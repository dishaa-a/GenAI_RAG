from pathlib import Path
import argparse, hashlib, json, math, os, re, sqlite3, time
ROOT=Path(__file__).resolve().parent; DB=ROOT/'rag.sqlite3'; D=384

def vec(s):
 v=[0.0]*D
 for w in re.findall(r"\w+",s.lower()):
  h=int(hashlib.sha256(w.encode()).hexdigest()[:8],16); v[h%D]+=1 if h&256 else -1
 n=math.sqrt(sum(x*x for x in v)) or 1
 return [x/n for x in v]
def db():
 c=sqlite3.connect(DB); c.execute('CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,source TEXT,digest TEXT,text TEXT,category TEXT,vec TEXT)'); return c
def split(s,size=500,overlap=80):
 w=s.split(); step=max(1,size-overlap); return [' '.join(w[i:i+size]) for i in range(0,len(w),step)]
def ingest(folder):
 c=db(); added=0
 for p in Path(folder).rglob('*'):
  if not p.is_file() or p.suffix.lower() not in ('.md','.txt','.html','.htm','.pdf'): continue
  if p.suffix.lower()=='.pdf':
   from pypdf import PdfReader
   text=' '.join(pg.extract_text() or '' for pg in PdfReader(str(p)).pages)
  else: text=p.read_text(encoding='utf8',errors='ignore'); text=re.sub('<[^>]+>',' ',text)
  src=str(p.resolve()); dg=hashlib.sha256(text.encode()).hexdigest()
  old=c.execute('SELECT digest FROM chunks WHERE source=? LIMIT 1',(src,)).fetchone()
  if old and old[0]==dg: continue
  c.execute('DELETE FROM chunks WHERE source=?',(src,))
  for i,chunk in enumerate(split(text)):
   cid=hashlib.sha1(f'{src}{dg}{i}'.encode()).hexdigest()[:12]; c.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?)',(cid,src,dg,chunk,p.parent.name,json.dumps(vec(chunk)))); added+=1
 c.commit(); c.close(); return added
def retrieve(q,k=4,category=None):
 c=db(); rows=c.execute('SELECT id,source,text,category,vec FROM chunks'+(' WHERE category=?' if category else ''),([category] if category else [])).fetchall(); c.close(); qv=vec(q)
 return sorted([(sum(a*b for a,b in zip(qv,json.loads(v))),i,s,t,cat) for i,s,t,cat,v in rows],reverse=True)[:k]
def answer(q,k=4,category=None):
 t=time.perf_counter(); hits=retrieve(q,k,category); ms=(time.perf_counter()-t)*1000
 if not hits or hits[0][0]<.08: return {'answer':'I do not have relevant context to answer that.','citations':[],'chunks':0,'retrieval_ms':round(ms,2),'tokens':{'prompt':0,'completion':0}}
 ctx='\n'.join(f'[{i}] {s}' for _,i,_,s,_ in hits); base=os.getenv('LLM_BASE_URL'); key=os.getenv('LLM_API_KEY'); toks={'prompt':0,'completion':0}
 if base and key:
  import urllib.request
  prompt=f'Answer only from context, cite claims as [chunk_id], abstain if unsupported.\nQuestion: {q}\nContext:\n{ctx}'
  body=json.dumps({'model':os.getenv('LLM_MODEL','gpt-4o-mini'),'messages':[{'role':'user','content':prompt}],'temperature':0}).encode(); req=urllib.request.Request(base.rstrip('/')+'/chat/completions',body,{'Content-Type':'application/json','Authorization':'Bearer '+key})
  with urllib.request.urlopen(req,timeout=90) as res: d=json.loads(res.read()); out=d['choices'][0]['message']['content']; toks={'prompt':d.get('usage',{}).get('prompt_tokens',0),'completion':d.get('usage',{}).get('completion_tokens',0)}
 else: out=' '.join(re.split(r'(?<=[.!?])\s+',hits[0][3])[:2])+f' [{hits[0][1]}]'
 return {'answer':out,'citations':[i for _,i,_,_,_ in hits if f'[{i}]' in out] or [hits[0][1]],'chunks':len(hits),'retrieval_ms':round(ms,2),'tokens':toks}
def evaluate():
 qs=json.loads((ROOT/'eval_questions.json').read_text(encoding='utf8')); rows=[]
 for x in qs:
  t=time.perf_counter(); got=retrieve(x['question'],4); ms=(time.perf_counter()-t)*1000; relevant=[i for i,r in enumerate(got) if x['relevant_text'].lower() in r[3].lower()]; ranks=[i+1 for i in relevant]; ideal=sum(1/math.log2(i+2) for i in range(min(1,len(got)))); dcg=sum(1/math.log2(r+1) for r in ranks)
  rows.append({'id':x['id'],'recall_at_k':float(bool(ranks)),'hit_rate':int(bool(ranks)),'mrr':1/ranks[0] if ranks else 0,'ndcg_at_k':dcg/ideal if ideal else 0,'context_precision':len(ranks)/len(got) if got else 0,'retrieval_ms':ms})
 vals=[r['retrieval_ms'] for r in rows]; out={'n':len(rows),'means':{k:sum(r[k] for r in rows)/len(rows) for k in rows[0] if k not in ('id',)},'p50_ms':sorted(vals)[len(vals)//2],'p95_ms':sorted(vals)[math.ceil(.95*len(vals))-1],'answer_quality':'not measured; requires independent judge/human labels','cases':rows}
 (ROOT/'eval_results.json').write_text(json.dumps(out,indent=2)); return out
try:
 from fastapi import FastAPI
 from pydantic import BaseModel
 api=FastAPI()
 class Req(BaseModel): question:str; k:int=4; category:str|None=None
 @api.post('/ask')
 def ask(r:Req): return answer(r.question,r.k,r.category)
except ImportError: api=None
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('command',choices=['ingest','eval','ask']); p.add_argument('value',nargs='?'); p.add_argument('--k',type=int,default=4); p.add_argument('--category'); a=p.parse_args(); result=ingest(a.value or ROOT/'data') if a.command=='ingest' else evaluate() if a.command=='eval' else answer(a.value or '',a.k,a.category); print(json.dumps(result,indent=2))
