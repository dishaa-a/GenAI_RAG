from pathlib import Path
import argparse,json,os,re,urllib.request,random
RUBRIC=['correctness','faithfulness','completeness','instruction_following','concise_grounded_style']
def demo(case,a,b):
 expected=case.get('expected','').lower(); good=expected in a.lower(); bad=expected in b.lower();
 if good and not bad: winner='A'
 elif bad and not good: winner='B'
 elif good and bad: winner='tie'
 else: winner='tie'
 scores={k:{'A':(5 if good else 2),'B':(5 if bad else 1),'rationale':f"Compared with expected evidence: {expected}"} for k in RUBRIC}
 return {'winner':winner,'scores':scores,'rationale':'Preference based on expected answer evidence; unsupported extra claims reduce faithfulness.'}
def call(prompt,case,a,b):
 base=os.getenv('JUDGE_BASE_URL'); key=os.getenv('JUDGE_API_KEY')
 if not base or not key: return json.dumps(demo(case,a,b))
 body=json.dumps({'model':os.getenv('JUDGE_MODEL','judge-model'),'messages':[{'role':'system','content':'Return only valid JSON with winner (A/B/tie), scores per criterion for A and B, rationale. Score 1-5. Penalize unsupported verbosity.'},{'role':'user','content':prompt}],'temperature':0}).encode(); req=urllib.request.Request(base.rstrip('/')+'/chat/completions',body,{'Content-Type':'application/json','Authorization':'Bearer '+key})
 with urllib.request.urlopen(req,timeout=90) as res: return json.loads(res.read())['choices'][0]['message']['content']
def parse(raw):
 s=raw.strip(); s=re.sub(r'^```(?:json)?|```$','',s,flags=re.I|re.M).strip()
 try:return json.loads(s)
 except: pass
 m=re.search(r'\{.*\}',s,re.S)
 if m: return json.loads(m.group())
 raise ValueError('Judge returned malformed JSON: '+raw[:250])
def run(suite,out):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); audit=[]; cases=[]; flips=0; wins={'A':0,'B':0,'tie':0};
 for c in suite:
  verdicts=[]
  for aa,bb,order in [(c['a'],c['b'],'AB'),(c['b'],c['a'],'BA')]:
   prompt=json.dumps({'input':c['input'],'expected':c.get('expected'),'A':aa,'B':bb,'rubric':RUBRIC,'anchors':'1 poor, 3 mixed, 5 fully correct and grounded; score each criterion separately.'})
   raw=call(prompt,c,aa,bb)
   try:v=parse(raw)
   except Exception as e: v={'winner':'tie','parse_error':str(e),'scores':{}}
   # Map order-reversed verdict back to original labels.
   if order=='BA' and v.get('winner') in ('A','B'): v['winner']='B' if v['winner']=='A' else 'A'
   verdicts.append(v); audit.append({'case':c['id'],'order':order,'prompt':prompt,'raw_response':raw,'parsed':v})
  w1,w2=(v.get('winner','tie') for v in verdicts); flip=int(w1!=w2); flips+=flip; winner=w1 if w1==w2 else 'tie'; wins[winner]+=1
  cases.append({'id':c['id'],'winner':winner,'order_flip':bool(flip),'preferred':c.get('preferred'),'adversarial':bool(c.get('adversarial'))})
 (out/'judge_audit.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in audit),encoding='utf8')
 labeled=[x for x in cases if x['preferred'] in ('A','B','tie')]; agree=sum(x['winner']==x['preferred'] for x in labeled); n=len(labeled); po=agree/n if n else 0
 actual={'A':sum(x['winner']=='A' for x in labeled)/n if n else 0,'B':sum(x['winner']=='B' for x in labeled)/n if n else 0,'tie':sum(x['winner']=='tie' for x in labeled)/n if n else 0}; exp=sum((sum(x['preferred']==z for x in labeled)/n)*actual[z] for z in actual) if n else 0; kappa=(po-exp)/(1-exp) if n and exp<1 else None
 report={'n':len(cases),'pass_rate':sum(x['winner']==x.get('preferred') for x in cases)/len(cases),'win_rate':{k:v/len(cases) for k,v in wins.items()},'position_flip_rate':flips/len(cases),'human_gold_agreement':po,'cohen_kappa':kappa,'test_retest_consistency':1-flips/len(cases),'adversarial':[{**x,'fooled':x['winner']=='B'} for x in cases if x['adversarial']],'cases':cases,'caveat':'Deterministic demo mode is pipeline evidence only, not a real LLM judge validation.'}
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8'); (out/'validation.json').write_text(json.dumps({k:report[k] for k in ['human_gold_agreement','cohen_kappa','test_retest_consistency','adversarial','caveat']},indent=2),encoding='utf8'); return report
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('suite'); p.add_argument('--out',default='judge/artifacts'); a=p.parse_args(); print(json.dumps(run(json.loads(Path(a.suite).read_text(encoding='utf8')),a.out),indent=2))
