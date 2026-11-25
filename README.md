# Assignment solutions: RAG + LLM judge

Requires Python 3.10+. Install with `pip install -r requirements.txt`.

## Problem 1

Run from this directory:

```powershell
python rag/app.py ingest rag/data
python rag/app.py eval
python rag/app.py ask "How long is the refund window?"
uvicorn rag.app:api --host 127.0.0.1 --port 8000
```

POST `/ask` accepts `{"question":"...","k":4,"category":"policy"}`. Uses SQLite, 384-dimensional signed feature hashing, 500-token chunks and 80-token overlap. Re-ingest is idempotent by source and content hash. PDF/HTML/MD/TXT are accepted. Embedding is a no-cost lexical baseline; use a semantic embedder for stronger retrieval. Optional OpenAI-compatible generation: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. Every generated claim is instructed to cite chunk IDs; insufficient context must abstain. `eval_questions.json` has 15 gold evidence snippets; eval maps snippets to retrieved chunks and computes Recall@k, Hit Rate, MRR, nDCG@k, context precision, p50/p95 latency. Answer evaluation is deliberately marked unmeasured in baseline mode; live answer judging needs labeled model calls.

Cost assumptions (planning only, USD/month): local fixed VM $50 + disk at $0.10/GB-month; each vector is 384 float32 values + 1 KB metadata; managed assumption $0.10/vector-month. This yields local $50.14/$51.44/$64.40 vs managed $10,000/$100,000/$1,000,000 at 100K/1M/10M vectors. Replace illustrative prices with vendor quotes. Trade-offs: single-node operations and weaker lexical retrieval. Switch to managed when uptime, replicas, high QPS, operations burden, or latency objectives justify its cost.

## Problem 2

`python judge/pipeline.py judge/data/suite.json --out judge/artifacts` runs a pairwise A/B suite, reverses order, logs prompt and raw verdict per call, aggregates pass rate, mean criterion score, win rate, and position flip rate. Configure judge independently with `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`. Runnable local demo mode produces deterministic mechanics evidence, not evidence of real judge quality. Rubric scores correctness, faithfulness, completeness, instruction-following, concise grounded style (1–5 with rationale). Pairwise is suitable for comparing two configurations; reference-based scores fit tasks with gold answers; pointwise suits a fixed release gate; reference-free can judge style but has weaker factual validation.

Validation artifact records labeled agreement, Cohen's kappa, repeated-run consistency, and verbose-wrong/terse-correct adversarial results. A different model family should judge the generator. Both A/B orders are required; disagreements count as flips. The demo artifacts must not be presented as empirical LLM validation or used as a release gate. Calibrate on human labels before gating.
