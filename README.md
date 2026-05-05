Permission-aware Retrieval & QA MVP

This small Python project demonstrates a permission-filtered retrieval-and-QA pipeline.

Features
- Document-level metadata with allowed_roles / allowed_users
- Permission-first filtering before retrieval
- Hybrid retrieval (TF-IDF cosine + token-overlap)
- Optional AI answer synthesis over retrieved authorized chunks
- Extractive fallback QA that returns answers with per-sentence citations
- Browser demo showing final answer, citations, retrieved chunks, and trace
- Evaluation suite including permission-leak test

Run
1. Create a venv and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Run the demo:
```bash
python -m src.main
```

3. Run the browser RAG demo:
```bash
python3 -m src.web_demo
```

Then open http://127.0.0.1:8000.

AI synthesis is enabled when `OPENAI_API_KEY` is set. Optional environment variables:
- `OPENAI_MODEL` defaults to `gpt-4o-mini`
- `OPENAI_API_URL` defaults to `https://api.openai.com/v1/chat/completions`
- `AI_API_TIMEOUT` defaults to `30`

Files
- data/docs.jsonl: sample documents
- src/: core modules (data.py, retriever.py, qa.py, llm_answer.py, web_demo.py, eval.py, main.py)
