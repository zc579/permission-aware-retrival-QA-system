Permission-aware Retrieval & QA MVP

This small Python project demonstrates a permission-filtered retrieval-and-QA pipeline.

Features
- Document-level metadata with allowed_roles / allowed_users
- Permission-first filtering before retrieval
- Hybrid retrieval (TF-IDF cosine + token-overlap)
- Extractive QA that returns answers with per-sentence citations
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

Files
- data/docs.jsonl: sample documents
- src/: core modules (data.py, retriever.py, qa.py, eval.py, main.py)
