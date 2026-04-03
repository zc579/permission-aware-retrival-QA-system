from typing import List, Dict
from .data import load_docs, chunk_documents
from .qa import answer_query


SAMPLE_QUERIES = [
    {"q": "What improvement in throughput did Alpha show?", "sensitivity": "medium"},
    {"q": "Summarize confidential design choices for Project Alpha", "sensitivity": "high"},
    {"q": "What does the external paper say about throughput?", "sensitivity": "public"},
    {"q": "List intern observations on Beta project", "sensitivity": "low"},
]


def run_evals(users: List[Dict[str, str]], docs_path: str):
    docs = load_docs(docs_path)
    chunks = chunk_documents(docs)
    results = []
    for user in users:
        for q in SAMPLE_QUERIES:
            res = answer_query(user, q['q'], chunks, top_k=5)
            results.append({"user": user, "query": q['q'], "result": res})
    return results
