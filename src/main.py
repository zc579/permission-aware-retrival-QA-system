from src.data import load_docs, chunk_documents
from src.qa import answer_query
from src.eval import run_evals
import json


def demo_interactive():
    docs = load_docs('data/docs.jsonl')
    chunks = chunk_documents(docs)

    users = [
        {'user': 'alice', 'role': 'researcher'},
        {'user': 'bob', 'role': 'manager'},
        {'user': 'charlie', 'role': 'intern'},
    ]

    queries = [
        "What improvement in throughput did Alpha show?",
        "Summarize confidential design choices for Project Alpha",
        "What does the external paper say about throughput?",
    ]

    for u in users:
        print(f"\n=== User: {u['user']} ({u['role']}) ===")
        for q in queries:
            res = answer_query(u, q, chunks, top_k=3)
            print(f"Query: {q}")
            print(f"Answer: {res['answer']}")
            print(f"Citations: {json.dumps(res['citations'], indent=2)}")
            print(f"Filtered out doc count: {res['filtered_out_docs']}\n")


def run_eval():
    users = [
        {'user': 'alice', 'role': 'researcher'},
        {'user': 'bob', 'role': 'manager'},
        {'user': 'charlie', 'role': 'intern'},
    ]
    results = run_evals(users, 'data/docs.jsonl')
    print('\n=== Eval results (sample) ===')
    for r in results:
        print(f"User={r['user']['user']} role={r['user']['role']} Query={r['query']} Filtered={r['result']['filtered_out_docs']}")


if __name__ == '__main__':
    demo_interactive()
    run_eval()
