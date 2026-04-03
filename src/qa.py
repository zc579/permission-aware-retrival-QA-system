from typing import Dict, Any, List
from .data import load_docs, chunk_documents, filter_by_permissions
from .retriever import HybridRetriever


def answer_query(user: Dict[str, Any], query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    # permission filter
    allowed_chunks, filtered_doc_count = filter_by_permissions(chunks, user)

    if not allowed_chunks:
        return {"answer": "I don't have access to documents relevant to that query.", "citations": [], "filtered_out_docs": filtered_doc_count}

    retriever = HybridRetriever(allowed_chunks)
    results = retriever.retrieve(query, top_k=top_k)

    # build extractive answer: pick top result sentences and attach citations
    answer_sentences = []
    citations = []
    for r in results:
        sent = r['chunk']['text']
        src = r['chunk']['doc_id']
        answer_sentences.append(sent)
        citations.append({'chunk_id': r['chunk']['chunk_id'], 'doc_id': src, 'score': r['score']})

    answer = ' '.join(answer_sentences)
    return {"answer": answer, "citations": citations, "filtered_out_docs": filtered_doc_count}
