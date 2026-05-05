from typing import Dict, Any, List
from .data import filter_by_permissions
from .llm_answer import generate_llm_answer
from .retriever import HybridRetriever


def _serialize_retrieved_chunk(result: Dict[str, Any]) -> Dict[str, Any]:
    chunk = result["chunk"]
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "title": chunk.get("title"),
        "text": chunk["text"],
        "metadata": chunk.get("metadata", {}),
        "score": result["score"],
        "rank": result["rank"],
    }


def answer_query(
    user: Dict[str, Any],
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    use_llm: bool = False,
) -> Dict[str, Any]:
    # permission filter
    allowed_chunks, filtered_doc_count = filter_by_permissions(chunks, user)

    if not allowed_chunks:
        return {
            "answer": "I don't have access to documents relevant to that query.",
            "citations": [],
            "filtered_out_docs": filtered_doc_count,
            "retrieved_chunks": [],
            "trace": {
                "user": user,
                "permission_filter": "before_retrieval",
                "allowed_chunk_count": 0,
                "filtered_out_docs": filtered_doc_count,
                "retriever": "not_run_no_allowed_chunks",
                "top_k": top_k,
            },
            "llm": {
                "used_ai_api": False,
                "provider": None,
                "model": None,
                "error": "No allowed chunks after permission filtering.",
            },
        }

    retriever = HybridRetriever(allowed_chunks)
    results = retriever.retrieve(query, top_k=top_k)
    retrieved_chunks = [_serialize_retrieved_chunk(r) for r in results]

    # build extractive answer: pick top result sentences and attach citations
    answer_sentences = []
    citations = []
    for chunk in retrieved_chunks:
        sent = chunk['text']
        src = chunk['doc_id']
        answer_sentences.append(sent)
        citations.append({
            'chunk_id': chunk['chunk_id'],
            'doc_id': src,
            'title': chunk.get('title'),
            'score': chunk['score'],
        })

    extractive_answer = ' '.join(answer_sentences)
    llm_result = {
        "used_ai_api": False,
        "provider": "extractive",
        "model": None,
        "prompt_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
        "error": None,
    }
    answer = extractive_answer

    if use_llm:
        llm_result = generate_llm_answer(query, retrieved_chunks)
        answer = llm_result["answer"]
        if llm_result.get("is_refusal"):
            citations = []

    return {
        "answer": answer,
        "extractive_answer": extractive_answer,
        "citations": citations,
        "filtered_out_docs": filtered_doc_count,
        "retrieved_chunks": retrieved_chunks,
        "trace": {
            "user": user,
            "permission_filter": "before_retrieval",
            "allowed_chunk_count": len(allowed_chunks),
            "filtered_out_docs": filtered_doc_count,
            "retriever": "HybridRetriever: 0.7 TF-IDF cosine + 0.3 token overlap",
            "retrieved_count": len(retrieved_chunks),
            "top_k": top_k,
            "llm_input_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
        },
        "llm": llm_result,
    }
