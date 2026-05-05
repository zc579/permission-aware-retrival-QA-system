import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List


REFUSAL_ANSWER = "I do not have enough accessible information to answer this question."


def _citation_id(chunk: Dict[str, Any]) -> str:
    return f"{chunk.get('doc_id')}#{chunk.get('chunk_id')}"


def _max_retrieval_score(retrieved_chunks: List[Dict[str, Any]]) -> float:
    if not retrieved_chunks:
        return 0.0
    return max(float(c.get("score") or 0.0) for c in retrieved_chunks)


def _min_relevance_score() -> float:
    return float(os.getenv("MIN_RELEVANCE_SCORE", "0.05"))


def _is_relevant_enough(retrieved_chunks: List[Dict[str, Any]]) -> bool:
    return _max_retrieval_score(retrieved_chunks) >= _min_relevance_score()


def _fallback_answer(query: str, retrieved_chunks: List[Dict[str, Any]], reason: str) -> Dict[str, Any]:
    citations = [_citation_id(c) for c in retrieved_chunks]
    is_refusal = not retrieved_chunks or not _is_relevant_enough(retrieved_chunks)
    if is_refusal:
        answer = REFUSAL_ANSWER
    else:
        answer = " ".join(c.get("text", "") for c in retrieved_chunks)

    return {
        "answer": answer,
        "is_refusal": is_refusal,
        "citations": citations,
        "used_ai_api": False,
        "provider": "offline_fallback",
        "model": None,
        "prompt_chunk_ids": [c.get("chunk_id") for c in retrieved_chunks],
        "error": reason,
        "query": query,
        "max_retrieval_score": _max_retrieval_score(retrieved_chunks),
        "min_relevance_score": _min_relevance_score(),
    }


def _format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata") or {}
        lines.append(
            "\n".join(
                [
                    f"Citation: {_citation_id(chunk)}",
                    f"Title: {chunk.get('title') or 'Untitled'}",
                    f"Source: {metadata.get('source') or 'unknown'}",
                    f"Sensitivity: {metadata.get('sensitivity') or 'unknown'}",
                    f"Retrieval score: {chunk.get('score', 0):.4f}",
                    f"Text: {chunk.get('text', '')}",
                ]
            )
        )
    return "\n\n---\n\n".join(lines)


def generate_llm_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate an answer using only retrieved, permission-approved chunks.

    The function calls an OpenAI-compatible chat-completions endpoint when
    OPENAI_API_KEY is available. Without API credentials, it returns a local
    extractive fallback so the demo still works offline.
    """
    if not retrieved_chunks:
        return _fallback_answer(query, retrieved_chunks, "No authorized chunks were retrieved.")

    if not _is_relevant_enough(retrieved_chunks):
        return _fallback_answer(query, retrieved_chunks, "Retrieved chunks were below the relevance threshold.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_answer(query, retrieved_chunks, "OPENAI_API_KEY is not set.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    timeout = int(os.getenv("AI_API_TIMEOUT", "30"))

    system_prompt = (
        "You are a permission-aware RAG assistant. Answer only from the provided "
        "authorized retrieved context. Do not use outside knowledge. If the context "
        "does not contain enough information, reply exactly: "
        f"{REFUSAL_ANSWER} Include citations using the exact Citation values provided "
        "only when the answer is supported by context."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Authorized retrieved context:\n{_format_context(retrieved_chunks)}"
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        return _fallback_answer(query, retrieved_chunks, f"AI API call failed: {exc}")

    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return _fallback_answer(query, retrieved_chunks, "AI API returned an unexpected response shape.")

    return {
        "answer": answer,
        "is_refusal": answer.strip() == REFUSAL_ANSWER,
        "citations": [_citation_id(c) for c in retrieved_chunks],
        "used_ai_api": True,
        "provider": "openai_compatible_chat_completions",
        "model": model,
        "prompt_chunk_ids": [c.get("chunk_id") for c in retrieved_chunks],
        "error": None,
        "query": query,
        "max_retrieval_score": _max_retrieval_score(retrieved_chunks),
        "min_relevance_score": _min_relevance_score(),
    }
