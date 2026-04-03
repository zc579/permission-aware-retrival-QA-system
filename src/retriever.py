from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class HybridRetriever:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.texts = [c['text'] for c in chunks]
        self.vectorizer = TfidfVectorizer().fit(self.texts + [" "])
        self.tfidf = self.vectorizer.transform(self.texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.vectorizer.transform([query])
        cos = cosine_similarity(q_vec, self.tfidf).flatten()
        # simple token overlap
        q_tokens = set(query.lower().split())
        overlaps = np.array([len(q_tokens & set(t.lower().split())) for t in self.texts], dtype=float)
        if overlaps.max() > 0:
            overlaps = overlaps / (overlaps.max() + 1e-9)
        scores = 0.7 * cos + 0.3 * overlaps
        idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idx:
            results.append({
                'chunk': self.chunks[i],
                'score': float(scores[i]),
                'rank': int(np.where(idx == i)[0][0])
            })
        return results
