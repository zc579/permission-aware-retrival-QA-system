import json
import os
import re
from typing import List, Dict, Any


def load_docs(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def simple_sentence_split(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for doc in docs:
        sents = simple_sentence_split(doc.get('text', ''))
        for i, s in enumerate(sents):
            chunks.append({
                'chunk_id': f"{doc['id']}_c{i}",
                'doc_id': doc['id'],
                'text': s,
                'title': doc.get('title'),
                'metadata': {
                    'source': doc.get('source'),
                    'project': doc.get('project'),
                    'sensitivity': doc.get('sensitivity'),
                    'allowed_roles': doc.get('allowed_roles', []),
                    'allowed_users': doc.get('allowed_users', []),
                }
            })
    return chunks


def filter_by_permissions(chunks: List[Dict[str, Any]], user: Dict[str, Any]) -> (List[Dict[str, Any]], int):
    allowed = []
    filtered_doc_ids = set()
    user_role = user.get('role')
    user_id = user.get('user')
    for c in chunks:
        meta = c['metadata']
        allowed_roles = meta.get('allowed_roles') or []
        allowed_users = meta.get('allowed_users') or []
        if allowed_users and user_id in allowed_users:
            allowed.append(c)
        elif allowed_roles and user_role in allowed_roles:
            allowed.append(c)
        elif not allowed_roles and not allowed_users:
            allowed.append(c)
        else:
            filtered_doc_ids.add(c['doc_id'])
    return allowed, len(filtered_doc_ids)
