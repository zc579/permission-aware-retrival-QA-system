# 01 - Permission-Aware QA Implementation

# Permission-Aware QA Implementation Explanation

## 1. Project Overview

This project implements a small permission-aware retrieval and question-answering prototype. It loads sample documents from a JSONL file, splits them into sentence-level chunks, filters those chunks according to the current user's role or user ID, retrieves relevant authorized chunks, and returns an extractive answer with citations. The most important design choice is that permission filtering happens before retrieval, so unauthorized documents are not available to the retriever or the answer-generation step. The project is intentionally lightweight and uses local Python code rather than an external LLM or hosted search service.

## 2. Main Features Implemented

### Permission-filtered document access

- What it does: The system decides which document chunks a user is allowed to access based on document metadata and the user's role or ID.
- Why it matters: This directly addresses the assignment requirement that users should only retrieve and answer from content they are authorized to see.
- Where it is implemented: `src/data.py`, especially `filter_by_permissions`.
- How it works: Each chunk contains metadata copied from its parent document, including `allowed_roles` and `allowed_users`. `filter_by_permissions(chunks, user)` compares that metadata to the current `user` dictionary and returns only allowed chunks.

### Document loading and chunking

- What it does: The project loads documents from `data/docs.jsonl` and splits each document into sentence-sized chunks.
- Why it matters: Retrieval is performed over smaller text units rather than entire documents, which makes citations more specific and keeps answer context focused.
- Where it is implemented: `src/data.py`, especially `load_docs`, `simple_sentence_split`, and `chunk_documents`.
- How it works: `load_docs(path)` reads one JSON object per line. `chunk_documents(docs)` calls `simple_sentence_split` on each document's `text` field and creates chunk dictionaries with `chunk_id`, `doc_id`, `text`, `title`, and `metadata`.

### Hybrid retrieval

- What it does: The retriever ranks authorized chunks by combining TF-IDF cosine similarity with simple token overlap.
- Why it matters: This demonstrates search and retrieval without requiring a vector database or external embedding service.
- Where it is implemented: `src/retriever.py`, in the `HybridRetriever` class.
- How it works: `HybridRetriever.__init__` fits a `TfidfVectorizer` over the chunk texts. `retrieve(query, top_k)` transforms the query, computes cosine similarity against chunk vectors, computes token overlap between query tokens and each chunk, combines the two scores as `0.7 * cos + 0.3 * overlaps`, and returns the top ranked chunks.

### Extractive question answering

- What it does: The QA step creates an answer by concatenating the text of the top retrieved chunks.
- Why it matters: This keeps answers grounded in source material and makes it easy to trace each sentence back to a retrieved chunk.
- Where it is implemented: `src/qa.py`, in `answer_query`.
- How it works: `answer_query(user, query, chunks, top_k)` first filters chunks by permission, then builds a `HybridRetriever` only over allowed chunks, retrieves the top results, and joins the retrieved sentences into the final `answer`.

### Citation return

- What it does: The system returns citations for the chunks used in the answer.
- Why it matters: Citations help verify where each answer came from and demonstrate that the answer was built from retrieved source material.
- Where it is implemented: `src/qa.py`, inside the result-building loop in `answer_query`; citations are displayed in `src/main.py`.
- How it works: For each retrieved result, `answer_query` adds a citation dictionary containing `chunk_id`, `doc_id`, and retrieval `score`.

### Demo and evaluation harness

- What it does: The project includes a runnable demo and a simple evaluation loop over multiple users and sample queries.
- Why it matters: The examples show how different users receive different answers and filtered document counts based on permissions.
- Where it is implemented: `src/main.py` and `src/eval.py`.
- How it works: `demo_interactive()` runs three queries for Alice, Bob, and Charlie. `run_eval()` calls `run_evals`, which runs `SAMPLE_QUERIES` for each user and records the result from `answer_query`.

## 3. Permission-Aware Access Control

User permissions are defined in code as dictionaries with a `user` and `role`. In `src/main.py`, the demo users are:

```python
{'user': 'alice', 'role': 'researcher'}
{'user': 'bob', 'role': 'manager'}
{'user': 'charlie', 'role': 'intern'}
```

The same style of user dictionaries is used by `src/eval.py` when `run_evals` receives the user list from `src/main.py`.

Document permissions are defined in `data/docs.jsonl`. Each JSON document includes metadata fields such as `sensitivity`, `allowed_roles`, and `allowed_users`. For example:

- `doc_internal_a` is high-sensitivity Project Alpha content and allows only the `manager` role.
- `doc_internal_b` is medium-sensitivity Alpha content and allows `researcher` and `manager`.
- `doc_internal_c` is low-sensitivity Beta content and allows `intern`, `researcher`, and `manager`.
- `doc_external_1` has empty `allowed_roles` and `allowed_users`, which the code treats as public.

The permission check happens in `filter_by_permissions(chunks, user)` in `src/data.py`. For every chunk, the function reads:

- `user_role = user.get('role')`
- `user_id = user.get('user')`
- `allowed_roles = meta.get('allowed_roles') or []`
- `allowed_users = meta.get('allowed_users') or []`

The logic allows access in three cases:

- The document lists specific `allowed_users`, and the current user's ID appears in that list.
- The document lists `allowed_roles`, and the current user's role appears in that list.
- The document has no `allowed_roles` and no `allowed_users`, so it is treated as public.

If none of those conditions are true, the chunk is excluded and its `doc_id` is added to `filtered_doc_ids`. The function returns both the allowed chunks and the number of unique document IDs filtered out.

Filtering happens before retrieval. In `src/qa.py`, `answer_query` calls `filter_by_permissions(chunks, user)` before creating `HybridRetriever`. The retriever is initialized with `HybridRetriever(allowed_chunks)`, not with all chunks. This means unauthorized content is never indexed for that query, never scored by retrieval, and never passed into the answer-building step.

This design prevents unauthorized content from being used because later pipeline stages only receive the permission-approved subset. Even if a user's query asks directly about confidential content, the retriever cannot return chunks from documents the user cannot access.

## 4. Retrieval Pipeline

Documents are represented as JSON objects in `data/docs.jsonl`. Each document contains:

- `id`
- `title`
- `source`
- `project`
- `sensitivity`
- `allowed_roles`
- `allowed_users`
- `text`

The loading and chunking pipeline is implemented in `src/data.py`:

- `load_docs(path)` opens the JSONL file, skips blank lines, parses each line with `json.loads`, and returns a list of document dictionaries.
- `simple_sentence_split(text)` uses a regular expression, `r'(?<=[.!?])\s+'`, to split document text into sentences.
- `chunk_documents(docs)` creates one chunk per sentence and preserves the parent document's title and permission-related metadata.

Each chunk has this structure:

```python
{
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
}
```

The query is processed in `HybridRetriever.retrieve(query, top_k)` in `src/retriever.py`. The function transforms the query with the same `TfidfVectorizer` used for the chunks. It also lowercases and splits the query into simple whitespace tokens for the token-overlap score.

Relevant documents are selected by scoring chunks, not whole documents. The retriever computes:

- `cos`: cosine similarity between the query TF-IDF vector and each chunk TF-IDF vector.
- `overlaps`: a normalized count of shared tokens between the query and each chunk.
- `scores`: a weighted combination using `0.7 * cos + 0.3 * overlaps`.

The function sorts scores in descending order with `np.argsort(scores)[::-1]`, keeps the first `top_k` indices, and returns result dictionaries containing the matched `chunk`, the numeric `score`, and the `rank`.

Permission filtering interacts with retrieval through `src/qa.py`. `answer_query` first calls `filter_by_permissions`, then builds the retriever on `allowed_chunks`. This is the core permission-aware retrieval behavior: the search space is reduced to authorized content before ranking happens.

## 5. Question Answering Pipeline

The QA pipeline is implemented in `src/qa.py` through the `answer_query` function. Its inputs are:

- `user`: a dictionary such as `{'user': 'alice', 'role': 'researcher'}`
- `query`: the user's natural-language question
- `chunks`: all sentence chunks created from the loaded documents
- `top_k`: the number of retrieved chunks to include

The context passed into answer generation is the text of the retrieved, permission-approved chunks. There is no separate prompt template or external language model. Instead, the answer is extractive: it directly reuses source sentences from the retrieved chunks.

The answer is produced by this loop in `answer_query`:

```python
for r in results:
    sent = r['chunk']['text']
    src = r['chunk']['doc_id']
    answer_sentences.append(sent)
    citations.append({'chunk_id': r['chunk']['chunk_id'], 'doc_id': src, 'score': r['score']})
```

After the loop, the final answer is:

```python
answer = ' '.join(answer_sentences)
```

The answer is grounded in retrieved documents because every sentence in the answer comes directly from a retrieved chunk. The citations list is built from the same retrieved results, so the output records which chunks contributed to the answer.

The system avoids answering from unauthorized content by filtering before retrieval. If a user does not have access to a document, its chunks are not included in `allowed_chunks`, so they cannot appear in `results`, `answer_sentences`, or `citations`.

If a user has no allowed chunks at all, `answer_query` returns:

```python
{
    "answer": "I don't have access to documents relevant to that query.",
    "citations": [],
    "filtered_out_docs": filtered_doc_count
}
```

In the current sample dataset, all users can access at least the public external document, so the demo normally returns some answer text rather than this fallback.

## 6. Citation Support

Citation information is attached at the chunk level. The source document metadata is created during chunking in `src/data.py`, and the citation output is created in `src/qa.py`.

The stored chunk information includes:

- `chunk_id`: a unique sentence-level ID, such as `doc_internal_a_c2`
- `doc_id`: the parent document ID, such as `doc_internal_a`
- `title`: the parent document title
- `metadata.source`: whether the document is internal or external
- `metadata.project`
- `metadata.sensitivity`
- `metadata.allowed_roles`
- `metadata.allowed_users`

The final citation objects returned by `answer_query` include:

```python
{
    'chunk_id': r['chunk']['chunk_id'],
    'doc_id': src,
    'score': r['score']
}
```

These citations are connected to retrieved documents because they are generated from the same `results` list used to build the answer. In `src/main.py`, `demo_interactive()` prints the citations as formatted JSON:

```python
print(f"Citations: {json.dumps(res['citations'], indent=2)}")
```

In the demo output, a manager asking about Project Alpha's throughput receives a citation such as `doc_internal_a_c2`, which corresponds to the sentence "The preliminary results show a 20% improvement in throughput." An intern asking the same question does not receive citations from `doc_internal_a`, because that document is restricted to managers.

This citation design is useful for trust and verification because users can see the exact chunk IDs and document IDs behind the answer. It also makes permission behavior auditable: unauthorized document IDs should not appear in a user's citation list.

## 7. Demo and Evaluation Examples

The main demo is implemented in `demo_interactive()` in `src/main.py`. It loads documents, chunks them, defines three users, and runs three queries for each user.

The demo users test different access levels:

- Alice has role `researcher`.
- Bob has role `manager`.
- Charlie has role `intern`.

The demo queries are:

- `What improvement in throughput did Alpha show?`
- `Summarize confidential design choices for Project Alpha`
- `What does the external paper say about throughput?`

These examples demonstrate permission-aware retrieval because the same query produces different accessible source sets for different users. For example, Bob can access `doc_internal_a`, the confidential Project Alpha document, so the demo can return the sentence about a `20% improvement in throughput` with a citation to `doc_internal_a_c2`. Alice and Charlie cannot access that high-sensitivity manager-only document, so their filtered document counts are higher and their citations do not include `doc_internal_a`.

The evaluation examples are implemented in `src/eval.py`. The file defines `SAMPLE_QUERIES`:

```python
SAMPLE_QUERIES = [
    {"q": "What improvement in throughput did Alpha show?", "sensitivity": "medium"},
    {"q": "Summarize confidential design choices for Project Alpha", "sensitivity": "high"},
    {"q": "What does the external paper say about throughput?", "sensitivity": "public"},
    {"q": "List intern observations on Beta project", "sensitivity": "low"},
]
```

`run_evals(users, docs_path)` loads and chunks the documents, runs every sample query for every user, and appends result dictionaries containing the `user`, `query`, and `result` returned by `answer_query`.

`run_eval()` in `src/main.py` prints a compact summary showing each user, role, query, and `filtered_out_docs` count. In the current run:

- Alice, the researcher, filters out 1 document: the manager-only confidential Alpha document.
- Bob, the manager, filters out 0 documents.
- Charlie, the intern, filters out 2 documents: the high-sensitivity manager-only Alpha document and the medium-sensitivity researcher/manager Alpha literature review.

The examples show both authorized and unauthorized access cases. Bob demonstrates authorized access to confidential Alpha material. Alice and Charlie demonstrate unauthorized access cases because restricted documents are removed before retrieval and do not appear in their citations.

## 8. Assignment Alignment

- Search: The project implements search in `src/retriever.py` with `HybridRetriever`. It combines TF-IDF cosine similarity and token overlap to rank relevant chunks for a query.

- Context management: The project manages context by splitting full documents into sentence-level chunks in `src/data.py`. The QA step uses only the top retrieved chunks as answer context instead of passing all documents forward.

- Permissions: Permissions are represented in `data/docs.jsonl` with `allowed_roles` and `allowed_users`, and enforced in `src/data.py` by `filter_by_permissions`. `src/qa.py` applies this filter before retrieval, so unauthorized content is excluded from the search space.

- Agentic workflows: The project does not implement a multi-step autonomous agent, but it does implement a clear workflow pipeline: load documents, chunk documents, filter by permissions, retrieve relevant chunks, generate an extractive answer, and return citations. This structure demonstrates how an agent or assistant could use permission-aware retrieval as a tool before answering.

- Evals: The project includes a small evaluation harness in `src/eval.py`. `SAMPLE_QUERIES` covers public, low, medium, and high sensitivity scenarios, and `run_evals` runs those queries across users with different roles to show whether content is filtered appropriately.

## 9. Current Limitations

- The answer generation is extractive and concatenates retrieved sentences; it does not synthesize a natural-language response with an LLM.
- The evaluation harness records behavior but does not include formal pass/fail assertions.
- Citations include `chunk_id`, `doc_id`, and score, but not the document title or source in the final returned citation object.
- Retrieval uses simple whitespace token overlap and TF-IDF, so it may return weakly related chunks when few query terms match.
- Permissions are sample metadata in a local JSONL file rather than an integration with a real identity or access-control system.

## 10. Summary for Submission

I built a permission-aware retrieval and question-answering prototype that demonstrates how an assistant can search documents while respecting access control. The system loads JSONL documents with role/user permission metadata, splits them into sentence chunks, filters those chunks for the current user before retrieval, ranks the authorized chunks with a hybrid TF-IDF and token-overlap retriever, and produces an extractive answer with source citations. This design ensures that restricted documents are never searched or used to generate answers for unauthorized users, while the demo and evaluation examples show different behavior for manager, researcher, and intern roles across public, low-sensitivity, medium-sensitivity, and confidential queries.
