"""
Vector Store Manager
====================
Manages ChromaDB collections with HuggingFace sentence-transformer embeddings.
Handles: ingestion, cosine similarity search, metadata filtering, deletion.
"""

import os
import hashlib
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions
from config import EMBEDDING_MODEL, CHROMA_PERSIST


# Singleton embedding function (loaded once)
_ef = None

def get_embedding_function():
    global _ef
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _ef


def get_collection(collection_name: str = "rag_enterprise"):
    """Get or create ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST)
    col = client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}   # cosine similarity
    )
    return col, client


def ingest_chunks(chunks: List[Dict], collection_name: str = "rag_enterprise") -> Dict:
    """
    Ingest proposition chunks into ChromaDB.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        collection_name: ChromaDB collection name

    Returns:
        {"ingested": int, "skipped": int, "total": int}
    """
    col, _ = get_collection(collection_name)
    ingested = 0
    skipped  = 0

    texts     = []
    metadatas = []
    ids       = []

    for chunk in chunks:
        text = chunk["text"].strip()
        if not text or len(text) < 10:
            skipped += 1
            continue

        # Deterministic ID from content → prevents duplicates
        uid = hashlib.md5(text.encode()).hexdigest()

        # ChromaDB metadata values must be str/int/float/bool
        clean_meta = {}
        for k, v in chunk["metadata"].items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)

        texts.append(text)
        metadatas.append(clean_meta)
        ids.append(uid)
        ingested += 1

    if texts:
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            col.upsert(
                documents=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        # Re-add with actual text
        col.upsert(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    return {
        "ingested": ingested,
        "skipped":  skipped,
        "total":    ingested + skipped
    }


def search(
    query: str,
    top_k: int = 5,
    metadata_filter: Optional[Dict] = None,
    collection_name: str = "rag_enterprise"
) -> List[Dict]:
    """
    Cosine similarity search with optional metadata filtering.

    Args:
        query: Natural language query
        top_k: Number of results to return
        metadata_filter: ChromaDB where clause dict, e.g.
                         {"domain": "banking"} or
                         {"$and": [{"domain": "banking"}, {"date": "2024"}]}
        collection_name: ChromaDB collection name

    Returns:
        List of {"text": str, "metadata": dict, "score": float}
    """
    col, _ = get_collection(collection_name)

    count = col.count()
    if count == 0:
        return []

    actual_k = min(top_k, count)

    query_params = {
        "query_texts":     [query],
        "n_results":       actual_k,
        "include":         ["documents", "metadatas", "distances"]
    }
    if metadata_filter:
        query_params["where"] = metadata_filter

    try:
        results = col.query(**query_params)
    except Exception as e:
        # If filter fails, search without filter
        query_params.pop("where", None)
        results = col.query(**query_params)

    output = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance → similarity score
        score = round(1 - dist, 4)
        output.append({
            "text":     doc,
            "metadata": meta,
            "score":    score
        })

    # Sort by score descending
    output.sort(key=lambda x: x["score"], reverse=True)
    return output


def get_collection_stats(collection_name: str = "rag_enterprise") -> Dict:
    """Return stats about the current collection."""
    try:
        col, _ = get_collection(collection_name)
        count = col.count()
        if count == 0:
            return {"total_chunks": 0, "documents": [], "domains": [], "chunk_types": {}}

        # Sample up to 500 to get metadata overview
        sample = col.get(limit=min(500, count), include=["metadatas"])
        metas  = sample["metadatas"]

        docs     = list(set(m.get("filename", "unknown") for m in metas))
        domains  = list(set(m.get("domain", "") for m in metas if m.get("domain")))
        types    = {}
        for m in metas:
            t = m.get("chunk_type", "unknown")
            types[t] = types.get(t, 0) + 1

        return {
            "total_chunks": count,
            "documents":    docs,
            "domains":      domains,
            "chunk_types":  types
        }
    except Exception as e:
        return {"total_chunks": 0, "documents": [], "domains": [], "chunk_types": {}, "error": str(e)}


def delete_document(filename: str, collection_name: str = "rag_enterprise") -> int:
    """Delete all chunks belonging to a specific document."""
    col, _ = get_collection(collection_name)
    try:
        results = col.get(where={"filename": filename}, include=["metadatas"])
        ids = results["ids"]
        if ids:
            col.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def clear_collection(collection_name: str = "rag_enterprise"):
    """Wipe entire collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
