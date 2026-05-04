"""
RAG Answer Generator
====================
Takes retrieved chunks + user query → generates grounded answer via Groq LLM.
Uses only retrieved context, never makes up facts.
"""

import requests
from typing import List, Dict
from config import GROQ_API_KEY, GROQ_MODEL


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict],
    filters_used: Dict = None,
    conversation_history: List[Dict] = None
) -> Dict:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        query: User question
        retrieved_chunks: List of {"text", "metadata", "score"}
        filters_used: Metadata filters that were applied
        conversation_history: Prior Q&A for multi-turn context

    Returns:
        {
            "answer": str,
            "sources": List[str],
            "confidence": str,
            "chunk_count": int,
            "used_history": bool
        }
    """
    if not retrieved_chunks:
        return {
            "answer":      "No relevant documents found. Please upload documents and try again.",
            "sources":     [],
            "confidence":  "none",
            "chunk_count": 0,
            "used_history": False
        }

    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_your_groq_api_key_here":
        context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(retrieved_chunks)])
        return {
            "answer":      f"**Retrieved Context (offline mode — add Groq API key for AI answers):**\n\n{context}",
            "sources":     list(set(c["metadata"].get("filename", "unknown") for c in retrieved_chunks)),
            "confidence":  "offline",
            "chunk_count": len(retrieved_chunks),
            "used_history": False
        }

    # Build context from chunks
    context_parts = []
    sources = []
    for i, chunk in enumerate(retrieved_chunks):
        src = chunk["metadata"].get("filename", "unknown")
        score = chunk.get("score", 0)
        context_parts.append(f"[Source {i+1} | {src} | relevance: {score:.2f}]\n{chunk['text']}")
        if src not in sources:
            sources.append(src)

    context = "\n\n".join(context_parts)

    # Filter context
    filter_note = ""
    if filters_used:
        parts = [f"{k}={v}" for k, v in filters_used.items()]
        filter_note = f"\nFilters applied: {', '.join(parts)}"

    system = """You are an enterprise knowledge assistant. Answer questions using ONLY the provided context.

Rules:
- Answer based strictly on the provided context chunks
- If the answer is not in the context, say so clearly
- Cite source numbers like [Source 1] when referencing specific facts
- Be concise and professional
- Format with bullet points for multi-part answers
- Never make up information not present in context"""

    # Build messages
    messages = [{"role": "system", "content": system}]

    # Add conversation history (last 3 turns)
    used_history = False
    if conversation_history:
        for turn in conversation_history[-3:]:
            messages.append({"role": "user",      "content": turn["question"]})
            messages.append({"role": "assistant",  "content": turn["answer"]})
        used_history = True

    messages.append({
        "role": "user",
        "content": f"""Context:{filter_note}

{context}

Question: {query}

Answer based only on the context above:"""
    })

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "temperature": 0.2,
        "max_tokens":  1024
    }

    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip()

        # Estimate confidence from top chunk score
        top_score = retrieved_chunks[0].get("score", 0) if retrieved_chunks else 0
        if top_score >= 0.75:
            confidence = "high"
        elif top_score >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "answer":      answer,
            "sources":     sources,
            "confidence":  confidence,
            "chunk_count": len(retrieved_chunks),
            "used_history": used_history
        }

    except requests.exceptions.HTTPError as e:
        return {
            "answer":      f"API Error: {str(e)}. Check your Groq API key in config.py.",
            "sources":     [],
            "confidence":  "error",
            "chunk_count": 0,
            "used_history": False
        }
    except Exception as e:
        return {
            "answer":      f"Error generating answer: {str(e)}",
            "sources":     [],
            "confidence":  "error",
            "chunk_count": 0,
            "used_history": False
        }
