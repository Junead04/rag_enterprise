"""
Proposition-Based Chunking Agent (Agentic Chunking)
====================================================

HOW IT WORKS:
1. Split document into paragraphs (naive pass)
2. Send each paragraph to LLM → decompose into atomic propositions
3. Each proposition = one self-contained fact/claim
4. These propositions become the chunks stored in vector DB

WHY BETTER THAN NAIVE CHUNKING:
- Fixed-size chunks cut sentences mid-thought → lose context
- Propositions are semantically complete units
- Better cosine similarity matching → more relevant retrieval
- Reduces hallucination: model gets precise facts, not noisy paragraphs
"""

import re
import requests
from typing import List, Dict
from config import GROQ_API_KEY, GROQ_MODEL


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call_groq(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    """Raw Groq API call."""
    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_your_groq_api_key_here":
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    [
            {"role": "system", "content": system or "You are a precise text analysis assistant."},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens":  1024
    }
    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return ""


# ─────────────────────────────────────────────
# STEP 1 — Split into paragraphs
# ─────────────────────────────────────────────
def split_into_paragraphs(text: str, min_len: int = 80) -> List[str]:
    """Split document text into meaningful paragraphs."""
    # Split on double newlines or sentence boundaries
    raw = re.split(r'\n{2,}', text.strip())
    paragraphs = []
    buffer = ""

    for para in raw:
        para = para.strip()
        if not para:
            continue
        buffer = (buffer + " " + para).strip() if buffer else para
        # Flush when buffer is long enough
        if len(buffer) >= min_len:
            paragraphs.append(buffer)
            buffer = ""

    if buffer and len(buffer) >= 30:
        paragraphs.append(buffer)

    return paragraphs


# ─────────────────────────────────────────────
# STEP 2 — LLM: decompose paragraph → propositions
# ─────────────────────────────────────────────
def decompose_to_propositions(paragraph: str) -> List[str]:
    """
    Send a paragraph to Groq LLM and get back atomic propositions.
    Each proposition is a standalone, self-contained fact.
    """
    system = (
        "You are an expert at breaking text into atomic propositions. "
        "Each proposition must be: self-contained, factual, one idea only. "
        "Output ONLY a numbered list. No preamble. No explanation."
    )
    prompt = f"""Break this paragraph into atomic propositions.
Each proposition should be a complete, standalone factual statement.

Paragraph:
\"\"\"{paragraph}\"\"\"

Rules:
- One proposition per line
- Each must make sense on its own without context
- Keep proper nouns and specific details
- Number each line: 1. 2. 3. etc.
- Maximum 15 propositions per paragraph"""

    response = _call_groq(prompt, system, temperature=0.1)

    if not response:
        # Fallback: split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    # Parse numbered list
    lines = response.split("\n")
    propositions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1." "1)" "- " etc.
        cleaned = re.sub(r'^[\d]+[.)]\s*', '', line).strip()
        cleaned = re.sub(r'^[-•]\s*', '', cleaned).strip()
        if len(cleaned) > 15:
            propositions.append(cleaned)

    return propositions if propositions else [paragraph]


# ─────────────────────────────────────────────
# STEP 3 — Fallback: sentence-based chunking
# ─────────────────────────────────────────────
def sentence_chunk(text: str, chunk_size: int = 3) -> List[str]:
    """Simple sentence-based chunking as offline fallback."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────
# MAIN: Full proposition chunking pipeline
# ─────────────────────────────────────────────
def chunk_document(
    text: str,
    metadata: Dict,
    use_llm: bool = True,
    progress_callback=None
) -> List[Dict]:
    """
    Full agentic chunking pipeline.

    Args:
        text: Raw document text
        metadata: Document metadata dict
        use_llm: If True, use LLM proposition decomposition
                 If False, use sentence-based chunking (offline mode)
        progress_callback: Optional callable(current, total, msg)

    Returns:
        List of chunk dicts:
        {
            "text": str,          # the proposition/chunk
            "metadata": {
                ...doc metadata,
                "chunk_index": int,
                "chunk_type": "proposition" | "sentence",
                "source_para": int,
            }
        }
    """
    paragraphs = split_into_paragraphs(text)
    all_chunks = []
    chunk_index = 0

    for para_idx, para in enumerate(paragraphs):
        if progress_callback:
            progress_callback(para_idx, len(paragraphs),
                              f"Chunking paragraph {para_idx + 1}/{len(paragraphs)}")

        if use_llm and GROQ_API_KEY and GROQ_API_KEY != "gsk_your_groq_api_key_here":
            propositions = decompose_to_propositions(para)
            chunk_type = "proposition"
        else:
            propositions = sentence_chunk(para)
            chunk_type = "sentence"

        for prop in propositions:
            chunk_meta = {**metadata}
            chunk_meta["chunk_index"]  = str(chunk_index)
            chunk_meta["chunk_type"]   = chunk_type
            chunk_meta["source_para"]  = str(para_idx)
            chunk_meta["char_length"]  = str(len(prop))

            all_chunks.append({
                "text":     prop,
                "metadata": chunk_meta
            })
            chunk_index += 1

    return all_chunks
