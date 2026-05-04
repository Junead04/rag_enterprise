"""
Metadata Filter Agent
=====================
Takes a natural language query and extracts structured metadata filters.

Example:
  Query: "What are the credit policies from the banking domain in 2024?"
  Output: {"domain": "banking", "date": "2024", "category": "credit"}

This is what makes retrieval TARGETED instead of naive.
"""

import re
import json
import requests
from typing import Dict, List, Optional
from config import GROQ_API_KEY, GROQ_MODEL


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def extract_metadata_filters(
    query: str,
    available_fields: List[str],
    available_values: Dict[str, List[str]]
) -> Dict:
    """
    Use LLM to extract metadata filters from a natural language query.

    Args:
        query: User's natural language question
        available_fields: List of metadata field names in the collection
                          e.g. ["domain", "author", "date", "category"]
        available_values: Dict of field → known values
                          e.g. {"domain": ["banking", "finance", "hr"]}

    Returns:
        ChromaDB-compatible where clause dict, or {} if no filters found
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_your_groq_api_key_here":
        return {}

    if not available_fields:
        return {}

    # Build context for LLM
    fields_str = ", ".join(available_fields)
    values_str = ""
    for field, vals in available_values.items():
        if vals:
            values_str += f"\n  {field}: {', '.join(str(v) for v in vals[:10])}"

    system = (
        "You are a metadata filter extraction system. "
        "Extract structured filters from user queries. "
        "Return ONLY valid JSON. No explanation. No markdown."
    )

    prompt = f"""Extract metadata filters from this query for a document search system.

Query: "{query}"

Available metadata fields: {fields_str}
Known values:{values_str if values_str else " (none provided)"}

Rules:
- Return a JSON object with only fields that are clearly mentioned or implied
- Use exact field names from the available fields list
- Use exact values from known values when possible
- Return {{}} if no metadata filters are implied
- Do NOT include fields for semantic content (that's handled by vector search)
- Only include fields for: domain, time period, author, category, source, file type

Examples:
- "banking credit policies" → {{"domain": "banking"}}
- "HR documents from 2024" → {{"domain": "hr", "date": "2024"}}
- "what is machine learning" → {{}}

Return JSON only:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens":  200
    }

    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()

        # Clean markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        filters = json.loads(raw)

        # Validate: only keep known fields with non-empty values
        valid = {}
        for k, v in filters.items():
            if k in available_fields and v and str(v).strip():
                valid[k] = str(v).strip()

        return valid

    except Exception:
        return {}


def build_chroma_filter(filters: Dict) -> Optional[Dict]:
    """
    Convert simple filter dict to ChromaDB where clause.

    Single filter:  {"domain": "banking"}
                 →  {"domain": "banking"}

    Multi filter:   {"domain": "banking", "date": "2024"}
                 →  {"$and": [{"domain": "banking"}, {"date": "2024"}]}
    """
    if not filters:
        return None

    if len(filters) == 1:
        k, v = next(iter(filters.items()))
        return {k: v}

    return {"$and": [{k: v} for k, v in filters.items()]}


def explain_filters(filters: Dict) -> str:
    """Human-readable explanation of applied filters."""
    if not filters:
        return "No metadata filters applied — searching entire knowledge base"
    parts = [f"**{k}** = `{v}`" for k, v in filters.items()]
    return "Filters applied: " + " · ".join(parts)
