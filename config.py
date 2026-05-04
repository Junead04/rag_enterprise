# ─────────────────────────────────────────────
# RAG Enterprise — Configuration
# ─────────────────────────────────────────────

try:
    import streamlit as st
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    # LOCAL: paste your key here
    GROQ_API_KEY = "gsk_your_groq_api_key_here"

GROQ_MODEL        = "llama-3.1-8b-instant"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"       # free HuggingFace model
CHROMA_PERSIST    = "./vectorstore"            # local ChromaDB storage
MAX_CHUNK_TOKENS  = 120                        # max tokens per proposition chunk
TOP_K_RESULTS     = 5                          # default retrieval count
