import streamlit as st
import os, sys, time, base64, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GROQ_API_KEY
from utils.document_loader import load_document
from agents.chunking_agent import chunk_document
from agents.vector_store import (
    ingest_chunks, search, get_collection_stats,
    delete_document, clear_collection
)
from agents.metadata_filter_agent import (
    extract_metadata_filters, build_chroma_filter, explain_filters
)
from agents.rag_agent import generate_answer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# BACKGROUND IMAGE
# ─────────────────────────────────────────────
def get_bg_css() -> str:
    img_path = os.path.join(os.path.dirname(__file__), "assets", "background.png")
    if not os.path.exists(img_path):
        return "none"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"url('data:image/png;base64,{b64}')"

bg_css = get_bg_css()

# ─────────────────────────────────────────────
# CSS — Dark Navy & Purple Theme
# ─────────────────────────────────────────────
# CSS — Deep Teal & White Theme (high readability)
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}

.stApp {{
    background-color: #f0f4f8;
    background-image: {bg_css};
    background-size: cover;
    background-position: center right;
    background-attachment: fixed;
    color: #0f2b3d;
}}
.stApp::before {{
    content: '';
    position: fixed; inset: 0;
    background: rgba(240,244,248,0.94);
    pointer-events: none; z-index: 0;
}}

section[data-testid="stSidebar"] {{
    background: #ffffff !important;
    border-right: 1px solid #b2d8d8 !important;
}}
section[data-testid="stSidebar"] * {{ color: #0f2b3d !important; }}
section[data-testid="stSidebar"] label {{
    color: #3a7a7a !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 1px;
}}

.header-banner {{
    background: #ffffff; border: 1px solid #0d9488;
    border-radius: 12px; padding: 24px 36px;
    margin-bottom: 20px; position: relative; overflow: hidden;
    box-shadow: 0 2px 12px rgba(13,148,136,0.10);
}}
.header-banner::before {{
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #0d9488, #14b8a6, #0891b2, #0e7490);
}}
.header-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 26px; color: #0f2b3d; margin: 0;
}}
.header-sub {{
    color: #3a7a7a; font-size: 12px; margin-top: 5px;
    letter-spacing: 1px; text-transform: uppercase;
}}
.header-badge {{
    display: inline-block; background: #ccfbf1; color: #0f766e;
    border: 1px solid #0d948855; padding: 3px 12px;
    border-radius: 20px; font-size: 11px; margin-top: 8px; letter-spacing: 1px;
}}

.stat-card {{
    background: #ffffff; border: 1px solid #b2d8d8;
    border-top: 3px solid #0d9488; border-radius: 10px;
    padding: 16px 18px; text-align: center;
    box-shadow: 0 1px 6px rgba(13,148,136,0.08);
}}
.stat-val {{ font-size: 28px; font-weight: 600; color: #0d9488; line-height: 1; }}
.stat-lbl {{ font-size: 10px; color: #3a7a7a; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }}

.chunk-card {{
    background: #ffffff; border: 1px solid #b2d8d8;
    border-left: 4px solid #0d9488; border-radius: 0 8px 8px 0;
    padding: 14px 18px; margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(13,148,136,0.06);
}}
.chunk-score {{
    display: inline-block; padding: 2px 10px;
    border-radius: 12px; font-size: 11px; font-weight: 600;
}}
.score-high {{ background: #dcfce7; color: #166534; }}
.score-med  {{ background: #fef9c3; color: #854d0e; }}
.score-low  {{ background: #fee2e2; color: #991b1b; }}

.answer-box {{
    background: #ffffff; border: 1px solid #b2d8d8;
    border-radius: 10px; padding: 22px 26px;
    font-size: 14px; line-height: 1.8; color: #0f2b3d;
    box-shadow: 0 1px 6px rgba(13,148,136,0.08);
}}
.filter-pill {{
    display: inline-block; background: #ccfbf1;
    border: 1px solid #0d948855; color: #0f766e;
    padding: 3px 10px; border-radius: 12px;
    font-size: 11px; margin: 2px; font-weight: 500;
}}
.section-header {{
    font-family: 'DM Serif Display', serif;
    font-size: 17px; color: #0f2b3d;
    margin: 20px 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid #b2d8d8;
}}
.meta-tag {{
    display: inline-block; background: #e0f2f1;
    color: #0f766e; padding: 2px 8px;
    border-radius: 4px; font-size: 10px; margin: 2px;
}}
.confidence-badge {{
    display: inline-block; padding: 3px 12px;
    border-radius: 12px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
}}
.conf-high {{ background: #dcfce7; color: #166534; }}
.conf-med  {{ background: #fef9c3; color: #854d0e; }}
.conf-low  {{ background: #fee2e2; color: #991b1b; }}
.conf-offline {{ background: #e0f2f1; color: #0f766e; }}

.stButton > button {{
    background: #0d9488 !important; color: #ffffff !important;
    border: 1px solid #0b7a72 !important; border-radius: 8px !important;
    font-size: 13px !important; transition: all 0.2s !important;
}}
.stButton > button:hover {{
    background: #0b7a72 !important;
    box-shadow: 0 4px 12px rgba(13,148,136,0.30) !important;
}}
.stTextInput input, .stTextArea textarea, .stSelectbox select {{
    background: #ffffff !important; color: #0f2b3d !important;
    border: 1px solid #b2d8d8 !important; border-radius: 8px !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: #e0f2f1; border-radius: 8px; padding: 4px;
    border: 1px solid #b2d8d8;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important; color: #3a7a7a !important;
    border-radius: 6px !important; font-size: 13px !important;
}}
.stTabs [aria-selected="true"] {{
    background: #0d9488 !important; color: #ffffff !important;
}}
.stAlert {{ background: #f0fdfa !important; border: 1px solid #b2d8d8 !important; border-radius: 8px !important; }}
hr {{ border-color: #b2d8d8 !important; }}
.streamlit-expanderHeader {{
    background: #ffffff !important; border: 1px solid #b2d8d8 !important;
    border-radius: 8px !important; color: #0f2b3d !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key in ["conversation", "last_chunks", "last_filters", "last_answer"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["conversation", "last_chunks"] else None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 0 8px">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:34px;height:34px;background:#0d9488;border-radius:8px;
                        display:flex;align-items:center;justify-content:center;font-size:18px">🧠</div>
            <div>
                <div style="font-size:16px;font-weight:600;color:#0f2b3d">RAG Enterprise</div>
                <div style="font-size:10px;color:#3a7a7a;text-transform:uppercase;letter-spacing:2px">v1.0</div>
            </div>
        </div>
    </div>
    <hr style="border-color:#b2d8d8;margin:10px 0 14px">
    """, unsafe_allow_html=True)

    # API Status
    api_ok = bool(GROQ_API_KEY and GROQ_API_KEY != "gsk_your_groq_api_key_here")
    if api_ok:
        st.markdown('<div style="background:#ccfbf1;color:#0f766e;border:1px solid #0d948855;padding:6px 12px;border-radius:8px;font-size:12px">🟢 Groq API connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:#fee2e2;color:#991b1b;border:1px solid #f8717155;padding:6px 12px;border-radius:8px;font-size:12px">🔴 Add Groq key in config.py</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Collection stats
    st.markdown("**📊 Knowledge Base**")
    stats = get_collection_stats()
    c1, c2 = st.columns(2)
    c1.metric("Chunks", stats["total_chunks"])
    c2.metric("Docs", len(stats["documents"]))
    if stats["domains"]:
        st.markdown("**Domains:** " + " · ".join(f'<span style="background:#ccfbf1;color:#0f766e;padding:2px 8px;border-radius:4px;font-size:11px">{d}</span>' for d in stats["domains"]), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Chunking Settings**")
    use_llm_chunking = st.checkbox(
        "Proposition-based chunking",
        value=api_ok,
        help="Uses LLM to decompose text into atomic propositions. Requires Groq API key."
    )
    top_k = st.slider("Retrieve top-K chunks", 3, 10, 5)
    use_metadata_filter = st.checkbox("Auto metadata filtering", value=True,
                                       help="LLM extracts filters from your query automatically")

    st.markdown("---")
    st.markdown("**🗂️ Documents**")
    if stats["documents"]:
        for doc in stats["documents"]:
            col_d, col_x = st.columns([3, 1])
            col_d.markdown(f'<div style="font-size:11px;color:#3a7a7a;padding:3px 0">{doc[:22]}...</div>' if len(doc) > 22 else f'<div style="font-size:11px;color:#3a7a7a;padding:3px 0">{doc}</div>', unsafe_allow_html=True)
            if col_x.button("✕", key=f"del_{doc}", help=f"Remove {doc}"):
                n = delete_document(doc)
                st.success(f"Removed {n} chunks")
                st.rerun()
    else:
        st.markdown('<div style="font-size:12px;color:#3a7a7a">No documents yet</div>', unsafe_allow_html=True)

    if stats["total_chunks"] > 0:
        if st.button("🗑️ Clear All", use_container_width=True):
            clear_collection()
            st.session_state["conversation"] = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px;color:#3a7a7a;text-align:center;line-height:1.8">
    RAG Enterprise Agent<br>
    Proposition Chunking + Metadata Filtering<br>
    <span style="color:#0d9488;font-weight:500">ChromaDB + HuggingFace + Groq</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="header-banner">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
            <h1 class="header-title">🧠 RAG Enterprise Agent</h1>
            <p class="header-sub">Proposition-Based Chunking · LLM Metadata Filtering · Vector Search</p>
            <span class="header-badge">{"🟢 LLM ACTIVE" if api_ok else "🟡 OFFLINE MODE"}</span>
        </div>
        <div style="text-align:right">
            <div style="font-size:10px;color:#3a7a7a;text-transform:uppercase;letter-spacing:1px">Session</div>
            <div style="font-size:13px;color:#0d9488;font-weight:500;margin-top:4px">{datetime.now().strftime("%b %d, %Y · %H:%M")}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATS ROW
# ─────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{stats["total_chunks"]}</div><div class="stat-lbl">Total Chunks</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{len(stats["documents"])}</div><div class="stat-lbl">Documents</div></div>', unsafe_allow_html=True)
with s3:
    prop_count = stats.get("chunk_types", {}).get("proposition", 0)
    st.markdown(f'<div class="stat-card"><div class="stat-val">{prop_count}</div><div class="stat-lbl">Propositions</div></div>', unsafe_allow_html=True)
with s4:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{len(st.session_state["conversation"])}</div><div class="stat-lbl">Q&A Turns</div></div>', unsafe_allow_html=True)
st.markdown("")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Upload & Index", "💬 Query Knowledge Base", "🔍 Inspect Chunks"])

# ══════════════════════════════════════════════
# TAB 1 — UPLOAD & INDEX
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📄 Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="You can upload multiple files at once"
    )

    # Metadata fields
    st.markdown('<div class="section-header">🏷️ Define Metadata Fields</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#3a7a7a;margin-bottom:12px">These fields power the LLM metadata filtering — be specific for better retrieval.</div>', unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        meta_domain   = st.text_input("Domain", placeholder="e.g. banking, hr, legal, finance")
        meta_category = st.text_input("Category", placeholder="e.g. policy, report, guide")
    with mc2:
        meta_author   = st.text_input("Author", placeholder="e.g. John Doe")
        meta_date     = st.text_input("Date / Year", placeholder="e.g. 2024, Q1-2024")
    with mc3:
        meta_source   = st.text_input("Source", placeholder="e.g. RBI Guidelines, Internal")
        meta_tags     = st.text_input("Tags", placeholder="e.g. credit, risk, compliance")

    metadata_fields = {
        "domain":   meta_domain,
        "category": meta_category,
        "author":   meta_author,
        "date":     meta_date,
        "source":   meta_source,
        "tags":     meta_tags,
    }

    st.markdown("")
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        index_btn = st.button("⚡ Chunk & Index Documents", use_container_width=True,
                               disabled=not uploaded_files)
    with col_info:
        mode = "Proposition-based (LLM)" if use_llm_chunking else "Sentence-based (offline)"
        st.markdown(f'<div style="padding:8px 0;font-size:12px;color:#8a7030">Chunking mode: <strong style="color:#B8892A">{mode}</strong></div>', unsafe_allow_html=True)

    if index_btn and uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(f"**Processing:** `{uploaded_file.name}`")

            # Load
            with st.spinner(f"Loading {uploaded_file.name}..."):
                doc = load_document(uploaded_file, metadata_fields)

            word_count = doc["metadata"].get("word_count", "?")
            st.markdown(f'<div style="font-size:12px;color:#3a7a7a">Loaded — {word_count} words</div>', unsafe_allow_html=True)

            # Chunk
            progress_bar = st.progress(0)
            status_text  = st.empty()

            def update_progress(current, total, msg):
                pct = int((current / max(total, 1)) * 100)
                progress_bar.progress(pct)
                status_text.markdown(f'<div style="font-size:11px;color:#3a7a7a">{msg}</div>', unsafe_allow_html=True)

            with st.spinner("Chunking into propositions..."):
                chunks = chunk_document(
                    doc["text"],
                    doc["metadata"],
                    use_llm=use_llm_chunking,
                    progress_callback=update_progress
                )

            progress_bar.progress(100)
            status_text.markdown(f'<div style="font-size:12px;color:#3B6D11">✅ Created {len(chunks)} chunks</div>', unsafe_allow_html=True)

            # Ingest
            with st.spinner("Embedding and storing in ChromaDB..."):
                result = ingest_chunks(chunks)

            st.markdown(f"""
            <div style="background:#ccfbf1;border:1px solid #0d948855;border-radius:8px;padding:12px 16px;margin:8px 0">
                <strong style="color:#0f766e">✅ Indexed: {uploaded_file.name}</strong><br>
                <span style="font-size:12px;color:#0d9488">
                {result['ingested']} chunks ingested · {result['skipped']} skipped
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.rerun()

    # How it works explanation
    st.markdown('<div class="section-header">🧠 How Proposition-Based Chunking Works</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="chunk-card">
            <div style="font-size:13px;font-weight:600;color:#0d9488;margin-bottom:8px">Step 1 — Split</div>
            <div style="font-size:12px;color:#3a7a7a;line-height:1.7">
            Document is split into paragraphs based on structure and length.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="chunk-card">
            <div style="font-size:13px;font-weight:600;color:#0d9488;margin-bottom:8px">Step 2 — Decompose</div>
            <div style="font-size:12px;color:#3a7a7a;line-height:1.7">
            LLM breaks each paragraph into atomic propositions — one self-contained fact per chunk.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="chunk-card">
            <div style="font-size:13px;font-weight:600;color:#0d9488;margin-bottom:8px">Step 3 — Embed & Store</div>
            <div style="font-size:12px;color:#3a7a7a;line-height:1.7">
            Each proposition is embedded with HuggingFace and stored in ChromaDB with metadata.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — QUERY
# ══════════════════════════════════════════════
with tab2:
    if stats["total_chunks"] == 0:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:#3a7a7a">
            <div style="font-size:48px;margin-bottom:12px">📭</div>
            <div style="font-size:16px;font-weight:500;color:#0f2b3d">No documents indexed yet</div>
            <div style="font-size:13px;margin-top:8px">Go to the Upload tab and add some documents first</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">💬 Ask Your Knowledge Base</div>', unsafe_allow_html=True)

        # Query input
        query = st.text_input(
            "",
            placeholder="e.g. What are the credit risk policies for banking customers in 2024?",
            label_visibility="collapsed"
        )

        # Manual filter override
        with st.expander("🎛️ Manual Metadata Filters (optional — overrides auto-detection)"):
            avail = stats.get("domains", [])
            mf1, mf2, mf3 = st.columns(3)
            with mf1:
                manual_domain = st.selectbox("Domain", ["(auto)"] + avail)
            with mf2:
                manual_date = st.text_input("Date filter", placeholder="e.g. 2024")
            with mf3:
                manual_category = st.text_input("Category filter", placeholder="e.g. policy")

        col_q1, col_q2 = st.columns([1, 1])
        with col_q1:
            ask_btn = st.button("🔍 Search & Answer", use_container_width=True, disabled=not query)
        with col_q2:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state["conversation"] = []
                st.session_state["last_answer"] = None
                st.rerun()

        if ask_btn and query:
            with st.spinner("Extracting metadata filters..."):
                # Build available field values for filter agent
                available_values = {}
                if stats["domains"]:
                    available_values["domain"] = stats["domains"]

                available_fields = ["domain", "date", "category", "author", "source", "tags"]

                # Auto extract filters
                if use_metadata_filter:
                    auto_filters = extract_metadata_filters(query, available_fields, available_values)
                else:
                    auto_filters = {}

                # Apply manual overrides
                if manual_domain and manual_domain != "(auto)":
                    auto_filters["domain"] = manual_domain
                if manual_date:
                    auto_filters["date"] = manual_date
                if manual_category:
                    auto_filters["category"] = manual_category

                chroma_filter = build_chroma_filter(auto_filters)

            with st.spinner("Searching vector database..."):
                results = search(query, top_k=top_k, metadata_filter=chroma_filter)

                # Fallback: if filtered search returns nothing, search without filter
                if not results and chroma_filter:
                    results = search(query, top_k=top_k, metadata_filter=None)
                    auto_filters = {}
                    chroma_filter = None

            with st.spinner("Generating grounded answer..."):
                answer_result = generate_answer(
                    query,
                    results,
                    filters_used=auto_filters,
                    conversation_history=st.session_state["conversation"]
                )

            # Store in session
            st.session_state["conversation"].append({
                "question": query,
                "answer":   answer_result["answer"],
                "filters":  auto_filters,
                "chunks":   results,
                "confidence": answer_result["confidence"]
            })
            st.session_state["last_chunks"]  = results
            st.session_state["last_filters"] = auto_filters
            st.session_state["last_answer"]  = answer_result

        # Display current answer
        if st.session_state["last_answer"]:
            ans = st.session_state["last_answer"]
            filters = st.session_state["last_filters"] or {}
            chunks  = st.session_state["last_chunks"]  or []

            # Filter pills
            if filters:
                pills = "".join(f'<span class="filter-pill">🏷️ {k}: {v}</span>' for k, v in filters.items())
                st.markdown(f'<div style="margin-bottom:10px">{pills}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:12px;color:#3a7a7a;margin-bottom:8px">No metadata filters — searching full knowledge base</div>', unsafe_allow_html=True)

            # Confidence badge
            conf = ans["confidence"]
            conf_class = {"high": "conf-high", "medium": "conf-med", "low": "conf-low"}.get(conf, "conf-offline")
            conf_label = {"high": "High Confidence", "medium": "Medium Confidence", "low": "Low Confidence", "offline": "Offline Mode", "error": "Error"}.get(conf, conf)

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
                <span class="confidence-badge {conf_class}">{conf_label}</span>
                <span style="font-size:12px;color:#3a7a7a">{ans["chunk_count"]} chunks retrieved · Sources: {", ".join(ans["sources"]) or "none"}</span>
            </div>
            """, unsafe_allow_html=True)

            # Answer
            st.markdown('<div class="section-header">🤖 Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{ans["answer"]}</div>', unsafe_allow_html=True)

            # Retrieved chunks
            st.markdown('<div class="section-header">📎 Retrieved Chunks</div>', unsafe_allow_html=True)
            for i, chunk in enumerate(chunks):
                score = chunk.get("score", 0)
                score_class = "score-high" if score >= 0.7 else ("score-med" if score >= 0.5 else "score-low")
                meta = chunk["metadata"]
                meta_tags = "".join([
                    f'<span class="meta-tag">{k}: {v}</span>'
                    for k, v in meta.items()
                    if k in ["domain", "category", "date", "filename", "chunk_type"] and v
                ])
                st.markdown(f"""
                <div class="chunk-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                        <div style="font-size:11px;color:#3a7a7a;font-weight:500">CHUNK {i+1}</div>
                        <span class="chunk-score {score_class}">Score: {score:.3f}</span>
                    </div>
                    <div style="font-size:13px;color:#0f2b3d;line-height:1.7;margin-bottom:8px">{chunk["text"]}</div>
                    <div>{meta_tags}</div>
                </div>
                """, unsafe_allow_html=True)

        # Conversation history
        if len(st.session_state["conversation"]) > 1:
            st.markdown('<div class="section-header">🗂️ Conversation History</div>', unsafe_allow_html=True)
            for turn in reversed(st.session_state["conversation"][:-1]):
                with st.expander(f"Q: {turn['question'][:70]}..."):
                    st.markdown(f"**Filters:** {turn.get('filters', {}) or 'None'}")
                    st.markdown(f"**Confidence:** {turn.get('confidence', 'N/A')}")
                    st.markdown(turn["answer"])

# ══════════════════════════════════════════════
# TAB 3 — INSPECT CHUNKS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🔍 Inspect Vector Store</div>', unsafe_allow_html=True)

    if stats["total_chunks"] == 0:
        st.info("No chunks stored yet. Upload documents in the Upload tab.")
    else:
        from agents.vector_store import get_collection

        # Stats
        t1, t2, t3 = st.columns(3)
        t1.metric("Total Chunks", stats["total_chunks"])
        t2.metric("Proposition Chunks", stats["chunk_types"].get("proposition", 0))
        t3.metric("Sentence Chunks", stats["chunk_types"].get("sentence", 0))

        st.markdown("**Documents in knowledge base:**")
        for doc in stats["documents"]:
            st.markdown(f'<div style="padding:3px 0;font-size:13px;color:#3a7a7a">— <code style="background:#e0f2f1;color:#0f766e;padding:2px 6px;border-radius:4px">{doc}</code></div>', unsafe_allow_html=True)

        if stats["domains"]:
            st.markdown("**Domains:** " + " · ".join(f'<span style="background:#ccfbf1;color:#0f766e;padding:2px 8px;border-radius:4px;font-size:11px">{d}</span>' for d in stats["domains"]), unsafe_allow_html=True)

        # Browse chunks
        st.markdown('<div class="section-header">Browse Chunks</div>', unsafe_allow_html=True)
        browse_query = st.text_input("Search chunks by keyword", placeholder="e.g. credit risk")
        if browse_query:
            browse_results = search(browse_query, top_k=8)
            for r in browse_results:
                score = r.get("score", 0)
                score_class = "score-high" if score >= 0.7 else ("score-med" if score >= 0.5 else "score-low")
                st.markdown(f"""
                <div class="chunk-card">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                        <span style="font-size:11px;color:#3a7a7a">{r["metadata"].get("filename","?")}</span>
                        <span class="chunk-score {score_class}">Score: {score:.3f}</span>
                    </div>
                    <div style="font-size:13px;color:#0f2b3d">{r["text"]}</div>
                </div>
                """, unsafe_allow_html=True)
