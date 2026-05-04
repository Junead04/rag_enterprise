# 🧠 RAG Enterprise Agent
### Proposition-Based Chunking + LLM Metadata Filtering + Vector Search

> Stack: Python · ChromaDB · HuggingFace · Groq LLM · Streamlit

---

## 🏗️ Project Structure

```
rag_enterprise/
├── app.py                          ← Main Streamlit app
├── config.py                       ← API keys + settings
├── requirements.txt
├── .gitignore
├── agents/
│   ├── chunking_agent.py           ← Proposition-based chunking (CORE)
│   ├── vector_store.py             ← ChromaDB ingest + search
│   ├── metadata_filter_agent.py    ← LLM metadata extraction
│   └── rag_agent.py                ← Answer generation
├── utils/
│   └── document_loader.py          ← PDF, TXT, DOCX loader
├── data/
│   └── sample_banking_policy.txt   ← Test document
├── assets/
│   └── background.png              ← UI background image
└── vectorstore/                    ← Auto-created by ChromaDB
```

---

## 🚀 Setup — Step by Step

### Step 1 — Python version
Make sure you have **Python 3.10 or 3.11** (NOT 3.12+)
```bash
python --version
```

### Step 2 — Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
> This will take 3-5 minutes — HuggingFace model download is ~90MB

### Step 4 — Add your Groq API key
Open `config.py` and replace:
```python
GROQ_API_KEY = "gsk_your_actual_key_here"
```
Get a free key at: **console.groq.com**

### Step 5 — Run the app
```bash
streamlit run app.py
```
Opens at: **http://localhost:8501**

---

## 🎯 How to Use

### Upload Documents
1. Go to **Upload & Index** tab
2. Upload a PDF, TXT, or DOCX file
3. Fill in metadata fields: domain, category, date, author
4. Click **Chunk & Index Documents**
5. Wait for chunking + embedding (first run downloads HuggingFace model)

### Query
1. Go to **Query Knowledge Base** tab
2. Type a natural language question
3. The system automatically:
   - Extracts metadata filters from your query
   - Searches ChromaDB with cosine similarity
   - Generates a grounded answer via Groq LLM

### Inspect
1. Go to **Inspect Chunks** tab
2. Browse indexed propositions
3. Search chunks by keyword

---

## 🧠 How It Works — Deep Explanation

### Problem with Naive Chunking
Fixed-size chunking (e.g. 500 chars) cuts text arbitrarily:
- Sentences get split mid-thought
- A chunk may contain multiple unrelated ideas
- Cosine similarity matches poorly → wrong chunks retrieved → hallucination

### Proposition-Based Chunking
1. **Split** document into paragraphs
2. **LLM decomposes** each paragraph into atomic propositions
   - Each proposition = one self-contained fact
   - Example: "The minimum credit score is 700" is one proposition
3. **Embed** each proposition with HuggingFace sentence-transformers
4. **Store** in ChromaDB with metadata

### Metadata Filtering
1. User asks: "What are credit policies for banking in 2024?"
2. LLM extracts: `{"domain": "banking", "date": "2024"}`
3. ChromaDB filters by these before vector search
4. Only relevant domain/time chunks are searched
5. Result: more precise, less noise, less hallucination

---
- **How to scale this?** Replace ChromaDB with Milvus, add reranking with cross-encoders, add hybrid BM25 + vector search

---

*RAG Enterprise Agent · Portfolio Project · Inspired by Katonic AI*
