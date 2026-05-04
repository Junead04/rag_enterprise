"""
Document Loader
Handles PDF, TXT, DOCX files and returns raw text + metadata.
"""

import os
import re
from typing import Dict, List
from datetime import datetime


def load_txt(file_obj) -> str:
    return file_obj.read().decode("utf-8", errors="ignore")


def load_pdf(file_obj) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file_obj)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        return f"PDF load error: {e}"


def load_docx(file_obj) -> str:
    try:
        from docx import Document
        import io
        doc = Document(io.BytesIO(file_obj.read()))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)
    except Exception as e:
        return f"DOCX load error: {e}"


def load_document(uploaded_file, metadata_fields: Dict = None) -> Dict:
    """
    Load any supported file and return structured document dict.

    Returns:
        {
            "text": str,
            "metadata": {
                "filename": str,
                "file_type": str,
                "upload_time": str,
                "domain": str,          # user-defined
                "author": str,          # user-defined
                "date": str,            # user-defined
                "tags": str,            # user-defined
                ... any extra fields
            }
        }
    """
    name = uploaded_file.name
    ext  = name.rsplit(".", 1)[-1].lower()

    if ext == "txt":
        text = load_txt(uploaded_file)
    elif ext == "pdf":
        text = load_pdf(uploaded_file)
    elif ext in ["docx", "doc"]:
        text = load_docx(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    # Base metadata
    meta = {
        "filename":    name,
        "file_type":   ext,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "char_count":  str(len(text)),
        "word_count":  str(len(text.split())),
    }

    # Merge user-defined metadata fields
    if metadata_fields:
        for k, v in metadata_fields.items():
            if v and str(v).strip():
                meta[k] = str(v).strip()

    return {"text": text, "metadata": meta}
