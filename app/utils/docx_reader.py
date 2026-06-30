# app/utils/docx_reader.py
# ------------------------
# Utility for extracting plain text from DOCX files using python-docx.

import os
from docx import Document


def read_docx(file_path: str) -> str:
    """
    Open a DOCX file and extract text paragraphs.
    Raises FileNotFoundError if the file is missing.
    Raises ValueError if the DOCX is empty, corrupted, or unreadable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DOCX file not found at path: {file_path}")

    try:
        doc = Document(file_path)
        paragraphs_text = [p.text for p in doc.paragraphs if p.text]
    except Exception as e:
        raise ValueError(f"Corrupted or invalid DOCX file: {e}")

    combined_text = "\n".join(paragraphs_text)
    if not combined_text.strip():
        raise ValueError("No text could be extracted from the DOCX")

    return combined_text
