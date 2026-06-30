# app/utils/pdf_reader.py
# -----------------------
# Utility for extracting plain text from PDF files using PyMuPDF (fitz).

import os
import fitz  # PyMuPDF


def read_pdf(file_path: str) -> str:
    """
    Open a PDF file and extract all text content page by page.
    Raises FileNotFoundError if the file is missing.
    Raises ValueError if the PDF is empty, corrupted, or unreadable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    text_content = []
    try:
        with fitz.open(file_path) as doc:
            if doc.page_count == 0:
                raise ValueError("PDF document is empty")
                
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text_content.append(page_text)
                    
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise ValueError(f"Corrupted or invalid PDF file: {e}")

    combined_text = "\n".join(text_content)
    if not combined_text.strip():
        raise ValueError("No text could be extracted from the PDF")

    return combined_text
