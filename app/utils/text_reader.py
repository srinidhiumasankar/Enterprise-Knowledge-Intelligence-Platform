# app/utils/text_reader.py
# ------------------------
# Utility for safely reading plain text files.

import os


def read_text(file_path: str) -> str:
    """
    Open a TXT file and return its text content safely.
    Raises FileNotFoundError if the file is missing.
    Raises ValueError if the text file is empty or unreadable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TXT file not found at path: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding if UTF-8 fails
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Could not decode text file: {e}")
    except Exception as e:
        raise ValueError(f"Unreadable text file: {e}")

    if not content.strip():
        raise ValueError("Text file is empty")

    return content
