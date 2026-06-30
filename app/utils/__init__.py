# app/utils/__init__.py
# Marks the `utils` directory as a Python package.
# Pure, stateless helper functions shared across the application will live here.
# Planned contents:
#   - file_handler.py    (safe file I/O, extension validation, path helpers)
#   - text_splitter.py   (chunking strategies for document ingestion)
#   - logger.py          (structured logging configuration)
#   - validators.py      (reusable input validation helpers)
# Utils must not import from app.services or app.api to avoid circular dependencies.

from app.utils.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token,
    get_token_expiry,
)
from app.utils.file_utils import (
    is_allowed_file,
    ensure_upload_directories,
    get_upload_path,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
)
from app.utils.pdf_reader import read_pdf
from app.utils.docx_reader import read_docx
from app.utils.text_reader import read_text


