# app/utils/file_utils.py
# -----------------------
# Utility functions for file paths, extension validations, and directory setups.

from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def ensure_upload_directories(base_upload_dir: str = "uploads") -> None:
    """
    Ensure the base upload directory and the subfolders for each supported format exist.
    """
    base_path = Path(base_upload_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    for ext in ALLOWED_EXTENSIONS:
        folder_name = ext.lstrip(".")
        sub_path = base_path / folder_name
        sub_path.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    """
    Check if the file has an allowed file extension.
    """
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def get_upload_path(filename: str, base_upload_dir: str = "uploads") -> Path:
    """
    Retrieve the specific directory path for saving the file based on its extension.
    """
    ext = Path(filename).suffix.lower()
    folder_name = ext.lstrip(".")
    return Path(base_upload_dir) / folder_name
