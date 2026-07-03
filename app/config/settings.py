# app/config/settings.py
# ----------------------
# Application configuration settings, loaded from environment variables or .env file.

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings container.
    Loads values from environment variables and an optional .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application Settings
    APP_NAME: str = "Enterprise Knowledge Intelligence Platform"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_VERSION: str = "0.2.0"

    # Database Settings
    DATABASE_URL: str = "sqlite:///./ekip.db"

    # Security Settings
    JWT_SECRET_KEY: str = "change-me-to-a-secure-random-string"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Embedding Settings
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768

    # Vector Database Settings
    CHROMA_DB_DIR: str = "chroma_db"
    CHROMA_DB_PATH: str = "./vector_store"
    CHROMA_COLLECTION: str = "document_chunks"
    CHROMA_COLLECTION_NAME: str = "document_chunks"
    
    # Search Settings
    TOP_K_DEFAULT: int = 5
    TOP_K_RESULTS: int = 5
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.55

    # AI / LLM Settings
    GEMINI_API_KEY: str = ""

    @field_validator("GEMINI_API_KEY")
    @classmethod
    def validate_gemini_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Configuration Error: GEMINI_API_KEY is missing. "
                "Please configure GEMINI_API_KEY in your environment or .env file."
            )
        return v


settings = Settings()

