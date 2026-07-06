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
    HYBRID_SEMANTIC_WEIGHT: float = 0.7
    HYBRID_KEYWORD_WEIGHT: float = 0.3

    # Adaptive Retriever Settings
    ENABLE_ADAPTIVE_RETRIEVAL: bool = True
    ADAPTIVE_RULES: dict = {
        "simple_factual": "hybrid",
        "metadata_filtering": "self_query",
        "long_explanatory": "parent",
        "ambiguous": "multi_query",
        "comparison_multi_topic": "ensemble"
    }

    # Query Rewriter Settings
    ENABLE_QUERY_REWRITER: bool = True
    QUERY_REWRITE_RULES: dict = {
        "finance": "finance reports and financial documents",
        "vacation": "employee vacation leave policy",
        "ai": "Artificial Intelligence",
        "leave after 2022": "employee leave policy after year 2022"
    }
    SYNONYM_MAP: dict = {
        "vacation": "vacation leave policy",
        "leave": "employee leave policy",
        "report": "business report",
        "policy": "company policy"
    }
    ABBREVIATION_MAP: dict = {
        "ai": "Artificial Intelligence",
        "ml": "Machine Learning",
        "pto": "Paid Time Off",
        "hr": "Human Resources",
        "it": "Information Technology"
    }

    # Conversation Memory Settings
    ENABLE_CONVERSATION_MEMORY: bool = True
    MAX_CONVERSATION_MESSAGES: int = 20
    MAX_CONVERSATION_TOKENS: int = 4000
    ENABLE_HISTORY_TRIMMING: bool = True
    MAX_HISTORY_MESSAGES: int = 20

    # Collection Filtering Settings
    ENABLE_COLLECTION_FILTERING: bool = True
    MAX_COLLECTION_FILTERS: int = 20

    # Workspace Settings
    ENABLE_MULTI_WORKSPACE: bool = True
    DEFAULT_WORKSPACE_NAME: str = "My Workspace"
    ENABLE_WORKSPACE_SWITCHING: bool = True
    CACHE_ACTIVE_WORKSPACE: bool = True

    # Search History Settings
    ENABLE_SEARCH_HISTORY: bool = True
    MAX_RECENT_SEARCHES: int = 50
    ENABLE_SEARCH_ANALYTICS: bool = True

    # Dashboard Settings
    ENABLE_DASHBOARD: bool = True
    DASHBOARD_ACTIVITY_LIMIT: int = 20

    # Metadata Ranker Settings
    ENABLE_METADATA_RANKER: bool = True
    METADATA_RANKING_WEIGHTS: dict = {
        "semantic": 0.45,
        "rrf": 0.20,
        "freshness": 0.10,
        "importance": 0.10,
        "type": 0.05,
        "citation": 0.05,
        "completeness": 0.05,
    }
    FRESHNESS_WEIGHT: float = 0.10
    IMPORTANCE_WEIGHT: float = 0.10
    TYPE_WEIGHT: float = 0.05
    CITATION_WEIGHT: float = 0.05

    # Result Scorer Settings
    ENABLE_RESULT_SCORER: bool = True
    RESULT_SCORING_WEIGHTS: dict = {
        "semantic": 0.30,
        "metadata": 0.15,
        "rrf": 0.15,
        "agreement": 0.10,
        "chunk_completeness": 0.10,
        "citation_completeness": 0.10,
        "history_boost": 0.05,
        "rewrite_boost": 0.05,
    }
    MIN_CONFIDENCE_SCORE: float = 0.0

    # Answer Verifier Settings
    ENABLE_ANSWER_VERIFIER: bool = True
    GROUNDING_THRESHOLD: float = 70.0
    HALLUCINATION_THRESHOLD: float = 30.0
    MIN_SUPPORTED_KEYWORDS: int = 3

    # Retrieval Analytics Settings
    ENABLE_RETRIEVAL_ANALYTICS: bool = True
    EXPORT_ANALYTICS: bool = False
    MAX_ANALYTICS_HISTORY: int = 1000
    ANALYTICS_VERBOSE: bool = True

    # Health Monitor Settings
    ENABLE_HEALTH_MONITOR: bool = True
    HEALTH_WARNING_LATENCY_MS: float = 500.0
    ENABLE_DIAGNOSTICS: bool = True
    EXPORT_HEALTH_REPORT: bool = False

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

