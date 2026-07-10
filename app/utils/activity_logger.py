# app/utils/activity_logger.py
# ----------------------------
# Helper function to write generic activity records to the database.

import logging
from typing import Optional
from sqlalchemy.orm import Session
from app.models.activity_log import ActivityLog

logger = logging.getLogger(__name__)


def log_activity(
    db: Session,
    user_id: Optional[int],
    workspace_id: Optional[int],
    event_type: str,
    description: str
) -> Optional[ActivityLog]:
    """
    Creates and persists a generic activity log entry in the database.
    """
    try:
        log = ActivityLog(
            user_id=user_id,
            workspace_id=workspace_id,
            event_type=event_type,
            description=description
        )
        db.add(log)
        db.commit()
        logger.info(f"Workspace event logged: '{event_type}' - {description}")
        return log
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to record activity log '{event_type}': {e}", exc_info=True)
        return None


def record_rag_search_history(
    user_id: int,
    workspace_id: int,
    query: str,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    total_latency_ms: float,
    result_count: int,
    retrieved_document_names: list,
    similarity_scores: list,
    response_status: str
):
    """
    Asynchronously records search history entries and search_executed activity events.
    """
    from app.database.connection import SessionLocal
    from app.models.search_history import SearchHistory
    from app.models.activity_log import ActivityLog
    
    db = SessionLocal()
    try:
        filters_json = {
            "retrieval_latency_ms": int(retrieval_latency_ms),
            "generation_latency_ms": int(generation_latency_ms),
            "total_latency_ms": int(total_latency_ms),
            "retrieved_document_names": retrieved_document_names,
            "similarity_scores": [float(score) for score in similarity_scores],
            "query_length": len(query),
            "response_status": response_status
        }
        
        # Save to SearchHistory
        history = SearchHistory(
            user_id=user_id,
            workspace_id=workspace_id,
            query=query.strip(),
            filters_json=filters_json,
            execution_time_ms=int(total_latency_ms),
            result_count=result_count
        )
        db.add(history)
        
        # Save to ActivityLog
        log = ActivityLog(
            user_id=user_id,
            workspace_id=workspace_id,
            event_type="search_executed",
            description=f"Executed search query: '{query}'"
        )
        db.add(log)
        
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to record RAG search metrics: {e}", exc_info=True)
    finally:
        db.close()
