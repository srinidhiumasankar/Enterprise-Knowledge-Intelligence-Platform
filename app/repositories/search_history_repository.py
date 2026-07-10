# app/repositories/search_history_repository.py
# ---------------------------------------------
# Data access repository layer for managing search query history.

import logging
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import select, func, desc, delete
from sqlalchemy.orm import Session

from app.models.search_history import SearchHistory

logger = logging.getLogger(__name__)


class SearchHistoryRepository:
    """
    Repository class encapsulating database CRUD operations for SearchHistory.
    """
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        user_id: int,
        workspace_id: int,
        query: str,
        filters_json: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[int] = None,
        result_count: Optional[int] = None
    ) -> SearchHistory:
        """
        Records a new search entry in database.
        """
        history = SearchHistory(
            user_id=user_id,
            workspace_id=workspace_id,
            query=query.strip(),
            filters_json=filters_json,
            execution_time_ms=execution_time_ms,
            result_count=result_count
        )
        self.db.add(history)
        self.db.commit()
        self.db.refresh(history)
        logger.info(f"Recorded search history ID {history.id} for user {user_id}")
        return history

    def get(self, history_id: int) -> Optional[SearchHistory]:
        """
        Retrieves a history entry by ID.
        """
        return self.db.get(SearchHistory, history_id)

    def list(self, user_id: int, workspace_id: int, page: int = 1, page_size: int = 20) -> Tuple[List[SearchHistory], int]:
        """
        Lists search history belonging to the user and workspace with pagination.
        """
        query = select(SearchHistory).where(
            SearchHistory.user_id == user_id,
            SearchHistory.workspace_id == workspace_id
        )
        
        # Count total records
        count_query = select(func.count()).select_from(query.subquery())
        total_records = self.db.scalar(count_query) or 0
        
        # Offsets & ordering (newest first)
        query = query.order_by(SearchHistory.created_at.desc(), SearchHistory.id.desc())
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        items = list(self.db.scalars(query).all())
        return items, total_records

    def recent(self, user_id: int, workspace_id: int, limit: int = 50) -> List[SearchHistory]:
        """
        Returns recent search entries.
        """
        query = select(SearchHistory).where(
            SearchHistory.user_id == user_id,
            SearchHistory.workspace_id == workspace_id
        ).order_by(
            SearchHistory.created_at.desc(),
            SearchHistory.id.desc()
        ).limit(limit)
        return list(self.db.scalars(query).all())

    def frequent(self, user_id: int, workspace_id: int, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Returns frequently searched queries sorted by search count.
        """
        query = select(
            SearchHistory.query,
            func.count(SearchHistory.id).label("search_count")
        ).where(
            SearchHistory.user_id == user_id,
            SearchHistory.workspace_id == workspace_id
        ).group_by(
            SearchHistory.query
        ).order_by(
            desc("search_count")
        ).limit(limit)
        
        res = self.db.execute(query).all()
        return [(row[0], row[1]) for row in res]

    def delete(self, history_id: int) -> bool:
        """
        Deletes a single history record.
        """
        entry = self.db.get(SearchHistory, history_id)
        if entry:
            self.db.delete(entry)
            self.db.commit()
            logger.info(f"Deleted search history entry ID {history_id}")
            return True
        return False

    def clear(self, user_id: int, workspace_id: int) -> bool:
        """
        Clears all search history for a user in the active workspace.
        """
        stmt = delete(SearchHistory).where(
            SearchHistory.user_id == user_id,
            SearchHistory.workspace_id == workspace_id
        )
        self.db.execute(stmt)
        self.db.commit()
        logger.info(f"Cleared search history for user {user_id} in workspace {workspace_id}")
        return True

    def statistics(self, user_id: int, workspace_id: int) -> Dict[str, Any]:
        """
        Aggregates search history analytics inside a workspace.
        """
        now = datetime.utcnow()
        today_start = datetime(now.year, now.month, now.day)
        weekly_start = now - timedelta(days=7)
        monthly_start = now - timedelta(days=30)

        base_where = (SearchHistory.user_id == user_id, SearchHistory.workspace_id == workspace_id)

        # Count metrics
        total = self.db.scalar(select(func.count(SearchHistory.id)).where(*base_where)) or 0
        today = self.db.scalar(select(func.count(SearchHistory.id)).where(*base_where, SearchHistory.created_at >= today_start)) or 0
        weekly = self.db.scalar(select(func.count(SearchHistory.id)).where(*base_where, SearchHistory.created_at >= weekly_start)) or 0
        monthly = self.db.scalar(select(func.count(SearchHistory.id)).where(*base_where, SearchHistory.created_at >= monthly_start)) or 0

        # Calculations
        stats_query = select(
            func.avg(func.length(SearchHistory.query)).label("avg_len"),
            func.avg(SearchHistory.execution_time_ms).label("avg_latency"),
            func.max(SearchHistory.created_at).label("last_search")
        ).where(*base_where)

        stats_res = self.db.execute(stats_query).first()
        avg_len = float(stats_res.avg_len) if stats_res and stats_res.avg_len is not None else 0.0
        avg_latency = float(stats_res.avg_latency) if stats_res and stats_res.avg_latency is not None else 0.0
        last_search_time = stats_res.last_search if stats_res else None

        # Most frequent query
        freq_query = select(
            SearchHistory.query,
            func.count(SearchHistory.id).label("count")
        ).where(*base_where).group_by(SearchHistory.query).order_by(desc("count")).limit(1)
        freq_res = self.db.execute(freq_query).first()
        most_frequent = freq_res[0] if freq_res else None

        # Top 10 queries
        top_query = select(
            SearchHistory.query,
            func.count(SearchHistory.id).label("count")
        ).where(*base_where).group_by(SearchHistory.query).order_by(desc("count")).limit(10)
        top_res = self.db.execute(top_query).all()
        top_queries = [{"query": row[0], "count": row[1]} for row in top_res]

        # Daily Query Trend (last 7 days)
        trend_query = select(
            func.date(SearchHistory.created_at).label("day"),
            func.count(SearchHistory.id).label("count")
        ).where(
            *base_where,
            SearchHistory.created_at >= weekly_start
        ).group_by(
            func.date(SearchHistory.created_at)
        ).order_by(
            "day"
        )
        trend_res = self.db.execute(trend_query).all()
        
        daily_trend = {}
        for i in range(7):
            d = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_trend[d] = 0
            
        for row in trend_res:
            if row.day:
                daily_trend[row.day] = row.count
                
        daily_trend = dict(sorted(daily_trend.items()))

        return {
            "total_searches": total,
            "today_searches": today,
            "weekly_searches": weekly,
            "monthly_searches": monthly,
            "average_query_length": round(avg_len, 2),
            "average_latency_ms": round(avg_latency, 2),
            "most_frequent_query": most_frequent,
            "last_search_time": last_search_time,
            "top_queries": top_queries,
            "daily_query_trend": daily_trend
        }
