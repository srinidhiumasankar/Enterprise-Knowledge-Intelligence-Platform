# app/services/langchain/health_service.py
# ----------------------------------------
# Service layer managing configuration, wraps, exports, and lifespan of the HealthMonitor.

import logging
from typing import Any, Dict, List, Optional
from app.services.langchain.health_monitor import HealthMonitor, HealthReport

logger = logging.getLogger(__name__)


class HealthService:
    """
    Service wrapper managing telemetry resets, serializations, and access to HealthMonitor.
    """
    def __init__(self):
        self.monitor = HealthMonitor.get_instance()

    def get_monitor(self) -> HealthMonitor:
        """
        Returns the core HealthMonitor manager instance.
        """
        return self.monitor

    def export_reports(self) -> List[Dict[str, Any]]:
        """
        Exports health report diagnostics history list to dictionaries.
        """
        return self.monitor.export_report()

    def reset_health(self):
        """
        Clears status records and diagnostic logs database.
        """
        self.monitor.reset()
