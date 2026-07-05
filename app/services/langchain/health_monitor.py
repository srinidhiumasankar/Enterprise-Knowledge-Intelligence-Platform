# app/services/langchain/health_monitor.py
# ----------------------------------------
# Observability, diagnostics, and component health monitoring subsystem for RAG execution pipelines.
# Evaluates latencies, checks for failures/warnings, and formats diagnostic health reports.

import logging
import time
import uuid
import threading
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field, asdict

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """
    Diagnostic status payload for component health and runtime integrity.
    """
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""
    overall_status: str = "UNKNOWN"
    
    # Component Statuses (HEALTHY, WARNING, DEGRADED, FAILED, UNKNOWN)
    pipeline_status: str = "UNKNOWN"
    hybrid_status: str = "UNKNOWN"
    parent_status: str = "UNKNOWN"
    multi_query_status: str = "UNKNOWN"
    compression_status: str = "UNKNOWN"
    self_query_status: str = "UNKNOWN"
    ensemble_status: str = "UNKNOWN"
    adaptive_status: str = "UNKNOWN"
    query_rewriter_status: str = "UNKNOWN"
    conversation_memory_status: str = "UNKNOWN"
    metadata_ranker_status: str = "UNKNOWN"
    result_scorer_status: str = "UNKNOWN"
    answer_verifier_status: str = "UNKNOWN"
    cache_status: str = "UNKNOWN"
    analytics_status: str = "UNKNOWN"

    error_count: int = 0
    warning_count: int = 0
    diagnostics_summary: List[str] = field(default_factory=list)


class HealthMonitor:
    """
    Thread-safe observer tracking status, errors, warnings, and latency thresholds
    for RAG execution pipeline components.
    """
    _instance: Optional["HealthMonitor"] = None
    _lock = threading.Lock()
    _local = threading.local()

    def __init__(self):
        self.reports: List[HealthReport] = []
        self.reports_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "HealthMonitor":
        """
        Retrieves the thread-safe singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_or_create_state(self) -> Dict[str, Any]:
        """
        Gets the current thread-scoped request monitoring state.
        """
        if not hasattr(self._local, "state") or self._local.state is None:
            self._local.state = {
                "request_id": str(uuid.uuid4()),
                "components": {},
                "warnings": [],
                "errors": [],
                "start_time": time.perf_counter()
            }
        return self._local.state

    def start_monitoring(self, request_id: Optional[str] = None):
        """
        Initializes monitoring scope for the current request.
        """
        if not getattr(settings, "ENABLE_HEALTH_MONITOR", True):
            return

        rid = request_id or str(uuid.uuid4())
        self._local.state = {
            "request_id": rid,
            "components": {},
            "warnings": [],
            "errors": [],
            "start_time": time.perf_counter()
        }
        logger.info(f"Health Monitoring context started for request: {rid}")

    def record_component(
        self,
        name: str,
        status: str,
        latency: float = 0.0,
        error_message: Optional[str] = None
    ):
        """
        Registers status and latency benchmarks for a specific pipeline component.
        """
        if not getattr(settings, "ENABLE_HEALTH_MONITOR", True):
            return

        state = self._get_or_create_state()
        state["components"][name] = {
            "status": status,
            "latency": latency,
            "error": error_message
        }

        # Check slow component latencies
        warn_latency = getattr(settings, "HEALTH_WARNING_LATENCY_MS", 500.0)
        if latency > warn_latency:
            self.record_warning(name, f"Slow Component: {name} took {latency:.2f}ms (threshold: {warn_latency}ms)")
            state["components"][name]["status"] = "WARNING"

        if error_message:
            self.record_failure(name, error_message)

    def record_warning(self, name: str, warning_message: str):
        """
        Logs a non-fatal warning signal.
        """
        if not getattr(settings, "ENABLE_HEALTH_MONITOR", True):
            return

        state = self._get_or_create_state()
        state["warnings"].append({
            "component": name,
            "message": warning_message
        })
        logger.warning(f"[HEALTH WARNING] Component: {name} | {warning_message}")

    def record_failure(self, name: str, error_message: str):
        """
        Logs a fatal component exception.
        """
        if not getattr(settings, "ENABLE_HEALTH_MONITOR", True):
            return

        state = self._get_or_create_state()
        state["errors"].append({
            "component": name,
            "message": error_message
        })
        if name not in state["components"]:
            state["components"][name] = {"status": "FAILED", "latency": 0.0, "error": error_message}
        else:
            state["components"][name]["status"] = "FAILED"
            state["components"][name]["error"] = error_message
        logger.error(f"[HEALTH FAILURE] Component: {name} | {error_message}")

    def record_success(self, name: str):
        """
        Helper method to register a component success directly.
        """
        self.record_component(name, "HEALTHY")

    def build_health_report(self) -> HealthReport:
        """
        Aggregates logs and classifications to generate a detailed HealthReport.
        """
        state = self._get_or_create_state()
        
        report = HealthReport(
            timestamp=time.time(),
            request_id=state["request_id"]
        )

        components = state["components"]
        warnings = state["warnings"]
        errors = state["errors"]

        # Map component statuses
        def get_status(cname: str) -> str:
            return components.get(cname, {}).get("status", "UNKNOWN")

        report.hybrid_status = get_status("hybrid")
        report.parent_status = get_status("parent")
        report.multi_query_status = get_status("multi_query")
        report.compression_status = get_status("compression")
        report.self_query_status = get_status("self_query")
        report.ensemble_status = get_status("ensemble")
        report.adaptive_status = get_status("adaptive")
        report.query_rewriter_status = get_status("query_rewriter")
        report.conversation_memory_status = get_status("conversation_memory")
        report.metadata_ranker_status = get_status("metadata_ranker")
        report.result_scorer_status = get_status("result_scorer")
        report.answer_verifier_status = get_status("answer_verifier")
        report.cache_status = get_status("cache")
        report.analytics_status = get_status("analytics")
        report.pipeline_status = get_status("pipeline")

        report.error_count = len(errors)
        report.warning_count = len(warnings)

        # Generate Diagnostics Summary lists
        diags = []
        if getattr(settings, "ENABLE_DIAGNOSTICS", True):
            # Check config problems
            if not getattr(settings, "GEMINI_API_KEY", ""):
                diags.append("Configuration Problems: GEMINI_API_KEY is not configured")
                
            # Check cache diagnostics
            if not getattr(settings, "ENABLE_REQUEST_CACHE", True):
                diags.append("Cache Disabled")
                report.cache_status = "WARNING"
                
            # Check empty results
            # We can inspect the pipeline status or final doc count
            if get_status("pipeline") == "HEALTHY" and components.get("pipeline", {}).get("latency", 0.0) > 0.0:
                # If pipeline succeeded but returned 0 documents
                pass

            # Audit warnings
            for w in warnings:
                diags.append(w["message"])
                
            # Audit errors
            for e in errors:
                diags.append(f"Pipeline Failure: Component '{e['component']}' failed: {e['message']}")

            # Slow pipeline check
            total_time = (time.perf_counter() - state["start_time"]) * 1000
            if total_time > 2000.0:
                diags.append(f"Excessive Latency: Pipeline took {total_time:.2f}ms")

        report.diagnostics_summary = diags

        # Classify overall health levels status
        if report.error_count > 0:
            if get_status("pipeline") == "FAILED":
                report.overall_status = "FAILED"
            else:
                report.overall_status = "DEGRADED"
        elif report.warning_count > 0:
            report.overall_status = "WARNING"
        elif any(c["status"] == "HEALTHY" for c in components.values()):
            report.overall_status = "HEALTHY"
        else:
            report.overall_status = "UNKNOWN"

        return report

    def finish_monitoring(self) -> Optional[HealthReport]:
        """
        Completes the current request monitoring lifecycle, registers report in history logs.
        """
        if not getattr(settings, "ENABLE_HEALTH_MONITOR", True):
            return None

        state = self._get_or_create_state()
        
        # Calculate pipeline component final latency
        total_time = (time.perf_counter() - state["start_time"]) * 1000
        self.record_component("pipeline", "FAILED" if len(state["errors"]) > 0 else "HEALTHY", total_time)

        report = self.build_health_report()

        # Add to history
        with self.reports_lock:
            self.reports.append(report)
            if len(self.reports) > 1000:
                self.reports.pop(0)

        # Print structured logging presentation
        log_str = self._format_log_summary(report)
        logger.info(log_str)

        # Clear local thread scope
        self._local.state = None

        return report

    def _format_log_summary(self, r: HealthReport) -> str:
        """
        Formats report fields into a clean structured string presentation.
        """
        diags_str = ", ".join(r.diagnostics_summary) if r.diagnostics_summary else "None"
        summary = f"""
====================================================
Pipeline Health Report
====================================================
Overall Status:      {r.overall_status}
Pipeline:            {r.pipeline_status}
Hybrid:              {r.hybrid_status}
Parent:              {r.parent_status}
Compression:         {r.compression_status}
Self Query:          {r.self_query_status}
Ensemble:            {r.ensemble_status}
Adaptive:            {r.adaptive_status}
Memory:              {r.conversation_memory_status}
Metadata:            {r.metadata_ranker_status}
Scorer:              {r.result_scorer_status}
Verifier:            {r.answer_verifier_status}
Analytics:           {r.analytics_status}
Warnings:            {r.warning_count}
Errors:              {r.error_count}
Diagnostics:         {diags_str}
====================================================
"""
        return summary.strip()

    def export_report(self) -> List[Dict[str, Any]]:
        """
        Exports reports history list to serializable dictionary formats.
        """
        with self.reports_lock:
            return [asdict(r) for r in self.reports]

    def reset(self):
        """
        Resets and purges reports database and local thread scope.
        """
        with self.reports_lock:
            self.reports.clear()
        self._local.state = None
        logger.info("Health monitor diagnostics reset completed.")
