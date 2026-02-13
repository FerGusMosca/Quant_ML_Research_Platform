"""
Observability Context
=====================
Thread-safe context for tracking the current operation.
Set once at operation start, read by MCPLogObserver.

Usage:
    # At operation start (in handler or logic layer)
    ObsContext.set(job_id, service_id, operation_name)

    # At operation end
    ObsContext.clear(job_id)
"""

import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class OperationContext:
    """Context for a running operation."""
    job_id: str
    service_id: str
    operation_name: str
    metadata: Dict[str, Any] = None


class ObsContext:
    """
    Thread-safe global context for observability.
    Maps job_id -> OperationContext
    """
    _lock = threading.Lock()
    _contexts: Dict[str, OperationContext] = {}

    @classmethod
    def set(
            cls,
            job_id: str,
            service_id: str,
            operation_name: str,
            metadata: Dict[str, Any] = None
    ) -> None:
        """
        Set context for a job. Call at operation start.

        Args:
            job_id: Unique job identifier
            service_id: Service identifier (e.g., "mcp-sec-filings")
            operation_name: Name of the operation (e.g., "download_k8")
            metadata: Optional additional context
        """
        with cls._lock:
            cls._contexts[job_id] = OperationContext(
                job_id=job_id,
                service_id=service_id,
                operation_name=operation_name,
                metadata=metadata or {}
            )

    @classmethod
    def get(cls, job_id: str) -> Optional[OperationContext]:
        """Get context for a job."""
        with cls._lock:
            return cls._contexts.get(job_id)

    @classmethod
    def clear(cls, job_id: str) -> None:
        """Clear context for a job. Call at operation end."""
        with cls._lock:
            cls._contexts.pop(job_id, None)