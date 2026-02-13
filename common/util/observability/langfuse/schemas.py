"""
Observability Message Schema
============================
Defines the structured message format for all observability events.
Ensures consistent logging across all MCP services.

Usage from MCP clients:
    from langfuse.schemas import ObservabilityMessage, ServiceId, LogLevel

    message = ObservabilityMessage(
        service_id=ServiceId.MCP_SEC_REPORTS,
        node_name="download_10k",
        input_data={"symbol": "AAPL", "year": 2024},
        output_data={"status": "completed", "files": 3}
    )

    # Send to observability proxy
    requests.post("http://localhost:7003/log", json=message.to_dict())
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime


class ServiceId(str, Enum):
    """
    Identifiers for each MCP service.
    Add new services here as they are created.

    IMPORTANT: Use these consistently across all services
    to enable filtering and grouping in Langfuse dashboard.
    """
    MCP_SEC_REPORTS = "mcp-sec-reports"  # SEC filings service (10K, 10Q, F4, etc.)
    MCP_MARKET_DATA = "mcp-market-data"  # Market data service (prices, quotes)
    MCP_PORTFOLIO = "mcp-portfolio"  # Portfolio management service
    MCP_NEWS = "mcp-news"  # News aggregation service
    MCP_ANALYSIS = "mcp-analysis"  # Analysis/AI service
    ORCHESTRATOR = "orchestrator"  # Main orchestrator
    UI_CLIENT = "ui-client"  # Frontend UI
    UNKNOWN = "unknown"


class LogLevel(str, Enum):
    """Log severity levels supported by Langfuse."""
    DEFAULT = "DEFAULT"
    DEBUG = "DEBUG"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SpanType(str, Enum):
    """Types of spans/observations."""
    SPAN = "span"
    GENERATION = "generation"
    EVENT = "event"


class OperationType(str, Enum):
    """
    Standard operation types for consistent naming.
    Use these to build node_name: f"{operation}_{target}"

    Example: OperationType.DOWNLOAD + "10k_AAPL" = "download_10k_AAPL"
    """
    # Data operations
    DOWNLOAD = "download"
    FETCH = "fetch"
    PARSE = "parse"
    PROCESS = "process"
    EXTRACT = "extract"
    TRANSFORM = "transform"

    # LLM operations
    LLM_CALL = "llm_call"
    LLM_SUMMARIZE = "llm_summarize"
    LLM_ANALYZE = "llm_analyze"
    EMBEDDING = "embedding"

    # Validation
    VALIDATE = "validate"
    RESOLVE = "resolve"

    # I/O
    READ = "read"
    WRITE = "write"
    SAVE = "save"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"

    # Lifecycle
    START = "start"
    COMPLETE = "complete"
    ERROR = "error"
    RETRY = "retry"


@dataclass
class ObservabilityMessage:
    """
    Standard message format for observability events.

    This is the contract between MCP services and the Observability Proxy.
    All fields are serializable to JSON for HTTP transport.

    Example:
        msg = ObservabilityMessage(
            service_id=ServiceId.MCP_SEC_REPORTS,
            node_name="download_10k",
            input_data={"symbol": "AAPL"},
            output_data={"status": "ok", "files": 3}
        )
        requests.post("http://localhost:7003/log", json=msg.to_dict())
    """

    # Required: identifies which service is logging
    service_id: ServiceId

    # Required: name of the operation (use OperationType + target)
    node_name: str

    # Data payload
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None

    # Trace correlation (for grouping related operations)
    trace_id: Optional[str] = None
    trace_name: Optional[str] = "mcp-process"
    parent_id: Optional[str] = None

    # Severity
    level: LogLevel = LogLevel.DEFAULT

    # Span type (span, generation, event)
    span_type: SpanType = SpanType.SPAN

    # LLM-specific fields (only for generations)
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # {prompt_tokens, completion_tokens, total_tokens}

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "service_id": self.service_id.value if isinstance(self.service_id, Enum) else self.service_id,
            "node_name": self.node_name,
            "input": self.input_data,
            "output": self.output_data,
            "trace_id": self.trace_id,
            "trace_name": self.trace_name,
            "parent_id": self.parent_id,
            "level": self.level.value if isinstance(self.level, Enum) else self.level,
            "span_type": self.span_type.value if isinstance(self.span_type, Enum) else self.span_type,
            "model": self.model,
            "usage": self.usage,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservabilityMessage":
        """Create instance from dictionary (e.g., from JSON payload)."""
        # Handle service_id
        service_id = data.get("service_id", "unknown")
        if isinstance(service_id, str):
            try:
                service_id = ServiceId(service_id)
            except ValueError:
                service_id = ServiceId.UNKNOWN

        # Handle level
        level = data.get("level", "DEFAULT")
        if isinstance(level, str):
            try:
                level = LogLevel(level)
            except ValueError:
                level = LogLevel.DEFAULT

        # Handle span_type
        span_type = data.get("span_type", "span")
        if isinstance(span_type, str):
            try:
                span_type = SpanType(span_type)
            except ValueError:
                span_type = SpanType.SPAN

        return cls(
            service_id=service_id,
            node_name=data.get("node_name", "unknown"),
            input_data=data.get("input"),
            output_data=data.get("output"),
            trace_id=data.get("trace_id"),
            trace_name=data.get("trace_name", "mcp-process"),
            parent_id=data.get("parent_id"),
            level=level,
            span_type=span_type,
            model=data.get("model"),
            usage=data.get("usage"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat())
        )


@dataclass
class ErrorInfo:
    """Structured error information."""
    error_type: str
    message: str
    stack_trace: Optional[str] = None
    recoverable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingResult:
    """
    Standard output format for processing operations.
    Use for consistent output structure across services.
    """
    status: str  # "completed", "error", "partial", "started"
    count: int = 0
    items: List[Any] = field(default_factory=list)
    error_type: Optional[str] = None
    message: Optional[str] = None
    elapsed_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# HELPER FUNCTIONS FOR BUILDING MESSAGES
# ============================================================

def build_message(
        service_id: ServiceId,
        operation: OperationType,
        target: str,
        input_data: Any = None,
        output_data: Any = None,
        level: LogLevel = LogLevel.DEFAULT,
        trace_id: Optional[str] = None,
        trace_name: str = "mcp-process",
        metadata: Optional[Dict[str, Any]] = None
) -> ObservabilityMessage:
    """
    Helper to build a standard span message.

    Args:
        service_id: The calling service
        operation: Type of operation (from OperationType enum)
        target: What is being operated on (e.g., "10k_AAPL", "market_prices")
        input_data: Input to the operation
        output_data: Output from the operation
        level: Log level
        trace_id: Optional trace ID for correlation
        trace_name: Name for the trace group
        metadata: Additional metadata

    Returns:
        ObservabilityMessage ready to send

    Example:
        msg = build_message(
            service_id=ServiceId.MCP_SEC_REPORTS,
            operation=OperationType.DOWNLOAD,
            target="10k_AAPL_2024",
            input_data={"symbol": "AAPL", "year": 2024},
            output_data={"files": 3, "status": "ok"}
        )
        requests.post("http://localhost:7003/log", json=msg.to_dict())
    """
    node_name = f"{operation.value}_{target}"

    return ObservabilityMessage(
        service_id=service_id,
        node_name=node_name,
        input_data=input_data,
        output_data=output_data,
        level=level,
        trace_id=trace_id,
        trace_name=trace_name,
        metadata=metadata or {}
    )


def build_error_message(
        service_id: ServiceId,
        operation: str,
        error: Exception,
        input_data: Any = None,
        trace_id: Optional[str] = None,
        trace_name: str = "mcp-process"
) -> ObservabilityMessage:
    """
    Helper to build an error message.

    Args:
        service_id: The calling service
        operation: Name of the failed operation
        error: The exception that occurred
        input_data: Input that caused the error
        trace_id: Optional trace ID
        trace_name: Name for the trace group

    Returns:
        ObservabilityMessage with error details
    """
    import traceback

    error_info = ErrorInfo(
        error_type=type(error).__name__,
        message=str(error),
        stack_trace=traceback.format_exc()
    )

    return ObservabilityMessage(
        service_id=service_id,
        node_name=f"error_{operation}",
        input_data=input_data,
        output_data=error_info.to_dict(),
        level=LogLevel.ERROR,
        trace_id=trace_id,
        trace_name=trace_name,
        metadata={"error_type": error_info.error_type}
    )


def build_llm_message(
        service_id: ServiceId,
        name: str,
        model: str,
        input_messages: Any,
        output_text: str,
        usage: Optional[Dict[str, int]] = None,
        trace_id: Optional[str] = None,
        trace_name: str = "mcp-process"
) -> ObservabilityMessage:
    """
    Helper to build an LLM generation message.

    Args:
        service_id: The calling service
        name: Name of the LLM operation
        model: Model identifier (e.g., "gpt-4", "claude-3-sonnet")
        input_messages: Input prompt/messages
        output_text: Model response
        usage: Token usage {prompt_tokens, completion_tokens, total_tokens}
        trace_id: Optional trace ID
        trace_name: Name for the trace group

    Returns:
        ObservabilityMessage for LLM generation
    """
    return ObservabilityMessage(
        service_id=service_id,
        node_name=name,
        input_data=input_messages,
        output_data=output_text,
        span_type=SpanType.GENERATION,
        model=model,
        usage=usage,
        trace_id=trace_id,
        trace_name=trace_name
    )