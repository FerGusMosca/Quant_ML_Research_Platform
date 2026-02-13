"""
Observability Orchestration Logic
=================================
Orchestration layer for system-wide observability.
Receives messages from the FastAPI proxy and delegates to LangfuseClient.

Responsibilities:
    - Validate incoming messages
    - Route to appropriate Langfuse operations (span, generation, event)
    - Handle errors gracefully
    - Provide logging
"""

from typing import Optional, Dict, Any
from datetime import datetime

from common.util.observability.langfuse.schemas import SpanType, ObservabilityMessage
from service_layer.client.langfuse.langfuse_client import LangfuseClient


class ObservabilityOrchestationLogic:
    """
    Orchestration layer for system-wide observability.
    Acts as a bridge between the HTTP proxy and Langfuse client.
    """

    def __init__(self, config: Dict[str, str], logger=None):
        """
        Initialize the orchestration layer.

        Args:
            config: Dictionary containing Langfuse credentials
            logger: Optional logger instance
        """
        self.logger = logger
        self._client = LangfuseClient(config=config, logger=logger)
        self._log_info("ObservabilityOrchestationLogic initialized")

    def log_process_step(
        self,
        step_name: str,
        input_data: Any,
        output_data: Any,
        trace_id: Optional[str] = None,
        trace_name: str = "mcp-process",
        level: str = "DEFAULT",
        parent_id: Optional[str] = None,
        service_id: Optional[str] = None,
        span_type: str = "span",
        model: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a process step to Langfuse.
        Routes to span or generation based on span_type.

        Args:
            step_name: Name of the step/operation
            input_data: Input data for the step
            output_data: Output data from the step
            trace_id: Optional trace ID for grouping
            trace_name: Name of the parent trace
            level: Log level (DEFAULT, DEBUG, WARNING, ERROR)
            parent_id: Optional parent span ID
            service_id: Identifier of the calling service
            span_type: Type of span ("span" or "generation")
            model: Model name (required for generations)
            usage: Token usage (for generations)
            metadata: Additional metadata

        Returns:
            Dict with operation result
        """
        start_time = datetime.utcnow()

        result = {
            "success": False,
            "step_name": step_name,
            "service_id": service_id,
            "error": None,
            "elapsed_ms": None
        }

        try:
            # Validate required fields
            if not step_name:
                raise ValueError("step_name is required")

            # Route based on span type
            if span_type == "generation" or span_type == SpanType.GENERATION.value:
                if not model:
                    raise ValueError("model is required for generation spans")

                response = self._client.create_generation(
                    name=step_name,
                    model=model,
                    input_messages=input_data,
                    output_text=output_data,
                    trace_name=trace_name,
                    service_id=service_id,
                    usage=usage,
                    metadata=metadata
                )
            else:
                response = self._client.create_span(
                    name=step_name,
                    input_data=input_data,
                    output_data=output_data,
                    trace_name=trace_name,
                    level=level,
                    service_id=service_id,
                    trace_id=trace_id,
                    parent_span_id=parent_id,
                    metadata=metadata
                )

            result["success"] = response.get("success", False)
            result["span_id"] = response.get("span_id")
            result["trace_id"] = response.get("trace_id")

            if not result["success"]:
                result["error"] = response.get("error")

        except Exception as e:
            result["error"] = str(e)
            self._log_error(f"Error in log_process_step: {e}")

        result["elapsed_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        return result

    def log_from_message(self, message: ObservabilityMessage) -> Dict[str, Any]:
        """
        Log using a structured ObservabilityMessage object.

        Args:
            message: ObservabilityMessage instance

        Returns:
            Dict with operation result
        """
        return self.log_process_step(
            step_name=message.node_name,
            input_data=message.input_data,
            output_data=message.output_data,
            trace_id=message.trace_id,
            trace_name=message.trace_name,
            level=message.level.value if hasattr(message.level, 'value') else message.level,
            parent_id=message.parent_id,
            service_id=message.service_id.value if hasattr(message.service_id, 'value') else message.service_id,
            span_type=message.span_type.value if hasattr(message.span_type, 'value') else message.span_type,
            model=message.model,
            usage=message.usage,
            metadata=message.metadata
        )

    def log_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log from a raw dictionary (e.g., from HTTP request body).

        Args:
            data: Dictionary with message fields

        Returns:
            Dict with operation result
        """
        message = ObservabilityMessage.from_dict(data)
        return self.log_from_message(message)

    def shutdown(self) -> None:
        """Gracefully shutdown the orchestration layer."""
        self._log_info("Shutting down ObservabilityOrchestationLogic")
        self._client.shutdown()

    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            self.logger.info(f"[OBSERVABILITY] {message}")

    def _log_error(self, message: str) -> None:
        """Log error message."""
        if self.logger:
            self.logger.error(f"[OBSERVABILITY] {message}")