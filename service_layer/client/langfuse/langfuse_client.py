"""
Langfuse Client Service
=======================
Encapsulates all communication with Langfuse SDK v3.
This is the ONLY class that should interact directly with Langfuse.

Architecture:
    MCP Services -> Observability Proxy -> OrchestrationLogic -> LangfuseClient -> Langfuse Cloud
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime
from langfuse import get_client


class LangfuseClient:
    """
    Service client for Langfuse observability platform.
    Handles all trace and span operations with proper error handling.
    """

    def __init__(self, config: Dict[str, str], logger=None):
        """
        Initialize the Langfuse client.

        Args:
            config: Dictionary with keys:
                - LANGFUSE_PUBLIC_KEY
                - LANGFUSE_SECRET_KEY
                - LANGFUSE_BASE_URL (optional, defaults to EU cloud)
            logger: Optional logger instance
        """
        self.logger = logger
        self._configure_environment(config)
        self._client = get_client()

    def _configure_environment(self, config: Dict[str, str]) -> None:
        """Set up environment variables for Langfuse SDK."""
        os.environ["LANGFUSE_PUBLIC_KEY"] = config.get("LANGFUSE_PUBLIC_KEY", "")
        os.environ["LANGFUSE_SECRET_KEY"] = config.get("LANGFUSE_SECRET_KEY", "")
        os.environ["LANGFUSE_HOST"] = config.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"

    def create_span(
            self,
            name: str,
            input_data: Any,
            output_data: Any,
            trace_name: str = "mcp-process",
            level: str = "DEFAULT",
            service_id: Optional[str] = None,
            trace_id: Optional[str] = None,
            parent_span_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a span in Langfuse.

        Note: SDK v3 does not support manual trace_id assignment.
        trace_id is stored in metadata for reference only.
        Traces are grouped automatically by the SDK context.
        """
        result = {
            "success": False,
            "span_id": None,
            "trace_id": None,
            "error": None
        }

        try:
            # Build metadata (trace_id goes here for reference, not for grouping)
            span_metadata = {
                "service_id": service_id,
                "timestamp": datetime.utcnow().isoformat(),
                "custom_trace_id": trace_id,  # Stored for reference only
                "parent_span_id": parent_span_id
            }

            if metadata:
                span_metadata.update(metadata)

            # Remove None values
            span_metadata = {k: v for k, v in span_metadata.items() if v is not None}

            # SDK v3: trace_id is NOT supported in start_as_current_observation
            # Traces are managed automatically by the SDK context
            with self._client.start_as_current_observation(
                    as_type="span",
                    name=name
            ) as span:
                span.update(
                    input=input_data,
                    output=output_data,
                    level=level,
                    metadata=span_metadata if span_metadata else None
                )

                if trace_name:
                    span.update_trace(name=trace_name)

                result["span_id"] = self._client.get_current_observation_id()
                result["trace_id"] = self._client.get_current_trace_id()

            self._client.flush()
            result["success"] = True

            self._log_debug(f"Span '{name}' created [service={service_id}, trace={result['trace_id']}]")

        except Exception as e:
            result["error"] = str(e)
            self._log_error(f"Failed to create span '{name}': {e}")

        return result
    def create_generation(
            self,
            name: str,
            model: str,
            input_messages: Any,
            output_text: str,
            trace_name: str = "mcp-process",
            service_id: Optional[str] = None,
            usage: Optional[Dict[str, int]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a generation span for LLM calls.

        Args:
            name: Name of the generation
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet")
            input_messages: Input messages/prompt
            output_text: Model response
            trace_name: Name of the parent trace
            service_id: Identifier of the calling service
            usage: Token usage dict {prompt_tokens, completion_tokens, total_tokens}
            metadata: Additional metadata

        Returns:
            Dict with success status, span_id, trace_id, and error if any
        """
        result = {
            "success": False,
            "span_id": None,
            "trace_id": None,
            "error": None
        }

        try:
            gen_metadata = {
                "service_id": service_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            if metadata:
                gen_metadata.update(metadata)

            gen_metadata = {k: v for k, v in gen_metadata.items() if v is not None}

            with self._client.start_as_current_observation(
                    as_type="generation",
                    name=name,
                    model=model
            ) as generation:
                generation.update(
                    input=input_messages,
                    output=output_text,
                    usage=usage,
                    metadata=gen_metadata if gen_metadata else None
                )
                generation.update_trace(name=trace_name)

                result["span_id"] = self._client.get_current_observation_id()
                result["trace_id"] = self._client.get_current_trace_id()

            self._client.flush()
            result["success"] = True

            self._log_debug(f"Generation '{name}' created [model={model}, service={service_id}]")

        except Exception as e:
            result["error"] = str(e)
            self._log_error(f"Failed to create generation '{name}': {e}")

        return result

    def flush(self) -> None:
        """Force flush all pending events to Langfuse."""
        self._client.flush()

    def shutdown(self) -> None:
        """Gracefully shutdown the client."""
        self._client.shutdown()

    def _log_debug(self, message: str) -> None:
        """Log debug message if logger is available."""
        if self.logger:
            self.logger.info(message)

    def _log_error(self, message: str) -> None:
        """Log error message if logger is available."""
        if self.logger:
            self.logger.error(message)