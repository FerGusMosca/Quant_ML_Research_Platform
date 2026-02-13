import requests
from typing import Optional, Dict, Any, List
from datetime import datetime
import traceback



from common.util.observability.langfuse.schemas import (
    ServiceId,
    LogLevel,
    SpanType,
    OperationType,
    ObservabilityMessage,
    ErrorInfo
)


class ObsHttpClient:
    """
    HTTP client for sending observability events to the proxy.

    Each MCP service should create one instance with its ServiceId.
    """

    def __init__(
            self,
            service_id: ServiceId,
            proxy_url: str = "http://localhost:7003",
            timeout: int = 5,
            logger=None
    ):
        """
        Initialize the observability client.

        Args:
            service_id: Identifier for this service (from ServiceId enum)
            proxy_url: URL of the observability proxy
            timeout: Request timeout in seconds
            logger: Optional logger instance
        """
        self.service_id = service_id
        self.proxy_url = proxy_url.rstrip("/")
        self.timeout = timeout
        self.logger = logger
        self._session = requests.Session()

    def log_span(
            self,
            operation: OperationType,
            target: str,
            input_data: Any = None,
            output_data: Any = None,
            level: LogLevel = LogLevel.DEFAULT,
            trace_id: Optional[str] = None,
            trace_name: str = "mcp-process",
            parent_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a span (generic operation).

        Args:
            operation: Type of operation (from OperationType enum)
            target: What is being operated on (e.g., "10k_AAPL")
            input_data: Input to the operation
            output_data: Output from the operation
            level: Log level
            trace_id: Optional trace ID for correlation
            trace_name: Name for the trace group
            parent_id: Optional parent span ID
            metadata: Additional metadata

        Returns:
            Response from the proxy

        Example:
            obs_client.log_span(
                operation=OperationType.DOWNLOAD,
                target="f4_MSFT_2024",
                input_data={"symbol": "MSFT", "year": 2024},
                output_data={"files_count": 15, "status": "completed"}
            )
        """
        node_name = f"{operation.value}_{target}"

        payload = {
            "service_id": self.service_id.value,
            "node_name": node_name,
            "input": input_data,
            "output": output_data,
            "level": level.value,
            "trace_id": trace_id,
            "trace_name": trace_name,
            "parent_id": parent_id,
            "span_type": SpanType.SPAN.value,
            "metadata": metadata or {}
        }

        return self._send("/log", payload)

    def log_error(
            self,
            operation: str,
            error: Exception,
            input_data: Any = None,
            trace_id: Optional[str] = None,
            trace_name: str = "mcp-process",
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an error event.

        Args:
            operation: Name of the failed operation
            error: The exception that occurred
            input_data: Input that caused the error
            trace_id: Optional trace ID
            trace_name: Name for the trace group
            metadata: Additional metadata

        Returns:
            Response from the proxy

        Example:
            try:
                download_file(...)
            except Exception as e:
                obs_client.log_error(
                    operation="download_file",
                    error=e,
                    input_data={"url": file_url}
                )
        """
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error),
            "stack_trace": traceback.format_exc()
        }

        payload = {
            "service_id": self.service_id.value,
            "node_name": f"error_{operation}",
            "input": input_data,
            "output": error_info,
            "level": LogLevel.ERROR.value,
            "trace_id": trace_id,
            "trace_name": trace_name,
            "span_type": SpanType.SPAN.value,
            "metadata": {
                "error_type": error_info["error_type"],
                **(metadata or {})
            }
        }

        return self._send("/log", payload)

    def log_llm(
            self,
            name: str,
            model: str,
            input_messages: Any,
            output_text: str,
            usage: Optional[Dict[str, int]] = None,
            trace_id: Optional[str] = None,
            trace_name: str = "mcp-process",
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an LLM generation.

        Args:
            name: Name of the LLM operation
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet")
            input_messages: Input prompt/messages
            output_text: Model response
            usage: Token usage {prompt_tokens, completion_tokens, total_tokens}
            trace_id: Optional trace ID
            trace_name: Name for the trace group
            metadata: Additional metadata

        Returns:
            Response from the proxy

        Example:
            obs_client.log_llm(
                name="summarize_filing",
                model="gpt-4-turbo",
                input_messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": "Summarize this 10-K..."}
                ],
                output_text="The company reported revenue of...",
                usage={"prompt_tokens": 1500, "completion_tokens": 300, "total_tokens": 1800}
            )
        """
        payload = {
            "service_id": self.service_id.value,
            "node_name": name,
            "input": input_messages,
            "output": output_text,
            "level": LogLevel.DEFAULT.value,
            "trace_id": trace_id,
            "trace_name": trace_name,
            "span_type": SpanType.GENERATION.value,
            "model": model,
            "usage": usage,
            "metadata": metadata or {}
        }

        return self._send("/log", payload)

    def log_batch(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Log multiple events in a batch.

        Args:
            events: List of event dictionaries

        Returns:
            Batch processing summary
        """
        return self._send("/log/batch", events)

    def log_start(
            self,
            operation: str,
            input_data: Any = None,
            trace_id: Optional[str] = None,
            trace_name: str = "mcp-process"
    ) -> Dict[str, Any]:
        """
        Log the start of an operation.

        Args:
            operation: Name of the operation starting
            input_data: Input data
            trace_id: Trace ID for correlation
            trace_name: Name for the trace group

        Returns:
            Response from the proxy
        """
        payload = {
            "service_id": self.service_id.value,
            "node_name": f"start_{operation}",
            "input": input_data,
            "output": {"status": "started", "timestamp": datetime.utcnow().isoformat()},
            "level": LogLevel.DEFAULT.value,
            "trace_id": trace_id,
            "trace_name": trace_name,
            "span_type": SpanType.SPAN.value
        }

        return self._send("/log", payload)

    def log_complete(
            self,
            operation: str,
            output_data: Any = None,
            trace_id: Optional[str] = None,
            trace_name: str = "mcp-process",
            elapsed_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Log the completion of an operation.

        Args:
            operation: Name of the completed operation
            output_data: Output/result data
            trace_id: Trace ID for correlation
            trace_name: Name for the trace group
            elapsed_sec: Elapsed time in seconds

        Returns:
            Response from the proxy
        """
        output = output_data if isinstance(output_data, dict) else {"result": output_data}
        output["status"] = "completed"
        output["timestamp"] = datetime.utcnow().isoformat()
        if elapsed_sec is not None:
            output["elapsed_sec"] = elapsed_sec

        payload = {
            "service_id": self.service_id.value,
            "node_name": f"complete_{operation}",
            "input": None,
            "output": output,
            "level": LogLevel.DEFAULT.value,
            "trace_id": trace_id,
            "trace_name": trace_name,
            "span_type": SpanType.SPAN.value
        }

        return self._send("/log", payload)

    def _send(self, endpoint: str, payload: Any) -> Dict[str, Any]:
        """
        Send request to the observability proxy.

        Args:
            endpoint: API endpoint (e.g., "/log")
            payload: Request payload

        Returns:
            Response as dictionary
        """
        url = f"{self.proxy_url}{endpoint}"

        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            self._log_warning(f"Timeout sending to {endpoint}")
            return {"status": "error", "message": "timeout"}

        except requests.exceptions.ConnectionError:
            self._log_warning(f"Connection error to {self.proxy_url}")
            return {"status": "error", "message": "connection_error"}

        except Exception as e:
            self._log_warning(f"Error sending to {endpoint}: {e}")
            return {"status": "error", "message": str(e)}

    def _log_warning(self, message: str) -> None:
        """Log warning if logger is available."""
        if self.logger:
            self.logger.warning(f"[OBS-CLIENT] {message}")

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()