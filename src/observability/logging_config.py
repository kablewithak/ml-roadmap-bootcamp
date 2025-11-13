"""
Structured Logging Configuration

WHY STRUCTURED LOGGING MATTERS:
-------------------------------
Traditional logs:  "User 12345 made payment of $500 at 2024-01-15 10:23:45"
Structured logs:   {"user_id": 12345, "amount": 500, "timestamp": "2024-01-15T10:23:45Z"}

Benefits:
1. QUERYABLE: Filter "show me all payments > $1000" (impossible with text logs)
2. CORRELATION: Every log has trace_id → click to see full request trace
3. AGGREGATION: "Average payment amount per hour" (parse-free)
4. COST: Loki stores JSON more efficiently than text

BUSINESS IMPACT:
When a customer calls "My payment failed at 3pm", you can:
1. Search logs: timestamp=3pm, user_id=X
2. Get trace_id from log
3. Open Jaeger trace → see exactly which microservice failed

Without structured logging: grep through 1GB of text files for 30 minutes.

FAILURE MODE:
If you log sensitive data (credit_card_number), it's stored in plain text.
SOLUTION: Use log scrubbing (see scrub_sensitive_data below)
"""

import logging
import sys
from typing import Any, Dict
from pythonjsonlogger import jsonlogger

from opentelemetry import trace


class CorrelationJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter that injects trace_id and span_id into every log.

    HOW IT WORKS:
    1. Python logger calls: logger.info("Payment processed", extra={"user_id": 123})
    2. This formatter adds: trace_id, span_id, service_name
    3. Output: {"msg": "Payment processed", "user_id": 123, "trace_id": "abc...", ...}

    CRITICAL: Correlation IDs
    -------------------------
    The trace_id links logs to traces:
    - Grafana Log Panel: Click trace_id → Opens Jaeger trace
    - Jaeger Trace: Click "Logs" → Shows all logs for that request

    This is the "holy grail" of observability: unified view of logs + traces + metrics.
    """

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """
        Inject custom fields into every log record.

        Args:
            log_record: The dict that will be serialized to JSON
            record: Python LogRecord object
            message_dict: Extra fields from logger.info("msg", extra={...})
        """
        super().add_fields(log_record, record, message_dict)

        # Add trace context (if exists)
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            log_record['trace_id'] = format(ctx.trace_id, '032x')
            log_record['span_id'] = format(ctx.span_id, '016x')

        # Add standard fields
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['timestamp'] = self.formatTime(record, self.datefmt)

        # CRITICAL: Scrub sensitive data
        # This prevents accidentally logging credit cards, SSNs, etc.
        log_record = self.scrub_sensitive_data(log_record)

    def scrub_sensitive_data(self, log_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive fields from logs.

        COMPLIANCE REQUIREMENT:
        PCI-DSS: Cannot log full credit card numbers
        GDPR: Cannot log PII without consent
        HIPAA: Cannot log health data

        TRADE-OFF:
        - ✅ Compliance: No PII leaks
        - ❌ Debuggability: Can't see actual values

        SOLUTION: Log last 4 digits or hashed values
        """
        sensitive_keys = ['credit_card', 'ssn', 'password', 'api_key', 'secret']

        for key in sensitive_keys:
            if key in log_record:
                # Show last 4 digits only
                if isinstance(log_record[key], str) and len(log_record[key]) > 4:
                    log_record[key] = f"***{log_record[key][-4:]}"
                else:
                    log_record[key] = "***REDACTED***"

        return log_record


def setup_logging(level: str = "INFO", service_name: str = "unknown") -> None:
    """
    Configure structured JSON logging for the application.

    This replaces the default Python logging format with JSON output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service_name: Service name to include in logs

    Example output:
        {
            "timestamp": "2024-01-15T10:23:45.123Z",
            "level": "INFO",
            "logger": "fraud_service",
            "message": "Payment processed successfully",
            "trace_id": "abc123...",
            "span_id": "xyz789...",
            "user_id": 12345,
            "amount": 500.0
        }
    """
    # Create handler (stdout so Docker/K8s can collect logs)
    handler = logging.StreamHandler(sys.stdout)

    # Use our custom formatter
    formatter = CorrelationJsonFormatter(
        fmt='%(timestamp)s %(level)s %(logger)s %(message)s',
        rename_fields={'message': 'msg'},  # Shorter field name
    )
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    root_logger.info(f"Structured logging initialized for {service_name}")


# ============================================================================
# HELPER: Create child logger with context
# ============================================================================

def get_logger(name: str, **context) -> logging.LoggerAdapter:
    """
    Create a logger with pre-bound context.

    USE CASE:
    Instead of repeating logger.info("msg", extra={"user_id": 123}) everywhere,
    create a logger with user_id already bound:

        user_logger = get_logger("payment", user_id=123)
        user_logger.info("Payment started")  # Automatically includes user_id
        user_logger.info("Payment completed")

    Args:
        name: Logger name (usually __name__)
        **context: Key-value pairs to include in every log from this logger

    Returns:
        LoggerAdapter with pre-bound context
    """
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, extra=context)
