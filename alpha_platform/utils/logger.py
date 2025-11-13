"""Structured logging for the Alpha Platform."""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from structlog.stdlib import BoundLogger

from alpha_platform.utils.config import get_log_path


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
) -> None:
    """
    Configure structured logging for the platform.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        json_format: Whether to use JSON format for logs
    """
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        stream=sys.stdout,
    )

    # Add file handler if specified
    if log_file:
        log_path = get_log_path() / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


# Initialize logging on import
setup_logging()
