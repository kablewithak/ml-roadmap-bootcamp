"""Logging utilities for fraud detection system."""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class FraudDetectionLogger:
    """
    Structured logging for fraud detection system.

    Provides:
    - Structured JSON logging
    - Different log levels for different components
    - Audit trail for fraud decisions
    """

    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger("fraud_detection")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / "fraud_detection.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Audit log (separate file for fraud decisions)
        self.audit_logger = logging.getLogger("fraud_detection.audit")
        self.audit_logger.setLevel(logging.INFO)
        audit_handler = logging.FileHandler(
            self.log_dir / "fraud_audit.log"
        )
        audit_handler.setFormatter(file_formatter)
        self.audit_logger.addHandler(audit_handler)

    def log_fraud_decision(
        self,
        transaction_id: str,
        is_fraud: bool,
        score: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a fraud decision for audit trail.

        Args:
            transaction_id: Transaction ID
            is_fraud: Whether classified as fraud
            score: Fraud score
            details: Additional details
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "transaction_id": transaction_id,
            "decision": "BLOCK" if is_fraud else "ALLOW",
            "fraud_score": score,
            "details": details or {}
        }

        self.audit_logger.info(json.dumps(log_entry))

    def log_attack_detected(
        self,
        attack_type: str,
        num_transactions: int,
        success_rate: float
    ) -> None:
        """Log detected attack pattern."""
        self.logger.warning(
            f"Attack detected: {attack_type}, "
            f"{num_transactions} transactions, "
            f"success rate: {success_rate:.2%}"
        )

    def log_model_update(
        self,
        model_type: str,
        metrics: Dict[str, float]
    ) -> None:
        """Log model update event."""
        self.logger.info(
            f"Model updated: {model_type}, metrics: {json.dumps(metrics)}"
        )

    def log_performance_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Log system performance metrics."""
        self.logger.info(f"Performance metrics: {json.dumps(metrics)}")

    def log_error(self, error_message: str, exception: Optional[Exception] = None):
        """Log error."""
        if exception:
            self.logger.error(error_message, exc_info=exception)
        else:
            self.logger.error(error_message)
