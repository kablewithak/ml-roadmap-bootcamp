"""Real-time metrics tracking."""

from typing import Dict, Any, List
from collections import deque
from datetime import datetime, timedelta
import time


class MetricsTracker:
    """
    Track real-time metrics for monitoring.

    Provides:
    - Real-time performance metrics
    - Sliding window statistics
    - Alert triggering
    """

    def __init__(self, window_size_seconds: int = 300):
        """
        Initialize metrics tracker.

        Args:
            window_size_seconds: Size of sliding window in seconds
        """
        self.window_size = timedelta(seconds=window_size_seconds)

        # Sliding windows for metrics
        self.latencies: deque = deque()
        self.predictions: deque = deque()  # (timestamp, is_fraud, score)
        self.errors: deque = deque()

        # Counters
        self.total_processed = 0
        self.total_blocked = 0
        self.total_errors = 0

    def record_prediction(
        self,
        is_fraud: bool,
        score: float,
        latency_ms: float
    ) -> None:
        """
        Record a prediction.

        Args:
            is_fraud: Whether classified as fraud
            score: Fraud score
            latency_ms: Latency in milliseconds
        """
        now = datetime.utcnow()

        self.predictions.append((now, is_fraud, score))
        self.latencies.append((now, latency_ms))

        self.total_processed += 1
        if is_fraud:
            self.total_blocked += 1

        # Clean old data
        self._clean_old_data()

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        now = datetime.utcnow()
        self.errors.append((now, error_type))
        self.total_errors += 1

        self._clean_old_data()

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics over the sliding window."""
        self._clean_old_data()

        # Latency metrics
        recent_latencies = [lat for _, lat in self.latencies]
        if recent_latencies:
            avg_latency = sum(recent_latencies) / len(recent_latencies)
            p95_latency = sorted(recent_latencies)[int(len(recent_latencies) * 0.95)] if len(recent_latencies) > 0 else 0
            p99_latency = sorted(recent_latencies)[int(len(recent_latencies) * 0.99)] if len(recent_latencies) > 0 else 0
        else:
            avg_latency = 0
            p95_latency = 0
            p99_latency = 0

        # Prediction metrics
        recent_predictions = list(self.predictions)
        num_predictions = len(recent_predictions)
        num_blocked = sum(1 for _, is_fraud, _ in recent_predictions if is_fraud)

        # Throughput
        if recent_predictions:
            time_span = (recent_predictions[-1][0] - recent_predictions[0][0]).total_seconds()
            throughput = num_predictions / time_span if time_span > 0 else 0
        else:
            throughput = 0

        # Error rate
        num_errors = len(self.errors)
        error_rate = num_errors / num_predictions if num_predictions > 0 else 0

        return {
            "window_size_seconds": self.window_size.total_seconds(),
            "timestamp": datetime.utcnow().isoformat(),

            # Throughput
            "transactions_in_window": num_predictions,
            "throughput_tps": throughput,

            # Latency
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,

            # Predictions
            "blocked_count": num_blocked,
            "block_rate": num_blocked / num_predictions if num_predictions > 0 else 0,

            # Errors
            "error_count": num_errors,
            "error_rate": error_rate,

            # Totals (all-time)
            "total_processed": self.total_processed,
            "total_blocked": self.total_blocked,
            "total_errors": self.total_errors,
        }

    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for alert conditions.

        Returns:
            List of alerts
        """
        alerts = []
        metrics = self.get_current_metrics()

        # High latency alert
        if metrics["p99_latency_ms"] > 500:
            alerts.append({
                "level": "warning",
                "type": "high_latency",
                "message": f"P99 latency is {metrics['p99_latency_ms']:.1f}ms (threshold: 500ms)",
                "value": metrics["p99_latency_ms"]
            })

        # High error rate alert
        if metrics["error_rate"] > 0.01:  # 1% error rate
            alerts.append({
                "level": "critical",
                "type": "high_error_rate",
                "message": f"Error rate is {metrics['error_rate'] * 100:.2f}% (threshold: 1%)",
                "value": metrics["error_rate"]
            })

        # Low throughput alert
        if metrics["throughput_tps"] < 10 and self.total_processed > 100:
            alerts.append({
                "level": "warning",
                "type": "low_throughput",
                "message": f"Throughput is {metrics['throughput_tps']:.1f} TPS (threshold: 10 TPS)",
                "value": metrics["throughput_tps"]
            })

        # Unusual block rate
        if metrics["block_rate"] > 0.5:  # More than 50% blocked
            alerts.append({
                "level": "warning",
                "type": "high_block_rate",
                "message": f"Block rate is {metrics['block_rate'] * 100:.1f}% (unusually high)",
                "value": metrics["block_rate"]
            })

        return alerts

    def _clean_old_data(self) -> None:
        """Remove data outside the sliding window."""
        cutoff = datetime.utcnow() - self.window_size

        # Clean latencies
        while self.latencies and self.latencies[0][0] < cutoff:
            self.latencies.popleft()

        # Clean predictions
        while self.predictions and self.predictions[0][0] < cutoff:
            self.predictions.popleft()

        # Clean errors
        while self.errors and self.errors[0][0] < cutoff:
            self.errors.popleft()

    def reset(self) -> None:
        """Reset all metrics."""
        self.latencies.clear()
        self.predictions.clear()
        self.errors.clear()
        self.total_processed = 0
        self.total_blocked = 0
        self.total_errors = 0
