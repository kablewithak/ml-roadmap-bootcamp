"""
Resilience patterns for streaming infrastructure.

Implements circuit breakers, backpressure, and adaptive rate limiting
to prevent cascade failures and maintain system stability.
"""

import logging
import time
import asyncio
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
from confluent_kafka import TopicPartition

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, rejecting requests
    HALF_OPEN = "HALF_OPEN"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration.

    Attributes:
        failure_threshold: Failure rate to open circuit (0.0-1.0)
        success_threshold: Success rate to close circuit (0.0-1.0)
        timeout: Time to wait before attempting recovery
        window_size: Number of recent calls to track
        min_calls: Minimum calls before evaluating state
    """
    failure_threshold: float = 0.5  # 50% failure rate opens circuit
    success_threshold: float = 0.8  # 80% success rate closes circuit
    timeout: timedelta = timedelta(seconds=60)
    window_size: int = 100
    min_calls: int = 10


class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open."""
    pass


class StreamingCircuitBreaker:
    """
    Circuit breaker for preventing cascade failures.

    When downstream systems fail (database, external API), the circuit breaker
    prevents the system from being overwhelmed by failing requests.

    Real scenario: Database goes down, 1M messages queue up, OOM crash.
    Solution: Open circuit, fail fast, prevent resource exhaustion.

    States:
    - CLOSED: Normal operation, all requests allowed
    - OPEN: Too many failures, rejecting requests
    - HALF_OPEN: Testing recovery, allowing limited requests

    Example:
        >>> breaker = StreamingCircuitBreaker()
        >>> async def risky_operation():
        ...     # Call external service
        ...     pass
        >>> result = await breaker.call(risky_operation)
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.call_history = deque(maxlen=self.config.window_size)
        self.state_change_time = datetime.now()

        logger.info(f"Circuit breaker initialized: {self.config}")

    def _record_success(self):
        """Record successful call."""
        self.success_count += 1
        self.call_history.append(True)

        if self.state == CircuitState.HALF_OPEN:
            # Check if we can close the circuit
            if self._should_close_circuit():
                self._transition_to_closed()

    def _record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.call_history.append(False)

        if self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._transition_to_open()

        elif self.state == CircuitState.HALF_OPEN:
            # Failed during recovery, back to open
            self._transition_to_open()

    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open."""
        if len(self.call_history) < self.config.min_calls:
            return False

        failure_rate = 1 - (sum(self.call_history) / len(self.call_history))
        return failure_rate >= self.config.failure_threshold

    def _should_close_circuit(self) -> bool:
        """Determine if circuit should close."""
        if len(self.call_history) < self.config.min_calls:
            return False

        success_rate = sum(self.call_history) / len(self.call_history)
        return success_rate >= self.config.success_threshold

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        elapsed = datetime.now() - self.last_failure_time
        return elapsed >= self.config.timeout

    def _transition_to_open(self):
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.state_change_time = datetime.now()
        logger.warning(
            f"Circuit breaker OPENED. "
            f"Failure rate: {1 - (sum(self.call_history) / len(self.call_history)):.2%}"
        )

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = datetime.now()
        logger.info("Circuit breaker HALF_OPEN (testing recovery)")

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.state_change_time = datetime.now()
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker CLOSED (recovered)")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenException: If circuit is open
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker is OPEN. "
                    f"Wait {self.config.timeout.total_seconds()}s before retry."
                )

        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._record_success()
            return result

        except Exception as e:
            self._record_failure()
            raise

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state information."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_calls': len(self.call_history),
            'failure_rate': (1 - (sum(self.call_history) / len(self.call_history))
                           if self.call_history else 0.0),
            'time_in_state': (datetime.now() - self.state_change_time).total_seconds()
        }


@dataclass
class BackpressureConfig:
    """
    Backpressure configuration.

    Attributes:
        target_latency_ms: Target processing latency (p99)
        min_rate: Minimum consumption rate (msg/sec)
        max_rate: Maximum consumption rate (msg/sec)
        adjustment_factor: Rate adjustment factor (0.0-1.0)
        measurement_window: Window for latency measurement
    """
    target_latency_ms: float = 100.0
    min_rate: float = 1000.0  # 1k msg/sec minimum
    max_rate: float = 100000.0  # 100k msg/sec maximum
    adjustment_factor: float = 0.1  # 10% adjustment per iteration
    measurement_window: int = 1000  # Measure over 1000 messages


class AdaptiveBackpressure:
    """
    Adaptive backpressure mechanism for stream processing.

    Dynamically adjusts consumption rate based on processing capacity
    and latency to prevent OOM and maintain SLAs.

    Real scenario: Processing slows down due to downstream bottleneck.
    Without backpressure: Consumer keeps pulling, queues grow, OOM crash.
    With backpressure: Consumption rate adjusts, system stays stable.

    Example:
        >>> backpressure = AdaptiveBackpressure(consumer)
        >>> while True:
        ...     message = consumer.poll()
        ...     latency = process(message)
        ...     await backpressure.record_latency(latency)
        ...     # Backpressure automatically adjusts consumption
    """

    def __init__(
        self,
        consumer: Any,
        config: Optional[BackpressureConfig] = None
    ):
        """
        Initialize adaptive backpressure.

        Args:
            consumer: Kafka consumer instance
            config: Backpressure configuration
        """
        self.consumer = consumer
        self.config = config or BackpressureConfig()
        self.current_rate = self.config.max_rate
        self.latencies = deque(maxlen=self.config.measurement_window)
        self.is_paused = False
        self.adjustment_count = 0

        logger.info(f"Adaptive backpressure initialized: {self.config}")

    def record_latency(self, latency_ms: float):
        """
        Record processing latency and adjust rate if needed.

        Args:
            latency_ms: Processing latency in milliseconds
        """
        self.latencies.append(latency_ms)

        # Only adjust after collecting enough samples
        if len(self.latencies) < self.config.measurement_window:
            return

        # Calculate p99 latency
        sorted_latencies = sorted(self.latencies)
        p99_index = int(len(sorted_latencies) * 0.99)
        p99_latency = sorted_latencies[p99_index]

        # Adjust rate based on latency
        self._adjust_rate(p99_latency)

    def _adjust_rate(self, current_latency_p99: float):
        """
        Adjust consumption rate based on latency.

        Args:
            current_latency_p99: Current p99 latency
        """
        target = self.config.target_latency_ms
        adjustment = self.config.adjustment_factor

        if current_latency_p99 > target * 1.2:
            # Latency too high, reduce rate
            new_rate = self.current_rate * (1 - adjustment)
            new_rate = max(new_rate, self.config.min_rate)

            if new_rate != self.current_rate:
                self.current_rate = new_rate
                self.adjustment_count += 1
                logger.warning(
                    f"Backpressure: Reducing rate to {new_rate:.0f} msg/s "
                    f"(p99 latency: {current_latency_p99:.2f}ms)"
                )

                # Pause consumption if rate drops too low
                if new_rate <= self.config.min_rate * 1.1:
                    self._pause_consumption()

        elif current_latency_p99 < target * 0.8:
            # Latency low, can increase rate
            new_rate = self.current_rate * (1 + adjustment)
            new_rate = min(new_rate, self.config.max_rate)

            if new_rate != self.current_rate:
                self.current_rate = new_rate
                self.adjustment_count += 1
                logger.info(
                    f"Backpressure: Increasing rate to {new_rate:.0f} msg/s "
                    f"(p99 latency: {current_latency_p99:.2f}ms)"
                )

                # Resume if paused
                if self.is_paused:
                    self._resume_consumption()

    def _pause_consumption(self):
        """Pause consumption from all partitions."""
        if not self.is_paused:
            try:
                assignment = self.consumer.consumer.assignment()
                if assignment:
                    self.consumer.pause(assignment)
                    self.is_paused = True
                    logger.warning("Backpressure: PAUSED consumption")
            except Exception as e:
                logger.error(f"Error pausing consumption: {e}")

    def _resume_consumption(self):
        """Resume consumption from all partitions."""
        if self.is_paused:
            try:
                assignment = self.consumer.consumer.assignment()
                if assignment:
                    self.consumer.resume(assignment)
                    self.is_paused = False
                    logger.info("Backpressure: RESUMED consumption")
            except Exception as e:
                logger.error(f"Error resuming consumption: {e}")

    async def wait_for_capacity(self):
        """
        Wait until system has capacity to process more messages.

        Uses current rate to determine wait time.
        """
        if self.is_paused:
            # Wait for resume
            await asyncio.sleep(1.0)
            return

        # Calculate wait time based on current rate
        wait_time = 1.0 / self.current_rate if self.current_rate > 0 else 0.001
        await asyncio.sleep(wait_time)

    def get_metrics(self) -> Dict[str, Any]:
        """Get backpressure metrics."""
        if not self.latencies:
            return {
                'current_rate': self.current_rate,
                'is_paused': self.is_paused,
                'adjustment_count': self.adjustment_count,
                'p50_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'p99_latency_ms': 0.0
            }

        sorted_latencies = sorted(self.latencies)
        return {
            'current_rate': self.current_rate,
            'is_paused': self.is_paused,
            'adjustment_count': self.adjustment_count,
            'p50_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.50)],
            'p95_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.95)],
            'p99_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.99)],
            'target_latency_ms': self.config.target_latency_ms
        }
