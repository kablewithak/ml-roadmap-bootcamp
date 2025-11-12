"""Realistic timing utilities for attack simulation."""

import random
import time
from datetime import datetime, timedelta
from typing import Callable, Optional


class RealisticTimer:
    """
    Provides realistic timing patterns for attack simulation.

    Attackers don't execute attacks all at once - they use realistic timing
    to avoid detection.
    """

    @staticmethod
    def human_typing_delay() -> float:
        """Simulate human typing speed (200-400ms between keystrokes)."""
        return random.uniform(0.2, 0.4)

    @staticmethod
    def network_latency() -> float:
        """Simulate network latency (50-200ms)."""
        return random.uniform(0.05, 0.2)

    @staticmethod
    def think_time() -> float:
        """Simulate human think time (1-5 seconds)."""
        return random.uniform(1.0, 5.0)

    @staticmethod
    def automation_delay() -> float:
        """Simulate automated script delay (100-500ms)."""
        return random.uniform(0.1, 0.5)

    @staticmethod
    def stealthy_delay() -> float:
        """Simulate stealthy delay to avoid velocity detection (5-30 seconds)."""
        return random.uniform(5.0, 30.0)

    @staticmethod
    def exponential_backoff(attempt: int, base_delay: float = 1.0) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Attempt number (0-indexed)
            base_delay: Base delay in seconds

        Returns:
            Delay in seconds
        """
        return base_delay * (2 ** attempt) + random.uniform(0, 1)

    @staticmethod
    def poisson_arrival(rate: float) -> float:
        """
        Generate Poisson-distributed arrival time.

        Args:
            rate: Average number of events per second

        Returns:
            Time until next event in seconds
        """
        return random.expovariate(rate)

    @staticmethod
    def time_window_schedule(
        start_time: datetime,
        end_time: datetime,
        num_events: int,
        distribution: str = "uniform"
    ) -> list[datetime]:
        """
        Schedule events within a time window.

        Args:
            start_time: Start of time window
            end_time: End of time window
            num_events: Number of events to schedule
            distribution: Distribution type ("uniform", "poisson", "burst")

        Returns:
            List of scheduled timestamps
        """
        total_seconds = (end_time - start_time).total_seconds()
        timestamps = []

        if distribution == "uniform":
            for i in range(num_events):
                offset = (i + random.random()) * total_seconds / num_events
                timestamps.append(start_time + timedelta(seconds=offset))

        elif distribution == "poisson":
            rate = num_events / total_seconds
            current_time = start_time
            for _ in range(num_events):
                delay = RealisticTimer.poisson_arrival(rate)
                current_time += timedelta(seconds=delay)
                if current_time > end_time:
                    break
                timestamps.append(current_time)

        elif distribution == "burst":
            # Random bursts throughout the window
            num_bursts = max(1, num_events // 10)
            events_per_burst = num_events // num_bursts

            for i in range(num_bursts):
                burst_start = start_time + timedelta(
                    seconds=random.uniform(0, total_seconds)
                )
                for j in range(events_per_burst):
                    offset = random.uniform(0, 60)  # Burst within 1 minute
                    timestamp = burst_start + timedelta(seconds=offset)
                    if timestamp <= end_time:
                        timestamps.append(timestamp)

        return sorted(timestamps)

    @staticmethod
    def adaptive_delay(
        success_rate: float,
        min_delay: float = 0.1,
        max_delay: float = 30.0
    ) -> float:
        """
        Calculate adaptive delay based on success rate.

        If success rate is low (many blocks), increase delay to be stealthier.

        Args:
            success_rate: Current success rate (0.0 to 1.0)
            min_delay: Minimum delay in seconds
            max_delay: Maximum delay in seconds

        Returns:
            Adaptive delay in seconds
        """
        # Lower success rate = longer delay
        delay = min_delay + (max_delay - min_delay) * (1 - success_rate)
        return delay + random.uniform(0, delay * 0.2)  # Add jitter
