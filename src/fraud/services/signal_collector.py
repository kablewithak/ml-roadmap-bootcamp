"""
Transaction pattern signal collection service.
Analyzes transaction patterns beyond simple velocity.
"""

import logging
from datetime import datetime
from typing import Dict, Tuple

from ..models import TransactionRequest, PatternSignals, VelocitySignals
from ...infrastructure.redis.velocity_tracker import VelocityTracker

logger = logging.getLogger(__name__)


class SignalCollector:
    """
    Collects and analyzes transaction patterns for fraud detection.
    Works in conjunction with VelocityTracker for comprehensive signals.
    """

    def __init__(
        self,
        velocity_tracker: VelocityTracker,
        small_tx_threshold: float = 10.0,
        new_card_high_value: float = 1000.0,
        merchant_switch_threshold: int = 3,
    ):
        self.velocity_tracker = velocity_tracker
        self.small_tx_threshold = small_tx_threshold
        self.new_card_high_value = new_card_high_value
        self.merchant_switch_threshold = merchant_switch_threshold

    async def collect_all_signals(
        self,
        transaction: TransactionRequest
    ) -> Tuple[VelocitySignals, PatternSignals]:
        """
        Collect all fraud signals for a transaction.

        Returns:
            Tuple of (VelocitySignals, PatternSignals)
        """
        # Get velocity signals from Redis
        velocity_data = await self.velocity_tracker.get_velocity_signals(
            card_id=transaction.card_id,
            user_id=transaction.user_id,
            ip_address=transaction.ip_address,
            amount=transaction.amount,
            timestamp=transaction.timestamp
        )

        velocity_signals = VelocitySignals(**velocity_data)

        # Collect pattern signals
        pattern_signals = await self._collect_pattern_signals(transaction, velocity_data)

        return velocity_signals, pattern_signals

    async def _collect_pattern_signals(
        self,
        transaction: TransactionRequest,
        velocity_data: Dict
    ) -> PatternSignals:
        """Analyze transaction patterns."""

        # Check card testing pattern
        is_card_testing, card_testing_details = await self.velocity_tracker.check_card_testing_pattern(
            card_id=transaction.card_id,
            amount=transaction.amount,
            small_tx_threshold=self.small_tx_threshold
        )

        # First-time card usage
        is_first_card_use = velocity_data.get("is_first_card_use", False)

        # Merchant category switching
        merchant_category_count = velocity_data.get("merchant_category_count", 0)
        is_merchant_category_switch = merchant_category_count >= self.merchant_switch_threshold

        # Time pattern analysis
        is_unusual_hour = self._analyze_time_patterns(
            transaction.timestamp,
            velocity_data.get("hour_patterns", {}),
            velocity_data.get("current_hour_tx_count", 0)
        )

        return PatternSignals(
            is_first_card_use=is_first_card_use,
            merchant_category_count=merchant_category_count,
            current_hour_tx_count=velocity_data.get("current_hour_tx_count", 0),
            hour_patterns=velocity_data.get("hour_patterns", {}),
            is_card_testing=is_card_testing,
            card_testing_details=card_testing_details,
            is_unusual_hour=is_unusual_hour,
            is_merchant_category_switch=is_merchant_category_switch,
        )

    def _analyze_time_patterns(
        self,
        timestamp: datetime,
        hour_patterns: Dict[int, int],
        current_hour_count: int
    ) -> bool:
        """
        Detect unusual transaction timing.

        A transaction is unusual if:
        1. It occurs in an hour with very few historical transactions
        2. It's outside typical business hours (2am-6am) with no history
        """
        current_hour = timestamp.hour

        # No historical data yet
        if not hour_patterns:
            # Flag transactions in early morning hours
            return 2 <= current_hour <= 6

        # Calculate average transactions per hour
        total_tx = sum(hour_patterns.values())
        avg_per_hour = total_tx / len(hour_patterns) if hour_patterns else 0

        # Current hour has significantly fewer transactions than average
        if current_hour in hour_patterns:
            hour_tx = hour_patterns[current_hour]
            is_unusual = hour_tx < avg_per_hour * 0.3  # Less than 30% of average
        else:
            # New hour, flag if in unusual time range
            is_unusual = 2 <= current_hour <= 6

        return is_unusual

    async def track_transaction(self, transaction: TransactionRequest) -> Dict:
        """
        Record transaction in Redis for future velocity calculations.
        Called AFTER fraud check but BEFORE payment processing.
        """
        result = await self.velocity_tracker.track_transaction(
            card_id=transaction.card_id,
            user_id=transaction.user_id,
            ip_address=transaction.ip_address,
            amount=transaction.amount,
            merchant_category=transaction.merchant_category,
            timestamp=transaction.timestamp
        )

        logger.info(
            f"Transaction tracked: {transaction.transaction_id}, "
            f"latency: {result.get('latency_ms', 0):.2f}ms"
        )

        return result
