"""Base attack class for all fraud attack patterns."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fraud_detection.types import (
    Transaction,
    AttackType,
    AttackPattern,
    AttackResult,
    FraudLabel,
)


class BaseAttack(ABC):
    """
    Base class for all attack patterns.

    Provides common functionality for attack simulation including:
    - Adaptive behavior (learning from blocks)
    - Success metrics tracking
    - Historical loss estimation
    """

    def __init__(self, attack_pattern: AttackPattern):
        self.pattern = attack_pattern
        self.success_history: List[bool] = []
        self.blocked_count = 0
        self.success_count = 0
        self.total_attempted = 0

    @abstractmethod
    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        """
        Generate attack transactions.

        Args:
            start_time: When the attack starts
            defense_callback: Optional callback to check if transaction is blocked

        Returns:
            List of attack transactions
        """
        pass

    def execute_attack(
        self,
        start_time: datetime = None,
        defense_callback: Optional[callable] = None
    ) -> AttackResult:
        """
        Execute the attack and track results.

        Args:
            start_time: When to start the attack
            defense_callback: Function to check if transaction is blocked

        Returns:
            AttackResult with metrics
        """
        if start_time is None:
            start_time = datetime.utcnow()

        # Generate attack transactions
        transactions = self.generate_attack_transactions(start_time, defense_callback)

        # Track success/failure
        successful = []
        blocked = []

        for txn in transactions:
            self.total_attempted += 1

            if defense_callback:
                is_blocked = defense_callback(txn)
                if is_blocked:
                    self.blocked_count += 1
                    self.success_history.append(False)
                    blocked.append(txn)
                else:
                    self.success_count += 1
                    self.success_history.append(True)
                    successful.append(txn)
            else:
                # No defense, all succeed
                self.success_count += 1
                self.success_history.append(True)
                successful.append(txn)

        # Calculate metrics
        success_rate = self.success_count / self.total_attempted if self.total_attempted > 0 else 0
        detection_rate = self.blocked_count / self.total_attempted if self.total_attempted > 0 else 0

        # Estimate loss (successful fraud amount)
        estimated_loss = sum(txn.amount for txn in successful)

        end_time = start_time + timedelta(hours=self.pattern.duration_hours)

        return AttackResult(
            attack_type=self.pattern.attack_type,
            start_time=start_time,
            end_time=end_time,
            total_transactions=len(transactions),
            successful_transactions=len(successful),
            blocked_transactions=len(blocked),
            success_rate=success_rate,
            estimated_loss=estimated_loss,
            detection_rate=detection_rate,
            transactions=transactions,
            metadata={
                "pattern": self.pattern.name,
                "description": self.pattern.description,
            }
        )

    def get_current_success_rate(self) -> float:
        """Get current success rate from recent history."""
        if not self.success_history:
            return 0.5  # Assume 50% if no history

        # Use last 20 transactions
        recent = self.success_history[-20:]
        return sum(recent) / len(recent)

    def adapt_strategy(self) -> Dict[str, Any]:
        """
        Adapt attack strategy based on success rate.

        Returns:
            Dictionary with adapted parameters
        """
        success_rate = self.get_current_success_rate()

        adaptations = {
            "original_success_rate": success_rate,
            "adjustments": []
        }

        # If success rate is low, become more stealthy
        if success_rate < 0.3:
            adaptations["adjustments"].extend([
                "increase_delay",
                "reduce_velocity",
                "rotate_identifiers"
            ])
            adaptations["stealth_multiplier"] = 2.0

        # If success rate is medium, make minor adjustments
        elif success_rate < 0.7:
            adaptations["adjustments"].append("moderate_changes")
            adaptations["stealth_multiplier"] = 1.5

        # If success rate is high, can be more aggressive
        else:
            adaptations["adjustments"].append("increase_velocity")
            adaptations["stealth_multiplier"] = 0.8

        return adaptations

    def estimate_potential_loss(self) -> float:
        """
        Estimate potential loss if attack is successful.

        Returns:
            Estimated loss in dollars
        """
        # Estimate based on attack type and volume
        avg_transaction_value = self._get_average_transaction_value()
        return self.pattern.num_transactions * avg_transaction_value * self.pattern.success_threshold

    def _get_average_transaction_value(self) -> float:
        """Get average transaction value for this attack type."""
        # Default values by attack type
        avg_values = {
            AttackType.CARD_TESTING: 1.0,
            AttackType.ACCOUNT_TAKEOVER: 250.0,
            AttackType.VELOCITY_EVASION: 150.0,
            AttackType.SYNTHETIC_IDENTITY: 500.0,
            AttackType.DEVICE_ROTATION: 200.0,
            AttackType.IP_ROTATION: 180.0,
            AttackType.BLEND_ATTACK: 120.0,
            AttackType.SLOW_BURN: 300.0,
            AttackType.BIN_ATTACK: 5.0,
            AttackType.GIFT_CARD_CASHOUT: 100.0,
        }
        return avg_values.get(self.pattern.attack_type, 100.0)
