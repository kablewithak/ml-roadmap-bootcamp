"""Attack simulator that coordinates multiple attack patterns."""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from fraud_detection.attacks.patterns import (
    CardTestingAttack,
    AccountTakeoverAttack,
    VelocityEvasionAttack,
    SyntheticIdentityAttack,
    DeviceRotationAttack,
    IPRotationAttack,
    BlendAttack,
    SlowBurnAttack,
    BINAttack,
    GiftCardCashoutAttack,
)
from fraud_detection.types import AttackType, AttackPattern, AttackResult


class AttackSimulator:
    """
    Coordinates and executes multiple attack patterns.

    Supports:
    - Individual attack execution
    - Multi-day attack campaigns
    - Adaptive attacks that learn from defense
    """

    def __init__(self):
        self.attack_registry = {
            AttackType.CARD_TESTING: CardTestingAttack,
            AttackType.ACCOUNT_TAKEOVER: AccountTakeoverAttack,
            AttackType.VELOCITY_EVASION: VelocityEvasionAttack,
            AttackType.SYNTHETIC_IDENTITY: SyntheticIdentityAttack,
            AttackType.DEVICE_ROTATION: DeviceRotationAttack,
            AttackType.IP_ROTATION: IPRotationAttack,
            AttackType.BLEND_ATTACK: BlendAttack,
            AttackType.SLOW_BURN: SlowBurnAttack,
            AttackType.BIN_ATTACK: BINAttack,
            AttackType.GIFT_CARD_CASHOUT: GiftCardCashoutAttack,
        }

        self.attack_history: List[AttackResult] = []

    def create_attack_pattern(
        self,
        attack_type: AttackType,
        num_transactions: int = 100,
        duration_hours: float = 24.0,
        success_threshold: float = 0.5,
        adaptive: bool = True
    ) -> AttackPattern:
        """
        Create an attack pattern configuration.

        Args:
            attack_type: Type of attack
            num_transactions: Number of transactions in attack
            duration_hours: Duration of attack in hours
            success_threshold: Minimum success rate to consider attack successful
            adaptive: Whether attack adapts to defense responses

        Returns:
            AttackPattern configuration
        """
        attack_descriptions = {
            AttackType.CARD_TESTING: "Test 1000 cards with $1 transactions",
            AttackType.ACCOUNT_TAKEOVER: "Login → Change Email → Purchase chain",
            AttackType.VELOCITY_EVASION: "Stay under velocity detection limits",
            AttackType.SYNTHETIC_IDENTITY: "Fake person with real SSN pattern",
            AttackType.DEVICE_ROTATION: "Rotate device IDs to avoid detection",
            AttackType.IP_ROTATION: "Use proxy network to rotate IPs",
            AttackType.BLEND_ATTACK: "Mix fraud with legitimate traffic",
            AttackType.SLOW_BURN: "Low volume over extended time",
            AttackType.BIN_ATTACK: "Test card number ranges systematically",
            AttackType.GIFT_CARD_CASHOUT: "Buy gift cards → quick cashout",
        }

        return AttackPattern(
            attack_type=attack_type,
            name=attack_type.value,
            description=attack_descriptions.get(attack_type, "Unknown attack"),
            num_transactions=num_transactions,
            duration_hours=duration_hours,
            success_threshold=success_threshold,
            adaptive=adaptive
        )

    def execute_attack(
        self,
        attack_pattern: AttackPattern,
        start_time: Optional[datetime] = None,
        defense_callback: Optional[Callable] = None
    ) -> AttackResult:
        """
        Execute a single attack pattern.

        Args:
            attack_pattern: Attack configuration
            start_time: When to start the attack
            defense_callback: Function to check if transaction is blocked

        Returns:
            AttackResult with metrics
        """
        if start_time is None:
            start_time = datetime.utcnow()

        # Get attack class
        attack_class = self.attack_registry.get(attack_pattern.attack_type)
        if not attack_class:
            raise ValueError(f"Unknown attack type: {attack_pattern.attack_type}")

        # Create and execute attack
        attack = attack_class(attack_pattern)
        result = attack.execute_attack(start_time, defense_callback)

        # Store in history
        self.attack_history.append(result)

        return result

    def execute_campaign(
        self,
        attack_patterns: List[AttackPattern],
        campaign_duration_days: int = 7,
        start_time: Optional[datetime] = None,
        defense_callback: Optional[Callable] = None
    ) -> Dict[str, List[AttackResult]]:
        """
        Execute a multi-day attack campaign with multiple patterns.

        Args:
            attack_patterns: List of attack patterns to execute
            campaign_duration_days: Duration of campaign in days
            start_time: Campaign start time
            defense_callback: Function to check if transaction is blocked

        Returns:
            Dictionary mapping attack type to list of results
        """
        if start_time is None:
            start_time = datetime.utcnow()

        campaign_results: Dict[str, List[AttackResult]] = {}

        # Distribute attacks across campaign duration
        for pattern in attack_patterns:
            attack_type_name = pattern.attack_type.value
            campaign_results[attack_type_name] = []

            # Execute attack multiple times across campaign
            num_iterations = max(1, campaign_duration_days // 2)

            for i in range(num_iterations):
                # Space out attacks
                attack_start = start_time + timedelta(
                    days=i * (campaign_duration_days / num_iterations)
                )

                result = self.execute_attack(pattern, attack_start, defense_callback)
                campaign_results[attack_type_name].append(result)

        return campaign_results

    def get_all_attack_patterns(self) -> List[AttackPattern]:
        """
        Get default configurations for all 10 attack patterns.

        Returns:
            List of all attack patterns
        """
        patterns = [
            self.create_attack_pattern(
                AttackType.CARD_TESTING,
                num_transactions=1000,
                duration_hours=2.0,
                success_threshold=0.1
            ),
            self.create_attack_pattern(
                AttackType.ACCOUNT_TAKEOVER,
                num_transactions=10,
                duration_hours=1.0,
                success_threshold=0.7
            ),
            self.create_attack_pattern(
                AttackType.VELOCITY_EVASION,
                num_transactions=100,
                duration_hours=24.0,
                success_threshold=0.8
            ),
            self.create_attack_pattern(
                AttackType.SYNTHETIC_IDENTITY,
                num_transactions=50,
                duration_hours=168.0,  # 7 days
                success_threshold=0.7
            ),
            self.create_attack_pattern(
                AttackType.DEVICE_ROTATION,
                num_transactions=200,
                duration_hours=4.0,
                success_threshold=0.6
            ),
            self.create_attack_pattern(
                AttackType.IP_ROTATION,
                num_transactions=150,
                duration_hours=6.0,
                success_threshold=0.6
            ),
            self.create_attack_pattern(
                AttackType.BLEND_ATTACK,
                num_transactions=300,
                duration_hours=48.0,
                success_threshold=0.3
            ),
            self.create_attack_pattern(
                AttackType.SLOW_BURN,
                num_transactions=100,
                duration_hours=720.0,  # 30 days
                success_threshold=0.9
            ),
            self.create_attack_pattern(
                AttackType.BIN_ATTACK,
                num_transactions=500,
                duration_hours=3.0,
                success_threshold=0.2
            ),
            self.create_attack_pattern(
                AttackType.GIFT_CARD_CASHOUT,
                num_transactions=50,
                duration_hours=2.0,
                success_threshold=0.5
            ),
        ]

        return patterns

    def get_campaign_summary(self) -> Dict[str, any]:
        """
        Get summary of all executed attacks.

        Returns:
            Dictionary with campaign statistics
        """
        if not self.attack_history:
            return {"message": "No attacks executed yet"}

        total_transactions = sum(r.total_transactions for r in self.attack_history)
        total_successful = sum(r.successful_transactions for r in self.attack_history)
        total_blocked = sum(r.blocked_transactions for r in self.attack_history)
        total_loss = sum(r.estimated_loss for r in self.attack_history)

        avg_success_rate = total_successful / total_transactions if total_transactions > 0 else 0
        avg_detection_rate = total_blocked / total_transactions if total_transactions > 0 else 0

        # Group by attack type
        by_type: Dict[str, Dict] = {}
        for result in self.attack_history:
            attack_name = result.attack_type.value
            if attack_name not in by_type:
                by_type[attack_name] = {
                    "count": 0,
                    "total_transactions": 0,
                    "successful": 0,
                    "blocked": 0,
                    "total_loss": 0.0,
                }

            by_type[attack_name]["count"] += 1
            by_type[attack_name]["total_transactions"] += result.total_transactions
            by_type[attack_name]["successful"] += result.successful_transactions
            by_type[attack_name]["blocked"] += result.blocked_transactions
            by_type[attack_name]["total_loss"] += result.estimated_loss

        # Calculate rates by type
        for attack_name, stats in by_type.items():
            if stats["total_transactions"] > 0:
                stats["success_rate"] = stats["successful"] / stats["total_transactions"]
                stats["detection_rate"] = stats["blocked"] / stats["total_transactions"]

        return {
            "total_attacks": len(self.attack_history),
            "total_transactions": total_transactions,
            "successful_transactions": total_successful,
            "blocked_transactions": total_blocked,
            "overall_success_rate": avg_success_rate,
            "overall_detection_rate": avg_detection_rate,
            "total_estimated_loss": total_loss,
            "by_attack_type": by_type,
        }
