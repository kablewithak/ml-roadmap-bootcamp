"""Implementation of 10 realistic fraud attack patterns."""

import random
from datetime import datetime, timedelta
from typing import List, Optional
from fraud_detection.attacks.base import BaseAttack
from fraud_detection.types import (
    Transaction,
    TransactionType,
    FraudLabel,
    AttackType,
)
from fraud_detection.utils.data_generation import (
    generate_user_id,
    generate_device_id,
    generate_card_bin,
    generate_card_last4,
)
from fraud_detection.utils.timing import RealisticTimer
from faker import Faker


fake = Faker()


class CardTestingAttack(BaseAttack):
    """
    Card Testing Attack: Test 1000 cards with $1 each.

    Attackers test stolen card numbers with small transactions to identify
    valid cards before making larger fraudulent purchases.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        transactions = []
        device_id = generate_device_id()
        ip_address = fake.ipv4()
        merchant_id = f"merchant_{random.randint(1, 20)}"

        # Adaptive delay based on success rate
        adaptations = self.adapt_strategy()
        base_delay = 0.5 * adaptations.get("stealth_multiplier", 1.0)

        current_time = start_time

        for i in range(self.pattern.num_transactions):
            # Generate different card numbers
            card_bin = generate_card_bin()
            card_last4 = generate_card_last4()

            # Test with $1 transaction
            txn = Transaction(
                timestamp=current_time,
                user_id=generate_user_id(),
                merchant_id=merchant_id,
                amount=1.0,  # Small test amount
                transaction_type=TransactionType.PURCHASE,
                device_id=device_id,  # Same device
                ip_address=ip_address,  # Same IP
                user_agent=fake.user_agent(),
                card_bin=card_bin,
                card_last4=card_last4,
                country="US",
                is_first_transaction=True,
                time_since_account_creation_hours=0.1,
                transactions_last_24h=i,
                is_fraud=True,
                fraud_label=FraudLabel.FRAUDULENT,
                metadata={"attack_type": "card_testing", "sequence": i}
            )

            transactions.append(txn)

            # Realistic timing with adaptive delays
            delay = RealisticTimer.automation_delay() * adaptations.get("stealth_multiplier", 1.0)
            current_time += timedelta(seconds=delay)

            # If getting blocked, slow down
            if defense_callback and i > 0 and i % 10 == 0:
                success_rate = self.get_current_success_rate()
                if success_rate < 0.5:
                    current_time += timedelta(seconds=RealisticTimer.stealthy_delay())

        return transactions


class AccountTakeoverAttack(BaseAttack):
    """
    Account Takeover: Login → Change Email → Purchase.

    Multi-step attack that takes over a legitimate account.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        transactions = []
        device_id = generate_device_id()  # Attacker's device
        ip_address = fake.ipv4()  # Attacker's IP

        adaptations = self.adapt_strategy()
        current_time = start_time

        # Step 1: Small test purchase after takeover
        txn1 = Transaction(
            timestamp=current_time,
            user_id=f"user_{fake.uuid4()[:8]}",  # Existing user
            merchant_id=f"merchant_{random.randint(1, 50)}",
            amount=random.uniform(10, 50),
            transaction_type=TransactionType.PURCHASE,
            device_id=device_id,  # New device
            ip_address=ip_address,  # New IP
            user_agent=fake.user_agent(),
            card_bin=generate_card_bin(),
            card_last4=generate_card_last4(),
            country=random.choice(["US", "CA", "GB", "AU"]),
            is_first_transaction=False,  # Existing account
            time_since_account_creation_hours=random.uniform(720, 8760),  # 30-365 days
            transactions_last_24h=0,
            is_fraud=True,
            fraud_label=FraudLabel.FRAUDULENT,
            metadata={"attack_type": "account_takeover", "stage": "test_purchase"}
        )
        transactions.append(txn1)

        # Human think time
        current_time += timedelta(seconds=RealisticTimer.think_time())

        # Step 2: Larger purchase if test succeeds
        if not defense_callback or not defense_callback(txn1):
            current_time += timedelta(minutes=random.uniform(5, 30))

            txn2 = Transaction(
                timestamp=current_time,
                user_id=txn1.user_id,  # Same account
                merchant_id=f"merchant_{random.randint(1, 50)}",
                amount=random.uniform(200, 1000),
                transaction_type=TransactionType.PURCHASE,
                device_id=device_id,
                ip_address=ip_address,
                user_agent=txn1.user_agent,
                card_bin=txn1.card_bin,
                card_last4=txn1.card_last4,
                country=txn1.country,
                is_first_transaction=False,
                time_since_account_creation_hours=txn1.time_since_account_creation_hours,
                transactions_last_24h=1,
                is_fraud=True,
                fraud_label=FraudLabel.FRAUDULENT,
                metadata={"attack_type": "account_takeover", "stage": "large_purchase"}
            )
            transactions.append(txn2)

        return transactions


class VelocityEvasionAttack(BaseAttack):
    """
    Velocity Evasion: Stay under velocity limits.

    Attackers carefully time transactions to stay under detection thresholds.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        transactions = []
        user_id = generate_user_id()
        device_id = generate_device_id()

        # Stay under typical velocity limits (e.g., 5 transactions per hour)
        transactions_per_hour = 4
        hours_duration = self.pattern.duration_hours

        current_time = start_time

        for hour in range(int(hours_duration)):
            for i in range(transactions_per_hour):
                # Spread transactions within the hour
                offset_minutes = (60 / transactions_per_hour) * i + random.uniform(-5, 5)
                txn_time = current_time + timedelta(minutes=offset_minutes)

                txn = Transaction(
                    timestamp=txn_time,
                    user_id=user_id,
                    merchant_id=f"merchant_{random.randint(1, 100)}",
                    amount=random.uniform(50, 200),
                    transaction_type=TransactionType.PURCHASE,
                    device_id=device_id,
                    ip_address=fake.ipv4(),
                    user_agent=fake.user_agent(),
                    card_bin=generate_card_bin(),
                    card_last4=generate_card_last4(),
                    country="US",
                    is_first_transaction=(hour == 0 and i == 0),
                    time_since_account_creation_hours=random.uniform(24, 720),
                    transactions_last_24h=hour * transactions_per_hour + i,
                    is_fraud=True,
                    fraud_label=FraudLabel.FRAUDULENT,
                    metadata={"attack_type": "velocity_evasion", "hour": hour}
                )
                transactions.append(txn)

            current_time += timedelta(hours=1)

        return sorted(transactions, key=lambda t: t.timestamp)


class SyntheticIdentityAttack(BaseAttack):
    """
    Synthetic Identity: Fake person with real SSN pattern.

    Creates a fake identity that passes basic validation checks.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        from fraud_detection.utils.data_generation import generate_synthetic_identity

        transactions = []

        # Create synthetic identity
        identity = generate_synthetic_identity()

        current_time = start_time

        # Build credit slowly over time (slow burn approach)
        num_phases = 3
        transactions_per_phase = self.pattern.num_transactions // num_phases

        for phase in range(num_phases):
            phase_duration = self.pattern.duration_hours / num_phases

            for i in range(transactions_per_phase):
                # Transaction amounts increase over time (building trust)
                if phase == 0:
                    amount = random.uniform(10, 50)  # Small amounts
                elif phase == 1:
                    amount = random.uniform(50, 200)  # Medium amounts
                else:
                    amount = random.uniform(200, 1000)  # Large cashout

                offset_hours = random.uniform(0, phase_duration)
                txn_time = current_time + timedelta(hours=offset_hours)

                txn = Transaction(
                    timestamp=txn_time,
                    user_id=identity["user_id"],
                    merchant_id=f"merchant_{random.randint(1, 100)}",
                    amount=amount,
                    transaction_type=TransactionType.PURCHASE,
                    device_id=identity["device_id"],
                    ip_address=identity["ip_address"],
                    user_agent=fake.user_agent(),
                    card_bin=generate_card_bin(),
                    card_last4=generate_card_last4(),
                    country="US",
                    is_first_transaction=(phase == 0 and i == 0),
                    time_since_account_creation_hours=(current_time - datetime.fromisoformat(
                        identity["created_at"]
                    )).total_seconds() / 3600,
                    transactions_last_24h=i,
                    is_fraud=True,
                    fraud_label=FraudLabel.FRAUDULENT,
                    metadata={
                        "attack_type": "synthetic_identity",
                        "phase": phase,
                        "identity_name": identity["name"]
                    }
                )
                transactions.append(txn)

            current_time += timedelta(hours=phase_duration)

        return sorted(transactions, key=lambda t: t.timestamp)


class DeviceRotationAttack(BaseAttack):
    """
    Device Rotation: Rotate device IDs to avoid device-based detection.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        transactions = []
        user_id = generate_user_id()

        # Create pool of devices to rotate through
        device_pool = [generate_device_id() for _ in range(10)]

        adaptations = self.adapt_strategy()
        current_time = start_time

        for i in range(self.pattern.num_transactions):
            # Rotate devices
            device_id = device_pool[i % len(device_pool)]

            txn = Transaction(
                timestamp=current_time,
                user_id=user_id,
                merchant_id=f"merchant_{random.randint(1, 50)}",
                amount=random.uniform(100, 300),
                transaction_type=TransactionType.PURCHASE,
                device_id=device_id,  # Rotating device
                ip_address=fake.ipv4(),
                user_agent=fake.user_agent(),
                card_bin=generate_card_bin(),
                card_last4=generate_card_last4(),
                country="US",
                is_first_transaction=(i == 0),
                time_since_account_creation_hours=random.uniform(1, 24),
                transactions_last_24h=i,
                is_fraud=True,
                fraud_label=FraudLabel.FRAUDULENT,
                metadata={"attack_type": "device_rotation", "device_index": i % len(device_pool)}
            )
            transactions.append(txn)

            delay = RealisticTimer.automation_delay() * adaptations.get("stealth_multiplier", 1.0)
            current_time += timedelta(seconds=delay)

        return transactions


class IPRotationAttack(BaseAttack):
    """
    IP Rotation: Use proxy network to rotate IP addresses.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        transactions = []
        user_id = generate_user_id()
        device_id = generate_device_id()

        # Create pool of IPs (simulating proxy network)
        ip_pool = [fake.ipv4() for _ in range(20)]

        current_time = start_time

        for i in range(self.pattern.num_transactions):
            # Rotate IPs frequently
            ip_address = ip_pool[i % len(ip_pool)]

            txn = Transaction(
                timestamp=current_time,
                user_id=user_id,
                merchant_id=f"merchant_{random.randint(1, 50)}",
                amount=random.uniform(150, 400),
                transaction_type=TransactionType.PURCHASE,
                device_id=device_id,
                ip_address=ip_address,  # Rotating IP
                user_agent=fake.user_agent(),
                card_bin=generate_card_bin(),
                card_last4=generate_card_last4(),
                country=random.choice(["US", "CA", "MX", "BR", "AR"]),
                is_first_transaction=(i == 0),
                time_since_account_creation_hours=random.uniform(1, 48),
                transactions_last_24h=i,
                is_fraud=True,
                fraud_label=FraudLabel.FRAUDULENT,
                metadata={"attack_type": "ip_rotation", "ip_index": i % len(ip_pool)}
            )
            transactions.append(txn)

            current_time += timedelta(seconds=random.uniform(30, 120))

        return transactions


class BlendAttack(BaseAttack):
    """
    Blend Attack: Mix fraudulent transactions with legitimate ones.

    Makes detection harder by blending in with normal traffic.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        from fraud_detection.utils.data_generation import generate_legitimate_transactions

        transactions = []

        # Generate mix of fraud and legitimate
        num_fraud = int(self.pattern.num_transactions * 0.3)
        num_legit = self.pattern.num_transactions - num_fraud

        # Generate legitimate transactions
        legit_txns = generate_legitimate_transactions(num_legit, start_time)
        transactions.extend(legit_txns)

        # Interleave fraudulent transactions
        user_id = generate_user_id()
        device_id = generate_device_id()

        for i in range(num_fraud):
            # Time fraud transactions to blend with legitimate traffic
            offset_hours = random.uniform(0, self.pattern.duration_hours)
            txn_time = start_time + timedelta(hours=offset_hours)

            txn = Transaction(
                timestamp=txn_time,
                user_id=user_id,
                merchant_id=f"merchant_{random.randint(1, 100)}",
                amount=random.uniform(50, 300),  # Similar to legitimate amounts
                transaction_type=TransactionType.PURCHASE,
                device_id=device_id,
                ip_address=fake.ipv4(),
                user_agent=fake.user_agent(),
                card_bin=generate_card_bin(),
                card_last4=generate_card_last4(),
                country="US",
                is_first_transaction=(i == 0),
                time_since_account_creation_hours=random.uniform(100, 1000),
                transactions_last_24h=random.randint(0, 3),  # Normal velocity
                is_fraud=True,
                fraud_label=FraudLabel.FRAUDULENT,
                metadata={"attack_type": "blend_attack"}
            )
            transactions.append(txn)

        return sorted(transactions, key=lambda t: t.timestamp)


class SlowBurnAttack(BaseAttack):
    """
    Slow Burn: Low volume over extended time period.

    Extremely stealthy, hard to detect due to low velocity.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        transactions = []
        user_id = generate_user_id()
        device_id = generate_device_id()

        # Spread transactions over entire duration
        total_hours = self.pattern.duration_hours
        hours_between_txns = total_hours / self.pattern.num_transactions

        current_time = start_time

        for i in range(self.pattern.num_transactions):
            txn = Transaction(
                timestamp=current_time,
                user_id=user_id,
                merchant_id=f"merchant_{random.randint(1, 100)}",
                amount=random.uniform(200, 500),
                transaction_type=TransactionType.PURCHASE,
                device_id=device_id,
                ip_address=fake.ipv4() if random.random() < 0.3 else fake.ipv4(),
                user_agent=fake.user_agent(),
                card_bin=generate_card_bin(),
                card_last4=generate_card_last4(),
                country="US",
                is_first_transaction=(i == 0),
                time_since_account_creation_hours=i * hours_between_txns,
                transactions_last_24h=max(1, int(24 / hours_between_txns)),
                is_fraud=True,
                fraud_label=FraudLabel.FRAUDULENT,
                metadata={"attack_type": "slow_burn", "sequence": i}
            )
            transactions.append(txn)

            # Long delay between transactions
            current_time += timedelta(hours=hours_between_txns + random.uniform(-1, 1))

        return transactions


class BINAttack(BaseAttack):
    """
    BIN Attack: Test card number ranges systematically.

    Tests Bank Identification Numbers to find valid card ranges.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        transactions = []
        device_id = generate_device_id()
        ip_address = fake.ipv4()

        # Test a range of BINs
        base_bin = "424242"  # Start with a common test BIN

        adaptations = self.adapt_strategy()
        current_time = start_time

        for i in range(self.pattern.num_transactions):
            # Increment BIN systematically
            bin_number = int(base_bin) + i
            card_bin = str(bin_number)[:6]

            txn = Transaction(
                timestamp=current_time,
                user_id=generate_user_id(),
                merchant_id=f"merchant_{random.randint(1, 20)}",
                amount=random.uniform(1, 10),  # Small test amounts
                transaction_type=TransactionType.PURCHASE,
                device_id=device_id,
                ip_address=ip_address,
                user_agent=fake.user_agent(),
                card_bin=card_bin,
                card_last4=generate_card_last4(),
                country="US",
                is_first_transaction=True,
                time_since_account_creation_hours=0.1,
                transactions_last_24h=i,
                is_fraud=True,
                fraud_label=FraudLabel.FRAUDULENT,
                metadata={"attack_type": "bin_attack", "bin_tested": card_bin}
            )
            transactions.append(txn)

            delay = RealisticTimer.automation_delay() * adaptations.get("stealth_multiplier", 1.0)
            current_time += timedelta(seconds=delay)

        return transactions


class GiftCardCashoutAttack(BaseAttack):
    """
    Gift Card Cashout: Chain of gift card purchases and usage.

    Buy gift cards with stolen cards, then use them quickly.
    """

    def generate_attack_transactions(
        self,
        start_time: datetime,
        defense_callback: Optional[callable] = None
    ) -> List[Transaction]:
        transactions = []
        device_id = generate_device_id()

        current_time = start_time

        # Phase 1: Buy gift cards
        gift_card_amounts = [25, 50, 100, 250]
        num_cards = self.pattern.num_transactions // 2

        for i in range(num_cards):
            amount = random.choice(gift_card_amounts)

            txn = Transaction(
                timestamp=current_time,
                user_id=generate_user_id(),
                merchant_id="merchant_giftcard_store",
                amount=float(amount),
                transaction_type=TransactionType.PURCHASE,
                device_id=device_id,
                ip_address=fake.ipv4(),
                user_agent=fake.user_agent(),
                card_bin=generate_card_bin(),
                card_last4=generate_card_last4(),
                country="US",
                is_first_transaction=True,
                time_since_account_creation_hours=0.5,
                transactions_last_24h=i,
                is_fraud=True,
                fraud_label=FraudLabel.FRAUDULENT,
                metadata={
                    "attack_type": "gift_card_cashout",
                    "phase": "purchase",
                    "gift_card_value": amount
                }
            )
            transactions.append(txn)

            current_time += timedelta(minutes=random.uniform(2, 10))

        # Phase 2: Quickly use gift cards
        current_time += timedelta(minutes=random.uniform(10, 30))

        for i in range(num_cards):
            txn = Transaction(
                timestamp=current_time,
                user_id=generate_user_id(),
                merchant_id=f"merchant_{random.randint(50, 100)}",
                amount=random.choice(gift_card_amounts),
                transaction_type=TransactionType.PURCHASE,
                device_id=generate_device_id(),  # Different device
                ip_address=fake.ipv4(),
                user_agent=fake.user_agent(),
                card_bin="999999",  # Gift card indicator
                card_last4="0000",
                country="US",
                is_first_transaction=True,
                time_since_account_creation_hours=0.1,
                transactions_last_24h=i,
                is_fraud=True,
                fraud_label=FraudLabel.FRAUDULENT,
                metadata={
                    "attack_type": "gift_card_cashout",
                    "phase": "redemption",
                    "payment_method": "gift_card"
                }
            )
            transactions.append(txn)

            current_time += timedelta(minutes=random.uniform(1, 5))

        return sorted(transactions, key=lambda t: t.timestamp)
