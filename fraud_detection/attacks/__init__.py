"""Attack simulation module for fraud detection."""

from fraud_detection.attacks.simulator import AttackSimulator
from fraud_detection.attacks.base import BaseAttack
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

__all__ = [
    "AttackSimulator",
    "BaseAttack",
    "CardTestingAttack",
    "AccountTakeoverAttack",
    "VelocityEvasionAttack",
    "SyntheticIdentityAttack",
    "DeviceRotationAttack",
    "IPRotationAttack",
    "BlendAttack",
    "SlowBurnAttack",
    "BINAttack",
    "GiftCardCashoutAttack",
]
