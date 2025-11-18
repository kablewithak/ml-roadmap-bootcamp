"""Safety Valve: The compliance layer that prevents catastrophic exploration.

This module implements hard-coded rules that override the exploration logic
to ensure legal compliance and risk management.
"""

import structlog
from dataclasses import dataclass
from typing import NamedTuple

from .models import UserFeatures

logger = structlog.get_logger(__name__)


class ComplianceCheck(NamedTuple):
    """Result of a compliance check."""
    passed: bool
    reason: str | None


@dataclass
class SafetyValve:
    """
    The Safety Valve (The "Brakes").

    Prevents the exploration mode from making decisions that are:
    - Illegal (e.g., discriminatory)
    - Catastrophic (e.g., approving obviously fraudulent applications)
    - Non-compliant (e.g., violating lending regulations)

    This runs BEFORE the traffic router to override any exploration decision.
    """

    # Compliance thresholds
    min_age: int = 18
    max_debt_to_income_ratio: float = 0.65
    min_income_for_loan: float = 12000.0  # Annual income floor

    def check_all_rules(self, features: UserFeatures) -> ComplianceCheck:
        """
        Run all compliance rules against user features.

        Args:
            features: User features to check

        Returns:
            ComplianceCheck with pass/fail status and reason
        """
        # Rule 1: Age check (legal requirement)
        if features.age < self.min_age:
            logger.warning(
                "compliance_block",
                user_id=features.user_id,
                rule="age_requirement",
                age=features.age
            )
            return ComplianceCheck(
                passed=False,
                reason=f"Applicant must be at least {self.min_age} years old"
            )

        # Rule 2: Recent bankruptcy
        if features.recent_bankruptcy:
            logger.warning(
                "compliance_block",
                user_id=features.user_id,
                rule="recent_bankruptcy"
            )
            return ComplianceCheck(
                passed=False,
                reason="Recent bankruptcy within 7 years"
            )

        # Rule 3: Debt-to-income ratio
        if features.income > 0:
            dti_ratio = features.debt / features.income
            if dti_ratio > self.max_debt_to_income_ratio:
                logger.warning(
                    "compliance_block",
                    user_id=features.user_id,
                    rule="debt_to_income",
                    dti_ratio=dti_ratio
                )
                return ComplianceCheck(
                    passed=False,
                    reason=f"Debt-to-income ratio {dti_ratio:.2f} exceeds limit"
                )

        # Rule 4: Minimum income
        if features.income < self.min_income_for_loan:
            logger.warning(
                "compliance_block",
                user_id=features.user_id,
                rule="minimum_income",
                income=features.income
            )
            return ComplianceCheck(
                passed=False,
                reason=f"Income below minimum threshold of ${self.min_income_for_loan}"
            )

        # Rule 5: Credit score floor (regulatory requirement)
        if features.credit_score < 300:
            logger.warning(
                "compliance_block",
                user_id=features.user_id,
                rule="invalid_credit_score",
                credit_score=features.credit_score
            )
            return ComplianceCheck(
                passed=False,
                reason="Invalid credit score"
            )

        # All rules passed
        return ComplianceCheck(passed=True, reason=None)

    def is_high_risk_exploration(self, features: UserFeatures) -> bool:
        """
        Check if exploration would be too risky even if compliant.

        This is a soft check for cases where we want to reduce
        exploration probability but not block entirely.

        Args:
            features: User features to check

        Returns:
            True if this is a high-risk exploration candidate
        """
        # Very low credit score
        if features.credit_score < 550:
            return True

        # High existing debt
        if features.debt > 100000:
            return True

        # Short employment history with low income
        if features.employment_years < 1 and features.income < 30000:
            return True

        return False


def is_legally_prohibited(features: UserFeatures) -> bool:
    """
    Quick check if a loan would be legally prohibited.

    This is the function referenced in the policy.py code block.

    Args:
        features: User features to check

    Returns:
        True if the loan would be legally prohibited
    """
    valve = SafetyValve()
    check = valve.check_all_rules(features)
    return not check.passed


def get_compliance_reason(features: UserFeatures) -> str | None:
    """
    Get the reason why a loan is prohibited, if any.

    Args:
        features: User features to check

    Returns:
        Reason string if prohibited, None if allowed
    """
    valve = SafetyValve()
    check = valve.check_all_rules(features)
    return check.reason
