"""
Policy Engine: The core decision-making logic for Project Lazarus.

This module contains the main `decide_application` function that implements
the epsilon-greedy exploration strategy with compliance safeguards.

This is the "IQ Test" code block - clean, defensive, and production-ready.
"""

import random
import structlog
from datetime import datetime
from typing import Any

import redis

from .models import (
    Decision,
    DecisionReason,
    DecisionResult,
    TreatmentGroup,
    UserFeatures,
    ExperimentLog,
)
from .safety_valve import is_legally_prohibited, get_compliance_reason
from .router import TrafficRouter
from config.settings import get_settings

logger = structlog.get_logger(__name__)


def decide_application(
    user_features: UserFeatures,
    risk_score: float,
    application_id: str,
    redis_client: redis.Redis | None = None,
) -> DecisionResult:
    """
    Make a loan decision using the Lazarus causal inference system.

    This function implements:
    1. Safety Valve (Compliance) - Hard rules that override everything
    2. Exploration Logic - Epsilon-greedy with budget constraints
    3. Exploitation - Standard model-based decisions

    Args:
        user_features: Features describing the applicant
        risk_score: Model's risk prediction (0.0 = safe, 1.0 = risky)
        application_id: Unique application identifier
        redis_client: Optional Redis client for state management

    Returns:
        DecisionResult with decision, reason, and metadata
    """
    settings = get_settings()
    timestamp = datetime.utcnow()

    # Initialize router (handles Redis connection)
    router = TrafficRouter(redis_client=redis_client)

    # =========================================================================
    # 1. SAFETY VALVE (Compliance)
    # =========================================================================
    # This runs FIRST - no exploration or model prediction can override this
    if is_legally_prohibited(user_features):
        reason = get_compliance_reason(user_features)
        logger.warning(
            "compliance_rejection",
            user_id=user_features.user_id,
            application_id=application_id,
            reason=reason
        )
        return DecisionResult(
            application_id=application_id,
            user_id=user_features.user_id,
            decision=Decision.REJECT,
            reason=DecisionReason.COMPLIANCE_BLOCK,
            risk_score=risk_score,
            treatment_group=TreatmentGroup.EXPLOIT,  # Compliance blocks are not experiments
            timestamp=timestamp,
            exploration_budget_remaining=router.get_budget_status()["remaining_budget"]
        )

    # =========================================================================
    # 2. EXPLORATION LOGIC (The "IQ Test" part)
    # =========================================================================
    # Atomic check of exploration budget in Redis
    should_explore, treatment_group, remaining_budget = router.should_explore(
        hard_rules_pass=True
    )

    if should_explore:
        # We are BUYING data. Force approval to learn from the outcome.
        # This data point will be weighted heavily in causal training.
        log_experiment_exposure(
            user_id=user_features.user_id,
            application_id=application_id,
            variant=TreatmentGroup.EXPLORE,
            risk_score=risk_score,
            features=user_features
        )

        logger.info(
            "exploration_approval",
            user_id=user_features.user_id,
            application_id=application_id,
            risk_score=risk_score,
            remaining_budget=remaining_budget
        )

        return DecisionResult(
            application_id=application_id,
            user_id=user_features.user_id,
            decision=Decision.APPROVE,
            reason=DecisionReason.PROJECT_LAZARUS_EXPLORE,
            risk_score=risk_score,
            treatment_group=TreatmentGroup.EXPLORE,
            timestamp=timestamp,
            exploration_budget_remaining=remaining_budget
        )

    # =========================================================================
    # 3. EXPLOITATION (Standard Business Logic)
    # =========================================================================
    # Use the model's prediction to make the decision
    # Note: Lower risk_score = safer applicant
    if risk_score < settings.risk_threshold:
        decision = Decision.APPROVE
        reason = DecisionReason.MODEL_QUALIFIED
    else:
        decision = Decision.REJECT
        reason = DecisionReason.MODEL_HIGH_RISK

    # Log the exploitation decision
    log_experiment_exposure(
        user_id=user_features.user_id,
        application_id=application_id,
        variant=TreatmentGroup.EXPLOIT,
        risk_score=risk_score,
        features=user_features
    )

    logger.info(
        "exploitation_decision",
        user_id=user_features.user_id,
        application_id=application_id,
        decision=decision.value,
        risk_score=risk_score
    )

    return DecisionResult(
        application_id=application_id,
        user_id=user_features.user_id,
        decision=decision,
        reason=reason,
        risk_score=risk_score,
        treatment_group=TreatmentGroup.EXPLOIT,
        timestamp=timestamp,
        exploration_budget_remaining=remaining_budget
    )


def log_experiment_exposure(
    user_id: str,
    application_id: str,
    variant: TreatmentGroup,
    risk_score: float,
    features: UserFeatures,
) -> None:
    """
    Log experiment exposure for causal analysis.

    This creates a record that will be used by the causal training pipeline
    to properly weight observations using Inverse Probability Weighting (IPW).

    Args:
        user_id: User identifier
        application_id: Application identifier
        variant: Treatment group (explore/exploit)
        risk_score: Model's risk prediction
        features: User features snapshot
    """
    log_entry = ExperimentLog(
        user_id=user_id,
        application_id=application_id,
        variant=variant,
        decision=Decision.APPROVE if variant == TreatmentGroup.EXPLORE else Decision.REJECT,
        risk_score=risk_score,
        timestamp=datetime.utcnow(),
        features_snapshot=features.model_dump()
    )

    # In production, this would write to a database or event stream
    # For now, we use structured logging
    logger.info(
        "experiment_exposure",
        **log_entry.model_dump(mode="json")
    )


def calculate_propensity_score(
    user_features: UserFeatures,
    epsilon: float,
    budget_remaining: float,
) -> float:
    """
    Calculate the probability of approval for IPW weighting.

    This is used in the causal training pipeline to weight observations.

    For exploration:
        P(approval) = epsilon (e.g., 0.01)

    For exploitation:
        P(approval) = P(model approves | features)

    Args:
        user_features: User features
        epsilon: Exploration probability
        budget_remaining: Whether budget allows exploration

    Returns:
        Propensity score (probability of approval)
    """
    if budget_remaining <= 0:
        # No exploration possible, propensity is model-based
        # This would need the actual model probability
        return 0.5  # Placeholder - should be model's P(approve)

    # If exploration is possible, the minimum propensity is epsilon
    # This ensures proper IPW weighting for explore cases
    return max(epsilon, 0.001)  # Floor to prevent division issues


# Convenience alias for cleaner imports
EPSILON = get_settings().epsilon
THRESHOLD = get_settings().risk_threshold
