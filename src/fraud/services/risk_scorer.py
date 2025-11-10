"""
Risk score calculation engine with weighted rule system.
Fast, configurable, and explainable scoring.
"""

import time
import logging
from typing import Dict, List, Tuple

from ..models import (
    TransactionRequest,
    VelocitySignals,
    PatternSignals,
    RiskScore,
    RiskDecision,
    SignalWeights,
    RiskThresholds
)

logger = logging.getLogger(__name__)


class RiskScorer:
    """
    Weighted rule-based risk scoring engine.
    Calculates risk scores and makes approve/review/decline decisions.
    """

    def __init__(
        self,
        weights: SignalWeights,
        thresholds: RiskThresholds,
        velocity_limits: Dict[str, float]
    ):
        self.weights = weights
        self.thresholds = thresholds
        self.velocity_limits = velocity_limits

    def calculate_risk_score(
        self,
        transaction: TransactionRequest,
        velocity_signals: VelocitySignals,
        pattern_signals: PatternSignals
    ) -> RiskScore:
        """
        Calculate comprehensive risk score.

        Returns RiskScore with:
        - Overall risk score (0-1)
        - Decision (approve/review/decline)
        - Triggered signals
        - Individual signal scores
        """
        start_time = time.perf_counter()

        # Calculate individual signal scores
        signal_scores = {}
        signals_triggered = []

        # 1. Velocity count risk
        velocity_count_score, velocity_count_signals = self._score_velocity_count(
            velocity_signals
        )
        signal_scores["velocity_count"] = velocity_count_score
        signals_triggered.extend(velocity_count_signals)

        # 2. Velocity amount risk
        velocity_amount_score, velocity_amount_signals = self._score_velocity_amount(
            velocity_signals
        )
        signal_scores["velocity_amount"] = velocity_amount_score
        signals_triggered.extend(velocity_amount_signals)

        # 3. New card risk
        new_card_score, new_card_signals = self._score_new_card_risk(
            transaction.amount,
            pattern_signals.is_first_card_use
        )
        signal_scores["new_card_risk"] = new_card_score
        signals_triggered.extend(new_card_signals)

        # 4. Merchant pattern risk
        merchant_score, merchant_signals = self._score_merchant_patterns(
            pattern_signals
        )
        signal_scores["merchant_pattern"] = merchant_score
        signals_triggered.extend(merchant_signals)

        # 5. Time pattern risk
        time_score, time_signals = self._score_time_patterns(
            pattern_signals
        )
        signal_scores["time_pattern"] = time_score
        signals_triggered.extend(time_signals)

        # 6. Card testing risk
        card_testing_score, card_testing_signals = self._score_card_testing(
            pattern_signals
        )
        signal_scores["card_testing_pattern"] = card_testing_score
        signals_triggered.extend(card_testing_signals)

        # Calculate weighted overall score
        overall_score = (
            velocity_count_score * self.weights.velocity_count +
            velocity_amount_score * self.weights.velocity_amount +
            new_card_score * self.weights.new_card_risk +
            merchant_score * self.weights.merchant_pattern +
            time_score * self.weights.time_pattern +
            card_testing_score * self.weights.card_testing_pattern
        )

        # Ensure score is in [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))

        # Make decision based on thresholds
        decision = self._make_decision(overall_score)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Risk score calculated: {overall_score:.3f}, "
            f"decision: {decision.value}, "
            f"signals: {len(signals_triggered)}, "
            f"time: {processing_time_ms:.2f}ms"
        )

        return RiskScore(
            risk_score=overall_score,
            decision=decision,
            signals_triggered=signals_triggered,
            signal_scores=signal_scores,
            velocity_signals=velocity_signals,
            pattern_signals=pattern_signals,
            processing_time_ms=processing_time_ms
        )

    def _score_velocity_count(
        self,
        velocity_signals: VelocitySignals
    ) -> Tuple[float, List[str]]:
        """Score transaction count velocity."""
        score = 0.0
        signals = []

        # Card velocity
        if velocity_signals.card_count_5min > self.velocity_limits["transaction_count_5min"]:
            score = max(score, 0.8)
            signals.append(f"card_velocity_5min_exceeded:{velocity_signals.card_count_5min}")

        if velocity_signals.card_count_1hr > self.velocity_limits["transaction_count_1hr"]:
            score = max(score, 0.7)
            signals.append(f"card_velocity_1hr_exceeded:{velocity_signals.card_count_1hr}")

        # User velocity
        if velocity_signals.user_count_5min > self.velocity_limits["transaction_count_5min"]:
            score = max(score, 0.6)
            signals.append(f"user_velocity_5min_exceeded:{velocity_signals.user_count_5min}")

        # IP velocity
        if velocity_signals.ip_count_5min > self.velocity_limits["transaction_count_5min"]:
            score = max(score, 0.5)
            signals.append(f"ip_velocity_5min_exceeded:{velocity_signals.ip_count_5min}")

        # Gradual scoring for approaching limits
        if score == 0:
            max_ratio = max(
                velocity_signals.card_count_5min / max(1, self.velocity_limits["transaction_count_5min"]),
                velocity_signals.user_count_5min / max(1, self.velocity_limits["transaction_count_5min"])
            )
            score = min(0.3, max_ratio * 0.3)

        return score, signals

    def _score_velocity_amount(
        self,
        velocity_signals: VelocitySignals
    ) -> Tuple[float, List[str]]:
        """Score transaction amount velocity."""
        score = 0.0
        signals = []

        # Card amount velocity
        if velocity_signals.card_amount_5min > self.velocity_limits["amount_sum_5min"]:
            score = max(score, 0.9)
            signals.append(f"card_amount_5min_exceeded:{velocity_signals.card_amount_5min:.2f}")

        if velocity_signals.card_amount_1hr > self.velocity_limits["amount_sum_1hr"]:
            score = max(score, 0.8)
            signals.append(f"card_amount_1hr_exceeded:{velocity_signals.card_amount_1hr:.2f}")

        # User amount velocity
        if velocity_signals.user_amount_5min > self.velocity_limits["amount_sum_5min"]:
            score = max(score, 0.7)
            signals.append(f"user_amount_5min_exceeded:{velocity_signals.user_amount_5min:.2f}")

        # Gradual scoring
        if score == 0:
            max_ratio = max(
                velocity_signals.card_amount_5min / max(1, self.velocity_limits["amount_sum_5min"]),
                velocity_signals.user_amount_5min / max(1, self.velocity_limits["amount_sum_5min"])
            )
            score = min(0.4, max_ratio * 0.4)

        return score, signals

    def _score_new_card_risk(
        self,
        amount: float,
        is_first_use: bool
    ) -> Tuple[float, List[str]]:
        """Score risk of new card usage."""
        score = 0.0
        signals = []

        if is_first_use:
            new_card_threshold = self.velocity_limits["new_card_high_value"]

            if amount > new_card_threshold:
                score = 0.8
                signals.append(f"first_card_use_high_value:{amount:.2f}")
            elif amount > new_card_threshold * 0.5:
                score = 0.5
                signals.append(f"first_card_use_moderate_value:{amount:.2f}")
            else:
                score = 0.2
                signals.append("first_card_use_low_value")

        return score, signals

    def _score_merchant_patterns(
        self,
        pattern_signals: PatternSignals
    ) -> Tuple[float, List[str]]:
        """Score merchant category switching patterns."""
        score = 0.0
        signals = []

        if pattern_signals.is_merchant_category_switch:
            score = 0.6
            signals.append(
                f"merchant_category_switch:{pattern_signals.merchant_category_count}"
            )

        return score, signals

    def _score_time_patterns(
        self,
        pattern_signals: PatternSignals
    ) -> Tuple[float, List[str]]:
        """Score unusual time patterns."""
        score = 0.0
        signals = []

        if pattern_signals.is_unusual_hour:
            score = 0.4
            signals.append("unusual_hour_pattern")

        return score, signals

    def _score_card_testing(
        self,
        pattern_signals: PatternSignals
    ) -> Tuple[float, List[str]]:
        """Score card testing patterns."""
        score = 0.0
        signals = []

        if pattern_signals.is_card_testing:
            score = 0.9
            signals.append("card_testing_pattern_detected")
            details = pattern_signals.card_testing_details
            signals.append(f"small_tx_count:{details.get('small_tx_count', 0)}")

        return score, signals

    def _make_decision(self, risk_score: float) -> RiskDecision:
        """Convert risk score to decision."""
        if risk_score < self.thresholds.approve_below:
            return RiskDecision.APPROVE
        elif risk_score < self.thresholds.review_below:
            return RiskDecision.REVIEW
        else:
            return RiskDecision.DECLINE
