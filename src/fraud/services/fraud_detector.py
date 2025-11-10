"""
Main fraud detection service integrating all components.
Fast, production-ready fraud detection with <50ms latency.
"""

import time
import logging
from typing import Dict

from ..models import (
    TransactionRequest,
    FraudDecision,
    RiskDecision,
    KafkaFraudEvent,
    SignalWeights,
    RiskThresholds
)
from .signal_collector import SignalCollector
from .risk_scorer import RiskScorer
from ...infrastructure.kafka.producer import FraudEventProducer

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Main fraud detection service.

    Orchestrates:
    1. Signal collection (velocity + patterns)
    2. Risk scoring
    3. Decision making
    4. Kafka logging for ML training

    Target: <50ms total latency
    """

    def __init__(
        self,
        signal_collector: SignalCollector,
        weights: SignalWeights,
        thresholds: RiskThresholds,
        velocity_limits: Dict[str, float],
        kafka_producer: FraudEventProducer
    ):
        self.signal_collector = signal_collector
        self.kafka_producer = kafka_producer

        # Initialize risk scorer
        self.risk_scorer = RiskScorer(
            weights=weights,
            thresholds=thresholds,
            velocity_limits=velocity_limits
        )

    async def assess_transaction(
        self,
        transaction: TransactionRequest
    ) -> FraudDecision:
        """
        Assess transaction for fraud risk BEFORE payment processing.

        Flow:
        1. Collect all signals (velocity + patterns)
        2. Calculate risk score
        3. Make decision
        4. Log to Kafka (async)
        5. Return decision

        Returns:
            FraudDecision with decision, risk_score, and signals
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Collect all signals
            velocity_signals, pattern_signals = await self.signal_collector.collect_all_signals(
                transaction
            )

            # Step 2: Calculate risk score
            risk_score = self.risk_scorer.calculate_risk_score(
                transaction=transaction,
                velocity_signals=velocity_signals,
                pattern_signals=pattern_signals
            )

            # Step 3: Make decision
            decision = self._make_fraud_decision(risk_score)

            total_latency = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"Transaction {transaction.transaction_id} assessed: "
                f"decision={decision.decision.value}, "
                f"risk_score={decision.risk_score:.3f}, "
                f"latency={total_latency:.2f}ms"
            )

            # Step 4: Log to Kafka asynchronously (don't wait)
            self._log_to_kafka(transaction, risk_score, decision)

            # Step 5: Track transaction for future velocity calculations
            # Only track if not declined (to avoid polluting velocity data)
            if decision.decision != RiskDecision.DECLINE:
                await self.signal_collector.track_transaction(transaction)

            return decision

        except Exception as e:
            logger.error(f"Fraud detection failed for {transaction.transaction_id}: {e}", exc_info=True)
            # Fail open - allow transaction but flag for review
            return FraudDecision(
                transaction_id=transaction.transaction_id,
                decision=RiskDecision.REVIEW,
                risk_score=0.5,
                signals_triggered=["fraud_detection_error"],
                should_process_payment=True,
                requires_manual_review=True,
                decline_reason="Fraud detection system error",
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                raw_signals={"error": str(e)}
            )

    def _make_fraud_decision(self, risk_score) -> FraudDecision:
        """Convert risk score to fraud decision."""
        decision = risk_score.decision

        should_process_payment = decision != RiskDecision.DECLINE
        requires_manual_review = decision == RiskDecision.REVIEW

        decline_reason = None
        if decision == RiskDecision.DECLINE:
            # Pick most critical signal
            if "card_testing_pattern_detected" in risk_score.signals_triggered:
                decline_reason = "Card testing pattern detected"
            elif any("amount" in s for s in risk_score.signals_triggered):
                decline_reason = "Transaction amount velocity exceeded"
            elif any("velocity" in s for s in risk_score.signals_triggered):
                decline_reason = "Transaction count velocity exceeded"
            else:
                decline_reason = "High fraud risk detected"

        return FraudDecision(
            transaction_id="",  # Will be set by caller
            decision=decision,
            risk_score=risk_score.risk_score,
            signals_triggered=risk_score.signals_triggered,
            should_process_payment=should_process_payment,
            requires_manual_review=requires_manual_review,
            decline_reason=decline_reason,
            processing_time_ms=risk_score.processing_time_ms,
            raw_signals={
                "velocity": risk_score.velocity_signals.dict(),
                "patterns": risk_score.pattern_signals.dict(),
                "signal_scores": risk_score.signal_scores
            },
            thresholds_used={
                "approve_below": self.risk_scorer.thresholds.approve_below,
                "review_below": self.risk_scorer.thresholds.review_below,
                "decline_above": self.risk_scorer.thresholds.decline_above
            }
        )

    def _log_to_kafka(
        self,
        transaction: TransactionRequest,
        risk_score,
        decision: FraudDecision
    ):
        """Log fraud signal to Kafka for ML training (async, non-blocking)."""
        try:
            event = KafkaFraudEvent(
                event_type="fraud_signal_collection",
                transaction_id=transaction.transaction_id,
                user_id=transaction.user_id,
                card_id=transaction.card_id,
                ip_address=transaction.ip_address,
                amount=transaction.amount,
                merchant_category=transaction.merchant_category,
                risk_score=risk_score.risk_score,
                decision=decision.decision.value,
                signals_triggered=risk_score.signals_triggered,
                velocity_signals=risk_score.velocity_signals.dict(),
                pattern_signals=risk_score.pattern_signals.dict(),
                signal_scores=risk_score.signal_scores,
                timestamp=transaction.timestamp,
                processing_time_ms=decision.processing_time_ms
            )

            # Publish to Kafka (async, non-blocking)
            self.kafka_producer.publish_fraud_signal(
                transaction_id=transaction.transaction_id,
                event_data=event.dict()
            )

            # Also publish decision separately
            self.kafka_producer.publish_payment_decision(
                transaction_id=transaction.transaction_id,
                decision_data=decision.dict()
            )

        except Exception as e:
            logger.error(f"Failed to log to Kafka: {e}")
            # Don't fail the transaction due to logging errors


async def create_fraud_detector(
    redis_client,
    kafka_config: Dict,
    fraud_config: Dict
) -> FraudDetector:
    """Factory function to create configured FraudDetector."""
    from ...infrastructure.redis.velocity_tracker import VelocityTracker

    # Create velocity tracker
    velocity_tracker = VelocityTracker(redis_client)

    # Create signal collector
    signal_collector = SignalCollector(
        velocity_tracker=velocity_tracker,
        small_tx_threshold=fraud_config["patterns"]["small_transaction_threshold"],
        new_card_high_value=fraud_config["patterns"]["new_card_high_value"],
        merchant_switch_threshold=fraud_config["patterns"]["merchant_category_switch_threshold"]
    )

    # Create Kafka producer
    kafka_producer = FraudEventProducer(
        bootstrap_servers=kafka_config["bootstrap_servers"],
        topics=kafka_config["topics"],
        compression_type=kafka_config["producer"]["compression_type"]
    )

    # Create weights and thresholds
    weights = SignalWeights(**fraud_config["weights"])
    thresholds = RiskThresholds(**fraud_config["thresholds"])

    # Create fraud detector
    return FraudDetector(
        signal_collector=signal_collector,
        weights=weights,
        thresholds=thresholds,
        velocity_limits=fraud_config["velocity"],
        kafka_producer=kafka_producer
    )
