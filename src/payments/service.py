"""
Payment processing service integrated with fraud detection.
Checks fraud signals BEFORE Stripe charge.
"""

import logging
import stripe
from typing import Dict, Optional
from datetime import datetime

from ..fraud.models import TransactionRequest, RiskDecision
from ..fraud.services.fraud_detector import FraudDetector

logger = logging.getLogger(__name__)


class PaymentService:
    """
    Payment processing service with fraud detection integration.

    Flow:
    1. Validate payment request
    2. Check fraud signals (BEFORE Stripe)
    3. If approved: process Stripe charge
    4. If declined: reject payment
    5. If review: flag for manual review
    """

    def __init__(
        self,
        fraud_detector: FraudDetector,
        stripe_api_key: str
    ):
        self.fraud_detector = fraud_detector
        stripe.api_key = stripe_api_key

    async def process_payment(
        self,
        transaction_id: str,
        user_id: str,
        card_id: str,
        card_token: str,  # Stripe token
        ip_address: str,
        amount: float,
        currency: str,
        merchant_id: str,
        merchant_category: str,
        merchant_name: str,
        description: Optional[str] = None
    ) -> Dict:
        """
        Process payment with fraud detection.

        Returns:
            Dict with payment result, including fraud decision
        """
        start_time = datetime.utcnow()

        # Step 1: Create transaction request for fraud check
        transaction = TransactionRequest(
            transaction_id=transaction_id,
            user_id=user_id,
            card_id=card_id,
            ip_address=ip_address,
            amount=amount,
            currency=currency,
            merchant_id=merchant_id,
            merchant_category=merchant_category,
            merchant_name=merchant_name,
            timestamp=start_time
        )

        # Step 2: Fraud detection check
        fraud_decision = await self.fraud_detector.assess_transaction(transaction)

        # Step 3: Make payment decision
        if fraud_decision.decision == RiskDecision.DECLINE:
            # DECLINE: Do not process payment
            logger.warning(
                f"Payment declined due to fraud: {transaction_id}, "
                f"risk_score={fraud_decision.risk_score:.3f}, "
                f"reason={fraud_decision.decline_reason}"
            )
            return {
                "status": "declined",
                "transaction_id": transaction_id,
                "decline_reason": fraud_decision.decline_reason,
                "fraud_decision": fraud_decision.dict(),
                "stripe_charge_id": None
            }

        elif fraud_decision.decision == RiskDecision.REVIEW:
            # REVIEW: Flag for manual review but process payment
            logger.info(
                f"Payment flagged for review: {transaction_id}, "
                f"risk_score={fraud_decision.risk_score:.3f}"
            )
            # Could implement hold funds / manual approval flow here
            # For now, we'll process but flag

        # Step 4: Process Stripe payment
        try:
            charge = stripe.Charge.create(
                amount=int(amount * 100),  # Stripe uses cents
                currency=currency.lower(),
                source=card_token,
                description=description or f"Payment for {merchant_name}",
                metadata={
                    "transaction_id": transaction_id,
                    "user_id": user_id,
                    "merchant_id": merchant_id,
                    "fraud_risk_score": fraud_decision.risk_score,
                    "fraud_decision": fraud_decision.decision.value
                }
            )

            logger.info(
                f"Payment processed: {transaction_id}, "
                f"stripe_charge={charge.id}, "
                f"amount={amount}, "
                f"fraud_score={fraud_decision.risk_score:.3f}"
            )

            return {
                "status": "approved" if fraud_decision.decision == RiskDecision.APPROVE else "approved_review_flagged",
                "transaction_id": transaction_id,
                "stripe_charge_id": charge.id,
                "amount": amount,
                "currency": currency,
                "fraud_decision": fraud_decision.dict(),
                "requires_review": fraud_decision.requires_manual_review
            }

        except stripe.error.CardError as e:
            # Card declined by Stripe
            logger.warning(f"Stripe card error: {transaction_id}, error={e.user_message}")
            return {
                "status": "card_declined",
                "transaction_id": transaction_id,
                "decline_reason": e.user_message,
                "fraud_decision": fraud_decision.dict(),
                "stripe_charge_id": None
            }

        except stripe.error.StripeError as e:
            # Other Stripe errors
            logger.error(f"Stripe error: {transaction_id}, error={str(e)}")
            return {
                "status": "error",
                "transaction_id": transaction_id,
                "error": str(e),
                "fraud_decision": fraud_decision.dict(),
                "stripe_charge_id": None
            }

    async def refund_payment(
        self,
        stripe_charge_id: str,
        amount: Optional[float] = None,
        reason: str = "requested_by_customer"
    ) -> Dict:
        """Refund a payment."""
        try:
            refund_params = {
                "charge": stripe_charge_id,
                "reason": reason
            }

            if amount is not None:
                refund_params["amount"] = int(amount * 100)

            refund = stripe.Refund.create(**refund_params)

            logger.info(f"Refund processed: charge={stripe_charge_id}, refund={refund.id}")

            return {
                "status": "refunded",
                "stripe_charge_id": stripe_charge_id,
                "refund_id": refund.id,
                "amount": refund.amount / 100,
                "reason": reason
            }

        except stripe.error.StripeError as e:
            logger.error(f"Refund failed: {stripe_charge_id}, error={str(e)}")
            return {
                "status": "error",
                "stripe_charge_id": stripe_charge_id,
                "error": str(e)
            }
