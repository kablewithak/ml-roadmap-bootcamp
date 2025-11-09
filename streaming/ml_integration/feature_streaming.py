"""
ML feature streaming integration.

Bridges streaming infrastructure with ML models for real-time fraud detection
and credit risk assessment. Computes streaming features and serves to models.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import redis
import json
from decimal import Decimal

logger = logging.getLogger(__name__)


class MLFeatureStreaming:
    """
    Computes and serves real-time features for ML models.

    This is where streaming meets ML - computing features from event streams
    for fraud detection, credit scoring, and other real-time ML use cases.

    Features computed:
    1. Velocity features (transaction counts over time windows)
    2. Amount aggregations (spending patterns)
    3. Behavioral patterns (unusual times, locations)
    4. Merchant risk scores

    Example:
        >>> feature_stream = MLFeatureStreaming()
        >>> features = await feature_stream.compute_streaming_features(payment_event)
        >>> fraud_score = ml_model.predict(features)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        feature_ttl_seconds: int = 86400  # 24 hours
    ):
        """
        Initialize ML feature streaming.

        Args:
            redis_url: Redis connection URL for feature store
            feature_ttl_seconds: Feature TTL in Redis
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.feature_ttl = feature_ttl_seconds

        logger.info("ML feature streaming initialized")

    async def compute_streaming_features(
        self,
        event: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute streaming features from an event.

        Args:
            event: Payment or user action event

        Returns:
            Dictionary of features for ML model
        """
        user_id = event.get('user_id')
        merchant_id = event.get('merchant_id')
        amount = event.get('amount')
        timestamp = event.get('timestamp')
        country_code = event.get('country_code')

        features = {}

        # 1. Velocity features (critical for fraud detection)
        features['tx_count_5min'] = await self._get_velocity(
            user_id, window_seconds=300
        )
        features['tx_count_1hr'] = await self._get_velocity(
            user_id, window_seconds=3600
        )
        features['tx_count_24hr'] = await self._get_velocity(
            user_id, window_seconds=86400
        )

        # 2. Amount aggregations
        features['tx_amount_1hr'] = await self._get_amount_sum(
            user_id, window_seconds=3600
        )
        features['tx_amount_24hr'] = await self._get_amount_sum(
            user_id, window_seconds=86400
        )
        features['avg_amount_30d'] = await self._get_avg_amount(
            user_id, days=30
        )

        # 3. Behavioral features
        features['unusual_time'] = self._is_unusual_time(timestamp, user_id)
        features['unusual_country'] = await self._is_unusual_country(
            user_id, country_code
        )
        features['new_merchant'] = await self._is_new_merchant(
            user_id, merchant_id
        )

        # 4. Merchant risk features
        features['merchant_risk_score'] = await self._get_merchant_risk_score(
            merchant_id
        )
        features['merchant_fraud_rate'] = await self._get_merchant_fraud_rate(
            merchant_id
        )

        # 5. Amount-based features
        if amount:
            amount_float = float(amount) if isinstance(amount, (Decimal, str)) else amount
            features['amount'] = amount_float
            features['amount_deviation'] = await self._compute_amount_deviation(
                user_id, amount_float
            )
            features['is_round_amount'] = float(amount_float % 10 == 0)

        # 6. Time-based features
        features['hour_of_day'] = datetime.fromtimestamp(timestamp / 1000).hour
        features['day_of_week'] = datetime.fromtimestamp(timestamp / 1000).weekday()
        features['is_weekend'] = float(features['day_of_week'] >= 5)

        # Update feature store
        await self._write_to_feature_store(user_id, features)

        return features

    async def _get_velocity(
        self,
        user_id: str,
        window_seconds: int
    ) -> float:
        """
        Get transaction velocity for user over time window.

        Args:
            user_id: User identifier
            window_seconds: Time window in seconds

        Returns:
            Transaction count
        """
        key = f"velocity:{window_seconds}:{user_id}"

        try:
            # Use Redis sorted set with timestamps
            now = datetime.utcnow().timestamp()
            cutoff = now - window_seconds

            # Remove old entries
            self.redis_client.zremrangebyscore(key, 0, cutoff)

            # Get count
            count = self.redis_client.zcard(key)

            # Add current transaction
            self.redis_client.zadd(key, {str(now): now})
            self.redis_client.expire(key, window_seconds)

            return float(count)

        except Exception as e:
            logger.error(f"Error computing velocity: {e}")
            return 0.0

    async def _get_amount_sum(
        self,
        user_id: str,
        window_seconds: int
    ) -> float:
        """
        Get sum of transaction amounts over time window.

        Args:
            user_id: User identifier
            window_seconds: Time window in seconds

        Returns:
            Sum of amounts
        """
        key = f"amount:{window_seconds}:{user_id}"

        try:
            amounts = self.redis_client.lrange(key, 0, -1)
            total = sum(float(a) for a in amounts)
            return total

        except Exception as e:
            logger.error(f"Error computing amount sum: {e}")
            return 0.0

    async def _get_avg_amount(
        self,
        user_id: str,
        days: int
    ) -> float:
        """
        Get average transaction amount over period.

        Args:
            user_id: User identifier
            days: Number of days

        Returns:
            Average amount
        """
        key = f"avg_amount:{days}d:{user_id}"

        try:
            avg = self.redis_client.get(key)
            return float(avg) if avg else 0.0

        except Exception as e:
            logger.error(f"Error getting average amount: {e}")
            return 0.0

    def _is_unusual_time(self, timestamp: int, user_id: str) -> float:
        """
        Check if transaction time is unusual for user.

        Args:
            timestamp: Transaction timestamp (ms)
            user_id: User identifier

        Returns:
            1.0 if unusual, 0.0 otherwise
        """
        dt = datetime.fromtimestamp(timestamp / 1000)
        hour = dt.hour

        # Unusual if between 2 AM and 6 AM
        if 2 <= hour < 6:
            return 1.0

        return 0.0

    async def _is_unusual_country(
        self,
        user_id: str,
        country_code: str
    ) -> float:
        """
        Check if country is unusual for user.

        Args:
            user_id: User identifier
            country_code: ISO country code

        Returns:
            1.0 if unusual, 0.0 otherwise
        """
        key = f"user_countries:{user_id}"

        try:
            countries = self.redis_client.smembers(key)

            if not countries:
                # First transaction, add country
                self.redis_client.sadd(key, country_code)
                self.redis_client.expire(key, 2592000)  # 30 days
                return 0.0

            if country_code not in countries:
                # New country
                self.redis_client.sadd(key, country_code)
                return 1.0

            return 0.0

        except Exception as e:
            logger.error(f"Error checking country: {e}")
            return 0.0

    async def _is_new_merchant(
        self,
        user_id: str,
        merchant_id: str
    ) -> float:
        """
        Check if merchant is new for user.

        Args:
            user_id: User identifier
            merchant_id: Merchant identifier

        Returns:
            1.0 if new, 0.0 otherwise
        """
        key = f"user_merchants:{user_id}"

        try:
            merchants = self.redis_client.smembers(key)

            if merchant_id not in merchants:
                self.redis_client.sadd(key, merchant_id)
                self.redis_client.expire(key, 2592000)  # 30 days
                return 1.0

            return 0.0

        except Exception as e:
            logger.error(f"Error checking merchant: {e}")
            return 0.0

    async def _get_merchant_risk_score(self, merchant_id: str) -> float:
        """
        Get risk score for merchant.

        Args:
            merchant_id: Merchant identifier

        Returns:
            Risk score (0.0-1.0)
        """
        key = f"merchant_risk:{merchant_id}"

        try:
            score = self.redis_client.get(key)
            return float(score) if score else 0.5  # Default medium risk

        except Exception as e:
            logger.error(f"Error getting merchant risk: {e}")
            return 0.5

    async def _get_merchant_fraud_rate(self, merchant_id: str) -> float:
        """
        Get fraud rate for merchant.

        Args:
            merchant_id: Merchant identifier

        Returns:
            Fraud rate (0.0-1.0)
        """
        key = f"merchant_fraud_rate:{merchant_id}"

        try:
            rate = self.redis_client.get(key)
            return float(rate) if rate else 0.0

        except Exception as e:
            logger.error(f"Error getting merchant fraud rate: {e}")
            return 0.0

    async def _compute_amount_deviation(
        self,
        user_id: str,
        amount: float
    ) -> float:
        """
        Compute deviation from user's average amount.

        Args:
            user_id: User identifier
            amount: Current amount

        Returns:
            Deviation ratio
        """
        avg_amount = await self._get_avg_amount(user_id, 30)

        if avg_amount == 0:
            return 0.0

        deviation = abs(amount - avg_amount) / avg_amount
        return deviation

    async def _write_to_feature_store(
        self,
        user_id: str,
        features: Dict[str, float]
    ):
        """
        Write features to feature store for model serving.

        Args:
            user_id: User identifier
            features: Feature dictionary
        """
        key = f"features:{user_id}"

        try:
            self.redis_client.setex(
                key,
                self.feature_ttl,
                json.dumps(features)
            )

        except Exception as e:
            logger.error(f"Error writing to feature store: {e}")

    async def get_features_for_user(self, user_id: str) -> Optional[Dict[str, float]]:
        """
        Retrieve features for a user from feature store.

        Args:
            user_id: User identifier

        Returns:
            Feature dictionary or None
        """
        key = f"features:{user_id}"

        try:
            features_json = self.redis_client.get(key)
            if features_json:
                return json.loads(features_json)
            return None

        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            return None

    def close(self):
        """Close Redis connection."""
        self.redis_client.close()
        logger.info("ML feature streaming closed")
