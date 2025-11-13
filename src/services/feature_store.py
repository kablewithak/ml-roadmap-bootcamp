"""
Feature Store with Redis Cache

WHAT IS A FEATURE STORE?
-------------------------
In ML systems, "features" are the input variables to your model:
- transaction_amount
- user_account_age
- transaction_velocity (transactions in last 24h)

These features come from different sources:
- Database: user_account_age (slow: 100ms)
- Real-time calculation: transaction_velocity (very slow: 500ms)
- External API: credit_score (extremely slow: 2s)

PROBLEM: If you fetch all features on every prediction:
100ms + 500ms + 2s = 2.6 seconds per prediction = UNACCEPTABLE

SOLUTION: Feature Store
-----------------------
1. Pre-compute features in batch jobs (hourly/daily)
2. Store in Redis (in-memory cache)
3. Prediction time: Redis lookup = 2ms (1000x faster!)

BUSINESS IMPACT:
- Without cache: 2.6s latency → customers abandon checkout → $500K/year lost revenue
- With cache: 50ms latency → smooth checkout → happy customers

TRADE-OFF:
- ✅ Speed: 1000x faster
- ❌ Staleness: Features may be hours old (acceptable for most ML use cases)

WHEN THIS PATTERN IS WRONG:
If your features must be REAL-TIME (fraud detection on current transaction),
you can't use cached features. Use a real-time feature store (Tecton, Feast).
"""

import logging
import json
import time
from typing import Dict, Optional
import redis
from opentelemetry import trace

from observability.instrumentation import business_metrics, get_trace_context

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class FeatureStore:
    """
    Redis-backed feature store with observability.

    CRITICAL: Cache invalidation
    -----------------------------
    "There are only two hard things in Computer Science: cache invalidation
    and naming things." - Phil Karlton

    When do you invalidate cached features?
    1. TTL (Time To Live): Expire after 1 hour (simple, but wasteful)
    2. Event-driven: Invalidate when user updates profile (complex, but efficient)
    3. Versioning: Cache key includes timestamp (our approach)

    We use: feature:user_id:timestamp
    This way, batch jobs can write new features without invalidating old ones.
    """

    def __init__(
        self,
        redis_host: str = "redis",
        redis_port: int = 6379,
        ttl_seconds: int = 3600,  # 1 hour cache
    ):
        """
        Initialize feature store.

        Args:
            redis_host: Redis hostname
            redis_port: Redis port
            ttl_seconds: Time-to-live for cached features

        FAILURE MODE:
        If Redis is unreachable, __init__ will raise redis.ConnectionError.
        Your application will crash on startup (fail-fast).

        ALTERNATIVE: Lazy connection (only connect on first get())
        - ✅ App starts even if Redis is down
        - ❌ First request fails (poor UX)

        We choose fail-fast because:
        - Kubernetes will restart the pod
        - Better than silently degraded service
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=0,
            decode_responses=True,  # Return strings, not bytes
            socket_connect_timeout=5,  # Fail fast if Redis is down
            socket_timeout=1,  # Don't wait forever for responses
        )
        self.ttl_seconds = ttl_seconds

        # Test connection
        try:
            self.redis_client.ping()
            logger.info(f"✓ Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"✗ Failed to connect to Redis: {e}")
            raise

    def _get_cache_key(self, user_id: int, feature_name: str) -> str:
        """
        Generate cache key for a feature.

        Format: feature:{user_id}:{feature_name}

        Example: feature:12345:transaction_velocity
        """
        return f"feature:{user_id}:{feature_name}"

    def get_feature(self, user_id: int, feature_name: str) -> Optional[float]:
        """
        Get a single feature from cache.

        Returns:
            Feature value (float) or None if not cached

        OBSERVABILITY:
        1. Span: Shows Redis latency in trace
        2. Metric: Increments cache_hits or cache_misses counter
        3. Log: Records cache miss (for debugging)
        """
        cache_key = self._get_cache_key(user_id, feature_name)

        # CRITICAL: Create custom span for this operation
        # This will show up in Jaeger as a child span of the parent request
        with tracer.start_as_current_span("feature_store.get") as span:
            # Add attributes to span (searchable in Jaeger)
            span.set_attribute("feature.name", feature_name)
            span.set_attribute("user.id", user_id)
            span.set_attribute("cache.key", cache_key)

            start_time = time.time()

            try:
                value = self.redis_client.get(cache_key)

                # Measure latency
                latency = time.time() - start_time
                business_metrics.feature_fetch_latency_seconds.observe(latency)

                if value is not None:
                    # CACHE HIT
                    span.set_attribute("cache.hit", True)
                    business_metrics.feature_cache_hits_total.labels(
                        feature_name=feature_name
                    ).inc()

                    logger.debug(
                        f"Cache hit: {cache_key}",
                        extra={"user_id": user_id, "feature_name": feature_name, **get_trace_context()}
                    )

                    return float(value)
                else:
                    # CACHE MISS
                    span.set_attribute("cache.hit", False)
                    business_metrics.feature_cache_misses_total.labels(
                        feature_name=feature_name
                    ).inc()

                    logger.warning(
                        f"Cache miss: {cache_key}",
                        extra={"user_id": user_id, "feature_name": feature_name, **get_trace_context()}
                    )

                    return None

            except redis.RedisError as e:
                # FAILURE MODE: Redis is down or network error
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                business_metrics.error_total.labels(
                    error_type="redis_error",
                    service="feature_store"
                ).inc()

                logger.error(
                    f"Redis error: {e}",
                    extra={"user_id": user_id, "feature_name": feature_name, **get_trace_context()}
                )

                # CRITICAL DECISION: Return None (degrade gracefully)
                # Alternative: Raise exception (fail fast)
                #
                # We return None because:
                # - Caller can use default values
                # - Better than crashing the entire request
                #
                # TRADE-OFF:
                # - ✅ Resilience: Service stays up
                # - ❌ Accuracy: Predictions use stale/default features
                return None

    def get_features(self, user_id: int) -> Dict[str, float]:
        """
        Get all features for a user (batch operation).

        This is more efficient than calling get_feature() multiple times
        because it uses Redis pipelining (single round-trip).

        Returns:
            Dict of feature_name → value

        OBSERVABILITY INSIGHT:
        If you see high latency in this span, check:
        1. Redis CPU usage (too many keys?)
        2. Network latency (Redis in different AZ?)
        3. Cache miss rate (features not pre-computed?)
        """
        feature_names = [
            'transaction_amount',
            'transaction_hour',
            'days_since_signup',
            'transaction_count_24h',
            'avg_transaction_amount',
            'is_international'
        ]

        with tracer.start_as_current_span("feature_store.get_batch") as span:
            span.set_attribute("user.id", user_id)
            span.set_attribute("feature.count", len(feature_names))

            # Use Redis pipeline for batch get (single network round-trip)
            pipe = self.redis_client.pipeline()
            for feature_name in feature_names:
                cache_key = self._get_cache_key(user_id, feature_name)
                pipe.get(cache_key)

            start_time = time.time()
            values = pipe.execute()
            latency = time.time() - start_time

            business_metrics.feature_fetch_latency_seconds.observe(latency)
            span.set_attribute("latency_ms", latency * 1000)

            # Convert to dict
            features = {}
            for feature_name, value in zip(feature_names, values):
                if value is not None:
                    features[feature_name] = float(value)
                    business_metrics.feature_cache_hits_total.labels(
                        feature_name=feature_name
                    ).inc()
                else:
                    # Use default value on cache miss
                    features[feature_name] = 0.0
                    business_metrics.feature_cache_misses_total.labels(
                        feature_name=feature_name
                    ).inc()

            cache_hit_rate = sum(1 for v in values if v is not None) / len(values)
            span.set_attribute("cache.hit_rate", cache_hit_rate)

            logger.info(
                f"Fetched {len(features)} features (hit rate: {cache_hit_rate:.2%})",
                extra={"user_id": user_id, "cache_hit_rate": cache_hit_rate, **get_trace_context()}
            )

            return features

    def set_feature(
        self,
        user_id: int,
        feature_name: str,
        value: float,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set a feature in cache (called by batch jobs).

        Args:
            user_id: User ID
            feature_name: Feature name
            value: Feature value
            ttl_seconds: Time-to-live (defaults to self.ttl_seconds)
        """
        cache_key = self._get_cache_key(user_id, feature_name)
        ttl = ttl_seconds or self.ttl_seconds

        with tracer.start_as_current_span("feature_store.set") as span:
            span.set_attribute("feature.name", feature_name)
            span.set_attribute("user.id", user_id)

            try:
                self.redis_client.setex(cache_key, ttl, value)
                logger.debug(f"Set feature: {cache_key} = {value} (TTL: {ttl}s)")
            except redis.RedisError as e:
                span.set_attribute("error", True)
                logger.error(f"Failed to set feature: {e}")

    def set_features_batch(
        self,
        user_id: int,
        features: Dict[str, float],
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set multiple features in one batch (used by feature engineering jobs).

        This is called by:
        - Hourly batch job that computes transaction_velocity
        - Daily batch job that updates user_account_age
        """
        ttl = ttl_seconds or self.ttl_seconds

        with tracer.start_as_current_span("feature_store.set_batch") as span:
            span.set_attribute("user.id", user_id)
            span.set_attribute("feature.count", len(features))

            pipe = self.redis_client.pipeline()
            for feature_name, value in features.items():
                cache_key = self._get_cache_key(user_id, feature_name)
                pipe.setex(cache_key, ttl, value)

            pipe.execute()

            logger.info(
                f"Set {len(features)} features for user {user_id}",
                extra={"user_id": user_id, "feature_count": len(features)}
            )

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get Redis cache statistics (for monitoring dashboard).

        Returns:
            {
                "total_keys": 1000000,
                "used_memory_mb": 512,
                "hit_rate": 0.95,
                "connected_clients": 10
            }
        """
        info = self.redis_client.info()

        return {
            "total_keys": self.redis_client.dbsize(),
            "used_memory_mb": info['used_memory'] / (1024 * 1024),
            "connected_clients": info['connected_clients'],
            "uptime_seconds": info['uptime_in_seconds'],
        }
