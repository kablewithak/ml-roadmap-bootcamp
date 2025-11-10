"""
Redis-based velocity tracking for fraud detection.
Provides fast (<10ms) lookups for real-time fraud signals.
"""

import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import redis.asyncio as aioredis
from redis.asyncio import Redis
import logging

logger = logging.getLogger(__name__)


class VelocityTracker:
    """
    High-performance Redis-based velocity tracking system.
    Tracks transaction counts and amounts across multiple time windows.
    """

    # Time windows in seconds
    WINDOW_5MIN = 300
    WINDOW_1HR = 3600
    WINDOW_24HR = 86400

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def track_transaction(
        self,
        card_id: str,
        user_id: str,
        ip_address: str,
        amount: float,
        merchant_category: str,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, int]:
        """
        Track a transaction and update all velocity counters.

        Returns current counts for all dimensions.
        Time complexity: O(1) per counter
        Target latency: < 5ms
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        ts = int(timestamp.timestamp())

        # Build pipeline for atomic operations
        pipe = self.redis.pipeline()

        # Track counts for each dimension
        dimensions = [
            f"card:{card_id}",
            f"user:{user_id}",
            f"ip:{ip_address}",
        ]

        for dimension in dimensions:
            # Count velocity (5min, 1hr windows)
            for window, ttl in [(self.WINDOW_5MIN, self.WINDOW_5MIN),
                                (self.WINDOW_1HR, self.WINDOW_1HR)]:
                key = f"{dimension}:count:{window}"
                pipe.incr(key)
                pipe.expire(key, ttl)

            # Amount velocity
            for window, ttl in [(self.WINDOW_5MIN, self.WINDOW_5MIN),
                                (self.WINDOW_1HR, self.WINDOW_1HR)]:
                key = f"{dimension}:amount:{window}"
                pipe.incrbyfloat(key, amount)
                pipe.expire(key, ttl)

        # Track merchant category for this user
        merchant_key = f"user:{user_id}:merchants:{self.WINDOW_1HR}"
        pipe.sadd(merchant_key, merchant_category)
        pipe.expire(merchant_key, self.WINDOW_1HR)

        # Track first-time card usage
        first_use_key = f"card:{card_id}:first_seen"
        pipe.set(first_use_key, ts, ex=self.WINDOW_24HR, nx=True)

        # Track hour of day pattern
        hour = timestamp.hour
        hour_key = f"user:{user_id}:hours:{self.WINDOW_24HR}"
        pipe.hincrby(hour_key, str(hour), 1)
        pipe.expire(hour_key, self.WINDOW_24HR)

        start_time = time.perf_counter()
        await pipe.execute()
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(f"Velocity tracking latency: {latency_ms:.2f}ms")

        return {"latency_ms": latency_ms}

    async def get_velocity_signals(
        self,
        card_id: str,
        user_id: str,
        ip_address: str,
        amount: float,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Get all velocity signals for fraud detection.
        Fast parallel lookup using Redis pipeline.

        Target latency: < 10ms
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        start_time = time.perf_counter()

        pipe = self.redis.pipeline()

        # Build lookup keys
        dimensions = {
            "card": f"card:{card_id}",
            "user": f"user:{user_id}",
            "ip": f"ip:{ip_address}",
        }

        # Queue all lookups
        for dim_name, dimension in dimensions.items():
            # Count velocity
            pipe.get(f"{dimension}:count:{self.WINDOW_5MIN}")
            pipe.get(f"{dimension}:count:{self.WINDOW_1HR}")

            # Amount velocity
            pipe.get(f"{dimension}:amount:{self.WINDOW_5MIN}")
            pipe.get(f"{dimension}:amount:{self.WINDOW_1HR}")

        # Merchant categories
        pipe.smembers(f"user:{user_id}:merchants:{self.WINDOW_1HR}")

        # First-time card check
        pipe.get(f"card:{card_id}:first_seen")

        # Hour patterns
        pipe.hgetall(f"user:{user_id}:hours:{self.WINDOW_24HR}")

        # Execute all lookups in parallel
        results = await pipe.execute()

        # Parse results
        idx = 0
        signals = {}

        for dim_name in ["card", "user", "ip"]:
            signals[f"{dim_name}_count_5min"] = int(results[idx] or 0)
            signals[f"{dim_name}_count_1hr"] = int(results[idx + 1] or 0)
            signals[f"{dim_name}_amount_5min"] = float(results[idx + 2] or 0.0)
            signals[f"{dim_name}_amount_1hr"] = float(results[idx + 3] or 0.0)
            idx += 4

        # Merchant categories
        merchant_categories = results[idx] or set()
        signals["merchant_category_count"] = len(merchant_categories)
        idx += 1

        # First-time card
        first_seen = results[idx]
        signals["is_first_card_use"] = first_seen is not None and \
                                       int(first_seen) >= (timestamp.timestamp() - 60)
        idx += 1

        # Hour patterns
        hour_patterns = results[idx] or {}
        signals["hour_patterns"] = {int(k): int(v) for k, v in hour_patterns.items()} if hour_patterns else {}
        current_hour = timestamp.hour
        signals["current_hour_tx_count"] = int(hour_patterns.get(str(current_hour).encode(), 0)) if hour_patterns else 0

        latency_ms = (time.perf_counter() - start_time) * 1000
        signals["lookup_latency_ms"] = latency_ms

        logger.debug(f"Velocity signal lookup latency: {latency_ms:.2f}ms")

        return signals

    async def check_card_testing_pattern(
        self,
        card_id: str,
        amount: float,
        small_tx_threshold: float = 10.0
    ) -> Tuple[bool, Dict]:
        """
        Detect card testing patterns (multiple small charges).

        Card testing: fraudsters test stolen cards with small charges
        before making large purchases.
        """
        pipe = self.redis.pipeline()

        # Get recent transaction count
        pipe.get(f"card:{card_id}:count:{self.WINDOW_5MIN}")
        pipe.get(f"card:{card_id}:amount:{self.WINDOW_5MIN}")

        # Track small transactions specifically
        small_tx_key = f"card:{card_id}:small_tx:{self.WINDOW_5MIN}"
        if amount < small_tx_threshold:
            pipe.incr(small_tx_key)
            pipe.expire(small_tx_key, self.WINDOW_5MIN)

        pipe.get(small_tx_key)

        results = await pipe.execute()

        tx_count = int(results[0] or 0)
        total_amount = float(results[1] or 0.0)
        small_tx_count = int(results[-1] or 0)

        # Pattern detection: 3+ transactions, mostly small amounts
        is_card_testing = (
            tx_count >= 3 and
            small_tx_count >= 2 and
            total_amount < small_tx_threshold * tx_count * 1.5
        )

        return is_card_testing, {
            "tx_count_5min": tx_count,
            "small_tx_count": small_tx_count,
            "total_amount_5min": total_amount,
            "is_card_testing": is_card_testing
        }

    async def get_health_metrics(self) -> Dict:
        """Get Redis health and performance metrics."""
        try:
            start = time.perf_counter()
            await self.redis.ping()
            ping_ms = (time.perf_counter() - start) * 1000

            info = await self.redis.info()

            return {
                "status": "healthy",
                "ping_ms": ping_ms,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                "ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


async def create_redis_client(host: str = "localhost", port: int = 6379, db: int = 0) -> Redis:
    """Create and configure Redis client with optimal settings."""
    return await aioredis.from_url(
        f"redis://{host}:{port}/{db}",
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
        socket_timeout=5,
        socket_connect_timeout=5,
    )
