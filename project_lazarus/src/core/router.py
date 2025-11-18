"""State-Aware Traffic Router: The decision brain using Redis and Lua scripts.

This module implements the epsilon-greedy exploration strategy with
atomic budget management using Redis Lua scripts.
"""

import random
import structlog
from typing import Any

import redis

from .models import TreatmentGroup
from config.settings import get_settings

logger = structlog.get_logger(__name__)


# Lua script for atomic budget decrement and exploration check
EXPLORATION_LUA_SCRIPT = """
-- Atomic exploration decision with budget check
-- KEYS[1] = loss_budget key
-- ARGV[1] = exploration cost (estimated loss per explore)
-- Returns: {can_explore: 0/1, remaining_budget: float}

local budget_key = KEYS[1]
local exploration_cost = tonumber(ARGV[1])

-- Get current budget
local current_budget = tonumber(redis.call('GET', budget_key) or 0)

if current_budget <= 0 then
    return {0, 0}
end

-- Decrement budget atomically
local new_budget = current_budget - exploration_cost
redis.call('SET', budget_key, new_budget)

-- Log the exploration
redis.call('INCR', 'lazarus:explore_count')

return {1, new_budget}
"""


class TrafficRouter:
    """
    The State-Aware Traffic Router (The "Brain").

    Decides whether to exploit (use model prediction) or explore
    (force approval to learn from outcome).

    Uses Redis with Lua scripts for atomic budget management.
    """

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        epsilon: float | None = None,
        exploration_cost: float = 50.0  # Estimated cost per exploration
    ):
        """
        Initialize the traffic router.

        Args:
            redis_client: Redis client instance
            epsilon: Exploration probability (default from settings)
            exploration_cost: Estimated cost per exploration decision
        """
        settings = get_settings()

        if redis_client is None:
            self.redis = redis.from_url(settings.redis_url)
        else:
            self.redis = redis_client

        self.epsilon = epsilon if epsilon is not None else settings.epsilon
        self.exploration_cost = exploration_cost

        # Register the Lua script
        self._exploration_script = self.redis.register_script(EXPLORATION_LUA_SCRIPT)

        logger.info(
            "traffic_router_initialized",
            epsilon=self.epsilon,
            exploration_cost=self.exploration_cost
        )

    def initialize_budget(self, budget: float | None = None) -> None:
        """
        Initialize or reset the exploration budget.

        Args:
            budget: Budget amount (default from settings)
        """
        if budget is None:
            budget = get_settings().initial_loss_budget

        self.redis.set("lazarus:loss_budget", budget)
        self.redis.set("lazarus:explore_count", 0)
        self.redis.set("lazarus:exploit_count", 0)

        logger.info("budget_initialized", budget=budget)

    def get_budget_status(self) -> dict[str, Any]:
        """
        Get current budget and exploration statistics.

        Returns:
            Dictionary with budget status
        """
        budget = float(self.redis.get("lazarus:loss_budget") or 0)
        explore_count = int(self.redis.get("lazarus:explore_count") or 0)
        exploit_count = int(self.redis.get("lazarus:exploit_count") or 0)

        return {
            "remaining_budget": budget,
            "explore_count": explore_count,
            "exploit_count": exploit_count,
            "exploration_rate": explore_count / max(1, explore_count + exploit_count)
        }

    def should_explore(self, hard_rules_pass: bool) -> tuple[bool, TreatmentGroup, float]:
        """
        Decide whether to explore or exploit.

        This implements the epsilon-greedy strategy with budget constraints.

        Args:
            hard_rules_pass: Whether the safety valve rules passed

        Returns:
            Tuple of (should_explore, treatment_group, remaining_budget)
        """
        # If hard rules failed, never explore
        if not hard_rules_pass:
            self.redis.incr("lazarus:exploit_count")
            budget = float(self.redis.get("lazarus:loss_budget") or 0)
            return False, TreatmentGroup.EXPLOIT, budget

        # Check if we should explore (epsilon probability)
        r = random.random()

        if r < self.epsilon:
            # Try to explore - use Lua script for atomic budget check
            result = self._exploration_script(
                keys=["lazarus:loss_budget"],
                args=[self.exploration_cost]
            )

            can_explore = bool(result[0])
            remaining_budget = float(result[1])

            if can_explore:
                logger.info(
                    "exploration_decision",
                    treatment="explore",
                    remaining_budget=remaining_budget
                )
                return True, TreatmentGroup.EXPLORE, remaining_budget
            else:
                # Budget exhausted, fall back to exploit
                logger.warning(
                    "budget_exhausted",
                    attempted_explore=True
                )
                self.redis.incr("lazarus:exploit_count")
                return False, TreatmentGroup.EXPLOIT, 0.0

        # Exploit mode
        self.redis.incr("lazarus:exploit_count")
        budget = float(self.redis.get("lazarus:loss_budget") or 0)

        return False, TreatmentGroup.EXPLOIT, budget

    def record_exploration_outcome(
        self,
        application_id: str,
        defaulted: bool,
        actual_loss: float
    ) -> None:
        """
        Record the outcome of an exploration decision.

        This updates the budget based on actual outcomes.

        Args:
            application_id: The application ID
            defaulted: Whether the loan defaulted
            actual_loss: Actual loss amount (if defaulted)
        """
        if defaulted:
            # Record the actual loss
            self.redis.incr("lazarus:total_exploration_loss", actual_loss)
            self.redis.incr("lazarus:exploration_defaults")

            logger.info(
                "exploration_default",
                application_id=application_id,
                actual_loss=actual_loss
            )
        else:
            # Record successful exploration
            self.redis.incr("lazarus:exploration_successes")

            logger.info(
                "exploration_success",
                application_id=application_id
            )

    def get_exploration_metrics(self) -> dict[str, Any]:
        """
        Get detailed exploration metrics.

        Returns:
            Dictionary with exploration metrics
        """
        status = self.get_budget_status()

        total_loss = float(self.redis.get("lazarus:total_exploration_loss") or 0)
        defaults = int(self.redis.get("lazarus:exploration_defaults") or 0)
        successes = int(self.redis.get("lazarus:exploration_successes") or 0)

        return {
            **status,
            "total_exploration_loss": total_loss,
            "exploration_defaults": defaults,
            "exploration_successes": successes,
            "default_rate": defaults / max(1, defaults + successes)
        }
