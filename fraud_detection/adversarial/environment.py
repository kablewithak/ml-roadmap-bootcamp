"""Reinforcement learning environment for fraud attacks."""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from fraud_detection.types import Transaction, AttackType
from fraud_detection.defense.system import DefenseSystem


class FraudAttackEnv(gym.Env):
    """
    OpenAI Gym environment for adversarial fraud attacks.

    The agent learns to execute fraud attacks that evade detection.
    State: Current attack parameters and defense response
    Action: Adjust attack parameters (amount, velocity, etc.)
    Reward: Successful fraud - detection penalty
    """

    def __init__(
        self,
        defense_system: DefenseSystem,
        max_steps: int = 100
    ):
        super(FraudAttackEnv, self).__init__()

        self.defense_system = defense_system
        self.max_steps = max_steps

        # Define action space
        # Actions: [amount_multiplier, delay_multiplier, device_rotation, ip_rotation]
        self.action_space = spaces.Box(
            low=np.array([0.5, 0.5, 0.0, 0.0]),
            high=np.array([2.0, 3.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Define observation space
        # State: [current_success_rate, avg_detection_score, recent_blocks, step_count]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 100.0, float(max_steps)]),
            dtype=np.float32
        )

        # State tracking
        self.step_count = 0
        self.success_count = 0
        self.block_count = 0
        self.total_reward = 0.0
        self.recent_blocks = 0

        # Attack parameters
        self.base_amount = 100.0
        self.base_delay = 1.0

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.step_count = 0
        self.success_count = 0
        self.block_count = 0
        self.total_reward = 0.0
        self.recent_blocks = 0

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Attack parameters adjustment

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.step_count += 1

        # Parse action
        amount_multiplier = action[0]
        delay_multiplier = action[1]
        device_rotation = action[2] > 0.5
        ip_rotation = action[3] > 0.5

        # Generate transaction with current parameters
        from fraud_detection.utils.data_generation import (
            generate_user_id,
            generate_device_id,
            generate_card_bin,
            generate_card_last4,
            fake
        )
        from datetime import datetime

        txn = Transaction(
            timestamp=datetime.utcnow(),
            user_id=generate_user_id(),
            merchant_id=f"merchant_{np.random.randint(1, 50)}",
            amount=self.base_amount * amount_multiplier,
            device_id=generate_device_id() if device_rotation else "device_attacker",
            ip_address=fake.ipv4() if ip_rotation else "10.0.0.1",
            user_agent=fake.user_agent(),
            card_bin=generate_card_bin(),
            card_last4=generate_card_last4(),
            country="US",
            is_first_transaction=True,
            time_since_account_creation_hours=1.0,
            transactions_last_24h=self.step_count,
            is_fraud=True,
        )

        # Check if defense blocks it
        is_blocked, detection_score, _ = self.defense_system.predict(txn)

        # Calculate reward
        reward = self._calculate_reward(
            is_blocked,
            detection_score,
            amount_multiplier,
            delay_multiplier
        )

        # Update stats
        if is_blocked:
            self.block_count += 1
            self.recent_blocks += 1
        else:
            self.success_count += 1
            self.recent_blocks = 0

        self.total_reward += reward

        # Check if done
        done = (
            self.step_count >= self.max_steps or
            self.recent_blocks >= 10  # Stop if blocked 10 times in a row
        )

        # Get new observation
        observation = self._get_observation()

        # Info
        info = {
            "success_count": self.success_count,
            "block_count": self.block_count,
            "success_rate": self.success_count / self.step_count if self.step_count > 0 else 0,
            "detection_score": detection_score,
            "total_reward": self.total_reward,
        }

        return observation, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        success_rate = self.success_count / self.step_count if self.step_count > 0 else 0.5

        # Get average detection score from recent transactions
        # (simplified - in practice, track this)
        avg_detection_score = 0.5

        observation = np.array([
            success_rate,
            avg_detection_score,
            float(self.recent_blocks),
            float(self.step_count),
        ], dtype=np.float32)

        return observation

    def _calculate_reward(
        self,
        is_blocked: bool,
        detection_score: float,
        amount_multiplier: float,
        delay_multiplier: float
    ) -> float:
        """
        Calculate reward for the agent.

        Args:
            is_blocked: Whether transaction was blocked
            detection_score: Detection score from defense
            amount_multiplier: Amount adjustment
            delay_multiplier: Delay adjustment

        Returns:
            Reward value
        """
        if is_blocked:
            # Penalize blocking
            reward = -10.0
        else:
            # Reward successful fraud (proportional to amount)
            reward = 5.0 * amount_multiplier

            # Small penalty for high detection score (risky)
            reward -= detection_score * 2.0

            # Small penalty for excessive delays (inefficient)
            reward -= (delay_multiplier - 1.0) * 0.5

        return reward

    def render(self, mode='human'):
        """Render the environment (optional)."""
        print(f"Step: {self.step_count}, Success: {self.success_count}, "
              f"Blocked: {self.block_count}, Recent Blocks: {self.recent_blocks}")
