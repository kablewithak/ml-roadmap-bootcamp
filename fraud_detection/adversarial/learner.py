"""Adversarial learner using reinforcement learning."""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fraud_detection.adversarial.environment import FraudAttackEnv
from fraud_detection.defense.system import DefenseSystem
from fraud_detection.types import Transaction, AttackResult, AttackType


class AdversarialLearner:
    """
    RL agent that learns to execute fraud attacks.

    Uses reinforcement learning to adaptively evolve attack strategies
    based on defense responses.
    """

    def __init__(self, defense_system: DefenseSystem):
        self.defense_system = defense_system
        self.env = FraudAttackEnv(defense_system)

        # RL agent (using stable-baselines3)
        self.agent = None
        self.is_trained = False

        # History
        self.training_history: List[Dict] = []
        self.attack_history: List[AttackResult] = []

    def train(
        self,
        num_episodes: int = 100,
        algorithm: str = "PPO"
    ) -> Dict[str, Any]:
        """
        Train the RL agent.

        Args:
            num_episodes: Number of training episodes
            algorithm: RL algorithm (PPO, A2C, SAC)

        Returns:
            Training metrics
        """
        from stable_baselines3 import PPO, A2C, SAC
        from stable_baselines3.common.callbacks import BaseCallback

        # Choose algorithm
        if algorithm == "PPO":
            self.agent = PPO("MlpPolicy", self.env, verbose=0)
        elif algorithm == "A2C":
            self.agent = A2C("MlpPolicy", self.env, verbose=0)
        elif algorithm == "SAC":
            self.agent = SAC("MlpPolicy", self.env, verbose=0)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Training callback to track metrics
        class MetricsCallback(BaseCallback):
            def __init__(self, history_list):
                super(MetricsCallback, self).__init__()
                self.history = history_list

            def _on_step(self) -> bool:
                return True

            def _on_rollout_end(self) -> None:
                # Track episode metrics
                if len(self.model.ep_info_buffer) > 0:
                    episode_info = self.model.ep_info_buffer[-1]
                    self.history.append({
                        "episode": len(self.history),
                        "reward": episode_info.get("r", 0),
                        "length": episode_info.get("l", 0),
                    })

        callback = MetricsCallback(self.training_history)

        # Train
        total_timesteps = num_episodes * self.env.max_steps
        self.agent.learn(total_timesteps=total_timesteps, callback=callback)

        self.is_trained = True

        # Return summary metrics
        if self.training_history:
            avg_reward = np.mean([h["reward"] for h in self.training_history[-10:]])
            avg_length = np.mean([h["length"] for h in self.training_history[-10:]])
        else:
            avg_reward = 0.0
            avg_length = 0.0

        return {
            "num_episodes": num_episodes,
            "algorithm": algorithm,
            "avg_reward_last_10": avg_reward,
            "avg_episode_length": avg_length,
            "total_episodes": len(self.training_history),
        }

    def execute_learned_attack(
        self,
        attack_type: AttackType,
        num_transactions: int = 100,
        start_time: Optional[datetime] = None
    ) -> AttackResult:
        """
        Execute an attack using the learned policy.

        Args:
            attack_type: Type of attack
            num_transactions: Number of transactions
            start_time: Attack start time

        Returns:
            AttackResult with metrics
        """
        if not self.is_trained:
            raise ValueError("Agent not trained yet")

        if start_time is None:
            start_time = datetime.utcnow()

        transactions = []
        successful = 0
        blocked = 0

        # Reset environment
        obs = self.env.reset()

        for i in range(num_transactions):
            # Get action from learned policy
            action, _ = self.agent.predict(obs, deterministic=False)

            # Execute action
            obs, reward, done, info = self.env.step(action)

            # Track results
            if info.get("detection_score", 1.0) > 0.7:
                blocked += 1
            else:
                successful += 1

            if done:
                obs = self.env.reset()

        # Create attack result
        success_rate = successful / num_transactions if num_transactions > 0 else 0
        detection_rate = blocked / num_transactions if num_transactions > 0 else 0

        result = AttackResult(
            attack_type=attack_type,
            start_time=start_time,
            end_time=start_time + timedelta(hours=1),
            total_transactions=num_transactions,
            successful_transactions=successful,
            blocked_transactions=blocked,
            success_rate=success_rate,
            estimated_loss=successful * 100.0,  # Simplified
            detection_rate=detection_rate,
            transactions=transactions,
            metadata={
                "method": "reinforcement_learning",
                "learned_policy": True,
            }
        )

        self.attack_history.append(result)

        return result

    def evolve_strategy(
        self,
        num_evolution_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Evolve attack strategy based on recent defense performance.

        Args:
            num_evolution_steps: Number of evolution steps

        Returns:
            Evolution metrics
        """
        if not self.is_trained:
            raise ValueError("Agent not trained yet")

        evolution_metrics = []

        for step in range(num_evolution_steps):
            # Execute attack
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0

            done = False
            while not done:
                action, _ = self.agent.predict(obs, deterministic=False)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

            evolution_metrics.append({
                "step": step,
                "reward": episode_reward,
                "length": episode_length,
                "success_rate": info.get("success_rate", 0),
            })

            # Continue training (online learning)
            self.agent.learn(total_timesteps=100)

        return {
            "num_steps": num_evolution_steps,
            "final_success_rate": evolution_metrics[-1]["success_rate"] if evolution_metrics else 0,
            "avg_reward": np.mean([m["reward"] for m in evolution_metrics]),
            "metrics": evolution_metrics,
        }

    def get_learned_strategy(self) -> Dict[str, Any]:
        """
        Get the current learned attack strategy.

        Returns:
            Dictionary describing the strategy
        """
        if not self.is_trained:
            return {"message": "Agent not trained yet"}

        # Sample the policy to understand the strategy
        obs = self.env.reset()
        sampled_actions = []

        for _ in range(10):
            action, _ = self.agent.predict(obs, deterministic=True)
            sampled_actions.append(action)
            obs, _, done, _ = self.env.step(action)
            if done:
                obs = self.env.reset()

        sampled_actions = np.array(sampled_actions)

        strategy = {
            "avg_amount_multiplier": float(np.mean(sampled_actions[:, 0])),
            "avg_delay_multiplier": float(np.mean(sampled_actions[:, 1])),
            "device_rotation_prob": float(np.mean(sampled_actions[:, 2])),
            "ip_rotation_prob": float(np.mean(sampled_actions[:, 3])),
        }

        # Add interpretation
        if strategy["avg_amount_multiplier"] < 0.7:
            strategy["interpretation"] = "Conservative: Using small transaction amounts"
        elif strategy["avg_amount_multiplier"] > 1.5:
            strategy["interpretation"] = "Aggressive: Using large transaction amounts"
        else:
            strategy["interpretation"] = "Balanced: Using moderate transaction amounts"

        if strategy["avg_delay_multiplier"] > 2.0:
            strategy["interpretation"] += " with slow, stealthy timing"
        elif strategy["avg_delay_multiplier"] < 1.0:
            strategy["interpretation"] += " with fast timing"

        return strategy

    def get_training_history(self) -> List[Dict]:
        """Get training history."""
        return self.training_history

    def get_attack_history(self) -> List[AttackResult]:
        """Get attack execution history."""
        return self.attack_history
