"""Adversarial training for robust fraud detection models."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split

from fraud_detection.types import Transaction
from fraud_detection.utils.features import FeatureEngineer


class AdversarialTrainer:
    """
    Trains models with adversarial robustness.

    Techniques:
    - Data augmentation with adversarial examples
    - Poisoning detection
    - Robust feature engineering
    - Online learning with drift detection
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.poisoning_detector = PoisoningDetector()

    def generate_adversarial_examples(
        self,
        transactions: List[Transaction],
        perturbation_ratio: float = 0.1
    ) -> List[Transaction]:
        """
        Generate adversarial examples by perturbing transactions.

        Args:
            transactions: Original transactions
            perturbation_ratio: Fraction of features to perturb

        Returns:
            List of adversarial transactions
        """
        adversarial_txns = []

        for txn in transactions:
            if not txn.is_fraud:
                continue  # Only perturb fraud transactions

            # Create adversarial copy
            adv_txn = Transaction(**txn.__dict__)

            # Perturb features to evade detection
            # 1. Split large transactions into smaller ones
            if adv_txn.amount > 500:
                adv_txn.amount = adv_txn.amount / 2

            # 2. Add delays to reduce velocity
            # (handled in timing during attack simulation)

            # 3. Vary device/IP slightly
            if np.random.random() < perturbation_ratio:
                from fraud_detection.utils.data_generation import generate_device_id, fake
                adv_txn.device_id = generate_device_id()
                adv_txn.ip_address = fake.ipv4()

            adversarial_txns.append(adv_txn)

        return adversarial_txns

    def train_with_adversarial_data(
        self,
        legitimate_transactions: List[Transaction],
        fraud_transactions: List[Transaction],
        model: Any,
        adversarial_ratio: float = 0.3
    ) -> Dict[str, float]:
        """
        Train model with adversarial examples.

        Args:
            legitimate_transactions: Legitimate transactions
            fraud_transactions: Fraud transactions
            model: Model to train (should have train() method)
            adversarial_ratio: Ratio of adversarial examples to include

        Returns:
            Training metrics
        """
        # Generate adversarial examples
        num_adversarial = int(len(fraud_transactions) * adversarial_ratio)
        adversarial_txns = self.generate_adversarial_examples(
            fraud_transactions[:num_adversarial]
        )

        # Combine all data
        all_transactions = (
            legitimate_transactions +
            fraud_transactions +
            adversarial_txns
        )

        # Extract features
        features_df = self.feature_engineer.extract_batch_features(all_transactions)

        # Prepare training data
        X = features_df.drop(['transaction_id', 'is_fraud'], axis=1, errors='ignore')
        y = features_df['is_fraud'].astype(int).values

        # Check for data poisoning
        is_poisoned, poison_stats = self.poisoning_detector.detect_poisoning(X, y)
        if is_poisoned:
            print(f"Warning: Potential data poisoning detected. Stats: {poison_stats}")
            # Remove suspicious samples
            clean_indices = self.poisoning_detector.get_clean_indices(X, y)
            X = X.iloc[clean_indices]
            y = y[clean_indices]

        # Train model
        metrics = model.train(X, y)

        return metrics

    def online_learning_update(
        self,
        model: Any,
        new_transactions: List[Transaction],
        drift_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Update model with new data using online learning.

        Args:
            model: Trained model
            new_transactions: New transactions
            drift_threshold: Threshold for drift detection

        Returns:
            Update metrics and drift information
        """
        # Extract features
        features_df = self.feature_engineer.extract_batch_features(new_transactions)
        X = features_df.drop(['transaction_id', 'is_fraud'], axis=1, errors='ignore')
        y = features_df['is_fraud'].astype(int).values

        # Check for drift
        drift_detected = self._detect_drift(X, drift_threshold)

        # Check for poisoning
        is_poisoned, poison_stats = self.poisoning_detector.detect_poisoning(X, y)

        if is_poisoned:
            return {
                "updated": False,
                "reason": "Data poisoning detected",
                "poison_stats": poison_stats,
                "drift_detected": drift_detected
            }

        # Retrain if drift detected and data is clean
        if drift_detected:
            metrics = model.train(X, y)
            return {
                "updated": True,
                "metrics": metrics,
                "drift_detected": True,
                "num_samples": len(new_transactions)
            }

        return {
            "updated": False,
            "reason": "No drift detected",
            "drift_detected": False
        }

    def _detect_drift(self, X: pd.DataFrame, threshold: float) -> bool:
        """
        Detect concept drift in features.

        Args:
            X: Feature DataFrame
            threshold: Drift threshold

        Returns:
            True if drift detected
        """
        # Simple drift detection: check if feature distributions have changed
        # In production, use more sophisticated methods like Kolmogorov-Smirnov test

        # For now, check if mean features differ significantly
        if not hasattr(self, '_feature_means'):
            self._feature_means = X.mean()
            return False

        current_means = X.mean()
        relative_changes = np.abs(
            (current_means - self._feature_means) / (self._feature_means + 1e-6)
        )

        max_change = relative_changes.max()

        if max_change > threshold:
            self._feature_means = current_means  # Update baseline
            return True

        return False


class PoisoningDetector:
    """
    Detects data poisoning attacks in training data.

    Poisoning attacks attempt to manipulate the model by injecting
    malicious training samples.
    """

    def detect_poisoning(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        contamination: float = 0.1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if training data is poisoned.

        Args:
            X: Features
            y: Labels
            contamination: Expected contamination rate

        Returns:
            Tuple of (is_poisoned, statistics)
        """
        from sklearn.ensemble import IsolationForest

        # Use Isolation Forest to detect outliers
        detector = IsolationForest(contamination=contamination, random_state=42)
        outlier_predictions = detector.fit_predict(X)

        # Check label consistency
        fraud_mask = (y == 1)
        legit_mask = (y == 0)

        fraud_outliers = (outlier_predictions[fraud_mask] == -1).sum()
        legit_outliers = (outlier_predictions[legit_mask] == -1).sum()

        total_outliers = (outlier_predictions == -1).sum()

        # If too many outliers, might be poisoning
        outlier_ratio = total_outliers / len(X)

        stats = {
            "total_outliers": int(total_outliers),
            "outlier_ratio": float(outlier_ratio),
            "fraud_outliers": int(fraud_outliers),
            "legit_outliers": int(legit_outliers),
        }

        # Simple heuristic: if outlier ratio is much higher than expected
        is_poisoned = outlier_ratio > contamination * 1.5

        return is_poisoned, stats

    def get_clean_indices(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        contamination: float = 0.1
    ) -> np.ndarray:
        """
        Get indices of clean (non-poisoned) samples.

        Args:
            X: Features
            y: Labels
            contamination: Expected contamination rate

        Returns:
            Array of clean indices
        """
        from sklearn.ensemble import IsolationForest

        detector = IsolationForest(contamination=contamination, random_state=42)
        predictions = detector.fit_predict(X)

        # Keep only inliers
        clean_indices = np.where(predictions == 1)[0]

        return clean_indices
