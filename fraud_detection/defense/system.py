"""Main defense system coordinator."""

from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

from fraud_detection.types import Transaction, DefenseMetrics
from fraud_detection.defense.models import EnsembleDetector
from fraud_detection.defense.training import AdversarialTrainer
from fraud_detection.utils.features import FeatureEngineer


class DefenseSystem:
    """
    Main defense system coordinating all defense mechanisms.

    Components:
    - Ensemble detection (rules + ML + graph)
    - Adversarial training
    - Online learning
    - Performance monitoring
    """

    def __init__(self):
        self.ensemble_detector = EnsembleDetector()
        self.adversarial_trainer = AdversarialTrainer()
        self.feature_engineer = FeatureEngineer()

        self.is_trained = False
        self.performance_history: List[DefenseMetrics] = []

        # Stats
        self.total_processed = 0
        self.total_blocked = 0
        self.total_allowed = 0

        # Latency tracking
        self.latencies_ms: List[float] = []

    def train(
        self,
        legitimate_transactions: List[Transaction],
        fraud_transactions: List[Transaction],
        use_adversarial_training: bool = True
    ) -> Dict[str, float]:
        """
        Train the defense system.

        Args:
            legitimate_transactions: Legitimate training data
            fraud_transactions: Fraud training data
            use_adversarial_training: Whether to use adversarial training

        Returns:
            Training metrics
        """
        if use_adversarial_training:
            metrics = self.adversarial_trainer.train_with_adversarial_data(
                legitimate_transactions,
                fraud_transactions,
                self.ensemble_detector
            )
        else:
            # Standard training
            all_transactions = legitimate_transactions + fraud_transactions
            features_df = self.feature_engineer.extract_batch_features(all_transactions)
            X = features_df.drop(['transaction_id', 'is_fraud'], axis=1, errors='ignore')
            y = features_df['is_fraud'].astype(int).values
            metrics = self.ensemble_detector.train(X, y)

        self.is_trained = True
        return metrics

    def predict(
        self,
        transaction: Transaction,
        return_details: bool = False
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Predict if transaction is fraudulent.

        Args:
            transaction: Transaction to check
            return_details: Whether to return detailed information

        Returns:
            Tuple of (is_fraud, score, details)
        """
        start_time = time.time()

        # Extract features
        features = self.feature_engineer.extract_features(transaction)

        # Get prediction
        is_fraud, score, details = self.ensemble_detector.predict(features)

        # Update history
        self.feature_engineer.update_history(transaction)

        # Track latency
        latency_ms = (time.time() - start_time) * 1000
        self.latencies_ms.append(latency_ms)

        # Update stats
        self.total_processed += 1
        if is_fraud:
            self.total_blocked += 1
        else:
            self.total_allowed += 1

        if return_details:
            details.update({
                "latency_ms": latency_ms,
                "features": features
            })
            return is_fraud, score, details
        else:
            return is_fraud, score, None

    def predict_batch(
        self,
        transactions: List[Transaction]
    ) -> List[Tuple[bool, float]]:
        """
        Predict fraud for batch of transactions.

        Args:
            transactions: List of transactions

        Returns:
            List of (is_fraud, score) tuples
        """
        results = []

        for txn in transactions:
            is_fraud, score, _ = self.predict(txn)
            results.append((is_fraud, score))

        return results

    def update_with_feedback(
        self,
        transactions: List[Transaction],
        true_labels: List[bool]
    ) -> Dict[str, Any]:
        """
        Update model with labeled feedback (online learning).

        Args:
            transactions: Transactions
            true_labels: True fraud labels

        Returns:
            Update metrics
        """
        # Update transaction labels
        for txn, label in zip(transactions, true_labels):
            txn.is_fraud = label

        # Online learning update
        update_result = self.adversarial_trainer.online_learning_update(
            self.ensemble_detector,
            transactions
        )

        return update_result

    def evaluate_performance(
        self,
        test_transactions: List[Transaction]
    ) -> DefenseMetrics:
        """
        Evaluate defense performance.

        Args:
            test_transactions: Test transactions with known labels

        Returns:
            DefenseMetrics with performance statistics
        """
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score
        )

        # Get predictions
        predictions = []
        scores = []
        true_labels = []

        for txn in test_transactions:
            is_fraud, score, _ = self.predict(txn)
            predictions.append(is_fraud)
            scores.append(score)
            true_labels.append(txn.is_fraud)

        # Calculate metrics
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        roc_auc = roc_auc_score(true_labels, scores)

        # Calculate latency metrics
        avg_latency = sum(self.latencies_ms[-len(test_transactions):]) / len(test_transactions)

        # Calculate throughput
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        metrics = DefenseMetrics(
            timestamp=datetime.utcnow(),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            avg_detection_latency_ms=avg_latency,
            throughput_tps=throughput
        )

        self.performance_history.append(metrics)

        return metrics

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        return {
            "total_processed": self.total_processed,
            "total_blocked": self.total_blocked,
            "total_allowed": self.total_allowed,
            "block_rate": self.total_blocked / self.total_processed if self.total_processed > 0 else 0,
            "avg_latency_ms": sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0,
            "p95_latency_ms": sorted(self.latencies_ms)[int(len(self.latencies_ms) * 0.95)] if self.latencies_ms else 0,
            "p99_latency_ms": sorted(self.latencies_ms)[int(len(self.latencies_ms) * 0.99)] if self.latencies_ms else 0,
            "is_trained": self.is_trained,
        }

    def get_performance_history(self) -> List[DefenseMetrics]:
        """Get historical performance metrics."""
        return self.performance_history

    def reset_stats(self):
        """Reset system statistics."""
        self.total_processed = 0
        self.total_blocked = 0
        self.total_allowed = 0
        self.latencies_ms = []
