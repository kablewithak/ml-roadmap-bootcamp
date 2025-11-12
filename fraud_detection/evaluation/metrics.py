"""Metrics calculation for fraud detection evaluation."""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

from fraud_detection.types import AttackResult, DefenseMetrics


class MetricsCalculator:
    """Calculate comprehensive metrics for fraud detection."""

    @staticmethod
    def calculate_detection_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate standard detection metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_score: Prediction scores

        Returns:
            Dictionary of metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            # Basic metrics
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),

            # Performance metrics
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),

            # ROC metrics
            "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0,
            "average_precision": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0,

            # Rates
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            "true_negative_rate": tn / (tn + fp) if (tn + fp) > 0 else 0.0,

            # Accuracy
            "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0,
        }

        return metrics

    @staticmethod
    def calculate_attack_success_metrics(
        attack_results: List[AttackResult]
    ) -> Dict[str, Any]:
        """
        Calculate metrics for attack success over time.

        Args:
            attack_results: List of attack results

        Returns:
            Dictionary of attack metrics
        """
        if not attack_results:
            return {}

        total_transactions = sum(r.total_transactions for r in attack_results)
        total_successful = sum(r.successful_transactions for r in attack_results)
        total_blocked = sum(r.blocked_transactions for r in attack_results)
        total_loss = sum(r.estimated_loss for r in attack_results)

        # By attack type
        by_type: Dict[str, Dict] = {}
        for result in attack_results:
            attack_name = result.attack_type.value
            if attack_name not in by_type:
                by_type[attack_name] = {
                    "attempts": 0,
                    "successful": 0,
                    "blocked": 0,
                    "loss": 0.0,
                }

            by_type[attack_name]["attempts"] += result.total_transactions
            by_type[attack_name]["successful"] += result.successful_transactions
            by_type[attack_name]["blocked"] += result.blocked_transactions
            by_type[attack_name]["loss"] += result.estimated_loss

        # Calculate rates
        for attack_name, stats in by_type.items():
            if stats["attempts"] > 0:
                stats["success_rate"] = stats["successful"] / stats["attempts"]
                stats["detection_rate"] = stats["blocked"] / stats["attempts"]

        return {
            "total_transactions": total_transactions,
            "total_successful": total_successful,
            "total_blocked": total_blocked,
            "overall_success_rate": total_successful / total_transactions if total_transactions > 0 else 0,
            "overall_detection_rate": total_blocked / total_transactions if total_transactions > 0 else 0,
            "total_estimated_loss": total_loss,
            "by_attack_type": by_type,
        }

    @staticmethod
    def calculate_defense_degradation(
        metrics_history: List[DefenseMetrics]
    ) -> Dict[str, Any]:
        """
        Calculate defense degradation over time.

        Args:
            metrics_history: Historical defense metrics

        Returns:
            Degradation analysis
        """
        if len(metrics_history) < 2:
            return {"message": "Insufficient history for degradation analysis"}

        # Compare first and last metrics
        first = metrics_history[0]
        last = metrics_history[-1]

        precision_change = last.precision - first.precision
        recall_change = last.recall - first.recall
        f1_change = last.f1_score - first.f1_score
        roc_auc_change = last.roc_auc - first.roc_auc

        # Calculate trend
        precisions = [m.precision for m in metrics_history]
        recalls = [m.recall for m in metrics_history]
        f1_scores = [m.f1_score for m in metrics_history]

        return {
            "precision_change": precision_change,
            "recall_change": recall_change,
            "f1_change": f1_change,
            "roc_auc_change": roc_auc_change,

            "precision_trend": "improving" if precision_change > 0.01 else "degrading" if precision_change < -0.01 else "stable",
            "recall_trend": "improving" if recall_change > 0.01 else "degrading" if recall_change < -0.01 else "stable",
            "f1_trend": "improving" if f1_change > 0.01 else "degrading" if f1_change < -0.01 else "stable",

            "avg_precision": np.mean(precisions),
            "avg_recall": np.mean(recalls),
            "avg_f1": np.mean(f1_scores),

            "precision_std": np.std(precisions),
            "recall_std": np.std(recalls),
            "f1_std": np.std(f1_scores),
        }

    @staticmethod
    def calculate_cost_analysis(
        defense_metrics: DefenseMetrics,
        prevented_loss_per_tp: float = 250.0,
        fp_cost_per_transaction: float = 5.0,
        fn_cost_per_transaction: float = 250.0,
        operational_cost_per_transaction: float = 0.01
    ) -> Dict[str, float]:
        """
        Calculate cost-benefit analysis.

        Args:
            defense_metrics: Defense performance metrics
            prevented_loss_per_tp: Average prevented loss per true positive
            fp_cost_per_transaction: Cost per false positive (customer friction)
            fn_cost_per_transaction: Cost per false negative (fraud loss)
            operational_cost_per_transaction: Cost to run defense system

        Returns:
            Cost analysis
        """
        total_transactions = (
            defense_metrics.true_positives +
            defense_metrics.false_positives +
            defense_metrics.true_negatives +
            defense_metrics.false_negatives
        )

        prevented_loss = defense_metrics.true_positives * prevented_loss_per_tp
        fp_cost = defense_metrics.false_positives * fp_cost_per_transaction
        fn_cost = defense_metrics.false_negatives * fn_cost_per_transaction
        operational_cost = total_transactions * operational_cost_per_transaction

        total_cost = fp_cost + fn_cost + operational_cost
        net_savings = prevented_loss - total_cost

        roi = (net_savings / total_cost * 100) if total_cost > 0 else 0

        return {
            "prevented_loss": prevented_loss,
            "false_positive_cost": fp_cost,
            "false_negative_cost": fn_cost,
            "operational_cost": operational_cost,
            "total_cost": total_cost,
            "net_savings": net_savings,
            "roi_percentage": roi,
            "cost_per_transaction": total_cost / total_transactions if total_transactions > 0 else 0,
        }

    @staticmethod
    def calculate_roc_curve(
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve.

        Args:
            y_true: True labels
            y_score: Prediction scores

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_score)

    @staticmethod
    def calculate_precision_recall_curve(
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate precision-recall curve.

        Args:
            y_true: True labels
            y_score: Prediction scores

        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(y_true, y_score)
