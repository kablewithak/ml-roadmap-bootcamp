"""Visualization tools for fraud detection analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from fraud_detection.types import AttackResult, DefenseMetrics
from fraud_detection.evaluation.metrics import MetricsCalculator


class FraudVisualizer:
    """
    Visualization tools for fraud detection system.

    Creates comprehensive visualizations for:
    - Attack patterns
    - Defense effectiveness
    - ROC curves
    - Cost analysis
    - Time series
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize visualizer with style."""
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        sns.set_palette("husl")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        title: str = "ROC Curve",
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_score: Prediction scores
            title: Plot title
            output_path: Path to save plot
        """
        fpr, tpr, _ = MetricsCalculator.calculate_roc_curve(y_true, y_score)
        auc = MetricsCalculator.calculate_detection_metrics(
            y_true,
            (y_score > 0.5).astype(int),
            y_score
        )["roc_auc"]

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        title: str = "Precision-Recall Curve",
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot precision-recall curve.

        Args:
            y_true: True labels
            y_score: Prediction scores
            title: Plot title
            output_path: Path to save plot
        """
        precision, recall, _ = MetricsCalculator.calculate_precision_recall_curve(
            y_true, y_score
        )

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_attack_success_over_time(
        self,
        attack_results: List[AttackResult],
        title: str = "Attack Success Rate Over Time",
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot attack success rate over time.

        Args:
            attack_results: List of attack results
            title: Plot title
            output_path: Path to save plot
        """
        if not attack_results:
            return

        # Group by attack type
        by_type: Dict[str, List] = {}
        for result in attack_results:
            attack_name = result.attack_type.value
            if attack_name not in by_type:
                by_type[attack_name] = []
            by_type[attack_name].append({
                "time": result.start_time,
                "success_rate": result.success_rate
            })

        plt.figure(figsize=(14, 8))

        for attack_name, data in by_type.items():
            times = [d["time"] for d in data]
            success_rates = [d["success_rate"] for d in data]
            plt.plot(times, success_rates, marker='o', label=attack_name, linewidth=2)

        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_defense_metrics_over_time(
        self,
        metrics_history: List[DefenseMetrics],
        title: str = "Defense Performance Over Time",
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot defense metrics over time.

        Args:
            metrics_history: Historical metrics
            title: Plot title
            output_path: Path to save plot
        """
        if not metrics_history:
            return

        times = [m.timestamp for m in metrics_history]
        precisions = [m.precision for m in metrics_history]
        recalls = [m.recall for m in metrics_history]
        f1_scores = [m.f1_score for m in metrics_history]

        plt.figure(figsize=(14, 8))
        plt.plot(times, precisions, marker='o', label='Precision', linewidth=2)
        plt.plot(times, recalls, marker='s', label='Recall', linewidth=2)
        plt.plot(times, f1_scores, marker='^', label='F1 Score', linewidth=2)

        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            output_path: Path to save plot
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud']
        )

        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_cost_analysis(
        self,
        cost_data: Dict[str, float],
        title: str = "Cost-Benefit Analysis",
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot cost-benefit analysis.

        Args:
            cost_data: Cost analysis dictionary
            title: Plot title
            output_path: Path to save plot
        """
        categories = ['Prevented\nLoss', 'False Positive\nCost', 'False Negative\nCost',
                     'Operational\nCost', 'Net\nSavings']
        values = [
            cost_data.get('prevented_loss', 0),
            -cost_data.get('false_positive_cost', 0),
            -cost_data.get('false_negative_cost', 0),
            -cost_data.get('operational_cost', 0),
            cost_data.get('net_savings', 0),
        ]

        colors = ['green', 'red', 'red', 'orange', 'blue']

        plt.figure(figsize=(12, 8))
        bars = plt.bar(categories, values, color=colors, alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'${abs(height):,.0f}',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=10,
                fontweight='bold'
            )

        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.ylabel('Amount ($)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_attack_comparison(
        self,
        attack_metrics: Dict[str, Any],
        title: str = "Attack Success by Type",
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of different attack types.

        Args:
            attack_metrics: Attack metrics dictionary
            title: Plot title
            output_path: Path to save plot
        """
        by_type = attack_metrics.get('by_attack_type', {})

        if not by_type:
            return

        attack_names = list(by_type.keys())
        success_rates = [by_type[name].get('success_rate', 0) for name in attack_names]
        detection_rates = [by_type[name].get('detection_rate', 0) for name in attack_names]

        x = np.arange(len(attack_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(x - width/2, success_rates, width, label='Success Rate', alpha=0.8)
        bars2 = ax.bar(x + width/2, detection_rates, width, label='Detection Rate', alpha=0.8)

        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attack_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def create_dashboard(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
        attack_results: List[AttackResult],
        defense_metrics: List[DefenseMetrics],
        output_path: str = "fraud_detection_dashboard.png"
    ) -> None:
        """
        Create comprehensive dashboard with all visualizations.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_score: Prediction scores
            attack_results: Attack results
            defense_metrics: Defense metrics history
            output_path: Path to save dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = MetricsCalculator.calculate_roc_curve(y_true, y_score)
        auc = MetricsCalculator.calculate_detection_metrics(
            y_true, y_pred, y_score
        )["roc_auc"]
        ax1.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=['Legit', 'Fraud'],
                   yticklabels=['Legit', 'Fraud'])
        ax2.set_title('Confusion Matrix', fontweight='bold')

        # Metrics text
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        metrics = MetricsCalculator.calculate_detection_metrics(y_true, y_pred, y_score)
        metrics_text = f"""
        Detection Metrics:

        Precision: {metrics['precision']:.3f}
        Recall: {metrics['recall']:.3f}
        F1 Score: {metrics['f1_score']:.3f}
        ROC AUC: {metrics['roc_auc']:.3f}

        FPR: {metrics['false_positive_rate']:.3f}
        FNR: {metrics['false_negative_rate']:.3f}
        Accuracy: {metrics['accuracy']:.3f}
        """
        ax3.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace')

        # Additional plots would go here...
        # (truncated for space, but would include attack success, defense metrics, etc.)

        plt.suptitle('Fraud Detection System Dashboard', fontsize=16, fontweight='bold')

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
