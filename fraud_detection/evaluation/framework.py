"""Main evaluation framework coordinating all evaluation components."""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from fraud_detection.types import Transaction, AttackResult, DefenseMetrics
from fraud_detection.defense.system import DefenseSystem
from fraud_detection.attacks.simulator import AttackSimulator
from fraud_detection.evaluation.metrics import MetricsCalculator
from fraud_detection.evaluation.visualizer import FraudVisualizer


class EvaluationFramework:
    """
    Comprehensive evaluation framework for fraud detection system.

    Evaluates:
    - Attack success rates
    - Defense effectiveness
    - ROC curves under adversarial conditions
    - Cost analysis
    - Defense degradation over time
    """

    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = FraudVisualizer()

        # Results storage
        self.evaluation_results: List[Dict] = []

    def evaluate_defense_system(
        self,
        defense_system: DefenseSystem,
        test_transactions: List[Transaction],
        create_visualizations: bool = True,
        output_dir: str = "artifacts/figures"
    ) -> Dict[str, Any]:
        """
        Comprehensively evaluate defense system.

        Args:
            defense_system: Defense system to evaluate
            test_transactions: Test transactions
            create_visualizations: Whether to create visualizations
            output_dir: Directory for outputs

        Returns:
            Comprehensive evaluation results
        """
        # Get predictions
        y_true = []
        y_pred = []
        y_score = []

        for txn in test_transactions:
            is_fraud, score, _ = defense_system.predict(txn)
            y_true.append(txn.is_fraud)
            y_pred.append(is_fraud)
            y_score.append(score)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)

        # Calculate metrics
        detection_metrics = self.metrics_calculator.calculate_detection_metrics(
            y_true, y_pred, y_score
        )

        # Evaluate on defense metrics object
        defense_metrics = defense_system.evaluate_performance(test_transactions)

        # Cost analysis
        cost_analysis = self.metrics_calculator.calculate_cost_analysis(defense_metrics)

        # Get system stats
        system_stats = defense_system.get_system_stats()

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "num_transactions": len(test_transactions),
            "detection_metrics": detection_metrics,
            "cost_analysis": cost_analysis,
            "system_stats": system_stats,
            "defense_metrics": {
                "precision": defense_metrics.precision,
                "recall": defense_metrics.recall,
                "f1_score": defense_metrics.f1_score,
                "roc_auc": defense_metrics.roc_auc,
            }
        }

        # Create visualizations
        if create_visualizations:
            import os
            os.makedirs(output_dir, exist_ok=True)

            self.visualizer.plot_roc_curve(
                y_true, y_score,
                output_path=f"{output_dir}/roc_curve.png"
            )
            self.visualizer.plot_precision_recall_curve(
                y_true, y_score,
                output_path=f"{output_dir}/precision_recall.png"
            )
            self.visualizer.plot_confusion_matrix(
                y_true, y_pred,
                output_path=f"{output_dir}/confusion_matrix.png"
            )
            self.visualizer.plot_cost_analysis(
                cost_analysis,
                output_path=f"{output_dir}/cost_analysis.png"
            )

        self.evaluation_results.append(results)

        return results

    def evaluate_attack_campaign(
        self,
        attack_results: List[AttackResult],
        create_visualizations: bool = True,
        output_dir: str = "artifacts/figures"
    ) -> Dict[str, Any]:
        """
        Evaluate attack campaign results.

        Args:
            attack_results: List of attack results
            create_visualizations: Whether to create visualizations
            output_dir: Directory for outputs

        Returns:
            Attack campaign analysis
        """
        attack_metrics = self.metrics_calculator.calculate_attack_success_metrics(
            attack_results
        )

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "num_attacks": len(attack_results),
            "attack_metrics": attack_metrics,
        }

        # Create visualizations
        if create_visualizations:
            import os
            os.makedirs(output_dir, exist_ok=True)

            self.visualizer.plot_attack_success_over_time(
                attack_results,
                output_path=f"{output_dir}/attack_success_over_time.png"
            )
            self.visualizer.plot_attack_comparison(
                attack_metrics,
                output_path=f"{output_dir}/attack_comparison.png"
            )

        return results

    def evaluate_adversarial_robustness(
        self,
        defense_system: DefenseSystem,
        attack_simulator: AttackSimulator,
        num_iterations: int = 5,
        create_visualizations: bool = True,
        output_dir: str = "artifacts/figures"
    ) -> Dict[str, Any]:
        """
        Evaluate defense robustness under adversarial attacks.

        Args:
            defense_system: Defense system
            attack_simulator: Attack simulator
            num_iterations: Number of attack-defense iterations
            create_visualizations: Whether to create visualizations
            output_dir: Directory for outputs

        Returns:
            Robustness evaluation results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "num_iterations": num_iterations,
            "iterations": []
        }

        metrics_history = []

        # Run multiple iterations
        for i in range(num_iterations):
            print(f"Running iteration {i + 1}/{num_iterations}...")

            # Execute all attack patterns
            attack_patterns = attack_simulator.get_all_attack_patterns()

            iteration_results = []
            for pattern in attack_patterns:
                # Execute attack with defense callback
                def defense_callback(txn):
                    is_blocked, _, _ = defense_system.predict(txn)
                    return is_blocked

                result = attack_simulator.execute_attack(
                    pattern,
                    defense_callback=defense_callback
                )
                iteration_results.append(result)

            # Evaluate defense performance
            all_transactions = []
            for result in iteration_results:
                all_transactions.extend(result.transactions)

            if all_transactions:
                defense_metrics = defense_system.evaluate_performance(all_transactions)
                metrics_history.append(defense_metrics)

                iteration_summary = {
                    "iteration": i,
                    "num_attacks": len(iteration_results),
                    "total_transactions": len(all_transactions),
                    "defense_metrics": {
                        "precision": defense_metrics.precision,
                        "recall": defense_metrics.recall,
                        "f1_score": defense_metrics.f1_score,
                        "roc_auc": defense_metrics.roc_auc,
                    }
                }
                results["iterations"].append(iteration_summary)

        # Calculate degradation
        if metrics_history:
            degradation = self.metrics_calculator.calculate_defense_degradation(
                metrics_history
            )
            results["degradation_analysis"] = degradation

        # Create visualizations
        if create_visualizations and metrics_history:
            import os
            os.makedirs(output_dir, exist_ok=True)

            self.visualizer.plot_defense_metrics_over_time(
                metrics_history,
                output_path=f"{output_dir}/defense_degradation.png"
            )

        return results

    def generate_report(
        self,
        output_path: str = "fraud_detection_report.txt"
    ) -> str:
        """
        Generate comprehensive text report.

        Args:
            output_path: Path to save report

        Returns:
            Report text
        """
        if not self.evaluation_results:
            return "No evaluation results available."

        report_lines = [
            "=" * 80,
            "FRAUD DETECTION SYSTEM EVALUATION REPORT",
            "=" * 80,
            f"\nGenerated: {datetime.utcnow().isoformat()}",
            f"\nTotal Evaluations: {len(self.evaluation_results)}",
            "\n"
        ]

        for i, result in enumerate(self.evaluation_results):
            report_lines.extend([
                f"\n{'=' * 80}",
                f"Evaluation #{i + 1}",
                f"{'=' * 80}",
                f"\nTimestamp: {result.get('timestamp', 'N/A')}",
            ])

            # Detection metrics
            if "detection_metrics" in result:
                metrics = result["detection_metrics"]
                report_lines.extend([
                    "\n--- Detection Metrics ---",
                    f"Precision: {metrics.get('precision', 0):.4f}",
                    f"Recall: {metrics.get('recall', 0):.4f}",
                    f"F1 Score: {metrics.get('f1_score', 0):.4f}",
                    f"ROC AUC: {metrics.get('roc_auc', 0):.4f}",
                    f"Accuracy: {metrics.get('accuracy', 0):.4f}",
                    f"False Positive Rate: {metrics.get('false_positive_rate', 0):.4f}",
                    f"False Negative Rate: {metrics.get('false_negative_rate', 0):.4f}",
                ])

            # Cost analysis
            if "cost_analysis" in result:
                cost = result["cost_analysis"]
                report_lines.extend([
                    "\n--- Cost Analysis ---",
                    f"Prevented Loss: ${cost.get('prevented_loss', 0):,.2f}",
                    f"False Positive Cost: ${cost.get('false_positive_cost', 0):,.2f}",
                    f"False Negative Cost: ${cost.get('false_negative_cost', 0):,.2f}",
                    f"Operational Cost: ${cost.get('operational_cost', 0):,.2f}",
                    f"Net Savings: ${cost.get('net_savings', 0):,.2f}",
                    f"ROI: {cost.get('roi_percentage', 0):.2f}%",
                ])

        report_text = "\n".join(report_lines)

        # Save to file
        with open(output_path, 'w') as f:
            f.write(report_text)

        return report_text
