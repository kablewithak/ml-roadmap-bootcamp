#!/usr/bin/env python3
"""
Complete demo of the adversarial fraud detection system.

This script demonstrates:
1. Generating training data
2. Training the defense system
3. Simulating various attacks
4. Evaluating performance
5. Calculating business impact
6. Visualizing results
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fraud_detection.attacks import AttackSimulator
from fraud_detection.defense import DefenseSystem
from fraud_detection.graph import GraphFraudDetector
from fraud_detection.adversarial import AdversarialLearner
from fraud_detection.evaluation import EvaluationFramework
from fraud_detection.business import BusinessImpactCalculator
from fraud_detection.monitoring import FraudDetectionLogger, MetricsTracker
from fraud_detection.utils.data_generation import generate_legitimate_transactions, generate_fraud_transactions
from fraud_detection.types import AttackType


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Run complete system demo."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "ADVERSARIAL FRAUD DETECTION SYSTEM DEMO" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝\n")

    # Initialize components
    print_section("1. INITIALIZING COMPONENTS")

    attack_simulator = AttackSimulator()
    defense_system = DefenseSystem()
    graph_detector = GraphFraudDetector()
    evaluator = EvaluationFramework()
    business_calculator = BusinessImpactCalculator()
    logger = FraudDetectionLogger()
    metrics_tracker = MetricsTracker()

    print("✓ Attack simulator initialized")
    print("✓ Defense system initialized")
    print("✓ Graph detector initialized")
    print("✓ Evaluation framework initialized")
    print("✓ Business calculator initialized")
    print("✓ Monitoring tools initialized")

    # Generate training data
    print_section("2. GENERATING TRAINING DATA")

    legit_transactions = generate_legitimate_transactions(1000)
    fraud_transactions = generate_fraud_transactions(500, attack_signature="generic")

    print(f"✓ Generated {len(legit_transactions)} legitimate transactions")
    print(f"✓ Generated {len(fraud_transactions)} fraudulent transactions")
    print(f"✓ Total training samples: {len(legit_transactions) + len(fraud_transactions)}")
    print(f"✓ Fraud rate: {len(fraud_transactions) / (len(legit_transactions) + len(fraud_transactions)) * 100:.2f}%")

    # Train defense system
    print_section("3. TRAINING DEFENSE SYSTEM")

    print("Training with adversarial robustness...")
    training_metrics = defense_system.train(
        legit_transactions,
        fraud_transactions,
        use_adversarial_training=True
    )

    print(f"\nTraining Results:")
    print(f"  ROC AUC:   {training_metrics.get('roc_auc', 0):.4f}")
    print(f"  Precision: {training_metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {training_metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:  {training_metrics.get('f1_score', 0):.4f}")

    # Build transaction graph
    print_section("4. BUILDING TRANSACTION GRAPH")

    all_transactions = legit_transactions + fraud_transactions
    graph_detector.build_graph(all_transactions)

    graph_stats = graph_detector.get_graph_statistics()
    print(f"Graph Statistics:")
    print(f"  Total nodes: {graph_stats['num_nodes']}")
    print(f"  Total edges: {graph_stats['num_edges']}")
    print(f"  Users: {graph_stats['num_users']}")
    print(f"  Devices: {graph_stats['num_devices']}")
    print(f"  IPs: {graph_stats['num_ips']}")
    print(f"  Fraud rings detected: {graph_stats.get('num_fraud_rings', 0)}")

    # Simulate attacks
    print_section("5. SIMULATING ADVERSARIAL ATTACKS")

    def defense_callback(txn):
        """Check if transaction is blocked by defense."""
        is_blocked, score, _ = defense_system.predict(txn)
        metrics_tracker.record_prediction(is_blocked, score, 50.0)
        return is_blocked

    # Get some attack patterns to test
    attack_patterns_to_test = [
        attack_simulator.create_attack_pattern(AttackType.CARD_TESTING, 100, 0.5),
        attack_simulator.create_attack_pattern(AttackType.ACCOUNT_TAKEOVER, 20, 1.0),
        attack_simulator.create_attack_pattern(AttackType.VELOCITY_EVASION, 50, 12.0),
        attack_simulator.create_attack_pattern(AttackType.BLEND_ATTACK, 100, 24.0),
    ]

    attack_results = []
    for i, pattern in enumerate(attack_patterns_to_test):
        print(f"\n[{i+1}/{len(attack_patterns_to_test)}] Executing {pattern.attack_type.value}...")
        result = attack_simulator.execute_attack(pattern, defense_callback=defense_callback)

        print(f"  Transactions: {result.total_transactions}")
        print(f"  Successful: {result.successful_transactions}")
        print(f"  Blocked: {result.blocked_transactions}")
        print(f"  Success Rate: {result.success_rate * 100:.2f}%")
        print(f"  Estimated Loss: ${result.estimated_loss:,.2f}")

        attack_results.append(result)

    # Campaign summary
    campaign_summary = attack_simulator.get_campaign_summary()
    print(f"\nCampaign Summary:")
    print(f"  Total attacks: {campaign_summary['total_attacks']}")
    print(f"  Total transactions: {campaign_summary['total_transactions']}")
    print(f"  Overall success rate: {campaign_summary['overall_success_rate'] * 100:.2f}%")
    print(f"  Overall detection rate: {campaign_summary['overall_detection_rate'] * 100:.2f}%")
    print(f"  Total estimated loss: ${campaign_summary['total_estimated_loss']:,.2f}")

    # Evaluate defense performance
    print_section("6. EVALUATING DEFENSE PERFORMANCE")

    test_transactions = generate_legitimate_transactions(500) + generate_fraud_transactions(250)

    evaluation_results = evaluator.evaluate_defense_system(
        defense_system,
        test_transactions,
        create_visualizations=True,
        output_dir="artifacts/figures"
    )

    print("Defense Performance:")
    defense_metrics = evaluation_results["defense_metrics"]
    print(f"  Precision: {defense_metrics['precision']:.4f}")
    print(f"  Recall: {defense_metrics['recall']:.4f}")
    print(f"  F1 Score: {defense_metrics['f1_score']:.4f}")
    print(f"  ROC AUC: {defense_metrics['roc_auc']:.4f}")

    print("\n✓ Visualizations saved to artifacts/figures/")

    # Calculate business impact
    print_section("7. CALCULATING BUSINESS IMPACT")

    from fraud_detection.types import DefenseMetrics
    defense_perf = defense_system.evaluate_performance(test_transactions)

    business_metrics = business_calculator.calculate_business_impact(defense_perf)

    print("Financial Impact:")
    print(f"  Prevented Loss: ${business_metrics.prevented_loss:,.2f}")
    print(f"  False Positive Cost: ${business_metrics.false_positive_cost:,.2f}")
    print(f"  Operational Cost: ${business_metrics.operational_cost:,.2f}")
    print(f"  NET SAVINGS: ${business_metrics.net_savings:,.2f}")
    print(f"  ROI: {business_metrics.roi:.1f}%")

    # Annual projection
    annual_projection = business_calculator.calculate_annual_projection(
        business_metrics,
        transactions_per_day=10000
    )
    print(f"\n{annual_projection['summary']}")

    # Generate executive summary
    print_section("8. EXECUTIVE SUMMARY")

    exec_summary = business_calculator.generate_executive_summary(
        business_metrics,
        defense_perf
    )
    print(exec_summary)

    # System monitoring stats
    print_section("9. SYSTEM MONITORING")

    current_metrics = metrics_tracker.get_current_metrics()
    print("Current System Metrics:")
    print(f"  Throughput: {current_metrics['throughput_tps']:.2f} TPS")
    print(f"  Avg Latency: {current_metrics['avg_latency_ms']:.2f} ms")
    print(f"  P95 Latency: {current_metrics['p95_latency_ms']:.2f} ms")
    print(f"  P99 Latency: {current_metrics['p99_latency_ms']:.2f} ms")
    print(f"  Block Rate: {current_metrics['block_rate'] * 100:.2f}%")

    alerts = metrics_tracker.check_alerts()
    if alerts:
        print(f"\n⚠️  Active Alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  [{alert['level'].upper()}] {alert['message']}")
    else:
        print("\n✓ No active alerts - system operating normally")

    # Generate final report
    print_section("10. GENERATING FINAL REPORT")

    report = evaluator.generate_report(output_path="fraud_detection_report.txt")
    print("✓ Comprehensive report generated: fraud_detection_report.txt")

    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "DEMO COMPLETED SUCCESSFULLY" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝\n")

    print("Next Steps:")
    print("  1. Review visualizations in artifacts/figures/")
    print("  2. Read the full report in fraud_detection_report.txt")
    print("  3. Check logs in logs/ directory")
    print("  4. Explore the codebase in fraud_detection/")
    print("\nFor more examples, see the examples/ directory")
    print("For tests, run: pytest tests/")
    print()


if __name__ == "__main__":
    main()
