"""Comprehensive integration tests for the fraud detection system."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from fraud_detection.attacks import AttackSimulator
from fraud_detection.defense import DefenseSystem
from fraud_detection.graph import GraphFraudDetector
from fraud_detection.adversarial import AdversarialLearner
from fraud_detection.evaluation import EvaluationFramework
from fraud_detection.business import BusinessImpactCalculator
from fraud_detection.monitoring import FraudDetectionLogger, MetricsTracker
from fraud_detection.utils.data_generation import generate_legitimate_transactions, generate_fraud_transactions
from fraud_detection.types import AttackType


class TestCompleteSystem:
    """Integration tests for complete fraud detection system."""

    def test_attack_simulator(self):
        """Test attack simulator with all 10 attack patterns."""
        simulator = AttackSimulator()

        # Test individual attack patterns
        for attack_type in AttackType:
            pattern = simulator.create_attack_pattern(
                attack_type=attack_type,
                num_transactions=10,
                duration_hours=1.0
            )

            result = simulator.execute_attack(pattern)

            assert result.total_transactions > 0
            assert result.attack_type == attack_type
            assert len(result.transactions) > 0

        print("✓ All 10 attack patterns working")

    def test_defense_system(self):
        """Test defense system training and prediction."""
        defense = DefenseSystem()

        # Generate training data
        legit_txns = generate_legitimate_transactions(100)
        fraud_txns = generate_fraud_transactions(50, attack_signature="generic")

        # Train system
        metrics = defense.train(legit_txns, fraud_txns)

        assert defense.is_trained
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] > 0

        # Test prediction
        test_txn = legit_txns[0]
        is_fraud, score, details = defense.predict(test_txn, return_details=True)

        assert isinstance(is_fraud, bool)
        assert 0 <= score <= 1
        assert details is not None

        print("✓ Defense system training and prediction working")

    def test_graph_fraud_detection(self):
        """Test graph-based fraud detection."""
        graph_detector = GraphFraudDetector()

        # Generate transactions
        transactions = generate_legitimate_transactions(50) + generate_fraud_transactions(25)

        # Build graph
        graph_detector.build_graph(transactions)

        # Test fraud ring detection
        fraud_rings = graph_detector.detect_fraud_rings()
        assert isinstance(fraud_rings, list)

        # Test graph features
        graph_features = graph_detector.compute_graph_features(transactions[0])
        assert len(graph_features) > 0
        assert "user_degree" in graph_features

        # Test fraud score
        score = graph_detector.compute_fraud_score(transactions[0])
        assert 0 <= score <= 1

        print("✓ Graph-based fraud detection working")

    def test_adversarial_learning(self):
        """Test adversarial learning with RL agent."""
        defense = DefenseSystem()

        # Train defense first
        legit_txns = generate_legitimate_transactions(100)
        fraud_txns = generate_fraud_transactions(50)
        defense.train(legit_txns, fraud_txns)

        # Create adversarial learner
        learner = AdversarialLearner(defense)

        # Train RL agent (short training for test)
        training_metrics = learner.train(num_episodes=5)

        assert learner.is_trained
        assert "avg_reward_last_10" in training_metrics

        # Get learned strategy
        strategy = learner.get_learned_strategy()
        assert "avg_amount_multiplier" in strategy

        print("✓ Adversarial learning working")

    def test_evaluation_framework(self):
        """Test evaluation framework."""
        evaluator = EvaluationFramework()

        # Setup defense and test data
        defense = DefenseSystem()
        legit_txns = generate_legitimate_transactions(100)
        fraud_txns = generate_fraud_transactions(50)
        defense.train(legit_txns, fraud_txns)

        test_txns = generate_legitimate_transactions(50) + generate_fraud_transactions(25)

        # Evaluate
        results = evaluator.evaluate_defense_system(
            defense,
            test_txns,
            create_visualizations=False
        )

        assert "detection_metrics" in results
        assert "cost_analysis" in results
        assert results["num_transactions"] == len(test_txns)

        print("✓ Evaluation framework working")

    def test_business_impact_calculator(self):
        """Test business impact calculations."""
        from fraud_detection.types import DefenseMetrics

        calculator = BusinessImpactCalculator()

        # Create mock defense metrics
        metrics = DefenseMetrics(
            timestamp=datetime.utcnow(),
            true_positives=90,
            false_positives=10,
            true_negatives=890,
            false_negatives=10,
            precision=0.9,
            recall=0.9,
            f1_score=0.9,
            roc_auc=0.95,
            avg_detection_latency_ms=50.0,
            throughput_tps=100.0
        )

        # Calculate business impact
        business_metrics = calculator.calculate_business_impact(metrics)

        assert business_metrics.prevented_loss > 0
        assert business_metrics.net_savings is not None
        assert business_metrics.roi is not None

        # Test annual projection
        annual = calculator.calculate_annual_projection(business_metrics)
        assert "annual_net_savings" in annual

        print("✓ Business impact calculator working")

    def test_monitoring(self):
        """Test monitoring and logging."""
        logger = FraudDetectionLogger(log_dir="test_logs")
        tracker = MetricsTracker()

        # Log decision
        logger.log_fraud_decision(
            transaction_id="test_123",
            is_fraud=True,
            score=0.95,
            details={"reason": "test"}
        )

        # Track metrics
        tracker.record_prediction(is_fraud=True, score=0.95, latency_ms=45.0)
        tracker.record_prediction(is_fraud=False, score=0.2, latency_ms=50.0)

        current_metrics = tracker.get_current_metrics()
        assert current_metrics["transactions_in_window"] == 2

        print("✓ Monitoring and logging working")

    def test_end_to_end_attack_defense(self):
        """Test complete end-to-end attack and defense scenario."""
        # Initialize components
        simulator = AttackSimulator()
        defense = DefenseSystem()
        evaluator = EvaluationFramework()

        # Generate training data and train defense
        legit_txns = generate_legitimate_transactions(200)
        fraud_txns = generate_fraud_transactions(100)
        defense.train(legit_txns, fraud_txns, use_adversarial_training=True)

        # Simulate attacks
        attack_patterns = simulator.get_all_attack_patterns()[:3]  # Test first 3 attacks

        def defense_callback(txn):
            is_blocked, _, _ = defense.predict(txn)
            return is_blocked

        attack_results = []
        for pattern in attack_patterns:
            result = simulator.execute_attack(
                pattern,
                defense_callback=defense_callback
            )
            attack_results.append(result)

        # Evaluate results
        attack_metrics = evaluator.evaluate_attack_campaign(
            attack_results,
            create_visualizations=False
        )

        assert "attack_metrics" in attack_metrics
        assert attack_metrics["num_attacks"] == 3

        # Verify some attacks were blocked
        total_blocked = sum(r.blocked_transactions for r in attack_results)
        assert total_blocked > 0

        print("✓ End-to-end attack-defense scenario working")
        print(f"  Blocked {total_blocked} out of {sum(r.total_transactions for r in attack_results)} attack transactions")


def run_all_tests():
    """Run all tests manually."""
    test_suite = TestCompleteSystem()

    print("\n" + "=" * 80)
    print("RUNNING FRAUD DETECTION SYSTEM TESTS")
    print("=" * 80 + "\n")

    tests = [
        ("Attack Simulator", test_suite.test_attack_simulator),
        ("Defense System", test_suite.test_defense_system),
        ("Graph Fraud Detection", test_suite.test_graph_fraud_detection),
        ("Adversarial Learning", test_suite.test_adversarial_learning),
        ("Evaluation Framework", test_suite.test_evaluation_framework),
        ("Business Impact Calculator", test_suite.test_business_impact_calculator),
        ("Monitoring", test_suite.test_monitoring),
        ("End-to-End", test_suite.test_end_to_end_attack_defense),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\nTesting {test_name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed out of {len(tests)} total")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()
