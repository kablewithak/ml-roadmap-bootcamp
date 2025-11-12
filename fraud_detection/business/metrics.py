"""Business impact and ROI calculations."""

from typing import Dict, Any, List
from datetime import datetime
from fraud_detection.types import DefenseMetrics, BusinessMetrics


class BusinessImpactCalculator:
    """
    Calculate business impact and ROI of fraud detection system.

    Quantifies:
    - Prevented losses
    - Cost of false positives (customer friction)
    - Operational costs
    - Net savings and ROI
    - Customer lifetime value impact
    """

    def __init__(
        self,
        avg_fraud_amount: float = 250.0,
        fp_customer_friction_cost: float = 5.0,
        fn_fraud_loss: float = 250.0,
        operational_cost_per_txn: float = 0.01,
        customer_lifetime_value: float = 1000.0,
        customer_churn_rate_per_fp: float = 0.05
    ):
        """
        Initialize with business parameters.

        Args:
            avg_fraud_amount: Average fraudulent transaction amount
            fp_customer_friction_cost: Cost per false positive (customer service, etc.)
            fn_fraud_loss: Average loss per false negative
            operational_cost_per_txn: Cost to run system per transaction
            customer_lifetime_value: Average customer lifetime value
            customer_churn_rate_per_fp: Probability customer churns due to false positive
        """
        self.avg_fraud_amount = avg_fraud_amount
        self.fp_customer_friction_cost = fp_customer_friction_cost
        self.fn_fraud_loss = fn_fraud_loss
        self.operational_cost_per_txn = operational_cost_per_txn
        self.customer_lifetime_value = customer_lifetime_value
        self.customer_churn_rate_per_fp = customer_churn_rate_per_fp

    def calculate_business_impact(
        self,
        defense_metrics: DefenseMetrics
    ) -> BusinessMetrics:
        """
        Calculate comprehensive business impact.

        Args:
            defense_metrics: Defense system metrics

        Returns:
            BusinessMetrics with financial impact
        """
        tp = defense_metrics.true_positives
        fp = defense_metrics.false_positives
        tn = defense_metrics.true_negatives
        fn = defense_metrics.false_negatives

        total_transactions = tp + fp + tn + fn

        # Prevented loss (true positives caught)
        prevented_loss = tp * self.avg_fraud_amount

        # False positive cost
        # 1. Direct customer service cost
        fp_direct_cost = fp * self.fp_customer_friction_cost

        # 2. Customer churn cost (some customers leave due to false blocks)
        expected_churned_customers = fp * self.customer_churn_rate_per_fp
        customer_churn_cost = expected_churned_customers * self.customer_lifetime_value

        total_fp_cost = fp_direct_cost + customer_churn_cost

        # False negative cost (fraud that got through)
        fn_cost = fn * self.fn_fraud_loss

        # Operational cost
        operational_cost = total_transactions * self.operational_cost_per_txn

        # Total cost
        total_cost = total_fp_cost + fn_cost + operational_cost

        # Net savings
        net_savings = prevented_loss - total_cost

        # ROI
        roi = (net_savings / total_cost * 100) if total_cost > 0 else 0

        # Cost per transaction
        cost_per_transaction = total_cost / total_transactions if total_transactions > 0 else 0

        # Customer lifetime value impact
        clv_impact = -customer_churn_cost  # Negative because we're losing customers

        return BusinessMetrics(
            timestamp=datetime.utcnow(),
            prevented_loss=prevented_loss,
            false_positive_cost=total_fp_cost,
            operational_cost=operational_cost,
            net_savings=net_savings,
            roi=roi,
            customer_lifetime_value_impact=clv_impact,
            transactions_processed=total_transactions,
            cost_per_transaction=cost_per_transaction
        )

    def calculate_annual_projection(
        self,
        current_metrics: BusinessMetrics,
        transactions_per_day: int = 10000
    ) -> Dict[str, Any]:
        """
        Project annual business impact.

        Args:
            current_metrics: Current business metrics
            transactions_per_day: Expected daily transaction volume

        Returns:
            Annual projections
        """
        days_per_year = 365
        annual_transactions = transactions_per_day * days_per_year

        # Scale up current metrics
        scale_factor = annual_transactions / current_metrics.transactions_processed if current_metrics.transactions_processed > 0 else 1

        annual_prevented_loss = current_metrics.prevented_loss * scale_factor
        annual_fp_cost = current_metrics.false_positive_cost * scale_factor
        annual_operational_cost = current_metrics.operational_cost * scale_factor
        annual_net_savings = current_metrics.net_savings * scale_factor

        return {
            "annual_transactions": annual_transactions,
            "annual_prevented_loss": annual_prevented_loss,
            "annual_false_positive_cost": annual_fp_cost,
            "annual_operational_cost": annual_operational_cost,
            "annual_net_savings": annual_net_savings,
            "annual_roi": current_metrics.roi,
            "summary": f"System saves ${annual_net_savings:,.0f} annually at {transactions_per_day:,} transactions/day"
        }

    def compare_to_baseline(
        self,
        current_metrics: BusinessMetrics,
        baseline_fraud_rate: float = 0.02,
        baseline_detection_rate: float = 0.5
    ) -> Dict[str, Any]:
        """
        Compare to baseline (without ML system).

        Args:
            current_metrics: Current system metrics
            baseline_fraud_rate: Baseline fraud rate without system
            baseline_detection_rate: Baseline detection rate (rules only)

        Returns:
            Comparison results
        """
        # Calculate baseline metrics
        baseline_fraud_loss = (
            current_metrics.transactions_processed *
            baseline_fraud_rate *
            (1 - baseline_detection_rate) *
            self.fn_fraud_loss
        )

        # Improvement
        improvement = current_metrics.prevented_loss - baseline_fraud_loss

        return {
            "baseline_fraud_loss": baseline_fraud_loss,
            "current_prevented_loss": current_metrics.prevented_loss,
            "improvement": improvement,
            "improvement_percentage": (improvement / baseline_fraud_loss * 100) if baseline_fraud_loss > 0 else 0,
            "summary": f"ML system prevents ${improvement:,.0f} more fraud than baseline ({improvement / baseline_fraud_loss * 100:.1f}% improvement)"
        }

    def calculate_sla_compliance(
        self,
        defense_metrics: DefenseMetrics,
        target_latency_ms: float = 100.0,
        target_throughput_tps: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Calculate SLA compliance.

        Args:
            defense_metrics: Defense metrics
            target_latency_ms: Target latency in ms
            target_throughput_tps: Target throughput in TPS

        Returns:
            SLA compliance metrics
        """
        latency_compliance = defense_metrics.avg_detection_latency_ms <= target_latency_ms
        throughput_compliance = defense_metrics.throughput_tps >= target_throughput_tps

        return {
            "latency": {
                "actual_ms": defense_metrics.avg_detection_latency_ms,
                "target_ms": target_latency_ms,
                "compliant": latency_compliance,
                "margin": target_latency_ms - defense_metrics.avg_detection_latency_ms
            },
            "throughput": {
                "actual_tps": defense_metrics.throughput_tps,
                "target_tps": target_throughput_tps,
                "compliant": throughput_compliance,
                "margin": defense_metrics.throughput_tps - target_throughput_tps
            },
            "overall_compliant": latency_compliance and throughput_compliance
        }

    def generate_executive_summary(
        self,
        business_metrics: BusinessMetrics,
        defense_metrics: DefenseMetrics
    ) -> str:
        """
        Generate executive summary for stakeholders.

        Args:
            business_metrics: Business impact metrics
            defense_metrics: Defense performance metrics

        Returns:
            Executive summary text
        """
        summary = f"""
FRAUD DETECTION SYSTEM - EXECUTIVE SUMMARY
==========================================

FINANCIAL IMPACT
----------------
• Prevented Fraud Loss: ${business_metrics.prevented_loss:,.0f}
• System Costs: ${business_metrics.false_positive_cost + business_metrics.operational_cost:,.0f}
• NET SAVINGS: ${business_metrics.net_savings:,.0f}
• ROI: {business_metrics.roi:.1f}%
• Cost per Transaction: ${business_metrics.cost_per_transaction:.4f}

OPERATIONAL PERFORMANCE
-----------------------
• Transactions Processed: {business_metrics.transactions_processed:,}
• Precision (Accuracy): {defense_metrics.precision * 100:.1f}%
• Recall (Fraud Caught): {defense_metrics.recall * 100:.1f}%
• False Positive Rate: {defense_metrics.false_positive_rate * 100:.2f}%
• Average Latency: {defense_metrics.avg_detection_latency_ms:.1f}ms

KEY INSIGHTS
------------
"""

        if business_metrics.roi > 500:
            summary += "• EXCELLENT ROI - System is highly profitable\n"
        elif business_metrics.roi > 200:
            summary += "• STRONG ROI - System provides significant value\n"
        elif business_metrics.roi > 0:
            summary += "• POSITIVE ROI - System is profitable\n"
        else:
            summary += "• NEGATIVE ROI - System costs exceed benefits\n"

        if defense_metrics.precision > 0.95:
            summary += "• Minimal customer friction from false positives\n"
        elif defense_metrics.precision > 0.85:
            summary += "• Acceptable customer friction levels\n"
        else:
            summary += "• High false positives may impact customer satisfaction\n"

        if defense_metrics.recall > 0.90:
            summary += "• Excellent fraud detection coverage\n"
        elif defense_metrics.recall > 0.75:
            summary += "• Good fraud detection coverage\n"
        else:
            summary += "• Opportunity to improve fraud detection rate\n"

        return summary
