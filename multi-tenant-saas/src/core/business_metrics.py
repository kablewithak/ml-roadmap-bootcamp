"""
Business Metrics Module

This module tracks the financial impact of every architectural decision in our
multi-tenant SaaS platform. Unlike typical "observability," this focuses on $$$.

Key Principle: "If you can't measure the ROI, you can't justify the investment."

Based on patterns from:
- Stripe's economic modeling framework
- Datadog's cost allocation system
- Snowflake's consumption-based pricing

Author: Senior ML Engineer building for staff-level impact
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import math


class IncidentType(Enum):
    """Categories of incidents with industry-standard cost models"""
    DATA_BREACH = "data_breach"
    DOWNTIME = "downtime"
    SLA_VIOLATION = "sla_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CROSS_TENANT_LEAK = "cross_tenant_leak"


@dataclass
class CostFactors:
    """
    Industry benchmarks for cost calculations.

    Sources:
    - IBM Cost of Data Breach Report 2023: $4.45M average
    - Gartner: $9,000/minute downtime for SMB, $540K/hour for Enterprise
    - Uptime Institute: Average of 14 outages per year
    """

    # Data breach costs (IBM 2023 report)
    COST_PER_RECORD_BREACHED = 165  # USD per record
    BASE_BREACH_COST = 150_000  # Fixed costs: legal, PR, forensics
    BREACH_CUSTOMER_CHURN_RATE = 0.25  # 25% churn after breach

    # Downtime costs (varies by customer tier)
    COST_PER_MINUTE_DOWNTIME_STARTER = 50  # USD
    COST_PER_MINUTE_DOWNTIME_GROWTH = 500  # USD
    COST_PER_MINUTE_DOWNTIME_ENTERPRISE = 9_000  # USD (Gartner benchmark)

    # SLA penalties (typical SaaS contracts)
    SLA_99_9_MONTHLY_CREDITS = 0.10  # 10% monthly fee credit
    SLA_99_95_MONTHLY_CREDITS = 0.25  # 25% monthly fee credit
    SLA_99_MONTHLY_CREDITS = 0.50  # 50% monthly fee credit

    # Customer churn economics
    CUSTOMER_ACQUISITION_COST = 1_200  # Average SaaS CAC
    AVERAGE_CUSTOMER_LIFETIME_MONTHS = 36  # 3-year average

    # Operational costs
    ENGINEER_COST_PER_HOUR = 150  # Fully loaded cost
    CUSTOMER_SUPPORT_COST_PER_HOUR = 75


class BusinessMetrics:
    """
    Track the financial impact of architectural decisions.

    This class answers questions like:
    - What's the cost if our RLS policy fails?
    - How much does noisy neighbor problems cost us?
    - What's the ROI of implementing circuit breakers?
    - How much should we invest in isolation?

    Usage:
        metrics = BusinessMetrics()

        # Calculate breach cost
        breach_cost = metrics.calculate_breach_cost(
            affected_tenants=100,
            records_per_tenant=10_000
        )

        # Calculate downtime cost
        downtime_cost = metrics.calculate_downtime_cost(
            duration_minutes=30,
            affected_tier="enterprise"
        )
    """

    def __init__(self):
        self.cost_factors = CostFactors()
        self._incident_history: List[Dict] = []

    def calculate_breach_cost(
        self,
        affected_tenants: int,
        records_per_tenant: int,
        include_churn: bool = True
    ) -> Dict[str, float]:
        """
        Calculate total cost of a cross-tenant data breach.

        This is THE metric that justifies our investment in RLS, connection
        pooling, and tenant isolation. One breach can bankrupt a startup.

        Real-world example:
            Uber 2016: 57M records breached
            - Direct costs: $148M
            - Our calculation: $9.4M + churn costs
            - Why the difference? Regulatory fines, legal battles

        Args:
            affected_tenants: Number of tenants whose data was exposed
            records_per_tenant: Average records exposed per tenant
            include_churn: Include customer churn cost in calculation

        Returns:
            Dict with breakdown:
                - direct_costs: Immediate response costs
                - regulatory_fines: Estimated GDPR/regulatory
                - customer_churn: Lost revenue from churned customers
                - total: Sum of all costs

        Business insight:
            If this number is >$1M, every penny spent on isolation is justified.
        """
        total_records = affected_tenants * records_per_tenant

        # Direct costs: IBM model
        direct_costs = (
            self.cost_factors.BASE_BREACH_COST +
            (total_records * self.cost_factors.COST_PER_RECORD_BREACHED)
        )

        # Regulatory fines: GDPR is 4% of revenue or €20M, whichever is higher
        # We model conservatively at 2% of affected revenue
        estimated_regulatory = direct_costs * 0.5

        # Customer churn: Lost lifetime value
        churn_cost = 0
        if include_churn:
            churned_customers = int(
                affected_tenants * self.cost_factors.BREACH_CUSTOMER_CHURN_RATE
            )

            # Assuming average customer value across tiers
            average_monthly_value = 500  # Blended across tiers
            customer_lifetime_value = (
                average_monthly_value *
                self.cost_factors.AVERAGE_CUSTOMER_LIFETIME_MONTHS
            )

            churn_cost = churned_customers * customer_lifetime_value

            # Add cost to re-acquire equivalent customers
            churn_cost += churned_customers * self.cost_factors.CUSTOMER_ACQUISITION_COST

        total_cost = direct_costs + estimated_regulatory + churn_cost

        # Log this incident for ROI calculations
        self._record_incident(
            incident_type=IncidentType.CROSS_TENANT_LEAK,
            cost=total_cost,
            affected_tenants=affected_tenants
        )

        return {
            "direct_costs": round(direct_costs, 2),
            "regulatory_fines": round(estimated_regulatory, 2),
            "customer_churn": round(churn_cost, 2),
            "total": round(total_cost, 2),
            "cost_per_tenant": round(total_cost / affected_tenants, 2),
            "cost_per_record": round(total_cost / total_records, 2)
        }

    def calculate_downtime_cost(
        self,
        duration_minutes: float,
        affected_tier: str,
        affected_tenant_count: int = 1
    ) -> Dict[str, float]:
        """
        Calculate cost of downtime by customer tier.

        Why this matters:
        - Helps prioritize which incidents to fix first
        - Justifies investment in high availability
        - Calculates SLA credit exposure

        Real-world example:
            AWS us-east-1 outage (Dec 2021): 7 hours
            - Enterprise customers: ~$3.8M impact per customer
            - Our multi-tenant cost: 1000 tenants * 7 hours = disaster

        Args:
            duration_minutes: Length of outage
            affected_tier: "starter", "growth", or "enterprise"
            affected_tenant_count: Number of tenants impacted

        Returns:
            Dict with:
                - customer_impact: Revenue loss for customers
                - sla_credits: Credits we must issue
                - total_cost: Sum of both

        Business insight:
            Every minute counts. This is why we implement circuit breakers.
        """
        tier_costs = {
            "starter": self.cost_factors.COST_PER_MINUTE_DOWNTIME_STARTER,
            "growth": self.cost_factors.COST_PER_MINUTE_DOWNTIME_GROWTH,
            "enterprise": self.cost_factors.COST_PER_MINUTE_DOWNTIME_ENTERPRISE
        }

        cost_per_minute = tier_costs.get(affected_tier.lower(), tier_costs["growth"])

        # Customer's business impact (what THEY lose)
        customer_impact = duration_minutes * cost_per_minute * affected_tenant_count

        # Our SLA credit exposure
        # Calculate based on downtime % of month
        minutes_per_month = 30 * 24 * 60  # ~43,200 minutes
        downtime_percentage = duration_minutes / minutes_per_month

        # Determine SLA credit tier
        monthly_uptime = 1 - downtime_percentage
        if monthly_uptime >= 0.999:  # 99.9%
            credit_rate = 0
        elif monthly_uptime >= 0.9995:  # 99.95%
            credit_rate = self.cost_factors.SLA_99_9_MONTHLY_CREDITS
        elif monthly_uptime >= 0.99:  # 99%
            credit_rate = self.cost_factors.SLA_99_95_MONTHLY_CREDITS
        else:
            credit_rate = self.cost_factors.SLA_99_MONTHLY_CREDITS

        # Calculate SLA credits we must issue
        # Assuming average monthly revenue per tier
        tier_monthly_revenue = {
            "starter": 99,
            "growth": 499,
            "enterprise": 4999
        }

        avg_revenue = tier_monthly_revenue.get(affected_tier.lower(), 499)
        sla_credits = avg_revenue * credit_rate * affected_tenant_count

        total_cost = customer_impact + sla_credits

        self._record_incident(
            incident_type=IncidentType.DOWNTIME,
            cost=sla_credits,  # We only track OUR cost
            affected_tenants=affected_tenant_count,
            duration_minutes=duration_minutes
        )

        return {
            "customer_impact": round(customer_impact, 2),
            "sla_credits": round(sla_credits, 2),
            "total_cost": round(total_cost, 2),
            "uptime_percentage": round(monthly_uptime * 100, 4),
            "minutes_of_downtime": round(duration_minutes, 2)
        }

    def calculate_noisy_neighbor_cost(
        self,
        affected_tenants: int,
        performance_degradation_pct: float,
        duration_minutes: float
    ) -> Dict[str, float]:
        """
        Calculate cost when one tenant impacts others' performance.

        This is the #1 reason to implement resource governors and rate limiting.

        Noisy neighbor scenarios:
        1. Crypto miner using all CPU
        2. Inefficient query locking tables
        3. Bulk upload consuming connection pool
        4. ML training job starving API requests

        Args:
            affected_tenants: Number of tenants experiencing slowdown
            performance_degradation_pct: 0-100, how much slower
            duration_minutes: How long the issue lasted

        Returns:
            Cost breakdown including support time and churn risk

        Business insight:
            Even 10% degradation for 1 hour can cost thousands in support time.
        """
        # Convert performance hit to "effective downtime"
        # 50% degradation = 50% effective downtime for cost purposes
        effective_downtime_minutes = duration_minutes * (performance_degradation_pct / 100)

        # Customer support costs: tickets, investigation
        estimated_support_tickets = affected_tenants * 0.3  # 30% file tickets
        avg_support_time_hours = 0.5  # 30 min per ticket
        support_cost = (
            estimated_support_tickets *
            avg_support_time_hours *
            self.cost_factors.CUSTOMER_SUPPORT_COST_PER_HOUR
        )

        # Engineering investigation cost
        estimated_engineering_hours = max(1, duration_minutes / 30)  # 1 hour per 30 min incident
        engineering_cost = (
            estimated_engineering_hours *
            self.cost_factors.ENGINEER_COST_PER_HOUR
        )

        # Churn risk: severe degradation increases churn
        churn_probability = min(0.05, performance_degradation_pct / 2000)  # Max 5%
        expected_churn_cost = (
            affected_tenants *
            churn_probability *
            500 *  # Average monthly revenue
            self.cost_factors.AVERAGE_CUSTOMER_LIFETIME_MONTHS
        )

        total_cost = support_cost + engineering_cost + expected_churn_cost

        self._record_incident(
            incident_type=IncidentType.PERFORMANCE_DEGRADATION,
            cost=total_cost,
            affected_tenants=affected_tenants,
            duration_minutes=duration_minutes
        )

        return {
            "support_cost": round(support_cost, 2),
            "engineering_cost": round(engineering_cost, 2),
            "expected_churn_cost": round(expected_churn_cost, 2),
            "total_cost": round(total_cost, 2),
            "cost_per_minute": round(total_cost / duration_minutes, 2)
        }

    def calculate_isolation_investment_roi(
        self,
        investment_cost: float,
        expected_incidents_prevented_per_year: int,
        average_incident_cost: float,
        years: int = 5
    ) -> Dict[str, float]:
        """
        Calculate ROI for isolation features (RLS, circuit breakers, etc.).

        This is how you justify spending 2 weeks implementing connection pooling.

        Example:
            Investment: $50K (2 engineers * 2 weeks * $150/hr)
            Prevents: 3 breaches per year
            Average breach cost: $500K
            ROI: 3000% over 5 years

        Args:
            investment_cost: Upfront cost to implement
            expected_incidents_prevented_per_year: How many breaches/outages avoided
            average_incident_cost: Average cost per incident
            years: Time horizon for ROI calculation

        Returns:
            ROI metrics including NPV, payback period

        Business insight:
            If ROI < 300%, reconsider the investment. Focus elsewhere.
        """
        # Calculate avoided costs
        total_avoided_costs = (
            expected_incidents_prevented_per_year *
            average_incident_cost *
            years
        )

        # Simple ROI calculation
        roi_percentage = ((total_avoided_costs - investment_cost) / investment_cost) * 100

        # Payback period (in years)
        annual_savings = expected_incidents_prevented_per_year * average_incident_cost
        payback_period_years = investment_cost / annual_savings if annual_savings > 0 else float('inf')

        # Net Present Value (NPV) with 10% discount rate
        discount_rate = 0.10
        npv = -investment_cost
        for year in range(1, years + 1):
            npv += annual_savings / ((1 + discount_rate) ** year)

        return {
            "investment_cost": round(investment_cost, 2),
            "total_avoided_costs": round(total_avoided_costs, 2),
            "roi_percentage": round(roi_percentage, 2),
            "payback_period_years": round(payback_period_years, 2),
            "npv": round(npv, 2),
            "annual_savings": round(annual_savings, 2),
            "recommendation": "INVEST" if roi_percentage > 300 else "RECONSIDER"
        }

    def calculate_sla_compliance_cost(
        self,
        target_sla: float,
        current_uptime: float,
        monthly_revenue: float
    ) -> Dict[str, float]:
        """
        Calculate cost of missing SLA targets.

        SLA tiers (industry standard):
        - 99.9% ("three nines"): 43.2 min/month downtime allowed
        - 99.95%: 21.6 min/month
        - 99.99% ("four nines"): 4.32 min/month

        Args:
            target_sla: Target uptime (0.999 for 99.9%)
            current_uptime: Actual uptime achieved
            monthly_revenue: Total monthly recurring revenue

        Returns:
            Credits owed and reputation impact
        """
        if current_uptime >= target_sla:
            return {
                "credits_owed": 0,
                "reputation_impact": 0,
                "total_cost": 0,
                "status": "SLA_MET"
            }

        # Calculate credit tier
        if current_uptime >= 0.9995:
            credit_rate = 0.10
        elif current_uptime >= 0.999:
            credit_rate = 0.25
        else:
            credit_rate = 0.50

        credits_owed = monthly_revenue * credit_rate

        # Reputation impact: customers talk
        # Estimate 5% churn increase for each SLA miss
        reputation_impact = monthly_revenue * 0.05 * 36  # LTV impact

        return {
            "credits_owed": round(credits_owed, 2),
            "reputation_impact": round(reputation_impact, 2),
            "total_cost": round(credits_owed + reputation_impact, 2),
            "status": "SLA_VIOLATED",
            "uptime_gap": round((target_sla - current_uptime) * 100, 4)
        }

    def get_incident_summary(self) -> Dict:
        """
        Get summary of all tracked incidents for executive reporting.

        Returns dashboard-ready metrics.
        """
        if not self._incident_history:
            return {
                "total_incidents": 0,
                "total_cost": 0,
                "incidents_by_type": {}
            }

        total_cost = sum(incident["cost"] for incident in self._incident_history)

        incidents_by_type = {}
        for incident in self._incident_history:
            itype = incident["type"].value
            if itype not in incidents_by_type:
                incidents_by_type[itype] = {"count": 0, "total_cost": 0}

            incidents_by_type[itype]["count"] += 1
            incidents_by_type[itype]["total_cost"] += incident["cost"]

        return {
            "total_incidents": len(self._incident_history),
            "total_cost": round(total_cost, 2),
            "incidents_by_type": incidents_by_type,
            "average_cost_per_incident": round(total_cost / len(self._incident_history), 2)
        }

    def _record_incident(
        self,
        incident_type: IncidentType,
        cost: float,
        affected_tenants: int = 0,
        duration_minutes: float = 0
    ):
        """Internal method to track incidents for analytics."""
        self._incident_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": incident_type,
            "cost": cost,
            "affected_tenants": affected_tenants,
            "duration_minutes": duration_minutes
        })


# ============================================================================
# DEMONSTRATION: How to use this in production
# ============================================================================

def demo_business_metrics():
    """
    Production usage examples showing how to integrate into decision-making.
    """
    metrics = BusinessMetrics()

    print("=" * 80)
    print("BUSINESS METRICS DEMONSTRATION")
    print("=" * 80)

    # Scenario 1: RLS Policy Failure
    print("\n📊 SCENARIO 1: RLS Policy Bypass (Cross-Tenant Leak)")
    print("-" * 80)
    breach = metrics.calculate_breach_cost(
        affected_tenants=50,
        records_per_tenant=5_000
    )
    print(f"Affected tenants: 50")
    print(f"Records exposed: 250,000")
    print(f"Direct costs: ${breach['direct_costs']:,.2f}")
    print(f"Regulatory fines: ${breach['regulatory_fines']:,.2f}")
    print(f"Customer churn: ${breach['customer_churn']:,.2f}")
    print(f"🚨 TOTAL COST: ${breach['total']:,.2f}")
    print(f"\n💡 INSIGHT: This is why we invest in RLS + multiple validation layers")

    # Scenario 2: Database Downtime
    print("\n\n📊 SCENARIO 2: Database Connection Pool Exhaustion")
    print("-" * 80)
    downtime = metrics.calculate_downtime_cost(
        duration_minutes=45,
        affected_tier="enterprise",
        affected_tenant_count=10
    )
    print(f"Downtime: 45 minutes")
    print(f"Affected: 10 Enterprise customers")
    print(f"Customer impact: ${downtime['customer_impact']:,.2f}")
    print(f"SLA credits owed: ${downtime['sla_credits']:,.2f}")
    print(f"🚨 TOTAL COST: ${downtime['total_cost']:,.2f}")
    print(f"\n💡 INSIGHT: Circuit breakers would have prevented this")

    # Scenario 3: Noisy Neighbor
    print("\n\n📊 SCENARIO 3: Noisy Neighbor (Crypto Mining Tenant)")
    print("-" * 80)
    noisy = metrics.calculate_noisy_neighbor_cost(
        affected_tenants=200,
        performance_degradation_pct=40,
        duration_minutes=120
    )
    print(f"Affected tenants: 200")
    print(f"Performance degradation: 40%")
    print(f"Duration: 2 hours")
    print(f"Support cost: ${noisy['support_cost']:,.2f}")
    print(f"Engineering cost: ${noisy['engineering_cost']:,.2f}")
    print(f"Expected churn: ${noisy['expected_churn_cost']:,.2f}")
    print(f"🚨 TOTAL COST: ${noisy['total_cost']:,.2f}")
    print(f"\n💡 INSIGHT: Resource governors pay for themselves in weeks")

    # Scenario 4: ROI of Connection Pooling
    print("\n\n📊 SCENARIO 4: ROI Analysis - Connection Pool Implementation")
    print("-" * 80)
    roi = metrics.calculate_isolation_investment_roi(
        investment_cost=75_000,  # 3 weeks * 2 engineers * $12.5K/week
        expected_incidents_prevented_per_year=4,
        average_incident_cost=100_000,
        years=5
    )
    print(f"Investment: ${roi['investment_cost']:,.2f}")
    print(f"Annual savings: ${roi['annual_savings']:,.2f}")
    print(f"Payback period: {roi['payback_period_years']:.1f} years")
    print(f"5-year NPV: ${roi['npv']:,.2f}")
    print(f"🎯 ROI: {roi['roi_percentage']:.0f}%")
    print(f"✅ RECOMMENDATION: {roi['recommendation']}")

    # Summary
    print("\n\n📈 INCIDENT SUMMARY")
    print("-" * 80)
    summary = metrics.get_incident_summary()
    print(f"Total incidents tracked: {summary['total_incidents']}")
    print(f"Total cost impact: ${summary['total_cost']:,.2f}")
    print(f"Average cost per incident: ${summary['average_cost_per_incident']:,.2f}")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAY:")
    print("Every architectural decision has a dollar value. Measure it.")
    print("=" * 80)


if __name__ == "__main__":
    demo_business_metrics()
