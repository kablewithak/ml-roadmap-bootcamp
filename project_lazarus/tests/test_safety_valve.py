"""Tests for the safety valve compliance layer."""

import pytest
from src.core.safety_valve import SafetyValve, is_legally_prohibited
from src.core.models import UserFeatures


@pytest.fixture
def safety_valve():
    """Create a safety valve instance."""
    return SafetyValve()


@pytest.fixture
def compliant_user():
    """Create a fully compliant user."""
    return UserFeatures(
        user_id="compliant_123",
        age=30,
        income=60000.0,
        debt=20000.0,
        credit_score=700,
        employment_years=3.0,
        recent_bankruptcy=False,
        num_credit_lines=2,
        avg_txn_amt_30d=200.0,
        credit_history_months=60
    )


class TestSafetyValve:
    """Test safety valve rule checks."""

    def test_compliant_user_passes(self, safety_valve, compliant_user):
        """Test that a compliant user passes all checks."""
        check = safety_valve.check_all_rules(compliant_user)
        assert check.passed is True
        assert check.reason is None

    def test_age_rule_blocks_minors(self, safety_valve):
        """Test that users under 18 are blocked."""
        minor = UserFeatures(
            user_id="minor_123",
            age=17,
            income=20000.0,
            debt=0.0,
            credit_score=650,
            employment_years=0.0,
            recent_bankruptcy=False,
            num_credit_lines=0,
            avg_txn_amt_30d=50.0,
            credit_history_months=0
        )
        check = safety_valve.check_all_rules(minor)
        assert check.passed is False
        assert "18 years old" in check.reason

    def test_bankruptcy_rule(self, safety_valve):
        """Test that recent bankruptcy blocks approval."""
        bankrupt = UserFeatures(
            user_id="bankrupt_123",
            age=40,
            income=50000.0,
            debt=5000.0,
            credit_score=550,
            employment_years=5.0,
            recent_bankruptcy=True,
            num_credit_lines=1,
            avg_txn_amt_30d=100.0,
            credit_history_months=120
        )
        check = safety_valve.check_all_rules(bankrupt)
        assert check.passed is False
        assert "bankruptcy" in check.reason.lower()

    def test_dti_rule(self, safety_valve):
        """Test debt-to-income ratio check."""
        high_dti = UserFeatures(
            user_id="high_dti_123",
            age=35,
            income=50000.0,
            debt=40000.0,  # 80% DTI
            credit_score=680,
            employment_years=4.0,
            recent_bankruptcy=False,
            num_credit_lines=5,
            avg_txn_amt_30d=150.0,
            credit_history_months=80
        )
        check = safety_valve.check_all_rules(high_dti)
        assert check.passed is False
        assert "debt-to-income" in check.reason.lower()

    def test_minimum_income_rule(self, safety_valve):
        """Test minimum income requirement."""
        low_income = UserFeatures(
            user_id="low_income_123",
            age=25,
            income=10000.0,  # Below $12k minimum
            debt=0.0,
            credit_score=700,
            employment_years=1.0,
            recent_bankruptcy=False,
            num_credit_lines=1,
            avg_txn_amt_30d=30.0,
            credit_history_months=24
        )
        check = safety_valve.check_all_rules(low_income)
        assert check.passed is False
        assert "income below minimum" in check.reason.lower()

    def test_is_legally_prohibited_convenience_function(self, compliant_user):
        """Test the convenience function for prohibition check."""
        assert is_legally_prohibited(compliant_user) is False

        minor = UserFeatures(
            user_id="minor_123",
            age=16,
            income=15000.0,
            debt=0.0,
            credit_score=600,
            employment_years=0.0,
            recent_bankruptcy=False,
            num_credit_lines=0,
            avg_txn_amt_30d=0.0,
            credit_history_months=0
        )
        assert is_legally_prohibited(minor) is True


class TestHighRiskExploration:
    """Test high-risk exploration detection."""

    def test_low_credit_score_is_high_risk(self, safety_valve):
        """Test that very low credit score flags high risk."""
        user = UserFeatures(
            user_id="low_credit_123",
            age=30,
            income=50000.0,
            debt=10000.0,
            credit_score=500,  # Below 550 threshold
            employment_years=3.0,
            recent_bankruptcy=False,
            num_credit_lines=2,
            avg_txn_amt_30d=100.0,
            credit_history_months=60
        )
        assert safety_valve.is_high_risk_exploration(user) is True

    def test_high_debt_is_high_risk(self, safety_valve):
        """Test that very high debt flags high risk."""
        user = UserFeatures(
            user_id="high_debt_123",
            age=45,
            income=100000.0,
            debt=150000.0,  # Over $100k threshold
            credit_score=700,
            employment_years=10.0,
            recent_bankruptcy=False,
            num_credit_lines=8,
            avg_txn_amt_30d=500.0,
            credit_history_months=200
        )
        assert safety_valve.is_high_risk_exploration(user) is True

    def test_normal_user_not_high_risk(self, safety_valve, compliant_user):
        """Test that normal users are not flagged as high risk."""
        assert safety_valve.is_high_risk_exploration(compliant_user) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
