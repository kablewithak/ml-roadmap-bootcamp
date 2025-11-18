"""Tests for the policy engine."""

import pytest
from unittest.mock import MagicMock, patch
from src.core.policy import decide_application
from src.core.models import UserFeatures, Decision, DecisionReason, TreatmentGroup


@pytest.fixture
def valid_user_features():
    """Create valid user features for testing."""
    return UserFeatures(
        user_id="test_user_123",
        age=35,
        income=75000.0,
        debt=15000.0,
        credit_score=720,
        employment_years=5.0,
        recent_bankruptcy=False,
        num_credit_lines=3,
        avg_txn_amt_30d=250.0,
        credit_history_months=120
    )


@pytest.fixture
def underage_user_features():
    """Create underage user features for testing compliance."""
    return UserFeatures(
        user_id="test_minor_123",
        age=17,
        income=25000.0,
        debt=0.0,
        credit_score=680,
        employment_years=0.5,
        recent_bankruptcy=False,
        num_credit_lines=0,
        avg_txn_amt_30d=50.0,
        credit_history_months=6
    )


@pytest.fixture
def bankruptcy_user_features():
    """Create user with recent bankruptcy."""
    return UserFeatures(
        user_id="test_bankrupt_123",
        age=45,
        income=60000.0,
        debt=5000.0,
        credit_score=550,
        employment_years=10.0,
        recent_bankruptcy=True,
        num_credit_lines=1,
        avg_txn_amt_30d=100.0,
        credit_history_months=200
    )


class TestSafetyValve:
    """Test the safety valve compliance checks."""

    @patch('src.core.policy.TrafficRouter')
    def test_underage_rejection(self, mock_router, underage_user_features):
        """Test that underage applicants are rejected."""
        mock_router_instance = MagicMock()
        mock_router_instance.get_budget_status.return_value = {"remaining_budget": 1000}
        mock_router.return_value = mock_router_instance

        result = decide_application(
            user_features=underage_user_features,
            risk_score=0.3,
            application_id="app_test_001"
        )

        assert result.decision == Decision.REJECT
        assert result.reason == DecisionReason.COMPLIANCE_BLOCK

    @patch('src.core.policy.TrafficRouter')
    def test_bankruptcy_rejection(self, mock_router, bankruptcy_user_features):
        """Test that recent bankruptcy leads to rejection."""
        mock_router_instance = MagicMock()
        mock_router_instance.get_budget_status.return_value = {"remaining_budget": 1000}
        mock_router.return_value = mock_router_instance

        result = decide_application(
            user_features=bankruptcy_user_features,
            risk_score=0.4,
            application_id="app_test_002"
        )

        assert result.decision == Decision.REJECT
        assert result.reason == DecisionReason.COMPLIANCE_BLOCK


class TestExplorationLogic:
    """Test the exploration/exploitation decision logic."""

    @patch('src.core.policy.TrafficRouter')
    def test_exploration_approval(self, mock_router, valid_user_features):
        """Test that exploration forces approval."""
        mock_router_instance = MagicMock()
        mock_router_instance.should_explore.return_value = (True, TreatmentGroup.EXPLORE, 950.0)
        mock_router_instance.get_budget_status.return_value = {"remaining_budget": 950}
        mock_router.return_value = mock_router_instance

        result = decide_application(
            user_features=valid_user_features,
            risk_score=0.8,  # High risk, would normally reject
            application_id="app_test_003"
        )

        assert result.decision == Decision.APPROVE
        assert result.reason == DecisionReason.PROJECT_LAZARUS_EXPLORE
        assert result.treatment_group == TreatmentGroup.EXPLORE

    @patch('src.core.policy.TrafficRouter')
    def test_exploitation_approval(self, mock_router, valid_user_features):
        """Test standard approval in exploitation mode."""
        mock_router_instance = MagicMock()
        mock_router_instance.should_explore.return_value = (False, TreatmentGroup.EXPLOIT, 1000.0)
        mock_router_instance.get_budget_status.return_value = {"remaining_budget": 1000}
        mock_router.return_value = mock_router_instance

        result = decide_application(
            user_features=valid_user_features,
            risk_score=0.3,  # Low risk
            application_id="app_test_004"
        )

        assert result.decision == Decision.APPROVE
        assert result.reason == DecisionReason.MODEL_QUALIFIED
        assert result.treatment_group == TreatmentGroup.EXPLOIT

    @patch('src.core.policy.TrafficRouter')
    def test_exploitation_rejection(self, mock_router, valid_user_features):
        """Test standard rejection in exploitation mode."""
        mock_router_instance = MagicMock()
        mock_router_instance.should_explore.return_value = (False, TreatmentGroup.EXPLOIT, 1000.0)
        mock_router_instance.get_budget_status.return_value = {"remaining_budget": 1000}
        mock_router.return_value = mock_router_instance

        result = decide_application(
            user_features=valid_user_features,
            risk_score=0.7,  # High risk
            application_id="app_test_005"
        )

        assert result.decision == Decision.REJECT
        assert result.reason == DecisionReason.MODEL_HIGH_RISK
        assert result.treatment_group == TreatmentGroup.EXPLOIT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
