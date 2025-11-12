"""
VaR Backtesting Framework

Backtesting is mandatory for validating VaR models.

Regulatory Requirements:
- Basel III: Daily VaR backtesting required for IMA (Internal Models Approach)
- SR 11-7 (Federal Reserve): Model validation and backtesting
- MiFID II: Regular backtesting of risk models

Statistical Tests:
1. Kupiec POF Test: Tests if violation rate matches confidence level
2. Christoffersen Test: Tests independence of violations
3. Mixed Kupiec-Christoffersen: Combined test

Traffic Light System (Basel):
- Green Zone: 0-4 violations in 250 days (99% VaR) → OK
- Yellow Zone: 5-9 violations → Increase capital multiplier
- Red Zone: 10+ violations → Model rejected, use standard approach

Key Concept: VaR violations should be:
1. Infrequent (matching confidence level)
2. Independent (no clustering)
3. Random (no predictable pattern)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from datetime import datetime, timedelta

from ..utils.data_structures import BacktestResult


class VaRBacktester:
    """
    Comprehensive VaR backtesting framework.

    Tests whether VaR model is correctly specified by comparing
    predicted VaR to actual losses.
    """

    def __init__(self, confidence_level: float = 0.99):
        """
        Initialize backtester.

        Args:
            confidence_level: VaR confidence level
        """
        self.confidence_level = confidence_level
        self.expected_violation_rate = 1 - confidence_level

    def calculate_violations(
        self,
        actual_losses: np.ndarray,
        predicted_var: np.ndarray
    ) -> np.ndarray:
        """
        Identify VaR violations (actual loss > predicted VaR).

        Args:
            actual_losses: Actual realized losses (positive = loss)
            predicted_var: Predicted VaR values (positive)

        Returns:
            Binary array (1 = violation, 0 = no violation)
        """
        # Violation occurs when actual loss exceeds VaR
        violations = (actual_losses > predicted_var).astype(int)
        return violations

    def kupiec_pof_test(
        self,
        violations: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Kupiec Proportion of Failures (POF) test.

        Tests null hypothesis: H0: p = p_expected (violation rate is correct)

        Test statistic:
        LR = -2 * ln[(p_expected^N * (1-p_expected)^(T-N)) /
                     (p_observed^N * (1-p_observed)^(T-N))]

        Under H0, LR ~ χ²(1)

        Intuition:
        - Compares observed vs expected violation rate
        - Uses likelihood ratio test
        - Doesn't check if violations are clustered

        Args:
            violations: Binary array of violations
            alpha: Significance level

        Returns:
            Dict with test statistic, p-value, and decision
        """
        T = len(violations)
        N = np.sum(violations)  # Number of violations

        p_expected = self.expected_violation_rate
        p_observed = N / T

        # Likelihood ratio test statistic
        if N == 0 or N == T:
            # Edge cases
            if N == 0 and p_expected > 0:
                lr_stat = -2 * T * np.log(1 - p_expected)
            elif N == T:
                lr_stat = -2 * T * np.log(p_expected)
            else:
                lr_stat = 0
        else:
            lr_stat = -2 * (
                N * np.log(p_expected / p_observed) +
                (T - N) * np.log((1 - p_expected) / (1 - p_observed))
            )

        # P-value from chi-squared distribution (df=1)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

        # Decision
        reject_h0 = p_value < alpha

        return {
            'test_statistic': lr_stat,
            'p_value': p_value,
            'reject_h0': reject_h0,
            'conclusion': 'Model rejected' if reject_h0 else 'Model accepted',
            'n_violations': int(N),
            'violation_rate': p_observed,
            'expected_rate': p_expected
        }

    def christoffersen_independence_test(
        self,
        violations: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Christoffersen Independence test.

        Tests if violations are independent (no clustering).

        H0: Violations occur independently
        H1: Violations are clustered (serial correlation)

        Method:
        - Construct 2x2 transition matrix for violations
        - Test if P(violation_t | violation_{t-1}) = P(violation_t | no_violation_{t-1})

        Important: VaR model can have correct coverage but still fail if
        violations cluster (model underestimates risk in stress periods).

        Args:
            violations: Binary violation array
            alpha: Significance level

        Returns:
            Dict with test results
        """
        # Build 2x2 transition matrix
        # State 0 = no violation, State 1 = violation
        n_00 = 0  # no violation → no violation
        n_01 = 0  # no violation → violation
        n_10 = 0  # violation → no violation
        n_11 = 0  # violation → violation

        for t in range(len(violations) - 1):
            if violations[t] == 0 and violations[t+1] == 0:
                n_00 += 1
            elif violations[t] == 0 and violations[t+1] == 1:
                n_01 += 1
            elif violations[t] == 1 and violations[t+1] == 0:
                n_10 += 1
            elif violations[t] == 1 and violations[t+1] == 1:
                n_11 += 1

        # Calculate transition probabilities
        n_0 = n_00 + n_01  # Total no violations at t
        n_1 = n_10 + n_11  # Total violations at t

        if n_0 == 0 or n_1 == 0:
            # Not enough data
            return {
                'test_statistic': 0.0,
                'p_value': 1.0,
                'reject_h0': False,
                'conclusion': 'Insufficient violations for independence test'
            }

        # Conditional probabilities
        p_01 = n_01 / n_0 if n_0 > 0 else 0  # P(viol | no viol)
        p_11 = n_11 / n_1 if n_1 > 0 else 0  # P(viol | viol)

        # Overall violation probability
        p = (n_01 + n_11) / (n_0 + n_1)

        # Likelihood ratio test
        if p_01 == 0 or p_11 == 0 or p == 0 or p_01 == 1 or p_11 == 1 or p == 1:
            lr_stat = 0.0
        else:
            lr_stat = -2 * (
                n_00 * np.log((1-p) / (1-p_01)) +
                n_01 * np.log(p / p_01) +
                n_10 * np.log((1-p) / (1-p_11)) +
                n_11 * np.log(p / p_11)
            )

        # P-value (chi-squared, df=1)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

        # Decision
        reject_h0 = p_value < alpha

        return {
            'test_statistic': lr_stat,
            'p_value': p_value,
            'reject_h0': reject_h0,
            'conclusion': 'Violations are clustered' if reject_h0 else 'Violations are independent',
            'p_01': p_01,
            'p_11': p_11,
            'clustering_ratio': p_11 / p_01 if p_01 > 0 else None
        }

    def christoffersen_combined_test(
        self,
        violations: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Combined Christoffersen test (coverage + independence).

        Tests both:
        1. Correct violation rate (Kupiec)
        2. Independence of violations

        LR_combined = LR_kupiec + LR_independence

        Under H0, LR_combined ~ χ²(2)

        Model must pass BOTH tests to be considered valid.

        Args:
            violations: Binary violation array
            alpha: Significance level

        Returns:
            Dict with combined test results
        """
        # Run individual tests
        kupiec = self.kupiec_pof_test(violations, alpha=alpha)
        independence = self.christoffersen_independence_test(violations, alpha=alpha)

        # Combined test statistic
        lr_combined = kupiec['test_statistic'] + independence['test_statistic']

        # P-value (chi-squared, df=2)
        p_value = 1 - stats.chi2.cdf(lr_combined, df=2)

        # Decision
        reject_h0 = p_value < alpha

        return {
            'test_statistic': lr_combined,
            'p_value': p_value,
            'reject_h0': reject_h0,
            'conclusion': 'Model rejected' if reject_h0 else 'Model accepted',
            'kupiec_statistic': kupiec['test_statistic'],
            'independence_statistic': independence['test_statistic'],
            'kupiec_passed': not kupiec['reject_h0'],
            'independence_passed': not independence['reject_h0']
        }

    def calculate_basel_zone(
        self,
        n_violations: int,
        n_observations: int = 250
    ) -> str:
        """
        Calculate Basel traffic light zone.

        Basel Committee thresholds (for 99% VaR, 250 days):
        - Green: 0-4 violations
        - Yellow: 5-9 violations
        - Red: 10+ violations

        Args:
            n_violations: Number of VaR violations
            n_observations: Number of observations (typically 250)

        Returns:
            Zone color ('green', 'yellow', 'red')
        """
        if n_observations == 250 and self.confidence_level == 0.99:
            # Standard Basel zones
            if n_violations <= 4:
                return 'green'
            elif n_violations <= 9:
                return 'yellow'
            else:
                return 'red'
        else:
            # Generalized zones based on binomial distribution
            expected_violations = n_observations * (1 - self.confidence_level)

            # Green: within 1.5 standard deviations
            # Yellow: 1.5 to 2.5 standard deviations
            # Red: > 2.5 standard deviations

            std_violations = np.sqrt(
                n_observations * (1 - self.confidence_level) * self.confidence_level
            )

            if n_violations <= expected_violations + 1.5 * std_violations:
                return 'green'
            elif n_violations <= expected_violations + 2.5 * std_violations:
                return 'yellow'
            else:
                return 'red'

    def backtest(
        self,
        actual_pnl: np.ndarray,
        predicted_var: np.ndarray,
        dates: Optional[np.ndarray] = None
    ) -> BacktestResult:
        """
        Comprehensive VaR backtesting.

        Args:
            actual_pnl: Actual P&L (positive = profit, negative = loss)
            predicted_var: Predicted VaR (positive)
            dates: Optional dates for each observation

        Returns:
            BacktestResult with all statistics
        """
        # Convert P&L to losses
        actual_losses = -actual_pnl

        # Calculate violations
        violations = self.calculate_violations(actual_losses, predicted_var)

        n_obs = len(violations)
        n_violations = int(np.sum(violations))
        violation_rate = n_violations / n_obs

        # Statistical tests
        kupiec_result = self.kupiec_pof_test(violations)
        christoffersen_result = self.christoffersen_combined_test(violations)

        # Calculate average violation size
        violation_indices = violations == 1
        if np.any(violation_indices):
            avg_violation_size = np.mean(
                actual_losses[violation_indices] - predicted_var[violation_indices]
            )
        else:
            avg_violation_size = None

        # Create result object
        start_date = dates[0] if dates is not None else datetime.now() - timedelta(days=n_obs)
        end_date = dates[-1] if dates is not None else datetime.now()

        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            confidence_level=self.confidence_level,
            n_observations=n_obs,
            n_violations=n_violations,
            violation_rate=violation_rate,
            kupiec_pof_statistic=kupiec_result['test_statistic'],
            kupiec_pof_pvalue=kupiec_result['p_value'],
            christoffersen_statistic=christoffersen_result['test_statistic'],
            christoffersen_pvalue=christoffersen_result['p_value'],
            avg_var=float(np.mean(predicted_var)),
            max_var=float(np.max(predicted_var)),
            avg_violation_size=avg_violation_size
        )

        return result

    def rolling_backtest(
        self,
        actual_pnl: np.ndarray,
        predicted_var: np.ndarray,
        window: int = 250
    ) -> pd.DataFrame:
        """
        Perform rolling window backtesting.

        Useful for detecting model degradation over time.

        Args:
            actual_pnl: Actual P&L
            predicted_var: Predicted VaR
            window: Rolling window size

        Returns:
            DataFrame with rolling backtest statistics
        """
        n = len(actual_pnl)
        results = []

        for i in range(window, n):
            # Window data
            pnl_window = actual_pnl[i-window:i]
            var_window = predicted_var[i-window:i]

            # Backtest
            result = self.backtest(pnl_window, var_window)

            results.append({
                'end_idx': i,
                'n_violations': result.n_violations,
                'violation_rate': result.violation_rate,
                'kupiec_pvalue': result.kupiec_pof_pvalue,
                'christoffersen_pvalue': result.christoffersen_pvalue,
                'basel_zone': self.calculate_basel_zone(result.n_violations, window)
            })

        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Simulate backtesting data
    np.random.seed(42)

    n_days = 500

    # Generate "actual" returns (with fat tails)
    actual_returns = stats.t.rvs(df=5, loc=0.0005, scale=0.015, size=n_days)

    # Generate VaR predictions (assuming normal distribution - will be wrong!)
    predicted_var = np.abs(stats.norm.ppf(0.01) * 0.015 * np.ones(n_days))

    # Convert returns to P&L (assume $10M portfolio)
    portfolio_value = 10_000_000
    actual_pnl = actual_returns * portfolio_value
    predicted_var_dollar = predicted_var * portfolio_value

    print("=== VaR Backtesting ===\n")

    # Run backtest
    backtester = VaRBacktester(confidence_level=0.99)
    result = backtester.backtest(actual_pnl, predicted_var_dollar)

    # Print results
    print(f"Backtest Period: {result.n_observations} days")
    print(f"Confidence Level: {result.confidence_level * 100}%")
    print(f"\nViolations:")
    print(f"  Count: {result.n_violations}")
    print(f"  Rate: {result.violation_rate * 100:.2f}%")
    print(f"  Expected Rate: {(1-result.confidence_level) * 100:.2f}%")

    # Basel zone
    zone = backtester.calculate_basel_zone(result.n_violations, 250)
    print(f"\nBasel Zone (250-day): {zone.upper()}")

    # Statistical tests
    print(f"\nKupiec POF Test:")
    print(f"  Test Statistic: {result.kupiec_pof_statistic:.4f}")
    print(f"  P-value: {result.kupiec_pof_pvalue:.4f}")
    print(f"  Result: {'PASS' if result.passed_kupiec else 'FAIL'}")

    print(f"\nChristoffersen Test:")
    print(f"  Test Statistic: {result.christoffersen_statistic:.4f}")
    print(f"  P-value: {result.christoffersen_pvalue:.4f}")
    print(f"  Result: {'PASS' if result.passed_christoffersen else 'FAIL'}")

    if result.avg_violation_size:
        print(f"\nAverage Violation Size: ${result.avg_violation_size:,.0f}")

    # Rolling backtest
    print("\n=== Rolling Backtest (Last 5 Windows) ===")
    rolling_results = backtester.rolling_backtest(actual_pnl, predicted_var_dollar, window=250)
    print(rolling_results.tail().to_string(index=False))
