"""
Value-at-Risk (VaR) Calculator

Implements multiple VaR methodologies:
1. Historical Simulation - Non-parametric, uses actual return distribution
2. Parametric (Variance-Covariance) - Assumes normal distribution
3. Monte Carlo Simulation - Simulates future scenarios

VaR measures the maximum expected loss at a given confidence level over a time horizon.
For example, 1-day 99% VaR of $1M means there's a 1% chance of losing more than $1M tomorrow.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy import stats
from datetime import datetime

from ..config import config
from ..utils.data_structures import Portfolio, RiskMetrics, Position


class VaRCalculator:
    """
    Calculate Value-at-Risk using multiple methodologies.

    Value-at-Risk (VaR) is the maximum loss not exceeded with a given confidence level
    over a specified time horizon. It's a key risk metric used by banks and regulators.

    Regulatory Context:
    - Basel III requires banks to calculate VaR for market risk capital
    - SEC requires VaR disclosure for certain investment companies
    - UCITS funds must use VaR for risk management
    """

    def __init__(
        self,
        confidence_level: float = None,
        horizon_days: int = None,
        historical_window_days: int = None
    ):
        """
        Initialize VaR calculator.

        Args:
            confidence_level: Confidence level (e.g., 0.99 for 99%)
            horizon_days: Time horizon in days (typically 1 or 10)
            historical_window_days: Lookback period for historical simulation
        """
        self.confidence_level = confidence_level or config.risk_params.var_confidence_level
        self.horizon_days = horizon_days or config.risk_params.var_horizon_days
        self.historical_window_days = (
            historical_window_days or config.risk_params.historical_window_days
        )

    def calculate_historical_var(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> float:
        """
        Calculate VaR using Historical Simulation method.

        This is the simplest and most intuitive VaR method:
        1. Collect historical returns
        2. Sort them from worst to best
        3. Find the return at the confidence level percentile
        4. Scale by portfolio value

        Advantages:
        - Non-parametric (no distribution assumptions)
        - Captures fat tails and skewness
        - Easy to understand and implement

        Disadvantages:
        - Assumes future will resemble past
        - Requires significant historical data
        - Slow to react to regime changes

        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value

        Returns:
            VaR in dollars (positive number represents potential loss)
        """
        if len(returns) == 0:
            raise ValueError("No historical returns provided")

        # Calculate the percentile (alpha = 1 - confidence_level)
        alpha = 1 - self.confidence_level

        # Find the return at the alpha percentile (worst case)
        var_return = np.percentile(returns, alpha * 100)

        # Scale by portfolio value and horizon (assuming IID returns)
        # Note: sqrt(horizon) scaling assumes returns are independent
        var_dollar = -var_return * portfolio_value * np.sqrt(self.horizon_days)

        return float(var_dollar)

    def calculate_parametric_var(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> float:
        """
        Calculate VaR using Parametric (Variance-Covariance) method.

        This method assumes returns are normally distributed:
        VaR = μ + σ * z * sqrt(horizon) * portfolio_value

        Where:
        - μ is the mean return
        - σ is the standard deviation of returns
        - z is the z-score at the confidence level
        - horizon is the time horizon

        Advantages:
        - Fast to calculate
        - Requires less data than historical simulation
        - Good for portfolios with near-normal returns

        Disadvantages:
        - Assumes normal distribution (fat tails underestimated)
        - Poor for options or non-linear instruments
        - Can significantly underestimate risk in crisis

        Regulatory Note:
        RiskMetrics (J.P. Morgan, 1994) popularized this approach with
        exponentially weighted moving average (EWMA) for volatility.

        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value

        Returns:
            VaR in dollars
        """
        if len(returns) == 0:
            raise ValueError("No historical returns provided")

        # Calculate mean and standard deviation
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)  # Sample std deviation

        # Get z-score for confidence level (e.g., 2.33 for 99%)
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(alpha)  # Negative value for losses

        # Calculate VaR
        # VaR = -1 * (expected return - z * volatility) * value * sqrt(horizon)
        var_return = -(mu + z_score * sigma)
        var_dollar = var_return * portfolio_value * np.sqrt(self.horizon_days)

        return float(var_dollar)

    def calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        n_simulations: Optional[int] = None,
        method: str = 'normal'
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate VaR using Monte Carlo simulation.

        Simulates many possible future scenarios and calculates the loss
        distribution. More flexible than parametric methods as it can
        incorporate non-normal distributions, correlations, and path dependency.

        Methods:
        - 'normal': Simulate from normal distribution (simple)
        - 'bootstrap': Resample historical returns (captures actual distribution)

        Process:
        1. Estimate return distribution parameters from historical data
        2. Simulate N possible future return paths
        3. Calculate portfolio value under each scenario
        4. Sort scenarios and find VaR at confidence level

        Advantages:
        - Flexible - can model complex payoffs
        - Can incorporate non-normal distributions
        - Good for options and derivatives

        Disadvantages:
        - Computationally intensive
        - Results vary between runs (random)
        - Model risk if assumptions are wrong

        Regulatory Note:
        Basel III allows Monte Carlo for Internal Models Approach (IMA)
        but requires backtesting and model validation.

        Args:
            returns: Historical returns array
            portfolio_value: Current portfolio value
            n_simulations: Number of Monte Carlo paths (default from config)
            method: 'normal' or 'bootstrap'

        Returns:
            Tuple of (VaR in dollars, array of simulated returns)
        """
        if len(returns) == 0:
            raise ValueError("No historical returns provided")

        n_sims = n_simulations or config.risk_params.mc_simulations

        if method == 'normal':
            # Simulate from normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)

            # Generate random returns
            np.random.seed(config.risk_params.mc_random_seed)
            simulated_returns = np.random.normal(
                mu * self.horizon_days,
                sigma * np.sqrt(self.horizon_days),
                n_sims
            )

        elif method == 'bootstrap':
            # Bootstrap resampling from historical returns
            np.random.seed(config.risk_params.mc_random_seed)

            # For multi-period horizon, sample with replacement
            if self.horizon_days == 1:
                simulated_returns = np.random.choice(returns, size=n_sims, replace=True)
            else:
                # For multi-day horizon, sum random walks
                daily_samples = np.random.choice(
                    returns,
                    size=(n_sims, self.horizon_days),
                    replace=True
                )
                simulated_returns = np.sum(daily_samples, axis=1)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate portfolio values under each scenario
        simulated_losses = -simulated_returns * portfolio_value

        # VaR is the percentile of the loss distribution
        alpha = 1 - self.confidence_level
        var_dollar = np.percentile(simulated_losses, (1 - alpha) * 100)

        return float(var_dollar), simulated_returns

    def calculate_all_var_methods(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate VaR using all methods for comparison.

        Best Practice:
        Compare multiple methods to understand model risk. Large discrepancies
        indicate that distribution assumptions matter significantly.

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with VaR from each method
        """
        results = {}

        try:
            results['historical'] = self.calculate_historical_var(returns, portfolio_value)
        except Exception as e:
            results['historical'] = None
            print(f"Historical VaR failed: {e}")

        try:
            results['parametric'] = self.calculate_parametric_var(returns, portfolio_value)
        except Exception as e:
            results['parametric'] = None
            print(f"Parametric VaR failed: {e}")

        try:
            results['monte_carlo_normal'], _ = self.calculate_monte_carlo_var(
                returns, portfolio_value, method='normal'
            )
        except Exception as e:
            results['monte_carlo_normal'] = None
            print(f"Monte Carlo (normal) VaR failed: {e}")

        try:
            results['monte_carlo_bootstrap'], _ = self.calculate_monte_carlo_var(
                returns, portfolio_value, method='bootstrap'
            )
        except Exception as e:
            results['monte_carlo_bootstrap'] = None
            print(f"Monte Carlo (bootstrap) VaR failed: {e}")

        return results

    def calculate_component_var(
        self,
        portfolio_returns: np.ndarray,
        position_returns: Dict[str, np.ndarray],
        portfolio_value: float,
        position_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate Component VaR (CVaR) - each position's contribution to portfolio VaR.

        Component VaR answers: "How much does each position contribute to total portfolio risk?"

        Formula:
        CVaR_i = (weight_i * β_i * Portfolio_VaR)

        Where β_i is the beta of position i to the portfolio:
        β_i = Cov(r_i, r_portfolio) / Var(r_portfolio)

        Key Property: Sum of all Component VaRs equals Portfolio VaR

        Use Cases:
        - Risk budgeting: Allocate risk limits to traders/strategies
        - Attribution: Understand sources of risk
        - Optimization: Identify positions to reduce for max VaR reduction

        Regulatory Note:
        FRTB (Fundamental Review of the Trading Book) requires granular
        risk attribution for trading book capital.

        Args:
            portfolio_returns: Portfolio-level returns
            position_returns: Dict of returns for each position
            portfolio_value: Total portfolio value
            position_values: Dict of market values for each position

        Returns:
            Dict mapping position symbol to Component VaR
        """
        # Calculate portfolio VaR
        portfolio_var = self.calculate_historical_var(portfolio_returns, portfolio_value)

        # Calculate portfolio variance
        portfolio_variance = np.var(portfolio_returns, ddof=1)

        if portfolio_variance == 0:
            return {symbol: 0.0 for symbol in position_returns.keys()}

        component_vars = {}

        for symbol, returns in position_returns.items():
            # Calculate beta of this position to portfolio
            covariance = np.cov(returns, portfolio_returns)[0, 1]
            beta = covariance / portfolio_variance

            # Weight of position in portfolio
            weight = position_values[symbol] / portfolio_value

            # Component VaR
            component_vars[symbol] = weight * beta * portfolio_var

        return component_vars

    def calculate_marginal_var(
        self,
        portfolio_returns: np.ndarray,
        position_returns: Dict[str, np.ndarray],
        portfolio_value: float,
        position_values: Dict[str, float],
        delta_pct: float = 0.01
    ) -> Dict[str, float]:
        """
        Calculate Marginal VaR (MVaR) - impact of small position change on portfolio VaR.

        Marginal VaR answers: "If I increase this position by $1, how much does portfolio VaR increase?"

        Formula (approximate):
        MVaR_i ≈ ∂VaR/∂w_i ≈ (VaR(portfolio + Δw_i) - VaR(portfolio)) / Δw_i

        Use Cases:
        - Position sizing: Determine optimal position size
        - Hedging: Identify which positions to hedge first
        - Limit monitoring: Predict VaR impact before trading

        Args:
            portfolio_returns: Portfolio returns
            position_returns: Returns for each position
            portfolio_value: Total portfolio value
            position_values: Market value of each position
            delta_pct: Percentage change for finite difference (1% default)

        Returns:
            Dict mapping symbol to Marginal VaR (VaR change per $1 invested)
        """
        # Calculate base portfolio VaR
        base_var = self.calculate_historical_var(portfolio_returns, portfolio_value)

        marginal_vars = {}

        for symbol, returns in position_returns.items():
            # Calculate new portfolio returns with slightly increased position
            weight = position_values[symbol] / portfolio_value
            delta_weight = delta_pct * weight

            # Shocked portfolio returns (approximate)
            shocked_returns = portfolio_returns + delta_weight * returns

            # Calculate VaR with shocked position
            shocked_var = self.calculate_historical_var(shocked_returns, portfolio_value)

            # Marginal VaR (derivative approximation)
            marginal_var = (shocked_var - base_var) / (delta_weight * portfolio_value)

            marginal_vars[symbol] = marginal_var

        return marginal_vars


def calculate_portfolio_returns(
    positions: Dict[str, Position],
    price_data: pd.DataFrame,
    portfolio_value: float
) -> np.ndarray:
    """
    Calculate portfolio returns from position weights and price data.

    Args:
        positions: Dict of Position objects
        price_data: DataFrame with columns for each symbol, index is datetime
        portfolio_value: Total portfolio value

    Returns:
        Array of portfolio returns
    """
    # Calculate weights
    weights = {
        symbol: pos.market_value / portfolio_value
        for symbol, pos in positions.items()
    }

    # Calculate returns for each symbol
    returns_df = price_data.pct_change().dropna()

    # Calculate weighted portfolio returns
    portfolio_returns = np.zeros(len(returns_df))

    for symbol, weight in weights.items():
        if symbol in returns_df.columns:
            portfolio_returns += weight * returns_df[symbol].values

    return portfolio_returns


# Example usage and testing
if __name__ == "__main__":
    # Simulate some test data
    np.random.seed(42)

    # Generate synthetic returns (daily, 1 year)
    n_days = 252
    mu = 0.0005  # 0.05% daily return
    sigma = 0.02  # 2% daily volatility

    returns = np.random.normal(mu, sigma, n_days)
    portfolio_value = 10_000_000  # $10M portfolio

    # Initialize calculator
    var_calc = VaRCalculator(confidence_level=0.99, horizon_days=1)

    # Calculate VaR using all methods
    print("=== VaR Calculation Results ===")
    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    print(f"Confidence Level: {var_calc.confidence_level * 100}%")
    print(f"Horizon: {var_calc.horizon_days} day(s)")
    print()

    var_results = var_calc.calculate_all_var_methods(returns, portfolio_value)

    for method, var_value in var_results.items():
        if var_value is not None:
            print(f"{method:25s}: ${var_value:,.0f}")

    print("\n=== Interpretation ===")
    hist_var = var_results.get('historical', 0)
    print(f"Historical VaR: ${hist_var:,.0f}")
    print(f"This means there is a {(1-var_calc.confidence_level)*100}% chance")
    print(f"of losing more than ${hist_var:,.0f} in the next {var_calc.horizon_days} day(s).")
