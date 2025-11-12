"""
Expected Shortfall (ES) / Conditional Value-at-Risk (CVaR) Calculator

Expected Shortfall measures the average loss in the worst (1-α)% of cases,
where α is the confidence level. It's a coherent risk measure that addresses
VaR's limitations.

Why ES is important:
- ES is "coherent" (satisfies subadditivity, monotonicity, etc.)
- VaR is NOT coherent - can encourage excessive risk-taking
- Basel III moved from VaR to ES for market risk capital (2019)
- ES considers the severity of tail losses, not just frequency

Example: 99% ES of $2M means the average loss in the worst 1% scenarios is $2M.
This is always ≥ VaR since it includes more extreme losses.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import stats

from ..config import config


class ExpectedShortfallCalculator:
    """
    Calculate Expected Shortfall (ES) / Conditional VaR.

    Regulatory Context:
    - Basel III FRTB (2019): Replaced VaR with ES for market risk capital
    - ES at 97.5% confidence over 10-day horizon
    - Reason: ES is more sensitive to tail risk than VaR
    """

    def __init__(
        self,
        confidence_level: Optional[float] = None,
        horizon_days: Optional[int] = None
    ):
        """
        Initialize ES calculator.

        Args:
            confidence_level: Confidence level (e.g., 0.975 for Basel III)
            horizon_days: Time horizon in days
        """
        self.confidence_level = (
            confidence_level or config.risk_params.expected_shortfall_confidence
        )
        self.horizon_days = horizon_days or config.risk_params.var_horizon_days

    def calculate_historical_es(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> float:
        """
        Calculate ES using Historical Simulation.

        Method:
        1. Sort returns from worst to best
        2. Take the worst (1-α)% of returns
        3. Calculate the average of these tail returns
        4. Scale by portfolio value and horizon

        Args:
            returns: Historical returns array
            portfolio_value: Current portfolio value

        Returns:
            Expected Shortfall in dollars
        """
        if len(returns) == 0:
            raise ValueError("No historical returns provided")

        # Calculate the percentile (alpha = 1 - confidence_level)
        alpha = 1 - self.confidence_level

        # Get VaR threshold
        var_threshold = np.percentile(returns, alpha * 100)

        # Find all returns worse than VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            # No tail events in sample - use VaR as approximation
            tail_mean = var_threshold
        else:
            # Average of tail losses
            tail_mean = np.mean(tail_returns)

        # Scale by portfolio value and horizon
        es_dollar = -tail_mean * portfolio_value * np.sqrt(self.horizon_days)

        return float(es_dollar)

    def calculate_parametric_es(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> float:
        """
        Calculate ES assuming normal distribution.

        For normal distribution, there's a closed-form formula:
        ES = μ + σ * φ(z_α) / α * sqrt(horizon)

        Where:
        - φ is the standard normal PDF
        - z_α is the α-quantile of standard normal
        - α = 1 - confidence_level

        Args:
            returns: Historical returns
            portfolio_value: Portfolio value

        Returns:
            Expected Shortfall in dollars
        """
        if len(returns) == 0:
            raise ValueError("No historical returns provided")

        # Calculate mean and std
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Calculate for normal distribution
        alpha = 1 - self.confidence_level
        z_alpha = stats.norm.ppf(alpha)

        # Normal PDF at z_alpha
        phi_z = stats.norm.pdf(z_alpha)

        # Expected Shortfall for normal distribution
        es_return = -(mu + sigma * phi_z / alpha)

        # Scale by value and horizon
        es_dollar = es_return * portfolio_value * np.sqrt(self.horizon_days)

        return float(es_dollar)

    def calculate_monte_carlo_es(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        n_simulations: Optional[int] = None,
        method: str = 'normal'
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate ES using Monte Carlo simulation.

        Process:
        1. Simulate N return scenarios
        2. Calculate loss in each scenario
        3. Take worst (1-α)% of scenarios
        4. Average these tail losses

        Args:
            returns: Historical returns
            portfolio_value: Portfolio value
            n_simulations: Number of simulations
            method: 'normal' or 'bootstrap'

        Returns:
            Tuple of (ES in dollars, simulated tail returns)
        """
        if len(returns) == 0:
            raise ValueError("No historical returns provided")

        n_sims = n_simulations or config.risk_params.mc_simulations

        if method == 'normal':
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)

            np.random.seed(config.risk_params.mc_random_seed)
            simulated_returns = np.random.normal(
                mu * self.horizon_days,
                sigma * np.sqrt(self.horizon_days),
                n_sims
            )

        elif method == 'bootstrap':
            np.random.seed(config.risk_params.mc_random_seed)

            if self.horizon_days == 1:
                simulated_returns = np.random.choice(returns, size=n_sims, replace=True)
            else:
                daily_samples = np.random.choice(
                    returns,
                    size=(n_sims, self.horizon_days),
                    replace=True
                )
                simulated_returns = np.sum(daily_samples, axis=1)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate losses
        simulated_losses = -simulated_returns * portfolio_value

        # Get tail losses (worst (1-α)%)
        alpha = 1 - self.confidence_level
        var_threshold = np.percentile(simulated_losses, (1 - alpha) * 100)

        tail_losses = simulated_losses[simulated_losses >= var_threshold]

        # Expected Shortfall is mean of tail
        es_dollar = np.mean(tail_losses)

        return float(es_dollar), simulated_returns[simulated_returns <= -var_threshold/portfolio_value]

    def calculate_es_ratio(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        var_value: float
    ) -> float:
        """
        Calculate ES/VaR ratio.

        The ES/VaR ratio indicates tail risk severity:
        - Ratio close to 1.0: Thin tails (near-normal)
        - Ratio >> 1.0: Fat tails (extreme events likely)

        Typical values:
        - Normal distribution: ~1.15 at 99% confidence
        - Fat-tailed distribution: 1.5 - 3.0+
        - Crisis periods: Often >2.0

        Args:
            returns: Historical returns
            portfolio_value: Portfolio value
            var_value: Pre-calculated VaR value

        Returns:
            ES/VaR ratio
        """
        es = self.calculate_historical_es(returns, portfolio_value)

        if var_value == 0:
            return 0.0

        return es / var_value


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Test with normal returns
    print("=== Normal Distribution ===")
    normal_returns = np.random.normal(0.0005, 0.02, 252)
    portfolio_value = 10_000_000

    es_calc = ExpectedShortfallCalculator(confidence_level=0.975)

    hist_es = es_calc.calculate_historical_es(normal_returns, portfolio_value)
    param_es = es_calc.calculate_parametric_es(normal_returns, portfolio_value)
    mc_es, _ = es_calc.calculate_monte_carlo_es(normal_returns, portfolio_value)

    print(f"Historical ES:  ${hist_es:,.0f}")
    print(f"Parametric ES:  ${param_es:,.0f}")
    print(f"Monte Carlo ES: ${mc_es:,.0f}")

    # Test with fat-tailed returns (Student's t)
    print("\n=== Fat-Tailed Distribution (Student's t, df=5) ===")
    fat_tail_returns = stats.t.rvs(df=5, loc=0.0005, scale=0.02, size=252)

    hist_es_fat = es_calc.calculate_historical_es(fat_tail_returns, portfolio_value)
    param_es_fat = es_calc.calculate_parametric_es(fat_tail_returns, portfolio_value)

    print(f"Historical ES:  ${hist_es_fat:,.0f}")
    print(f"Parametric ES:  ${param_es_fat:,.0f}")
    print("\nNote: Parametric ES underestimates risk for fat-tailed distributions")
