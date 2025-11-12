"""
GARCH Volatility Modeling

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models capture
volatility clustering - the empirical fact that large price changes tend to be
followed by large changes, and small changes by small changes.

Key insight: Volatility is NOT constant over time - it varies predictably.

GARCH(1,1) model (most common):
σ²_t = ω + α * ε²_(t-1) + β * σ²_(t-1)

Where:
- σ²_t is the conditional variance at time t
- ε²_(t-1) is the squared residual (shock) from previous period
- ω is the long-run average variance
- α captures reaction to market shocks (ARCH effect)
- β captures persistence of volatility

Why GARCH matters for risk:
- VaR should use conditional volatility, not historical average
- GARCH predicts higher volatility after market shocks
- Better forecasts → better risk management

Developed by Robert Engle (2003 Nobel Prize in Economics)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from arch import arch_model
from scipy import stats

from ..config import config


class GARCHVolatilityModel:
    """
    GARCH volatility forecasting for risk management.

    Applications:
    - VaR forecasting with time-varying volatility
    - Option pricing (volatility input)
    - Position sizing based on current volatility regime
    - Risk parity allocation
    """

    def __init__(
        self,
        p: int = None,
        q: int = None,
        mean_model: str = 'Zero',
        vol_model: str = 'GARCH',
        dist: str = 'normal'
    ):
        """
        Initialize GARCH model.

        Args:
            p: GARCH lag order (default from config)
            q: ARCH lag order (default from config)
            mean_model: Mean model ('Zero', 'Constant', 'AR')
            vol_model: Volatility model ('GARCH', 'EGARCH', 'GJR-GARCH')
            dist: Error distribution ('normal', 't', 'skewt')
        """
        self.p = p or config.risk_params.garch_p
        self.q = q or config.risk_params.garch_q
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.dist = dist

        self.model = None
        self.model_fit = None

    def fit(self, returns: np.ndarray, **kwargs) -> 'GARCHVolatilityModel':
        """
        Fit GARCH model to historical returns.

        Args:
            returns: Array of returns (typically daily)
            **kwargs: Additional arguments for arch_model

        Returns:
            Self (for method chaining)
        """
        # Convert to percentage returns for numerical stability
        returns_pct = returns * 100

        # Create and fit model
        self.model = arch_model(
            returns_pct,
            mean=self.mean_model,
            vol=self.vol_model,
            p=self.p,
            q=self.q,
            dist=self.dist,
            **kwargs
        )

        # Fit with error handling
        try:
            self.model_fit = self.model.fit(disp='off', show_warning=False)
        except Exception as e:
            print(f"GARCH fitting failed: {e}")
            # Fallback to simpler model
            self.model = arch_model(returns_pct, mean='Zero', vol='GARCH', p=1, q=1)
            self.model_fit = self.model.fit(disp='off', show_warning=False)

        return self

    def forecast_volatility(
        self,
        horizon: int = 1,
        method: str = 'analytic'
    ) -> np.ndarray:
        """
        Forecast future volatility.

        Args:
            horizon: Forecast horizon in periods
            method: 'analytic' or 'simulation'

        Returns:
            Array of volatility forecasts (annualized if daily data)
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before forecasting")

        # Generate forecast
        forecast = self.model_fit.forecast(horizon=horizon, method=method)

        # Extract variance forecast and convert to volatility
        variance_forecast = forecast.variance.values[-1, :]

        # Convert from percentage to decimal and annualize
        volatility_forecast = np.sqrt(variance_forecast) / 100 * np.sqrt(252)

        return volatility_forecast

    def get_conditional_volatility(self) -> np.ndarray:
        """
        Get conditional volatility for the fitted sample.

        Returns:
            Array of conditional volatilities (annualized)
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")

        # Get conditional volatility from model
        cond_vol = self.model_fit.conditional_volatility

        # Convert from percentage to decimal and annualize
        return cond_vol.values / 100 * np.sqrt(252)

    def calculate_garch_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        confidence_level: float = 0.99,
        horizon: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate VaR using GARCH volatility forecast.

        This is more accurate than using historical volatility because
        it accounts for current market conditions.

        Args:
            returns: Historical returns
            portfolio_value: Portfolio value
            confidence_level: Confidence level
            horizon: Forecast horizon

        Returns:
            Tuple of (VaR, forecasted_volatility)
        """
        # Fit model if not already fitted
        if self.model_fit is None:
            self.fit(returns)

        # Forecast volatility
        vol_forecast = self.forecast_volatility(horizon=horizon)[0]

        # Calculate VaR using forecasted volatility
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)

        # Assume zero mean for simplicity (conservative)
        var_dollar = -z_score * vol_forecast * portfolio_value * np.sqrt(horizon / 252)

        return var_dollar, vol_forecast

    def get_model_parameters(self) -> Dict[str, float]:
        """
        Extract fitted GARCH parameters.

        Returns:
            Dictionary of parameter estimates
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")

        params = self.model_fit.params.to_dict()

        # Calculate persistence (α + β)
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
        persistence = alpha + beta

        # Calculate long-run volatility
        omega = params.get('omega', 0)
        if persistence < 1:
            long_run_var = omega / (1 - persistence)
            long_run_vol = np.sqrt(long_run_var) / 100 * np.sqrt(252)
        else:
            long_run_vol = None

        return {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'persistence': persistence,
            'long_run_vol_annual': long_run_vol
        }

    def calculate_volatility_ratio(
        self,
        returns: np.ndarray
    ) -> float:
        """
        Calculate ratio of current volatility to long-run volatility.

        Ratio > 1: Current volatility higher than average (risky period)
        Ratio < 1: Current volatility lower than average (calm period)

        Useful for:
        - Dynamic position sizing
        - Risk regime detection
        - VaR scaling

        Args:
            returns: Historical returns

        Returns:
            Volatility ratio
        """
        if self.model_fit is None:
            self.fit(returns)

        # Get current conditional volatility (last value)
        current_vol = self.get_conditional_volatility()[-1]

        # Get long-run volatility
        params = self.get_model_parameters()
        long_run_vol = params['long_run_vol_annual']

        if long_run_vol is None or long_run_vol == 0:
            return 1.0

        return current_vol / long_run_vol


class GJRGARCHModel(GARCHVolatilityModel):
    """
    GJR-GARCH model (Glosten-Jagannathan-Runkle)

    Extends GARCH to capture leverage effect:
    - Negative shocks (bad news) increase volatility more than positive shocks
    - Asymmetric response to up/down moves

    Model:
    σ²_t = ω + α * ε²_(t-1) + γ * ε²_(t-1) * I_(ε<0) + β * σ²_(t-1)

    Where γ captures the leverage effect (typically γ > 0)

    Empirical fact: Stock volatility rises more after drops than gains
    """

    def __init__(self, p: int = None, q: int = None, **kwargs):
        super().__init__(p=p, q=q, vol_model='GARCH', **kwargs)
        # Override to use GJR-GARCH
        self.vol_model = 'GARCH'  # Will manually add asymmetry

    def fit(self, returns: np.ndarray, **kwargs):
        """Fit GJR-GARCH with asymmetric response"""
        returns_pct = returns * 100

        # GJR-GARCH specification
        self.model = arch_model(
            returns_pct,
            mean=self.mean_model,
            vol='GARCH',
            p=self.p,
            o=1,  # Asymmetric term
            q=self.q,
            dist=self.dist
        )

        try:
            self.model_fit = self.model.fit(disp='off', show_warning=False)
        except:
            # Fallback
            self.model = arch_model(returns_pct, mean='Zero', vol='GARCH', p=1, q=1)
            self.model_fit = self.model.fit(disp='off', show_warning=False)

        return self


class EGARCHModel(GARCHVolatilityModel):
    """
    EGARCH (Exponential GARCH) model

    Advantages:
    - Log-volatility (ensures σ² > 0 without constraints)
    - Captures leverage effect naturally
    - More flexible asymmetric response

    Model:
    log(σ²_t) = ω + α * |z_(t-1)| + γ * z_(t-1) + β * log(σ²_(t-1))

    Popular for equity indices and individual stocks
    """

    def __init__(self, p: int = None, q: int = None, **kwargs):
        super().__init__(p=p, q=q, vol_model='EGARCH', **kwargs)


def create_volatility_model(
    model_type: str = 'garch',
    **kwargs
) -> GARCHVolatilityModel:
    """
    Factory function to create volatility models.

    Args:
        model_type: 'garch', 'gjr-garch', or 'egarch'
        **kwargs: Model parameters

    Returns:
        Volatility model instance
    """
    if model_type.lower() == 'garch':
        return GARCHVolatilityModel(**kwargs)
    elif model_type.lower() == 'gjr-garch':
        return GJRGARCHModel(**kwargs)
    elif model_type.lower() == 'egarch':
        return EGARCHModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Generate synthetic returns with volatility clustering
    np.random.seed(42)

    # Simulate GARCH(1,1) process
    n = 1000
    omega = 0.01
    alpha = 0.1
    beta = 0.85

    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.randn()

    print("=== GARCH Volatility Modeling ===\n")

    # Fit GARCH model
    garch = GARCHVolatilityModel(p=1, q=1)
    garch.fit(returns)

    # Get parameters
    params = garch.get_model_parameters()
    print("Fitted Parameters:")
    for name, value in params.items():
        if value is not None:
            print(f"  {name:20s}: {value:.6f}")

    # Forecast volatility
    vol_forecast = garch.forecast_volatility(horizon=10)
    print(f"\n10-Day Volatility Forecast:")
    for i, vol in enumerate(vol_forecast, 1):
        print(f"  Day {i}: {vol*100:.2f}%")

    # Calculate GARCH-based VaR
    portfolio_value = 10_000_000
    var, current_vol = garch.calculate_garch_var(
        returns, portfolio_value, confidence_level=0.99
    )

    print(f"\nGARCH-based VaR:")
    print(f"  Current Volatility: {current_vol*100:.2f}%")
    print(f"  1-Day 99% VaR: ${var:,.0f}")

    # Volatility ratio
    vol_ratio = garch.calculate_volatility_ratio(returns)
    print(f"\nVolatility Ratio: {vol_ratio:.2f}x")
    if vol_ratio > 1.2:
        print("  ⚠ Current volatility is elevated - consider reducing risk")
    elif vol_ratio < 0.8:
        print("  ✓ Current volatility is low - may increase risk capacity")
