"""
Liquidity-Adjusted VaR (L-VaR) Calculator

Standard VaR assumes positions can be liquidated instantly at mid-market prices.
This is unrealistic - large positions have market impact and require time to unwind.

Liquidity-Adjusted VaR incorporates:
1. Bid-ask spreads (immediate execution cost)
2. Market impact (price moves against you when trading)
3. Liquidation time (longer positions = more risk)

Components of Liquidity Risk:
- Exogenous Spread: Normal bid-ask spread
- Endogenous Spread: Spread widening due to your trade size
- Market Impact: Permanent price change from your order
- Timing: How long it takes to fully liquidate

This is crucial for:
- Large positions relative to average daily volume
- Illiquid markets (small-cap stocks, corporate bonds, exotic derivatives)
- Stress scenarios when liquidity dries up
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from ..config import config


@dataclass
class LiquidityMetrics:
    """Liquidity characteristics of an instrument"""
    symbol: str
    bid_ask_spread_bps: float  # Normal spread in basis points
    average_daily_volume: float  # Average daily trading volume
    daily_volatility: float  # Daily return volatility
    market_cap: Optional[float] = None  # For equities

    # Stress multipliers
    stress_spread_multiplier: float = 3.0  # Spread widens 3x in stress
    stress_volume_reduction: float = 0.5  # Volume drops 50% in stress


class AlmgrenChrissModel:
    """
    Almgren-Chriss optimal execution model.

    This model calculates the optimal way to liquidate a position
    to minimize total cost (market impact + volatility risk).

    References:
    - Almgren & Chriss (2000): "Optimal execution of portfolio transactions"
    - Used by major investment banks for execution algorithms

    The model balances:
    - Market impact: Trading faster increases price impact
    - Timing risk: Trading slower exposes you to market volatility
    """

    def __init__(
        self,
        permanent_impact_coeff: float = 0.1,
        temporary_impact_coeff: float = 0.01,
        risk_aversion: float = 1e-6
    ):
        """
        Initialize Almgren-Chriss model.

        Args:
            permanent_impact_coeff: Permanent price impact per share (η)
            temporary_impact_coeff: Temporary price impact per share (γ)
            risk_aversion: Risk aversion parameter (λ)
        """
        self.eta = permanent_impact_coeff
        self.gamma = temporary_impact_coeff
        self.lambda_risk = risk_aversion

    def calculate_market_impact(
        self,
        shares_to_liquidate: float,
        average_daily_volume: float,
        volatility: float,
        liquidation_days: float
    ) -> Tuple[float, float]:
        """
        Calculate expected market impact cost.

        Market impact has two components:
        1. Permanent impact: Price moves and stays moved
        2. Temporary impact: Price moves but reverts after trade

        Args:
            shares_to_liquidate: Number of shares to sell
            average_daily_volume: Average daily volume
            volatility: Daily volatility
            liquidation_days: Days to complete liquidation

        Returns:
            Tuple of (permanent_impact, temporary_impact) in basis points
        """
        # Participation rate (what fraction of daily volume you trade)
        participation_rate = shares_to_liquidate / (liquidation_days * average_daily_volume)

        # Permanent impact (in basis points)
        # Increases with participation rate
        permanent_impact_bps = self.eta * participation_rate * 10000

        # Temporary impact (in basis points)
        # Decreases with liquidation time (spreading it out helps)
        daily_shares = shares_to_liquidate / liquidation_days
        temporary_impact_bps = self.gamma * (daily_shares / average_daily_volume) * 10000

        return permanent_impact_bps, temporary_impact_bps

    def optimal_liquidation_schedule(
        self,
        shares: float,
        volatility: float,
        adv: float,
        max_days: int = 5
    ) -> np.ndarray:
        """
        Calculate optimal liquidation schedule (how many shares per period).

        Uses Almgren-Chriss closed-form solution for optimal execution.

        Args:
            shares: Total shares to liquidate
            volatility: Daily volatility
            adv: Average daily volume
            max_days: Maximum days to liquidate

        Returns:
            Array of shares to trade each period
        """
        n_periods = max_days
        tau = 1.0  # Time between trades (1 day)

        # Calculate kappa (adjusted impact coefficient)
        kappa = np.sqrt(self.lambda_risk * volatility**2 / (2 * self.gamma))

        # Optimal trajectory (closed form)
        schedule = np.zeros(n_periods)

        for t in range(n_periods):
            remaining_time = (n_periods - t) * tau
            schedule[t] = (2 * shares * np.sinh(0.5 * kappa * tau) /
                          np.sinh(kappa * remaining_time))

        return schedule


class LiquidityAdjustedVaRCalculator:
    """
    Calculate VaR adjusted for liquidity risk.

    L-VaR = Market VaR + Liquidation Cost

    Where Liquidation Cost includes:
    - Bid-ask spread cost
    - Market impact cost
    - Stress scenario adjustments
    """

    def __init__(self, market_var_calculator=None):
        """
        Initialize L-VaR calculator.

        Args:
            market_var_calculator: VaRCalculator instance for market risk
        """
        self.market_var_calc = market_var_calculator
        self.almgren_chriss = AlmgrenChrissModel()

    def calculate_spread_cost(
        self,
        position_value: float,
        spread_bps: float,
        stress_scenario: bool = False,
        stress_multiplier: float = 3.0
    ) -> float:
        """
        Calculate bid-ask spread cost.

        In normal conditions: Cost = Position * Spread / 2
        In stress: Spreads can widen 3-10x

        Args:
            position_value: Market value of position
            spread_bps: Normal spread in basis points
            stress_scenario: Whether to apply stress multipliers
            stress_multiplier: How much spread widens in stress

        Returns:
            Spread cost in dollars
        """
        effective_spread = spread_bps

        if stress_scenario:
            effective_spread *= stress_multiplier

        # Half-spread cost (you pay half the spread on average)
        cost = position_value * (effective_spread / 10000) * 0.5

        return cost

    def calculate_market_impact_cost(
        self,
        position_shares: float,
        position_value: float,
        adv: float,
        volatility: float,
        max_liquidation_days: int = 5
    ) -> float:
        """
        Calculate market impact cost using Almgren-Chriss.

        Args:
            position_shares: Number of shares
            position_value: Dollar value of position
            adv: Average daily volume (shares)
            volatility: Daily volatility
            max_liquidation_days: Maximum days to liquidate

        Returns:
            Market impact cost in dollars
        """
        # Calculate participation rate
        participation_rate = position_shares / (max_liquidation_days * adv)

        # Prevent trading more than 10% ADV per day
        if participation_rate > 0.10:
            # Need more days
            required_days = int(np.ceil(position_shares / (0.10 * adv)))
            max_liquidation_days = min(required_days, 20)  # Cap at 20 days

        # Calculate impact
        perm_impact_bps, temp_impact_bps = self.almgren_chriss.calculate_market_impact(
            position_shares, adv, volatility, max_liquidation_days
        )

        # Total impact cost
        total_impact_bps = perm_impact_bps + temp_impact_bps
        impact_cost = position_value * (total_impact_bps / 10000)

        return impact_cost

    def calculate_liquidity_adjusted_var(
        self,
        market_var: float,
        position_value: float,
        liquidity_metrics: LiquidityMetrics,
        position_shares: Optional[float] = None,
        stress_scenario: bool = False
    ) -> Dict[str, float]:
        """
        Calculate full Liquidity-Adjusted VaR.

        L-VaR = Market VaR + Spread Cost + Market Impact Cost + Liquidation Time Premium

        Regulatory Context:
        - Basel III FRTB requires liquidity horizons based on instrument type
        - Equities: 10 days (large cap) to 60 days (small cap)
        - Corporate bonds: 60-120 days
        - Illiquid derivatives: 120-250 days

        Args:
            market_var: Standard market VaR
            position_value: Position market value
            liquidity_metrics: Liquidity characteristics
            position_shares: Number of shares (if equity)
            stress_scenario: Use stress assumptions

        Returns:
            Dictionary with breakdown of L-VaR components
        """
        # Component 1: Market VaR (from standard calculation)
        components = {
            'market_var': market_var,
        }

        # Component 2: Spread cost
        spread_multiplier = (liquidity_metrics.stress_spread_multiplier
                           if stress_scenario else 1.0)

        spread_cost = self.calculate_spread_cost(
            position_value,
            liquidity_metrics.bid_ask_spread_bps,
            stress_scenario,
            spread_multiplier
        )
        components['spread_cost'] = spread_cost

        # Component 3: Market impact cost
        if position_shares and liquidity_metrics.average_daily_volume > 0:
            adv = liquidity_metrics.average_daily_volume

            if stress_scenario:
                # ADV drops in stress
                adv *= liquidity_metrics.stress_volume_reduction

            impact_cost = self.calculate_market_impact_cost(
                position_shares,
                position_value,
                adv,
                liquidity_metrics.daily_volatility
            )
            components['market_impact_cost'] = impact_cost
        else:
            components['market_impact_cost'] = 0.0

        # Component 4: Liquidation time risk
        # Additional volatility risk from extended liquidation period
        # L-VaR increases with sqrt(liquidation_time)
        liquidation_horizon = self._estimate_liquidation_time(
            position_shares or 0,
            liquidity_metrics.average_daily_volume,
            config.risk_limits.max_adv_pct
        )

        # Horizon adjustment (sqrt scaling)
        if liquidation_horizon > 1:
            horizon_multiplier = np.sqrt(liquidation_horizon)
            timing_risk = market_var * (horizon_multiplier - 1)
            components['timing_risk'] = timing_risk
        else:
            components['timing_risk'] = 0.0

        # Total L-VaR
        components['total_lvar'] = sum(components.values())

        return components

    def _estimate_liquidation_time(
        self,
        position_shares: float,
        adv: float,
        max_participation_rate: float = 0.10
    ) -> int:
        """
        Estimate days needed to liquidate position.

        Args:
            position_shares: Shares to liquidate
            adv: Average daily volume
            max_participation_rate: Max fraction of ADV to trade per day

        Returns:
            Number of days needed
        """
        if adv == 0 or position_shares == 0:
            return 1

        max_daily_shares = adv * max_participation_rate
        days_needed = int(np.ceil(position_shares / max_daily_shares))

        return max(1, days_needed)


# Example usage
if __name__ == "__main__":
    from .var_calculator import VaRCalculator

    # Simulate a large-cap equity position
    print("=== Liquidity-Adjusted VaR Example ===\n")

    # Market VaR calculation
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)
    position_value = 50_000_000  # $50M position

    var_calc = VaRCalculator()
    market_var = var_calc.calculate_historical_var(returns, position_value)

    print(f"Position Value: ${position_value:,.0f}")
    print(f"Market VaR (99%, 1-day): ${market_var:,.0f}\n")

    # Define liquidity characteristics
    liquidity = LiquidityMetrics(
        symbol="AAPL",
        bid_ask_spread_bps=2.0,  # 2 bps (liquid)
        average_daily_volume=50_000_000,  # 50M shares/day
        daily_volatility=0.02,  # 2% daily vol
        market_cap=2_500_000_000_000  # $2.5T market cap
    )

    # Calculate L-VaR
    lvar_calc = LiquidityAdjustedVaRCalculator(var_calc)

    # Normal conditions
    print("--- Normal Market Conditions ---")
    lvar_normal = lvar_calc.calculate_liquidity_adjusted_var(
        market_var=market_var,
        position_value=position_value,
        liquidity_metrics=liquidity,
        position_shares=position_value / 150,  # Assume $150/share
        stress_scenario=False
    )

    for component, value in lvar_normal.items():
        print(f"{component:25s}: ${value:,.0f}")

    print("\n--- Stress Scenario ---")
    lvar_stress = lvar_calc.calculate_liquidity_adjusted_var(
        market_var=market_var * 2,  # VaR doubles in stress
        position_value=position_value,
        liquidity_metrics=liquidity,
        position_shares=position_value / 150,
        stress_scenario=True
    )

    for component, value in lvar_stress.items():
        print(f"{component:25s}: ${value:,.0f}")

    # Calculate increase
    increase_pct = ((lvar_stress['total_lvar'] - lvar_normal['total_lvar']) /
                   lvar_normal['total_lvar'] * 100)

    print(f"\nStress L-VaR is {increase_pct:.1f}% higher than normal L-VaR")
