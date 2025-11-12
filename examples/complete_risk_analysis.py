"""
Complete Risk Analysis Example

This script demonstrates the full capability of the risk management system,
showing how all components work together for comprehensive risk analysis.

Scenario: Portfolio manager analyzing a $50M equity portfolio
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.risk_engine.var_calculator import VaRCalculator
from src.risk_engine.expected_shortfall import ExpectedShortfallCalculator
from src.risk_engine.garch_volatility import GARCHVolatilityModel
from src.risk_engine.liquidity_adjusted_var import (
    LiquidityAdjustedVaRCalculator,
    LiquidityMetrics
)
from src.regime_detection.hmm_regime import HMMRegimeDetector
from src.regime_detection.change_point import ChangePointDetector
from src.backtesting.var_backtest import VaRBacktester
from src.utils.data_structures import MarketRegime


def generate_synthetic_market_data(n_days=500):
    """
    Generate synthetic market data with realistic regime changes.

    Returns:
        tuple: (returns, dates)
    """
    np.random.seed(42)

    # Different regime periods
    # Bull market (200 days)
    bull_returns = np.random.normal(0.001, 0.012, 200)

    # Transition to neutral (100 days)
    neutral_returns = np.random.normal(0.0003, 0.018, 100)

    # Bear market begins (100 days)
    bear_returns = np.random.normal(-0.001, 0.025, 100)

    # Crisis/crash (50 days)
    crisis_returns = np.random.normal(-0.004, 0.045, 50)

    # Recovery (50 days)
    recovery_returns = np.random.normal(0.002, 0.020, 50)

    # Concatenate all regimes
    returns = np.concatenate([
        bull_returns,
        neutral_returns,
        bear_returns,
        crisis_returns,
        recovery_returns
    ])

    # Generate dates
    start_date = datetime.now() - timedelta(days=len(returns))
    dates = [start_date + timedelta(days=i) for i in range(len(returns))]

    return returns, np.array(dates)


def main():
    print("=" * 80)
    print("COMPREHENSIVE RISK MANAGEMENT SYSTEM DEMO")
    print("=" * 80)
    print()

    # Portfolio setup
    portfolio_value = 50_000_000  # $50M
    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    print()

    # Generate market data
    print("Generating synthetic market data...")
    returns, dates = generate_synthetic_market_data()
    print(f"Generated {len(returns)} days of return data")
    print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print()

    # =========================================================================
    # 1. MARKET REGIME DETECTION
    # =========================================================================
    print("=" * 80)
    print("1. MARKET REGIME DETECTION")
    print("=" * 80)
    print()

    print("Fitting Hidden Markov Model for regime detection...")
    regime_detector = HMMRegimeDetector(n_regimes=4)
    regime_detector.fit(returns, feature_window=20)

    # Get current regime
    current_regime, regime_probs = regime_detector.predict_regime(returns)

    print(f"\nCurrent Market Regime: {current_regime.value.upper()}")
    print("\nRegime Probabilities:")
    for regime, prob in sorted(regime_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"  {regime.value:10s}: {prob*100:5.1f}% {bar}")

    # Regime statistics
    print("\nHistorical Regime Statistics:")
    regime_stats = regime_detector.get_regime_statistics(returns)
    print(regime_stats[['regime', 'frequency', 'avg_return', 'volatility', 'sharpe']].to_string(index=False))

    # Expected durations
    print("\nExpected Regime Durations (days):")
    durations = regime_detector.get_expected_duration()
    for regime, duration in durations.items():
        print(f"  {regime.value:10s}: {duration:6.1f} days")

    print()

    # =========================================================================
    # 2. CHANGE POINT DETECTION
    # =========================================================================
    print("=" * 80)
    print("2. CHANGE POINT DETECTION (STRUCTURAL BREAKS)")
    print("=" * 80)
    print()

    change_detector = ChangePointDetector(method='pelt', penalty_value=2.0)

    # Detect mean changes
    mean_cps = change_detector.detect_mean_changes(returns, n_changepoints=5)
    print(f"Detected {len(mean_cps)} mean change points at days: {mean_cps}")

    # Detect variance changes
    var_cps = change_detector.detect_variance_changes(returns, n_changepoints=5)
    print(f"Detected {len(var_cps)} variance change points at days: {var_cps}")

    # Volatility breakouts
    breakouts = change_detector.detect_volatility_breakout(returns, window=20, n_std=2.0)
    print(f"Detected {len(breakouts)} volatility breakouts")

    # Online detection
    recent_change = change_detector.online_change_detection(
        returns[:400],
        returns[400:450],
        method='cusum'
    )
    print(f"\nReal-time change detected in recent data: {recent_change}")
    print()

    # =========================================================================
    # 3. GARCH VOLATILITY MODELING
    # =========================================================================
    print("=" * 80)
    print("3. GARCH VOLATILITY MODELING")
    print("=" * 80)
    print()

    print("Fitting GARCH(1,1) model...")
    garch = GARCHVolatilityModel(p=1, q=1)
    garch.fit(returns)

    # Get parameters
    params = garch.get_model_parameters()
    print("\nGARCH Parameters:")
    print(f"  ω (omega):      {params['omega']:.6f}")
    print(f"  α (alpha):      {params['alpha']:.6f}")
    print(f"  β (beta):       {params['beta']:.6f}")
    print(f"  Persistence:    {params['persistence']:.6f}")

    if params['long_run_vol_annual']:
        print(f"  Long-run Vol:   {params['long_run_vol_annual']*100:.2f}%")

    # Volatility forecast
    vol_forecast = garch.forecast_volatility(horizon=10)
    print("\n10-Day Volatility Forecast:")
    for i, vol in enumerate(vol_forecast[:5], 1):
        print(f"  Day {i}: {vol*100:5.2f}%")

    # Volatility ratio
    vol_ratio = garch.calculate_volatility_ratio(returns)
    print(f"\nVolatility Ratio (current/long-run): {vol_ratio:.2f}x")
    if vol_ratio > 1.5:
        print("  ⚠️  WARNING: Current volatility significantly elevated")
    elif vol_ratio < 0.7:
        print("  ✅ Current volatility below long-run average")

    print()

    # =========================================================================
    # 4. VAR CALCULATIONS
    # =========================================================================
    print("=" * 80)
    print("4. VALUE-AT-RISK (VaR) CALCULATIONS")
    print("=" * 80)
    print()

    var_calc = VaRCalculator(confidence_level=0.99, horizon_days=1)

    # Calculate VaR using all methods
    print("Calculating VaR using multiple methodologies...")
    var_results = var_calc.calculate_all_var_methods(returns, portfolio_value)

    print("\n1-Day 99% VaR Results:")
    print("-" * 50)
    for method, var_value in var_results.items():
        if var_value is not None:
            pct = (var_value / portfolio_value) * 100
            print(f"  {method:25s}: ${var_value:>12,.0f} ({pct:.2f}%)")

    # GARCH-based VaR
    garch_var, current_vol = garch.calculate_garch_var(returns, portfolio_value)
    print(f"  {'GARCH VaR':25s}: ${garch_var:>12,.0f} ({current_vol*100:.2f}% vol)")

    print()

    # =========================================================================
    # 5. EXPECTED SHORTFALL
    # =========================================================================
    print("=" * 80)
    print("5. EXPECTED SHORTFALL (CVaR)")
    print("=" * 80)
    print()

    es_calc = ExpectedShortfallCalculator(confidence_level=0.975)

    hist_es = es_calc.calculate_historical_es(returns, portfolio_value)
    param_es = es_calc.calculate_parametric_es(returns, portfolio_value)
    mc_es, _ = es_calc.calculate_monte_carlo_es(returns, portfolio_value, method='bootstrap')

    print("1-Day 97.5% Expected Shortfall:")
    print("-" * 50)
    print(f"  Historical ES:  ${hist_es:>12,.0f}")
    print(f"  Parametric ES:  ${param_es:>12,.0f}")
    print(f"  Monte Carlo ES: ${mc_es:>12,.0f}")

    # ES/VaR ratio
    historical_var = var_results['historical']
    if historical_var:
        es_var_ratio = hist_es / historical_var
        print(f"\nES/VaR Ratio: {es_var_ratio:.2f}")
        if es_var_ratio > 1.3:
            print("  ⚠️  Fat-tailed distribution detected (high tail risk)")
        else:
            print("  ✅ Tail risk consistent with near-normal distribution")

    print()

    # =========================================================================
    # 6. LIQUIDITY-ADJUSTED VAR
    # =========================================================================
    print("=" * 80)
    print("6. LIQUIDITY-ADJUSTED VaR (L-VaR)")
    print("=" * 80)
    print()

    # Define liquidity characteristics (large-cap equity)
    liquidity = LiquidityMetrics(
        symbol="PORTFOLIO",
        bid_ask_spread_bps=3.0,  # 3 bps
        average_daily_volume=10_000_000,  # 10M shares
        daily_volatility=np.std(returns),
        market_cap=500_000_000_000  # $500B
    )

    lvar_calc = LiquidityAdjustedVaRCalculator(var_calc)

    # Calculate in normal and stress scenarios
    market_var = var_results['historical']

    print("Normal Market Conditions:")
    print("-" * 50)
    lvar_normal = lvar_calc.calculate_liquidity_adjusted_var(
        market_var=market_var,
        position_value=portfolio_value,
        liquidity_metrics=liquidity,
        position_shares=portfolio_value / 100,  # Assume $100/share
        stress_scenario=False
    )

    for component, value in lvar_normal.items():
        print(f"  {component:25s}: ${value:>12,.0f}")

    print("\nStress Scenario (3x spreads, 50% volume drop):")
    print("-" * 50)
    lvar_stress = lvar_calc.calculate_liquidity_adjusted_var(
        market_var=market_var * 1.5,  # Market VaR increases
        position_value=portfolio_value,
        liquidity_metrics=liquidity,
        position_shares=portfolio_value / 100,
        stress_scenario=True
    )

    for component, value in lvar_stress.items():
        print(f"  {component:25s}: ${value:>12,.0f}")

    increase_pct = ((lvar_stress['total_lvar'] - lvar_normal['total_lvar']) /
                   lvar_normal['total_lvar'] * 100)
    print(f"\nL-VaR increase in stress: +{increase_pct:.1f}%")

    print()

    # =========================================================================
    # 7. VAR BACKTESTING
    # =========================================================================
    print("=" * 80)
    print("7. VaR MODEL BACKTESTING")
    print("=" * 80)
    print()

    # Simulate P&L
    actual_pnl = returns * portfolio_value

    # Use historical VaR predictions (constant for simplicity)
    predicted_var = np.full(len(returns), var_results['historical'])

    # Run backtest
    backtester = VaRBacktester(confidence_level=0.99)
    backtest_result = backtester.backtest(actual_pnl, predicted_var, dates)

    print("Backtest Results:")
    print("-" * 50)
    print(f"  Period:           {backtest_result.n_observations} days")
    print(f"  Confidence Level: {backtest_result.confidence_level * 100}%")
    print(f"  Violations:       {backtest_result.n_violations}")
    print(f"  Violation Rate:   {backtest_result.violation_rate * 100:.2f}%")
    print(f"  Expected Rate:    {(1-backtest_result.confidence_level) * 100:.2f}%")

    # Basel zone
    zone = backtester.calculate_basel_zone(backtest_result.n_violations, min(250, len(returns)))
    zone_emoji = {"green": "✅", "yellow": "⚠️", "red": "❌"}
    print(f"\nBasel Traffic Light: {zone_emoji[zone]} {zone.upper()}")

    # Statistical tests
    print("\nStatistical Tests:")
    print(f"  Kupiec POF:  p-value = {backtest_result.kupiec_pof_pvalue:.4f}", end="")
    print(f" ({'PASS ✅' if backtest_result.passed_kupiec else 'FAIL ❌'})")

    print(f"  Christoffersen: p-value = {backtest_result.christoffersen_pvalue:.4f}", end="")
    print(f" ({'PASS ✅' if backtest_result.passed_christoffersen else 'FAIL ❌'})")

    if backtest_result.avg_violation_size:
        print(f"\n  Avg Violation Size: ${backtest_result.avg_violation_size:,.0f}")

    print()

    # =========================================================================
    # 8. RISK SUMMARY AND RECOMMENDATIONS
    # =========================================================================
    print("=" * 80)
    print("8. RISK SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print()

    print("Current Risk Assessment:")
    print("-" * 50)

    # Risk level determination
    if current_regime in [MarketRegime.CRISIS, MarketRegime.BEAR]:
        risk_level = "HIGH"
        risk_emoji = "🔴"
    elif vol_ratio > 1.3:
        risk_level = "ELEVATED"
        risk_emoji = "🟡"
    else:
        risk_level = "NORMAL"
        risk_emoji = "🟢"

    print(f"  Overall Risk Level: {risk_emoji} {risk_level}")
    print(f"  Current Regime:     {current_regime.value.upper()}")
    print(f"  Volatility Status:  {vol_ratio:.2f}x long-run average")

    # Recommendations
    print("\nRecommendations:")
    print("-" * 50)

    if current_regime == MarketRegime.CRISIS:
        print("  1. ⚠️  CRISIS MODE: Reduce positions immediately")
        print("  2. ⚠️  Tighten VaR limits by 50%")
        print("  3. ⚠️  Increase hedging (consider options)")
        print("  4. ⚠️  Monitor for liquidity stress")
    elif current_regime == MarketRegime.BEAR:
        print("  1. Reduce risk exposure by 20-30%")
        print("  2. Consider defensive positioning")
        print("  3. Use wider risk limits for stop-losses")
    elif vol_ratio > 1.5:
        print("  1. Volatility elevated - consider reducing leverage")
        print("  2. Review position concentrations")
        print("  3. Update VaR using GARCH forecasts")
    else:
        print("  1. ✅ Risk levels normal")
        print("  2. ✅ Continue current strategy")
        print("  3. Monitor for regime changes")

    # Key metrics summary
    print("\nKey Metrics Summary:")
    print("-" * 50)
    print(f"  1-Day 99% VaR (Historical):    ${var_results['historical']:,.0f}")
    print(f"  1-Day 99% VaR (GARCH):         ${garch_var:,.0f}")
    print(f"  1-Day 97.5% Expected Shortfall: ${hist_es:,.0f}")
    print(f"  L-VaR (Normal):                ${lvar_normal['total_lvar']:,.0f}")
    print(f"  L-VaR (Stress):                ${lvar_stress['total_lvar']:,.0f}")
    print(f"  VaR as % of Portfolio:         {(var_results['historical']/portfolio_value)*100:.2f}%")

    print()
    print("=" * 80)
    print("END OF RISK ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
