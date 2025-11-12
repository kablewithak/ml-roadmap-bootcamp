# Real-Time Market Risk Management System

A production-grade real-time market risk management system implementing liquidity-adjusted Value-at-Risk (VaR), regime detection, and comprehensive risk analytics for multi-asset portfolios managing $500M+ AUM.

## 🎯 Project Overview

This system demonstrates enterprise-grade risk management capabilities used by major banks and hedge funds, combining advanced quantitative finance techniques with modern distributed computing infrastructure.

### Key Features

- **Advanced VaR Methodologies**: Historical simulation, parametric (variance-covariance), and Monte Carlo VaR
- **Liquidity-Adjusted VaR**: Incorporates bid-ask spreads, market impact (Almgren-Chriss), and liquidation time
- **Expected Shortfall (CVaR)**: Basel III compliant tail risk measurement
- **GARCH Volatility Modeling**: Time-varying volatility forecasting for conditional VaR
- **Market Regime Detection**: Hidden Markov Models (HMM) for bull/bear/crisis regime identification
- **Change Point Detection**: Real-time structural break detection using CUSUM, MOSUM, and PELT
- **Comprehensive Backtesting**: Kupiec POF and Christoffersen tests with Basel traffic light system
- **Component & Marginal VaR**: Granular risk attribution and marginal contribution analysis

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Risk Management System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Market Data  │  │   Regime     │  │     VaR      │      │
│  │  Ingestion   │→ │  Detection   │→ │  Calculator  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         ↓                  ↓                  ↓              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Liquidity   │  │   GARCH      │  │ Backtesting  │      │
│  │  Adjustment  │  │  Volatility  │  │  Framework   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for infrastructure)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ml-roadmap-bootcamp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start infrastructure (TimescaleDB, Redis, Pulsar, Grafana)
docker-compose up -d
```

### Basic Usage

```python
import numpy as np
from src.risk_engine.var_calculator import VaRCalculator
from src.risk_engine.expected_shortfall import ExpectedShortfallCalculator
from src.risk_engine.garch_volatility import GARCHVolatilityModel
from src.regime_detection.hmm_regime import HMMRegimeDetector

# Load your returns data
returns = np.random.normal(0.0005, 0.02, 252)  # Example: 1 year daily returns
portfolio_value = 10_000_000  # $10M portfolio

# 1. Calculate VaR using multiple methods
var_calc = VaRCalculator(confidence_level=0.99, horizon_days=1)
var_results = var_calc.calculate_all_var_methods(returns, portfolio_value)

print(f"Historical VaR: ${var_results['historical']:,.0f}")
print(f"Monte Carlo VaR: ${var_results['monte_carlo_bootstrap']:,.0f}")

# 2. Calculate Expected Shortfall
es_calc = ExpectedShortfallCalculator(confidence_level=0.975)
expected_shortfall = es_calc.calculate_historical_es(returns, portfolio_value)

print(f"Expected Shortfall (97.5%): ${expected_shortfall:,.0f}")

# 3. GARCH Volatility Forecast
garch = GARCHVolatilityModel(p=1, q=1)
garch.fit(returns)
garch_var, current_vol = garch.calculate_garch_var(returns, portfolio_value)

print(f"GARCH VaR: ${garch_var:,.0f}")
print(f"Current Volatility: {current_vol*100:.2f}%")

# 4. Detect Market Regime
regime_detector = HMMRegimeDetector(n_regimes=3)
regime_detector.fit(returns)
current_regime, regime_probs = regime_detector.predict_regime(returns)

print(f"\nCurrent Market Regime: {current_regime.value}")
print("Regime Probabilities:")
for regime, prob in regime_probs.items():
    print(f"  {regime.value}: {prob*100:.1f}%")
```

## 📁 Project Structure

```
ml-roadmap-bootcamp/
├── src/
│   ├── risk_engine/              # Core risk calculation modules
│   │   ├── var_calculator.py           # VaR calculations
│   │   ├── expected_shortfall.py       # Expected Shortfall (CVaR)
│   │   ├── liquidity_adjusted_var.py   # L-VaR with market impact
│   │   └── garch_volatility.py         # GARCH volatility models
│   ├── regime_detection/         # Market regime analysis
│   │   ├── hmm_regime.py               # Hidden Markov Models
│   │   └── change_point.py             # Change point detection
│   ├── backtesting/              # Model validation
│   │   └── var_backtest.py             # VaR backtesting framework
│   ├── portfolio_optimization/   # Portfolio construction (TBD)
│   ├── data_ingestion/           # Market data processing (TBD)
│   ├── monitoring/               # System monitoring (TBD)
│   ├── utils/                    # Shared utilities
│   │   └── data_structures.py          # Core data models
│   └── config.py                 # System configuration
├── tests/                        # Unit and integration tests
├── config/                       # Configuration files
│   ├── prometheus.yml                  # Metrics configuration
│   └── grafana/                        # Dashboard configs
├── docs/                         # Documentation
├── docker-compose.yml            # Infrastructure setup
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎓 Technical Deep Dive

### Value-at-Risk (VaR)

VaR answers: "What is the maximum loss I expect with X% confidence over Y time period?"

**Example**: 1-day 99% VaR of $1M means there's a 1% chance of losing more than $1M tomorrow.

#### Methods Implemented

1. **Historical Simulation**
   - Non-parametric (no distribution assumptions)
   - Uses actual historical returns
   - Captures fat tails and skewness
   - Implementation: `VaRCalculator.calculate_historical_var()`

2. **Parametric VaR (RiskMetrics)**
   - Assumes normal distribution
   - Fast computation
   - Formula: VaR = μ + σ × z × √(horizon)
   - Implementation: `VaRCalculator.calculate_parametric_var()`

3. **Monte Carlo Simulation**
   - Flexible (can model any distribution)
   - Good for complex portfolios
   - Computationally intensive
   - Implementation: `VaRCalculator.calculate_monte_carlo_var()`

### Liquidity-Adjusted VaR (L-VaR)

Standard VaR assumes you can liquidate instantly at mid-market price. Reality:
- Large positions have **market impact** (your selling moves prices)
- **Bid-ask spreads** widen in stress
- Liquidation takes **time** (participation rate limits)

**Formula**: L-VaR = Market VaR + Spread Cost + Market Impact + Timing Risk

**Almgren-Chriss Model**: Optimal execution balancing market impact vs. timing risk

```
Market Impact = η × (Position Size / ADV) × σ × √T
```

Implementation: `LiquidityAdjustedVaRCalculator`

### GARCH Volatility Modeling

Markets exhibit **volatility clustering**: large moves follow large moves, small follow small.

**GARCH(1,1) Model**:
```
σ²_t = ω + α × ε²_(t-1) + β × σ²_(t-1)
```

- ω: Long-run average variance
- α: Reaction to shocks (ARCH effect)
- β: Persistence of volatility
- α + β: Total persistence (typically ~0.95)

**Why GARCH matters**: VaR should use **current** volatility forecast, not historical average.

Implementation: `GARCHVolatilityModel`

### Market Regime Detection

Markets transition between distinct states:
- **Bull**: Low vol, positive drift, high correlations
- **Bear**: High vol, negative drift, flight to quality
- **Crisis**: Extreme vol, correlation breakdown
- **Neutral**: Normal conditions

**Hidden Markov Model (HMM)**:
- States are latent (can't directly observe regimes)
- Observable: returns, volatility, correlations
- Transition matrix: P(regime_t | regime_{t-1})
- Applications: Dynamic asset allocation, regime-dependent VaR scaling

Implementation: `HMMRegimeDetector`

### Expected Shortfall (ES)

ES answers: "In the worst X% of cases, what's my average loss?"

**Why ES > VaR**:
- ES is **coherent** (VaR is not)
- ES considers **severity** of tail losses, not just frequency
- **Basel III** replaced VaR with ES for market risk capital (2019)

**Formula**: ES = E[Loss | Loss > VaR]

Implementation: `ExpectedShortfallCalculator`

### Backtesting

**Regulatory requirement**: Must validate VaR models against actual losses

**Kupiec POF Test**: Tests if violation rate matches confidence level
- H0: p = p_expected
- LR ~ χ²(1)

**Christoffersen Test**: Tests if violations are independent
- H0: Violations don't cluster
- Important: Model can have correct coverage but still fail due to clustering

**Basel Traffic Light System**:
- **Green** (0-4 violations in 250 days): Model OK
- **Yellow** (5-9): Increase capital
- **Red** (10+): Model rejected

Implementation: `VaRBacktester`

## 📈 Performance Benchmarks

Target performance for production system:

| Metric | Target | Status |
|--------|--------|--------|
| Risk calculation latency (p99) | < 50ms | ⏳ TBD |
| Events per second | 500,000+ | ⏳ TBD |
| Regime detection lag | < 5 seconds | ⏳ TBD |
| System uptime | 99.99% | ⏳ TBD |
| VaR backtesting accuracy | Within 1% CI | ✅ Implemented |

## 🏛️ Regulatory Compliance

This system addresses key regulatory requirements:

- **Basel III** (BCBS 239): VaR/ES calculation and backtesting
- **FRTB** (Fundamental Review of Trading Book): Expected Shortfall, liquidity horizons
- **SR 11-7** (Federal Reserve): Model risk management
- **MiFID II**: Transaction reporting readiness
- **Dodd-Frank**: Risk aggregation and reporting

## 🔬 Mathematical Foundations

### Key Formulas

**Value-at-Risk (Historical)**:
```
VaR_α = -percentile(returns, α) × Portfolio_Value × √(horizon)
```

**Expected Shortfall**:
```
ES_α = -E[returns | returns ≤ percentile(returns, α)] × Portfolio_Value × √(horizon)
```

**Component VaR**:
```
CVaR_i = weight_i × β_i × Portfolio_VaR
where β_i = Cov(r_i, r_portfolio) / Var(r_portfolio)
```

**Market Impact (Almgren-Chriss)**:
```
Permanent Impact = η × (shares / ADV) × price
Temporary Impact = γ × (shares_per_period / ADV) × price
```

## 📚 References

### Academic Papers
- Almgren & Chriss (2000): "Optimal execution of portfolio transactions"
- Engle (1982): "Autoregressive Conditional Heteroskedasticity" (GARCH foundation)
- Kupiec (1995): "Techniques for Verifying the Accuracy of Risk Measurement Models"
- Christoffersen (1998): "Evaluating Interval Forecasts"
- Hamilton (1989): "A New Approach to the Economic Analysis of Nonstationary Time Series" (HMM)

### Regulatory Documents
- Basel Committee (2019): "Minimum capital requirements for market risk" (FRTB)
- Basel Committee (2013): "Principles for effective risk data aggregation" (BCBS 239)
- Federal Reserve SR 11-7: "Guidance on Model Risk Management"

### Books
- "Value at Risk: The New Benchmark for Managing Financial Risk" - Philippe Jorion
- "Quantitative Risk Management" - McNeil, Frey, Embrechts
- "Active Portfolio Management" - Grinold & Kahn

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_var_calculator.py -v

# Run with coverage
pytest --cov=src tests/

# Run backtests
python src/backtesting/var_backtest.py
```

## 🚧 Roadmap

### Completed ✅
- [x] Basic VaR calculations (Historical, Parametric, Monte Carlo)
- [x] Expected Shortfall (CVaR)
- [x] Liquidity-Adjusted VaR with Almgren-Chriss
- [x] Component VaR and Marginal VaR
- [x] GARCH volatility modeling
- [x] HMM regime detection
- [x] Change point detection (CUSUM, MOSUM, PELT)
- [x] VaR backtesting framework (Kupiec, Christoffersen)
- [x] Project infrastructure and configuration

### In Progress 🔨
- [ ] Market data ingestion pipeline (TimescaleDB, Pulsar)
- [ ] Real-time order book reconstruction
- [ ] Market microstructure features
- [ ] Options Greeks calculator

### Planned 📋
- [ ] Black-Litterman portfolio optimization
- [ ] Risk parity allocation
- [ ] Dynamic hedging system
- [ ] LSTM regime detection
- [ ] Copula-based dependency modeling
- [ ] Stress testing scenarios
- [ ] Real-time monitoring dashboards (Grafana)
- [ ] Production deployment guides

## 💡 Use Cases

### 1. Hedge Fund Risk Manager
**Scenario**: Monitor multi-strategy fund with $500M AUM

```python
# Daily risk monitoring
for strategy in strategies:
    var = calculate_var(strategy.positions)
    if var > strategy.risk_limit:
        send_alert(f"Strategy {strategy.id} exceeds VaR limit")
        auto_reduce_positions(strategy)
```

### 2. Proprietary Trading Desk
**Scenario**: Real-time position sizing based on regime

```python
# Regime-dependent position sizing
regime, probs = detector.predict_regime(recent_returns)

if regime == MarketRegime.CRISIS:
    max_position_size *= 0.5  # Cut risk in half
elif regime == MarketRegime.BULL:
    max_position_size *= 1.5  # Increase risk in calm markets
```

### 3. Risk Committee Reporting
**Scenario**: Monthly risk report for board

```python
# Generate comprehensive risk report
report = {
    'var_99': calculate_var(portfolio, 0.99),
    'expected_shortfall': calculate_es(portfolio, 0.975),
    'regime': current_regime,
    'var_violations_ytd': backtest_ytd(),
    'top_risk_contributors': component_var_analysis()
}
```

## 🤝 Contributing

This is a learning project demonstrating production-grade risk management systems. Contributions welcome!

## 📄 License

MIT License - See LICENSE file for details

## 📧 Contact

For questions about this implementation or risk management in general, please open an issue.

---

**Disclaimer**: This is an educational project. Not intended for actual trading or risk management without thorough testing and validation. Always consult qualified risk professionals before deploying any risk management system.
