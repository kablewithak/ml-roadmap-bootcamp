"""
Core data structures for market data, positions, and risk metrics.
These dataclasses provide type-safe representations of financial objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List
import numpy as np


class AssetClass(Enum):
    """Asset class categories"""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    FX = "fx"
    COMMODITY = "commodity"
    DERIVATIVE = "derivative"
    CRYPTO = "crypto"


class OrderSide(Enum):
    """Order side"""
    BID = "bid"
    ASK = "ask"


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"


class MarketRegime(Enum):
    """Market regime states"""
    BULL = "bull"  # Low volatility, positive returns
    BEAR = "bear"  # High volatility, negative returns
    NEUTRAL = "neutral"  # Normal volatility, mixed returns
    CRISIS = "crisis"  # Extreme volatility, liquidity crisis


@dataclass
class TickData:
    """
    Individual tick (trade or quote) from market data feed.
    Represents the most granular level of market data.
    """
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    side: Optional[OrderSide] = None
    exchange: Optional[str] = None
    sequence_number: Optional[int] = None

    def __post_init__(self):
        """Validate tick data"""
        if self.price <= 0:
            raise ValueError(f"Price must be positive: {self.price}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")


@dataclass
class OrderBookLevel:
    """Single level in the order book"""
    price: float
    volume: float
    num_orders: int = 1


@dataclass
class OrderBook:
    """
    Full order book reconstruction for a symbol.
    Maintains bid/ask levels for liquidity analysis.
    """
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        """Highest bid price"""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest ask price"""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid-market price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread in absolute terms"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Bid-ask spread in basis points"""
        if self.spread and self.mid_price and self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return None

    def total_bid_volume(self, levels: int = 10) -> float:
        """Total volume on bid side"""
        return sum(level.volume for level in self.bids[:levels])

    def total_ask_volume(self, levels: int = 10) -> float:
        """Total volume on ask side"""
        return sum(level.volume for level in self.asks[:levels])

    def order_flow_imbalance(self, levels: int = 10) -> float:
        """
        Order flow imbalance: (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
        Positive values indicate buying pressure, negative indicates selling pressure.
        """
        bid_vol = self.total_bid_volume(levels)
        ask_vol = self.total_ask_volume(levels)
        total = bid_vol + ask_vol

        if total > 0:
            return (bid_vol - ask_vol) / total
        return 0.0


@dataclass
class Position:
    """
    Represents a portfolio position in a single instrument.
    """
    symbol: str
    asset_class: AssetClass
    quantity: float
    entry_price: float
    current_price: float
    timestamp: datetime
    currency: str = "USD"
    strategy_id: Optional[str] = None
    trader_id: Optional[str] = None

    @property
    def side(self) -> PositionSide:
        """Position side based on quantity"""
        return PositionSide.LONG if self.quantity >= 0 else PositionSide.SHORT

    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        """Unrealized P&L"""
        return self.quantity * (self.current_price - self.entry_price)

    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of entry value"""
        if self.entry_price != 0:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        return 0.0


@dataclass
class Portfolio:
    """
    Collection of positions representing a portfolio or sub-portfolio.
    """
    portfolio_id: str
    positions: Dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    timestamp: Optional[datetime] = None

    @property
    def total_value(self) -> float:
        """Total portfolio value (positions + cash)"""
        return sum(pos.market_value for pos in self.positions.values()) + self.cash

    @property
    def total_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(pos.pnl for pos in self.positions.values())

    def get_positions_by_asset_class(self, asset_class: AssetClass) -> List[Position]:
        """Filter positions by asset class"""
        return [pos for pos in self.positions.values() if pos.asset_class == asset_class]

    def get_exposure(self, asset_class: Optional[AssetClass] = None) -> float:
        """
        Calculate gross exposure (sum of absolute values).
        Optionally filter by asset class.
        """
        positions = (self.get_positions_by_asset_class(asset_class)
                    if asset_class else self.positions.values())
        return sum(abs(pos.market_value) for pos in positions)

    def add_position(self, position: Position):
        """Add or update position"""
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str):
        """Remove position"""
        if symbol in self.positions:
            del self.positions[symbol]


@dataclass
class RiskMetrics:
    """
    Comprehensive risk metrics for a portfolio or position.
    """
    timestamp: datetime
    portfolio_id: str

    # VaR metrics
    var_historical: Optional[float] = None
    var_parametric: Optional[float] = None
    var_monte_carlo: Optional[float] = None
    var_liquidity_adjusted: Optional[float] = None

    # Expected Shortfall (CVaR)
    expected_shortfall: Optional[float] = None

    # Component VaR (contribution of each position to portfolio VaR)
    component_var: Dict[str, float] = field(default_factory=dict)

    # Marginal VaR (impact of small position change)
    marginal_var: Dict[str, float] = field(default_factory=dict)

    # Volatility metrics
    volatility_daily: Optional[float] = None
    volatility_annualized: Optional[float] = None

    # Correlation metrics
    avg_correlation: Optional[float] = None
    max_correlation: Optional[float] = None

    # Greeks (for options portfolios)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None

    # Regime information
    current_regime: Optional[MarketRegime] = None
    regime_probabilities: Dict[MarketRegime, float] = field(default_factory=dict)

    # Liquidity metrics
    avg_bid_ask_spread_bps: Optional[float] = None
    liquidation_time_days: Optional[float] = None
    market_impact_bps: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_id': self.portfolio_id,
            'var_historical': self.var_historical,
            'var_parametric': self.var_parametric,
            'var_monte_carlo': self.var_monte_carlo,
            'var_liquidity_adjusted': self.var_liquidity_adjusted,
            'expected_shortfall': self.expected_shortfall,
            'volatility_daily': self.volatility_daily,
            'current_regime': self.current_regime.value if self.current_regime else None,
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
        }


@dataclass
class MarketMicrostructureFeatures:
    """
    Market microstructure features extracted from tick data.
    Used for liquidity analysis and high-frequency trading signals.
    """
    symbol: str
    timestamp: datetime

    # Spread metrics
    bid_ask_spread: float
    bid_ask_spread_bps: float
    effective_spread: Optional[float] = None  # Actual execution vs mid

    # Volume metrics
    trade_intensity: float  # Trades per minute
    volume_imbalance: float  # Buy volume - Sell volume

    # Order flow
    order_flow_imbalance: float

    # Price impact
    price_impact_bps: Optional[float] = None

    # Volatility
    realized_volatility: Optional[float] = None

    # Other
    tick_direction: Optional[int] = None  # +1 uptick, -1 downtick, 0 no change


@dataclass
class StressScenario:
    """
    Definition of a stress testing scenario.
    """
    name: str
    description: str

    # Shocks to apply (symbol -> percentage change)
    price_shocks: Dict[str, float] = field(default_factory=dict)

    # Volatility multipliers
    volatility_multiplier: float = 1.0

    # Correlation adjustments
    correlation_shift: float = 0.0  # Add to all correlations

    # Liquidity adjustments
    spread_multiplier: float = 1.0  # Multiply all spreads

    # Market regime
    forced_regime: Optional[MarketRegime] = None


@dataclass
class BacktestResult:
    """
    Results from VaR backtesting.
    """
    start_date: datetime
    end_date: datetime
    confidence_level: float

    # Statistics
    n_observations: int
    n_violations: int
    violation_rate: float

    # Test statistics
    kupiec_pof_statistic: Optional[float] = None  # Proportion of Failures test
    kupiec_pof_pvalue: Optional[float] = None

    christoffersen_statistic: Optional[float] = None  # Independence test
    christoffersen_pvalue: Optional[float] = None

    # Additional metrics
    avg_var: float = 0.0
    max_var: float = 0.0
    avg_violation_size: Optional[float] = None

    @property
    def passed_kupiec(self, alpha: float = 0.05) -> bool:
        """Check if passed Kupiec test at given significance level"""
        if self.kupiec_pof_pvalue is None:
            return False
        return self.kupiec_pof_pvalue > alpha

    @property
    def passed_christoffersen(self, alpha: float = 0.05) -> bool:
        """Check if passed Christoffersen test"""
        if self.christoffersen_pvalue is None:
            return False
        return self.christoffersen_pvalue > alpha
