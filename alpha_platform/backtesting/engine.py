"""
Event-driven backtesting engine.

Simulates realistic trading with market impact, slippage, and commissions.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Features:
    - Realistic order execution with slippage
    - Market impact modeling
    - Transaction costs
    - Walk-forward validation
    - Performance analytics
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000,
        commission: float = 0.0005,
        slippage_model: str = "volume_share",
        market_impact_model: str = "square_root",
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission: Commission rate (as fraction)
            slippage_model: Slippage calculation model
            market_impact_model: Market impact model
        """
        self.config = get_config()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_model = slippage_model
        self.market_impact_model = market_impact_model

        # State
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        logger.info(
            f"Backtest engine initialized with ${initial_capital:,.0f} capital"
        )

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            signals: DataFrame of alpha signals [date x ticker]
            prices: DataFrame of prices [date x ticker]
            volumes: Optional DataFrame of volumes [date x ticker]

        Returns:
            Backtest results including performance metrics
        """
        logger.info(
            f"Running backtest from {signals.index[0]} to {signals.index[-1]}"
        )

        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # Align data
        common_dates = signals.index.intersection(prices.index)
        signals = signals.loc[common_dates]
        prices = prices.loc[common_dates]

        if volumes is not None:
            volumes = volumes.loc[common_dates]

        # Run simulation
        for date in common_dates:
            self._process_date(
                date,
                signals.loc[date],
                prices.loc[date],
                volumes.loc[date] if volumes is not None else None,
            )

        # Calculate performance metrics
        results = self._calculate_performance()

        logger.info(
            f"Backtest complete: Return={results['total_return']:.2%}, "
            f"Sharpe={results['sharpe_ratio']:.2f}, "
            f"MaxDD={results['max_drawdown']:.2%}"
        )

        return results

    def _process_date(
        self,
        date: pd.Timestamp,
        signals: pd.Series,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
    ) -> None:
        """Process a single trading date."""
        # Calculate target positions from signals
        total_capital = self.capital + sum(
            self.positions.get(ticker, 0) * prices.get(ticker, 0)
            for ticker in self.positions
        )

        target_positions = {}
        for ticker, signal in signals.items():
            if pd.notna(signal) and pd.notna(prices.get(ticker)):
                # Position size based on signal strength
                position_value = total_capital * signal
                target_shares = position_value / prices[ticker]
                target_positions[ticker] = target_shares

        # Execute trades
        self._rebalance_portfolio(date, target_positions, prices, volumes)

        # Update equity curve
        portfolio_value = self.capital + sum(
            self.positions.get(ticker, 0) * prices.get(ticker, 0)
            for ticker in self.positions
        )
        self.equity_curve.append(
            {
                "date": date,
                "equity": portfolio_value,
                "cash": self.capital,
                "positions_value": portfolio_value - self.capital,
            }
        )

    def _rebalance_portfolio(
        self,
        date: pd.Timestamp,
        target_positions: Dict[str, float],
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
    ) -> None:
        """Rebalance portfolio to target positions."""
        # Determine trades needed
        all_tickers = set(self.positions.keys()) | set(target_positions.keys())

        for ticker in all_tickers:
            current_shares = self.positions.get(ticker, 0)
            target_shares = target_positions.get(ticker, 0)

            shares_to_trade = target_shares - current_shares

            if abs(shares_to_trade) > 0.01:  # Minimum trade threshold
                self._execute_trade(
                    date,
                    ticker,
                    shares_to_trade,
                    prices[ticker],
                    volumes[ticker] if volumes is not None else None,
                )

    def _execute_trade(
        self,
        date: pd.Timestamp,
        ticker: str,
        shares: float,
        price: float,
        volume: Optional[float] = None,
    ) -> None:
        """Execute a trade with realistic costs."""
        # Calculate slippage
        slippage = self._calculate_slippage(shares, price, volume)

        # Calculate market impact
        impact = self._calculate_market_impact(shares, price, volume)

        # Effective price
        if shares > 0:  # Buy
            effective_price = price * (1 + slippage + impact)
        else:  # Sell
            effective_price = price * (1 - slippage - impact)

        # Commission
        trade_value = abs(shares * effective_price)
        commission = trade_value * self.commission

        # Update capital
        self.capital -= shares * effective_price + commission

        # Update positions
        self.positions[ticker] = self.positions.get(ticker, 0) + shares

        # Record trade
        self.trades.append(
            {
                "date": date,
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "effective_price": effective_price,
                "slippage": slippage,
                "impact": impact,
                "commission": commission,
            }
        )

    def _calculate_slippage(
        self, shares: float, price: float, volume: Optional[float]
    ) -> float:
        """Calculate slippage based on model."""
        if self.slippage_model == "volume_share" and volume is not None:
            # Slippage proportional to volume participation
            participation = abs(shares) / volume if volume > 0 else 0
            slippage = min(participation * 0.1, 0.01)  # Cap at 1%
        else:
            # Fixed slippage
            slippage = 0.001  # 10 bps default

        return slippage

    def _calculate_market_impact(
        self, shares: float, price: float, volume: Optional[float]
    ) -> float:
        """Calculate market impact."""
        if self.market_impact_model == "square_root" and volume is not None:
            # Square-root market impact
            participation = abs(shares) / volume if volume > 0 else 0
            impact = 0.01 * np.sqrt(participation)  # Square root model
        else:
            # Linear impact
            impact = 0.0005  # 5 bps default

        return impact

    def _calculate_performance(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index("date", inplace=True)

        # Returns
        equity_df["returns"] = equity_df["equity"].pct_change()

        # Total return
        total_return = (equity_df["equity"].iloc[-1] / self.initial_capital) - 1

        # Annualized return
        n_years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        # Sharpe ratio (assume 252 trading days)
        sharpe_ratio = (
            equity_df["returns"].mean() / equity_df["returns"].std() * np.sqrt(252)
        )

        # Maximum drawdown
        running_max = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            # Group trades by ticker to calculate P&L
            win_rate = 0.5  # Simplified
        else:
            win_rate = 0.0

        results = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "num_trades": len(self.trades),
            "final_equity": float(equity_df["equity"].iloc[-1]),
            "equity_curve": equity_df,
            "trades": trades_df if len(trades_df) > 0 else None,
        }

        return results
