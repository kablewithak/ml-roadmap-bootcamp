"""Risk management for portfolio and trading."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Manage trading and portfolio risk.

    Features:
    - Position limits
    - Drawdown controls
    - VaR monitoring
    - Stop losses
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,
        var_limit: float = 0.02,
        confidence_level: float = 0.99,
    ):
        """
        Initialize risk manager.

        Args:
            max_drawdown: Maximum allowable drawdown
            var_limit: Daily VaR limit
            confidence_level: VaR confidence level
        """
        self.config = get_config()
        self.max_drawdown = max_drawdown
        self.var_limit = var_limit
        self.confidence_level = confidence_level

        logger.info("Risk manager initialized")

    def calculate_var(
        self, returns: pd.Series, method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Historical returns
            method: VaR calculation method

        Returns:
            VaR value
        """
        if method == "historical":
            var = np.percentile(returns, (1 - self.confidence_level) * 100)
        else:
            # Parametric VaR
            mean = returns.mean()
            std = returns.std()
            var = mean - 2.33 * std  # 99% confidence

        return float(var)

    def check_risk_limits(
        self,
        portfolio_value: float,
        peak_value: float,
        var: float,
    ) -> Dict[str, bool]:
        """
        Check if risk limits are exceeded.

        Returns:
            Dictionary of risk check results
        """
        # Drawdown check
        drawdown = (peak_value - portfolio_value) / peak_value
        drawdown_ok = drawdown < self.max_drawdown

        # VaR check
        var_ok = abs(var) < self.var_limit

        return {
            "drawdown_ok": drawdown_ok,
            "var_ok": var_ok,
            "all_ok": drawdown_ok and var_ok,
            "drawdown": float(drawdown),
            "var": float(var),
        }
