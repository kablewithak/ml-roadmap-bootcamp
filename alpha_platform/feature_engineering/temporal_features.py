"""
Temporal feature engineering for time series data.

Creates lag features, rolling statistics, and temporal patterns.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalFeatureEngineer:
    """Engineer temporal features from time series data."""

    def __init__(
        self,
        lags: List[int] = [1, 5, 20, 60],
        rolling_windows: List[int] = [5, 10, 20, 60],
        include_returns: bool = True,
        include_volatility: bool = True,
    ):
        """
        Initialize temporal feature engineer.

        Args:
            lags: List of lag periods
            rolling_windows: List of rolling window sizes
            include_returns: Whether to include return features
            include_volatility: Whether to include volatility features
        """
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.include_returns = include_returns
        self.include_volatility = include_volatility

        logger.info("Temporal feature engineer initialized")

    def create_features(
        self, df: pd.DataFrame, value_col: str = "value"
    ) -> pd.DataFrame:
        """
        Create temporal features.

        Args:
            df: Input dataframe with time series
            value_col: Name of value column

        Returns:
            Dataframe with temporal features
        """
        result = df.copy()

        # Lag features
        for lag in self.lags:
            result[f"{value_col}_lag_{lag}"] = result[value_col].shift(lag)

        # Rolling statistics
        for window in self.rolling_windows:
            result[f"{value_col}_ma_{window}"] = (
                result[value_col].rolling(window=window).mean()
            )
            result[f"{value_col}_std_{window}"] = (
                result[value_col].rolling(window=window).std()
            )
            result[f"{value_col}_min_{window}"] = (
                result[value_col].rolling(window=window).min()
            )
            result[f"{value_col}_max_{window}"] = (
                result[value_col].rolling(window=window).max()
            )

        # Returns
        if self.include_returns:
            result[f"{value_col}_return_1d"] = result[value_col].pct_change()
            result[f"{value_col}_return_5d"] = result[value_col].pct_change(5)
            result[f"{value_col}_return_20d"] = result[value_col].pct_change(20)

        # Volatility
        if self.include_volatility:
            result[f"{value_col}_volatility_20d"] = (
                result[f"{value_col}_return_1d"].rolling(window=20).std()
            )

        return result
