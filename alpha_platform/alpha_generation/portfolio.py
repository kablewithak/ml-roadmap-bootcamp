"""
Portfolio construction with hierarchical risk parity and optimization.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioConstructor:
    """
    Construct optimal portfolios using various methods.

    Supports:
    - Hierarchical Risk Parity (HRP)
    - Mean-variance optimization
    - Risk parity
    - Equal weight
    """

    def __init__(
        self,
        method: str = "hierarchical_risk_parity",
        max_position_size: float = 0.05,
        max_sector_exposure: float = 0.20,
    ):
        """
        Initialize portfolio constructor.

        Args:
            method: Portfolio construction method
            max_position_size: Maximum position size per asset
            max_sector_exposure: Maximum exposure per sector
        """
        self.config = get_config()
        self.method = method
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure

        logger.info(f"Portfolio constructor initialized with method: {method}")

    def construct_portfolio(
        self,
        signals: pd.Series,
        returns: pd.DataFrame,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> pd.Series:
        """
        Construct portfolio weights.

        Args:
            signals: Alpha signals for each asset
            returns: Historical returns for covariance estimation
            constraints: Optional additional constraints

        Returns:
            Portfolio weights
        """
        logger.info(f"Constructing portfolio for {len(signals)} assets")

        if self.method == "hierarchical_risk_parity":
            weights = self._hrp(returns)
        elif self.method == "equal_weight":
            weights = pd.Series(1.0 / len(signals), index=signals.index)
        elif self.method == "signal_weighted":
            # Weight by signal strength
            abs_signals = signals.abs()
            weights = abs_signals / abs_signals.sum()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply signal direction
        weights = weights * np.sign(signals)

        # Apply constraints
        weights = self._apply_constraints(weights)

        logger.info(f"Portfolio constructed: {len(weights[weights != 0])} positions")

        return weights

    def _hrp(self, returns: pd.DataFrame) -> pd.Series:
        """
        Hierarchical Risk Parity portfolio construction.

        Args:
            returns: Historical returns

        Returns:
            Portfolio weights
        """
        # Calculate correlation matrix
        corr = returns.corr()

        # Calculate distance matrix
        dist = np.sqrt((1 - corr) / 2)

        # Hierarchical clustering
        link = linkage(squareform(dist.values), method="single")

        # Get sorted indices
        sorted_idx = self._get_quasi_diag(link)

        # Recursive bisection
        weights = self._recursive_bisection(returns.iloc[:, sorted_idx])

        # Reorder to original order
        weights = pd.Series(weights, index=returns.columns[sorted_idx])
        weights = weights.reindex(returns.columns)

        return weights

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Get quasi-diagonal order from linkage matrix."""
        # Number of original observations
        n = link.shape[0] + 1

        # Initialize with first cluster
        sorted_idx = [n]

        # Recursive sorting
        self._quasi_diag_recursive(link, sorted_idx)

        # Convert cluster IDs to indices
        result = []
        for idx in sorted_idx:
            if idx < n:
                result.append(idx)

        return result

    def _quasi_diag_recursive(self, link: np.ndarray, sorted_idx: List[int]) -> None:
        """Recursively sort clusters."""
        n = link.shape[0] + 1

        if len(sorted_idx) == 0:
            return

        # Get last cluster
        cluster_id = sorted_idx.pop()

        if cluster_id < n:
            # Original observation
            sorted_idx.append(cluster_id)
        else:
            # Merged cluster
            left, right = int(link[cluster_id - n, 0]), int(link[cluster_id - n, 1])
            sorted_idx.extend([left, right])
            self._quasi_diag_recursive(link, sorted_idx)

    def _recursive_bisection(self, returns: pd.DataFrame) -> np.ndarray:
        """Allocate weights using recursive bisection."""
        weights = pd.Series(1.0, index=returns.columns)
        clusters = [returns.columns]

        while len(clusters) > 0:
            # Split first cluster
            cluster = clusters.pop(0)

            if len(cluster) == 1:
                continue

            # Split cluster in half
            mid = len(cluster) // 2
            cluster1 = cluster[:mid]
            cluster2 = cluster[mid:]

            # Calculate cluster variances
            var1 = self._get_cluster_var(returns[cluster1])
            var2 = self._get_cluster_var(returns[cluster2])

            # Allocate weights inversely proportional to variance
            alpha = 1 - var1 / (var1 + var2)

            weights[cluster1] *= alpha
            weights[cluster2] *= (1 - alpha)

            # Add sub-clusters for further splitting
            if len(cluster1) > 1:
                clusters.append(cluster1)
            if len(cluster2) > 1:
                clusters.append(cluster2)

        return weights.values

    def _get_cluster_var(self, returns: pd.DataFrame) -> float:
        """Calculate cluster variance."""
        cov = returns.cov()
        weights = np.ones(len(returns.columns)) / len(returns.columns)
        variance = weights.dot(cov).dot(weights)
        return variance

    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply position size constraints."""
        # Apply max position size
        weights = weights.clip(-self.max_position_size, self.max_position_size)

        # Renormalize
        total_abs_weight = weights.abs().sum()
        if total_abs_weight > 1.0:
            weights = weights / total_abs_weight

        return weights
