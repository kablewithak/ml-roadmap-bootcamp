"""
Change Point Detection for Market Regime Shifts

Change points are moments when statistical properties of data suddenly change.
In finance, these correspond to:
- Regime transitions (bull → bear)
- Policy changes (interest rate announcements)
- Black swan events (crashes, crises)
- Structural breaks (new regulations)

Methods implemented:
1. CUSUM (Cumulative Sum): Detects changes in mean
2. MOSUM (Moving Sum): Similar to CUSUM but more robust
3. Binary Segmentation: Recursive partitioning
4. PELT (Pruned Exact Linear Time): Optimal segmentation

Applications:
- Real-time crisis detection
- Model retraining triggers
- Risk limit adjustments
- Portfolio rebalancing signals
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import ruptures as rpt
from scipy import stats


class ChangePointDetector:
    """
    Detect change points in time series data using multiple algorithms.

    Change points indicate regime shifts that require immediate action:
    - Increase risk limits in calm periods
    - Decrease risk limits when volatility jumps
    - Retrain models when data distribution changes
    """

    def __init__(
        self,
        method: str = 'pelt',
        penalty_value: float = 1.0,
        min_segment_length: int = 20
    ):
        """
        Initialize change point detector.

        Args:
            method: Detection algorithm ('pelt', 'binseg', 'window')
            penalty_value: Penalty for adding change points (higher = fewer CPs)
            min_segment_length: Minimum length between change points
        """
        self.method = method
        self.penalty = penalty_value
        self.min_size = min_segment_length

    def detect_mean_changes(
        self,
        data: np.ndarray,
        n_changepoints: Optional[int] = None
    ) -> List[int]:
        """
        Detect changes in mean level.

        Useful for:
        - Detecting drift changes in returns
        - Identifying trend shifts
        - Finding structural breaks

        Args:
            data: Time series data
            n_changepoints: Number of change points (None = automatic)

        Returns:
            List of change point indices
        """
        # Reshape if needed
        if data.ndim == 1:
            signal = data.reshape(-1, 1)
        else:
            signal = data

        # Select algorithm
        if self.method == 'pelt':
            algo = rpt.Pelt(model='rbf', min_size=self.min_size)
        elif self.method == 'binseg':
            algo = rpt.Binseg(model='l2', min_size=self.min_size)
        elif self.method == 'window':
            algo = rpt.Window(width=50, model='l2', min_size=self.min_size)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Fit model
        algo.fit(signal)

        # Detect change points
        if n_changepoints is not None:
            change_points = algo.predict(n_bkps=n_changepoints)
        else:
            # Automatic detection using penalty
            change_points = algo.predict(pen=self.penalty * np.log(len(data)))

        # Remove the last point (end of series)
        if change_points and change_points[-1] == len(data):
            change_points = change_points[:-1]

        return change_points

    def detect_variance_changes(
        self,
        data: np.ndarray,
        n_changepoints: Optional[int] = None
    ) -> List[int]:
        """
        Detect changes in variance (volatility regime shifts).

        Critical for risk management:
        - Volatility regime shifts require immediate VaR adjustment
        - Detect crisis onset (volatility spike)
        - Identify calm periods (reduce hedging costs)

        Args:
            data: Time series data
            n_changepoints: Number of change points

        Returns:
            List of change point indices
        """
        # Square the data to focus on variance
        squared_data = data ** 2

        # Use PELT with RBF kernel (sensitive to variance changes)
        algo = rpt.Pelt(model='rbf', min_size=self.min_size)

        signal = squared_data.reshape(-1, 1) if squared_data.ndim == 1 else squared_data
        algo.fit(signal)

        if n_changepoints is not None:
            change_points = algo.predict(n_bkps=n_changepoints)
        else:
            change_points = algo.predict(pen=self.penalty * np.log(len(data)))

        if change_points and change_points[-1] == len(data):
            change_points = change_points[:-1]

        return change_points

    def cusum(
        self,
        data: np.ndarray,
        threshold: float = 5.0,
        drift: float = 0.0
    ) -> Tuple[List[int], np.ndarray]:
        """
        CUSUM (Cumulative Sum) change point detection.

        Classic quality control method adapted for finance.
        Detects changes in mean by tracking cumulative deviations.

        Formula:
        S_t = max(0, S_(t-1) + (x_t - μ - drift))

        Alert when S_t > threshold

        Args:
            data: Time series
            threshold: Detection threshold (in standard deviations)
            drift: Drift parameter (allowance for small changes)

        Returns:
            Tuple of (change_point_indices, cumsum_statistic)
        """
        n = len(data)

        # Center data
        mu = np.mean(data)
        sigma = np.std(data)
        centered = (data - mu) / sigma if sigma > 0 else data - mu

        # CUSUM statistics (two-sided)
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)

        change_points = []

        for t in range(1, n):
            # Positive CUSUM (upward shift)
            cusum_pos[t] = max(0, cusum_pos[t-1] + centered[t] - drift)

            # Negative CUSUM (downward shift)
            cusum_neg[t] = max(0, cusum_neg[t-1] - centered[t] - drift)

            # Check for change point
            if cusum_pos[t] > threshold or cusum_neg[t] > threshold:
                change_points.append(t)
                # Reset CUSUM
                cusum_pos[t] = 0
                cusum_neg[t] = 0

        # Combine both statistics
        cusum_stat = np.maximum(cusum_pos, cusum_neg)

        return change_points, cusum_stat

    def mosum(
        self,
        data: np.ndarray,
        window: int = 50,
        threshold: float = 3.0
    ) -> Tuple[List[int], np.ndarray]:
        """
        MOSUM (Moving Sum) change point detection.

        Similar to CUSUM but uses moving window (more robust to outliers).

        Args:
            data: Time series
            window: Window size for moving sum
            threshold: Detection threshold

        Returns:
            Tuple of (change_point_indices, mosum_statistic)
        """
        n = len(data)

        # Standardize
        mu = np.mean(data)
        sigma = np.std(data)
        standardized = (data - mu) / sigma if sigma > 0 else data - mu

        # MOSUM statistic
        mosum_stat = np.zeros(n)
        change_points = []

        for t in range(window, n - window):
            # Left window
            left = standardized[t-window:t]

            # Right window
            right = standardized[t:t+window]

            # MOSUM statistic (difference in means)
            mosum_stat[t] = abs(np.mean(right) - np.mean(left)) * np.sqrt(window / 2)

            # Check threshold
            if mosum_stat[t] > threshold:
                change_points.append(t)

        return change_points, mosum_stat

    def get_segment_statistics(
        self,
        data: np.ndarray,
        change_points: List[int]
    ) -> pd.DataFrame:
        """
        Calculate statistics for each segment between change points.

        Args:
            data: Time series
            change_points: List of change point indices

        Returns:
            DataFrame with segment statistics
        """
        segments = []

        # Add start and end
        boundaries = [0] + sorted(change_points) + [len(data)]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            segment_data = data[start:end]

            if len(segment_data) > 0:
                segments.append({
                    'segment': i,
                    'start_idx': start,
                    'end_idx': end,
                    'length': len(segment_data),
                    'mean': np.mean(segment_data),
                    'std': np.std(segment_data),
                    'min': np.min(segment_data),
                    'max': np.max(segment_data),
                    'skewness': pd.Series(segment_data).skew(),
                    'kurtosis': pd.Series(segment_data).kurtosis(),
                })

        return pd.DataFrame(segments)

    def detect_volatility_breakout(
        self,
        returns: np.ndarray,
        window: int = 20,
        n_std: float = 2.0
    ) -> List[int]:
        """
        Detect volatility breakouts (sudden vol spikes).

        Real-time crisis detection:
        - When volatility exceeds historical mean + n*std
        - Indicates regime shift or market stress

        Used for:
        - Automatic risk limit tightening
        - Circuit breaker triggers
        - Alert generation

        Args:
            returns: Return series
            window: Rolling window for volatility calculation
            n_std: Number of standard deviations for threshold

        Returns:
            List of breakout indices
        """
        # Calculate rolling volatility
        vol = pd.Series(returns).rolling(window=window, min_periods=1).std().values

        # Calculate threshold (mean + n*std)
        vol_mean = np.mean(vol)
        vol_std = np.std(vol)
        threshold = vol_mean + n_std * vol_std

        # Find breakouts
        breakouts = np.where(vol > threshold)[0].tolist()

        # Filter to only include start of each breakout period
        filtered_breakouts = []
        for i, idx in enumerate(breakouts):
            if i == 0 or idx > breakouts[i-1] + 1:
                filtered_breakouts.append(idx)

        return filtered_breakouts

    def online_change_detection(
        self,
        historical_data: np.ndarray,
        new_data: np.ndarray,
        method: str = 'cusum'
    ) -> bool:
        """
        Online change detection for real-time monitoring.

        Returns True if new data indicates a regime change.

        Args:
            historical_data: Historical baseline
            new_data: Recent observations
            method: 'cusum' or 'likelihood_ratio'

        Returns:
            True if change detected
        """
        if method == 'cusum':
            # Combine data
            combined = np.concatenate([historical_data, new_data])

            # Run CUSUM
            change_points, _ = self.cusum(combined, threshold=4.0)

            # Check if any change points in new data region
            new_data_start = len(historical_data)
            recent_changes = [cp for cp in change_points if cp >= new_data_start]

            return len(recent_changes) > 0

        elif method == 'likelihood_ratio':
            # Likelihood ratio test for distribution change
            # H0: new_data comes from same distribution as historical_data

            # Calculate statistics
            hist_mean = np.mean(historical_data)
            hist_std = np.std(historical_data)

            new_mean = np.mean(new_data)
            new_std = np.std(new_data)

            # Z-test for mean difference
            n1, n2 = len(historical_data), len(new_data)
            pooled_std = np.sqrt((hist_std**2 / n1) + (new_std**2 / n2))

            if pooled_std > 0:
                z_stat = abs(new_mean - hist_mean) / pooled_std
                p_value = 2 * (1 - stats.norm.cdf(z_stat))

                # Reject H0 if p < 0.01 (strong evidence of change)
                return p_value < 0.01
            else:
                return False

        else:
            raise ValueError(f"Unknown method: {method}")


# Example usage
if __name__ == "__main__":
    # Generate synthetic data with regime changes
    np.random.seed(42)

    # Regime 1: Low vol (300 points)
    regime1 = np.random.normal(0.001, 0.01, 300)

    # Regime 2: High vol, negative mean (200 points)
    regime2 = np.random.normal(-0.002, 0.03, 200)

    # Regime 3: Return to normal (200 points)
    regime3 = np.random.normal(0.001, 0.015, 200)

    # Crisis: Extreme volatility (100 points)
    crisis = np.random.normal(-0.005, 0.06, 100)

    # Combine
    data = np.concatenate([regime1, regime2, regime3, crisis])

    print("=== Change Point Detection ===\n")

    # Initialize detector
    detector = ChangePointDetector(method='pelt', penalty_value=2.0)

    # Detect mean changes
    print("Detecting Mean Change Points...")
    mean_cps = detector.detect_mean_changes(data)
    print(f"Found {len(mean_cps)} change points at indices: {mean_cps}")
    print(f"True change points: [300, 500, 700]")

    # Detect variance changes
    print("\nDetecting Variance Change Points...")
    var_cps = detector.detect_variance_changes(data)
    print(f"Found {len(var_cps)} change points at indices: {var_cps}")

    # CUSUM
    print("\nCUSUM Detection...")
    cusum_cps, cusum_stat = detector.cusum(data, threshold=5.0)
    print(f"Found {len(cusum_cps)} change points: {cusum_cps}")

    # Segment statistics
    print("\nSegment Statistics:")
    stats_df = detector.get_segment_statistics(data, mean_cps)
    print(stats_df.to_string(index=False))

    # Volatility breakout detection
    print("\nVolatility Breakouts:")
    breakouts = detector.detect_volatility_breakout(data, window=20, n_std=2.0)
    print(f"Detected {len(breakouts)} volatility breakouts")

    # Online detection
    print("\nOnline Change Detection:")
    historical = data[:600]
    new_observations = data[700:750]  # Crisis period

    change_detected = detector.online_change_detection(
        historical, new_observations, method='cusum'
    )
    print(f"Change detected in new data: {change_detected}")
