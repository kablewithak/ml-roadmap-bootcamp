"""
Hidden Markov Model (HMM) for Market Regime Detection

Market regimes are distinct states that markets transition between:
- Bull regime: Low volatility, positive drift, high correlations
- Bear regime: High volatility, negative drift, flight to quality
- Neutral: Moderate volatility, sideways movement
- Crisis: Extreme volatility, correlation breakdown/convergence

HMM is perfect for regime detection because:
1. Markets have latent (hidden) states we can't directly observe
2. States persist over time (Markov property)
3. Observable features (returns, volatility) depend on hidden state
4. Provides probabilistic state estimates (not binary)

Applications:
- Adjust risk limits based on regime (tighter in bear markets)
- Dynamic asset allocation (regime-dependent portfolios)
- VaR scaling (higher multipliers in crisis regimes)
- Trading strategy selection (different strategies for different regimes)

Mathematical Framework:
- States: S = {Bull, Bear, Neutral, Crisis}
- Transition matrix: P(S_t | S_(t-1))
- Emission distributions: P(observations | S_t)
- Goal: Infer P(S_t | observations)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from ..config import config
from ..utils.data_structures import MarketRegime


class HMMRegimeDetector:
    """
    Detect market regimes using Hidden Markov Models.

    Features used for regime detection:
    1. Returns (mean regime behavior)
    2. Volatility (risk regime)
    3. Volume/liquidity (market participation)
    4. Correlations (contagion/diversification)
    """

    def __init__(
        self,
        n_regimes: int = None,
        covariance_type: str = 'full',
        n_iter: int = 100
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of market regimes to detect
            covariance_type: Covariance structure ('full', 'diag', 'tied')
            n_iter: Maximum iterations for EM algorithm
        """
        self.n_regimes = n_regimes or config.risk_params.n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter

        self.model: Optional[hmm.GaussianHMM] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

        # Regime interpretation (filled after fitting)
        self.regime_mapping: Dict[int, MarketRegime] = {}

    def _create_features(
        self,
        returns: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Create features for regime detection.

        Features:
        1. Returns (raw)
        2. Rolling volatility
        3. Rolling Sharpe ratio (return/vol)
        4. Volume ratio (if available)
        5. Drawdown

        Args:
            returns: Array of returns
            window: Rolling window for feature calculation

        Returns:
            Feature matrix (T x n_features)
        """
        n = len(returns)
        features = []

        # Feature 1: Returns (normalized)
        features.append(returns.reshape(-1, 1))

        # Feature 2: Rolling volatility
        returns_series = pd.Series(returns)
        rolling_vol = returns_series.rolling(window=window, min_periods=1).std().values
        features.append(rolling_vol.reshape(-1, 1))

        # Feature 3: Rolling Sharpe ratio
        rolling_mean = returns_series.rolling(window=window, min_periods=1).mean().values
        sharpe = np.where(rolling_vol > 0, rolling_mean / rolling_vol, 0)
        features.append(sharpe.reshape(-1, 1))

        # Feature 4: Volatility change (volatility clustering indicator)
        vol_change = np.diff(rolling_vol, prepend=rolling_vol[0])
        features.append(vol_change.reshape(-1, 1))

        # Feature 5: Cumulative returns (trend)
        cumulative_returns = (1 + returns_series).cumprod().values
        features.append(cumulative_returns.reshape(-1, 1))

        # Feature 6: Drawdown (distance from peak)
        running_max = pd.Series(cumulative_returns).expanding().max().values
        drawdown = (cumulative_returns - running_max) / running_max
        features.append(drawdown.reshape(-1, 1))

        # Concatenate all features
        feature_matrix = np.hstack(features)

        # Store feature names
        self.feature_names = [
            'returns',
            'volatility',
            'sharpe',
            'vol_change',
            'cum_returns',
            'drawdown'
        ]

        return feature_matrix

    def fit(
        self,
        returns: np.ndarray,
        feature_window: int = 20
    ) -> 'HMMRegimeDetector':
        """
        Fit HMM to historical data.

        Args:
            returns: Historical returns
            feature_window: Window for rolling features

        Returns:
            Self (for method chaining)
        """
        # Create features
        features = self._create_features(returns, window=feature_window)

        # Normalize features
        features_scaled = self.scaler.fit_transform(features)

        # Create and fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42
        )

        # Fit model
        self.model.fit(features_scaled)

        # Interpret regimes based on characteristics
        self._interpret_regimes(features)

        return self

    def _interpret_regimes(self, features: np.ndarray):
        """
        Interpret the discovered regimes based on their characteristics.

        Maps HMM states (0, 1, 2, ...) to meaningful regime names
        (Bull, Bear, Neutral, Crisis)

        Args:
            features: Feature matrix used for training
        """
        if self.model is None:
            return

        # Predict regimes for historical data
        features_scaled = self.scaler.transform(features)
        states = self.model.predict(features_scaled)

        # Calculate average characteristics for each state
        regime_chars = []

        for state in range(self.n_regimes):
            mask = states == state
            state_features = features[mask]

            if len(state_features) > 0:
                avg_return = np.mean(state_features[:, 0])
                avg_vol = np.mean(state_features[:, 1])
                avg_sharpe = np.mean(state_features[:, 2])
                avg_drawdown = np.mean(state_features[:, 5])

                regime_chars.append({
                    'state': state,
                    'return': avg_return,
                    'volatility': avg_vol,
                    'sharpe': avg_sharpe,
                    'drawdown': avg_drawdown,
                    'frequency': mask.sum() / len(states)
                })

        # Sort by volatility (simplest approach)
        regime_chars.sort(key=lambda x: x['volatility'])

        # Map states to regime types
        # Lowest vol → Bull, Highest vol → Crisis, etc.
        if self.n_regimes == 2:
            self.regime_mapping = {
                regime_chars[0]['state']: MarketRegime.BULL,
                regime_chars[1]['state']: MarketRegime.BEAR,
            }
        elif self.n_regimes == 3:
            self.regime_mapping = {
                regime_chars[0]['state']: MarketRegime.BULL,
                regime_chars[1]['state']: MarketRegime.NEUTRAL,
                regime_chars[2]['state']: MarketRegime.BEAR,
            }
        elif self.n_regimes >= 4:
            self.regime_mapping = {
                regime_chars[0]['state']: MarketRegime.BULL,
                regime_chars[1]['state']: MarketRegime.NEUTRAL,
                regime_chars[2]['state']: MarketRegime.BEAR,
                regime_chars[3]['state']: MarketRegime.CRISIS,
            }

            # Additional states → assign based on characteristics
            for i in range(4, self.n_regimes):
                char = regime_chars[i]
                if char['sharpe'] > 0:
                    self.regime_mapping[char['state']] = MarketRegime.BULL
                else:
                    self.regime_mapping[char['state']] = MarketRegime.BEAR

    def predict_regime(
        self,
        returns: np.ndarray,
        return_probabilities: bool = False
    ) -> Tuple[MarketRegime, Dict[MarketRegime, float]]:
        """
        Predict current market regime.

        Args:
            returns: Recent returns for feature calculation
            return_probabilities: If True, return regime probabilities

        Returns:
            Tuple of (most_likely_regime, regime_probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Create features for recent data
        features = self._create_features(returns)

        # Use only the most recent observation
        recent_features = features[-1:, :]
        recent_scaled = self.scaler.transform(recent_features)

        # Predict state probabilities
        state_probs = self.model.predict_proba(recent_scaled)[0]

        # Most likely state
        most_likely_state = np.argmax(state_probs)
        most_likely_regime = self.regime_mapping.get(
            most_likely_state,
            MarketRegime.NEUTRAL
        )

        # Calculate regime probabilities (aggregate by regime type)
        regime_probs = {}
        for state, prob in enumerate(state_probs):
            regime = self.regime_mapping.get(state, MarketRegime.NEUTRAL)
            regime_probs[regime] = regime_probs.get(regime, 0) + prob

        return most_likely_regime, regime_probs

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition probability matrix.

        Returns:
            Matrix where entry (i,j) is P(regime_j | regime_i)
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        return self.model.transmat_

    def get_regime_statistics(
        self,
        returns: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.

        Args:
            returns: Historical returns

        Returns:
            DataFrame with regime statistics
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        # Create features and predict regimes
        features = self._create_features(returns)
        features_scaled = self.scaler.transform(features)
        states = self.model.predict(features_scaled)

        # Calculate statistics for each regime
        stats = []

        for state in range(self.n_regimes):
            mask = states == state
            regime_returns = returns[mask]

            if len(regime_returns) > 0:
                regime = self.regime_mapping.get(state, MarketRegime.NEUTRAL)

                stats.append({
                    'regime': regime.value,
                    'state': state,
                    'frequency': mask.sum() / len(returns),
                    'avg_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'sharpe': (np.mean(regime_returns) / np.std(regime_returns)
                              if np.std(regime_returns) > 0 else 0),
                    'min_return': np.min(regime_returns),
                    'max_return': np.max(regime_returns),
                    'skewness': pd.Series(regime_returns).skew(),
                    'kurtosis': pd.Series(regime_returns).kurtosis(),
                })

        return pd.DataFrame(stats)

    def get_expected_duration(self) -> Dict[MarketRegime, float]:
        """
        Calculate expected duration (in periods) for each regime.

        Expected duration = 1 / (1 - P(stay in same state))

        Returns:
            Dict mapping regime to expected duration
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        trans_matrix = self.model.transmat_
        durations = {}

        for state in range(self.n_regimes):
            p_stay = trans_matrix[state, state]

            if p_stay < 1:
                expected_duration = 1 / (1 - p_stay)
            else:
                expected_duration = np.inf

            regime = self.regime_mapping.get(state, MarketRegime.NEUTRAL)
            durations[regime] = expected_duration

        return durations

    def simulate_regimes(
        self,
        n_periods: int,
        current_regime: Optional[MarketRegime] = None
    ) -> np.ndarray:
        """
        Simulate future regime path.

        Useful for:
        - Stress testing (simulate regime transitions)
        - Strategy backtesting (regime-dependent returns)
        - Risk scenario generation

        Args:
            n_periods: Number of periods to simulate
            current_regime: Starting regime (if None, sample from stationary dist)

        Returns:
            Array of regime states
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        # Determine starting state
        if current_regime is not None:
            # Find state corresponding to regime
            current_state = None
            for state, regime in self.regime_mapping.items():
                if regime == current_regime:
                    current_state = state
                    break

            if current_state is None:
                current_state = 0
        else:
            # Sample from stationary distribution
            current_state = np.random.choice(self.n_regimes, p=self.model.startprob_)

        # Simulate regime transitions
        states = [current_state]
        trans_matrix = self.model.transmat_

        for _ in range(n_periods - 1):
            next_state = np.random.choice(
                self.n_regimes,
                p=trans_matrix[current_state]
            )
            states.append(next_state)
            current_state = next_state

        # Convert states to regimes
        regimes = np.array([
            self.regime_mapping.get(state, MarketRegime.NEUTRAL).value
            for state in states
        ])

        return regimes


# Example usage
if __name__ == "__main__":
    # Generate synthetic market data with regime switches
    np.random.seed(42)

    n = 1000

    # Regime 1: Bull (200 days, low vol, positive drift)
    bull_returns = np.random.normal(0.001, 0.01, 300)

    # Regime 2: Neutral (400 days, medium vol, zero drift)
    neutral_returns = np.random.normal(0.0, 0.015, 400)

    # Regime 3: Bear (200 days, high vol, negative drift)
    bear_returns = np.random.normal(-0.002, 0.03, 200)

    # Crisis (100 days, extreme vol, extreme negative drift)
    crisis_returns = np.random.normal(-0.005, 0.05, 100)

    # Concatenate
    returns = np.concatenate([
        bull_returns, neutral_returns, bear_returns, crisis_returns
    ])

    print("=== HMM Regime Detection ===\n")

    # Fit HMM
    detector = HMMRegimeDetector(n_regimes=4)
    detector.fit(returns)

    # Get regime statistics
    print("Regime Statistics:")
    stats = detector.get_regime_statistics(returns)
    print(stats.to_string(index=False))

    # Predict current regime
    current_regime, probs = detector.predict_regime(returns[-60:])
    print(f"\nCurrent Regime: {current_regime.value}")
    print("Regime Probabilities:")
    for regime, prob in probs.items():
        print(f"  {regime.value:10s}: {prob*100:5.1f}%")

    # Expected durations
    print("\nExpected Regime Durations (days):")
    durations = detector.get_expected_duration()
    for regime, duration in durations.items():
        print(f"  {regime.value:10s}: {duration:5.1f}")

    # Transition matrix
    print("\nTransition Matrix:")
    trans_matrix = detector.get_transition_matrix()
    print(trans_matrix)
