"""
Configuration management for the risk management system.
Handles environment variables, risk limits, and system parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """TimescaleDB configuration"""
    host: str = os.getenv("TIMESCALE_HOST", "localhost")
    port: int = int(os.getenv("TIMESCALE_PORT", "5432"))
    database: str = os.getenv("TIMESCALE_DB", "market_data")
    user: str = os.getenv("TIMESCALE_USER", "risk_admin")
    password: str = os.getenv("TIMESCALE_PASSWORD", "")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    db: int = 0
    max_connections: int = 50


@dataclass
class PulsarConfig:
    """Apache Pulsar streaming configuration"""
    service_url: str = os.getenv("PULSAR_URL", "pulsar://localhost:6650")
    admin_url: str = os.getenv("PULSAR_HTTP_URL", "http://localhost:8080")
    topic_prefix: str = "persistent://public/default/"

    # Topics
    market_data_topic: str = "market-data"
    risk_metrics_topic: str = "risk-metrics"
    alerts_topic: str = "alerts"
    positions_topic: str = "positions"


@dataclass
class RiskLimits:
    """Portfolio risk limits and thresholds"""
    # VaR limits
    max_portfolio_var_usd: float = float(os.getenv("MAX_PORTFOLIO_VAR_USD", "5000000"))
    max_strategy_var_pct: float = 0.30  # 30% of portfolio VaR
    max_position_var_pct: float = 0.10  # 10% of portfolio VaR

    # Position limits
    max_position_size_pct: float = float(os.getenv("MAX_POSITION_SIZE_PCT", "0.05"))
    max_sector_exposure_pct: float = 0.25
    max_single_name_exposure_usd: float = 10_000_000

    # Liquidity limits
    max_adv_pct: float = 0.10  # Max 10% of average daily volume
    min_holding_period_days: int = 1

    # Greeks limits (for options)
    max_delta: float = 1_000_000
    max_gamma: float = 100_000
    max_vega: float = 500_000

    # Concentration limits
    max_correlation_exposure: float = 0.80
    max_leverage: float = 3.0


@dataclass
class RiskParameters:
    """Risk calculation parameters"""
    # VaR parameters
    var_confidence_level: float = float(os.getenv("VAR_CONFIDENCE_LEVEL", "0.99"))
    var_horizon_days: int = 1
    expected_shortfall_confidence: float = float(os.getenv("EXPECTED_SHORTFALL_CONFIDENCE", "0.975"))

    # Historical simulation
    historical_window_days: int = 252  # 1 year

    # Monte Carlo simulation
    mc_simulations: int = 100_000
    mc_random_seed: int = 42

    # GARCH parameters
    garch_p: int = 1  # GARCH lag order
    garch_q: int = 1  # ARCH lag order

    # Regime detection
    n_regimes: int = 3  # Bull, Bear, Neutral
    regime_lookback_days: int = 60

    # Correlation parameters
    correlation_window_days: int = 60
    correlation_ewma_lambda: float = 0.94  # RiskMetrics standard


@dataclass
class PerformanceConfig:
    """System performance requirements"""
    max_events_per_second: int = int(os.getenv("MAX_EVENTS_PER_SECOND", "500000"))
    target_latency_ms: int = int(os.getenv("TARGET_LATENCY_MS", "50"))
    regime_update_interval_sec: int = int(os.getenv("REGIME_UPDATE_INTERVAL_SEC", "5"))

    # Computation
    n_workers: int = os.cpu_count() or 4
    use_gpu: bool = False
    batch_size: int = 10000

    # Caching
    cache_ttl_seconds: int = 60
    cache_refresh_interval: int = 30


@dataclass
class RegulatoryConfig:
    """Regulatory compliance settings"""
    jurisdiction: str = os.getenv("REGULATORY_JURISDICTION", "US")
    frtb_enabled: bool = os.getenv("FRTB_ENABLED", "true").lower() == "true"
    reporting_currency: str = os.getenv("REPORTING_CURRENCY", "USD")

    # FRTB parameters (Fundamental Review of Trading Book)
    frtb_liquidity_horizon_days: int = 10
    frtb_period_of_stress: str = "2008"  # Global Financial Crisis

    # Reporting
    daily_report_time: str = "17:00:00"  # UTC
    regulatory_reporting_enabled: bool = True


@dataclass
class SystemConfig:
    """Master system configuration"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    pulsar: PulsarConfig = field(default_factory=PulsarConfig)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    risk_params: RiskParameters = field(default_factory=RiskParameters)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    regulatory: RegulatoryConfig = field(default_factory=RegulatoryConfig)

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")


# Global config instance
config = SystemConfig()


def validate_config() -> list[str]:
    """Validate configuration and return list of warnings/errors"""
    issues = []

    # Check database password
    if not config.database.password:
        issues.append("WARNING: Database password not set")

    # Validate risk limits
    if config.risk_limits.max_portfolio_var_usd <= 0:
        issues.append("ERROR: max_portfolio_var_usd must be positive")

    # Validate confidence levels
    if not 0 < config.risk_params.var_confidence_level < 1:
        issues.append("ERROR: var_confidence_level must be between 0 and 1")

    # Check performance settings
    if config.performance.target_latency_ms > 1000:
        issues.append("WARNING: Target latency >1s may not meet real-time requirements")

    return issues


if __name__ == "__main__":
    # Validate configuration
    issues = validate_config()
    for issue in issues:
        print(issue)

    if not any("ERROR" in issue for issue in issues):
        print("✓ Configuration validated successfully")
    else:
        print("✗ Configuration has errors")
