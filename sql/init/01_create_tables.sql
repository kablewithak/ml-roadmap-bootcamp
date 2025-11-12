-- TimescaleDB initialization script for market risk management system
-- Creates hypertables optimized for tick data and risk metrics

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- MARKET DATA TABLES
-- ============================================================================

-- Tick data table (trades and quotes)
CREATE TABLE IF NOT EXISTS tick_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    side TEXT,  -- 'bid' or 'ask'
    exchange TEXT,
    sequence_number BIGINT,
    CONSTRAINT tick_data_price_positive CHECK (price > 0),
    CONSTRAINT tick_data_volume_nonnegative CHECK (volume >= 0)
);

-- Convert to hypertable (partitioned by time)
SELECT create_hypertable('tick_data', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_tick_data_symbol_time
    ON tick_data (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_tick_data_exchange
    ON tick_data (exchange, time DESC);


-- Order book snapshots
CREATE TABLE IF NOT EXISTS order_book (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'bid' or 'ask'
    level INTEGER NOT NULL,  -- 0 = best, 1 = second best, etc.
    price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    num_orders INTEGER DEFAULT 1,
    CONSTRAINT order_book_level_nonnegative CHECK (level >= 0),
    CONSTRAINT order_book_price_positive CHECK (price > 0)
);

SELECT create_hypertable('order_book', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_order_book_symbol_time
    ON order_book (symbol, time DESC, side, level);


-- Market microstructure features (aggregated metrics)
CREATE TABLE IF NOT EXISTS market_microstructure (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bid_ask_spread DOUBLE PRECISION,
    bid_ask_spread_bps DOUBLE PRECISION,
    trade_intensity DOUBLE PRECISION,  -- trades per minute
    volume_imbalance DOUBLE PRECISION,
    order_flow_imbalance DOUBLE PRECISION,
    realized_volatility DOUBLE PRECISION,
    effective_spread DOUBLE PRECISION,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('market_microstructure', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_microstructure_symbol
    ON market_microstructure (symbol, time DESC);


-- ============================================================================
-- POSITION AND PORTFOLIO TABLES
-- ============================================================================

-- Positions (current and historical)
CREATE TABLE IF NOT EXISTS positions (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION NOT NULL,
    market_value DOUBLE PRECISION NOT NULL,
    unrealized_pnl DOUBLE PRECISION,
    strategy_id TEXT,
    trader_id TEXT,
    PRIMARY KEY (time, portfolio_id, symbol)
);

SELECT create_hypertable('positions', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_positions_portfolio
    ON positions (portfolio_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_positions_strategy
    ON positions (strategy_id, time DESC);


-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id TEXT NOT NULL,
    total_value DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    unrealized_pnl DOUBLE PRECISION,
    gross_exposure DOUBLE PRECISION,
    net_exposure DOUBLE PRECISION,
    num_positions INTEGER,
    PRIMARY KEY (time, portfolio_id)
);

SELECT create_hypertable('portfolio_snapshots', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);


-- ============================================================================
-- RISK METRICS TABLES
-- ============================================================================

-- VaR calculations
CREATE TABLE IF NOT EXISTS var_metrics (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id TEXT NOT NULL,
    confidence_level DOUBLE PRECISION NOT NULL,
    horizon_days INTEGER NOT NULL,

    -- VaR values
    var_historical DOUBLE PRECISION,
    var_parametric DOUBLE PRECISION,
    var_monte_carlo DOUBLE PRECISION,
    var_garch DOUBLE PRECISION,
    var_liquidity_adjusted DOUBLE PRECISION,

    -- Expected Shortfall
    expected_shortfall DOUBLE PRECISION,

    -- Volatility
    volatility_daily DOUBLE PRECISION,
    volatility_annualized DOUBLE PRECISION,

    -- Regime
    current_regime TEXT,
    regime_probability DOUBLE PRECISION,

    PRIMARY KEY (time, portfolio_id, confidence_level, horizon_days)
);

SELECT create_hypertable('var_metrics', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_var_metrics_portfolio
    ON var_metrics (portfolio_id, time DESC);


-- Component VaR (position-level risk contribution)
CREATE TABLE IF NOT EXISTS component_var (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    component_var DOUBLE PRECISION NOT NULL,
    marginal_var DOUBLE PRECISION,
    contribution_pct DOUBLE PRECISION,
    PRIMARY KEY (time, portfolio_id, symbol)
);

SELECT create_hypertable('component_var', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);


-- Greeks for options portfolios
CREATE TABLE IF NOT EXISTS portfolio_greeks (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id TEXT NOT NULL,
    delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION,
    vega DOUBLE PRECISION,
    theta DOUBLE PRECISION,
    rho DOUBLE PRECISION,
    PRIMARY KEY (time, portfolio_id)
);

SELECT create_hypertable('portfolio_greeks', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);


-- ============================================================================
-- BACKTESTING AND VALIDATION TABLES
-- ============================================================================

-- VaR backtesting results
CREATE TABLE IF NOT EXISTS var_backtest (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id TEXT NOT NULL,
    predicted_var DOUBLE PRECISION NOT NULL,
    actual_loss DOUBLE PRECISION NOT NULL,
    is_violation BOOLEAN NOT NULL,
    violation_size DOUBLE PRECISION,
    confidence_level DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, portfolio_id)
);

SELECT create_hypertable('var_backtest', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);


-- Model validation statistics
CREATE TABLE IF NOT EXISTS model_validation (
    time TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION,
    passed BOOLEAN,
    notes TEXT,
    PRIMARY KEY (time, model_name, metric_name)
);

SELECT create_hypertable('model_validation', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);


-- ============================================================================
-- REGIME DETECTION TABLES
-- ============================================================================

-- Market regime classifications
CREATE TABLE IF NOT EXISTS market_regimes (
    time TIMESTAMPTZ NOT NULL,
    asset_class TEXT NOT NULL,
    regime TEXT NOT NULL,  -- 'bull', 'bear', 'neutral', 'crisis'
    probability DOUBLE PRECISION NOT NULL,
    volatility DOUBLE PRECISION,
    expected_duration_days DOUBLE PRECISION,
    PRIMARY KEY (time, asset_class)
);

SELECT create_hypertable('market_regimes', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);


-- Change points detected
CREATE TABLE IF NOT EXISTS change_points (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    change_type TEXT NOT NULL,  -- 'mean', 'variance', 'volatility_breakout'
    detection_method TEXT NOT NULL,  -- 'cusum', 'mosum', 'pelt'
    confidence DOUBLE PRECISION,
    notes TEXT,
    PRIMARY KEY (time, symbol, change_type)
);

SELECT create_hypertable('change_points', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);


-- ============================================================================
-- ALERT AND MONITORING TABLES
-- ============================================================================

-- Risk limit violations and alerts
CREATE TABLE IF NOT EXISTS risk_alerts (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,  -- 'var_breach', 'position_limit', 'greek_limit'
    severity TEXT NOT NULL,  -- 'info', 'warning', 'critical'
    metric_name TEXT NOT NULL,
    current_value DOUBLE PRECISION NOT NULL,
    limit_value DOUBLE PRECISION NOT NULL,
    message TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at TIMESTAMPTZ
);

SELECT create_hypertable('risk_alerts', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_alerts_portfolio_severity
    ON risk_alerts (portfolio_id, severity, time DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_unacknowledged
    ON risk_alerts (acknowledged, time DESC)
    WHERE acknowledged = FALSE;


-- ============================================================================
-- RETENTION POLICIES
-- ============================================================================

-- Tick data: Keep 90 days, then downsample to 1-minute bars
SELECT add_retention_policy('tick_data', INTERVAL '90 days', if_not_exists => TRUE);

-- Order book: Keep 30 days (very high volume)
SELECT add_retention_policy('order_book', INTERVAL '30 days', if_not_exists => TRUE);

-- Microstructure features: Keep 1 year
SELECT add_retention_policy('market_microstructure', INTERVAL '1 year', if_not_exists => TRUE);

-- Risk metrics: Keep 2 years
SELECT add_retention_policy('var_metrics', INTERVAL '2 years', if_not_exists => TRUE);

-- Alerts: Keep 1 year
SELECT add_retention_policy('risk_alerts', INTERVAL '1 year', if_not_exists => TRUE);


-- ============================================================================
-- CONTINUOUS AGGREGATES (for performance)
-- ============================================================================

-- 1-minute OHLCV bars from tick data
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS time,
    symbol,
    FIRST(price, time) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, time) AS close,
    SUM(volume) AS volume,
    COUNT(*) AS tick_count
FROM tick_data
GROUP BY time_bucket('1 minute', time), symbol;


-- Daily portfolio statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS portfolio_daily_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS time,
    portfolio_id,
    AVG(total_value) AS avg_value,
    MAX(total_value) AS max_value,
    MIN(total_value) AS min_value,
    LAST(total_value, time) AS close_value,
    AVG(gross_exposure) AS avg_gross_exposure
FROM portfolio_snapshots
GROUP BY time_bucket('1 day', time), portfolio_id;


-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO risk_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO risk_admin;

-- Create read-only user for reporting
-- CREATE USER risk_readonly WITH PASSWORD 'readonly_password';
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO risk_readonly;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB risk management database initialized successfully!';
    RAISE NOTICE 'Created hypertables for tick data, order books, positions, and risk metrics.';
    RAISE NOTICE 'Configured retention policies and continuous aggregates.';
END $$;
