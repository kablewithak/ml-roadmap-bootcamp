-- Initialize database for streaming state management
-- Supports exactly-once semantics with transactional state

-- Transaction state tracking table
CREATE TABLE IF NOT EXISTS transaction_state (
    transaction_id UUID PRIMARY KEY,
    status VARCHAR(20) NOT NULL CHECK (status IN ('PREPARED', 'COMMITTED', 'ROLLED_BACK')),
    kafka_topic VARCHAR(255) NOT NULL,
    kafka_partition INTEGER NOT NULL,
    kafka_offset BIGINT NOT NULL,
    message_key VARCHAR(255),
    processing_started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    processing_completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index for efficient lookup
CREATE INDEX IF NOT EXISTS idx_transaction_status ON transaction_state(status, created_at);
CREATE INDEX IF NOT EXISTS idx_kafka_offset ON transaction_state(kafka_topic, kafka_partition, kafka_offset);

-- Processed messages deduplication table
CREATE TABLE IF NOT EXISTS processed_messages (
    message_id VARCHAR(255) PRIMARY KEY,
    kafka_topic VARCHAR(255) NOT NULL,
    kafka_partition INTEGER NOT NULL,
    kafka_offset BIGINT NOT NULL,
    kafka_timestamp TIMESTAMP NOT NULL,
    idempotency_key VARCHAR(255) UNIQUE,
    processed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    processing_duration_ms INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index for idempotency checks
CREATE INDEX IF NOT EXISTS idx_idempotency_key ON processed_messages(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_processed_at ON processed_messages(processed_at);

-- Offset tracking table (for manual offset management)
CREATE TABLE IF NOT EXISTS consumer_offsets (
    consumer_group VARCHAR(255) NOT NULL,
    topic VARCHAR(255) NOT NULL,
    partition INTEGER NOT NULL,
    offset_value BIGINT NOT NULL,
    metadata TEXT,
    committed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (consumer_group, topic, partition)
);

-- Dead letter queue table
CREATE TABLE IF NOT EXISTS dead_letter_queue (
    id BIGSERIAL PRIMARY KEY,
    original_topic VARCHAR(255) NOT NULL,
    original_partition INTEGER NOT NULL,
    original_offset BIGINT NOT NULL,
    message_key VARCHAR(255),
    message_value BYTEA,
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT,
    stack_trace TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    next_retry_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dlq_retry ON dead_letter_queue(next_retry_at, resolved_at) WHERE resolved_at IS NULL;

-- Business domain tables (example: payment processing)
CREATE TABLE IF NOT EXISTS payment_events (
    payment_id UUID PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    merchant_id VARCHAR(255) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    status VARCHAR(50) NOT NULL,
    fraud_score DECIMAL(3, 2),
    transaction_id UUID REFERENCES transaction_state(transaction_id),
    kafka_offset BIGINT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_payment_user ON payment_events(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_payment_merchant ON payment_events(merchant_id, created_at);

-- Fraud decisions table
CREATE TABLE IF NOT EXISTS fraud_decisions (
    decision_id UUID PRIMARY KEY,
    payment_id UUID REFERENCES payment_events(payment_id),
    decision VARCHAR(50) NOT NULL CHECK (decision IN ('APPROVE', 'DECLINE', 'REVIEW')),
    fraud_score DECIMAL(3, 2) NOT NULL,
    fraud_reasons JSONB,
    model_version VARCHAR(50),
    transaction_id UUID REFERENCES transaction_state(transaction_id),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fraud_payment ON fraud_decisions(payment_id);

-- Metrics aggregation table
CREATE TABLE IF NOT EXISTS processing_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 4) NOT NULL,
    metric_type VARCHAR(50) NOT NULL CHECK (metric_type IN ('COUNTER', 'GAUGE', 'HISTOGRAM', 'SUMMARY')),
    tags JSONB,
    recorded_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON processing_metrics(metric_name, recorded_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_transaction_state_updated_at BEFORE UPDATE ON transaction_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_payment_events_updated_at BEFORE UPDATE ON payment_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for cleaning old processed messages (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_processed_messages(retention_days INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM processed_messages
    WHERE processed_at < NOW() - (retention_days || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function for cleaning old transaction state
CREATE OR REPLACE FUNCTION cleanup_old_transactions(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM transaction_state
    WHERE status IN ('COMMITTED', 'ROLLED_BACK')
    AND created_at < NOW() - (retention_days || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO streaming;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO streaming;
