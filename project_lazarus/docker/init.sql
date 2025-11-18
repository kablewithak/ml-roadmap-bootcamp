-- Initialize Project Lazarus database

-- Create tables
CREATE TABLE IF NOT EXISTS loan_applications (
    application_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    requested_amount FLOAT NOT NULL,
    loan_purpose VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- User features snapshot
    age INTEGER NOT NULL,
    income FLOAT NOT NULL,
    debt FLOAT NOT NULL,
    credit_score INTEGER NOT NULL,
    employment_years FLOAT,
    recent_bankruptcy BOOLEAN DEFAULT FALSE,
    num_credit_lines INTEGER,
    avg_txn_amt_30d FLOAT,
    credit_history_months INTEGER
);

CREATE TABLE IF NOT EXISTS loan_decisions (
    id SERIAL PRIMARY KEY,
    application_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,

    -- Decision details
    decision VARCHAR(20) NOT NULL,
    reason VARCHAR(50) NOT NULL,
    risk_score FLOAT NOT NULL,
    treatment_group VARCHAR(20) NOT NULL,

    -- Metadata
    model_version VARCHAR(20) DEFAULT 'v1',
    exploration_budget_remaining FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS loan_outcomes (
    id SERIAL PRIMARY KEY,
    application_id VARCHAR(50) NOT NULL UNIQUE,
    user_id VARCHAR(50) NOT NULL,

    -- Outcome details
    defaulted BOOLEAN NOT NULL,
    days_to_default INTEGER,
    amount_recovered FLOAT DEFAULT 0.0,

    -- Treatment group for causal analysis
    treatment_group VARCHAR(20) NOT NULL,

    -- Timestamps
    loan_disbursed_at TIMESTAMP,
    outcome_recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment_exposures (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    application_id VARCHAR(50) NOT NULL,

    -- Experiment details
    variant VARCHAR(20) NOT NULL,
    risk_score FLOAT NOT NULL,
    decision VARCHAR(20) NOT NULL,

    -- Features snapshot
    features_snapshot JSONB,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,

    -- Performance metrics
    auc_roc FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1 FLOAT,

    -- Causal metrics
    blindness_score FLOAT,
    ate_estimate FLOAT,
    exploration_cost FLOAT,
    projected_revenue_lift FLOAT,

    -- Training details
    training_samples INTEGER,
    explore_samples INTEGER,
    exploit_samples INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_applications_user ON loan_applications(user_id);
CREATE INDEX IF NOT EXISTS idx_applications_created ON loan_applications(created_at);

CREATE INDEX IF NOT EXISTS idx_decisions_application ON loan_decisions(application_id);
CREATE INDEX IF NOT EXISTS idx_decisions_user ON loan_decisions(user_id);
CREATE INDEX IF NOT EXISTS idx_decisions_treatment ON loan_decisions(treatment_group);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON loan_decisions(created_at);

CREATE INDEX IF NOT EXISTS idx_outcomes_application ON loan_outcomes(application_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_user ON loan_outcomes(user_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_defaulted ON loan_outcomes(defaulted);
CREATE INDEX IF NOT EXISTS idx_outcomes_treatment ON loan_outcomes(treatment_group);

CREATE INDEX IF NOT EXISTS idx_exposures_user ON experiment_exposures(user_id);
CREATE INDEX IF NOT EXISTS idx_exposures_application ON experiment_exposures(application_id);
CREATE INDEX IF NOT EXISTS idx_exposures_variant ON experiment_exposures(variant);
CREATE INDEX IF NOT EXISTS idx_exposures_created ON experiment_exposures(created_at);

CREATE INDEX IF NOT EXISTS idx_metrics_version ON model_metrics(model_version);
CREATE INDEX IF NOT EXISTS idx_metrics_created ON model_metrics(created_at);
