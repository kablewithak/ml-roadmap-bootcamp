"""Feature definitions for Project Lazarus using Feast."""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, Field
from feast.types import Float32, Int32, Bool, String


# Define entities
user = Entity(
    name="user",
    join_keys=["user_id"],
    description="A loan applicant",
)

# Define data sources
user_features_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
)

transaction_features_source = FileSource(
    path="data/transaction_features.parquet",
    timestamp_field="event_timestamp",
)

# User demographic features
user_demographics_fv = FeatureView(
    name="user_demographics",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age", dtype=Int32),
        Field(name="income", dtype=Float32),
        Field(name="employment_years", dtype=Float32),
        Field(name="credit_history_months", dtype=Int32),
    ],
    source=user_features_source,
    online=True,
    tags={"team": "risk"},
)

# User credit features
user_credit_fv = FeatureView(
    name="user_credit",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="credit_score", dtype=Int32),
        Field(name="debt", dtype=Float32),
        Field(name="num_credit_lines", dtype=Int32),
        Field(name="recent_bankruptcy", dtype=Bool),
    ],
    source=user_features_source,
    online=True,
    tags={"team": "risk"},
)

# Transaction-based features
transaction_features_fv = FeatureView(
    name="transaction_features",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="avg_txn_amt_30d", dtype=Float32),
        Field(name="num_txns_30d", dtype=Int32),
        Field(name="max_txn_amt_30d", dtype=Float32),
        Field(name="min_txn_amt_30d", dtype=Float32),
    ],
    source=transaction_features_source,
    online=True,
    tags={"team": "risk"},
)
