"""Data processing utilities."""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def generate_cache_key(*args: Any) -> str:
    """Generate a cache key from arguments."""
    key_str = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_trading_days(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    exchange: str = "NYSE",
) -> pd.DatetimeIndex:
    """
    Get trading days between two dates.

    Args:
        start_date: Start date
        end_date: End date
        exchange: Exchange calendar to use

    Returns:
        DatetimeIndex of trading days
    """
    import pandas_market_calendars as mcal

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    calendar = mcal.get_calendar(exchange)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    return schedule.index


def align_timestamps(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    time_col: str = "timestamp",
    method: str = "asof",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align timestamps between two dataframes.

    Args:
        df1: First dataframe
        df2: Second dataframe
        time_col: Name of timestamp column
        method: Alignment method ('asof', 'ffill', 'bfill')

    Returns:
        Tuple of aligned dataframes
    """
    df1 = df1.sort_values(time_col)
    df2 = df2.sort_values(time_col)

    if method == "asof":
        df2_aligned = pd.merge_asof(
            df1[[time_col]], df2, on=time_col, direction="backward"
        )
        return df1, df2_aligned
    else:
        merged = pd.merge(df1, df2, on=time_col, how="outer", suffixes=("_1", "_2"))
        merged = merged.sort_values(time_col)
        if method == "ffill":
            merged = merged.fillna(method="ffill")
        elif method == "bfill":
            merged = merged.fillna(method="bfill")
        return merged.filter(like="_1"), merged.filter(like="_2")


def remove_outliers(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Remove outliers from dataframe.

    Args:
        data: Input dataframe
        columns: Columns to check for outliers (None = all numeric)
        method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection

    Returns:
        Dataframe with outliers removed
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    mask = pd.Series([True] * len(data), index=data.index)

    if method == "iqr":
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask &= (data[col] >= lower_bound) & (data[col] <= upper_bound)

    elif method == "zscore":
        for col in columns:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            mask &= z_scores < threshold

    elif method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(contamination=0.1, random_state=42)
        predictions = clf.fit_predict(data[columns])
        mask &= predictions == 1

    return data[mask]


def winsorize(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    limits: Tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Winsorize dataframe columns.

    Args:
        data: Input dataframe
        columns: Columns to winsorize (None = all numeric)
        limits: Lower and upper percentile limits

    Returns:
        Winsorized dataframe
    """
    from scipy.stats import mstats

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    result = data.copy()
    for col in columns:
        result[col] = mstats.winsorize(
            data[col], limits=(limits[0], 1 - limits[1])
        )

    return result


def create_time_features(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create time-based features from timestamp column.

    Args:
        df: Input dataframe
        time_col: Name of timestamp column
        features: List of features to create (None = all)

    Returns:
        Dataframe with time features added
    """
    result = df.copy()
    ts = pd.to_datetime(result[time_col])

    all_features = {
        "year": ts.dt.year,
        "month": ts.dt.month,
        "day": ts.dt.day,
        "dayofweek": ts.dt.dayofweek,
        "dayofyear": ts.dt.dayofyear,
        "hour": ts.dt.hour,
        "minute": ts.dt.minute,
        "quarter": ts.dt.quarter,
        "is_month_start": ts.dt.is_month_start.astype(int),
        "is_month_end": ts.dt.is_month_end.astype(int),
        "is_quarter_start": ts.dt.is_quarter_start.astype(int),
        "is_quarter_end": ts.dt.is_quarter_end.astype(int),
        "is_year_start": ts.dt.is_year_start.astype(int),
        "is_year_end": ts.dt.is_year_end.astype(int),
    }

    if features is None:
        features = list(all_features.keys())

    for feature in features:
        if feature in all_features:
            result[feature] = all_features[feature]

    return result


def normalize_features(
    train_data: pd.DataFrame,
    test_data: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
    method: str = "robust",
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Normalize features using various methods.

    Args:
        train_data: Training dataframe
        test_data: Optional test dataframe
        columns: Columns to normalize (None = all numeric)
        method: Normalization method ('robust', 'standard', 'minmax')

    Returns:
        Normalized dataframe(s)
    """
    if columns is None:
        columns = train_data.select_dtypes(include=[np.number]).columns.tolist()

    if method == "robust":
        scaler = RobustScaler()
    elif method == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    train_result = train_data.copy()
    train_result[columns] = scaler.fit_transform(train_data[columns])

    if test_data is not None:
        test_result = test_data.copy()
        test_result[columns] = scaler.transform(test_data[columns])
        return train_result, test_result
    else:
        return train_result


def check_data_leakage(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    time_col: str = "timestamp",
) -> Dict[str, Any]:
    """
    Check for data leakage between train and test sets.

    Args:
        train_data: Training dataframe
        test_data: Test dataframe
        time_col: Name of timestamp column

    Returns:
        Dictionary with leakage check results
    """
    results = {}

    # Check time overlap
    if time_col in train_data.columns and time_col in test_data.columns:
        train_max_time = train_data[time_col].max()
        test_min_time = test_data[time_col].min()
        results["time_overlap"] = train_max_time >= test_min_time
        results["train_max_time"] = train_max_time
        results["test_min_time"] = test_min_time

    # Check for duplicate indices
    common_indices = set(train_data.index).intersection(set(test_data.index))
    results["duplicate_indices"] = len(common_indices)
    results["has_duplicate_indices"] = len(common_indices) > 0

    # Check for identical rows
    if len(train_data.columns) == len(test_data.columns):
        train_hashes = train_data.apply(
            lambda row: hashlib.md5(str(row.values).encode()).hexdigest(), axis=1
        )
        test_hashes = test_data.apply(
            lambda row: hashlib.md5(str(row.values).encode()).hexdigest(), axis=1
        )
        common_hashes = set(train_hashes).intersection(set(test_hashes))
        results["duplicate_rows"] = len(common_hashes)
        results["has_duplicate_rows"] = len(common_hashes) > 0

    return results
