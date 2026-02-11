"""Generate statistical profiles and anomaly reports for synthetic datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats


RANDOM_SEED = 42
DATE_COLUMNS = {
    "users": ["signup_date"],
    "transactions": ["transaction_date"],
}


def get_data_directory() -> Path:
    """Return the project data directory, creating it when needed."""

    data_directory = Path(__file__).resolve().parents[1] / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    return data_directory


def load_dataset(file_path: Path, date_columns: list[str]) -> pd.DataFrame:
    """Load a CSV dataset from disk and parse its date columns."""

    return pd.read_csv(file_path, parse_dates=date_columns)


def to_serializable(value: Any) -> Any:
    """Convert pandas and NumPy scalar values into JSON-serializable Python values."""

    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns excluding boolean fields."""

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if not pd.api.types.is_bool_dtype(df[column])]


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical columns, including booleans and excluding datetimes."""

    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def compute_column_completeness(df: pd.DataFrame) -> dict[str, Any]:
    """Compute null-count and null-percentage metrics for every column."""

    total_rows = len(df)
    completeness: dict[str, Any] = {}

    for column in df.columns:
        null_count = int(df[column].isna().sum())
        null_percentage = float(null_count / total_rows) if total_rows else 0.0
        completeness[column] = {
            "non_null_count": int(total_rows - null_count),
            "null_count": null_count,
            "null_percentage": round(null_percentage, 6),
        }

    return completeness


def compute_numeric_profile(series: pd.Series) -> dict[str, Any]:
    """Compute descriptive statistics, outlier counts, and quartiles for a numeric column."""

    clean_series = pd.to_numeric(series, errors="coerce").dropna()

    if clean_series.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "percentiles": {"25th": None, "50th": None, "75th": None},
            "skewness": None,
            "kurtosis": None,
            "iqr_outlier_count": 0,
        }

    q1 = clean_series.quantile(0.25)
    median = clean_series.quantile(0.50)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    iqr_outlier_count = int(((clean_series < lower_bound) | (clean_series > upper_bound)).sum())

    return {
        "count": int(clean_series.count()),
        "mean": to_serializable(clean_series.mean()),
        "median": to_serializable(clean_series.median()),
        "std": to_serializable(clean_series.std()),
        "min": to_serializable(clean_series.min()),
        "max": to_serializable(clean_series.max()),
        "percentiles": {
            "25th": to_serializable(q1),
            "50th": to_serializable(median),
            "75th": to_serializable(q3),
        },
        "skewness": to_serializable(clean_series.skew()),
        "kurtosis": to_serializable(clean_series.kurt()),
        "iqr_outlier_count": iqr_outlier_count,
    }


def compute_categorical_profile(series: pd.Series) -> dict[str, Any]:
    """Compute categorical distribution statistics and entropy for a column."""

    clean_series = series.dropna()

    if clean_series.empty:
        return {
            "value_counts": {},
            "percentages": {},
            "unique_values": 0,
            "most_frequent": None,
            "least_frequent": None,
            "entropy": None,
        }

    counts = clean_series.value_counts(dropna=False)
    percentages = (counts / counts.sum()) * 100
    probabilities = counts / counts.sum()
    most_frequent = counts.sort_values(ascending=False, kind="stable").index[0]
    least_frequent = counts.sort_values(ascending=True, kind="stable").index[0]

    return {
        "value_counts": {str(key): int(value) for key, value in counts.items()},
        "percentages": {
            str(key): round(float(value), 2) for key, value in percentages.items()
        },
        "unique_values": int(clean_series.nunique(dropna=True)),
        "most_frequent": str(most_frequent),
        "least_frequent": str(least_frequent),
        "entropy": to_serializable(stats.entropy(probabilities, base=2)),
    }


def compute_anomaly_details(
    dataset_name: str,
    column_name: str,
    series: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect anomalies using Z-score and IQR methods for a numeric column."""

    numeric_series = pd.to_numeric(series, errors="coerce")
    clean_series = numeric_series.dropna()

    if clean_series.empty:
        return [], {
            "z_score_anomalies": 0,
            "iqr_anomalies": 0,
            "agreement_rate": None,
        }

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    iqr_flags = (clean_series < lower_bound) | (clean_series > upper_bound)

    if clean_series.std(ddof=0) == 0:
        z_scores = pd.Series(0.0, index=clean_series.index)
    else:
        z_scores = pd.Series(stats.zscore(clean_series, nan_policy="omit"), index=clean_series.index)

    z_flags = z_scores.abs() > 3
    agreement_rate = float((iqr_flags == z_flags).mean() * 100)
    combined_flags = iqr_flags | z_flags
    anomalies: list[dict[str, Any]] = []

    for row_index in clean_series.index[combined_flags]:
        anomalies.append(
            {
                "dataset": dataset_name,
                "column": column_name,
                "row_index": int(row_index),
                "value": to_serializable(clean_series.loc[row_index]),
                "z_score": to_serializable(z_scores.loc[row_index]),
                "iqr_flag": bool(iqr_flags.loc[row_index]),
                "z_flag": bool(z_flags.loc[row_index]),
            }
        )

    return anomalies, {
        "z_score_anomalies": int(z_flags.sum()),
        "iqr_anomalies": int(iqr_flags.sum()),
        "agreement_rate": round(agreement_rate, 2),
    }


def test_normality(series: pd.Series) -> dict[str, Any]:
    """Run the Shapiro-Wilk normality test on up to 5000 non-null numeric values."""

    clean_series = pd.to_numeric(series, errors="coerce").dropna()

    if len(clean_series) < 3:
        return {
            "is_normal": None,
            "p_value": None,
            "test_statistic": None,
        }

    if len(clean_series) > 5000:
        clean_series = clean_series.sample(n=5000, random_state=RANDOM_SEED)

    test_statistic, p_value = stats.shapiro(clean_series)
    return {
        "is_normal": bool(p_value > 0.05),
        "p_value": to_serializable(p_value),
        "test_statistic": to_serializable(test_statistic),
    }


def profile_dataset(dataset_name: str, df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Generate a full statistical profile, anomaly list, and summary metrics."""

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)
    column_completeness = compute_column_completeness(df)

    numeric_profiles: dict[str, Any] = {}
    categorical_profiles: dict[str, Any] = {}
    distribution_analysis: dict[str, Any] = {}
    anomaly_detection: dict[str, Any] = {}
    anomalies: list[dict[str, Any]] = []
    columns_with_outliers: list[str] = []

    for column in numeric_columns:
        numeric_profiles[column] = compute_numeric_profile(df[column])
        column_anomalies, anomaly_summary = compute_anomaly_details(dataset_name, column, df[column])
        anomalies.extend(column_anomalies)
        anomaly_detection[column] = anomaly_summary
        distribution_analysis[column] = test_normality(df[column])

        if numeric_profiles[column]["iqr_outlier_count"] > 0:
            columns_with_outliers.append(column)

    for column in categorical_columns:
        categorical_profiles[column] = compute_categorical_profile(df[column])

    normal_distribution_count = sum(
        1 for result in distribution_analysis.values() if result["is_normal"] is True
    )

    profile = {
        "dataset": dataset_name,
        "total_rows": int(len(df)),
        "column_completeness": column_completeness,
        "numeric_profiles": numeric_profiles,
        "categorical_profiles": categorical_profiles,
        "distribution_analysis": distribution_analysis,
        "anomaly_detection": anomaly_detection,
    }

    summary = {
        "numeric_columns_profiled": len(numeric_columns),
        "categorical_columns_profiled": len(categorical_columns),
        "total_anomalies_detected": len(anomalies),
        "columns_with_outliers": columns_with_outliers,
        "normal_distributions": normal_distribution_count,
        "total_numeric_columns": len(numeric_columns),
    }

    return profile, anomalies, summary


def save_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Save a profile payload to disk as formatted JSON."""

    with file_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def save_anomalies(file_path: Path, anomalies: list[dict[str, Any]]) -> None:
    """Save anomaly records to CSV with the required column order."""

    columns = ["dataset", "column", "row_index", "value", "z_score", "iqr_flag", "z_flag"]
    anomalies_df = pd.DataFrame(anomalies, columns=columns)
    anomalies_df.sort_values(by=["dataset", "column", "row_index"], inplace=True)
    anomalies_df.to_csv(file_path, index=False)


def print_summary(user_summary: dict[str, Any], transaction_summary: dict[str, Any]) -> None:
    """Print the requested statistical profiling summary for both datasets."""

    print("=== STATISTICAL PROFILING REPORT ===")
    print("Users dataset:")
    print(f"Numeric columns profiled: {user_summary['numeric_columns_profiled']}")
    print(f"Categorical columns profiled: {user_summary['categorical_columns_profiled']}")
    print(f"Total anomalies detected: {user_summary['total_anomalies_detected']}")
    print(f"Columns with outliers: {user_summary['columns_with_outliers']}")
    print(
        "Normal distributions: "
        f"{user_summary['normal_distributions']}/{user_summary['total_numeric_columns']} columns"
    )
    print()
    print("Transactions dataset:")
    print(f"Numeric columns profiled: {transaction_summary['numeric_columns_profiled']}")
    print(f"Categorical columns profiled: {transaction_summary['categorical_columns_profiled']}")
    print(f"Total anomalies detected: {transaction_summary['total_anomalies_detected']}")
    print(f"Columns with outliers: {transaction_summary['columns_with_outliers']}")
    print(
        "Normal distributions: "
        f"{transaction_summary['normal_distributions']}/{transaction_summary['total_numeric_columns']} columns"
    )
    print("=====================================")


def main() -> None:
    """Run statistical profiling and anomaly detection for both synthetic datasets."""

    data_directory = get_data_directory()
    users_df = load_dataset(data_directory / "synthetic_users.csv", DATE_COLUMNS["users"])
    transactions_df = load_dataset(
        data_directory / "synthetic_transactions.csv", DATE_COLUMNS["transactions"]
    )

    user_profile, user_anomalies, user_summary = profile_dataset("users", users_df)
    transaction_profile, transaction_anomalies, transaction_summary = profile_dataset(
        "transactions", transactions_df
    )

    save_json(data_directory / "user_profile.json", user_profile)
    save_json(data_directory / "transaction_profile.json", transaction_profile)
    save_anomalies(data_directory / "anomalies.csv", user_anomalies + transaction_anomalies)
    print_summary(user_summary, transaction_summary)


if __name__ == "__main__":
    main()
