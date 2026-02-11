"""Generate synthetic user and transaction datasets with intentional quality issues."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from faker import Faker


SEED = 42
USER_COUNT = 2000
TRANSACTION_COUNT = 5000

COUNTRIES = ["USA", "UK", "Canada", "Germany", "France", "India", "Australia"]
SUBSCRIPTION_PLANS = ["free", "basic", "premium", "enterprise"]
CHURN_RISK_LEVELS = ["low", "medium", "high"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD"]
TRANSACTION_CATEGORIES = ["subscription", "refund", "upgrade", "addon", "penalty"]
TRANSACTION_STATUSES = ["completed", "pending", "failed", "reversed"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer"]


def seed_generators() -> Faker:
    """Seed NumPy and Faker so dataset generation is reproducible."""

    np.random.seed(SEED)
    Faker.seed(SEED)
    fake = Faker()
    fake.seed_instance(SEED)
    return fake


def ensure_data_directory() -> Path:
    """Create the project data directory when it does not already exist."""

    data_directory = Path(__file__).resolve().parents[1] / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    return data_directory


def sample_random_dates(start: str, end: str, size: int) -> pd.Series:
    """Return a series of random dates sampled uniformly between two bounds."""

    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    random_days = np.random.randint(0, (end_timestamp - start_timestamp).days + 1, size=size)
    return pd.Series(start_timestamp + pd.to_timedelta(random_days, unit="D"))


def sample_random_datetimes(start: str, end: str, size: int) -> pd.Series:
    """Return a series of random datetimes sampled uniformly between two bounds."""

    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    total_seconds = int((end_timestamp - start_timestamp).total_seconds())
    random_seconds = np.random.randint(0, total_seconds + 1, size=size)
    return pd.Series(start_timestamp + pd.to_timedelta(random_seconds, unit="s"))


def generate_user_dataset(fake: Faker, row_count: int = USER_COUNT) -> pd.DataFrame:
    """Build the base synthetic user dataset before quality issues are injected."""

    names = [fake.name() for _ in range(row_count)]
    emails = [fake.email() for _ in range(row_count)]

    users_df = pd.DataFrame(
        {
            "user_id": [f"USR_{index:04d}" for index in range(1, row_count + 1)],
            "name": names,
            "age": np.random.randint(18, 81, size=row_count),
            "email": emails,
            "signup_date": sample_random_dates("2020-01-01", "2024-12-31", row_count),
            "country": np.random.choice(COUNTRIES, size=row_count),
            "subscription_plan": np.random.choice(SUBSCRIPTION_PLANS, size=row_count),
            "monthly_spend": np.round(np.random.uniform(0, 500, size=row_count), 2),
            "is_active": np.random.choice([True, False], size=row_count),
            "churn_risk": np.random.choice(CHURN_RISK_LEVELS, size=row_count),
        }
    )

    return users_df


def generate_transaction_dataset(
    user_ids: Iterable[str], row_count: int = TRANSACTION_COUNT
) -> pd.DataFrame:
    """Build the base synthetic transaction dataset before quality issues are injected."""

    transactions_df = pd.DataFrame(
        {
            "transaction_id": [f"TXN_{index:05d}" for index in range(1, row_count + 1)],
            "user_id": np.random.choice(list(user_ids), size=row_count, replace=True),
            "transaction_date": sample_random_datetimes(
                "2020-01-01 00:00:00", "2024-12-31 23:59:59", row_count
            ),
            "amount": np.round(np.random.uniform(1.0, 1000.0, size=row_count), 2),
            "currency": np.random.choice(CURRENCIES, size=row_count),
            "category": np.random.choice(TRANSACTION_CATEGORIES, size=row_count),
            "status": np.random.choice(TRANSACTION_STATUSES, size=row_count),
            "payment_method": np.random.choice(PAYMENT_METHODS, size=row_count),
        }
    )

    return transactions_df


def allocate_issue_indices(total_rows: int, issue_counts: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Allocate disjoint row indices for each issue type and reserve a clean source pool."""

    if sum(issue_counts.values()) >= total_rows:
        raise ValueError("Issue counts must leave at least one clean row available.")

    shuffled_indices = np.random.permutation(total_rows)
    allocations: Dict[str, np.ndarray] = {}
    cursor = 0

    for issue_name, count in issue_counts.items():
        allocations[issue_name] = shuffled_indices[cursor : cursor + count]
        cursor += count

    allocations["clean_pool"] = shuffled_indices[cursor:]
    return allocations


def inject_null_values(df: pd.DataFrame, row_indices: np.ndarray, critical_columns: list[str]) -> None:
    """Inject null values into one randomly selected critical column per chosen row."""

    for row_index in row_indices:
        column_name = np.random.choice(critical_columns)
        null_value = pd.NaT if pd.api.types.is_datetime64_any_dtype(df[column_name]) else np.nan
        df.at[row_index, column_name] = null_value


def inject_out_of_range_values(
    df: pd.DataFrame, row_indices: np.ndarray, column_name: str, invalid_value: float | int
) -> None:
    """Inject an out-of-range value into the specified column for selected rows."""

    df.loc[row_indices, column_name] = invalid_value


def inject_duplicate_records(
    df: pd.DataFrame, target_indices: np.ndarray, source_pool: np.ndarray
) -> None:
    """Replace selected rows with copies of clean rows to create duplicates without changing row count."""

    source_indices = np.random.choice(source_pool, size=len(target_indices), replace=True)

    for target_index, source_index in zip(target_indices, source_indices):
        df.loc[target_index] = df.loc[source_index].copy()


def inject_invalid_email_formats(df: pd.DataFrame, row_indices: np.ndarray) -> None:
    """Replace valid email addresses with intentionally invalid email strings."""

    for invalid_counter, row_index in enumerate(row_indices, start=1):
        df.at[row_index, "email"] = f"invalid-email-{invalid_counter:04d}"


def inject_invalid_dates(
    df: pd.DataFrame, row_indices: np.ndarray, date_column: str, include_time: bool
) -> None:
    """Inject future dates beyond 2024 into the selected datetime column."""

    if include_time:
        invalid_dates = sample_random_datetimes(
            "2025-01-01 00:00:00", "2026-12-31 23:59:59", len(row_indices)
        )
    else:
        invalid_dates = sample_random_dates("2025-01-01", "2026-12-31", len(row_indices))

    for row_index, invalid_date in zip(row_indices, invalid_dates):
        df.at[row_index, date_column] = invalid_date


def inject_user_quality_issues(users_df: pd.DataFrame) -> Dict[str, int]:
    """Inject the required quality issues into the synthetic users dataset."""

    issue_counts = {
        "null_values": int(len(users_df) * 0.03),
        "out_of_range_values": int(len(users_df) * 0.02),
        "duplicate_records": int(len(users_df) * 0.02),
        "invalid_email_formats": int(len(users_df) * 0.01),
        "invalid_dates": int(len(users_df) * 0.01),
    }

    allocations = allocate_issue_indices(len(users_df), issue_counts)

    inject_null_values(
        users_df,
        allocations["null_values"],
        ["user_id", "name", "email", "signup_date", "country", "subscription_plan"],
    )
    inject_out_of_range_values(users_df, allocations["out_of_range_values"], "age", 999)
    inject_duplicate_records(users_df, allocations["duplicate_records"], allocations["clean_pool"])
    inject_invalid_email_formats(users_df, allocations["invalid_email_formats"])
    inject_invalid_dates(users_df, allocations["invalid_dates"], "signup_date", include_time=False)

    return issue_counts


def inject_transaction_quality_issues(transactions_df: pd.DataFrame) -> Dict[str, int]:
    """Inject the required quality issues into the synthetic transactions dataset."""

    issue_counts = {
        "null_values": int(len(transactions_df) * 0.03),
        "out_of_range_values": int(len(transactions_df) * 0.02),
        "duplicate_records": int(len(transactions_df) * 0.02),
        "invalid_dates": int(len(transactions_df) * 0.01),
    }

    allocations = allocate_issue_indices(len(transactions_df), issue_counts)

    inject_null_values(
        transactions_df,
        allocations["null_values"],
        ["user_id", "transaction_date", "category", "status", "payment_method"],
    )
    inject_out_of_range_values(transactions_df, allocations["out_of_range_values"], "amount", -500)
    inject_duplicate_records(
        transactions_df, allocations["duplicate_records"], allocations["clean_pool"]
    )
    inject_invalid_dates(
        transactions_df, allocations["invalid_dates"], "transaction_date", include_time=True
    )

    return issue_counts


def print_dataset_summary(dataset_name: str, df: pd.DataFrame, issue_counts: Dict[str, int]) -> None:
    """Print a concise summary of dataset size, issue counts, and column dtypes."""

    print(f"\nSummary for {dataset_name}")
    print(f"Row count: {len(df)}")
    print("Injected issue counts:")
    for issue_name, count in issue_counts.items():
        print(f"  - {issue_name}: {count}")
    print("Column dtypes:")
    for column_name, dtype in df.dtypes.items():
        print(f"  - {column_name}: {dtype}")


def main() -> None:
    """Generate datasets, inject quality issues, save outputs, and print summaries."""

    print("Starting synthetic data generation with seed 42...")
    fake = seed_generators()
    data_directory = ensure_data_directory()

    print("Generating synthetic users dataset...")
    users_df = generate_user_dataset(fake)
    clean_user_ids = users_df["user_id"].tolist()

    print("Generating synthetic transactions dataset...")
    transactions_df = generate_transaction_dataset(clean_user_ids)

    print("Injecting data quality issues into users dataset...")
    user_issue_counts = inject_user_quality_issues(users_df)

    print("Injecting data quality issues into transactions dataset...")
    transaction_issue_counts = inject_transaction_quality_issues(transactions_df)

    users_output_path = data_directory / "synthetic_users.csv"
    transactions_output_path = data_directory / "synthetic_transactions.csv"

    print(f"Saving users dataset to {users_output_path}...")
    users_df.to_csv(users_output_path, index=False)

    print(f"Saving transactions dataset to {transactions_output_path}...")
    transactions_df.to_csv(transactions_output_path, index=False)

    print("\nSynthetic data generation completed successfully.")
    print_dataset_summary("data/synthetic_users.csv", users_df, user_issue_counts)
    print_dataset_summary("data/synthetic_transactions.csv", transactions_df, transaction_issue_counts)


if __name__ == "__main__":
    main()
