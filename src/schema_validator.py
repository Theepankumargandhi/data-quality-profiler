"""Validate synthetic datasets against predefined schemas and emit validation reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_COUNTRIES = ["USA", "UK", "Canada", "Germany", "France", "India", "Australia"]
ALLOWED_SUBSCRIPTION_PLANS = ["free", "basic", "premium", "enterprise"]
ALLOWED_CHURN_RISK = ["low", "medium", "high"]
ALLOWED_CURRENCIES = ["USD", "EUR", "GBP", "CAD"]
ALLOWED_CATEGORIES = ["subscription", "refund", "upgrade", "addon", "penalty"]
ALLOWED_STATUSES = ["completed", "pending", "failed", "reversed"]
ALLOWED_PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer"]

USER_ID_PATTERN = re.compile(r"^USR_\d{4}$")
TRANSACTION_ID_PATTERN = re.compile(r"^TXN_\d{5}$")
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
MAX_ALLOWED_SIGNUP_DATE = pd.Timestamp("2024-12-31")
MAX_ALLOWED_TRANSACTION_DATE = pd.Timestamp("2024-12-31 23:59:59")

USER_SCHEMA = {
    "user_id": {"dtype": "string", "pattern": USER_ID_PATTERN},
    "age": {"dtype": "integer", "min_value": 18, "max_value": 80},
    "email": {"dtype": "string", "pattern": EMAIL_PATTERN},
    "signup_date": {"dtype": "date", "max_value": MAX_ALLOWED_SIGNUP_DATE},
    "country": {"dtype": "string", "allowed": ALLOWED_COUNTRIES},
    "subscription_plan": {"dtype": "string", "allowed": ALLOWED_SUBSCRIPTION_PLANS},
    "monthly_spend": {"dtype": "float", "min_value": 0.0, "max_value": 500.0},
    "is_active": {"dtype": "boolean"},
    "churn_risk": {"dtype": "string", "allowed": ALLOWED_CHURN_RISK},
}

TRANSACTION_SCHEMA = {
    "transaction_id": {"dtype": "string", "pattern": TRANSACTION_ID_PATTERN},
    "amount": {
        "dtype": "float",
        "min_value": 0.0,
        "max_value": 1000.0,
        "min_inclusive": False,
        "max_inclusive": False,
    },
    "currency": {"dtype": "string", "allowed": ALLOWED_CURRENCIES},
    "category": {"dtype": "string", "allowed": ALLOWED_CATEGORIES},
    "status": {"dtype": "string", "allowed": ALLOWED_STATUSES},
    "payment_method": {"dtype": "string", "allowed": ALLOWED_PAYMENT_METHODS},
    "transaction_date": {"dtype": "datetime", "max_value": MAX_ALLOWED_TRANSACTION_DATE},
}


def get_data_directory() -> Path:
    """Return the project data directory path, creating it if necessary."""

    data_directory = Path(__file__).resolve().parents[1] / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    return data_directory


def build_issue(
    row_index: int,
    column: str,
    check_type: str,
    expected: str,
    actual: Any,
    severity: str,
) -> dict[str, Any]:
    """Create a normalized validation issue record for report output."""

    actual_value = "null" if pd.isna(actual) else str(actual)
    return {
        "row_index": int(row_index),
        "column": column,
        "check_type": check_type,
        "expected": expected,
        "actual": actual_value,
        "severity": severity,
    }


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load a CSV dataset from disk into a pandas DataFrame."""

    return pd.read_csv(file_path)


def normalize_boolean_series(series: pd.Series) -> pd.Series:
    """Normalize common boolean representations while preserving nulls."""

    valid_boolean_map = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
    }

    return series.map(
        lambda value: np.nan if pd.isna(value) else valid_boolean_map.get(value, np.nan)
    )


def validate_nulls(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate null values across all columns and flag columns above 1 percent nulls."""

    issues: list[dict[str, Any]] = []
    null_percentages = df.isna().mean()

    for column, null_percentage in null_percentages.items():
        severity = "critical" if null_percentage > 0.01 else "medium"
        expected = f"non-null value (column null rate: {null_percentage:.2%}; critical threshold: 1.00%)"

        for row_index in df.index[df[column].isna()]:
            issues.append(build_issue(row_index, column, "null", expected, np.nan, severity))

    return issues


def validate_types(df: pd.DataFrame, schema: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate row-level values against expected logical data types."""

    issues: list[dict[str, Any]] = []

    for column, rules in schema.items():
        expected_dtype = rules["dtype"]
        series = df[column]

        if expected_dtype == "integer":
            converted = pd.to_numeric(series, errors="coerce")
            invalid_mask = series.notna() & (converted.isna() | (np.floor(converted) != converted))
        elif expected_dtype == "float":
            converted = pd.to_numeric(series, errors="coerce")
            invalid_mask = series.notna() & converted.isna()
        elif expected_dtype == "boolean":
            converted = normalize_boolean_series(series)
            invalid_mask = series.notna() & converted.isna()
        elif expected_dtype in {"date", "datetime"}:
            converted = pd.to_datetime(series, errors="coerce")
            invalid_mask = series.notna() & converted.isna()
        else:
            invalid_mask = series.notna() & ~series.map(lambda value: isinstance(value, str))

        for row_index in df.index[invalid_mask]:
            issues.append(
                build_issue(
                    row_index,
                    column,
                    "type",
                    f"{expected_dtype} value",
                    series.loc[row_index],
                    "high",
                )
            )

    return issues


def validate_ranges(df: pd.DataFrame, schema: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate numeric and temporal columns against configured bounds."""

    issues: list[dict[str, Any]] = []

    for column, rules in schema.items():
        expected_dtype = rules["dtype"]

        if expected_dtype in {"integer", "float"} and (
            "min_value" in rules or "max_value" in rules
        ):
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            valid_mask = df[column].notna() & numeric_series.notna()
            invalid_mask = pd.Series(False, index=df.index)

            if "min_value" in rules:
                min_inclusive = rules.get("min_inclusive", True)
                if min_inclusive:
                    invalid_mask |= valid_mask & (numeric_series < rules["min_value"])
                else:
                    invalid_mask |= valid_mask & (numeric_series <= rules["min_value"])

            if "max_value" in rules:
                max_inclusive = rules.get("max_inclusive", True)
                if max_inclusive:
                    invalid_mask |= valid_mask & (numeric_series > rules["max_value"])
                else:
                    invalid_mask |= valid_mask & (numeric_series >= rules["max_value"])

            expected = describe_range_expectation(rules)

            for row_index in df.index[invalid_mask]:
                issues.append(
                    build_issue(
                        row_index,
                        column,
                        "range",
                        expected,
                        df.loc[row_index, column],
                        "high",
                    )
                )

        if expected_dtype in {"date", "datetime"} and "max_value" in rules:
            parsed_dates = pd.to_datetime(df[column], errors="coerce")
            invalid_mask = df[column].notna() & parsed_dates.notna() & (parsed_dates > rules["max_value"])
            expected = f"date/datetime on or before {rules['max_value']}"

            for row_index in df.index[invalid_mask]:
                issues.append(
                    build_issue(
                        row_index,
                        column,
                        "range",
                        expected,
                        df.loc[row_index, column],
                        "high",
                    )
                )

    return issues


def describe_range_expectation(rules: dict[str, Any]) -> str:
    """Create a readable expected-range description from schema rules."""

    minimum = rules.get("min_value")
    maximum = rules.get("max_value")
    min_operator = ">=" if rules.get("min_inclusive", True) else ">"
    max_operator = "<=" if rules.get("max_inclusive", True) else "<"

    if minimum is not None and maximum is not None:
        return f"value {min_operator} {minimum} and {max_operator} {maximum}"
    if minimum is not None:
        return f"value {min_operator} {minimum}"
    return f"value {max_operator} {maximum}"


def validate_formats(df: pd.DataFrame, schema: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate regex patterns and allowed categorical values."""

    issues: list[dict[str, Any]] = []

    for column, rules in schema.items():
        series = df[column]

        if "pattern" in rules:
            pattern = rules["pattern"]
            invalid_mask = series.notna() & ~series.map(
                lambda value: bool(pattern.fullmatch(value)) if isinstance(value, str) else False
            )

            for row_index in df.index[invalid_mask]:
                issues.append(
                    build_issue(
                        row_index,
                        column,
                        "format",
                        f"match pattern {pattern.pattern}",
                        series.loc[row_index],
                        "high",
                    )
                )

        if "allowed" in rules:
            invalid_mask = series.notna() & ~series.isin(rules["allowed"])
            expected = f"one of {rules['allowed']}"

            for row_index in df.index[invalid_mask]:
                issues.append(
                    build_issue(
                        row_index,
                        column,
                        "format",
                        expected,
                        series.loc[row_index],
                        "high",
                    )
                )

    return issues


def validate_duplicates(df: pd.DataFrame, id_column: str) -> list[dict[str, Any]]:
    """Validate exact duplicate rows and duplicate identifier values."""

    issues: list[dict[str, Any]] = []
    exact_duplicate_mask = df.duplicated(keep="first")
    duplicate_id_mask = df[id_column].notna() & df[id_column].duplicated(keep="first")

    for row_index in df.index[exact_duplicate_mask]:
        issues.append(
            build_issue(
                row_index,
                "__row__",
                "duplicate",
                "unique row",
                "duplicate record",
                "high",
            )
        )

    for row_index in df.index[duplicate_id_mask]:
        issues.append(
            build_issue(
                row_index,
                id_column,
                "duplicate",
                "unique identifier",
                df.loc[row_index, id_column],
                "critical",
            )
        )

    return issues


def create_report_dataframe(issues: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a consistently shaped report DataFrame from collected issues."""

    columns = ["row_index", "column", "check_type", "expected", "actual", "severity"]

    if not issues:
        return pd.DataFrame(columns=columns)

    report_df = pd.DataFrame(issues, columns=columns)
    return report_df.sort_values(by=["row_index", "column", "check_type"]).reset_index(drop=True)


def build_summary(total_rows: int, report_df: pd.DataFrame) -> dict[str, Any]:
    """Build summary statistics for a validation run."""

    failed_rows = int(report_df["row_index"].nunique()) if not report_df.empty else 0
    passed_rows = int(total_rows - failed_rows)
    pass_rate = round((passed_rows / total_rows) * 100, 2) if total_rows else 0.0

    return {
        "total_rows": int(total_rows),
        "passed_rows": passed_rows,
        "failed_rows": failed_rows,
        "pass_rate": pass_rate,
        "issues_by_type": {
            "null": int((report_df["check_type"] == "null").sum()) if not report_df.empty else 0,
            "range": int((report_df["check_type"] == "range").sum()) if not report_df.empty else 0,
            "format": int((report_df["check_type"] == "format").sum()) if not report_df.empty else 0,
            "duplicate": int((report_df["check_type"] == "duplicate").sum()) if not report_df.empty else 0,
        },
    }


def validate_dataset(
    file_path: Path,
    schema: dict[str, dict[str, Any]],
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate a dataset file and return the raw data, report, and summary."""

    df = load_dataset(file_path)
    issues: list[dict[str, Any]] = []
    issues.extend(validate_nulls(df))
    issues.extend(validate_types(df, schema))
    issues.extend(validate_ranges(df, schema))
    issues.extend(validate_formats(df, schema))
    issues.extend(validate_duplicates(df, id_column))

    report_df = create_report_dataframe(issues)
    summary = build_summary(len(df), report_df)
    return df, report_df, summary


def save_validation_outputs(
    user_report: pd.DataFrame,
    transaction_report: pd.DataFrame,
    summary_payload: dict[str, Any],
    data_directory: Path,
) -> None:
    """Save validation reports and summary payload to the data directory."""

    user_report.to_csv(data_directory / "user_validation_report.csv", index=False)
    transaction_report.to_csv(data_directory / "transaction_validation_report.csv", index=False)

    with (data_directory / "validation_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary_payload, output_file, indent=2)


def print_report(user_summary: dict[str, Any], transaction_summary: dict[str, Any]) -> None:
    """Print the formatted schema validation report requested for both datasets."""

    print("=== SCHEMA VALIDATION REPORT ===")
    print("USERS dataset:")
    print(f"Total rows: {user_summary['total_rows']}")
    print(f"Passed: {user_summary['passed_rows']} ({user_summary['pass_rate']:.2f}%)")
    print(
        f"Failed: {user_summary['failed_rows']} "
        f"({100 - user_summary['pass_rate']:.2f}%)"
    )
    print("Issues by type:")
    print(f"  Null violations: {user_summary['issues_by_type']['null']}")
    print(f"  Range violations: {user_summary['issues_by_type']['range']}")
    print(f"  Format violations: {user_summary['issues_by_type']['format']}")
    print(f"  Duplicates: {user_summary['issues_by_type']['duplicate']}")
    print()
    print("TRANSACTIONS dataset:")
    print(f"Total rows: {transaction_summary['total_rows']}")
    print(f"Passed: {transaction_summary['passed_rows']} ({transaction_summary['pass_rate']:.2f}%)")
    print(
        f"Failed: {transaction_summary['failed_rows']} "
        f"({100 - transaction_summary['pass_rate']:.2f}%)"
    )
    print("Issues by type:")
    print(f"  Null violations: {transaction_summary['issues_by_type']['null']}")
    print(f"  Range violations: {transaction_summary['issues_by_type']['range']}")
    print(f"  Format violations: {transaction_summary['issues_by_type']['format']}")
    print(f"  Duplicates: {transaction_summary['issues_by_type']['duplicate']}")
    print("================================")


def main() -> None:
    """Run schema validation for the synthetic users and transactions datasets."""

    data_directory = get_data_directory()
    _, user_report, user_summary = validate_dataset(
        data_directory / "synthetic_users.csv", USER_SCHEMA, "user_id"
    )
    _, transaction_report, transaction_summary = validate_dataset(
        data_directory / "synthetic_transactions.csv",
        TRANSACTION_SCHEMA,
        "transaction_id",
    )

    summary_payload = {
        "users": user_summary,
        "transactions": transaction_summary,
    }

    save_validation_outputs(user_report, transaction_report, summary_payload, data_directory)
    print_report(user_summary, transaction_summary)


if __name__ == "__main__":
    main()

