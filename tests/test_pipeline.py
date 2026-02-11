"""Pytest coverage for the synthetic data quality validation pipeline."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src import generate_synthetic_data, quality_scorer, schema_validator, statistical_profiler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
USERS_CSV_PATH = DATA_DIR / "synthetic_users.csv"
QUALITY_SCORES_PATH = DATA_DIR / "quality_scores.json"
VALIDATION_SUMMARY_PATH = DATA_DIR / "validation_summary.json"
USER_PROFILE_PATH = DATA_DIR / "user_profile.json"
TRANSACTION_PROFILE_PATH = DATA_DIR / "transaction_profile.json"
ANOMALIES_PATH = DATA_DIR / "anomalies.csv"
REQUIRED_USER_COLUMNS = {
    "user_id",
    "name",
    "age",
    "email",
    "signup_date",
    "country",
    "subscription_plan",
    "monthly_spend",
    "is_active",
    "churn_risk",
}



def ensure_pipeline_artifacts() -> None:
    """Generate required pipeline artifacts when they are not already present."""

    required_files = [
        QUALITY_SCORES_PATH,
        VALIDATION_SUMMARY_PATH,
        USER_PROFILE_PATH,
        TRANSACTION_PROFILE_PATH,
        ANOMALIES_PATH,
    ]

    if all(file_path.exists() for file_path in required_files):
        return

    generate_synthetic_data.main()
    schema_validator.main()
    statistical_profiler.main()
    quality_scorer.main()



def test_data_generation() -> None:
    """Generate synthetic user data and verify the expected file, row count, and columns exist."""

    generate_synthetic_data.main()

    assert USERS_CSV_PATH.exists()

    users_df = pd.read_csv(USERS_CSV_PATH)
    assert len(users_df) == 2000
    assert REQUIRED_USER_COLUMNS.issubset(users_df.columns)



def test_schema_validation() -> None:
    """Validate that a single out-of-range age produces exactly one range violation."""

    test_df = pd.DataFrame(
        {
            "user_id": ["USR_0001", "USR_0002", "USR_0003"],
            "age": [25, 32, 999],
            "email": ["user1@example.com", "user2@example.com", "user3@example.com"],
            "signup_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "country": ["USA", "UK", "Canada"],
            "subscription_plan": ["free", "basic", "premium"],
            "monthly_spend": [0.0, 49.99, 99.99],
            "is_active": [True, False, True],
            "churn_risk": ["low", "medium", "high"],
        }
    )

    range_issues = schema_validator.validate_ranges(test_df, schema_validator.USER_SCHEMA)

    assert len(range_issues) == 1
    assert range_issues[0]["check_type"] == "range"
    assert range_issues[0]["column"] == "age"



def test_quality_scoring() -> None:
    """Verify saved quality scores stay within valid bounds and use an allowed grade."""

    ensure_pipeline_artifacts()
    quality_scores = json.loads(QUALITY_SCORES_PATH.read_text(encoding="utf-8"))

    for dataset_name in ["users", "transactions"]:
        dataset_score = quality_scores[dataset_name]
        assert 0 <= dataset_score["overall_score"] <= 100
        assert dataset_score["grade"] in ["A", "B", "C", "D", "F"]



def test_statistical_profiler() -> None:
    """Verify the IQR anomaly logic flags two obvious numeric outliers."""

    series = pd.Series([10] * 20 + [100, 120])
    anomalies, _ = statistical_profiler.compute_anomaly_details("test", "value", series)
    flagged_values = sorted(anomaly["value"] for anomaly in anomalies if anomaly["iqr_flag"])

    assert flagged_values == [100, 120]



def test_db_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log a dummy run to an in-memory SQLite database and verify one history row is returned."""

    monkeypatch.setenv("DB_CONNECTION", "sqlite:///:memory:")
    db_logger = importlib.import_module("src.db_logger")
    db_logger = importlib.reload(db_logger)
    db_logger._ENGINE = None
    db_logger._SESSION_FACTORY = None

    db_logger.init_db()
    db_logger.log_run(
        "users",
        {
            "total_rows": 10,
            "passed_rows": 9,
            "failed_rows": 1,
            "pass_rate": 90.0,
            "issues_by_type": {
                "null": 1,
                "range": 1,
                "format": 0,
                "duplicate": 0,
            },
        },
        {
            "completeness": 99.0,
            "validity": 90.0,
            "consistency": 100.0,
            "accuracy": 95.0,
            "overall_score": 96.0,
            "grade": "A",
        },
    )

    run_history = db_logger.get_run_history()
    assert len(run_history) == 1
