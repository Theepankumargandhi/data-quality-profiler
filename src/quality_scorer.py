"""Compute weighted data quality scores from validation, profiling, and anomaly outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REPORT_DIVIDER = "-" * 21


def get_data_directory() -> Path:
    """Return the project data directory path."""

    return Path(__file__).resolve().parents[1] / "data"


def load_json(file_path: Path) -> dict[str, Any]:
    """Load a JSON file from disk and return its parsed contents."""

    with file_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Load the required validation, profiling, and anomaly artifacts."""

    data_directory = get_data_directory()
    validation_summary = load_json(data_directory / "validation_summary.json")
    user_profile = load_json(data_directory / "user_profile.json")
    transaction_profile = load_json(data_directory / "transaction_profile.json")
    anomalies_df = pd.read_csv(data_directory / "anomalies.csv")
    return validation_summary, user_profile, transaction_profile, anomalies_df


def compute_completeness_score(profile: dict[str, Any]) -> tuple[float, int]:
    """Compute the completeness score and count columns above the null threshold."""

    column_completeness = profile["column_completeness"]
    completeness_scores = [
        100 - (metrics["null_percentage"] * 100) for metrics in column_completeness.values()
    ]
    columns_above_threshold = sum(
        1 for metrics in column_completeness.values() if metrics["null_percentage"] > 0.01
    )
    average_score = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
    return round(average_score, 2), columns_above_threshold


def compute_validity_score(validation_dataset_summary: dict[str, Any]) -> tuple[float, float]:
    """Compute the validity score and failed-record percentage from validation results."""

    validity = float(validation_dataset_summary["pass_rate"])
    failed_percentage = 100 - validity
    return round(validity, 2), round(failed_percentage, 2)


def compute_consistency_score(validation_dataset_summary: dict[str, Any]) -> float:
    """Compute the consistency score from duplicate validation counts."""

    duplicate_count = validation_dataset_summary["issues_by_type"]["duplicate"]
    total_rows = validation_dataset_summary["total_rows"]
    duplicate_percentage = duplicate_count / total_rows if total_rows else 0.0
    return round(max(0.0, 100 - (duplicate_percentage * 100)), 2)


def compute_accuracy_score(anomalies_df: pd.DataFrame, dataset_name: str, total_rows: int) -> tuple[float, int]:
    """Compute the accuracy score and number of anomalous columns for a dataset."""

    dataset_anomalies = anomalies_df[anomalies_df["dataset"] == dataset_name]
    anomaly_count = len(dataset_anomalies)
    anomaly_percentage = anomaly_count / total_rows if total_rows else 0.0
    anomaly_columns = int(dataset_anomalies["column"].nunique())
    score = max(0.0, 100 - (anomaly_percentage * 100))
    return round(score, 2), anomaly_columns


def compute_overall_score(
    completeness: float,
    validity: float,
    consistency: float,
    accuracy: float,
) -> float:
    """Compute the overall score using equal 25 percent weights."""

    weighted_score = (0.25 * completeness) + (0.25 * validity) + (0.25 * consistency) + (0.25 * accuracy)
    return round(weighted_score, 2)


def assign_grade(overall_score: float) -> str:
    """Assign a quality grade from the overall score."""

    if overall_score >= 90:
        return "A"
    if overall_score >= 80:
        return "B"
    if overall_score >= 70:
        return "C"
    if overall_score >= 60:
        return "D"
    return "F"


def generate_recommendations(
    completeness: float,
    validity: float,
    consistency: float,
    accuracy: float,
    columns_above_threshold: int,
    failed_percentage: float,
    anomaly_columns: int,
) -> list[str]:
    """Generate dataset recommendations based on score thresholds."""

    recommendations: list[str] = []

    if completeness < 90:
        recommendations.append(
            "Critical: "
            f"{columns_above_threshold} columns have null rates above threshold. "
            "Recommend data collection review."
        )

    if validity < 85:
        recommendations.append(
            f"Warning: {failed_percentage:.1f}% of records fail schema validation. "
            "Recommend data source audit."
        )

    if consistency < 95:
        recommendations.append(
            "Warning: Duplicate records detected. Recommend deduplication pipeline."
        )

    if accuracy < 90:
        recommendations.append(
            f"Info: Anomalies detected in {anomaly_columns} columns. "
            "Recommend outlier investigation."
        )

    return recommendations


def build_dataset_score(
    dataset_name: str,
    validation_dataset_summary: dict[str, Any],
    profile: dict[str, Any],
    anomalies_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute the full scorecard for one dataset."""

    completeness, columns_above_threshold = compute_completeness_score(profile)
    validity, failed_percentage = compute_validity_score(validation_dataset_summary)
    consistency = compute_consistency_score(validation_dataset_summary)
    accuracy, anomaly_columns = compute_accuracy_score(
        anomalies_df, dataset_name, validation_dataset_summary["total_rows"]
    )
    overall_score = compute_overall_score(completeness, validity, consistency, accuracy)
    grade = assign_grade(overall_score)
    recommendations = generate_recommendations(
        completeness,
        validity,
        consistency,
        accuracy,
        columns_above_threshold,
        failed_percentage,
        anomaly_columns,
    )

    return {
        "completeness": completeness,
        "validity": validity,
        "consistency": consistency,
        "accuracy": accuracy,
        "overall_score": overall_score,
        "grade": grade,
        "recommendations": recommendations,
    }


def save_quality_scores(quality_scores: dict[str, Any]) -> None:
    """Save the quality score payload to data/quality_scores.json."""

    output_path = get_data_directory() / "quality_scores.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(quality_scores, output_file, indent=2)


def print_dataset_report(dataset_label: str, dataset_score: dict[str, Any]) -> None:
    """Print the formatted scorecard for a single dataset."""

    print(f"{dataset_label} Dataset:")
    print(f"Completeness:  {dataset_score['completeness']:.1f}%")
    print(f"Validity:      {dataset_score['validity']:.1f}%")
    print(f"Consistency:   {dataset_score['consistency']:.1f}%")
    print(f"Accuracy:      {dataset_score['accuracy']:.1f}%")
    print(REPORT_DIVIDER)
    print(f"Overall Score: {dataset_score['overall_score']:.1f}% (Grade: {dataset_score['grade']})")
    print()
    print("Recommendations:")

    if dataset_score["recommendations"]:
        for recommendation in dataset_score["recommendations"]:
            print(recommendation)
    else:
        print("None")

    print()


def print_report(quality_scores: dict[str, Any]) -> None:
    """Print the complete data quality scorecard."""

    print("=== DATA QUALITY SCORECARD ===")
    print_dataset_report("USERS", quality_scores["users"])
    print_dataset_report("TRANSACTIONS", quality_scores["transactions"])
    print("==============================")


def main() -> None:
    """Load profiling artifacts, compute quality scores, save them, and print a report."""

    validation_summary, user_profile, transaction_profile, anomalies_df = load_inputs()

    quality_scores = {
        "users": build_dataset_score(
            "users", validation_summary["users"], user_profile, anomalies_df
        ),
        "transactions": build_dataset_score(
            "transactions", validation_summary["transactions"], transaction_profile, anomalies_df
        ),
    }

    save_quality_scores(quality_scores)
    print_report(quality_scores)


if __name__ == "__main__":
    main()

