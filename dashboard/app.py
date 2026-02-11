import json
import os
import sys

sys.path.append("..")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.db_logger import get_run_history


st.set_page_config(
    page_title="Synthetic Data Quality Validator",
    layout="wide"
)

BASE_DIR = PROJECT_ROOT
DATA_DIR = os.path.join(BASE_DIR, "data")
QUALITY_SCORES_PATH = os.path.join(DATA_DIR, "quality_scores.json")
VALIDATION_SUMMARY_PATH = os.path.join(DATA_DIR, "validation_summary.json")
USER_PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")
TRANSACTION_PROFILE_PATH = os.path.join(DATA_DIR, "transaction_profile.json")
USER_VALIDATION_REPORT_PATH = os.path.join(DATA_DIR, "user_validation_report.csv")
TRANSACTION_VALIDATION_REPORT_PATH = os.path.join(DATA_DIR, "transaction_validation_report.csv")
ANOMALIES_PATH = os.path.join(DATA_DIR, "anomalies.csv")
USERS_CSV_PATH = os.path.join(DATA_DIR, "synthetic_users.csv")
TRANSACTIONS_CSV_PATH = os.path.join(DATA_DIR, "synthetic_transactions.csv")
QUALITY_DB_PATH = os.path.join(DATA_DIR, "quality_logs.db")
ISSUE_ORDER = ["null", "range", "format", "duplicate"]
DATASET_LABELS = {
    "users": "Users",
    "transactions": "Transactions",
}

sns.set_theme(style="whitegrid")



def load_json_file(file_path: str) -> dict | None:
    """Load a JSON file from disk and return the parsed payload."""

    if not os.path.exists(file_path):
        st.warning(f"Missing file: {os.path.relpath(file_path, BASE_DIR)}")
        return None

    with open(file_path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)



def load_csv_file(file_path: str, **kwargs) -> pd.DataFrame | None:
    """Load a CSV file from disk and return a DataFrame."""

    if not os.path.exists(file_path):
        st.warning(f"Missing file: {os.path.relpath(file_path, BASE_DIR)}")
        return None

    return pd.read_csv(file_path, **kwargs)



def load_run_history_data() -> pd.DataFrame | None:
    """Load pipeline run history from the SQLite database."""

    if not os.path.exists(QUALITY_DB_PATH):
        st.warning("Missing file: data/quality_logs.db")
        return None

    try:
        return get_run_history()
    except Exception:
        st.warning("Unable to load run history from the SQLite database.")
        return None



def get_selected_dataset_keys(selection: str) -> list[str]:
    """Return dataset keys based on the sidebar selection."""

    if selection == "Both":
        return ["users", "transactions"]
    if selection == "Users":
        return ["users"]
    return ["transactions"]



def score_color(score: float) -> str:
    """Return a display color for a numeric quality score."""

    if score > 90:
        return "#2e7d32"
    if score > 80:
        return "#ef6c00"
    return "#c62828"



def grade_color(grade: str) -> str:
    """Return a display color for the overall quality grade."""

    grade_colors = {
        "A": "#2e7d32",
        "B": "#1565c0",
        "C": "#ef6c00",
        "D": "#d84315",
        "F": "#c62828",
    }
    return grade_colors.get(grade, "#455a64")



def render_metric_card(title: str, value_text: str, color: str, large: bool = False) -> None:
    """Render a styled metric card with color-coded text."""

    font_size = "2rem" if large else "1.5rem"
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 0.75rem; border: 1px solid #d9d9d9; background-color: #fafafa; min-height: 140px;">
            <div style="font-size: 0.95rem; color: #555; margin-bottom: 0.5rem;">{title}</div>
            <div style="font-size: {font_size}; font-weight: 700; color: {color}; line-height: 1.25;">{value_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_quality_scorecard(selected_dataset_keys: list[str], quality_scores: dict | None) -> None:
    """Render the quality scorecards for the selected datasets."""

    st.header("Quality Scorecard")

    if quality_scores is None:
        return

    for dataset_key in selected_dataset_keys:
        dataset_scores = quality_scores.get(dataset_key)
        if dataset_scores is None:
            st.warning(f"Missing quality score data for {DATASET_LABELS[dataset_key]}.")
            continue

        st.subheader(f"{DATASET_LABELS[dataset_key]} Dataset")
        columns = st.columns(5)
        with columns[0]:
            render_metric_card(
                "Completeness Score",
                f"{dataset_scores['completeness']:.1f}%",
                score_color(dataset_scores["completeness"]),
            )
        with columns[1]:
            render_metric_card(
                "Validity Score",
                f"{dataset_scores['validity']:.1f}%",
                score_color(dataset_scores["validity"]),
            )
        with columns[2]:
            render_metric_card(
                "Consistency Score",
                f"{dataset_scores['consistency']:.1f}%",
                score_color(dataset_scores["consistency"]),
            )
        with columns[3]:
            render_metric_card(
                "Accuracy Score",
                f"{dataset_scores['accuracy']:.1f}%",
                score_color(dataset_scores["accuracy"]),
            )
        with columns[4]:
            render_metric_card(
                "Overall Score + Grade",
                f"{dataset_scores['overall_score']:.1f}%<br><span style='font-size: 1.05rem;'>Grade: {dataset_scores['grade']}</span>",
                grade_color(dataset_scores["grade"]),
                large=True,
            )



def render_validation_issue_breakdown(dataset_key: str, report_df: pd.DataFrame | None) -> None:
    """Render pie and bar charts summarizing validation issues for a dataset."""

    st.subheader(f"{DATASET_LABELS[dataset_key]} Dataset")

    if report_df is None:
        return

    if report_df.empty:
        st.warning(f"No validation issues found for {DATASET_LABELS[dataset_key]}.")
        return

    issue_counts = report_df["check_type"].value_counts().reindex(ISSUE_ORDER, fill_value=0)
    issue_counts_for_pie = issue_counts[issue_counts > 0]
    pie_col, bar_col = st.columns([0.9, 1.1], gap="medium")

    with pie_col:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.pie(
            issue_counts_for_pie.values,
            labels=[label.title() for label in issue_counts_for_pie.index],
            autopct="%1.1f%%",
            pctdistance=0.72,
            startangle=90,
            textprops={"fontsize": 11},
            wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
            colors=["#6c9ae0", "#d35a52", "#efc15c", "#9b59b6"],
        )
        ax.set_title("Issues by Type", fontsize=13)
        ax.axis("equal")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with bar_col:
        column_counts = (
            report_df.assign(column=report_df["column"].replace({"__row__": "row_duplicate"}))
            ["column"]
            .value_counts()
            .head(10)
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        colors = sns.color_palette("flare", n_colors=len(column_counts))
        ax.barh(column_counts.index, column_counts.values, color=colors)
        ax.set_title("Top 10 Columns with Most Errors", fontsize=13)
        ax.set_xlabel("Error Count")
        ax.set_ylabel("Column")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_validation_section(
    selected_dataset_keys: list[str],
    report_map: dict[str, pd.DataFrame | None],
) -> None:
    """Render the validation issue section with dataset tabs when needed."""

    st.header("Validation Issue Breakdown")

    if len(selected_dataset_keys) > 1:
        validation_tabs = st.tabs([DATASET_LABELS[key] for key in selected_dataset_keys])
        for validation_tab, dataset_key in zip(validation_tabs, selected_dataset_keys):
            with validation_tab:
                render_validation_issue_breakdown(dataset_key, report_map.get(dataset_key))
    else:
        dataset_key = selected_dataset_keys[0]
        render_validation_issue_breakdown(dataset_key, report_map.get(dataset_key))



def build_numeric_profile_table(profile_data: dict) -> pd.DataFrame:
    """Build a numeric summary table from profile JSON data."""

    rows = []
    for column_name, metrics in profile_data.get("numeric_profiles", {}).items():
        rows.append(
            {
                "column": column_name,
                "mean": metrics.get("mean"),
                "median": metrics.get("median"),
                "std": metrics.get("std"),
                "outlier_count": metrics.get("iqr_outlier_count"),
            }
        )

    return pd.DataFrame(rows)



def highlight_outlier_rows(row: pd.Series) -> list[str]:
    """Highlight rows containing outliers in the numeric profile table."""

    if row.get("outlier_count", 0) > 0:
        return ["background-color: #fff7cc"] * len(row)
    return [""] * len(row)



def render_numeric_tab(profile_data: dict) -> None:
    """Render the numeric profile table with outlier highlighting."""

    numeric_df = build_numeric_profile_table(profile_data)
    if numeric_df.empty:
        st.warning("No numeric profile data available.")
        return

    styled_df = numeric_df.style.apply(highlight_outlier_rows, axis=1)
    st.dataframe(styled_df, width="stretch", hide_index=True)



def render_categorical_tab(profile_data: dict) -> None:
    """Render top-value bar charts for each categorical column."""

    categorical_profiles = profile_data.get("categorical_profiles", {})
    if not categorical_profiles:
        st.warning("No categorical profile data available.")
        return

    profile_items = list(categorical_profiles.items())
    for index in range(0, len(profile_items), 2):
        chart_columns = st.columns(2, gap="medium")
        for chart_column, (column_name, metrics) in zip(chart_columns, profile_items[index : index + 2]):
            value_counts = metrics.get("value_counts", {})
            if not value_counts:
                continue

            chart_df = pd.DataFrame(
                {
                    "value": list(value_counts.keys()),
                    "count": list(value_counts.values()),
                }
            ).sort_values("count", ascending=False).head(5)

            with chart_column:
                st.markdown(f"**{column_name}**")
                fig, ax = plt.subplots(figsize=(5.2, 2.8))
                colors = sns.color_palette("crest", n_colors=len(chart_df))
                ax.barh(chart_df["value"], chart_df["count"], color=colors)
                ax.invert_yaxis()
                ax.set_xlabel("Count")
                ax.set_ylabel("Value")
                ax.set_title(f"Top 5 Values for {column_name}", fontsize=12)
                ax.grid(axis="x", linestyle="--", alpha=0.35)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)



def render_anomalies_tab(dataset_key: str, anomalies_df: pd.DataFrame | None) -> None:
    """Render anomaly tables and plots for a dataset."""

    if anomalies_df is None:
        return

    dataset_anomalies = anomalies_df[anomalies_df["dataset"] == dataset_key].copy()
    st.metric("Total anomaly count", len(dataset_anomalies))

    if dataset_anomalies.empty:
        st.info("No anomalies detected for this dataset.")
        return

    dataset_anomalies["z_score"] = pd.to_numeric(dataset_anomalies["z_score"], errors="coerce")
    dataset_anomalies["row_index"] = pd.to_numeric(dataset_anomalies["row_index"], errors="coerce")
    dataset_anomalies["value"] = pd.to_numeric(dataset_anomalies["value"], errors="coerce")

    top_anomalies = (
        dataset_anomalies.assign(abs_z_score=dataset_anomalies["z_score"].abs())
        .sort_values("abs_z_score", ascending=False)
        .drop(columns=["abs_z_score"])
        .head(20)
    )
    st.dataframe(top_anomalies, width="stretch", hide_index=True)

    scatter_df = dataset_anomalies.dropna(subset=["row_index", "value"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.scatterplot(data=scatter_df, x="row_index", y="value", hue="column", ax=ax, s=58)
    ax.set_title("Anomaly Scatter Plot", fontsize=13)
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Value")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



def render_statistical_profiles(
    selected_dataset_keys: list[str],
    user_profile: dict | None,
    transaction_profile: dict | None,
    anomalies_df: pd.DataFrame | None,
) -> None:
    """Render statistical profile views for the selected datasets."""

    st.header("Statistical Profiles")
    profile_map = {
        "users": user_profile,
        "transactions": transaction_profile,
    }

    if len(selected_dataset_keys) > 1:
        dataset_tabs = st.tabs([DATASET_LABELS[key] for key in selected_dataset_keys])
        for dataset_tab, dataset_key in zip(dataset_tabs, selected_dataset_keys):
            with dataset_tab:
                profile_data = profile_map.get(dataset_key)
                if profile_data is None:
                    st.warning(f"Missing profile data for {DATASET_LABELS[dataset_key]}.")
                    continue
                numeric_tab, categorical_tab, anomalies_tab = st.tabs(
                    ["Numeric Columns", "Categorical Columns", "Anomalies"]
                )
                with numeric_tab:
                    render_numeric_tab(profile_data)
                with categorical_tab:
                    render_categorical_tab(profile_data)
                with anomalies_tab:
                    render_anomalies_tab(dataset_key, anomalies_df)
    else:
        dataset_key = selected_dataset_keys[0]
        profile_data = profile_map.get(dataset_key)
        st.subheader(f"{DATASET_LABELS[dataset_key]} Dataset")
        if profile_data is None:
            st.warning(f"Missing profile data for {DATASET_LABELS[dataset_key]}.")
            return
        numeric_tab, categorical_tab, anomalies_tab = st.tabs(
            ["Numeric Columns", "Categorical Columns", "Anomalies"]
        )
        with numeric_tab:
            render_numeric_tab(profile_data)
        with categorical_tab:
            render_categorical_tab(profile_data)
        with anomalies_tab:
            render_anomalies_tab(dataset_key, anomalies_df)



def render_quality_trends(run_history_df: pd.DataFrame | None) -> None:
    """Render historical quality score trends and recent pipeline runs."""

    st.header("Quality Trends")

    if run_history_df is None:
        return

    recent_runs = run_history_df.copy().sort_values("run_id", ascending=False).head(10)

    if len(run_history_df) < 2:
        st.info("Run pipeline multiple times to see trends")
    else:
        trend_df = run_history_df.copy()
        trend_df["run_timestamp"] = pd.to_datetime(trend_df["run_timestamp"], errors="coerce")
        trend_df["dataset_name"] = trend_df["dataset_name"].str.title()
        trend_df = trend_df.sort_values(["dataset_name", "run_id"])

        summary_df = (
            trend_df.groupby("dataset_name", as_index=False)
            .agg(run_count=("run_id", "count"), latest_score=("overall_score", "last"))
        )
        summary_columns = st.columns(max(len(summary_df), 1))
        for summary_column, (_, summary_row) in zip(summary_columns, summary_df.iterrows()):
            with summary_column:
                st.metric(
                    f"{summary_row['dataset_name']} Runs",
                    int(summary_row["run_count"]),
                    f"Latest: {summary_row['latest_score']:.2f}%",
                )

        fig, ax = plt.subplots(figsize=(8.6, 3.9))
        palette = {"Users": "#5b7cfa", "Transactions": "#d38a44"}
        for dataset_name in ["Users", "Transactions"]:
            dataset_trend = trend_df[trend_df["dataset_name"] == dataset_name]
            if dataset_trend.empty:
                continue
            ax.plot(
                dataset_trend["run_id"],
                dataset_trend["overall_score"],
                marker="o",
                linewidth=2.2,
                markersize=6,
                label=dataset_name,
                color=palette[dataset_name],
            )
            for _, row in dataset_trend.iterrows():
                ax.annotate(
                    f"{row['overall_score']:.2f}",
                    (row["run_id"], row["overall_score"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=palette[dataset_name],
                )

        ax.set_title("Overall Score Trends", fontsize=13)
        ax.set_xlabel("Pipeline Run ID")
        ax.set_ylabel("Overall Score")
        ax.set_xticks(sorted(trend_df["run_id"].unique()))
        ax.grid(linestyle="--", alpha=0.35)
        ax.legend(title="Dataset", frameon=True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "This project uses a fixed random seed (42), so trend lines remain flat unless the generation or validation logic changes."
        )

    st.dataframe(recent_runs, width="stretch", hide_index=True)



def render_recommendations(selected_dataset_keys: list[str], quality_scores: dict | None) -> None:
    """Render recommendations with severity-specific Streamlit alerts."""

    st.header("Recommendations")

    if quality_scores is None:
        return

    for dataset_key in selected_dataset_keys:
        dataset_scores = quality_scores.get(dataset_key)
        if dataset_scores is None:
            st.warning(f"Missing recommendations for {DATASET_LABELS[dataset_key]}.")
            continue

        st.subheader(f"{DATASET_LABELS[dataset_key]} Dataset")
        recommendations = dataset_scores.get("recommendations", [])
        if not recommendations:
            st.info("No recommendations for this dataset.")
            continue

        for recommendation in recommendations:
            if recommendation.startswith("Critical:"):
                st.error(recommendation)
            elif recommendation.startswith("Warning:"):
                st.warning(recommendation)
            else:
                st.info(recommendation)



def build_validation_errors_table(
    selected_dataset_keys: list[str],
    user_validation_report: pd.DataFrame | None,
    transaction_validation_report: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Build a combined validation errors table filtered by dataset selection."""

    report_frames = []

    if user_validation_report is not None and "users" in selected_dataset_keys:
        user_errors = user_validation_report.copy()
        user_errors.insert(0, "dataset", "users")
        report_frames.append(user_errors)

    if transaction_validation_report is not None and "transactions" in selected_dataset_keys:
        transaction_errors = transaction_validation_report.copy()
        transaction_errors.insert(0, "dataset", "transactions")
        report_frames.append(transaction_errors)

    if not report_frames:
        return None

    return pd.concat(report_frames, ignore_index=True)



def render_raw_data_explorer(
    selected_dataset_keys: list[str],
    users_df: pd.DataFrame | None,
    transactions_df: pd.DataFrame | None,
    user_validation_report: pd.DataFrame | None,
    transaction_validation_report: pd.DataFrame | None,
) -> None:
    """Render a collapsed raw data explorer with source data and validation errors."""

    with st.expander("Raw Data Explorer", expanded=False):
        users_tab, transactions_tab, validation_tab = st.tabs(
            ["Users Data", "Transactions Data", "Validation Errors"]
        )

        with users_tab:
            if users_df is None:
                st.warning("Missing file: data/synthetic_users.csv")
            else:
                st.dataframe(users_df.head(100), width="stretch", hide_index=True)

        with transactions_tab:
            if transactions_df is None:
                st.warning("Missing file: data/synthetic_transactions.csv")
            else:
                st.dataframe(transactions_df.head(100), width="stretch", hide_index=True)

        with validation_tab:
            validation_errors_df = build_validation_errors_table(
                selected_dataset_keys,
                user_validation_report,
                transaction_validation_report,
            )
            if validation_errors_df is None or validation_errors_df.empty:
                st.warning("No validation errors available to display.")
            else:
                st.dataframe(validation_errors_df, width="stretch", hide_index=True)



def main() -> None:
    """Render the synthetic data quality validation dashboard."""

    if "last_refreshed" not in st.session_state:
        st.session_state["last_refreshed"] = pd.Timestamp.now()

    st.title("\U0001F50D Synthetic Data Quality Validator")
    st.caption(
        "Automated quality scoring, anomaly detection, and statistical profiling for synthetic datasets"
    )

    st.sidebar.title("Dataset Controls")
    selected_option = st.sidebar.selectbox("Select Dataset", ["Users", "Transactions", "Both"])

    if st.sidebar.button("\U0001F504 Refresh Data"):
        st.session_state["last_refreshed"] = pd.Timestamp.now()

    st.sidebar.write(
        "Last refreshed:",
        st.session_state["last_refreshed"].strftime("%Y-%m-%d %H:%M:%S"),
    )

    selected_dataset_keys = get_selected_dataset_keys(selected_option)

    quality_scores = load_json_file(QUALITY_SCORES_PATH)
    user_profile = load_json_file(USER_PROFILE_PATH)
    transaction_profile = load_json_file(TRANSACTION_PROFILE_PATH)
    user_validation_report = load_csv_file(USER_VALIDATION_REPORT_PATH, keep_default_na=False)
    transaction_validation_report = load_csv_file(TRANSACTION_VALIDATION_REPORT_PATH, keep_default_na=False)
    anomalies_df = load_csv_file(ANOMALIES_PATH)
    users_df = load_csv_file(USERS_CSV_PATH)
    transactions_df = load_csv_file(TRANSACTIONS_CSV_PATH)
    run_history_df = load_run_history_data()
    _ = load_json_file(VALIDATION_SUMMARY_PATH)

    render_quality_scorecard(selected_dataset_keys, quality_scores)

    report_map = {
        "users": user_validation_report,
        "transactions": transaction_validation_report,
    }
    render_validation_section(selected_dataset_keys, report_map)

    render_statistical_profiles(
        selected_dataset_keys,
        user_profile,
        transaction_profile,
        anomalies_df,
    )
    render_quality_trends(run_history_df)
    render_recommendations(selected_dataset_keys, quality_scores)
    render_raw_data_explorer(
        selected_dataset_keys,
        users_df,
        transactions_df,
        user_validation_report,
        transaction_validation_report,
    )


if __name__ == "__main__":
    main()



