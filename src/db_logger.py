"""Log pipeline quality results to a SQLite database using SQLAlchemy."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Float, ForeignKey, Integer, String, create_engine, desc, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_CONNECTION = "sqlite:///data/quality_logs.db"

load_dotenv(PROJECT_ROOT / ".env")

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models in this module."""


class PipelineRun(Base):
    """ORM model for top-level pipeline quality run records."""

    __tablename__ = "pipeline_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_timestamp: Mapped[str] = mapped_column(String)
    dataset_name: Mapped[str] = mapped_column(String)
    total_rows: Mapped[int] = mapped_column(Integer)
    passed_rows: Mapped[int] = mapped_column(Integer)
    failed_rows: Mapped[int] = mapped_column(Integer)
    null_violations: Mapped[int] = mapped_column(Integer)
    range_violations: Mapped[int] = mapped_column(Integer)
    format_violations: Mapped[int] = mapped_column(Integer)
    duplicate_count: Mapped[int] = mapped_column(Integer)
    anomaly_count: Mapped[int] = mapped_column(Integer)
    completeness_score: Mapped[float] = mapped_column(Float)
    validity_score: Mapped[float] = mapped_column(Float)
    consistency_score: Mapped[float] = mapped_column(Float)
    accuracy_score: Mapped[float] = mapped_column(Float)
    overall_score: Mapped[float] = mapped_column(Float)
    quality_grade: Mapped[str] = mapped_column(String)

    column_stats: Mapped[list["ColumnStat"]] = relationship(back_populates="pipeline_run")


class ColumnStat(Base):
    """ORM model for per-column statistics linked to a pipeline run."""

    __tablename__ = "column_stats"

    stat_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("pipeline_runs.run_id"))
    dataset_name: Mapped[str] = mapped_column(String)
    column_name: Mapped[str] = mapped_column(String)
    null_count: Mapped[int] = mapped_column(Integer)
    null_percentage: Mapped[float] = mapped_column(Float)
    anomaly_count: Mapped[int] = mapped_column(Integer)
    mean_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    pipeline_run: Mapped[PipelineRun] = relationship(back_populates="column_stats")



def get_database_url() -> str:
    """Resolve the configured database connection string from .env with a default fallback."""

    raw_url = os.getenv("DB_CONNECTION", DEFAULT_DB_CONNECTION)
    parsed_url = make_url(raw_url)

    if parsed_url.drivername == "sqlite" and parsed_url.database and parsed_url.database != ":memory:":
        database_path = Path(parsed_url.database)
        if not database_path.is_absolute():
            absolute_path = (PROJECT_ROOT / database_path).resolve()
            return f"sqlite:///{absolute_path.as_posix()}"

    return raw_url



def ensure_database_directory(database_url: str) -> None:
    """Create the parent directory for SQLite database files when needed."""

    parsed_url = make_url(database_url)

    if parsed_url.drivername == "sqlite" and parsed_url.database and parsed_url.database != ":memory:":
        Path(parsed_url.database).resolve().parent.mkdir(parents=True, exist_ok=True)



def get_engine() -> Engine:
    """Return a lazily created SQLAlchemy engine for the configured database."""

    global _ENGINE

    if _ENGINE is None:
        database_url = get_database_url()
        ensure_database_directory(database_url)
        _ENGINE = create_engine(database_url, future=True)

    return _ENGINE



def get_session_factory() -> sessionmaker[Session]:
    """Return a lazily created SQLAlchemy session factory."""

    global _SESSION_FACTORY

    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = sessionmaker(bind=get_engine(), future=True)

    return _SESSION_FACTORY



def get_data_directory() -> Path:
    """Return the project's data directory path."""

    return PROJECT_ROOT / "data"



def load_anomalies() -> pd.DataFrame:
    """Load anomaly records from disk or return an empty DataFrame when absent."""

    anomalies_path = get_data_directory() / "anomalies.csv"

    if not anomalies_path.exists():
        return pd.DataFrame(columns=["dataset", "column", "row_index", "value", "z_score", "iqr_flag", "z_flag"])

    return pd.read_csv(anomalies_path)



def calculate_dataset_anomaly_count(dataset_name: str) -> int:
    """Calculate the total number of anomalies for a dataset from anomalies.csv."""

    anomalies_df = load_anomalies()
    return int((anomalies_df["dataset"] == dataset_name).sum()) if not anomalies_df.empty else 0



def calculate_column_anomaly_counts(dataset_name: str) -> dict[str, int]:
    """Calculate anomaly counts per column for a dataset from anomalies.csv."""

    anomalies_df = load_anomalies()

    if anomalies_df.empty:
        return {}

    dataset_anomalies = anomalies_df[anomalies_df["dataset"] == dataset_name]
    if dataset_anomalies.empty:
        return {}

    return dataset_anomalies.groupby("column").size().astype(int).to_dict()



def init_db() -> None:
    """Create all database tables if they do not already exist."""

    Base.metadata.create_all(get_engine())



def log_run(dataset_name: str, validation_summary: dict[str, Any], quality_scores: dict[str, Any]) -> int:
    """Insert a pipeline run summary row and return the created run ID."""

    anomaly_count = calculate_dataset_anomaly_count(dataset_name)
    issues_by_type = validation_summary["issues_by_type"]
    session_factory = get_session_factory()

    pipeline_run = PipelineRun(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name=dataset_name,
        total_rows=int(validation_summary["total_rows"]),
        passed_rows=int(validation_summary["passed_rows"]),
        failed_rows=int(validation_summary["failed_rows"]),
        null_violations=int(issues_by_type["null"]),
        range_violations=int(issues_by_type["range"]),
        format_violations=int(issues_by_type["format"]),
        duplicate_count=int(issues_by_type["duplicate"]),
        anomaly_count=anomaly_count,
        completeness_score=float(quality_scores["completeness"]),
        validity_score=float(quality_scores["validity"]),
        consistency_score=float(quality_scores["consistency"]),
        accuracy_score=float(quality_scores["accuracy"]),
        overall_score=float(quality_scores["overall_score"]),
        quality_grade=str(quality_scores["grade"]),
    )

    with session_factory() as session:
        session.add(pipeline_run)
        session.commit()
        session.refresh(pipeline_run)
        return int(pipeline_run.run_id)



def log_column_stats(run_id: int, dataset_name: str, profile_data: dict[str, Any]) -> None:
    """Insert per-column statistics rows for a pipeline run."""

    session_factory = get_session_factory()
    column_completeness = profile_data.get("column_completeness", {})
    numeric_profiles = profile_data.get("numeric_profiles", {})
    anomaly_counts = calculate_column_anomaly_counts(dataset_name)

    column_stats = []
    for column_name, completeness_metrics in column_completeness.items():
        numeric_profile = numeric_profiles.get(column_name, {})
        column_stats.append(
            ColumnStat(
                run_id=run_id,
                dataset_name=dataset_name,
                column_name=column_name,
                null_count=int(completeness_metrics.get("null_count", 0)),
                null_percentage=float(completeness_metrics.get("null_percentage", 0.0)),
                anomaly_count=int(anomaly_counts.get(column_name, 0)),
                mean_value=(
                    float(numeric_profile["mean"])
                    if numeric_profile.get("mean") is not None
                    else None
                ),
                std_value=(
                    float(numeric_profile["std"])
                    if numeric_profile.get("std") is not None
                    else None
                ),
            )
        )

    if not column_stats:
        return

    with session_factory() as session:
        session.add_all(column_stats)
        session.commit()



def get_run_history() -> pd.DataFrame:
    """Return the full pipeline run history as a pandas DataFrame."""

    query = select(PipelineRun).order_by(desc(PipelineRun.run_id))
    return pd.read_sql(query, get_engine())



def get_score_trends(dataset_name: str) -> pd.DataFrame:
    """Return run timestamp and overall score history for one dataset."""

    query = (
        select(PipelineRun.run_timestamp, PipelineRun.overall_score)
        .where(PipelineRun.dataset_name == dataset_name)
        .order_by(PipelineRun.run_id)
    )
    return pd.read_sql(query, get_engine())



def get_latest_run(dataset_name: str) -> dict[str, Any] | None:
    """Return the most recent run record for a dataset as a dictionary."""

    session_factory = get_session_factory()

    with session_factory() as session:
        latest_run = session.execute(
            select(PipelineRun)
            .where(PipelineRun.dataset_name == dataset_name)
            .order_by(desc(PipelineRun.run_id))
            .limit(1)
        ).scalar_one_or_none()

        if latest_run is None:
            return None

        return {
            "run_id": latest_run.run_id,
            "run_timestamp": latest_run.run_timestamp,
            "dataset_name": latest_run.dataset_name,
            "total_rows": latest_run.total_rows,
            "passed_rows": latest_run.passed_rows,
            "failed_rows": latest_run.failed_rows,
            "null_violations": latest_run.null_violations,
            "range_violations": latest_run.range_violations,
            "format_violations": latest_run.format_violations,
            "duplicate_count": latest_run.duplicate_count,
            "anomaly_count": latest_run.anomaly_count,
            "completeness_score": latest_run.completeness_score,
            "validity_score": latest_run.validity_score,
            "consistency_score": latest_run.consistency_score,
            "accuracy_score": latest_run.accuracy_score,
            "overall_score": latest_run.overall_score,
            "quality_grade": latest_run.quality_grade,
        }



def main() -> None:
    """Initialize the database, show prior run history if available, and confirm readiness."""

    init_db()
    run_history = get_run_history()

    if not run_history.empty:
        print(run_history.to_string(index=False))

    print("Database ready")


if __name__ == "__main__":
    main()

