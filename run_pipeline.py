"""Master runner for the synthetic data quality validation pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

from src import db_logger, generate_synthetic_data, quality_scorer, schema_validator, statistical_profiler


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
QUALITY_SCORES_PATH = DATA_DIR / "quality_scores.json"
VALIDATION_SUMMARY_PATH = DATA_DIR / "validation_summary.json"
START_MESSAGE = "\U0001F680 Starting Synthetic Data Quality Validator..."
SUCCESS_PREFIX = "\u2705"
FAILURE_PREFIX = "\u274C"
END_MESSAGE = "\U0001F389 Pipeline complete! Run: streamlit run dashboard/app.py"



def configure_console_output() -> None:
    """Configure stdout and stderr for UTF-8 so status emojis render correctly."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")



def load_json_file(file_path: Path) -> dict:
    """Load a JSON file from disk and return the parsed payload."""

    with file_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)



def run_step(step_number: int, module_name: str, action: Callable[[], None]) -> None:
    """Execute one pipeline step, printing success or error output without stopping the pipeline."""

    try:
        action()
        print(f"{SUCCESS_PREFIX} Step {step_number}/5 complete: {module_name}")
    except Exception as exc:
        print(f"{FAILURE_PREFIX} Step {step_number}/5 failed: {module_name} -> {exc}")



def log_pipeline_runs() -> None:
    """Initialize the database and log the latest user and transaction run summaries."""

    db_logger.init_db()
    quality_scores = load_json_file(QUALITY_SCORES_PATH)
    validation_summary = load_json_file(VALIDATION_SUMMARY_PATH)

    db_logger.log_run("users", validation_summary["users"], quality_scores["users"])
    db_logger.log_run("transactions", validation_summary["transactions"], quality_scores["transactions"])



def main() -> None:
    """Run the full synthetic data quality pipeline from generation through logging."""

    configure_console_output()
    print(START_MESSAGE)

    steps = [
        (1, "generate_synthetic_data", generate_synthetic_data.main),
        (2, "schema_validator", schema_validator.main),
        (3, "statistical_profiler", statistical_profiler.main),
        (4, "quality_scorer", quality_scorer.main),
        (5, "db_logger", log_pipeline_runs),
    ]

    for step_number, module_name, action in steps:
        run_step(step_number, module_name, action)

    print(END_MESSAGE)


if __name__ == "__main__":
    main()
