#!/usr/bin/env python3
"""
Build a pipeline-ready Racket dataset by merging:
- translated Racket-target tasks
- generated passing ground-truth solutions

Input:
    data/racket_from_humaneval.jsonl
    artifacts/racket_ground_truth_dataset.jsonl

Output:
    data/racket_pipeline_ready.jsonl

The result is designed to be consumed by the existing general pipeline.py
through source: jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TASKS = Path("data/racket_from_humaneval.jsonl")
DEFAULT_GROUND_TRUTH = Path("artifacts/racket_ground_truth_dataset.jsonl")
DEFAULT_OUTPUT = Path("data/racket_pipeline_ready.jsonl")
DEFAULT_MISSING = Path("artifacts/racket_pipeline_missing_gt.json")


class BuildError(Exception):
    """Raised when pipeline-ready dataset construction cannot proceed."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    if not path.exists():
        raise BuildError(f"Input file does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise BuildError(
                    f"Line {line_number} of {path} must be a JSON object"
                )

            rows.append(row)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    """Write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_task_row(row: dict[str, Any], index: int) -> None:
    """Validate a translated task row."""
    required_fields = [
        "task_id",
        "language",
        "task_type",
        "entry_point",
        "prompt",
        "question",
        "python_prompt",
        "python_tests",
        "racket_test_module",
        "tests",
    ]
    for field_name in required_fields:
        if field_name not in row:
            raise BuildError(
                f"Task row at index {index} is missing required field '{field_name}'"
            )

    if row["language"] != "racket":
        raise BuildError(
            f"Task '{row.get('task_id', f'index_{index}')}' has language='{row['language']}', expected 'racket'"
        )

    tests = row["tests"]
    if not isinstance(tests, dict):
        raise BuildError(
            f"Task '{row['task_id']}' field 'tests' must be an object"
        )

    for required_test_field in ("kind", "command"):
        if required_test_field not in tests:
            raise BuildError(
                f"Task '{row['task_id']}' tests object is missing '{required_test_field}'"
            )


def validate_gt_row(row: dict[str, Any], index: int) -> None:
    """Validate a GT row."""
    required_fields = [
        "task_id",
        "language",
        "entry_point",
        "ground_truth",
        "execution_status",
    ]
    for field_name in required_fields:
        if field_name not in row:
            raise BuildError(
                f"GT row at index {index} is missing required field '{field_name}'"
            )

    if row["language"] != "racket":
        raise BuildError(
            f"GT row '{row.get('task_id', f'index_{index}')}' has language='{row['language']}', expected 'racket'"
        )

    if row["execution_status"] != "passed":
        raise BuildError(
            f"GT row '{row['task_id']}' has execution_status='{row['execution_status']}', expected 'passed'"
        )


def build_pipeline_row(
    task_row: dict[str, Any],
    gt_row: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge translated task metadata with the discovered ground-truth solution.

    Output schema is aligned with the generic jsonl dataset loader expected by
    the general pipeline.
    """
    ground_truth = gt_row["ground_truth"]

    merged_meta = dict(task_row.get("meta", {}))
    merged_meta.update(
        {
            "source_dataset": "humaneval_python_to_racket",
            "entry_point": task_row["entry_point"],
            "language": "racket",
            "source_language": "python",
            "ground_truth_attempts_used": gt_row.get("num_attempts_used"),
            "ground_truth_model": gt_row.get("meta", {}).get("model_name_or_path"),
            "ground_truth_status": gt_row.get("execution_status"),
        }
    )

    return {
        "task_id": task_row["task_id"],
        "language": "racket",
        "task_type": "cg",
        "entry_point": task_row["entry_point"],
        "prompt": task_row["prompt"],
        "question": task_row["question"],
        "answer": ground_truth,
        "canonical_solutions": [ground_truth],
        "tests": task_row["tests"],
        "python_prompt": task_row["python_prompt"],
        "python_tests": task_row["python_tests"],
        "racket_test_module": task_row["racket_test_module"],
        "meta": merged_meta,
    }


def build_dataset(
    task_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build the pipeline-ready dataset and a report of tasks missing GT.
    """
    gt_by_task_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(gt_rows):
        validate_gt_row(row, index)
        gt_by_task_id[row["task_id"]] = row

    built_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for index, task_row in enumerate(task_rows):
        validate_task_row(task_row, index)
        task_id = task_row["task_id"]

        gt_row = gt_by_task_id.get(task_id)
        if gt_row is None:
            missing_rows.append(
                {
                    "task_id": task_id,
                    "entry_point": task_row["entry_point"],
                    "reason": "missing ground truth",
                }
            )
            continue

        built_rows.append(build_pipeline_row(task_row, gt_row))

    return built_rows, missing_rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a pipeline-ready Racket dataset from translated tasks and GT"
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_TASKS,
        help=f"Path to translated Racket task JSONL (default: {DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help=f"Path to GT JSONL (default: {DEFAULT_GROUND_TRUTH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output pipeline-ready JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=DEFAULT_MISSING,
        help=f"Output missing-GT report JSON (default: {DEFAULT_MISSING})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_rows = read_jsonl(args.tasks)
    gt_rows = read_jsonl(args.ground_truth)

    built_rows, missing_rows = build_dataset(task_rows, gt_rows)

    write_jsonl(args.output, built_rows)
    write_json(args.missing_report, missing_rows)

    print(f"Built {len(built_rows)} pipeline-ready rows")
    print(f"Missing GT for {len(missing_rows)} task(s)")
    print(f"Pipeline-ready dataset written to: {args.output}")
    print(f"Missing-GT report written to: {args.missing_report}")


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc