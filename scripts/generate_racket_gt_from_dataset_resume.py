#!/usr/bin/env python3
"""
Resume-capable GT generator for Racket tasks.

This script extends the basic GT builder by:
- reading already completed GT records
- skipping task_ids that already have a ground truth
- generating only for missing tasks
- merging newly found GT solutions with existing ones

Example:
    python scripts/generate_racket_gt_from_dataset_resume.py ^
        --input data/racket_from_humaneval.jsonl ^
        --output artifacts/racket_ground_truth_dataset.jsonl ^
        --failures artifacts/racket_ground_truth_failures.json ^
        --model deepseek-ai/deepseek-coder-6.7b-instruct ^
        --device cuda ^
        --dtype float16 ^
        --max-new-tokens 256 ^
        --temperature 0.4 ^
        --top-p 0.95 ^
        --attempts 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_racket_gt_from_dataset import (
    GTBuildError,
    GroundTruthRecord,
    build_prompt,
    cleanup_model_output,
    generate_single_completion,
    load_model_and_tokenizer,
    read_jsonl,
    run_tests_for_candidate,
    validate_task,
    write_failures,
    write_jsonl,
)


DEFAULT_INPUT = Path("data/racket_from_humaneval.jsonl")
DEFAULT_OUTPUT = Path("artifacts/racket_ground_truth_dataset.jsonl")
DEFAULT_FAILURES = Path("artifacts/racket_ground_truth_failures.json")


def read_existing_gt_records(path: Path) -> list[dict[str, Any]]:
    """Read existing GT JSONL if present, otherwise return an empty list."""
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GTBuildError(
                    f"Invalid JSON in existing GT file on line {line_number} of {path}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise GTBuildError(
                    f"Line {line_number} of existing GT file {path} must be a JSON object"
                )

            rows.append(row)

    return rows


def read_existing_failures(path: Path) -> list[dict[str, Any]]:
    """Read an existing failure report if present."""
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GTBuildError(f"Invalid JSON in failure report {path}: {exc}") from exc

    if not isinstance(data, list):
        raise GTBuildError(f"Failure report {path} must contain a JSON list")

    return data


def gt_record_from_dict(row: dict[str, Any]) -> GroundTruthRecord:
    """Convert an existing GT dict back into GroundTruthRecord."""
    return GroundTruthRecord(
        task_id=row["task_id"],
        language=row["language"],
        entry_point=row["entry_point"],
        prompt=row["prompt"],
        question=row["question"],
        python_prompt=row["python_prompt"],
        python_tests=row["python_tests"],
        racket_test_module=row["racket_test_module"],
        ground_truth=row["ground_truth"],
        num_attempts_used=row["num_attempts_used"],
        execution_status=row["execution_status"],
        meta=row.get("meta", {}),
    )


def try_generate_ground_truth_resume(
    row: dict[str, Any],
    *,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    attempts: int,
    dataset_path: Path,
    timeout: int,
    model_name_or_path: str,
) -> tuple[GroundTruthRecord | None, dict[str, Any]]:
    """
    Try multiple attempts until one generated solution passes tests.
    """
    prompt_text = build_prompt(row["prompt"], row["entry_point"])
    last_failure: dict[str, Any] = {
        "task_id": row["task_id"],
        "entry_point": row["entry_point"],
        "reason": "unknown",
        "attempts": [],
    }

    for attempt_idx in range(1, attempts + 1):
        raw_output, tokens, rendered_prompt = generate_single_completion(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        cleaned_output = cleanup_model_output(
            raw_output=raw_output,
            tokens=tokens,
            entry_point=row["entry_point"],
        )

        passed, stdout, stderr, exit_code = run_tests_for_candidate(
            code=cleaned_output,
            task_id=row["task_id"],
            dataset_path=dataset_path,
            timeout=timeout,
        )

        attempt_info = {
            "attempt": attempt_idx,
            "raw_output_preview": raw_output[:500],
            "cleaned_output_preview": cleaned_output[:500],
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
        }
        last_failure["attempts"].append(attempt_info)

        if passed:
            meta = dict(row.get("meta", {}))
            meta.update(
                {
                    "model_name_or_path": model_name_or_path,
                    "rendered_prompt_preview": rendered_prompt[:1000],
                    "generation_attempt": attempt_idx,
                    "generation_mode": "sample" if temperature > 0.0 else "greedy",
                    "resume_run": True,
                }
            )

            record = GroundTruthRecord(
                task_id=row["task_id"],
                language=row["language"],
                entry_point=row["entry_point"],
                prompt=row["prompt"],
                question=row["question"],
                python_prompt=row["python_prompt"],
                python_tests=row["python_tests"],
                racket_test_module=row["racket_test_module"],
                ground_truth=cleaned_output,
                num_attempts_used=attempt_idx,
                execution_status="passed",
                meta=meta,
            )
            return record, last_failure

    last_failure["reason"] = "no passing solution found"
    return None, last_failure


def merge_failures(
    existing_failures: list[dict[str, Any]],
    new_failures: list[dict[str, Any]],
    completed_task_ids: set[str],
) -> list[dict[str, Any]]:
    """
    Merge failures while removing tasks that are now completed.
    """
    merged: dict[str, dict[str, Any]] = {}

    for item in existing_failures:
        task_id = item.get("task_id")
        if isinstance(task_id, str) and task_id not in completed_task_ids:
            merged[task_id] = item

    for item in new_failures:
        task_id = item.get("task_id")
        if isinstance(task_id, str) and task_id not in completed_task_ids:
            merged[task_id] = item

    return list(merged.values())


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Resume-capable generator of Racket ground-truth solutions"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input translated dataset JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output GT dataset JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=DEFAULT_FAILURES,
        help=f"Output failure report JSON (default: {DEFAULT_FAILURES})",
    )
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Generation device: cpu or cuda (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Model dtype: float16, bfloat16, or float32 (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature; >0 enables multiple diverse attempts",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p for sampling",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=5,
        help="Maximum number of generation attempts per missing task",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Test execution timeout in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of missing tasks to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_rows = read_jsonl(args.input)
    for index, row in enumerate(source_rows):
        validate_task(row, index)

    existing_gt_dicts = read_existing_gt_records(args.output)
    existing_gt_records = [gt_record_from_dict(row) for row in existing_gt_dicts]
    existing_task_ids = {row.task_id for row in existing_gt_records}

    missing_rows = [row for row in source_rows if row["task_id"] not in existing_task_ids]

    if args.limit is not None:
        missing_rows = missing_rows[: args.limit]

    print(f"Existing GT records: {len(existing_gt_records)}")
    print(f"Missing tasks to process: {len(missing_rows)}")

    if not missing_rows:
        print("Nothing to do. All tasks already have GT.")
        return

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype_name=args.dtype,
    )

    new_success_rows: list[GroundTruthRecord] = []
    new_failure_rows: list[dict[str, Any]] = []

    for row in missing_rows:
        print(f"[GT-RESUME] Processing {row['task_id']}")
        record, failure_info = try_generate_ground_truth_resume(
            row,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            attempts=args.attempts,
            dataset_path=args.input,
            timeout=args.timeout,
            model_name_or_path=args.model,
        )

        if record is not None:
            print(
                f"[GT-RESUME] PASS {row['task_id']} in {record.num_attempts_used} attempt(s)"
            )
            new_success_rows.append(record)
        else:
            print(f"[GT-RESUME] FAIL {row['task_id']}")
            new_failure_rows.append(failure_info)

    all_success_rows = existing_gt_records + new_success_rows
    completed_task_ids = {row.task_id for row in all_success_rows}

    existing_failures = read_existing_failures(args.failures)
    merged_failures = merge_failures(
        existing_failures=existing_failures,
        new_failures=new_failure_rows,
        completed_task_ids=completed_task_ids,
    )

    write_jsonl(args.output, all_success_rows)
    write_failures(args.failures, merged_failures)

    print(f"Total GT records after merge: {len(all_success_rows)}")
    print(f"Remaining failures after merge: {len(merged_failures)}")
    print(f"GT dataset: {args.output}")
    print(f"Failure report: {args.failures}")


if __name__ == "__main__":
    try:
        main()
    except GTBuildError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc