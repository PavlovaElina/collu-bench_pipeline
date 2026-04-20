#!/usr/bin/env python3
"""
Bootstrap the full Racket benchmark workflow.

This script orchestrates:
1. Build deterministic Racket-target tasks from HumanEval
2. Generate ground-truth Racket solutions
3. Build a pipeline-ready Racket dataset
4. Optionally run the main evaluation pipeline

Example:
    python scripts/bootstrap_racket_benchmark.py ^
      --humaneval-input configs/data/HumanEval.jsonl ^
      --hybrid-output configs/data/racket_from_humaneval_hybrid.jsonl ^
      --gt-output artifacts/racket_ground_truth_dataset_hybrid.jsonl ^
      --pipeline-output configs/data/racket_pipeline_ready_hybrid.jsonl ^
      --model deepseek-ai/deepseek-coder-6.7b-instruct ^
      --device cuda ^
      --dtype float16 ^
      --gt-attempts 15 ^
      --gt-temperature 0.4 ^
      --gt-top-p 0.95 ^
      --gt-max-new-tokens 384

This is the recommended entry point when reproducing the complete Racket
benchmark from raw HumanEval data, because it preserves the intended ordering of
translation, native validation, dataset assembly, and optional evaluation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class BootstrapError(Exception):
    """Raised when one of the bootstrap steps fails."""


def run_command(command: list[str], *, cwd: Path) -> None:
    """Run one pipeline stage and fail fast if the stage exits unsuccessfully."""
    print("\n[BOOTSTRAP] Running:")
    print(" ".join(command))
    result = subprocess.run(command, cwd=str(cwd))
    if result.returncode != 0:
        raise BootstrapError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the Racket benchmark workflow from HumanEval"
    )

    parser.add_argument(
        "--humaneval-input",
        type=Path,
        default=Path("configs/data/HumanEval.jsonl"),
        help="Path to official HumanEval JSONL",
    )

    parser.add_argument(
        "--hybrid-output",
        type=Path,
        default=Path("configs/data/racket_from_humaneval_hybrid.jsonl"),
        help="Output path for deterministic/hybrid Racket task dataset",
    )
    parser.add_argument(
        "--hybrid-failures",
        type=Path,
        default=Path("artifacts/racket_from_humaneval_hybrid_failures.json"),
        help="Failure report for deterministic/hybrid task build",
    )

    parser.add_argument(
        "--gt-output",
        type=Path,
        default=Path("artifacts/racket_ground_truth_dataset_hybrid.jsonl"),
        help="Output GT dataset JSONL",
    )
    parser.add_argument(
        "--gt-failures",
        type=Path,
        default=Path("artifacts/racket_ground_truth_failures_hybrid.json"),
        help="Failure report for GT generation",
    )

    parser.add_argument(
        "--pipeline-output",
        type=Path,
        default=Path("configs/data/racket_pipeline_ready_hybrid.jsonl"),
        help="Output pipeline-ready dataset JSONL",
    )
    parser.add_argument(
        "--pipeline-missing",
        type=Path,
        default=Path("artifacts/racket_pipeline_missing_gt_hybrid.json"),
        help="Missing-GT report for pipeline-ready builder",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="HF model name or local path for GT generation",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Generation device: cpu or cuda",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Generation dtype: float16, bfloat16, or float32",
    )
    parser.add_argument(
        "--gt-max-new-tokens",
        type=int,
        default=384,
        help="Max new tokens for GT generation",
    )
    parser.add_argument(
        "--gt-temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for GT generation",
    )
    parser.add_argument(
        "--gt-top-p",
        type=float,
        default=0.95,
        help="Top-p for GT generation",
    )
    parser.add_argument(
        "--gt-attempts",
        type=int,
        default=15,
        help="Attempts per task for GT generation",
    )
    parser.add_argument(
        "--gt-timeout",
        type=int,
        default=20,
        help="Execution timeout per GT test run",
    )

    parser.add_argument(
        "--build-limit",
        type=int,
        default=None,
        help="Optional limit for deterministic dataset build",
    )
    parser.add_argument(
        "--gt-limit",
        type=int,
        default=None,
        help="Optional limit for GT generation",
    )

    parser.add_argument(
        "--resume-gt",
        action="store_true",
        help="Use resume-capable GT generation instead of fresh GT generation",
    )

    parser.add_argument(
        "--run-eval-pipeline",
        action="store_true",
        help="After building pipeline-ready dataset, also run pipeline.py",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path("configs/racket_pipeline.yaml"),
        help="Config path for pipeline.py if --run-eval-pipeline is set",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    python_exe = sys.executable

    # Step 1: build deterministic/hybrid dataset
    build_cmd = [
        python_exe,
        "scripts/build_racket_from_humaneval_hybrid.py",
        "--input",
        str(args.humaneval_input),
        "--output",
        str(args.hybrid_output),
        "--failures",
        str(args.hybrid_failures),
    ]
    if args.build_limit is not None:
        build_cmd.extend(["--limit", str(args.build_limit)])

    run_command(build_cmd, cwd=project_root)

    # Step 2: generate GT
    if args.resume_gt:
        gt_script = "scripts/generate_racket_gt_from_dataset_resume.py"
    else:
        gt_script = "scripts/generate_racket_gt_from_dataset.py"

    gt_cmd = [
        python_exe,
        gt_script,
        "--input",
        str(args.hybrid_output),
        "--output",
        str(args.gt_output),
        "--failures",
        str(args.gt_failures),
        "--model",
        str(args.model),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--max-new-tokens",
        str(args.gt_max_new_tokens),
        "--temperature",
        str(args.gt_temperature),
        "--top-p",
        str(args.gt_top_p),
        "--attempts",
        str(args.gt_attempts),
        "--timeout",
        str(args.gt_timeout),
    ]
    if args.gt_limit is not None:
        gt_cmd.extend(["--limit", str(args.gt_limit)])

    run_command(gt_cmd, cwd=project_root)

    # Step 3: build pipeline-ready dataset
    pipeline_ready_cmd = [
        python_exe,
        "scripts/build_pipeline_ready_racket_dataset.py",
        "--tasks",
        str(args.hybrid_output),
        "--ground-truth",
        str(args.gt_output),
        "--output",
        str(args.pipeline_output),
        "--missing-report",
        str(args.pipeline_missing),
    ]
    run_command(pipeline_ready_cmd, cwd=project_root)

    # Step 4: optional evaluation pipeline
    if args.run_eval_pipeline:
        eval_cmd = [
            python_exe,
            "pipeline.py",
            "--config",
            str(args.eval_config),
        ]
        run_command(eval_cmd, cwd=project_root)

    print("\n[BOOTSTRAP] Done.")
    print(f"Deterministic dataset: {args.hybrid_output}")
    print(f"GT dataset: {args.gt_output}")
    print(f"Pipeline-ready dataset: {args.pipeline_output}")


if __name__ == "__main__":
    try:
        main()
    except BootstrapError as exc:
        print(f"[BOOTSTRAP ERROR] {exc}")
        raise SystemExit(1) from exc
