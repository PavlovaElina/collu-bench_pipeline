#!/usr/bin/env python3
"""
Run translated Racket tests for a HumanEval-derived task.

This script is intended to be called as:

    python scripts/run_racket_humaneval_tests.py {code_path} {task_id}

It:
- loads data/racket_from_humaneval.jsonl
- finds the matching task by task_id
- takes its `racket_test_module`
- writes temporary files:
    - solution.rkt
    - runner.rkt
- executes the runner with the local `racket` executable
- exits with code 0 when tests pass

Operationally, this script is the bridge between Python orchestration and the
Racket runtime.  The benchmark pipeline can therefore stay language-agnostic at
the top level while still evaluating candidate Racket modules with native
RackUnit semantics underneath.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PATH = Path("data/racket_from_humaneval.jsonl")


class HarnessError(Exception):
    """Raised when the test harness cannot proceed."""


def ensure_racket_available() -> str:
    """Return path to the racket executable or raise an error."""
    racket_path = shutil.which("racket")
    if racket_path is None:
        raise HarnessError(
            "Racket executable was not found in PATH. "
            "Install Racket and make sure `racket` is available from the command line."
        )
    return racket_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    if not path.exists():
        raise HarnessError(f"Dataset file does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HarnessError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise HarnessError(
                    f"Line {line_number} of {path} must be a JSON object"
                )

            rows.append(row)

    return rows


def find_task_by_id(rows: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
    """Find a dataset record by task_id."""
    for row in rows:
        if row.get("task_id") == task_id:
            return row
    raise HarnessError(f"Task '{task_id}' was not found in the dataset")


def normalize_newlines(text: str) -> str:
    """Normalize all line endings to '\\n'."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def ensure_lang_header(code: str) -> str:
    """Ensure the solution starts with '#lang racket'."""
    stripped = code.lstrip()
    if stripped.startswith("#lang racket"):
        return code
    return "#lang racket\n\n" + code


def ensure_provide_all_defined_out(code: str) -> str:
    """
    Ensure solution.rkt exports all defined names so the runner can require them.

    This is important for Racket because a module does not automatically expose
    its internal bindings to another module.  By injecting
    `(provide (all-defined-out))`, the generated test runner can import the
    candidate function regardless of how the model structured the file.
    """
    normalized = normalize_newlines(code)

    if "(provide (all-defined-out))" in normalized:
        return normalized

    lines = normalized.splitlines()

    if lines and lines[0].strip() == "#lang racket":
        rebuilt = [lines[0], "", "(provide (all-defined-out))"]
        if len(lines) > 1:
            rebuilt.extend(lines[1:])
        result = "\n".join(rebuilt)
    else:
        result = "#lang racket\n\n(provide (all-defined-out))\n" + normalized

    return result


def load_candidate_code(code_path: Path) -> str:
    """Load generated candidate code from disk."""
    if not code_path.exists():
        raise HarnessError(f"Generated code file does not exist: {code_path}")

    code = code_path.read_text(encoding="utf-8")
    code = normalize_newlines(code).strip()

    if not code:
        raise HarnessError("Generated code file is empty")

    code = ensure_lang_header(code)
    code = ensure_provide_all_defined_out(code)

    if not code.endswith("\n"):
        code += "\n"

    return code


def load_racket_test_module(task: dict[str, Any]) -> str:
    """Extract the translated Racket test module from the task record."""
    racket_test_module = task.get("racket_test_module")
    if not isinstance(racket_test_module, str) or not racket_test_module.strip():
        raise HarnessError(
            f"Task '{task.get('task_id', '<unknown>')}' does not contain a valid racket_test_module"
        )

    module = normalize_newlines(racket_test_module).strip()
    if not module.startswith("#lang racket"):
        module = "#lang racket\n\n" + module

    if not module.endswith("\n"):
        module += "\n"

    return module


def run_racket_tests(
    racket_path: str,
    runner_path: Path,
    working_directory: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """
    Execute the generated runner module with the system `racket` binary.

    A direct `racket runner.rkt` invocation is sufficient because `runner.rkt`
    already requires RackUnit, imports `solution.rkt`, and exits with a
    meaningful process status.
    """
    return subprocess.run(
        [racket_path, str(runner_path)],
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=str(working_directory),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run translated Racket/HumanEval tests for a generated solution"
    )
    parser.add_argument("code_path", type=Path, help="Path to generated Racket code")
    parser.add_argument("task_id", type=str, help="Task identifier")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the translated Racket dataset JSONL (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Execution timeout in seconds (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    try:
        racket_path = ensure_racket_available()
        dataset_rows = read_jsonl(args.dataset)
        task = find_task_by_id(dataset_rows, args.task_id)

        candidate_code = load_candidate_code(args.code_path)
        runner_code = load_racket_test_module(task)

        with tempfile.TemporaryDirectory(prefix="racket_humaneval_exec_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            solution_path = tmpdir_path / "solution.rkt"
            runner_path = tmpdir_path / "runner.rkt"

            solution_path.write_text(candidate_code, encoding="utf-8")
            runner_path.write_text(runner_code, encoding="utf-8")

            result = run_racket_tests(
                racket_path=racket_path,
                runner_path=runner_path,
                working_directory=tmpdir_path,
                timeout=args.timeout,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                print(stdout)
            if stderr:
                print(stderr, file=sys.stderr)

            if result.returncode != 0:
                raise SystemExit(result.returncode)

    except subprocess.TimeoutExpired:
        print("Execution timed out", file=sys.stderr)
        raise SystemExit(124)
    except HarnessError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"Unexpected harness error: {exc}", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
