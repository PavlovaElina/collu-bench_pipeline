#!/usr/bin/env python3
"""
Build a deterministic Racket-target dataset from the official HumanEval dataset.

Strategy:
- task prompt is derived deterministically from Python signature + docstring
- Python tests are translated deterministically via Python AST
- a strict rackunit/text-ui runner is generated automatically

Input:
    HumanEval JSONL or JSONL.GZ with fields like:
      - task_id
      - prompt
      - test
      - entry_point

Output:
    configs/data/racket_from_humaneval_hybrid.jsonl
    artifacts/racket_from_humaneval_hybrid_failures.json
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT = Path("configs/data/racket_from_humaneval_hybrid.jsonl")
DEFAULT_FAILURES = Path("artifacts/racket_from_humaneval_hybrid_failures.json")


class BuildError(Exception):
    """Raised when dataset construction cannot proceed."""


@dataclass
class AdaptedTask:
    task_id: str
    language: str
    task_type: str
    entry_point: str
    source_language: str
    prompt: str
    question: str
    python_prompt: str
    python_tests: str
    racket_test_module: str
    canonical_solutions: list[str]
    tests: dict[str, Any]
    meta: dict[str, Any]


@dataclass
class TranslatedTestCase:
    python_source: str
    racket_check: str


def read_jsonl_or_gz(path: Path) -> list[dict[str, Any]]:
    """Read .jsonl or .jsonl.gz."""
    if not path.exists():
        raise BuildError(f"Input file does not exist: {path}")

    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []

    with opener(path, "rt", encoding="utf-8") as f:
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_humaneval_row(row: dict[str, Any], index: int) -> None:
    """Validate minimum HumanEval fields."""
    required_fields = ["task_id", "prompt", "test", "entry_point"]
    for field_name in required_fields:
        if field_name not in row:
            raise BuildError(
                f"HumanEval row at index {index} is missing required field '{field_name}'"
            )

    for field_name in required_fields:
        if not isinstance(row[field_name], str) or not row[field_name].strip():
            raise BuildError(
                f"HumanEval row '{row.get('task_id', f'index_{index}')}' has invalid '{field_name}'"
            )


def extract_prompt_from_python_source(python_prompt: str, entry_point: str) -> str:
    """
    Convert Python prompt source into a natural-language Racket task description.

    We parse the Python function, extract the docstring, remove doctest examples
    together with their output lines, and append a short Racket-specific
    instruction block.
    """
    try:
        tree = ast.parse(python_prompt)
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse python prompt: {exc}") from exc

    fn_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            fn_node = node
            break

    if fn_node is None:
        raise ValueError(f"Could not find function definition for '{entry_point}'")

    docstring = ast.get_docstring(fn_node) or ""
    docstring = docstring.strip()

    if not docstring:
        raise ValueError(f"No docstring found for '{entry_point}'")

    lines = docstring.splitlines()
    cleaned_lines: list[str] = []

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Skip doctest input line
        if stripped.startswith(">>>"):
            i += 1

            # Skip following doctest output lines until:
            # - blank line
            # - another doctest
            # - a normal paragraph-like sentence
            while i < len(lines):
                next_stripped = lines[i].strip()

                if not next_stripped:
                    i += 1
                    break

                if next_stripped.startswith(">>>"):
                    break

                # Heuristic: doctest output lines are usually short values /
                # literals like False, True, 0.5, ['a'], etc.
                # If the line looks like a normal sentence, keep it.
                looks_like_sentence = (
                    next_stripped[:1].isupper()
                    and (" " in next_stripped or next_stripped.endswith("."))
                )
                if looks_like_sentence:
                    break

                i += 1

            continue

        cleaned_lines.append(lines[i])
        i += 1

    cleaned_doc = "\n".join(cleaned_lines).strip()

    # Normalize whitespace while preserving sentence boundaries reasonably.
    cleaned_doc = re.sub(r"\n\s*\n+", "\n\n", cleaned_doc)
    cleaned_doc = re.sub(r"[ \t]+", " ", cleaned_doc)
    cleaned_doc = re.sub(r"\n\s*", " ", cleaned_doc)
    cleaned_doc = re.sub(r"\s+", " ", cleaned_doc).strip()

    if not cleaned_doc:
        raise ValueError(f"Docstring for '{entry_point}' became empty after cleaning")

    cleaned_doc = cleaned_doc.replace("Python", "Racket")
    cleaned_doc = cleaned_doc.replace("python", "Racket")

    final_prompt = (
        f"{cleaned_doc}\n\n"
        f"Implement the solution in Racket.\n"
        f"Use `#lang racket`.\n"
        f"Define the function `{entry_point}`.\n"
        f"When the original task refers to Python lists or tuples, use Racket lists.\n"
        f"Preserve the original task semantics.\n"
        f"Return only valid Racket code."
    )

    return final_prompt

def translate_humaneval_tests_to_racket(
    python_tests: str,
    entry_point: str,
) -> list[TranslatedTestCase]:
    """
    Deterministically translate Python HumanEval tests into Racket checks.
    """
    try:
        tree = ast.parse(python_tests)
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse Python tests: {exc}") from exc

    translated_cases: list[TranslatedTestCase] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            for stmt in node.body:
                if isinstance(stmt, ast.Assert):
                    translated_cases.append(_translate_assert_node(stmt, entry_point))
        elif isinstance(node, ast.Assert):
            translated_cases.append(_translate_assert_node(node, entry_point))

    if not translated_cases:
        raise ValueError("No translatable assert cases found")

    return translated_cases


def _translate_assert_node(node: ast.Assert, entry_point: str) -> TranslatedTestCase:
    source = ast.unparse(node.test)
    expr = node.test

    if isinstance(expr, ast.Compare):
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            raise ValueError(f"Unsupported multi-part comparison: {source}")

        left = _translate_python_expr_to_racket(expr.left, entry_point)
        right = _translate_python_expr_to_racket(expr.comparators[0], entry_point)
        op = expr.ops[0]

        if isinstance(op, ast.Eq):
            racket_check = f"(check-equal? {left} {right})"
        elif isinstance(op, ast.NotEq):
            racket_check = f"(check-true (not (equal? {left} {right})))"
        elif isinstance(op, ast.Lt):
            racket_check = f"(check-true (< {left} {right}))"
        elif isinstance(op, ast.LtE):
            racket_check = f"(check-true (<= {left} {right}))"
        elif isinstance(op, ast.Gt):
            racket_check = f"(check-true (> {left} {right}))"
        elif isinstance(op, ast.GtE):
            racket_check = f"(check-true (>= {left} {right}))"
        else:
            raise ValueError(f"Unsupported comparison operator: {source}")

        return TranslatedTestCase(
            python_source=source,
            racket_check=racket_check,
        )

    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        operand = _translate_python_expr_to_racket(expr.operand, entry_point)
        return TranslatedTestCase(
            python_source=source,
            racket_check=f"(check-false {operand})",
        )

    translated_expr = _translate_python_expr_to_racket(expr, entry_point)
    return TranslatedTestCase(
        python_source=source,
        racket_check=f"(check-true {translated_expr})",
    )


def _translate_python_expr_to_racket(expr: ast.AST, entry_point: str) -> str:
    if isinstance(expr, ast.Call):
        return _translate_call(expr, entry_point)

    if isinstance(expr, ast.Constant):
        return _translate_constant(expr.value)

    if isinstance(expr, ast.List):
        items = " ".join(_translate_python_expr_to_racket(elt, entry_point) for elt in expr.elts)
        return f"'({items})"

    if isinstance(expr, ast.Tuple):
        items = " ".join(_translate_python_expr_to_racket(elt, entry_point) for elt in expr.elts)
        return f"'({items})"

    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        operand = _translate_python_expr_to_racket(expr.operand, entry_point)
        if re.fullmatch(r"\d+(\.\d+)?", operand):
            return f"-{operand}"
        return f"(- {operand})"

    if isinstance(expr, ast.BinOp):
        left = _translate_python_expr_to_racket(expr.left, entry_point)
        right = _translate_python_expr_to_racket(expr.right, entry_point)

        if isinstance(expr.op, ast.Add):
            return f"(+ {left} {right})"
        if isinstance(expr.op, ast.Sub):
            return f"(- {left} {right})"
        if isinstance(expr.op, ast.Mult):
            return f"(* {left} {right})"
        if isinstance(expr.op, ast.Div):
            return f"(/ {left} {right})"
        if isinstance(expr.op, ast.Mod):
            return f"(modulo {left} {right})"

        raise ValueError(f"Unsupported binary operator: {ast.dump(expr)}")

    if isinstance(expr, ast.Name):
        if expr.id == "True":
            return "#t"
        if expr.id == "False":
            return "#f"
        if expr.id == "None":
            return "'()"
        raise ValueError(f"Unsupported bare name in tests: {expr.id}")

    raise ValueError(f"Unsupported Python expression in tests: {ast.dump(expr)}")


def _translate_call(call: ast.Call, entry_point: str) -> str:
    if isinstance(call.func, ast.Name) and call.func.id == "candidate":
        translated_args = [_translate_python_expr_to_racket(arg, entry_point) for arg in call.args]
        joined_args = " ".join(translated_args)
        if joined_args:
            return f"({entry_point} {joined_args})"
        return f"({entry_point})"

    if isinstance(call.func, ast.Name):
        name = call.func.id
        translated_args = [_translate_python_expr_to_racket(arg, entry_point) for arg in call.args]

        builtin_map = {
            "abs": "abs",
            "len": "length",
            "max": "max",
            "min": "min",
        }

        if name in builtin_map:
            target = builtin_map[name]
            joined_args = " ".join(translated_args)
            return f"({target} {joined_args})"

        if name == "sum":
            if len(translated_args) != 1:
                raise ValueError("sum(...) translation expects one argument")
            return f"(apply + {translated_args[0]})"

    raise ValueError(f"Unsupported function call in tests: {ast.dump(call)}")


def _translate_constant(value: object) -> str:
    if value is True:
        return "#t"
    if value is False:
        return "#f"
    if value is None:
        return "'()"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return repr(value)

    raise ValueError(f"Unsupported constant type in tests: {type(value).__name__}")


def build_racket_test_module_from_cases(translated_cases: list[TranslatedTestCase]) -> str:
    """
    Build a strict rackunit/text-ui module.

    This guarantees correct pass/fail exit codes.
    """
    test_case_lines = []
    for idx, case in enumerate(translated_cases, start=1):
        test_case_lines.append(
            f'   (test-case "case-{idx}" {case.racket_check})'
        )

    tests_block = "\n".join(test_case_lines)

    return (
        "#lang racket\n\n"
        "(require rackunit)\n"
        "(require rackunit/text-ui)\n"
        '(require "solution.rkt")\n\n'
        "(define generated-tests\n"
        '  (test-suite\n'
        '   "generated-tests"\n'
        f"{tests_block}\n"
        "   ))\n\n"
        "(define failures (run-tests generated-tests))\n\n"
        "(if (zero? failures)\n"
        '    (begin\n'
        '      (displayln "ALL_TESTS_PASSED")\n'
        "      (exit 0))\n"
        "    (exit 1))\n"
    )


def build_single_task(row: dict[str, Any]) -> AdaptedTask:
    task_id = row["task_id"]
    entry_point = row["entry_point"]
    python_prompt = row["prompt"]
    python_tests = row["test"]

    adapted_prompt = extract_prompt_from_python_source(
        python_prompt=python_prompt,
        entry_point=entry_point,
    )
    translated_cases = translate_humaneval_tests_to_racket(
        python_tests=python_tests,
        entry_point=entry_point,
    )
    racket_test_module = build_racket_test_module_from_cases(translated_cases)

    return AdaptedTask(
        task_id=task_id,
        language="racket",
        task_type="cg",
        entry_point=entry_point,
        source_language="python",
        prompt=adapted_prompt,
        question=adapted_prompt,
        python_prompt=python_prompt,
        python_tests=python_tests,
        racket_test_module=racket_test_module,
        canonical_solutions=[],
        tests={
            "kind": "external_command",
            "command": "python scripts/run_racket_humaneval_tests.py {code_path} {task_id}",
        },
        meta={
            "source_dataset": "humaneval_python",
            "entry_point": entry_point,
            "language": "racket",
            "source_language": "python",
            "prompt_adaptation_mode": "deterministic_docstring",
            "num_translated_tests": len(translated_cases),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Racket-target dataset from all HumanEval tasks"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to official HumanEval .jsonl or .jsonl.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output translated dataset JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=DEFAULT_FAILURES,
        help=f"Output failure report JSON (default: {DEFAULT_FAILURES})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of HumanEval tasks to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    humaneval_rows = read_jsonl_or_gz(args.input)
    for index, row in enumerate(humaneval_rows):
        validate_humaneval_row(row, index)

    if args.limit is not None:
        humaneval_rows = humaneval_rows[: args.limit]

    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for row in humaneval_rows:
        print(f"[DETERMINISTIC] {row['task_id']}")
        try:
            adapted = build_single_task(row)
            success_rows.append(
                {
                    "task_id": adapted.task_id,
                    "language": adapted.language,
                    "task_type": adapted.task_type,
                    "entry_point": adapted.entry_point,
                    "source_language": adapted.source_language,
                    "prompt": adapted.prompt,
                    "question": adapted.question,
                    "python_prompt": adapted.python_prompt,
                    "python_tests": adapted.python_tests,
                    "racket_test_module": adapted.racket_test_module,
                    "canonical_solutions": adapted.canonical_solutions,
                    "tests": adapted.tests,
                    "meta": adapted.meta,
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "task_id": row["task_id"],
                    "entry_point": row["entry_point"],
                    "reason": str(exc),
                }
            )

    write_jsonl(args.output, success_rows)
    write_json(args.failures, failure_rows)

    print(f"Built {len(success_rows)} deterministic Racket tasks")
    print(f"Failed to build {len(failure_rows)} task(s)")
    print(f"Dataset written to: {args.output}")
    print(f"Failure report written to: {args.failures}")


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc