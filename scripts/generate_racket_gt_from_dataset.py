#!/usr/bin/env python3
"""
Generate ground-truth Racket solutions for tasks from data/racket_from_humaneval.jsonl.

Pipeline:
- read the translated Racket-target dataset
- prompt a local Hugging Face model to solve each task in Racket
- run translated Racket tests
- accept the first passing solution as ground truth
- write successful tasks to artifacts/racket_ground_truth_dataset.jsonl
- write failures to artifacts/racket_ground_truth_failures.json

Example:
    python scripts/generate_racket_gt_from_dataset.py ^
        --input data/racket_from_humaneval.jsonl ^
        --output artifacts/racket_ground_truth_dataset.jsonl ^
        --model deepseek-ai/deepseek-coder-6.7b-instruct ^
        --device cuda ^
        --dtype float16 ^
        --max-new-tokens 256 ^
        --attempts 5

The script assumes that Racket itself is installed and visible in `PATH`,
because every candidate is validated by invoking
`scripts/run_racket_humaneval_tests.py`, which in turn shells out to the local
`racket` executable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUT = Path("data/racket_from_humaneval.jsonl")
DEFAULT_OUTPUT = Path("artifacts/racket_ground_truth_dataset.jsonl")
DEFAULT_FAILURES = Path("artifacts/racket_ground_truth_failures.json")


class GTBuildError(Exception):
    """Raised when GT generation cannot proceed."""


@dataclass
class GroundTruthRecord:
    task_id: str
    language: str
    entry_point: str
    prompt: str
    question: str
    python_prompt: str
    python_tests: str
    racket_test_module: str
    ground_truth: str
    num_attempts_used: int
    execution_status: str
    meta: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL into a list of dicts."""
    if not path.exists():
        raise GTBuildError(f"Input file does not exist: {path}")

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
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise GTBuildError(
                    f"Line {line_number} of {path} must be a JSON object"
                )

            rows.append(row)

    return rows


def write_jsonl(path: Path, rows: Iterable[GroundTruthRecord]) -> None:
    """Write successful ground-truth rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "task_id": row.task_id,
                        "language": row.language,
                        "entry_point": row.entry_point,
                        "prompt": row.prompt,
                        "question": row.question,
                        "python_prompt": row.python_prompt,
                        "python_tests": row.python_tests,
                        "racket_test_module": row.racket_test_module,
                        "ground_truth": row.ground_truth,
                        "num_attempts_used": row.num_attempts_used,
                        "execution_status": row.execution_status,
                        "meta": row.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_failures(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write failure report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def validate_task(row: dict[str, Any], index: int) -> None:
    """Validate a translated dataset record."""
    required_fields = [
        "task_id",
        "language",
        "entry_point",
        "prompt",
        "question",
        "python_prompt",
        "python_tests",
        "racket_test_module",
    ]
    for field_name in required_fields:
        if field_name not in row:
            raise GTBuildError(
                f"Task at index {index} is missing required field '{field_name}'"
            )

    if row["language"] != "racket":
        raise GTBuildError(
            f"Task '{row.get('task_id', f'index_{index}')}' has language='{row['language']}', expected 'racket'"
        )


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Convert string dtype name to torch dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.strip().lower()
    if key not in mapping:
        raise GTBuildError(
            f"Unsupported dtype '{dtype_name}'. Use one of: {', '.join(sorted(mapping))}"
        )
    return mapping[key]


def ensure_tokenizer_has_pad_token(tokenizer: AutoTokenizer) -> None:
    """Assign a pad token if missing."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device: str,
    dtype_name: str,
):
    """Load HF tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    ensure_tokenizer_has_pad_token(tokenizer)

    torch_dtype = resolve_torch_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device.lower() == "cuda":
        if not torch.cuda.is_available():
            raise GTBuildError("CUDA was requested but torch.cuda.is_available() is False")
        model = model.to("cuda")
    elif device.lower() == "cpu":
        model = model.to("cpu")
    else:
        raise GTBuildError("Only device='cpu' or device='cuda' is supported")

    model.eval()
    return model, tokenizer


def build_prompt(question: str, entry_point: str) -> str:
    """
    Build a strict prompt for generating Racket GT.

    The prompt asks for executable `#lang racket` code rather than a natural
    language explanation, because the downstream validator expects a module that
    can be imported and tested immediately.
    """
    return (
        "You are given a programming task originally written for Python.\n"
        "Implement the required function in Racket.\n"
        "Preserve the intended semantics of the task, not Python syntax.\n"
        "Return only valid executable Racket code.\n"
        "Do not include Markdown, explanations, comments, tests, or example calls.\n\n"
        f"Required function name: {entry_point}\n\n"
        "Task:\n"
        f"{question}\n"
    )


def normalize_visible_token_markers(text: str) -> str:
    """Convert visible tokenizer markers into readable text."""
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    text = text.replace("ĉ", "\n")
    text = text.replace("▁", " ")
    text = text.replace("<0x0A>", "\n")
    return text


def detokenize_fallback_from_token_strings(tokens: list[str]) -> str:
    """Fallback detokenization from tokenizer tokens."""
    if not tokens:
        return ""

    text = "".join(tokens)
    text = normalize_visible_token_markers(text)

    special_tokens_to_remove = [
        "<｜begin▁of▁sentence｜>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<s>",
        "</s>",
        "<pad>",
        "<|EOT|>",
    ]
    for token in special_tokens_to_remove:
        text = text.replace(token, "")

    return text


def isolate_racket_code_region(text: str) -> str:
    """Extract the most likely Racket code region."""
    lang_index = text.find("#lang racket")
    define_index = text.find("(define")

    if lang_index != -1:
        return text[lang_index:]
    if define_index != -1:
        return "#lang racket\n\n" + text[define_index:]
    return text


def remove_known_prose_prefixes(text: str) -> str:
    """Remove assistant-style lead-ins."""
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    patterns = [
        r"^A:\s*Here is.*?$",
        r"^Here is.*?$",
        r"^Sure,.*?$",
        r"^Certainly,.*?$",
    ]

    for line in lines:
        stripped = line.strip()
        matched = False
        for pattern in patterns:
            import re
            if re.match(pattern, stripped, flags=re.IGNORECASE):
                matched = True
                break
        if not matched:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_trailing_example_calls(text: str, entry_point: str) -> str:
    """Remove trailing example invocations after the main definition."""
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"({entry_point} "):
            break
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def truncate_after_last_balanced_form(text: str) -> str:
    """Keep text through the last balanced closing parenthesis."""
    last_balanced_end = -1
    depth = 0
    in_string = False
    escaped = False

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                last_balanced_end = idx + 1

    if last_balanced_end != -1:
        return text[:last_balanced_end]

    return text


def balance_racket_brackets(text: str) -> str:
    """Append missing closing brackets for (), [], {}."""
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {")", "]", "}"}

    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in closing:
            if stack and stack[-1] == ch:
                stack.pop()

    if stack:
        text += "".join(reversed(stack))

    return text


def cleanup_model_output(raw_output: str, tokens: list[str], entry_point: str) -> str:
    """
    Clean raw generation into executable Racket code.

    This recovery step is especially valuable for Racket outputs, where an
    otherwise correct solution may become non-runnable if the model adds one
    stray explanation line or leaves a parenthesis unmatched.
    """
    import re

    decoded = raw_output.replace("\r\n", "\n").replace("\r", "\n").strip()
    decoded = normalize_visible_token_markers(decoded)

    if "Ġ" in decoded or "Ċ" in decoded or "▁" in decoded:
        decoded = detokenize_fallback_from_token_strings(tokens).strip()

    fenced_match = re.search(
        r"```(?:racket|scheme|lisp)?\s*(.*?)```",
        decoded,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_match:
        decoded = fenced_match.group(1).strip()

    decoded = remove_known_prose_prefixes(decoded)
    decoded = isolate_racket_code_region(decoded)
    decoded = remove_trailing_example_calls(decoded, entry_point)
    decoded = truncate_after_last_balanced_form(decoded).strip()
    decoded = balance_racket_brackets(decoded).strip()

    if not decoded.startswith("#lang racket"):
        decoded = "#lang racket\n\n" + decoded

    decoded = decoded.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not decoded.endswith("\n"):
        decoded += "\n"

    return decoded


def encode_prompt(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    *,
    device: str,
) -> tuple[dict[str, torch.Tensor], str]:
    """
    Encode prompt; use chat template if available.
    """
    rendered_prompt = prompt_text

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {
                "role": "system",
                "content": "You are an expert Racket programmer. Return only executable Racket code.",
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    encoded = tokenizer(rendered_prompt, return_tensors="pt")

    if device.lower() == "cuda":
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
    else:
        encoded = {k: v.to("cpu") for k, v in encoded.items()}

    return encoded, rendered_prompt


def generate_single_completion(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, list[str], str]:
    """Generate one completion and return raw text, tokens, rendered prompt."""
    encoded, rendered_prompt = encode_prompt(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        device=device,
    )

    do_sample = temperature > 0.0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            **generate_kwargs,
        )

    full_sequence = generated[0]
    prompt_len = encoded["input_ids"].shape[1]
    generated_ids = full_sequence[prompt_len:]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    token_strings = tokenizer.convert_ids_to_tokens(generated_ids.tolist())

    return raw_output, token_strings, rendered_prompt


def run_tests_for_candidate(
    code: str,
    task_id: str,
    *,
    dataset_path: Path,
    timeout: int,
) -> tuple[bool, str, str, int]:
    """
    Run translated Racket tests on a candidate solution.

    Returns:
        (passed, stdout, stderr, exit_code)

    The candidate is written to a temporary `.rkt` file and evaluated through
    the same native-Racket harness used later by the main benchmark.
    """
    with tempfile.TemporaryDirectory(prefix="racket_gt_candidate_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        solution_path = tmpdir_path / "candidate_solution.rkt"
        solution_path.write_text(code, encoding="utf-8")

        command = [
            "python",
            "scripts/run_racket_humaneval_tests.py",
            str(solution_path),
            task_id,
            "--dataset",
            str(dataset_path),
            "--timeout",
            str(timeout),
        ]

        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        passed = result.returncode == 0

        return passed, stdout, stderr, result.returncode


def try_generate_ground_truth(
    row: dict[str, Any],
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    attempts: int,
    dataset_path: Path,
    timeout: int,
    model_name_or_path: str,
) -> tuple[Optional[GroundTruthRecord], dict[str, Any]]:
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Racket ground-truth solutions from the translated dataset"
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
        help="Maximum number of generation attempts per task",
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
        help="Optional limit on number of tasks to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_jsonl(args.input)
    for index, row in enumerate(rows):
        validate_task(row, index)

    if args.limit is not None:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype_name=args.dtype,
    )

    success_rows: list[GroundTruthRecord] = []
    failure_rows: list[dict[str, Any]] = []

    for row in rows:
        print(f"[GT] Processing {row['task_id']}")
        record, failure_info = try_generate_ground_truth(
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
            print(f"[GT] PASS {row['task_id']} in {record.num_attempts_used} attempt(s)")
            success_rows.append(record)
        else:
            print(f"[GT] FAIL {row['task_id']}")
            failure_rows.append(failure_info)

    write_jsonl(args.output, success_rows)
    write_failures(args.failures, failure_rows)

    print(f"Saved {len(success_rows)} ground-truth records")
    print(f"Saved {len(failure_rows)} failures")
    print(f"GT dataset: {args.output}")
    print(f"Failure report: {args.failures}")


if __name__ == "__main__":
    try:
        main()
    except GTBuildError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc
