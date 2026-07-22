from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from .execution import ExecutionResult


@dataclass
class ColluRecord:
    idx: int
    model: str
    dataset: str
    task_id: str
    meta: Dict[str, Any]
    model_output: str
    closest_gt: str
    hallucination_token_index: Optional[int]
    tokens: List[str]
    token_types: List[str]
    token_logprobs: List[Dict[str, Any]]
    execution: ExecutionResult
    question: str
    answer: str

    def to_row(self) -> Dict[str, Any]:
        """Convert to a Hugging Face dataset row with nested types."""
        return {
            "idx": self.idx,
            "model": self.model,
            "dataset": self.dataset,
            "task_id": self.task_id,
            # JSON string: task metadata keys/types differ across datasets and
            # would otherwise produce incompatible Arrow structs.
            "meta": json.dumps(_json_ready(self.meta), sort_keys=True, ensure_ascii=False),
            "model_output": self.model_output,
            "closest_gt": self.closest_gt,
            "hallucination_token_index": self.hallucination_token_index,
            "tokens": list(self.tokens),
            "token_types": list(self.token_types),
            "token_logprobs": _format_token_logprobs(self.token_logprobs),
            "execution": _format_execution(self.execution),
            "question": self.question,
            "answer": self.answer,
        }


def _json_ready(value: Any) -> Any:
    """Recursively convert values into JSON/Arrow-friendly Python objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return json.loads(json.dumps(value, default=str))


def _format_token_logprobs(
    token_logprobs: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    Normalize per-step logprobs to the Collu-Bench HF schema:

    List[List[{decoded_token, logprob, token_id}]]
    """
    formatted: List[List[Dict[str, Any]]] = []
    for step in token_logprobs:
        if isinstance(step, list):
            candidates = step
        elif isinstance(step, dict) and "top_logprobs" in step:
            candidates = step["top_logprobs"] or [
                {
                    "decoded_token": step.get("decoded_token", ""),
                    "logprob": step.get("logprob", 0.0),
                    "token_id": step.get("token_id", -1),
                }
            ]
        elif isinstance(step, dict):
            candidates = [step]
        else:
            candidates = []

        formatted.append(
            [
                {
                    "decoded_token": str(entry.get("decoded_token", "")),
                    "logprob": float(entry.get("logprob", 0.0)),
                    "token_id": int(entry.get("token_id", -1)),
                }
                for entry in candidates
                if isinstance(entry, dict)
            ]
        )
    return formatted


def _format_execution(execution: ExecutionResult) -> str:
    """Match the official Collu-Bench field: a single execution-feedback string."""
    if execution.status == "pass":
        return ""
    feedback = (execution.stderr or execution.stdout or "").strip()
    if feedback:
        return feedback
    return execution.status


class StorageWriter:
    """Collect dataset rows and flush to a Hugging Face datasets directory."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.records: List[Dict[str, Any]] = []

    def append(self, record: ColluRecord) -> None:
        self.records.append(record.to_row())

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists() and self.output_path.is_file():
            self.output_path.unlink()
        dataset = Dataset.from_list(self.records)
        dataset.save_to_disk(str(self.output_path))
