from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        return {
            "idx": self.idx,
            "model": self.model,
            "dataset": self.dataset,
            "task_id": self.task_id,
            "meta": json.dumps(self.meta, sort_keys=True),
            "model_output": self.model_output,
            "closest_gt": self.closest_gt,
            "hallucination_token_index": (
                "" if self.hallucination_token_index is None else self.hallucination_token_index
            ),
            "tokens": json.dumps(self.tokens),
            "token_types": json.dumps(self.token_types),
            "token_logprobs": json.dumps(self.token_logprobs),
            "execution": json.dumps(
                {
                    "status": self.execution.status,
                    "stdout": self.execution.stdout,
                    "stderr": self.execution.stderr,
                    "command": self.execution.command,
                }
            ),
            "question": self.question,
            "answer": self.answer,
        }


class StorageWriter:
    """Collect dataset rows and flush to CSV."""

    FIELDNAMES = [
        "idx",
        "model",
        "dataset",
        "task_id",
        "meta",
        "model_output",
        "closest_gt",
        "hallucination_token_index",
        "tokens",
        "token_types",
        "token_logprobs",
        "execution",
        "question",
        "answer",
    ]

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.records: List[Dict[str, Any]] = []

    def append(self, record: ColluRecord) -> None:
        self.records.append(record.to_row())

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES, delimiter=";")
            writer.writeheader()
            for row in self.records:
                writer.writerow(row)
