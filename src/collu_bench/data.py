from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus

from .config import DatasetConfig

TestKind = Literal["humaneval", "mbpp_assert", "script", "external_command"]


@dataclass
class TestSpec:
    """Executable specification for validating code."""

    kind: TestKind
    content: str = ""
    command: Optional[str] = None
    workdir: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskInstance:
    """Single dataset entry."""

    dataset: str
    task_id: str
    prompt: str
    question: str
    answer: str
    language: str
    entry_point: Optional[str]
    canonical_solutions: List[str]
    tests: TestSpec
    meta: Dict[str, Any] = field(default_factory=dict)


def load_dataset(config: DatasetConfig, repo_root: Path) -> List[TaskInstance]:
    """Load dataset tasks according to the config."""

    if config.source == "humaneval":
        return _load_humaneval(config)
    if config.source == "mbpp":
        return _load_mbpp(config)
    if config.source == "jsonl":
        return _load_jsonl(config, repo_root)
    raise ValueError(f"Unsupported dataset source {config.source}")


def _load_humaneval(config: DatasetConfig) -> List[TaskInstance]:
    data = get_human_eval_plus()
    tasks: List[TaskInstance] = []
    for idx, (raw_id, entry) in enumerate(sorted(data.items())):
        if config.limit and idx >= config.limit:
            break
        metadata = {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "task_id",
                "prompt",
                "canonical_solution",
                "test",
                "entry_point",
            }
        }
        question = entry["prompt"].strip()
        canonical = entry["canonical_solution"].strip()
        print(question, canonical)
        tasks.append(
            TaskInstance(
                dataset=config.name,
                task_id=entry["task_id"],
                prompt=_strip_triple_quotes(question),
                question=_strip_triple_quotes(question),
                answer=canonical,
                language="python",
                entry_point=entry.get("entry_point"),
                canonical_solutions=[canonical],
                tests=TestSpec(kind="humaneval", content=entry["test"]),
                meta={"source_task_id": raw_id, **metadata},
            )
        )
    return tasks


def _load_mbpp(config: DatasetConfig) -> List[TaskInstance]:
    data = get_mbpp_plus()
    tasks: List[TaskInstance] = []
    for idx, (raw_id, entry) in enumerate(sorted(data.items())):
        if config.limit and idx >= config.limit:
            break
        question = entry["prompt"].strip()
        canonical = entry["canonical_solution"].strip()
        metadata = {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "task_id",
                "prompt",
                "canonical_solution",
                "assertion",
                "entry_point",
            }
        }
        tasks.append(
            TaskInstance(
                dataset=config.name,
                task_id=entry["task_id"],
                prompt=_strip_triple_quotes(question),
                question=_strip_triple_quotes(question),
                answer=canonical,
                language="python",
                entry_point=entry.get("entry_point"),
                canonical_solutions=[canonical],
                tests=TestSpec(kind="mbpp_assert", content=entry.get("assertion", "")),
                meta={"source_task_id": raw_id, **metadata},
            )
        )
    return tasks


def _load_jsonl(config: DatasetConfig, repo_root: Path) -> List[TaskInstance]:
    path = (repo_root / config.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file {path} not found")
    tasks: List[TaskInstance] = []
    with path.open("r") as handle:
        for idx, line in enumerate(handle):
            if config.limit and idx >= config.limit:
                break
            record = json.loads(line)
            tests_dict = record.get("tests", {})
            tests = TestSpec(
                kind=tests_dict.get("kind", "script"),
                content=tests_dict.get("content", ""),
                command=tests_dict.get("command"),
                workdir=tests_dict.get("workdir"),
                environment=tests_dict.get("environment", {}),
            )
            canonical_raw = record.get("canonical_solutions")
            if not canonical_raw:
                canonical_raw = record.get("canonical_solution")
            if canonical_raw is None:
                canonical: List[str] = []
            elif isinstance(canonical_raw, str):
                canonical = [canonical_raw]
            else:
                canonical = list(canonical_raw)
            language = record.get("language") or config.language
            if not language:
                raise ValueError(f"Task {record.get('task_id')} missing language info")
            tasks.append(
                TaskInstance(
                    dataset=config.name,
                    task_id=str(record.get("task_id")),
                    prompt=record.get("prompt", "").strip(),
                    question=record.get("question", record.get("prompt", "").strip()),
                    answer=record.get("answer", canonical[0] if canonical else ""),
                    language=language,
                    entry_point=record.get("entry_point"),
                    canonical_solutions=canonical,
                    tests=tests,
                    meta=record.get("meta", {}),
                )
            )
    return tasks


def _strip_triple_quotes(text: str) -> str:
    text = text.strip()
    if text.startswith('"""') and text.endswith('"""'):
        return text[3:-3].strip()
    if text.startswith("'''") and text.endswith("'''"):
        return text[3:-3].strip()
    return text
