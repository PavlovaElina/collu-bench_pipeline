from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .config import CanonicalSamplingConfig, DatasetConfig
from .data import TaskInstance
from .execution import ExecutionRunner
from .llm import BaseLLMClient
from .normalization import NormalizerRegistry
from .prompt import PromptBuilder
from .utils import extract_code_snippet
from tqdm import tqdm
LOGGER = logging.getLogger(__name__)


@dataclass
class CanonicalRecord:
    original: str
    normalized_text: str


class CanonicalRepository:
    """Storage for canonical solutions per task."""

    def __init__(self):
        self._data: Dict[str, Dict[str, List[CanonicalRecord]]] = {}

    def add(self, task: TaskInstance, normalized_text: str, original: str) -> bool:
        dataset_store = self._data.setdefault(task.dataset, {})
        task_store = dataset_store.setdefault(task.task_id, [])
        if any(record.normalized_text == normalized_text for record in task_store):
            return False
        task_store.append(CanonicalRecord(original=original, normalized_text=normalized_text))
        return True

    def get(self, task: TaskInstance) -> List[CanonicalRecord]:
        return list(self._data.get(task.dataset, {}).get(task.task_id, []))

    def dump(self, path: Path) -> None:
        serializable = {
            dataset: {
                task_id: [
                    {"original": record.original, "normalized": record.normalized_text}
                    for record in task_records
                ]
                for task_id, task_records in tasks.items()
            }
            for dataset, tasks in self._data.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serializable, indent=2))

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text())
        for dataset, tasks in data.items():
            for task_id, records in tasks.items():
                for record in records:
                    self._data.setdefault(dataset, {}).setdefault(task_id, []).append(
                        CanonicalRecord(
                            original=record["original"], normalized_text=record["normalized"]
                        )
                    )

    def total(self) -> int:
        return sum(len(task_records) for tasks in self._data.values() for task_records in tasks.values())


class CanonicalCollector:
    """Collect additional canonical solutions using LLM sampling."""

    def __init__(
        self,
        repo: CanonicalRepository,
        normalizers: NormalizerRegistry,
        executor: ExecutionRunner,
        dataset_builders: Dict[str, PromptBuilder],
        config: CanonicalSamplingConfig,
    ):
        self.repo = repo
        self.normalizers = normalizers
        self.executor = executor
        self.dataset_builders = dataset_builders
        self.config = config

    def seed_with_dataset(self, dataset: Sequence[TaskInstance]) -> None:
        for task in dataset:
            normalizer = self.normalizers.for_language(task.language)
            for canonical in task.canonical_solutions:
                normalized = normalizer.normalize(canonical).text
                added = self.repo.add(task, normalized, canonical)
                if added:
                    LOGGER.debug(
                        "Seeded canonical solution for %s/%s", task.dataset, task.task_id
                    )

    def collect(
        self,
        dataset: Sequence[TaskInstance],
        sampler_clients: Sequence[BaseLLMClient],
        dataset_config: DatasetConfig,
    ) -> None:
        if not self.config.enabled or not sampler_clients:
            return
        target_per_model = self.config.samples_per_model
        if target_per_model <= 0:
            return
        max_attempts = max(
            target_per_model, int(target_per_model * self.config.max_attempts_multiplier)
        )
        builder = self.dataset_builders[dataset_config.name]
        for task in tqdm(dataset, desc="Collecting canonical solutions"):  
            normalizer = self.normalizers.for_language(task.language)
            existing_records = self.repo.get(task)
            if task.canonical_solutions:
                if not existing_records:
                    for canonical in task.canonical_solutions:
                        normalized = normalizer.normalize(canonical).text
                        self.repo.add(task, normalized, canonical)
                continue
            if existing_records:
                continue
            for client in sampler_clients:
                collected = 0
                attempts = 0
                while collected < target_per_model and attempts < max_attempts:
                    prompt = builder.build(task)

                    generation = client.generate(
                        prompt, request_id=f"canonical-{task.dataset}-{task.task_id}"
                    )
                    code = extract_code_snippet(generation.text)
                    execution = self.executor.run(task, code)
                    attempts += 1
                    if execution.status != "pass":
                        continue
                    normalized = normalizer.normalize(code).text
                    added = self.repo.add(task, normalized, code)
                    if added:
                        collected += 1
                        LOGGER.info(
                            "%s canonical sample collected for %s/%s using %s",
                            collected,
                            task.dataset,
                            task.task_id,
                            client,
                        )
                    if attempts >= max_attempts:
                        break
