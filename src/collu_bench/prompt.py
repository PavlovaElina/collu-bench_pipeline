from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from .config import DatasetConfig, PromptConfig
from .data import TaskInstance


@dataclass
class PromptPayload:
    """Prompt to be sent to an LLM."""

    mode: str
    content: Union[str, List[Dict[str, str]]]


class PromptBuilder:
    """Render dataset specific prompts."""

    def __init__(self, dataset_config: DatasetConfig, repo_root: Path):
        self.dataset_config = dataset_config
        self.repo_root = repo_root
        self.prompt_cfg = dataset_config.prompt
        self.few_shot = None
        if self.prompt_cfg.few_shot_path:
            few_shot_path = (repo_root / self.prompt_cfg.few_shot_path).expanduser()
            if not few_shot_path.exists():
                raise FileNotFoundError(f"few-shot file {few_shot_path} missing")
            self.few_shot = few_shot_path.read_text().strip()

    def build(self, task: TaskInstance) -> PromptPayload:
        context = {
            "dataset": task.dataset,
            "task_id": task.task_id,
            "question": task.question,
            "prompt": task.prompt,
            "meta": task.meta,
        }
        segments: List[str] = []
        if self.prompt_cfg.prefix:
            segments.append(self.prompt_cfg.prefix.format(**context))
        if self.few_shot:
            segments.append(self.few_shot)
        segments.append(task.prompt.strip())
        if self.prompt_cfg.suffix:
            segments.append(self.prompt_cfg.suffix.format(**context))
        payload = "\n\n".join(segment.strip() for segment in segments if segment.strip())
        if self.prompt_cfg.mode == "chat":
            messages = []
            if self.prompt_cfg.system:
                messages.append({"role": "system", "content": self.prompt_cfg.system})
            messages.append({"role": "user", "content": payload})
            return PromptPayload(mode="chat", content=messages)
        return PromptPayload(mode="text", content=payload)

