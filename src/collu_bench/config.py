from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator


SupportedLanguage = Literal["python", "java", "racket"]
DatasetSource = Literal["humaneval", "mbpp", "jsonl"]
TaskType = Literal["cg", "apr"]
PromptMode = Literal["text", "chat"]


class PromptConfig(BaseModel):
    """Prompt template settings for a dataset."""

    mode: PromptMode = "text"
    system: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    few_shot_path: Optional[str] = None

    def load_few_shot(self, root: Path) -> Optional[str]:
        """Load few-shot examples from disk if a path is provided."""
        if not self.few_shot_path:
            return None
        path = (root / self.few_shot_path).expanduser().resolve()
        return path.read_text(encoding="utf-8")


class DatasetConfig(BaseModel):
    """Configuration describing the origin of a task collection."""

    name: str
    source: DatasetSource
    task_type: TaskType
    language: Optional[SupportedLanguage] = None
    path: Optional[str] = None
    limit: Optional[int] = None
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @validator("path")
    def _validate_path(cls, value: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Require a path for JSONL-backed datasets."""
        if values.get("source") == "jsonl" and not value:
            raise ValueError("jsonl datasets must provide a path")
        return value

    @validator("language", always=True)
    def _validate_language_for_builtin_sources(
        cls,
        value: Optional[SupportedLanguage],
        values: Dict[str, Any],
    ) -> Optional[SupportedLanguage]:
        """
        Enforce consistent language settings.

        Rules:
        - humaneval / mbpp are Python datasets by definition
        - jsonl may specify python / java / racket
        """
        source = values.get("source")

        if source in {"humaneval", "mbpp"}:
            if value is None:
                return "python"
            if value != "python":
                raise ValueError(
                    f"source='{source}' only supports language='python', got '{value}'"
                )
            return value

        return value


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    name: str
    model: str
    local_model_path: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    logprobs: int = 5
    tokenizer: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    environment: Dict[str, str] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)


class CanonicalSamplingConfig(BaseModel):
    """Settings for optional canonical solution expansion."""

    enabled: bool = True
    samples_per_model: int = 100
    max_attempts_multiplier: float = 2.5
    cache_path: Optional[str] = "artifacts/canonical.json"
    sampler_models: List[LLMConfig] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    output_csv: str = "collu-bench.csv"
    datasets: List[DatasetConfig]
    eval_models: List[LLMConfig]
    canonical_sampling: CanonicalSamplingConfig = Field(
        default_factory=CanonicalSamplingConfig
    )
    execution_timeout: int = 120
    workspace: str = "artifacts/workspace"
    resume_path: Optional[str] = None


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config file into a validated PipelineConfig."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    pipeline_config = PipelineConfig(**data)
    return pipeline_config


def dump_config(config: PipelineConfig, path: str | Path) -> None:
    """Persist a PipelineConfig to disk for debugging."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(json.loads(config.model_dump_json()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )