from __future__ import annotations

"""
Validated configuration models for the benchmark pipeline.

The pipeline is restricted to HumanEval, MBPP, and HumanEval-Java. An optional
``target_language`` rewrites task prompts (via OpenAI Batch API) into an
arbitrary programming language before evaluation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import yaml
from pydantic import BaseModel, Field, validator


ALLOWED_DATASET_NAMES: Set[str] = {"humaneval", "mbpp", "humaneval_java"}
DatasetSource = Literal["humaneval", "mbpp", "jsonl"]
TaskType = Literal["cg", "apr"]
PromptMode = Literal["text", "chat"]

_QUANTIZATION_ALIASES = {
    "q8": "q8",
    "8bit": "q8",
    "int8": "q8",
    "llmint8": "q8",
    "bitsandbytes": "q8",
    "bitsandbytes8bit": "q8",
    "mxfp4": "mxfp4",
    "fp4": "mxfp4",
    "fp8": "fp8",
    "awq": "awq",
    "gptq": "gptq",
    "auto": "auto",
    "none": "none",
    "null": "none",
}


def _normalize_quantization_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip().lower().replace("_", "").replace("-", "")
    if not cleaned:
        return None
    if cleaned not in _QUANTIZATION_ALIASES:
        supported = ", ".join(sorted(set(_QUANTIZATION_ALIASES.values())))
        raise ValueError(
            f"Unsupported quantization='{value}'. Supported: {supported}"
        )
    return _QUANTIZATION_ALIASES[cleaned]


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
    language: Optional[str] = None
    path: Optional[str] = None
    limit: Optional[int] = None
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @validator("name")
    def _validate_allowed_dataset(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_DATASET_NAMES:
            allowed = ", ".join(sorted(ALLOWED_DATASET_NAMES))
            raise ValueError(
                f"Unsupported dataset name '{value}'. "
                f"Only {allowed} are allowed."
            )
        return normalized

    @validator("path")
    def _validate_path(cls, value: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Require a path for JSONL-backed datasets."""
        if values.get("source") == "jsonl" and not value:
            raise ValueError("jsonl datasets must provide a path")
        return value

    @validator("source")
    def _validate_source_matches_name(
        cls,
        value: DatasetSource,
        values: Dict[str, Any],
    ) -> DatasetSource:
        name = values.get("name")
        if name == "humaneval" and value != "humaneval":
            raise ValueError("dataset 'humaneval' must use source='humaneval'")
        if name == "mbpp" and value != "mbpp":
            raise ValueError("dataset 'mbpp' must use source='mbpp'")
        if name == "humaneval_java" and value != "jsonl":
            raise ValueError("dataset 'humaneval_java' must use source='jsonl'")
        return value

    @validator("language", always=True)
    def _default_source_language(
        cls,
        value: Optional[str],
        values: Dict[str, Any],
    ) -> Optional[str]:
        """
        Fill in the native language for built-in datasets when omitted.

        ``target_language`` on the pipeline config may later rewrite prompts into
        a different language; this field records the dataset's native language.
        """
        source = values.get("source")
        name = values.get("name")

        if value:
            return value.strip().lower()

        if source in {"humaneval", "mbpp"} or name in {"humaneval", "mbpp"}:
            return "python"
        if name == "humaneval_java":
            return "java"
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
    # Weight quantization, e.g. "q8", "mxfp4", "fp8", "auto".
    quantization: Optional[str] = None
    environment: Dict[str, str] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @validator("quantization", pre=True)
    def _normalize_quantization(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_quantization_name(value)


class CanonicalSamplingConfig(BaseModel):
    """Settings for optional canonical solution expansion."""

    enabled: bool = True
    samples_per_model: int = 100
    max_attempts_multiplier: float = 2.5
    cache_path: Optional[str] = "artifacts/canonical.json"
    sampler_models: List[LLMConfig] = Field(default_factory=list)


class PromptTranslationConfig(BaseModel):
    """Settings for OpenAI Batch API prompt translation into ``target_language``."""

    enabled: bool = True
    translate_solutions: bool = True
    cache_path: Optional[str] = "artifacts/prompt_translations.json"
    # Workflow mode:
    # - export: write a Batch API input JSONL for manual upload, then stop
    # - import: load a Batch API output JSONL and continue the pipeline
    # - api: submit/poll the Batch API automatically (requires API key)
    mode: str = "export"
    # OpenAI model used for every request in the translation batch.
    model: str = "gpt-5.4-nano"
    # Path to a local file containing the OpenAI API key (one line). Prefer this
    # over committing secrets; see ``secrets/openai_api_key.example``.
    # Only required when mode=api.
    api_key_path: str = "secrets/openai_api_key"
    # Separate folders for manually uploaded Batch inputs and downloaded outputs.
    input_dir: str = "artifacts/translation_inputs"
    output_dir: str = "artifacts/translation_outputs"
    # Optional basename for the exported input JSONL (written under input_dir).
    # When omitted, a timestamped name is generated.
    batch_input_file: Optional[str] = None
    # Basename or path of the Batch output JSONL to load (mode=import).
    # Relative paths are resolved under output_dir first.
    batch_output_file: Optional[str] = None
    # Batch completion window accepted by the OpenAI Batch API (mode=api).
    completion_window: str = "24h"
    # How often to poll batch status while waiting for completion (mode=api).
    poll_interval_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 2048
    # Reasoning effort for GPT-5 family models (none/low/medium/high/xhigh).
    reasoning_effort: Optional[str] = "none"

    @validator("mode", pre=True, always=True)
    def _normalize_mode(cls, value: Optional[str]) -> str:
        cleaned = str(value or "export").strip().lower()
        aliases = {
            "export": "export",
            "write": "export",
            "write_input": "export",
            "import": "import",
            "load": "import",
            "load_output": "import",
            "api": "api",
            "auto": "api",
            "submit": "api",
        }
        if cleaned not in aliases:
            raise ValueError(
                f"Unsupported prompt_translation.mode='{value}'. "
                "Supported: export, import, api"
            )
        return aliases[cleaned]

    @validator("model", pre=True, always=True)
    def _normalize_model(cls, value: Optional[str]) -> str:
        cleaned = str(value or "gpt-5.4-nano").strip()
        if not cleaned:
            raise ValueError("prompt_translation.model must be a non-empty string")
        return cleaned

    @validator("api_key_path", pre=True, always=True)
    def _normalize_api_key_path(cls, value: Optional[str]) -> str:
        cleaned = str(value or "secrets/openai_api_key").strip()
        if not cleaned:
            raise ValueError("prompt_translation.api_key_path must be a non-empty path")
        return cleaned

    @validator("input_dir", pre=True, always=True)
    def _normalize_input_dir(cls, value: Optional[str]) -> str:
        cleaned = str(value or "artifacts/translation_inputs").strip()
        if not cleaned:
            raise ValueError("prompt_translation.input_dir must be a non-empty path")
        return cleaned

    @validator("output_dir", pre=True, always=True)
    def _normalize_output_dir(cls, value: Optional[str]) -> str:
        cleaned = str(value or "artifacts/translation_outputs").strip()
        if not cleaned:
            raise ValueError("prompt_translation.output_dir must be a non-empty path")
        return cleaned

    @validator("batch_input_file", pre=True)
    def _normalize_batch_input_file(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @validator("batch_output_file", pre=True)
    def _normalize_batch_output_file(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @validator("completion_window", pre=True, always=True)
    def _normalize_completion_window(cls, value: Optional[str]) -> str:
        cleaned = str(value or "24h").strip().lower()
        if cleaned != "24h":
            raise ValueError(
                f"Unsupported prompt_translation.completion_window='{value}'. "
                "Only '24h' is currently supported by the OpenAI Batch API."
            )
        return cleaned

    @validator("poll_interval_seconds", pre=True, always=True)
    def _normalize_poll_interval(cls, value: Any) -> float:
        interval = float(value if value is not None else 30.0)
        if interval <= 0:
            raise ValueError("prompt_translation.poll_interval_seconds must be > 0")
        return interval

    @validator("reasoning_effort", pre=True)
    def _normalize_reasoning_effort(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip().lower()
        if not cleaned or cleaned == "null":
            return None
        allowed = {"none", "low", "medium", "high", "xhigh"}
        if cleaned not in allowed:
            raise ValueError(
                f"Unsupported prompt_translation.reasoning_effort='{value}'. "
                f"Supported: {', '.join(sorted(allowed))}"
            )
        return cleaned


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    output_csv: str = "collu-bench.csv"
    datasets: List[DatasetConfig]
    eval_models: List[LLMConfig]
    # Arbitrary programming language; when set, prompts are translated via LLM.
    target_language: Optional[str] = None
    canonical_sampling: CanonicalSamplingConfig = Field(
        default_factory=CanonicalSamplingConfig
    )
    prompt_translation: PromptTranslationConfig = Field(
        default_factory=PromptTranslationConfig
    )
    execution_timeout: int = 120
    workspace: str = "artifacts/workspace"
    resume_path: Optional[str] = None

    @validator("target_language", pre=True)
    def _normalize_target_language(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @validator("datasets")
    def _require_datasets(cls, value: List[DatasetConfig]) -> List[DatasetConfig]:
        if not value:
            raise ValueError("at least one dataset must be configured")
        names = {dataset.name for dataset in value}
        unexpected = names - ALLOWED_DATASET_NAMES
        if unexpected:
            raise ValueError(
                "Unsupported datasets: "
                + ", ".join(sorted(unexpected))
                + f". Allowed: {', '.join(sorted(ALLOWED_DATASET_NAMES))}"
            )
        return value

    def named_models(self) -> Dict[str, LLMConfig]:
        """Return all configured LLMs keyed by their ``name`` field."""
        models: Dict[str, LLMConfig] = {cfg.name: cfg for cfg in self.eval_models}
        for cfg in self.canonical_sampling.sampler_models:
            models[cfg.name] = cfg
        return models


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
