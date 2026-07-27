from __future__ import annotations

"""
LLM-based translation of coding-task prompts into a target programming language.

When ``PipelineConfig.target_language`` differs from a task's native language,
this module rewrites the task prompt (and optional reference solution) so the
rest of the Collu-Bench pipeline can generate and analyze code in the requested
language.

Supported ``prompt_translation.mode`` values:
- ``export``: compose Batch API input JSONL under ``input_dir`` for manual
  upload on the OpenAI platform, then stop the pipeline
- ``import``: load Batch API output JSONL from ``output_dir`` and apply
  translations before continuing
- ``api``: submit/poll via the OpenAI Batch API automatically

Results are cached on disk so repeated runs do not redo identical work.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from .config import PromptTranslationConfig
from .data import TaskInstance, TestSpec
from .secrets import load_openai_api_key, resolve_workdir_path
from .utils import extract_code_snippet

LOGGER = logging.getLogger(__name__)

SOURCE_LANGUAGE_BY_DATASET = {
    "humaneval": "python",
    "mbpp": "python",
    "humaneval_java": "java",
}

_BATCH_ENDPOINT = "/v1/chat/completions"
_TERMINAL_BATCH_STATUSES = frozenset(
    {"completed", "failed", "expired", "cancelled", "cancelling"}
)

_TRANSLATION_SYSTEM = (
    "You are an expert software engineer who translates coding-benchmark "
    "prompts between programming languages while preserving problem semantics."
)

# Bump when the required translated-prompt layout or included artifacts change
# so old cache entries are not reused.
_PROMPT_LAYOUT_VERSION = "sections_v2_with_tests"

_TRANSLATION_INSTRUCTIONS = """\
Rewrite the coding task below from {source_language} into {target_language}.

Output ONLY the translated prompt using this exact section layout.
Omit any section that would be empty (do not print its header).

TASK FORMULATION
<natural-language problem statement>

EXAMPLES
<examples rewritten into idiomatic {target_language}>

FUNCTION SIGNATURE
<idiomatic {target_language} function signature(s) to implement>

Rules:
- TASK FORMULATION must keep the original natural-language wording unchanged \
(copy it; do not paraphrase, shorten, or expand). Exclude source-language code \
stubs, imports, type annotations, and doctest/assert lines from this section; \
those belong in EXAMPLES / FUNCTION SIGNATURE when present.
- EXAMPLES must rewrite every original example/doctest/assert into idiomatic \
{target_language} call/result form. If the original has no examples, omit the \
EXAMPLES section entirely.
- FUNCTION SIGNATURE must give the idiomatic {target_language} signature(s) for \
the function(s) the model should implement. If no entry point can be inferred, \
omit the FUNCTION SIGNATURE section entirely.
- Do not solve the problem. Do not include implementations.
- Do not use markdown fences or any commentary outside the sections.
- Section headers must appear exactly as: TASK FORMULATION, EXAMPLES, \
FUNCTION SIGNATURE (when the section is included).
- Separate consecutive sections with a single blank line.
"""

_SOLUTION_INSTRUCTIONS = """\
Translate the reference solution below from {source_language} into {target_language}.

Requirements:
- Preserve the algorithm and behavior.
- Produce a complete, idiomatic {target_language} implementation.
- Output ONLY the translated code. No markdown fences, no commentary.

Original prompt (for context):
{prompt}

Reference solution:
{solution}
"""

_TEST_INSTRUCTIONS = """\
Translate the unit tests / assertions below from {source_language} into {target_language}.

Requirements:
- Preserve every test case and expected outcome.
- Rewrite calls, types, and assertions into idiomatic {target_language}.
- Keep the same coverage; do not add or remove cases.
- Output ONLY the translated tests. No markdown fences, no commentary.

Function / entry point (if known): {entry_point}

Original prompt (for context):
{prompt}

Tests:
{tests}
"""


class TranslationBatchExported(Exception):
    """Raised after writing a Batch input JSONL so the pipeline can stop cleanly."""

    def __init__(self, input_path: Path, manifest_path: Path, request_count: int):
        self.input_path = input_path
        self.manifest_path = manifest_path
        self.request_count = request_count
        super().__init__(
            f"Wrote {request_count} Batch API request(s) to {input_path}. "
            f"Upload this file on the OpenAI Batch platform, then re-run with "
            f"mode=import after placing the output JSONL under the configured "
            f"output_dir. Manifest: {manifest_path}"
        )


@dataclass
class _PendingTask:
    task: TaskInstance
    source_language: str
    cache_key: str
    prompt_custom_id: str
    solution_custom_id: Optional[str]
    test_custom_id: Optional[str]


class PromptTranslator:
    """Translate task prompts (and optional answers) via OpenAI Batch JSONL."""

    def __init__(
        self,
        config: PromptTranslationConfig,
        target_language: str,
        *,
        repo_root: Optional[Path] = None,
    ):
        self.config = config
        self.target_language = target_language.strip()
        self.repo_root = Path(repo_root).resolve() if repo_root else Path.cwd()
        self._cache: Dict[str, Dict[str, str]] = {}
        self._cache_path = (
            resolve_workdir_path(config.cache_path, repo_root=self.repo_root)
            if config.cache_path
            else None
        )
        self._input_dir = resolve_workdir_path(
            config.input_dir,
            repo_root=self.repo_root,
        )
        self._output_dir = resolve_workdir_path(
            config.output_dir,
            repo_root=self.repo_root,
        )
        if self._cache_path:
            self._load_cache()

    def translate_tasks(self, tasks: List[TaskInstance]) -> List[TaskInstance]:
        """
        Return tasks rewritten for ``target_language``.

        In ``export`` mode, writes the Batch input JSONL and raises
        ``TranslationBatchExported`` so the caller can stop the pipeline.
        """
        ready: Dict[Tuple[str, str], TaskInstance] = {}
        pending: List[_PendingTask] = []

        progress = tqdm(
            tasks,
            desc=f"prepare->{self.target_language}",
            unit="task",
            leave=True,
        )
        for task in progress:
            source_language = _infer_source_language(task)
            progress.set_postfix_str(f"{task.dataset}/{task.task_id}", refresh=False)
            key = (task.dataset, task.task_id)

            if source_language.lower() == self.target_language.lower():
                task.language = self.target_language
                ready[key] = task
                _display_prompt_pair(
                    task_id=f"{task.dataset}/{task.task_id}",
                    source_language=source_language,
                    target_language=self.target_language,
                    original=task.prompt,
                    translated_prompt=task.prompt,
                    original_tests=task.tests.content if task.tests else "",
                    translated_tests=task.tests.content if task.tests else "",
                    original_solution=task.answer,
                    translated_solution=task.answer,
                    skipped=True,
                )
                continue

            cache_key = _cache_key(
                task=task,
                source_language=source_language,
                target_language=self.target_language,
                prompt=task.prompt,
                solution=task.answer,
                tests=(task.tests.content if task.tests else ""),
                translate_solutions=self.config.translate_solutions,
            )
            cached = self._cache.get(cache_key)
            if cached:
                result = self._build_translated_task(
                    task=task,
                    source_language=source_language,
                    new_prompt=cached["prompt"],
                    new_answer=cached.get("answer", task.answer),
                    new_tests=cached.get("tests"),
                    from_cache=True,
                )
                _display_prompt_pair(
                    task_id=f"{task.dataset}/{task.task_id}",
                    source_language=source_language,
                    target_language=self.target_language,
                    original=task.prompt,
                    translated_prompt=result.prompt,
                    original_tests=task.tests.content if task.tests else "",
                    translated_tests=cached.get("tests")
                    or (result.tests.content if result.tests else ""),
                    original_solution=task.answer,
                    translated_solution=result.answer,
                    skipped=False,
                    from_cache=True,
                )
                ready[key] = result
                continue

            prompt_custom_id = _custom_id(task, "prompt")
            solution_custom_id = None
            if self.config.translate_solutions and task.answer.strip():
                solution_custom_id = _custom_id(task, "solution")
            test_custom_id = None
            if task.tests and task.tests.content.strip():
                test_custom_id = _custom_id(task, "tests")
            pending.append(
                _PendingTask(
                    task=task,
                    source_language=source_language,
                    cache_key=cache_key,
                    prompt_custom_id=prompt_custom_id,
                    solution_custom_id=solution_custom_id,
                    test_custom_id=test_custom_id,
                )
            )

        mode = self.config.mode
        if pending and mode == "export":
            input_path, manifest_path, request_count = self._export_batch_input(pending)
            raise TranslationBatchExported(input_path, manifest_path, request_count)

        if pending:
            if mode == "import":
                LOGGER.info(
                    "Loading %s uncached translation(s) from Batch output JSONL",
                    len(pending),
                )
                batch_outputs = self._load_batch_output()
            elif mode == "api":
                LOGGER.info(
                    "Submitting %s uncached translation(s) to OpenAI Batch API "
                    "(model=%s)",
                    len(pending),
                    self.config.model,
                )
                batch_outputs = self._run_openai_batch(pending)
            else:
                raise ValueError(f"Unsupported translation mode: {mode}")

            for item in pending:
                new_prompt = _clean_translation_output(
                    _require_batch_text(batch_outputs, item.prompt_custom_id)
                )
                new_answer = item.task.answer
                if item.solution_custom_id is not None:
                    new_answer = extract_code_snippet(
                        _clean_translation_output(
                            _require_batch_text(batch_outputs, item.solution_custom_id)
                        )
                    )
                new_tests = None
                if item.test_custom_id is not None:
                    new_tests = _clean_translation_output(
                        _require_batch_text(batch_outputs, item.test_custom_id)
                    )
                cache_entry: Dict[str, str] = {
                    "prompt": new_prompt,
                    "answer": new_answer,
                }
                if new_tests is not None:
                    cache_entry["tests"] = new_tests
                self._cache[item.cache_key] = cache_entry
                result = self._build_translated_task(
                    task=item.task,
                    source_language=item.source_language,
                    new_prompt=new_prompt,
                    new_answer=new_answer,
                    new_tests=new_tests,
                    from_cache=False,
                )
                _display_prompt_pair(
                    task_id=f"{item.task.dataset}/{item.task.task_id}",
                    source_language=item.source_language,
                    target_language=self.target_language,
                    original=item.task.prompt,
                    translated_prompt=result.prompt,
                    original_tests=item.task.tests.content if item.task.tests else "",
                    translated_tests=new_tests or "",
                    original_solution=item.task.answer,
                    translated_solution=new_answer,
                    skipped=False,
                    from_cache=False,
                )
                ready[(item.task.dataset, item.task.task_id)] = result

        ordered = [ready[(task.dataset, task.task_id)] for task in tasks]
        if self._cache_path:
            self._dump_cache()
        return ordered

    def _build_translated_task(
        self,
        *,
        task: TaskInstance,
        source_language: str,
        new_prompt: str,
        new_answer: str,
        new_tests: Optional[str],
        from_cache: bool,
    ) -> TaskInstance:
        original_tests = task.tests.content if task.tests else ""
        meta = {
            **task.meta,
            "source_language": source_language,
            "target_language": self.target_language,
            "original_prompt": task.prompt,
            "original_answer": task.answer,
            "original_tests": original_tests,
            "prompt_translated": True,
            "translation_from_cache": from_cache,
            "translation_model": self.config.model,
            "translation_backend": f"openai_batch:{self.config.mode}",
        }
        if new_tests is not None:
            meta["translated_tests"] = new_tests
        tests = task.tests
        if new_tests is not None and task.tests is not None:
            tests = TestSpec(
                kind=task.tests.kind,
                content=new_tests,
                command=task.tests.command,
                workdir=task.tests.workdir,
                environment=dict(task.tests.environment),
            )
        canonical = [new_answer] if new_answer.strip() else []
        return TaskInstance(
            dataset=task.dataset,
            task_id=task.task_id,
            prompt=new_prompt,
            question=new_prompt,
            answer=new_answer,
            language=self.target_language,
            entry_point=task.entry_point,
            canonical_solutions=canonical,
            tests=tests,
            meta=meta,
        )

    def _export_batch_input(
        self,
        pending: Sequence[_PendingTask],
    ) -> Tuple[Path, Path, int]:
        """Write Batch API input JSONL + manifest under ``input_dir``."""
        self._input_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lang = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.target_language.lower()) or "lang"
        default_name = f"translation_batch_{lang}_{stamp}.jsonl"
        filename = self.config.batch_input_file or default_name
        if not filename.endswith(".jsonl"):
            filename = f"{filename}.jsonl"

        input_path = self._resolve_under_dir(self._input_dir, filename)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = input_path.with_name(f"{input_path.stem}_manifest.json")

        requests = self._compose_batch_requests(pending)
        _write_jsonl(input_path, requests)
        _write_json(
            manifest_path,
            {
                "created_at": stamp,
                "mode": "export",
                "model": self.config.model,
                "target_language": self.target_language,
                "input_path": str(input_path),
                "output_dir": str(self._output_dir),
                "request_count": len(requests),
                "task_count": len(pending),
                "translate_solutions": self.config.translate_solutions,
                "tasks": [
                    {
                        "dataset": item.task.dataset,
                        "task_id": item.task.task_id,
                        "source_language": item.source_language,
                        "cache_key": item.cache_key,
                        "prompt_custom_id": item.prompt_custom_id,
                        "solution_custom_id": item.solution_custom_id,
                        "test_custom_id": item.test_custom_id,
                    }
                    for item in pending
                ],
            },
        )
        LOGGER.info(
            "Exported %s Batch API request(s) for manual upload to %s "
            "(manifest=%s). Place the downloaded output JSONL under %s "
            "and re-run with mode=import.",
            len(requests),
            input_path,
            manifest_path,
            self._output_dir,
        )
        return input_path, manifest_path, len(requests)

    def _load_batch_output(self) -> Dict[str, str]:
        """Parse a manually downloaded Batch output JSONL from ``output_dir``."""
        configured = self.config.batch_output_file
        if not configured:
            raise ValueError(
                "prompt_translation.mode=import requires "
                "prompt_translation.batch_output_file "
                "(filename under output_dir or an absolute path)"
            )
        output_path = self._resolve_output_file(configured)
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Batch output JSONL not found at {output_path}. "
                f"Download the OpenAI Batch results into {self._output_dir} "
                "and set prompt_translation.batch_output_file accordingly."
            )
        outputs = _parse_batch_output(output_path)
        LOGGER.info(
            "Loaded %s translation response(s) from %s",
            len(outputs),
            output_path,
        )
        return outputs

    def _resolve_output_file(self, configured: str) -> Path:
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path.resolve()
        under_output = (self._output_dir / path).resolve()
        if under_output.is_file() or not path.exists():
            return under_output
        return resolve_workdir_path(path, repo_root=self.repo_root)

    @staticmethod
    def _resolve_under_dir(directory: Path, filename: str) -> Path:
        path = Path(filename).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (directory / path.name).resolve()

    def _run_openai_batch(self, pending: Sequence[_PendingTask]) -> Dict[str, str]:
        """Compose, submit, poll, and parse one OpenAI Batch job."""
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The 'openai' package is required for Batch API translation. "
                "Install it with: pip install openai"
            ) from exc

        api_key = load_openai_api_key(
            self.config.api_key_path,
            repo_root=self.repo_root,
        )
        client = OpenAI(api_key=api_key)

        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lang = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.target_language.lower()) or "lang"
        input_name = self.config.batch_input_file or f"translation_batch_{lang}_{stamp}.jsonl"
        if not str(input_name).endswith(".jsonl"):
            input_name = f"{input_name}.jsonl"
        input_path = self._resolve_under_dir(self._input_dir, str(input_name))
        output_name = (
            self.config.batch_output_file
            or f"{Path(input_name).stem}_output.jsonl"
        )
        output_path = self._resolve_under_dir(self._output_dir, str(output_name))
        error_path = output_path.with_name(f"{output_path.stem}_errors.jsonl")
        meta_path = input_path.with_name(f"{input_path.stem}_meta.json")

        requests = self._compose_batch_requests(pending)
        _write_jsonl(input_path, requests)
        LOGGER.info(
            "Wrote %s Batch API request(s) to %s",
            len(requests),
            input_path,
        )

        with input_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        LOGGER.info("Uploaded batch input file id=%s", uploaded.id)

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint=_BATCH_ENDPOINT,
            completion_window=self.config.completion_window,
            metadata={
                "description": f"collu-bench prompt translation -> {self.target_language}",
                "model": self.config.model,
            },
        )
        LOGGER.info("Created OpenAI batch id=%s status=%s", batch.id, batch.status)
        _write_json(
            meta_path,
            {
                "batch_id": batch.id,
                "input_file_id": uploaded.id,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "model": self.config.model,
                "target_language": self.target_language,
                "request_count": len(requests),
                "created_at": stamp,
            },
        )

        batch = self._wait_for_batch(client, batch.id)
        _write_json(
            meta_path,
            {
                "batch_id": batch.id,
                "input_file_id": uploaded.id,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "status": batch.status,
                "request_counts": _request_counts_dict(batch),
                "model": self.config.model,
                "target_language": self.target_language,
                "request_count": len(requests),
                "created_at": stamp,
            },
        )

        if batch.status != "completed":
            if getattr(batch, "error_file_id", None):
                _download_file_content(client, batch.error_file_id, error_path)
            raise RuntimeError(
                f"OpenAI batch {batch.id} finished with status={batch.status}. "
                f"See {meta_path}"
                + (f" and {error_path}" if error_path.exists() else "")
            )

        if not batch.output_file_id:
            raise RuntimeError(
                f"OpenAI batch {batch.id} completed without an output_file_id"
            )

        _download_file_content(client, batch.output_file_id, output_path)
        outputs = _parse_batch_output(output_path)
        LOGGER.info(
            "Parsed %s successful translation response(s) from batch %s "
            "(saved to %s)",
            len(outputs),
            batch.id,
            output_path,
        )

        if getattr(batch, "error_file_id", None):
            _download_file_content(client, batch.error_file_id, error_path)
            LOGGER.warning(
                "Batch %s reported per-request errors; details written to %s",
                batch.id,
                error_path,
            )

        return outputs

    def _compose_batch_requests(
        self,
        pending: Sequence[_PendingTask],
    ) -> List[Dict[str, Any]]:
        requests: List[Dict[str, Any]] = []
        for item in pending:
            prompt_user = (
                _TRANSLATION_INSTRUCTIONS.format(
                    source_language=item.source_language,
                    target_language=self.target_language,
                )
                + "\n\n"
                + item.task.prompt.strip()
            )
            requests.append(
                _chat_completion_request(
                    custom_id=item.prompt_custom_id,
                    model=self.config.model,
                    user_content=prompt_user,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    reasoning_effort=self.config.reasoning_effort,
                )
            )
            if item.solution_custom_id is not None:
                solution_user = _SOLUTION_INSTRUCTIONS.format(
                    source_language=item.source_language,
                    target_language=self.target_language,
                    prompt=item.task.prompt.strip(),
                    solution=item.task.answer.strip(),
                )
                requests.append(
                    _chat_completion_request(
                        custom_id=item.solution_custom_id,
                        model=self.config.model,
                        user_content=solution_user,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        reasoning_effort=self.config.reasoning_effort,
                    )
                )
            if item.test_custom_id is not None:
                test_user = _TEST_INSTRUCTIONS.format(
                    source_language=item.source_language,
                    target_language=self.target_language,
                    entry_point=item.task.entry_point or "(unknown)",
                    prompt=item.task.prompt.strip(),
                    tests=item.task.tests.content.strip(),
                )
                requests.append(
                    _chat_completion_request(
                        custom_id=item.test_custom_id,
                        model=self.config.model,
                        user_content=test_user,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        reasoning_effort=self.config.reasoning_effort,
                    )
                )
        return requests

    def _wait_for_batch(self, client: Any, batch_id: str) -> Any:
        """Poll until the batch reaches a terminal status."""
        while True:
            batch = client.batches.retrieve(batch_id)
            counts = _request_counts_dict(batch)
            LOGGER.info(
                "Batch %s status=%s completed=%s failed=%s total=%s",
                batch_id,
                batch.status,
                counts.get("completed"),
                counts.get("failed"),
                counts.get("total"),
            )
            if batch.status in _TERMINAL_BATCH_STATUSES:
                if batch.status == "cancelling":
                    time.sleep(self.config.poll_interval_seconds)
                    continue
                return batch
            time.sleep(self.config.poll_interval_seconds)

    def _load_cache(self) -> None:
        assert self._cache_path is not None
        if not self._cache_path.exists():
            return
        try:
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            LOGGER.info(
                "Loaded %s cached prompt translations from %s",
                len(self._cache),
                self._cache_path,
            )
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "Failed to load translation cache %s: %s",
                self._cache_path,
                exc,
            )
            self._cache = {}

    def _dump_cache(self) -> None:
        assert self._cache_path is not None
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _chat_completion_request(
    *,
    custom_id: str,
    model: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _TRANSLATION_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        # GPT-5 family rejects legacy ``max_tokens``; use max_completion_tokens.
        "max_completion_tokens": max_tokens,
    }
    if reasoning_effort in (None, "none"):
        body["temperature"] = temperature
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": _BATCH_ENDPOINT,
        "body": body,
    }


def _custom_id(task: TaskInstance, kind: str) -> str:
    """
    Build a Batch API ``custom_id``.

    OpenAI requires unique ids; keep them filesystem-/JSON-safe and short.
    """
    digest = hashlib.sha1(
        f"{task.dataset}\0{task.task_id}\0{kind}".encode("utf-8")
    ).hexdigest()[:16]
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.dataset)[:32]
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.task_id)[:48]
    return f"{safe_dataset}__{safe_task}__{kind}__{digest}"


def _require_batch_text(outputs: Dict[str, str], custom_id: str) -> str:
    if custom_id not in outputs:
        raise KeyError(
            f"Missing Batch API result for custom_id={custom_id}. "
            "Check the batch output/error file under "
            "artifacts/translation_outputs/."
        )
    text = outputs[custom_id].strip()
    if not text:
        raise ValueError(f"Empty translation response for custom_id={custom_id}")
    return text


def _parse_batch_output(path: Path) -> Dict[str, str]:
    """Map ``custom_id`` -> assistant message content from a batch output JSONL."""
    results: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of batch output {path}"
                ) from exc
            custom_id = record.get("custom_id")
            if not custom_id:
                raise ValueError(
                    f"Batch output line {line_no} is missing custom_id: {path}"
                )
            if record.get("error"):
                raise RuntimeError(
                    f"Batch request custom_id={custom_id} failed: {record['error']}"
                )
            response = record.get("response") or {}
            status_code = response.get("status_code")
            body = response.get("body") or {}
            if status_code not in (None, 200):
                raise RuntimeError(
                    f"Batch request custom_id={custom_id} returned "
                    f"status_code={status_code}: {body}"
                )
            choices = body.get("choices") or []
            if not choices:
                raise RuntimeError(
                    f"Batch request custom_id={custom_id} has no choices: {body}"
                )
            message = choices[0].get("message") or {}
            content = message.get("content")
            if content is None:
                raise RuntimeError(
                    f"Batch request custom_id={custom_id} has empty message content"
                )
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                    elif isinstance(part, str):
                        parts.append(part)
                content = "".join(parts)
            results[str(custom_id)] = str(content)
    return results


def _download_file_content(client: Any, file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_response = client.files.content(file_id)
    text = getattr(file_response, "text", None)
    if text is None:
        text = file_response.read().decode("utf-8")
    destination.write_text(text, encoding="utf-8")
    LOGGER.info("Downloaded OpenAI file %s -> %s", file_id, destination)


def _request_counts_dict(batch: Any) -> Dict[str, Any]:
    counts = getattr(batch, "request_counts", None)
    if counts is None:
        return {}
    if isinstance(counts, dict):
        return counts
    return {
        "total": getattr(counts, "total", None),
        "completed": getattr(counts, "completed", None),
        "failed": getattr(counts, "failed", None),
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _infer_source_language(task: TaskInstance) -> str:
    meta_lang = task.meta.get("source_language")
    if isinstance(meta_lang, str) and meta_lang.strip():
        return meta_lang.strip()
    mapped = SOURCE_LANGUAGE_BY_DATASET.get(task.dataset.lower())
    if mapped:
        return mapped
    if task.language:
        return task.language
    return "python"


def _cache_key(
    task: TaskInstance,
    source_language: str,
    target_language: str,
    prompt: str,
    solution: str,
    tests: str,
    translate_solutions: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(_PROMPT_LAYOUT_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(task.dataset.encode("utf-8"))
    digest.update(b"\0")
    digest.update(task.task_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source_language.lower().encode("utf-8"))
    digest.update(b"\0")
    digest.update(target_language.lower().encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(tests.encode("utf-8"))
    if translate_solutions:
        digest.update(b"\0")
        digest.update(solution.encode("utf-8"))
    return digest.hexdigest()


def _clean_translation_output(text: str) -> str:
    cleaned = extract_code_snippet(text.strip())
    cleaned = re.sub(
        r"^(?:translated\s+prompt|translated\s+solution)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _display_prompt_pair(
    *,
    task_id: str,
    source_language: str,
    target_language: str,
    original: str,
    translated_prompt: str,
    original_tests: str = "",
    translated_tests: str = "",
    original_solution: str = "",
    translated_solution: str = "",
    skipped: bool = False,
    from_cache: bool = False,
) -> None:
    """Print original/translated prompts, tests, and solutions for a task."""
    status = (
        "skipped (same language)"
        if skipped
        else ("cached" if from_cache else "translated")
    )
    separator = "=" * 80
    parts = [
        f"\n{separator}",
        f"[{status}] {task_id}: {source_language} -> {target_language}",
        f"{'-' * 80}",
        "ORIGINAL PROMPT:",
        original.rstrip(),
        f"{'-' * 80}",
        "TRANSLATED PROMPT:",
        translated_prompt.rstrip(),
    ]
    if original_tests.strip() or translated_tests.strip():
        parts.extend(
            [
                f"{'-' * 80}",
                "ORIGINAL TESTS:",
                (original_tests.rstrip() or "(empty)"),
                f"{'-' * 80}",
                "TRANSLATED TESTS:",
                (translated_tests.rstrip() or "(empty)"),
            ]
        )
    if original_solution.strip() or translated_solution.strip():
        parts.extend(
            [
                f"{'-' * 80}",
                "ORIGINAL SOLUTION:",
                (original_solution.rstrip() or "(empty)"),
                f"{'-' * 80}",
                "TRANSLATED SOLUTION:",
                (translated_solution.rstrip() or "(empty)"),
            ]
        )
    parts.append(separator)
    parts.append("")
    tqdm.write("\n".join(parts))
    LOGGER.info(
        "Prompt translation %s for %s (%s -> %s)",
        status,
        task_id,
        source_language,
        target_language,
    )
