from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .canonical import CanonicalCollector, CanonicalRepository
from .config import LLMConfig, PipelineConfig, load_config
from .data import TaskInstance, load_dataset
from .execution import ExecutionRunner
from .hallucination import HallucinationLocator
from .llm import LocalHFClient
from .normalization import NormalizerRegistry
from .prompt import PromptBuilder
from .storage import ColluRecord, StorageWriter
from .token_types import TokenTypeAnnotator
from .utils import extract_code_snippet

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collu-Bench pipeline runner")
    parser.add_argument("--config", required=True, help="Path to the pipeline YAML config")
    parser.add_argument("--output", help="Override the output csv path")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run_pipeline(config_path: Path, override_output: str | None = None) -> None:
    config = load_config(config_path)
    if override_output:
        config.output_csv = override_output
    repo_root = config_path.parent
    dataset_tasks: Dict[str, List[TaskInstance]] = {}
    prompt_builders: Dict[str, PromptBuilder] = {}
    for dataset_cfg in config.datasets:
        tasks = load_dataset(dataset_cfg, repo_root)
        dataset_tasks[dataset_cfg.name] = tasks
        prompt_builders[dataset_cfg.name] = PromptBuilder(dataset_cfg, repo_root)
        LOGGER.info("Loaded %s tasks for %s", len(tasks), dataset_cfg.name)

    normalizers = NormalizerRegistry()
    execution_runner = ExecutionRunner(config.execution_timeout, Path(config.workspace))
    token_annotator = TokenTypeAnnotator(normalizers)
    canonical_repo = CanonicalRepository()
    cache_path = (
        Path(config.canonical_sampling.cache_path)
        if config.canonical_sampling.cache_path
        else None
    )
    if cache_path:
        canonical_repo.load(cache_path)
        LOGGER.info("Loaded cached canonical solutions (%s entries)", canonical_repo.total())

    canonical_collector = CanonicalCollector(
        repo=canonical_repo,
        normalizers=normalizers,
        executor=execution_runner,
        dataset_builders=prompt_builders,
        config=config.canonical_sampling,
    )
    for dataset_cfg in config.datasets:
        canonical_collector.seed_with_dataset(dataset_tasks[dataset_cfg.name])

    all_model_configs: Dict[str, LLMConfig] = {
        cfg.name: cfg for cfg in config.eval_models
    }
    for sampler_cfg in config.canonical_sampling.sampler_models:
        all_model_configs[sampler_cfg.name] = sampler_cfg
    LOGGER.info(f"Loaded configs: {all_model_configs}")
    model_clients = {name: LocalHFClient(cfg) for name, cfg in all_model_configs.items()}

    sampler_cfgs = config.canonical_sampling.sampler_models or config.eval_models
    sampler_clients = [model_clients[cfg.name] for cfg in sampler_cfgs]
    LOGGER.info(f"Loaded sampler clients {sampler_clients} ")
    for dataset_cfg in config.datasets:
        if not dataset_cfg.extra.get("sample_canonical", True):
            continue
        canonical_collector.collect(
            dataset_tasks[dataset_cfg.name],
            sampler_clients,
            dataset_cfg,
        )
    if cache_path:
        canonical_repo.dump(cache_path)

    locator = HallucinationLocator(normalizers)
    writer = StorageWriter(Path(config.output_csv))
    idx_counter = 0
    LOGGER.info(f"Running {len(config.eval_models)} models")
    for dataset_cfg in config.datasets:
        tasks = dataset_tasks[dataset_cfg.name]
        builder = prompt_builders[dataset_cfg.name]
        for model_cfg in config.eval_models:
            client = model_clients[model_cfg.name]
            LOGGER.info(
                "Generating for dataset=%s model=%s (%s tasks)",
                dataset_cfg.name,
                model_cfg.name,
                len(tasks),
            )
            task_iterator = tqdm(
                tasks,
                desc=f"{dataset_cfg.name}/{model_cfg.name}",
                leave=False,
            )
            for task in task_iterator:
                try:
                    prompt = builder.build(task)
                    generation = client.generate(
                        prompt, request_id=f"{model_cfg.name}-{task.dataset}-{task.task_id}"
                    )
                    code = extract_code_snippet(generation.text)
                    execution = execution_runner.run(task, code)
                    meta_payload = {**task.meta, "raw_completion": generation.text}
                    canonical_records = canonical_repo.get(task)
                    hallucination = locator.locate(
                        task, code, canonical_records, generation.tokens
                    )
                    token_types = token_annotator.annotate(task.language, code, generation.tokens)
                    
                    writer.append(
                        ColluRecord(
                            idx=idx_counter,
                            model=model_cfg.name,
                            dataset=task.dataset,
                            task_id=task.task_id,
                            meta=meta_payload,
                            model_output=code,
                            closest_gt=hallucination.closest_solution if hallucination else "",
                            hallucination_token_index=(
                                hallucination.token_index if hallucination else None
                            ),
                            tokens=generation.tokens,
                            token_types=token_types,
                            token_logprobs=generation.token_logprobs,
                            execution=execution,
                            question=task.question,
                            answer=task.answer,
                        )
                    )
                    idx_counter += 1
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.exception(
                        "Failed to process %s/%s with model %s: %s",
                        task.dataset,
                        task.task_id,
                        model_cfg.name,
                        exc,
                    )
        writer.write()
        LOGGER.info("Wrote %s rows to %s", len(writer.records), config.output_csv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print(f"Running with args: {args}")
    run_pipeline(Path(args.config).expanduser(), args.output)


if __name__ == "__main__":
    main()
