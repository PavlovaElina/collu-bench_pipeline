# Collu-Bench Reproduction Pipeline

This repository contains a fully automated pipeline that recreates the benchmark
generation procedure described in *Collu-Bench: Fine-grained Hallucination
Benchmark for Code LLMs*
([arXiv:2410.09997](https://arxiv.org/pdf/2410.09997)). The pipeline mirrors
the paper's main stages:

1. **Dataset ingestion** - load source benchmarks with their prompts, reference
   solutions, and execution harnesses.
2. **Canonical solution expansion** - sample diverse, test-passing programs
   from multiple LLMs and normalize identifier choices to reduce spurious
   hallucination detections.
3. **LLM inference** - run target models with prompt templates, record decoded
   tokens and per-step logprobs, and store the raw responses.
4. **Hallucination localization** - normalize generated code, align it with the
   closest canonical program, and report the first hallucinated token index.
5. **Execution feedback** - run benchmark-specific harnesses or external
   scripts to record pass/fail/error messages.
6. **Dataset export** - emit a `collu-bench.csv` with the fields used by the
   paper.

The default configuration collects canonical solutions for HumanEval and MBPP
with EvalPlus, while APR-style datasets can be plugged in later through JSONL
descriptors that point to scripts capable of applying candidate patches and
running their test suites.

## Racket workflow

In addition to the Python- and Java-oriented benchmark paths, the repository
contains an experimental Racket workflow built around HumanEval-derived tasks.
The workflow starts from the official Python HumanEval dataset, rewrites each
task so that it requests a `#lang racket` implementation, translates supported
Python assertions into `rackunit` checks, and validates generated solutions
with the native `racket` executable.

The Racket toolchain currently covers four stages:

1. **Task translation** -
   `scripts/build_racket_from_humaneval_hybrid.py` rewrites prompts and
   produces a `racket_test_module` for each supported HumanEval task.
2. **Ground-truth generation** -
   `scripts/generate_racket_gt_from_dataset.py` samples candidate Racket
   solutions from a local Hugging Face model and keeps the first one that
   passes the translated tests.
3. **Pipeline-ready dataset assembly** -
   `scripts/build_pipeline_ready_racket_dataset.py` merges translated tasks
   with passing Racket ground truths into a generic JSONL dataset consumable by
   the main pipeline.
4. **Benchmark execution** - `pipeline.py` can evaluate target models on that
   Racket JSONL dataset, using the same CSV export and hallucination-analysis
   machinery as for the original benchmark.

At the code level, the Racket-specific support lives mainly in:

* `src/collu_bench/python_to_racket_tests.py` - deterministic prompt and test translation
* `src/collu_bench/racket_normalization.py` - lightweight normalization for generated Racket code
* `src/collu_bench/racket_labeling.py` - token-level comparison against canonical Racket solutions
* `src/collu_bench/token_types.py` - Racket tokenization and token-type annotation
* `scripts/run_racket_humaneval_tests.py` - native execution harness that runs `racket runner.rkt`

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> **Hardware:** Code generation happens locally via Hugging Face
> `transformers`. Ensure you have enough GPU memory, or set `device: cpu` for
> debugging, and make sure the required checkpoints are available on disk or
> through your Hugging Face cache.
>
> **Racket runtime:** The Racket workflow requires a local installation of
> Racket with the `racket` executable available in `PATH`, because translated
> tests are executed through `scripts/run_racket_humaneval_tests.py`.

## Configuration

All behavior is driven by a YAML config. The included `configs/example.yaml`
shows the expected structure: HumanEval and MBPP are loaded directly from
EvalPlus, and additional datasets can be hooked in through JSONL manifests that
describe how to execute tests.

```yaml
output_csv: artifacts/collu-bench.csv
execution_timeout: 120
workspace: artifacts/workspace
datasets:
  - name: humaneval
    source: humaneval
    task_type: cg
    prompt:
      prefix: >
        You are a senior Python engineer. Provide only valid Python code.
  - name: mbpp
    source: mbpp
    task_type: cg
    prompt:
      prefix: >
        Write a correct Python function that satisfies the following specification.
canonical_sampling:
  enabled: true
  samples_per_model: 100
  sampler_models:
    - name: llama3-8b-sampler
      model: meta-llama/Meta-Llama-3-8B-Instruct
      device: cuda
      dtype: float16
      temperature: 0.8
    - name: mistral-nemo-sampler
      model: mistralai/Mistral-Nemo-Instruct-2407
      device: cuda
      dtype: float16
      temperature: 0.8
eval_models:
  - name: llama3-8b
    model: meta-llama/Meta-Llama-3-8B-Instruct
    device: cuda
    dtype: float16
    logprobs: 35
  - name: mistral-nemo
    model: mistralai/Mistral-Nemo-Instruct-2407
    device: cuda
    dtype: float16
```

Key sections:

* **datasets** - declares each source benchmark. `source` may be
  `humaneval`, `mbpp`, or `jsonl`. JSONL datasets must provide prompt text,
  canonical solutions, test metadata, and an execution command, so you can plug
  in HumanEval-Java, Defects4J, SWE-bench, or custom Racket datasets.
* **prompt** - optional prefix and suffix strings added before or after each
  task prompt. Use these to enforce "only output code" instructions and keep
  the decoded tokens aligned with executable code.
* **canonical_sampling** - controls the optional step that samples diverse
  canonical programs per task. You can disable it or reduce
  `samples_per_model` when debugging.
* **eval_models** - target models used during benchmark creation. Each entry
  maps to a local Hugging Face checkpoint, either a path on disk via
  `local_model_path` or an online repo id in `model`, plus runtime overrides
  such as `device`, `dtype`, decoding temperature, and logprob depth.

For the Racket workflow, the main pipeline should be pointed at a JSONL dataset
whose rows explicitly declare `language: racket`. The repository already
contains example artifacts such as
`configs/data/racket_pipeline_ready_hybrid.jsonl` and a sample config
`configs/racket_pipeline_hybrid.yaml`.

## Running the pipeline

Run the default pipeline setup with:

```bash
python3 pipeline.py --config configs/example.yaml
```

To evaluate the prepared Racket dataset instead of the default mixed setup, run:

```bash
python3 pipeline.py --config configs/racket_pipeline_hybrid.yaml
```

Results are written as a semicolon-separated CSV, matching the Collu-Bench
schema, and contain:

| Field | Description |
| --- | --- |
| `idx` | Unique row id |
| `model` | Evaluated LLM |
| `dataset` / `task_id` | Source benchmark identifiers |
| `meta` | Task metadata plus raw model output before sanitization |
| `model_output` | Cleaned code fed into execution and comparison |
| `closest_gt` | Canonical solution with the largest shared prefix |
| `hallucination_token_index` | Index of the first hallucinated token |
| `tokens` / `token_logprobs` | Per-step decoding trace |
| `token_types` | Token categories aligned with each decoded token |
| `execution` | Pass/fail status, stdout, stderr, and command details |
| `question` / `answer` | Prompt and canonical answer for quick reference |

For Racket rows, `model_output`, `closest_gt`, and `answer` contain executable
`#lang racket` modules. Test execution is delegated to the external command
stored in the dataset row, which typically calls
`scripts/run_racket_humaneval_tests.py`.

## Building the Racket dataset

If you want to reproduce the full Racket preparation pipeline from raw
HumanEval, the repository includes a dedicated bootstrap script:

```bash
python scripts/bootstrap_racket_benchmark.py ^
  --humaneval-input configs/data/HumanEval.jsonl ^
  --hybrid-output configs/data/racket_from_humaneval_hybrid.jsonl ^
  --gt-output artifacts/racket_ground_truth_dataset_hybrid.jsonl ^
  --pipeline-output configs/data/racket_pipeline_ready_hybrid.jsonl ^
  --model deepseek-ai/deepseek-coder-6.7b-instruct ^
  --device cuda ^
  --dtype float16 ^
  --gt-attempts 15 ^
  --gt-temperature 0.4 ^
  --gt-top-p 0.95 ^
  --gt-max-new-tokens 384
```

This command performs the following sequence:

1. Builds a deterministic Racket-target dataset from HumanEval prompts and tests.
2. Generates candidate Racket ground-truth solutions with a local HF model.
3. Validates those candidates by executing translated `rackunit` suites through
   the local `racket` interpreter.
4. Produces a pipeline-ready JSONL file that can be passed to `pipeline.py`.

If GT generation is interrupted, use
`scripts/generate_racket_gt_from_dataset_resume.py` directly or add
`--resume-gt` to the bootstrap command to continue from already completed
tasks.

## Using local Hugging Face models

The example configs load Hugging Face checkpoints directly through
`transformers`, execute them on CUDA by default, and record per-token log
probabilities. Make sure:

1. You have the required weights available locally, for example after
   `huggingface-cli login` for gated models.
2. Your machine has enough GPU memory for the selected models; otherwise set
   `device: cpu`.
3. You pick an appropriate `dtype`, such as `float16` on GPUs or
   `bfloat16`/`float32` on CPU.

To swap in another open-source model, edit the `model` or `local_model_path`
fields in the config and rerun `python3 pipeline.py --config ...`. Multiple
models can coexist, and they will be loaded once per unique `name` entry in the
configuration.

## Extending to APR datasets

For HumanEval-Java, Defects4J, and SWE-bench:

1. Prepare a JSONL file where each line does not contain secrets but includes:
   `task_id`, `prompt`, `language`, `entry_point`,
   `canonical_solutions`, and `tests`.
2. Use `tests.kind: external_command` together with `tests.command` to define
   a shell command that applies the generated patch and runs the corresponding
   regression suite. You can use placeholders such as `{code_path}`,
   `{task_id}`, and `{dataset}`, or rely on the injected `COLLU_*`
   environment variables inside your script.
3. Point a dataset config to this JSONL with `source: jsonl`, for example
   `path: data/defects4j.jsonl`, `task_type: apr`, and `language: java`.
4. Toggle `extra.sample_canonical: false` for heavy APR datasets where
   collecting new canonical solutions is impractical.

The same JSONL mechanism is what allows the Racket workflow to integrate with
the rest of the pipeline after the HumanEval-to-Racket translation stage.

## Next steps

* Plug in additional prompt templates; few-shot instructions can live next to
  the config so you can reuse templates from the paper's appendix.
* Integrate storage backends other than CSV, for example Hugging Face datasets,
  by extending `StorageWriter`.
* Enable model-parallel inference by spawning multiple workers or using
  `device_map: auto` and `accelerate` when loading large checkpoints.

Feel free to adapt the config to your hardware and iteration speed. Once
satisfied, the pipeline can be pointed at the entire dataset suite to recreate a
full Collu-Bench-style benchmark, including the Racket extension provided in
this repository.
