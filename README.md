# Collu-Bench Reproduction Pipeline

This repository contains a fully automated pipeline that recreates the benchmark generation
procedure described in *Collu-Bench: Fine-grained Hallucination Benchmark for Code LLMs*
([arXiv:2410.09997](https://arxiv.org/pdf/2410.09997)). The pipeline mirrors the paper's steps:

1. **Dataset ingestion** – load the five source benchmarks (HumanEval, MBPP, HumanEval-Java,
   Defects4J, and SWE-bench) with their prompts, reference solutions, and execution harnesses.
2. **Canonical solution expansion** – sample diverse, test-passing programs from multiple LLMs and
   normalize identifier choices to prevent spurious hallucination detections.
3. **LLM inference** – run target models with few-shot prompts, record decoded tokens and
   per-step logprobs, and store the raw responses.
4. **Hallucination localization** – normalize code with Tree-sitter, align it with the closest
   canonical program, and report the first hallucinated token index.
5. **Execution feedback** – run EvalPlus-style harnesses (or dataset-provided scripts) to record
   pass/fail/error messages.
6. **Dataset export** – emit a `collu-bench.csv` with the fields demanded by the paper.

The default configuration collects canonical solutions for HumanEval and MBPP with EvalPlus,
while APR datasets can be plugged in later through JSONL descriptors that point to scripts capable
of applying candidate patches and running their test suites.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> **Hardware:** Code generation happens locally via Hugging Face `transformers`. Ensure you have
> enough GPU memory (or set `device: cpu` for debugging) and that the required checkpoints are
> available on disk or through your Hugging Face cache.

## Configuration

All behaviour is driven by a YAML config. The included `configs/example.yaml` showcases the expected
structure – HumanEval & MBPP are loaded directly from EvalPlus, and additional APR datasets can be
hooked in through JSONL manifests that describe how to execute tests.

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

* **datasets** – declares each source benchmark. `source` may be `humaneval`, `mbpp`, or `jsonl`.
  JSONL datasets must provide `prompt`, `canonical_solutions`, test metadata, and the execution
  command (so you can plug HumanEval-Java, Defects4J, and SWE-bench).
* **prompt** – optional prefix/suffix strings added before/after each task prompt. Use these to
  enforce “only output code” instructions to keep the tokens aligned with executable code.
* **canonical_sampling** – controls the optional step that samples 100 diverse canonical programs
  per task (following the paper). You can disable it or reduce `samples_per_model` when debugging.
* **eval_models** – target models used during benchmark creation. Each entry maps to a local
  Hugging Face checkpoint (either a path on disk via `local_model_path` or an online repo id in
  `model`) plus runtime overrides such as `device`, `dtype`, decoding temperature, etc.

## Running the pipeline

```bash
python3 pipeline.py --config configs/example.yaml
```

Results are written as a semicolon-separated CSV (matching the official Collu-Bench schema) and
contain:

| Field | Description |
| --- | --- |
| `idx` | Unique row id |
| `model` | Evaluated LLM |
| `dataset` / `task_id` | Source benchmark identifiers |
| `meta` | Task metadata + raw model output before sanitization |
| `model_output` | Cleaned code fed into execution / comparison |
| `closest_gt` | Canonical solution with the largest shared prefix |
| `hallucination_token_index` | Index of the first hallucinated token |
| `tokens` / `token_logprobs` | Per-step decoding trace |
| `token_types` | AST leaf node type aligned with each decoded token |
| `execution` | Pass/fail, stdout, stderr, and command details |
| `question` / `answer` | Prompt and canonical answer for quick reference |

## Using local Hugging Face models

The example config loads Hugging Face checkpoints directly through `transformers`, executes them on
CUDA by default, and records per-token log probabilities. Make sure:

1. You have the required weights available locally (e.g., `huggingface-cli login` for gated models).
2. Your machine has enough GPU memory for the selected models; otherwise set `device: cpu`.
3. You pick an appropriate `dtype` (e.g., `float16` on GPUs, `bfloat16`/`float32` on CPU).

To swap in another open-source model, simply edit the `model` / `local_model_path` fields in the
config and rerun `python3 pipeline.py --config ...`. Multiple models can coexist, and they will be
loaded once per unique `name` entry in the configuration.

## Extending to APR datasets

For HumanEval-Java, Defects4J, and SWE-bench:

1. Prepare a JSONL file where each line does **not** contain secrets but includes:
   * `task_id`, `prompt`, `language`, `entry_point`
   * `canonical_solutions` – developer fixes or curated good patches
   * `tests.kind: external_command`, `tests.command` – shell command that applies the generated
     patch and runs the corresponding regression suite. Use placeholders `{code_path}`,
     `{task_id}`, `{dataset}` which the pipeline replaces per sample, or rely on the injected
     `COLLU_*` environment variables inside your script.
2. Point a dataset config to this JSONL (`source: jsonl`, `path: data/defects4j.jsonl`,
   `task_type: apr`, `language: java`).
3. Toggle `extra.sample_canonical: false` for heavy APR datasets where collecting new canonical
   solutions is impractical (matching the paper).

## Next steps

* Plug in additional prompt templates (few-shot instructions live next to the config so you can
  reuse the ones described in Appendix A of the paper).
* Integrate storage backends other than CSV (e.g., Hugging Face datasets) by extending
  `StorageWriter`.
* Enable model-parallel inference by spawning multiple workers or using `device_map: auto` /
  `accelerate` when loading large checkpoints.

Feel free to adapt the config to your API provider limits or to only run subsets of the benchmark
while iterating. Once satisfied, the pipeline can be pointed at the entire dataset suite to recreate
the 13,234-row Collu-Bench benchmark.
