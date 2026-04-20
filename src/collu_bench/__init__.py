"""
Utilities for reproducing the Collu-Bench benchmark generation pipeline.

The package exposes helpers to load datasets, drive LLM sampling, normalize
programs, align hallucination tokens, and export Collu-Bench compatible CSVs.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - best effort metadata
    __version__ = version("collu_bench")
except PackageNotFoundError:  # pragma: no cover - local usage without install
    __version__ = "0.0.0"

