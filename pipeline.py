#!/usr/bin/env python3
"""Convenience wrapper to run the Collu-Bench pipeline from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from collu_bench.pipeline import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()

