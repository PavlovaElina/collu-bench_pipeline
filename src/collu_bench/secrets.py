from __future__ import annotations

"""
Secure loading of the OpenAI API key from a local file.

The key file is intentionally kept outside version control (see
``secrets/openai_api_key.example``). ``OPENAI_API_KEY`` may override the file
when set in the environment.
"""

import os
from pathlib import Path
from typing import List, Optional


_PLACEHOLDER_MARKERS = (
    "REPLACE_WITH_YOUR_OPENAI_API_KEY",
    "YOUR_OPENAI_API_KEY",
    "sk-...",
)


def _candidate_paths(
    api_key_path: str | Path,
    *,
    repo_root: Optional[Path] = None,
) -> List[Path]:
    path = Path(api_key_path).expanduser()
    if path.is_absolute():
        return [path.resolve()]

    candidates: List[Path] = [(Path.cwd() / path).resolve()]
    if repo_root is not None:
        root = Path(repo_root).resolve()
        candidates.append((root / path).resolve())
        # Configs usually live under ``configs/``; also check the package root.
        candidates.append((root.parent / path).resolve())

    # Preserve order while dropping duplicates.
    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def load_openai_api_key(
    api_key_path: str | Path,
    *,
    repo_root: Optional[Path] = None,
) -> str:
    """
    Load the OpenAI API key.

    Resolution order:
    1. ``OPENAI_API_KEY`` environment variable (if non-empty)
    2. Contents of ``api_key_path``, searched relative to the current working
       directory, then ``repo_root``, then ``repo_root``'s parent (package root
       when configs live under ``configs/``)
    """
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    candidates = _candidate_paths(api_key_path, repo_root=repo_root)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"OpenAI API key file not found (searched: {searched}). "
            "Copy secrets/openai_api_key.example to secrets/openai_api_key "
            "and paste your key, or set OPENAI_API_KEY."
        )

    raw = path.read_text(encoding="utf-8")
    # Allow comment lines starting with '#'; take the first non-empty non-comment line.
    key = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key = stripped
        break

    if not key or key in _PLACEHOLDER_MARKERS:
        raise ValueError(
            f"OpenAI API key file {path} is empty or still contains a placeholder. "
            "Replace it with a real key, or set OPENAI_API_KEY."
        )
    return key


def resolve_workdir_path(
    path_value: str | Path,
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Resolve an artifact path the same way as other pipeline outputs.

    Relative paths prefer the current working directory (typical when running
    ``python3 pipeline.py`` from the package root). If that location does not
    exist yet and ``repo_root`` is the ``configs/`` directory, fall back to the
    package root so ``artifacts/`` and ``secrets/`` land next to ``pipeline.py``.
    """
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists() or repo_root is None:
        return cwd_candidate

    root = Path(repo_root).resolve()
    package_root = root.parent if root.name == "configs" else root
    return (package_root / path).resolve()
