from __future__ import annotations

import re
from typing import Optional

CODE_BLOCK_RE = re.compile(r"```(?:[\w+-]+)?\n(.*?)```", re.DOTALL)


def extract_code_snippet(text: str) -> str:
    """Attempt to extract the executable portion from a completion."""

    if not text:
        return ""
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()

