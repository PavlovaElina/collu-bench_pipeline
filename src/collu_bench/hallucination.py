from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .canonical import CanonicalRecord
from .data import TaskInstance
from .normalization import NormalizerRegistry


@dataclass
class HallucinationResult:
    closest_solution: str
    token_index: Optional[int]
    char_index: Optional[int]


class HallucinationLocator:
    """Identify the first hallucinated token."""

    def __init__(self, normalizers: NormalizerRegistry):
        self.normalizers = normalizers

    def locate(
        self,
        task: TaskInstance,
        generated_code: str,
        canonical_records: Sequence[CanonicalRecord],
        tokens: Sequence[str],
    ) -> Optional[HallucinationResult]:
        if not canonical_records:
            return None
        normalizer = self.normalizers.for_language(task.language)
        candidate = normalizer.normalize(generated_code)
        best_idx = -1
        best_record = None
        for record in canonical_records:
            mismatch = _first_difference(candidate.text, record.normalized_text)
            if mismatch > best_idx:
                best_idx = mismatch
                best_record = record
        if best_record is None:
            return None
        char_index = None
        if candidate.mapping:
            target = min(best_idx, len(candidate.mapping) - 1)
            if target >= 0:
                char_index = candidate.mapping[target]
        token_index = _map_char_to_token(tokens, generated_code, char_index)
        return HallucinationResult(
            closest_solution=best_record.original,
            token_index=token_index,
            char_index=char_index,
        )


def _first_difference(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return idx
    return limit


def _map_char_to_token(tokens: Sequence[str], text: str, char_index: Optional[int]) -> Optional[int]:
    if char_index is None or char_index < 0:
        return None
    cursor = 0
    for idx, token in enumerate(tokens):
        cursor += len(token)
        if cursor > char_index:
            return idx
    return len(tokens) - 1 if tokens else None
