from __future__ import annotations

"""
Token-level labeling for generated Racket programs.

This module compares a model-generated Racket solution to the closest canonical
reference after both sides have been normalized.  The goal is not to prove
semantic equivalence; rather, it provides a stable token-level signal that can
be used in downstream error analysis and hallucination studies on Racket code.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Sequence

from .racket_normalization import RacketNormalizer


@dataclass
class RacketTokenLabelingResult:
    generated_code: str
    normalized_generated_code: str
    closest_gt_code: str
    normalized_closest_gt_code: str
    generated_tokens: List[str]
    closest_gt_tokens: List[str]
    token_labels: List[int]
    similarity: float


class RacketCodeTokenizer:
    """
    Lightweight Racket tokenizer for normalized code.

    The tokenizer mirrors the basic lexer rules used by the normalizer:
    - skips whitespace
    - skips line comments
    - keeps delimiters as separate tokens
    - keeps strings as atomic tokens
    - keeps atoms/operators/identifiers as atomic tokens
    """

    DELIMITERS = {"(", ")", "[", "]", "{", "}"}
    QUOTE_PREFIXES = {"'", "`", ","}

    def tokenize(self, code: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        n = len(code)

        while i < n:
            ch = code[i]

            if ch.isspace():
                i += 1
                continue

            if ch == ";":
                i = self._skip_comment(code, i)
                continue

            if ch in self.DELIMITERS:
                tokens.append(ch)
                i += 1
                continue

            if ch in self.QUOTE_PREFIXES:
                tokens.append(ch)
                i += 1
                continue

            if ch == '"':
                token, i = self._read_string(code, i)
                tokens.append(token)
                continue

            token, i = self._read_atom(code, i)
            if token:
                tokens.append(token)

        return tokens

    @staticmethod
    def _skip_comment(code: str, start: int) -> int:
        i = start
        n = len(code)
        while i < n and code[i] != "\n":
            i += 1
        return i

    @staticmethod
    def _read_string(code: str, start: int) -> tuple[str, int]:
        assert code[start] == '"'
        chars = ['"']
        i = start + 1
        n = len(code)
        escaped = False

        while i < n:
            ch = code[i]
            chars.append(ch)

            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                i += 1
                break

            i += 1

        return "".join(chars), i

    def _read_atom(self, code: str, start: int) -> tuple[str, int]:
        i = start
        n = len(code)
        chars: List[str] = []

        while i < n:
            ch = code[i]
            if ch.isspace() or ch == ";" or ch == '"' or ch in self.DELIMITERS or ch in self.QUOTE_PREFIXES:
                break
            chars.append(ch)
            i += 1

        return "".join(chars), i


class RacketTokenLabeler:
    """
    Token-level labeler for Racket generated code.

    Pipeline:
    1. Normalize generated code
    2. Normalize each canonical solution
    3. Tokenize both sides
    4. Pick the closest canonical solution using token-level similarity
    5. Label each generated token:
       - 0 if aligned as equal
       - 1 if part of replace/delete region

    Because the alignment is sequence-based rather than AST-based, the labels
    should be interpreted as a reproducible approximation of token mismatch, not
    as a formal diagnosis of semantic error in a Racket program.
    """

    def __init__(self):
        self.normalizer = RacketNormalizer()
        self.tokenizer = RacketCodeTokenizer()

    def label_against_canonicals(
        self,
        generated_code: str,
        canonical_solutions: Sequence[str],
    ) -> RacketTokenLabelingResult:
        if not canonical_solutions:
            raise ValueError("canonical_solutions must contain at least one solution")

        normalized_generated = self.normalizer.normalize(generated_code or "")
        generated_tokens = self.tokenizer.tokenize(normalized_generated.normalized_code)

        best_similarity = -1.0
        best_canonical_raw = ""
        best_canonical_normalized = ""
        best_canonical_tokens: List[str] = []

        for canonical in canonical_solutions:
            normalized_canonical = self.normalizer.normalize(canonical or "")
            canonical_tokens = self.tokenizer.tokenize(normalized_canonical.normalized_code)

            similarity = self._token_similarity(generated_tokens, canonical_tokens)
            if similarity > best_similarity:
                best_similarity = similarity
                best_canonical_raw = canonical
                best_canonical_normalized = normalized_canonical.normalized_code
                best_canonical_tokens = canonical_tokens

        labels = self._label_generated_tokens(
            generated_tokens=generated_tokens,
            reference_tokens=best_canonical_tokens,
        )

        return RacketTokenLabelingResult(
            generated_code=generated_code,
            normalized_generated_code=normalized_generated.normalized_code,
            closest_gt_code=best_canonical_raw,
            normalized_closest_gt_code=best_canonical_normalized,
            generated_tokens=generated_tokens,
            closest_gt_tokens=best_canonical_tokens,
            token_labels=labels,
            similarity=best_similarity,
        )

    @staticmethod
    def _token_similarity(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
        return SequenceMatcher(a=list(tokens_a), b=list(tokens_b)).ratio()

    @staticmethod
    def _label_generated_tokens(
        generated_tokens: Sequence[str],
        reference_tokens: Sequence[str],
    ) -> List[int]:
        labels = [0] * len(generated_tokens)
        matcher = SequenceMatcher(a=list(generated_tokens), b=list(reference_tokens))

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            if tag in {"replace", "delete"}:
                for idx in range(i1, i2):
                    labels[idx] = 1

            elif tag == "insert":
                # Tokens exist in reference but not in generated.
                # Since we annotate only generated tokens, there is no direct
                # generated token span to mark here in the baseline version.
                continue

        return labels


def format_labeling_result(result: RacketTokenLabelingResult) -> str:
    """Pretty-print a labeling result for manual inspection."""
    lines: List[str] = []

    lines.append("=" * 80)
    lines.append("NORMALIZED GENERATED CODE")
    lines.append("=" * 80)
    lines.append(result.normalized_generated_code.rstrip())

    lines.append("")
    lines.append("=" * 80)
    lines.append("NORMALIZED CLOSEST GROUND TRUTH")
    lines.append("=" * 80)
    lines.append(result.normalized_closest_gt_code.rstrip())

    lines.append("")
    lines.append("=" * 80)
    lines.append("SIMILARITY")
    lines.append("=" * 80)
    lines.append(f"{result.similarity:.6f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("TOKEN LABELS")
    lines.append("=" * 80)

    for idx, (token, label) in enumerate(zip(result.generated_tokens, result.token_labels)):
        lines.append(f"{idx:03d} | {label} | {token}")

    return "\n".join(lines)
