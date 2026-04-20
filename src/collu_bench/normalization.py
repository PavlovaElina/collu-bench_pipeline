from __future__ import annotations

"""
Shared normalization interface for benchmark programs.

Python and Java currently use tree-sitter-backed normalization, while Racket is
adapted through a custom lexer-based normalizer.  This split exists because the
Racket workflow needs to stay tolerant to partially malformed generated code and
to preserve progress even when no full parser can be trusted to succeed.
"""

import keyword
from dataclasses import dataclass
from typing import Dict, List

from tree_sitter import Parser
from tree_sitter_languages import get_language

from .racket_normalization import RacketNormalizer


@dataclass
class NormalizedProgram:
    text: str
    mapping: List[int]


class ProgramNormalizer:
    def normalize(self, code: str) -> NormalizedProgram:
        raise NotImplementedError


class TreeSitterNormalizer(ProgramNormalizer):
    """Normalization that replaces identifiers and canonicalizes whitespace."""

    def __init__(self, language: str):
        self.language = language
        self.parser = Parser()
        tree_lang = get_language(language)
        if hasattr(self.parser, "set_language"):
            self.parser.set_language(tree_lang)
        else:  # backwards compat for newer tree-sitter releases
            self.parser.language = tree_lang
        self.reserved = _reserved_identifiers(language)
        self.ignored = {"self", "cls"} if language == "python" else set()

    def normalize(self, code: str) -> NormalizedProgram:
        text = code or ""
        replaced = self._replace_identifiers(text)
        return self._normalize_whitespace(replaced)

    def _replace_identifiers(self, code: str) -> NormalizedProgram:
        tree = self.parser.parse(code.encode("utf-8"))
        if not tree:
            return NormalizedProgram(code, list(range(len(code))))

        replacements: List[tuple[int, int, str]] = []
        mapping: Dict[str, str] = {}

        def visit(node) -> None:
            node_type = node.type
            if node_type == "identifier":
                token = code[node.start_byte : node.end_byte]
                if self._should_replace(token):
                    replacement = mapping.setdefault(token, f"v{len(mapping) + 1}")
                    replacements.append((node.start_byte, node.end_byte, replacement))
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        if not replacements:
            return NormalizedProgram(code, list(range(len(code))))

        replacements.sort()
        rebuilt, rebuilt_mapping = _rewrite_with_replacements(code, replacements)
        return NormalizedProgram(rebuilt, rebuilt_mapping)

    def _should_replace(self, identifier: str) -> bool:
        if not identifier:
            return False
        if identifier in self.reserved or identifier in self.ignored:
            return False
        if identifier.startswith("__") and identifier.endswith("__"):
            return False
        return identifier.isidentifier()

    def _normalize_whitespace(self, program: NormalizedProgram) -> NormalizedProgram:
        if self.language == "python":
            return _normalize_python_whitespace(program)
        if self.language == "java":
            return _normalize_java_whitespace(program)
        return program


class RacketProgramNormalizer(ProgramNormalizer):
    """
    Adapter that plugs the standalone Racket normalizer into the shared
    NormalizedProgram interface used by the rest of the pipeline.

    This adapter is what lets the rest of the benchmark treat Racket like any
    other supported language, even though the underlying implementation is not
    tree-sitter based.
    """

    def __init__(self):
        self.normalizer = RacketNormalizer()

    def normalize(self, code: str) -> NormalizedProgram:
        result = self.normalizer.normalize(code or "")
        normalized_text = result.normalized_code

        # For the first integration version we expose an identity-style mapping
        # over the normalized text. This is sufficient for canonical comparison
        # and selection of the closest solution. A richer original-to-normalized
        # mapping can be introduced later when we implement full token-level
        # alignment and labels.
        mapping = list(range(len(normalized_text)))

        return NormalizedProgram(
            text=normalized_text,
            mapping=mapping,
        )


class NormalizerRegistry:
    """Cache of normalizers keyed by language."""

    def __init__(self):
        self._cache: Dict[str, ProgramNormalizer] = {}

    def for_language(self, language: str) -> ProgramNormalizer:
        if language not in self._cache:
            if language == "racket":
                self._cache[language] = RacketProgramNormalizer()
            else:
                self._cache[language] = TreeSitterNormalizer(language)
        return self._cache[language]


def _reserved_identifiers(language: str) -> set[str]:
    if language == "python":
        return set(keyword.kwlist) | set(dir(__builtins__))

    if language == "java":
        return {
            "abstract",
            "assert",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extends",
            "final",
            "finally",
            "float",
            "for",
            "goto",
            "if",
            "implements",
            "import",
            "instanceof",
            "int",
            "interface",
            "long",
            "native",
            "new",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "short",
            "static",
            "strictfp",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "try",
            "void",
            "volatile",
            "while",
        }

    if language == "racket":
        return set()

    return set()


def _rewrite_with_replacements(
    code: str, replacements: List[tuple[int, int, str]]
) -> tuple[str, List[int]]:
    cursor = 0
    pieces: List[str] = []
    mapping: List[int] = []

    for start, end, text in replacements:
        if start < cursor:
            continue
        pieces.append(code[cursor:start])
        mapping.extend(range(cursor, start))
        pieces.append(text)
        mapping.extend([start] * len(text))
        cursor = end

    pieces.append(code[cursor:])
    mapping.extend(range(cursor, len(code)))
    rebuilt = "".join(pieces)
    return rebuilt, mapping


def _normalize_python_whitespace(program: NormalizedProgram) -> NormalizedProgram:
    text = program.text
    mapping = program.mapping

    builder: List[str] = []
    new_mapping: List[int] = []
    idx = 0

    while idx < len(text):
        line_end = text.find("\n", idx)
        if line_end == -1:
            line_end = len(text)

        line = text[idx:line_end]
        indent_len = len(line) - len(line.lstrip(" \t"))

        for offset in range(indent_len):
            builder.append(line[offset])
            new_mapping.append(mapping[idx + offset])

        for body_offset, ch in enumerate(line[indent_len:]):
            global_idx = idx + indent_len + body_offset
            if ch in {" ", "\t"}:
                continue
            builder.append(ch)
            new_mapping.append(mapping[global_idx])

        if line_end < len(text):
            builder.append("\n")
            new_mapping.append(mapping[line_end])

        idx = line_end + 1

    return NormalizedProgram("".join(builder), new_mapping)


def _normalize_java_whitespace(program: NormalizedProgram) -> NormalizedProgram:
    text = program.text
    mapping = program.mapping

    builder: List[str] = []
    new_mapping: List[int] = []

    for idx, ch in enumerate(text):
        if ch.isspace():
            continue
        builder.append(ch)
        new_mapping.append(mapping[idx])

    return NormalizedProgram("".join(builder), new_mapping)
