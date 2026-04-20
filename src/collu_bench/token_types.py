from __future__ import annotations

"""
Token-type annotation utilities for benchmark outputs.

Racket support is implemented with a small dedicated lexer and classifier
instead of tree-sitter.  That design keeps annotation available even when the
generated Racket code is only approximately well formed, which is common in
hallucination analysis pipelines.
"""

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Type

from tree_sitter import Node

from .normalization import NormalizerRegistry

LOGGER = logging.getLogger(__name__)

CATEGORY_KEYWORD = "Keyword"
CATEGORY_DELIMITER = "Delimiter"
CATEGORY_OPERATOR = "Operator"
CATEGORY_CONSTANT = "Constant"
CATEGORY_IDENTIFIER = "Identifier"
CATEGORY_TYPE_IDENTIFIER = "Type Identifier"
CATEGORY_SPACE = "Space"
CATEGORY_EOS = "<EOS>"
CATEGORY_UNKNOWN = "Unknown"

DATASET_LANGUAGE_MAP = {
    "humaneval": "python",
    "mbpp": "python",
    "humaneval_java": "java",
    "defects4j": "java",
    "swe-bench": "python",
    "swebench": "python",
    "racket_codegen": "racket",
    "racket": "racket",
}

JAVA_KEYWORDS = {
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

PYTHON_KEYWORDS = {
    "false",
    "none",
    "true",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}

RACKET_KEYWORDS = {
    "#lang",
    "racket",
    "define",
    "define-values",
    "define/contract",
    "lambda",
    "λ",
    "if",
    "cond",
    "else",
    "let",
    "let*",
    "letrec",
    "let-values",
    "let*-values",
    "begin",
    "begin0",
    "set!",
    "quote",
    "quasiquote",
    "unquote",
    "unquote-splicing",
    "match",
    "case",
    "and",
    "or",
    "when",
    "unless",
    "for",
    "for*",
    "for/list",
    "for/vector",
    "for/hash",
    "for/first",
    "for/last",
    "for/sum",
    "for/product",
    "for/fold",
    "require",
    "provide",
    "module",
    "module+",
    "struct",
    "define-struct",
    "parameterize",
    "values",
}

RACKET_BUILTINS = {
    "null?",
    "empty?",
    "pair?",
    "list?",
    "cons",
    "car",
    "cdr",
    "caar",
    "cadr",
    "cdar",
    "cddr",
    "list",
    "append",
    "reverse",
    "length",
    "map",
    "filter",
    "foldl",
    "foldr",
    "member",
    "memq",
    "memv",
    "apply",
    "andmap",
    "ormap",
    "build-list",
    "list-ref",
    "take",
    "drop",
    "string?",
    "string-length",
    "string-ref",
    "substring",
    "string-append",
    "string->list",
    "list->string",
    "symbol?",
    "symbol->string",
    "string->symbol",
    "number?",
    "integer?",
    "real?",
    "zero?",
    "positive?",
    "negative?",
    "odd?",
    "even?",
    "add1",
    "sub1",
    "sort",
    "min",
    "max",
    "abs",
    "modulo",
    "remainder",
    "quotient",
    "floor",
    "ceiling",
    "round",
    "expt",
    "sqrt",
    "in-range",
    "display",
    "displayln",
    "newline",
    "printf",
    "format",
    "error",
    "void",
}

KEYWORDS_BY_LANGUAGE = {
    "python": PYTHON_KEYWORDS,
    "java": JAVA_KEYWORDS,
    "racket": RACKET_KEYWORDS,
}

TYPE_KEYWORDS = {
    "boolean",
    "byte",
    "char",
    "double",
    "float",
    "int",
    "long",
    "short",
    "void",
}

BOOLEAN_LITERALS = {"true", "false", "#t", "#f"}
NULL_LITERALS = {"null", "none", "empty"}

IDENTIFIER_NODE_TYPES = {"identifier"}
TYPE_IDENTIFIER_NODE_TYPES = {"type_identifier", "void_type"}

STRING_NODE_TYPES = {
    "string_fragment",
    "multiline_string_fragment",
    "string_start",
    "string_end",
    "string_content",
    "\"",
    '"""',
}

CONSTANT_NODE_TYPES = {
    "decimal_integer_literal",
    "decimal_floating_point_literal",
    "integer",
    "character_literal",
    "escape_sequence",
    "line_comment",
}

DELIMITER_TOKENS = {
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ":",
    ",",
    ".",
    ";",
    "::",
    "\"",
    '"""',
    "'",
    "`",
}

OPERATOR_TOKENS = {
    "+",
    "-",
    "*",
    "**",
    "/",
    "//",
    "%",
    "+=",
    "-=",
    "*=",
    "/=",
    "//=",
    "=",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "<<",
    ">>",
    ">>>",
    "&",
    "|",
    "^",
    "&&",
    "||",
    "->",
    "--",
    "++",
    "is",
    "is not",
    "in",
    "not in",
    "and",
    "or",
    "not",
}

RACKET_OPERATOR_TOKENS = {
    "+",
    "-",
    "*",
    "/",
    "=",
    "<",
    "<=",
    ">",
    ">=",
}

RACKET_DELIMITERS = {"(", ")", "[", "]", "{", "}", "'", "`", ","}
RACKET_QUOTE_PREFIXES = {"'", "`", ","}
_RACKET_IDENTIFIER_REGEX = re.compile(r"^[^\d\W][\w!?+\-*/<>=:$%&~^.@]*$", re.UNICODE)


@dataclass
class CodeToken:
    text: str
    kind: str
    start: int
    end: int


def _normalize_language_name(language: str | None) -> str:
    if not language:
        return ""
    return language.strip().lower()


class TokenTypeAnnotator:
    """Annotate tokens with their corresponding token categories."""

    def __init__(
        self,
        normalizers: NormalizerRegistry | Type[NormalizerRegistry] | None = None,
    ):
        if normalizers is None:
            self.normalizers = NormalizerRegistry()
        elif isinstance(normalizers, NormalizerRegistry):
            self.normalizers = normalizers
        elif isinstance(normalizers, type) and issubclass(normalizers, NormalizerRegistry):
            self.normalizers = normalizers()
        else:
            raise TypeError(
                "TokenTypeAnnotator requires a NormalizerRegistry instance or class"
            )

    def annotate(self, language: str, code: str, tokens: Sequence[str]) -> List[str]:
        if not tokens:
            return []

        normalized_language = _normalize_language_name(language)
        spans = self._collect_ast_tokens(normalized_language, code)
        if spans is None:
            return [CATEGORY_UNKNOWN] * len(tokens)

        token_types: List[str] = []
        node_index = 0
        char_cursor = 0

        for token in tokens:
            token_length = len(token)
            start = char_cursor
            end = start + token_length
            char_cursor = end

            while node_index < len(spans) and spans[node_index].end <= start:
                node_index += 1

            kind = CATEGORY_UNKNOWN
            if node_index < len(spans):
                node = spans[node_index]
                if node.start < end and node.end > start:
                    kind = node.kind
            token_types.append(kind)

        return token_types

    def tokenize_code(self, language: str, code: str) -> List[CodeToken]:
        normalized_language = _normalize_language_name(language)
        spans = self._collect_ast_tokens(normalized_language, code)
        tokens: List[CodeToken] = []

        if spans is None:
            if code:
                tokens.append(CodeToken(code, CATEGORY_UNKNOWN, 0, len(code)))
            tokens.append(CodeToken(CATEGORY_EOS, CATEGORY_EOS, len(code), len(code)))
            return tokens

        cursor = 0
        for span in spans:
            if span.start > cursor:
                fragment = code[cursor:span.start]
                if fragment:
                    tokens.append(CodeToken(fragment, CATEGORY_SPACE, cursor, span.start))
            tokens.append(span)
            cursor = span.end

        if cursor < len(code):
            tokens.append(CodeToken(code[cursor:], CATEGORY_SPACE, cursor, len(code)))

        tokens.append(CodeToken(CATEGORY_EOS, CATEGORY_EOS, len(code), len(code)))
        return tokens

    def _collect_ast_tokens(self, language: str, code: str) -> List[CodeToken] | None:
        if not language:
            return None

        if language == "racket":
            return _collect_racket_tokens(code or "")

        parser_text, parser_mapping = _prepare_code_for_parser(code or "")
        try:
            normalizer = self.normalizers.for_language(language)
            parser = getattr(normalizer, "parser", None)
            if parser is None:
                return None
            tree = parser.parse(parser_text.encode("utf-8"))
        except Exception as exc:
            LOGGER.debug("Error parsing code for language=%s: %s", language, exc)
            return None

        if not tree or not tree.root_node:
            return None

        leaf_nodes = _collect_leaf_nodes(tree.root_node)
        if not leaf_nodes:
            return None

        byte_to_char = _build_byte_to_char_map(parser_text)
        spans: List[CodeToken] = []

        for node in leaf_nodes:
            if node.end_byte <= node.start_byte:
                continue

            start_norm = _byte_to_char(byte_to_char, node.start_byte)
            end_norm = _byte_to_char(byte_to_char, node.end_byte)

            if start_norm >= len(parser_mapping):
                start_norm = len(parser_mapping) - 1
            if end_norm >= len(parser_mapping):
                end_norm = len(parser_mapping) - 1

            start = parser_mapping[start_norm]
            end = parser_mapping[end_norm]
            if end <= start:
                continue

            text = code[start:end]
            kind = _classify_token(language, node.type, text)
            spans.append(CodeToken(text, kind, start, end))

        return spans


def annotate_csv_file(
    csv_path: Path,
    annotator: TokenTypeAnnotator,
    *,
    delimiter: str = ";",
    code_column: str = "model_output",
    language_column: str | None = None,
    dataset_column: str | None = "dataset",
    meta_column: str | None = "meta",
    output_path: Path | None = None,
    output_column: str = "token_types",
    default_language: str | None = None,
) -> Path:
    csv.field_size_limit(sys.maxsize)
    target_path = output_path or csv_path.with_suffix(".token_types.csv")

    with csv_path.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        if output_column not in fieldnames:
            fieldnames.append(output_column)

        with target_path.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()

            for row in reader:
                language = _infer_language(
                    row,
                    language_column=language_column,
                    dataset_column=dataset_column,
                    meta_column=meta_column,
                    default_language=default_language,
                )
                code = row.get(code_column, "") or ""

                if language:
                    tokens = annotator.tokenize_code(language, code)
                else:
                    LOGGER.warning(
                        "Skipping row idx=%s due to unknown language",
                        row.get("idx"),
                    )
                    tokens = [
                        CodeToken(code, CATEGORY_UNKNOWN, 0, len(code)),
                        CodeToken(CATEGORY_EOS, CATEGORY_EOS, len(code), len(code)),
                    ]

                row[output_column] = json.dumps(
                    [{"token": tok.text, "type": tok.kind} for tok in tokens],
                    ensure_ascii=False,
                )
                writer.writerow(row)

    return target_path


def _infer_language(
    row: Dict[str, str],
    *,
    language_column: str | None,
    dataset_column: str | None,
    meta_column: str | None,
    default_language: str | None,
) -> str | None:
    default_language = _normalize_language_name(default_language)

    if language_column:
        language = (row.get(language_column) or "").strip().lower()
        if language:
            return language

    if meta_column and row.get(meta_column):
        try:
            meta = json.loads(row[meta_column])
            if isinstance(meta, dict):
                language = meta.get("language")
                if isinstance(language, str) and language:
                    return language.lower()
        except json.JSONDecodeError:
            LOGGER.debug("Failed to parse meta JSON for row idx=%s", row.get("idx"))

    if dataset_column:
        dataset = (row.get(dataset_column) or "").lower()
        for key, language in DATASET_LANGUAGE_MAP.items():
            if key in dataset:
                return language

    return default_language


def _classify_token(language: str, node_type: str, text: str) -> str:
    normalized_language = _normalize_language_name(language)
    normalized_type = (node_type or "").lower()
    stripped = text.strip()
    lowered_text = stripped.lower()
    keyword_set = KEYWORDS_BY_LANGUAGE.get(normalized_language, set())

    if normalized_language == "racket":
        return _classify_racket_token(stripped)

    if normalized_type == "error":
        return CATEGORY_UNKNOWN

    if normalized_type in TYPE_IDENTIFIER_NODE_TYPES or stripped in TYPE_KEYWORDS:
        return CATEGORY_TYPE_IDENTIFIER

    if normalized_type in IDENTIFIER_NODE_TYPES:
        return CATEGORY_IDENTIFIER

    if normalized_type in OPERATOR_TOKENS or stripped in OPERATOR_TOKENS:
        return CATEGORY_OPERATOR

    if normalized_type in DELIMITER_TOKENS or text in DELIMITER_TOKENS:
        return CATEGORY_DELIMITER

    if lowered_text in BOOLEAN_LITERALS or lowered_text in NULL_LITERALS:
        return CATEGORY_CONSTANT

    if normalized_type in STRING_NODE_TYPES or normalized_type in CONSTANT_NODE_TYPES:
        return CATEGORY_CONSTANT

    if normalized_type.endswith("_literal"):
        return CATEGORY_CONSTANT

    if normalized_type in keyword_set or lowered_text in keyword_set:
        return CATEGORY_KEYWORD

    if lowered_text in TYPE_KEYWORDS:
        return CATEGORY_TYPE_IDENTIFIER

    if _looks_numeric(stripped):
        return CATEGORY_CONSTANT

    if stripped:
        if stripped[0].isalpha() or stripped[0] == "_":
            return CATEGORY_IDENTIFIER

    return CATEGORY_UNKNOWN


def _classify_racket_token(token: str) -> str:
    if not token:
        return CATEGORY_UNKNOWN

    if token in RACKET_DELIMITERS:
        return CATEGORY_DELIMITER

    if token in RACKET_OPERATOR_TOKENS:
        return CATEGORY_OPERATOR

    if token in RACKET_KEYWORDS:
        return CATEGORY_KEYWORD

    if token in RACKET_BUILTINS:
        return CATEGORY_IDENTIFIER

    lowered = token.lower()
    if lowered in BOOLEAN_LITERALS or lowered in NULL_LITERALS:
        return CATEGORY_CONSTANT

    if _is_racket_string(token):
        return CATEGORY_CONSTANT

    if _looks_numeric(token):
        return CATEGORY_CONSTANT

    if _looks_racket_character(token):
        return CATEGORY_CONSTANT

    if _looks_racket_identifier(token):
        return CATEGORY_IDENTIFIER

    return CATEGORY_UNKNOWN


def _is_racket_string(token: str) -> bool:
    return len(token) >= 2 and token.startswith('"') and token.endswith('"')


def _looks_racket_character(token: str) -> bool:
    return token.startswith("#\\") and len(token) > 2


def _looks_racket_identifier(token: str) -> bool:
    if not token:
        return False
    if token.startswith("#"):
        return False
    return bool(_RACKET_IDENTIFIER_REGEX.fullmatch(token))


def _looks_numeric(text: str) -> bool:
    if not text:
        return False
    try:
        float(text.replace("_", ""))
        return True
    except ValueError:
        return False


def _collect_leaf_nodes(node: Node) -> List[Node]:
    leaves: List[Node] = []

    def visit(current: Node) -> None:
        if current.child_count == 0:
            leaves.append(current)
            return
        for child in current.children:
            visit(child)

    visit(node)
    leaves.sort(key=lambda item: item.start_byte)
    return leaves


def _build_byte_to_char_map(text: str) -> List[int]:
    mapping: List[int] = []
    char_index = 0
    for ch in text:
        encoded = ch.encode("utf-8")
        mapping.extend([char_index] * len(encoded))
        char_index += 1
    mapping.append(char_index)
    return mapping


def _byte_to_char(mapping: List[int], byte_index: int) -> int:
    if byte_index < 0:
        return 0
    if byte_index >= len(mapping):
        return mapping[-1]
    return mapping[byte_index]


def _prepare_code_for_parser(code: str) -> tuple[str, List[int]]:
    text = code or ""
    normalized_chars: List[str] = []
    mapping: List[int] = [0]
    idx = 0
    length = len(text)

    while idx < length:
        ch = text[idx]
        if ch == "\r":
            if idx + 1 < length and text[idx + 1] == "\n":
                normalized_chars.append("\n")
                idx += 2
            else:
                normalized_chars.append("\n")
                idx += 1
        else:
            normalized_chars.append(ch)
            idx += 1
        mapping.append(idx)

    if not normalized_chars or normalized_chars[-1] != "\n":
        normalized_chars.append("\n")
        mapping.append(len(text))

    normalized = "".join(normalized_chars)
    return normalized, mapping


def _collect_racket_tokens(code: str) -> List[CodeToken]:
    """
    Lexer-style Racket token collector with character spans.

    This is intentionally lightweight and does not require tree-sitter.
    It is sufficient for token-type annotation and for keeping the pipeline
    operational on Racket tasks.

    In particular, the lexer is designed to preserve useful token boundaries
    for classic Racket surface syntax such as quoted lists, string literals, and
    semicolon comments, while remaining permissive about malformed input.
    """
    tokens: List[CodeToken] = []
    i = 0
    n = len(code)

    while i < n:
        ch = code[i]

        if ch.isspace():
            i += 1
            continue

        if ch == ";":
            i = _skip_racket_comment(code, i)
            continue

        if ch in RACKET_DELIMITERS:
            tokens.append(
                CodeToken(ch, _classify_racket_token(ch), i, i + 1)
            )
            i += 1
            continue

        if ch == '"':
            start = i
            token_text, i = _read_racket_string(code, i)
            tokens.append(
                CodeToken(token_text, _classify_racket_token(token_text), start, i)
            )
            continue

        start = i
        token_text, i = _read_racket_atom(code, i)
        if token_text:
            tokens.append(
                CodeToken(token_text, _classify_racket_token(token_text), start, i)
            )

    return tokens


def _skip_racket_comment(code: str, start: int) -> int:
    i = start
    n = len(code)
    while i < n and code[i] != "\n":
        i += 1
    return i


def _read_racket_string(code: str, start: int) -> tuple[str, int]:
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


def _read_racket_atom(code: str, start: int) -> tuple[str, int]:
    i = start
    n = len(code)
    chars: List[str] = []

    while i < n:
        ch = code[i]
        if ch.isspace() or ch == ";" or ch == '"' or ch in RACKET_DELIMITERS:
            break
        chars.append(ch)
        i += 1

    return "".join(chars), i


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate CSV rows with token types.")
    parser.add_argument("--csv", required=True, help="Path to the input semicolon-separated CSV.")
    parser.add_argument(
        "--output",
        help="Optional output CSV path. Defaults to creating <input>.token_types.csv",
    )
    parser.add_argument(
        "--delimiter",
        default=";",
        help="CSV delimiter (default: %(default)s)",
    )
    parser.add_argument(
        "--code-column",
        default="model_output",
        help="Name of the CSV column containing code (default: %(default)s)",
    )
    parser.add_argument(
        "--language-column",
        help="Optional column containing row-level language information.",
    )
    parser.add_argument(
        "--dataset-column",
        default="dataset",
        help="Column used to infer language when explicit information is missing (default: %(default)s)",
    )
    parser.add_argument(
        "--meta-column",
        default="meta",
        help="Column with JSON metadata that may contain a language field (default: %(default)s)",
    )
    parser.add_argument(
        "--output-column",
        default="token_types",
        help="Column where the serialized token annotations will be stored (default: %(default)s)",
    )
    parser.add_argument(
        "--default-language",
        help="Fallback language applied when a row does not provide one and it cannot be inferred.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (default: %(default)s)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    annotator = TokenTypeAnnotator()
    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    annotate_csv_file(
        csv_path,
        annotator,
        delimiter=args.delimiter,
        code_column=args.code_column,
        language_column=args.language_column,
        dataset_column=args.dataset_column,
        meta_column=args.meta_column,
        output_path=output_path,
        output_column=args.output_column,
        default_language=args.default_language,
    )


if __name__ == "__main__":
    main()
