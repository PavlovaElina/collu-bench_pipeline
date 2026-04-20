from __future__ import annotations

"""
Normalization utilities specialized for generated Racket code.

The broader benchmark uses normalization before comparing candidate solutions to
canonical references.  Racket is handled with a lightweight lexer rather than a
full parser because benchmark outputs often contain incomplete programs,
hallucinated prose, or minor syntax damage.  A tolerant lexical normalizer is
therefore more robust than an AST-based pass for this stage of the pipeline.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


RACKET_RESERVED_WORDS = {
    "#lang",
    "racket",
    "define",
    "define/contract",
    "define-values",
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
    "for* /fold".replace(" ", ""),
    "require",
    "provide",
    "module",
    "module+",
    "struct",
    "define-struct",
    "parameterize",
    "values",
    "call/cc",
    "call-with-current-continuation",
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
    "+",
    "-",
    "*",
    "/",
    "=",
    "<",
    ">",
    "<=",
    ">=",
    "not",
    "equal?",
    "eq?",
    "eqv?",
    "first",
    "second",
    "third",
    "rest",
    "empty",
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

RACKET_BOOLEAN_AND_LITERAL_WORDS = {
    "#t",
    "#f",
    "true",
    "false",
    "null",
}

DELIMITERS = {"(", ")", "[", "]", "{", "}"}
QUOTE_PREFIXES = {"'", "`", ","}


@dataclass
class RacketNormalizationResult:
    original_code: str
    normalized_code: str
    identifier_map: Dict[str, str] = field(default_factory=dict)
    tokens: List[str] = field(default_factory=list)


class RacketNormalizer:
    """
    A lightweight Racket code normalizer.

    This version is lexer-based, not AST-based. It is designed to:
    - normalize newlines
    - ensure '#lang racket' header
    - remove comments
    - collapse redundant whitespace
    - canonicalize user-defined identifiers

    It intentionally preserves:
    - strings
    - numeric literals
    - booleans
    - delimiters
    - built-ins / reserved words

    This means that two Racket programs that differ only in user-chosen
    identifier names, spacing, or end-of-line comments should collapse to the
    same normalized surface form, which is useful for similarity-based
    evaluation.
    """

    _identifier_regex = re.compile(r"^[^\d\W][\w!?+\-*/<>=:$%&~^.@]*$", re.UNICODE)

    def normalize(self, code: str) -> RacketNormalizationResult:
        """Normalize Racket source code."""
        original = self._normalize_newlines(code).strip()
        original = self._ensure_lang_header(original)

        uncommented = self._remove_line_comments(original)
        raw_tokens = self._tokenize(uncommented)
        normalized_tokens, identifier_map = self._canonicalize_identifiers(raw_tokens)
        normalized_code = self._untokenize(normalized_tokens).strip()

        if not normalized_code.startswith("#lang racket"):
            normalized_code = "#lang racket\n\n" + normalized_code

        if not normalized_code.endswith("\n"):
            normalized_code += "\n"

        return RacketNormalizationResult(
            original_code=original if original.endswith("\n") else original + "\n",
            normalized_code=normalized_code,
            identifier_map=identifier_map,
            tokens=normalized_tokens,
        )

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        """Normalize all line endings to '\\n'."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _ensure_lang_header(code: str) -> str:
        """Ensure '#lang racket' exists."""
        stripped = code.lstrip()
        if stripped.startswith("#lang racket"):
            return code
        return "#lang racket\n\n" + code

    def _remove_line_comments(self, code: str) -> str:
        """
        Remove ';' line comments while preserving strings.

        Racket comments start with ';' and continue to end of line.
        We only handle line comments here.

        The implementation deliberately ignores block comments such as `#| |#`.
        Those are uncommon in generated benchmark solutions, whereas `;` line
        comments appear frequently in model outputs and are easy to strip
        safely.
        """
        result_lines: List[str] = []
        for line in code.split("\n"):
            result_lines.append(self._strip_comment_from_line(line))
        return "\n".join(result_lines)

    @staticmethod
    def _strip_comment_from_line(line: str) -> str:
        """Strip '; ...' comment from a single line, respecting strings."""
        in_string = False
        escaped = False
        output_chars: List[str] = []

        for ch in line:
            if in_string:
                output_chars.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                output_chars.append(ch)
                continue

            if ch == ";":
                break

            output_chars.append(ch)

        return "".join(output_chars)

    def _tokenize(self, code: str) -> List[str]:
        """
        Tokenize Racket source.

        We keep:
        - delimiters as separate tokens
        - strings as atomic tokens
        - symbols/identifiers/operators as atomic tokens
        - quote prefixes as separate tokens

        Treating quote prefixes (`'`, `` ` ``, `,`) as standalone tokens makes
        it possible to reconstruct readable quoted forms such as `'(1 2 3)`
        without introducing parser dependencies.
        """
        tokens: List[str] = []
        i = 0
        n = len(code)

        while i < n:
            ch = code[i]

            if ch.isspace():
                i += 1
                continue

            if ch in DELIMITERS:
                tokens.append(ch)
                i += 1
                continue

            if ch in QUOTE_PREFIXES:
                tokens.append(ch)
                i += 1
                continue

            if ch == '"':
                string_token, i = self._read_string(code, i)
                tokens.append(string_token)
                continue

            atom, i = self._read_atom(code, i)
            if atom:
                tokens.append(atom)

        return tokens

    @staticmethod
    def _read_string(code: str, start: int) -> Tuple[str, int]:
        """Read a string literal."""
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

    @staticmethod
    def _read_atom(code: str, start: int) -> Tuple[str, int]:
        """Read a non-whitespace, non-delimiter atom."""
        i = start
        n = len(code)
        chars: List[str] = []

        while i < n:
            ch = code[i]
            if ch.isspace() or ch in DELIMITERS or ch in QUOTE_PREFIXES or ch == '"':
                break
            chars.append(ch)
            i += 1

        return "".join(chars), i

    def _canonicalize_identifiers(self, tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Replace user-defined identifiers with canonical names.

        Example:
            xs -> var_1
            acc -> var_2

        We do not rename:
        - reserved forms
        - built-ins
        - booleans/literals
        - numbers
        - quote tokens
        - delimiters
        """
        identifier_map: Dict[str, str] = {}
        normalized_tokens: List[str] = []
        next_id = 1

        for token in tokens:
            if self._should_preserve_token(token):
                normalized_tokens.append(token)
                continue

            if self._is_identifier(token):
                if token not in identifier_map:
                    identifier_map[token] = f"var_{next_id}"
                    next_id += 1
                normalized_tokens.append(identifier_map[token])
                continue

            normalized_tokens.append(token)

        return normalized_tokens, identifier_map

    def _should_preserve_token(self, token: str) -> bool:
        """Return True if a token should not be renamed."""
        if token in DELIMITERS:
            return True
        if token in QUOTE_PREFIXES:
            return True
        if token in RACKET_RESERVED_WORDS:
            return True
        if token in RACKET_BUILTINS:
            return True
        if token in RACKET_BOOLEAN_AND_LITERAL_WORDS:
            return True
        if self._is_string(token):
            return True
        if self._is_number(token):
            return True
        if token == ".":
            return True
        return False

    @staticmethod
    def _is_string(token: str) -> bool:
        """Check whether token is a string literal."""
        return len(token) >= 2 and token.startswith('"') and token.endswith('"')

    @staticmethod
    def _is_number(token: str) -> bool:
        """A permissive numeric check for common integer/float forms."""
        return bool(re.fullmatch(r"[+-]?\d+(\.\d+)?", token))

    def _is_identifier(self, token: str) -> bool:
        """Heuristic identifier check."""
        if not token:
            return False
        if token.startswith("#"):
            return False
        return bool(self._identifier_regex.fullmatch(token))

    def _untokenize(self, tokens: List[str]) -> str:
        """
        Convert token sequence back to readable normalized Racket code.

        This produces a stable single-space representation, while preserving
        no-space around bracket boundaries where appropriate.
        """
        pieces: List[str] = []
        prev: str | None = None

        for token in tokens:
            if not pieces:
                pieces.append(token)
                prev = token
                continue

            if token in {")", "]", "}"}:
                pieces.append(token)
            elif prev in {"(", "[", "{", "'", "`", ","}:
                pieces.append(token)
            elif token in {"(", "[", "{", "'", "`", ","}:
                pieces.append(" ")
                pieces.append(token)
            else:
                pieces.append(" ")
                pieces.append(token)

            prev = token

        text = "".join(pieces)
        text = self._postprocess_layout(text)
        return text

    @staticmethod
    def _postprocess_layout(text: str) -> str:
        """
        Improve readability of the normalized code.

        We keep this conservative:
        - restore '#lang racket' line header
        - put double newline after language header
        """
        text = text.strip()

        if text.startswith("#lang racket "):
            text = text.replace("#lang racket ", "#lang racket\n\n", 1)
        elif text == "#lang racket":
            text = "#lang racket\n"

        return text
