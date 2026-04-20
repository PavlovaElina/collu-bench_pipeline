from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import List, Optional


UNSUPPORTED_TEST_PATTERNS = (
    "lambda",
    "set(",
    "{",
    "dict(",
    "raise ",
    "try:",
    "except",
    "with ",
    "yield ",
    "sorted(",
)


PYTHON_SPECIFIC_TEXT_REPLACEMENTS = [
    (r"\bPython function\b", "Racket function"),
    (r"\bpython function\b", "Racket function"),
    (r"\bPython list\b", "list"),
    (r"\bpython list\b", "list"),
    (r"\bPython lists\b", "lists"),
    (r"\bPython tuple\b", "list"),
    (r"\bPython tuples\b", "lists"),
    (r"\bTrue\b", "#t"),
    (r"\bFalse\b", "#f"),
    (r"\bNone\b", "'()"),
]


@dataclass
class TranslatedTestCase:
    python_source: str
    racket_check: str


@dataclass
class TranslationResult:
    success: bool
    entry_point: str
    adapted_prompt: str
    racket_test_module: str
    translated_cases: List[TranslatedTestCase]
    failure_reason: Optional[str] = None


def is_humaneval_task_translatable(
    prompt: str,
    test_source: str,
    entry_point: str,
) -> tuple[bool, str]:
    """
    Heuristic filter for tasks that are realistic to translate automatically.

    We reject tasks that appear to use Python-specific or structurally complex
    testing patterns that this translator does not support yet.
    """
    if not prompt.strip():
        return False, "empty prompt"

    if not test_source.strip():
        return False, "empty tests"

    if not entry_point.strip():
        return False, "empty entry point"

    lowered_tests = test_source.lower()
    lowered_prompt = prompt.lower()

    for pattern in UNSUPPORTED_TEST_PATTERNS:
        if pattern in lowered_tests:
            return False, f"unsupported test pattern: {pattern}"

    disallowed_prompt_markers = [
        "dictionary",
        "dictionaries",
        "set of",
        "generator",
        "yield",
        "numpy",
        "pandas",
        "complex number",
    ]
    for marker in disallowed_prompt_markers:
        if marker in lowered_prompt:
            return False, f"unsupported prompt marker: {marker}"

    if "candidate(" not in test_source:
        return False, "tests do not call candidate(...)"

    if "assert" not in test_source:
        return False, "tests do not contain assert"

    return True, "ok"


def adapt_python_prompt_to_racket(prompt: str, entry_point: str) -> str:
    """
    Rewrite Python-oriented task text into a Racket-oriented task description.

    This version adds stronger semantic hints for generic list-processing tasks,
    which helps the model avoid over-specializing to numeric lists.
    """
    text = prompt.strip()

    for pattern, replacement in PYTHON_SPECIFIC_TEXT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    text = re.sub(
        r"\bwrite a function\b",
        "write a Racket function",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bwrite a python function\b",
        "write a Racket function",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\breturn (true|false)\b",
        "return a Racket boolean (#t or #f)",
        text,
        flags=re.IGNORECASE,
    )

    lowered = text.lower()
    semantic_notes: list[str] = []

    if "list" in lowered:
        semantic_notes.append(
            "When the original task refers to Python lists or tuples, use Racket lists."
        )

    if "last element" in lowered or "last item" in lowered or "last value" in lowered:
        semantic_notes.append(
            "The input list may contain values of any type, not only numbers."
        )
        semantic_notes.append(
            "Return the final element of the list without assuming numeric operations."
        )

    if "string" in lowered:
        semantic_notes.append(
            "Use standard Racket strings and preserve the intended behavior of the original task."
        )

    if "true" in prompt or "false" in prompt:
        semantic_notes.append(
            "Use Racket booleans #t and #f instead of Python booleans."
        )

    extra_note_lines = [
        "",
        "",
        "Implement the solution in Racket.",
        "Use `#lang racket`.",
        f"Define the function `{entry_point}`.",
        "Preserve the original task semantics, but write valid Racket code.",
    ]
    extra_note_lines.extend(semantic_notes)
    extra_note_lines.append("Return only valid Racket code.")

    extra_note = "\n".join(extra_note_lines)

    return text + extra_note


def translate_python_tests_to_racket(
    prompt: str,
    test_source: str,
    entry_point: str,
) -> TranslationResult:
    """
    Translate a subset of Python/HumanEval tests into a Racket rackunit module.

    Supported assert patterns:
    - assert candidate(...) == ...
    - assert candidate(...)
    - assert not candidate(...)
    """
    ok, reason = is_humaneval_task_translatable(prompt, test_source, entry_point)
    if not ok:
        return TranslationResult(
            success=False,
            entry_point=entry_point,
            adapted_prompt="",
            racket_test_module="",
            translated_cases=[],
            failure_reason=reason,
        )

    try:
        tree = ast.parse(test_source)
    except SyntaxError as exc:
        return TranslationResult(
            success=False,
            entry_point=entry_point,
            adapted_prompt="",
            racket_test_module="",
            translated_cases=[],
            failure_reason=f"python test parse error: {exc}",
        )

    translated_cases: List[TranslatedTestCase] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # HumanEval often stores checks inside:
            # def check(candidate): ...
            translated_cases.extend(_translate_check_function(node, entry_point))
        elif isinstance(node, ast.Assert):
            translated = _translate_assert_node(node, entry_point)
            translated_cases.append(translated)
        else:
            # Ignore harmless top-level nodes such as comments/docstrings/pass.
            continue

    if not translated_cases:
        return TranslationResult(
            success=False,
            entry_point=entry_point,
            adapted_prompt="",
            racket_test_module="",
            translated_cases=[],
            failure_reason="no translatable assert cases found",
        )

    adapted_prompt = adapt_python_prompt_to_racket(prompt, entry_point)
    racket_test_module = _build_racket_test_module(entry_point, translated_cases)

    return TranslationResult(
        success=True,
        entry_point=entry_point,
        adapted_prompt=adapted_prompt,
        racket_test_module=racket_test_module,
        translated_cases=translated_cases,
        failure_reason=None,
    )


def _translate_check_function(
    fn_node: ast.FunctionDef,
    entry_point: str,
) -> List[TranslatedTestCase]:
    """
    Translate asserts from a HumanEval-style check(candidate) function.
    """
    translated_cases: List[TranslatedTestCase] = []

    for stmt in fn_node.body:
        if isinstance(stmt, ast.Assert):
            translated_cases.append(_translate_assert_node(stmt, entry_point))

    return translated_cases


def _translate_assert_node(node: ast.Assert, entry_point: str) -> TranslatedTestCase:
    """
    Translate one Python assert statement into a rackunit check.
    """
    source = ast.unparse(node.test)

    expr = node.test

    if isinstance(expr, ast.Compare):
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            raise ValueError(f"unsupported comparison assert: {source}")

        op = expr.ops[0]
        left = expr.left
        right = expr.comparators[0]

        if isinstance(op, ast.Eq):
            left_rkt = _translate_python_expr_to_racket(left, entry_point)
            right_rkt = _translate_python_expr_to_racket(right, entry_point)
            return TranslatedTestCase(
                python_source=source,
                racket_check=f"(check-equal? {left_rkt} {right_rkt})",
            )

        raise ValueError(f"unsupported comparison operator in assert: {source}")

    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        operand = _translate_python_expr_to_racket(expr.operand, entry_point)
        return TranslatedTestCase(
            python_source=source,
            racket_check=f"(check-false {operand})",
        )

    translated_expr = _translate_python_expr_to_racket(expr, entry_point)
    return TranslatedTestCase(
        python_source=source,
        racket_check=f"(check-true {translated_expr})",
    )


def _translate_python_expr_to_racket(expr: ast.AST, entry_point: str) -> str:
    """
    Translate a restricted Python expression subset into a Racket expression.
    """
    if isinstance(expr, ast.Call):
        return _translate_call(expr, entry_point)

    if isinstance(expr, ast.Constant):
        return _translate_constant(expr.value)

    if isinstance(expr, ast.List):
        items = " ".join(_translate_python_expr_to_racket(elt, entry_point) for elt in expr.elts)
        return f"'({items})"

    if isinstance(expr, ast.Tuple):
        items = " ".join(_translate_python_expr_to_racket(elt, entry_point) for elt in expr.elts)
        return f"'({items})"

    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        operand = _translate_python_expr_to_racket(expr.operand, entry_point)
        if operand.isdigit():
            return f"-{operand}"
        return f"(- {operand})"

    if isinstance(expr, ast.Name):
        if expr.id == "True":
            return "#t"
        if expr.id == "False":
            return "#f"
        if expr.id == "None":
            return "'()"
        raise ValueError(f"unsupported bare name in tests: {expr.id}")

    raise ValueError(f"unsupported python expression in tests: {ast.dump(expr)}")


def _translate_call(call: ast.Call, entry_point: str) -> str:
    """
    Translate function calls.

    Main supported pattern:
        candidate(...)
    """
    if isinstance(call.func, ast.Name) and call.func.id == "candidate":
        translated_args = [_translate_python_expr_to_racket(arg, entry_point) for arg in call.args]
        joined_args = " ".join(translated_args)
        if joined_args:
            return f"({entry_point} {joined_args})"
        return f"({entry_point})"

    if isinstance(call.func, ast.Name):
        name = call.func.id
        translated_args = [_translate_python_expr_to_racket(arg, entry_point) for arg in call.args]
        joined_args = " ".join(translated_args)

        builtin_map = {
            "len": "length",
            "abs": "abs",
            "sum": "apply +",
            "max": "max",
            "min": "min",
        }

        if name in builtin_map:
            target = builtin_map[name]
            if target == "apply +":
                if len(translated_args) != 1:
                    raise ValueError("sum(...) translation expects one argument")
                return f"(apply + {translated_args[0]})"
            if joined_args:
                return f"({target} {joined_args})"
            return f"({target})"

    raise ValueError(f"unsupported function call in tests: {ast.dump(call)}")


def _translate_constant(value: object) -> str:
    """Translate a Python literal value into a Racket literal."""
    if value is True:
        return "#t"
    if value is False:
        return "#f"
    if value is None:
        return "'()"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return repr(value)

    raise ValueError(f"unsupported constant type in tests: {type(value).__name__}")


def _build_racket_test_module(
    entry_point: str,
    translated_cases: List[TranslatedTestCase],
) -> str:
    """
    Build a complete rackunit module.

    The generated solution is expected to be stored in `solution.rkt`.

    Important:
    - we wrap checks inside a rackunit test-suite
    - we execute it with run-tests
    - we return exit code 0 only if all tests pass
    """
    test_case_lines = []
    for idx, case in enumerate(translated_cases, start=1):
        escaped_name = f"case-{idx}"
        test_case_lines.append(
            f'   (test-case "{escaped_name}" {case.racket_check})'
        )

    tests_block = "\n".join(test_case_lines)

    return (
        "#lang racket\n\n"
        "(require rackunit)\n"
        "(require rackunit/text-ui)\n"
        '(require "solution.rkt")\n\n'
        "(define generated-tests\n"
        '  (test-suite\n'
        '   "generated-tests"\n'
        f"{tests_block}\n"
        "   ))\n\n"
        "(define failures (run-tests generated-tests))\n\n"
        "(if (zero? failures)\n"
        '    (begin\n'
        '      (displayln "ALL_TESTS_PASSED")\n'
        "      (exit 0))\n"
        "    (exit 1))\n"
    )