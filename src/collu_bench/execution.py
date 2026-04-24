from __future__ import annotations

"""
Execution harnesses for generated benchmark solutions.

Racket tasks are executed through the generic `external_command` pathway rather
than a language-specific inline runner.  This is a deliberate design choice: a
Racket task typically needs a small surrounding harness that writes
`solution.rkt`, imports it from a generated `runner.rkt`, and then invokes the
system `racket` executable so that RackUnit can determine the final exit code.
"""

import os
import shlex
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .data import TaskInstance, TestSpec


@dataclass
class ExecutionResult:
    status: str
    stdout: str
    stderr: str
    command: Optional[List[str]] = None


class ExecutionRunner:
    """Execute generated programs and capture feedback."""

    def __init__(self, timeout_seconds: int, workspace: Path):
        self.timeout = timeout_seconds
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, task: TaskInstance, code: str) -> ExecutionResult:
        if not code.strip():
            return ExecutionResult(status="empty", stdout="", stderr="empty generation")
        if task.language == "python":
            return self._run_python(task, code)
        if task.tests.kind == "external_command":
            return self._run_external_command(task, code)
        return ExecutionResult(status="unsupported", stdout="", stderr="Language not supported")

    def _run_python(self, task: TaskInstance, code: str) -> ExecutionResult:
        script_segments = [code.rstrip()]
        tests = task.tests
        if tests.kind == "humaneval":
            script_segments.append(tests.content.strip())
            if not task.entry_point:
                return ExecutionResult(
                    status="error", stdout="", stderr="missing entry point for HumanEval task"
                )
            script_segments.append(f"check({task.entry_point})")
        elif tests.kind in {"mbpp_assert", "script"}:
            if tests.content:
                script_segments.append(tests.content.strip())
        script = "\n\n".join(segment for segment in script_segments if segment)
        script_path = self._write_temp_file(script, suffix=".py")
        try:
            proc = subprocess.run(
                ["python3", script_path.name],
                cwd=script_path.parent,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            status = "pass" if proc.returncode == 0 else "fail"
            return ExecutionResult(status=status, stdout=proc.stdout, stderr=proc.stderr)
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                status="timeout",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            )
        finally:
            script_path.unlink(missing_ok=True)
            _cleanup_temp_dir(script_path.parent)

    def _run_external_command(self, task: TaskInstance, code: str) -> ExecutionResult:
        tests = task.tests
        if not tests.command:
            return ExecutionResult(status="error", stdout="", stderr="missing command")

        if task.language == "python":
            extension = ".py"
        elif task.language == "racket":
            # Racket candidates are materialized as `.rkt` files so that the
            # downstream harness can `require` them as ordinary modules.
            extension = ".rkt"
        else:
            extension = ".java"

        code_path = self._write_temp_file(code, suffix=extension)
        template = tests.command
        command = template.format(
            code_path=str(code_path),
            dataset=task.dataset,
            task_id=task.task_id,
        )

        env = {k: v for k, v in tests.environment.items() if v is not None}
        env.update(
            {
                "COLLU_CODE_PATH": str(code_path),
                "COLLU_TASK_ID": str(task.task_id),
                "COLLU_DATASET": task.dataset,
            }
        )
        merged_env = {**os.environ, **env}

        project_root = Path(__file__).resolve().parents[2]

        if os.name == "nt":
            command_parts = shlex.split(command, posix=False)
            if command_parts and command_parts[0].lower() == "python":
                command_parts[0] = sys.executable

            proc_command = command_parts
            proc_cwd = tests.workdir or str(project_root)
        else:
            proc_command = ["bash", "-lc", command]
            proc_cwd = tests.workdir or str(project_root)

        try:
            proc = subprocess.run(
                proc_command,
                cwd=proc_cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=merged_env,
            )
            status = "pass" if proc.returncode == 0 else "fail"
            return ExecutionResult(
                status=status,
                stdout=proc.stdout,
                stderr=proc.stderr,
                command=proc_command,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                status="timeout",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                command=proc_command,
            )
        finally:
            code_path.unlink(missing_ok=True)
            _cleanup_temp_dir(code_path.parent)

    def _write_temp_file(self, content: str, suffix: str) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(dir=self.workspace))
        tmp_path = tmp_dir / f"task-{uuid.uuid4().hex}{suffix}"
        tmp_path.write_text(content, encoding="utf-8")
        return tmp_path


def _cleanup_temp_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass
