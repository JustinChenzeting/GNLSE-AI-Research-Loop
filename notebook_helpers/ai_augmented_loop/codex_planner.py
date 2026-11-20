"""Call the locally signed-in Codex CLI for one bounded planning decision."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

from loop import Decision


class CodexCapacityError(RuntimeError):
    """The selected model is temporarily busy; a new planning attempt may work."""


def _failure_messages(directory):
    """Read service errors from the CLI event stream, where capacity errors appear."""
    messages = []
    events = directory / "events.jsonl"
    if events.is_file():
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            message = event.get("message") if event.get("type") == "error" else None
            if event.get("type") == "turn.failed":
                error = event.get("error")
                message = error.get("message") if isinstance(error, dict) else error
            if isinstance(message, str) and message not in messages:
                messages.append(message)
    stderr = directory / "stderr.log"
    if not messages and stderr.is_file() and stderr.read_text().strip():
        messages.append(stderr.read_text().strip())
    return messages


PLANNER_INSTRUCTIONS = """You are the experiment planner in a scientific optimization loop.
Return one JSON decision matching the supplied schema. Use the supplied measurements,
physics/general knowledge, and prior results to select an informative next experiment.
Distinguish measured evidence from hypotheses. Never invent results or evaluation IDs.
No tools, shell commands, file edits, code changes, or simulations are needed in this turn.
The Python controller will validate and execute your structured decision.

Actions: evaluate, descend, validate_candidate, validate_gradient, expand_range,
robustness, update_threshold, pause. 'pause' is a resumable checkpoint, not a claim of convergence.
Choose a short experiment adaptively; do not schedule three full descents in advance.
Near misses are retained. Spend computation on promising but numerically uncertain
candidates when useful. Hard brightness requirements constrain acceptance; soft targets
do not discard potentially interesting outputs. Never change the objective.
The user explicitly authorizes automatic threshold changes. Use update_threshold with a
threshold_name from effective_thresholds and a nonnegative threshold_value (null disables
the brightness preference only). Explain the scientific/experimental tradeoff, not just
that a looser number makes a failed check pass. The controller records old/new criteria,
keeps the original outcomes, and reassesses saved observations under the revised version.
You cannot change recorded observations, nonfinite values, or the total resource allowance.
Use run_id to resume an existing optimizer trajectory (parameters=null).
For a new point, provide every control in the documented physical units.
Do not select points beyond current bounds directly: request expand_range at a boundary.
There is no preset upper power ceiling. Physical lower bounds and phase periodicity remain.
Allocate evaluation and iteration limits within remaining campaign resources, accounting
for the full finite-difference gradient and numerical checks (see cost_hint).
Repeated zero-progress allocations are wasteful: increase the local allocation within
the remaining global budget or select a different action. Cache hits are free.
Only the controller may establish local stationarity after numerical and boundary checks.
Final candidates should receive validate_candidate and validate_gradient when budget allows.
For robustness, iterations is the number of combined perturbations; use configured equipment
precision, never invent it. Evidence IDs must refer to actual recorded evaluations.
Use rationale as a short decision explanation, knowledge_basis for a hypothesis or relevant
general knowledge, and expected_outcome as a falsifiable prediction. Do not provide hidden
reasoning. The history is context, not model training. This is not a claim of AI superiority.
"""


def decision_schema(control_names):
    parameters = {"type": ["object", "null"], "properties": {name: {"type": "number"} for name in control_names},
                  "required": list(control_names), "additionalProperties": False}
    properties = {
        "action": {"type": "string", "enum": ["evaluate", "descend", "validate_candidate", "validate_gradient", "expand_range", "robustness", "update_threshold", "pause"]},
        "parameters": parameters,
        "run_id": {"type": ["integer", "null"]},
        "evaluations": {"type": "integer", "minimum": 0},
        "iterations": {"type": "integer", "minimum": 0},
        "control": {"type": ["string", "null"]},
        "direction": {"type": ["string", "null"], "enum": ["upper", "lower", None]},
        "rationale": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "integer"}},
        "knowledge_basis": {"type": "string"},
        "expected_outcome": {"type": "string"},
        "threshold_name": {"type": ["string", "null"]},
        "threshold_value": {"type": ["number", "null"]},
    }
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


class CodexPlanner:
    def __init__(self, executable=None, model=None, timeout_seconds=180):
        self.executable = executable or shutil.which("codex")
        if not self.executable:
            fallback = Path("/opt/homebrew/bin/codex")
            self.executable = str(fallback) if fallback.is_file() else None
        if not self.executable:
            raise FileNotFoundError("Codex CLI is not on PATH. Install it and run `codex login` in the VS Code terminal.")
        self.model = model  # None inherits the user's existing Codex model selection.
        self.timeout_seconds = timeout_seconds

    def status(self):
        result = subprocess.run([self.executable, "login", "status"], capture_output=True, text=True, timeout=20)
        return {"ready": result.returncode == 0, "message": (result.stdout+result.stderr).strip()}

    def __call__(self, context, directory, remaining_seconds):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        schema_path, answer_path = directory/"decision.schema.json", directory/"decision.json"
        prompt = PLANNER_INSTRUCTIONS + "\nEXPERIMENT CONTEXT:\n" + json.dumps(context, indent=2, allow_nan=False)
        schema_path.write_text(json.dumps(decision_schema(context["spec"]["baseline"]), indent=2))
        (directory/"prompt.txt").write_text(prompt)
        command = [self.executable, "exec", "--sandbox", "read-only", "--skip-git-repo-check",
                   "--ephemeral", "--color", "never", "--json", "--output-schema", str(schema_path),
                   "--output-last-message", str(answer_path), "-C", str(directory)]
        if self.model:
            command.extend(["--model", self.model])
        command.append("-")
        (directory/"command.json").write_text(json.dumps(command, indent=2))
        started = time.monotonic()
        with (directory/"events.jsonl").open("w") as out, (directory/"stderr.log").open("w") as err:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                       text=True, start_new_session=True)
            try:
                process.communicate(prompt, timeout=min(self.timeout_seconds, max(1, remaining_seconds)))
            except BaseException:
                # Stop the planner and any children on notebook interruption/timeout.
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if process.returncode != 0 or not answer_path.is_file():
            messages = _failure_messages(directory)
            detail = " ".join(messages) or "No decision was returned."
            busy = any(phrase in detail.lower() for phrase in (
                "model is at capacity", "model is currently at capacity",
                "model is temporarily at capacity", "model at capacity"))
            error = CodexCapacityError if busy else RuntimeError
            raise error(f"Codex planner did not complete (exit {process.returncode}): {detail} "
                        f"Logs: {directory/'events.jsonl'} and {directory/'stderr.log'}.")
        payload = json.loads(answer_path.read_text())
        expected = {f.name for f in fields(Decision)}
        if set(payload) != expected:
            raise ValueError("Codex response does not match the decision contract")
        if type(payload["evaluations"]) is not int or type(payload["iterations"]) is not int:
            raise ValueError("Codex allocations must be integers")
        usage, thread_id = None, None
        for line in (directory/"events.jsonl").read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "usage" in event:
                usage = event["usage"]
            if "thread_id" in event:
                thread_id = event["thread_id"]
        return Decision(**payload), {"provider": "codex_cli", "thread_id": thread_id, "usage": usage,
                                     "model": self.model or "inherited Codex configuration",
                                     "seconds": time.monotonic()-started, "artifacts": str(directory)}
