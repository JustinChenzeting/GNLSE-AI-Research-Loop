"""One JSON-in/JSON-out connection to the user's signed-in Codex CLI."""

from pathlib import Path
import json
import os
import shutil
import signal
import subprocess
import time
import math


def write_json(path, value):
    """Replace a JSON file atomically, rejecting NaN and infinity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def object_schema(properties):
    """Describe a JSON object with exactly the named fields."""
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def points_schema(problem):
    """Describe a list of parameter dictionaries."""
    return object_schema({"points": {
        "type": "array", "items": object_schema(problem["parameters"])
    }})


def check_point(problem, point):
    """Reject missing, extra, nonfinite, out-of-range, or noninteger values."""
    parameters = problem["parameters"]
    if not isinstance(point, dict) or set(point) != set(parameters):
        raise ValueError(f"Expected these parameters: {list(parameters)}.")
    checked = {}
    for name, spec in parameters.items():
        value = point[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if not spec["minimum"] <= value <= spec["maximum"]:
            raise ValueError(f"{name} is outside its allowed range.")
        if spec["type"] == "integer" and value != int(value):
            raise ValueError(f"{name} must be an integer.")
        checked[name] = int(value) if spec["type"] == "integer" else float(value)
    return checked


def check_points(problem, response, count, history=()):
    """Validate a complete batch before evaluating any point in it."""
    if not isinstance(response, dict) or set(response) != {"points"}:
        raise ValueError("Expected a JSON object containing only 'points'.")
    points = response["points"]
    if not isinstance(points, list) or len(points) != count:
        raise ValueError(f"Expected exactly {count} points.")
    checked = [check_point(problem, point) for point in points]
    seen = {tuple(row["parameters"][key] for key in problem["parameters"]) for row in history}
    for point in checked:
        key = tuple(point.values())
        if key in seen:
            raise ValueError("The LLM repeated a point. No new experiment was run.")
        seen.add(key)
    return checked


def measured_examples(history):
    """Send measured parameter/score pairs, excluding prediction diagnostics."""
    return [{"parameters": row["parameters"], "score": row["score"]} for row in history]


def connect_llm(log_directory, *, max_calls, model=None, reasoning_effort=None,
                timeout_seconds=180):
    """Return ask_llm(request, schema), using existing Codex authentication.

    Each call gets a fresh context, a read-only sandbox, and its own log folder.
    No automatic retries or model substitutions are made. The returned function's
    `calls` list records attempts, including failures, against `max_calls`.
    """
    executable = shutil.which("codex")
    if executable is None:
        raise FileNotFoundError("Install the Codex CLI, then run `codex login`.")
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError("max_calls must be a positive integer.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    status = subprocess.run(
        [executable, "login", "status"], capture_output=True, text=True, timeout=20
    )
    if status.returncode:
        raise RuntimeError("Codex is not signed in. Run `codex login` in a terminal.")

    log_directory = Path(log_directory).resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    calls = []

    def ask_llm(request, schema):
        """Request structured numbers; never execute model-produced code."""
        if len(calls) >= max_calls:
            raise RuntimeError(f"The LLM call budget ({max_calls}) is exhausted.")
        directory = log_directory / f"call_{len(calls) + 1:03d}"
        directory.mkdir(exist_ok=False)
        schema_path = directory / "schema.json"
        answer_path = directory / "answer.json"
        write_json(directory / "request.json", request)
        write_json(schema_path, schema)
        prompt = (
            "You are a numerical assistant for Bayesian optimization. "
            "Use only the supplied problem description and measured observations. "
            "Do not use tools, inspect files, run experiments, or write code. "
            "Predictions are estimates, never new measurements. "
            "Return only JSON matching the supplied output schema.\n\n"
            + json.dumps(request, indent=2, allow_nan=False)
        )
        (directory / "prompt.txt").write_text(prompt)
        command = [
            executable, "exec", "--sandbox", "read-only",
            "--skip-git-repo-check", "--ephemeral", "--color", "never", "--json",
            "--output-schema", str(schema_path),
            "--output-last-message", str(answer_path), "-C", str(directory),
        ]
        if model is not None:
            command.extend(["--model", model])
        if reasoning_effort is not None:
            command.extend(["--config", "model_reasoning_effort=" + json.dumps(reasoning_effort)])
        command.append("-")
        write_json(directory / "command.json", command)
        record = {"call": len(calls) + 1, "operation": request["operation"],
                  "model": model or "inherited Codex configuration",
                  "reasoning_effort": reasoning_effort or "inherited Codex configuration",
                  "status": "started"}
        calls.append(record)
        write_json(directory / "metadata.json", record)
        started = time.monotonic()
        print(f"  LLM {len(calls)}/{max_calls}: {request['operation']}", flush=True)
        try:
            with (directory / "events.jsonl").open("w") as out, (directory / "stderr.log").open("w") as err:
                process = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                    text=True, start_new_session=True,
                )
                try:
                    process.communicate(prompt, timeout=timeout_seconds)
                except BaseException:
                    # Stop the process group when a notebook cell is interrupted.
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    raise
            if process.returncode or not answer_path.is_file():
                raise RuntimeError(f"Codex did not return a result. Inspect {directory}.")
            answer = json.loads(answer_path.read_text())
            for line in (directory / "events.jsonl").read_text().splitlines():
                event = json.loads(line)
                if "usage" in event:
                    record["usage"] = event["usage"]
            record["status"] = "completed"
            return answer
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["seconds"] = round(time.monotonic() - started, 3)
            write_json(directory / "metadata.json", record)

    ask_llm.calls = calls
    return ask_llm
