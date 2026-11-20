"""Continue a saved GD search in a new directory; keep the parent immutable."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time

from . import spectral_mask as sm
from .spectral_mask_gd import Campaign, DescentConfig, ROOT, load_run, save
from .spectral_mask_resources import ResourceMonitor


PRESERVED_NOTEBOOKS = [
    next((ROOT / "notebooks").rglob(name)) for name in
    ("dnl_gradient_descent_spectral_mask.ipynb", "dnl_llm_bo_spectral_mask.ipynb")
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fork_saved_measurements(parent, destination):
    """Copy metadata and link immutable raw artifacts, avoiding a duplicate 1.6 GB."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / "parent_snapshot"
    snapshot.mkdir()
    protected = {}
    for name in ("state.json", "run.json", "events.jsonl", "timing_audit.json"):
        source = parent.directory / name
        if source.exists():
            shutil.copy2(source, snapshot/name)
            protected[str(source)] = digest(source)
    for path in PRESERVED_NOTEBOOKS:
        protected[str(path)] = digest(path)
        shutil.copy2(path, snapshot/path.name)
    shutil.copy2(parent.directory/"events.jsonl", destination/"events.jsonl")
    for source in sorted(parent.directory.glob("gradient_*.json")):
        shutil.copy2(source, destination/source.name)
        protected[str(source)] = digest(source)
    gnlse = destination / "gnlse"
    gnlse.mkdir()
    identity = parent.directory/"gnlse/evaluator.json"
    shutil.copy2(identity, gnlse/"evaluator.json")
    protected[str(identity)] = digest(identity)
    records = []
    for source in sorted((parent.directory/"gnlse").glob("evaluation_*")):
        target = gnlse/source.name
        target.mkdir()
        for path in source.iterdir():
            if path.name == "measurement.json":
                shutil.copy2(path, target/path.name)
                protected[str(path)] = digest(path)
                row = json.loads(path.read_text())
                records.append({"evaluation": row["evaluation"], "measurement": str(path),
                                "fields": row.get("artifacts", {}).get("fields"),
                                "fields_sha256": row.get("artifacts", {}).get("fields_sha256")})
            else:
                (target/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    manifest = {"parent": str(parent.directory), "protected_sha256": protected,
                "inherited_measurements": records,
                "raw_artifacts": "Read-only reuse by symlink; parent directory must remain available."}
    save(destination/"parent_manifest.json", manifest)
    return manifest


def verify_parent_preserved(directory):
    manifest = json.loads((Path(directory)/"parent_manifest.json").read_text())
    changed = [path for path, expected in manifest["protected_sha256"].items() if digest(path) != expected]
    for row in manifest["inherited_measurements"]:
        if row["fields"] and digest(row["fields"]) != row["fields_sha256"]:
            changed.append(row["fields"])
    changed_notebooks = [path for path in changed if path.endswith(".ipynb")]
    changed_data = [path for path in changed if not path.endswith(".ipynb")]
    result = {"passed": not changed, "changed": changed, "raw_run_preserved":not changed_data,
              "changed_notebooks":changed_notebooks,
              "metadata_and_notebooks_checked": len(manifest["protected_sha256"]),
              "inherited_fields_checked": sum(bool(r["fields"]) for r in manifest["inherited_measurements"])}
    save(Path(directory)/"parent_preservation.json", result)
    if changed_data:
        raise RuntimeError(f"Preserved parent run data changed: {changed_data}")
    if changed_notebooks:
        print("Parent notebook changed since the fork; its saved snapshot and raw run remain available. Current notebook left untouched:",changed_notebooks,flush=True)
    return result


class ContinuationCampaign(Campaign):
    """Same FD, projection, backtracking and checks; start at the parent's last iterate."""
    def __init__(self, directory, physics, limits, evaluator, parent, **clocks):
        self.inherited_active = parent.result["total_wall_seconds"]
        self.inherited_pairs = parent.result["evaluation_attempts"]
        accepted = parent.result.get("accepted_steps", [])
        self.parent_iteration = accepted[-1]["iteration"] if accepted else 0
        if limits.max_iterations <= self.parent_iteration:
            raise ValueError("The total iteration limit must exceed the saved accepted iteration.")
        super().__init__(directory, physics, limits, evaluator, **clocks)
        self.state = deepcopy(parent.result)
        self.parent_validation = deepcopy(self.state.pop("validation"))
        self.state.pop("stop_reason", None)
        self.state.update(status="running", descent_config=asdict(limits),
                          parent_validation=self.parent_validation, result_label="preliminary")
        self.state["continuation"] = {
            "parent_directory": str(parent.directory), "parent_iteration": self.parent_iteration,
            "first_new_iteration": self.parent_iteration+1, "total_iteration_limit": limits.max_iterations,
            "inherited_objective_pairs": self.inherited_pairs,
            "max_new_objective_pairs": limits.max_evaluations-self.inherited_pairs,
            "new_wall_seconds_limit": limits.max_wall_seconds,
            "inherited_active_seconds": self.inherited_active,
            "runtime_basis": "Cumulative active time = recorded parent active time + this continuation; gaps between sessions excluded.",
            "calendar_basis": "Calendar deadline and calendar elapsed time apply to this continuation only.",
        }
        current_id = accepted[-1]["evaluation"] if accepted else self.state["restored_start"]["evaluation"]
        self.current = deepcopy(next(r for r in self.state["history"] if r["evaluation"] == current_id))
        if not self.current["feasible"]:
            raise ValueError("The saved current mask must be feasible on the search grid.")
        self.reference = self.parent_validation["baseline_fine"]["diagnostics"]["total_photons"]
        self.state["continuation"].update(start_evaluation=current_id, start_point=self.current["point"],
                                          start_D_NL=self.current["D_NL"],
                                          start_brightness=self.current["brightness_ratio"])

    def elapsed(self):
        return self.inherited_active + super().elapsed()

    def event(self, kind, **data):
        return super().event(kind, segment="continuation",
                             continuation_active_seconds=super().elapsed(), **data)

    def checkpoint(self):
        self.state.update(total_wall_seconds=self.elapsed(), active_runtime_seconds=self.elapsed(),
                          continuation_active_seconds=super().elapsed(),
                          calendar_elapsed_seconds=self.wall_elapsed(),
                          continuation_calendar_seconds=self.wall_elapsed(),
                          evaluation_attempts=self.evaluator.attempts,
                          new_evaluation_attempts=self.evaluator.attempts-self.inherited_pairs,
                          clock_policy="continuation calendar deadline; cumulative and additional active times recorded separately")
        save(self.directory/"state.json", self.state)

    def run(self):
        current = self.current
        base = self.state["history"][0]
        fine_base = self.parent_validation["baseline_fine"]
        best = deepcopy(self.state["best_search"])
        self.event("continuation_started", **self.state["continuation"])
        self.checkpoint()
        try:
            for iteration in range(self.parent_iteration+1, self.limits.max_iterations+1):
                self.state["last_attempted_iteration"] = iteration
                self.checkpoint()
                print(f"Gradient {iteration}/{self.limits.max_iterations}: continuing from evaluation {current['evaluation']}", flush=True)
                slopes = self.gradient(current, iteration)
                proposal, reason = self.line_search(current, slopes, iteration)
                if proposal is None:
                    self.state["stop_reason"] = reason
                    break
                current = proposal
                if current["D_NL"] < best["D_NL"]:
                    best = current
                self.state["best_search"] = best
                print(f"Accepted iteration {iteration}: D_NL={current['D_NL']:.8f}, brightness={current['brightness_ratio']:.6f}", flush=True)
                self.checkpoint()
            else:
                self.state["stop_reason"] = "iteration limit reached; no convergence claim"
        except Exception as error:
            self.state["stop_reason"] = f"{type(error).__name__}: {error}"
            self.event("search_stopped", reason=self.state["stop_reason"])
        finally:
            self.state["best_search"] = best
            self.validate(base, fine_base, best)
            self.evaluator.close()
            self.state["status"] = "completed"
            self.state["result_label"] = ("numerically validated and feasible" if
                self.state.get("validation", {}).get("status") == "pass" else "preliminary")
            self.checkpoint()
        return self.state


def continue_experiment(*, parent_directory, max_iterations=10, max_new_evaluations=1700,
                        max_minutes=180., validation_minutes=6., load_directory=None, directory=None):
    if load_directory is not None:
        run = load_run(load_directory)
        print("Loaded saved GD continuation:", run.directory)
        return run
    started, wall_started = time.monotonic(), time.time()
    parent = load_run(parent_directory)
    if parent.result["status"] != "completed":
        raise ValueError("Continue from a completed saved checkpoint.")
    last_iteration = max((s["iteration"] for s in parent.result["accepted_steps"]), default=0)
    if max_iterations <= last_iteration or max_new_evaluations < 6:
        raise ValueError("Increase the total iteration limit and allow at least six new objective pairs.")
    if sm.FLOOR != .50 or sm.TOLERANCE != 2e-4:
        raise ValueError("Keep the parent brightness floor and numerical-check tolerance.")
    physics = sm.MaskConfig(**parent.result["config"])
    limits = DescentConfig(**{**parent.result["descent_config"], "max_iterations": max_iterations,
        "max_evaluations": parent.result["evaluation_attempts"]+max_new_evaluations,
        "max_wall_seconds": 60*max_minutes, "validation_reserve_seconds": 60*validation_minutes})
    directory = Path(directory) if directory else ROOT/"output"/datetime.now(timezone.utc).strftime("%Y-%m-%d_gd_spectral_mask_continue_%H%M%S_%fZ")
    directory = directory.resolve()
    fork_saved_measurements(parent, directory)
    # Resume verifies evaluator identity and every cached field hash before any new solve.
    simulator = sm.MaskEvaluator(directory/"gnlse", physics, resume=True)
    simulator.max_evaluations = limits.max_evaluations
    campaign = ContinuationCampaign(directory, physics, limits, simulator, parent)
    campaign.started, campaign.wall_started = started, wall_started
    current = campaign.current
    resolution = current["resolution"]
    key = (sm.point_key(current["point"]), resolution["Nt"], resolution["time_window_ps"], resolution["dz_m"])
    if key not in simulator.cache or simulator.cache[key]["D_NL"] != current["D_NL"]:
        raise ValueError("Current iterate does not match the verified measurement cache.")
    metadata = deepcopy(json.loads((parent.directory/"run.json").read_text()))
    sources = [Path(__file__), Path(__file__).with_name("spectral_mask_gd.py"), Path(sm.__file__),
               simulator.worker, simulator.worker.with_name("dio_worker.jl")]
    (directory/"sources").mkdir()
    hashes = {}
    for path in sources:
        shutil.copy2(path, directory/"sources"/path.name)
        hashes[str(path)] = digest(path)
    metadata.update(limits=asdict(limits), continuation=campaign.state["continuation"],
                    evaluator=simulator.identity, source_sha256=hashes,
                    started_utc=datetime.fromtimestamp(wall_started, timezone.utc).isoformat(),
                    command_argv=sys.argv, controller_pid=os.getpid(),
                    hardware={"platform": platform.platform(), "machine": platform.machine(),
                              "python": platform.python_version(), "cpu_count": os.cpu_count()},
                    runtime_boundary="Continuation setup through search, checks and worker shutdown; cumulative active time adds the recorded parent run; plotting excluded.")
    save(directory/"run.json", metadata)
    campaign.checkpoint()
    print("Continuing GD in:", directory, flush=True)
    print(f"Parent iteration {last_iteration}, evaluation {current['evaluation']}: D_NL={current['D_NL']:.8f}, brightness={current['brightness_ratio']:.6f}", flush=True)
    print(f"Limits: {max_iterations} total iterations; {max_new_evaluations} NEW pairs; {max_minutes:g} additional calendar minutes including validation.", flush=True)
    monitor = ResourceMonitor(directory, started=started).start()
    try:
        result = campaign.run()
    finally:
        monitor.stop()
    verify_parent_preserved(directory)
    save(directory/"execution.json", {"controller_exit_status": 0, "stop_reason": result["stop_reason"],
                                       "validation_status": result["validation"]["status"],
                                       "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("Stop:", result["stop_reason"], flush=True)
    print(f"New pairs: {result['new_evaluation_attempts']}; additional active minutes: {result['continuation_active_seconds']/60:.3f}; calendar minutes: {result['continuation_calendar_seconds']/60:.3f}", flush=True)
    print("Validation:", result["validation"]["status"], flush=True)
    return SimpleNamespace(directory=directory, result=result, prior=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--directory")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-new-evaluations", type=int, default=1700)
    parser.add_argument("--max-minutes", type=float, default=180.)
    args = parser.parse_args()
    continue_experiment(parent_directory=args.parent, directory=args.directory,
                        max_iterations=args.max_iterations, max_new_evaluations=args.max_new_evaluations,
                        max_minutes=args.max_minutes)
