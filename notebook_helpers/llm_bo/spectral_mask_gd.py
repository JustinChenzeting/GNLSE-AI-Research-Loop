"""Projected finite-step descent using the unchanged spectral-mask GNLSE evaluator."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import platform
import threading
import time

import numpy as np

from . import spectral_mask as sm

ROOT = sm.ROOT
SOURCE_NOTEBOOK = next((ROOT / "notebooks").rglob("dnl_llm_bo_spectral_mask.ipynb"))


def save(path, value):
    """Atomic checkpoints; events and original measurements remain separate."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def project(values):
    """Nearest allowed factor; ties round upward. No derivative through this map."""
    values = np.asarray(values, dtype=float)
    if values.shape != (sm.BANDS,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected 100 finite factors.")
    return {"factors": (np.floor(np.clip(values, 0, 1)*100 + .5 + 1e-12)/100).tolist()}


@dataclass(frozen=True)
class DescentConfig:
    max_iterations: int = 2
    max_evaluations: int = 450  # Matched linear/nonlinear pairs, including checks.
    max_wall_seconds: float = 2700.
    validation_reserve_seconds: float = 360.
    validation_reserve_pairs: int = 3  # Selected fine, baseline refined, selected refined.
    finite_difference_increment: float = .01
    max_factor_step: float = .10
    backtracking_factor: float = .5
    max_backtracks: int = 12
    minimum_improvement: float = 1e-10
    restoration_steps: int = 10

    def __post_init__(self):
        for name in ("max_iterations", "max_evaluations", "max_backtracks", "restoration_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_evaluations < 6 or self.validation_reserve_pairs != 3:
            raise ValueError("Reserve the three final checks and at least three initial pairs.")
        if (not np.isfinite(self.max_wall_seconds) or not np.isfinite(self.validation_reserve_seconds)
                or not 0 < self.validation_reserve_seconds < self.max_wall_seconds):
            raise ValueError("Reserve validation time inside a finite positive wall-time limit.")
        if self.finite_difference_increment != .01 or not 0 < self.max_factor_step <= 1:
            raise ValueError("Use 0.01 finite differences and a bounded initial step.")
        if not 0 < self.backtracking_factor < 1 or not self.minimum_improvement >= 0:
            raise ValueError("Invalid backtracking settings.")


class BudgetReached(RuntimeError):
    pass


class Campaign:
    def __init__(self, directory, physics, limits, evaluator, *, clock=time.monotonic, wall_clock=None):
        self.directory, self.physics, self.limits, self.evaluator = Path(directory), physics, limits, evaluator
        self.clock, self.started = clock, clock()
        self.wall_clock = wall_clock or (time.time if clock is time.monotonic else clock)
        self.wall_started = self.wall_clock()
        self.reference = None
        self.state = {"status": "running", "method": "projected finite-difference gradient descent",
                      "config": asdict(physics), "descent_config": asdict(limits), "history": [],
                      "accepted_steps": [], "gradients": [], "restoration": [], "cache_hits": 0,
                      "ai_attempts": 0, "application_usefulness": "not_specified"}

    def elapsed(self):
        return self.clock() - self.started

    def wall_elapsed(self):
        return self.wall_clock() - self.wall_started

    def event(self, kind, **data):
        row = {"kind": kind, "elapsed_seconds": self.elapsed(),
               "calendar_elapsed_seconds": self.wall_elapsed(), **data}
        with (self.directory / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def checkpoint(self):
        self.state.update(total_wall_seconds=self.elapsed(), evaluation_attempts=self.evaluator.attempts)
        self.state.update(active_runtime_seconds=self.elapsed(), calendar_elapsed_seconds=self.wall_elapsed(),
                          clock_policy="calendar deadline; monotonic active runtime recorded separately")
        save(self.directory / "state.json", self.state)

    def measure(self, point, role, *, fine=False, refined=False, validation=False, **context):
        point = sm.check_point(point)
        grid = self.physics.grid(refined)
        resolution = dict(grid, dz_m=self.physics.check_dz_m if fine else self.physics.search_dz_m)
        key = (sm.point_key(point), grid["Nt"], grid["time_window_ps"], resolution["dz_m"])
        cached = key in self.evaluator.cache
        deadline = self.limits.max_wall_seconds - (0 if validation else self.limits.validation_reserve_seconds)
        remaining = deadline - self.wall_elapsed()
        cap = self.limits.max_evaluations - (0 if validation else self.limits.validation_reserve_pairs)
        if remaining <= 1:
            raise BudgetReached("wall-time limit reached; final checks may remain not_run")
        if not cached and self.evaluator.attempts >= cap:
            raise BudgetReached("evaluation allowance reached; final-check allowance retained")
        self.evaluator.config = replace(self.physics, solver_timeout_seconds=min(self.physics.solver_timeout_seconds, remaining))
        self.event("objective_request", role=role, point=point, resolution=resolution, cached=cached, **context)
        start = self.elapsed()
        done = threading.Event()
        def check_calendar_deadline():
            # macOS monotonic timers can exclude host suspension. Wake the existing
            # request wait on a calendar deadline; its normal error path stops Julia.
            while not done.wait(.25):
                if self.wall_elapsed() >= deadline:
                    self.evaluator.responses.put("CALENDAR_DEADLINE_REACHED")
                    return
        watchdog = None
        if not cached and hasattr(self.evaluator, "responses"):
            watchdog = threading.Thread(target=check_calendar_deadline, daemon=True)
            watchdog.start()
        try:
            measured = self.evaluator.measure(point, fine=fine, refined=refined)
        except BaseException as error:
            self.event("objective_failure", role=role, point=point, resolution=resolution,
                       error=f"{type(error).__name__}: {error}", **context)
            self.checkpoint()
            raise
        finally:
            done.set()
            if watchdog is not None:
                watchdog.join(timeout=1)
        if self.reference is None:
            if not fine or refined or sm.point_key(point) != (100,)*sm.BANDS:
                raise ValueError("Measure the unmasked 1 mm brightness reference first.")
            self.reference = measured["metrics"]["total_photons"]
            if not measured["valid"] or not self.reference > 0:
                raise ValueError("Invalid unmasked brightness reference.")
        row = sm.observation(measured, self.reference, role)
        row.update(elapsed_seconds=self.elapsed(), request_seconds=self.elapsed()-start, cached=cached,
                   brightness_reference_photons=self.reference)
        if cached:
            self.state["cache_hits"] += 1
        self.event("objective_result", **row, **context)
        self.checkpoint()
        return row

    def restore(self, initial):
        """Increase transmission along the seeded-mask-to-unity path, then stop."""
        if initial["feasible"]:
            self.state["restoration_status"] = "not_needed"
            return initial
        seed = np.asarray(initial["point"]["factors"])
        for step in range(1, self.limits.restoration_steps+1):
            fraction = step / self.limits.restoration_steps
            point = project(seed + fraction*(1-seed))
            self.event("restoration_proposal", fraction=fraction, point=point)
            row = self.measure(point, "feasibility restoration", fraction=fraction)
            self.state["history"].append(row)
            decision = {"fraction": fraction, "evaluation": row["evaluation"], "feasible": row["feasible"],
                        "D_NL": row["D_NL"], "brightness_ratio": row["brightness_ratio"],
                        "reason": "first measured feasible point" if row["feasible"] else "below floor or invalid"}
            self.state["restoration"].append(decision)
            self.event("restoration_decision", **decision)
            self.checkpoint()
            if row["feasible"]:
                self.state["restoration_status"] = "restored"
                return row
        raise RuntimeError("Restoration failed, including the already calibrated unmasked endpoint.")

    def gradient(self, current, iteration):
        ticks = np.asarray(sm.point_key(current["point"]))
        slopes = np.zeros(sm.BANDS)
        entries = []
        status = "partial"
        try:
            for i, tick in enumerate(ticks):
                lower, upper = current, current
                if tick > 0:
                    point = ticks.copy(); point[i] -= 1
                    lower = self.measure({"factors": (point/100).tolist()}, "gradient probe",
                                         iteration=iteration, band=i+1, side="minus")
                if tick < 100:
                    point = ticks.copy(); point[i] += 1
                    upper = self.measure({"factors": (point/100).tolist()}, "gradient probe",
                                         iteration=iteration, band=i+1, side="plus")
                if not lower["valid"] or not upper["valid"]:
                    raise ValueError(f"Undefined finite-step slope in band {i+1}.")
                delta = .01*(int(tick > 0)+int(tick < 100))
                slopes[i] = (upper["D_NL"]-lower["D_NL"])/delta
                entry = {"band": i+1, "scheme": "central" if 0 < tick < 100 else "one-sided",
                         "increment": .01, "denominator": delta, "lower_id": lower["evaluation"],
                         "upper_id": upper["evaluation"], "lower_D_NL": lower["D_NL"],
                         "upper_D_NL": upper["D_NL"], "slope": float(slopes[i])}
                entries.append(entry)
                self.event("finite_step_slope", iteration=iteration, **entry)
            status = "complete"
            return slopes
        finally:
            path = self.directory / f"gradient_{iteration:03d}.json"
            save(path, {"status": status, "iteration": iteration, "point": current["point"],
                        "center_evaluation": current["evaluation"], "entries": entries,
                        "elapsed_seconds": self.elapsed(), "interpretation": "finite-step slopes, not exact derivatives"})
            self.state["gradients"].append({"iteration": iteration, "status": status, "path": str(path),
                                             "completed_bands": len(entries)})
            self.checkpoint()

    def line_search(self, current, slopes, iteration):
        magnitude = float(np.max(np.abs(slopes)))
        if not np.isfinite(magnitude) or magnitude == 0:
            return None, "zero or invalid finite-step direction"
        alpha = self.limits.max_factor_step / magnitude
        x = np.asarray(current["point"]["factors"])
        for backtrack in range(self.limits.max_backtracks):
            point = project(x-alpha*slopes)
            self.event("step_proposal", iteration=iteration, backtrack=backtrack,
                       alpha=alpha, point=point, from_evaluation=current["evaluation"])
            if sm.point_key(point) == sm.point_key(current["point"]):
                self.event("step_rejected", iteration=iteration, backtrack=backtrack,
                           reason="projection gives unchanged mask", alpha=alpha, point=point)
                return None, "backtracking reached the factor-grid resolution"
            row = self.measure(point, "line search", iteration=iteration, backtrack=backtrack, alpha=alpha)
            self.state["history"].append(row)
            accepted = bool(row["feasible"] and row["D_NL"] < current["D_NL"]-self.limits.minimum_improvement)
            reason = ("feasible objective improvement" if accepted else
                      "brightness or numerical constraint" if not row["feasible"] else "no objective improvement")
            step = {"iteration": iteration, "backtrack": backtrack, "alpha": alpha,
                    "from_evaluation": current["evaluation"], "evaluation": row["evaluation"],
                    "point": point, "D_NL": row["D_NL"], "brightness_ratio": row["brightness_ratio"],
                    "elapsed_seconds": self.elapsed(), "reason": reason}
            self.event("step_accepted" if accepted else "step_rejected", **step)
            if accepted:
                self.state["accepted_steps"].append(step)
                self.checkpoint()
                return row, "accepted"
            self.checkpoint()
            alpha *= self.limits.backtracking_factor
        return None, "backtracking proposal limit reached"

    def validate(self, base, fine_base, best):
        checks = {name: "not_run" for name in (
            "baseline_step_agreement", "selected_step_agreement", "baseline_grid_agreement",
            "selected_grid_agreement", "selected_fine_and_refined_brightness", "finite_defined_objective")}
        checks["baseline_step_agreement"] = "pass" if abs(base["D_NL"]-fine_base["D_NL"]) <= sm.TOLERANCE else "fail"
        validation = {"status": "not_run", "checks": checks, "baseline_fine": fine_base,
                      "verified_improvement": False}
        self.state["validation"] = validation
        try:
            fine = self.measure(best["point"], "selected fine", fine=True, validation=True)
            validation["selected_fine"] = fine
            checks["selected_step_agreement"] = "pass" if abs(best["D_NL"]-fine["D_NL"]) <= sm.TOLERANCE else "fail"
            refined_base = self.measure(base["point"], "unmasked baseline refined", fine=True, refined=True, validation=True)
            refined_ref = refined_base["diagnostics"]["total_photons"]
            if not refined_base["valid"] or refined_ref <= 0:
                raise ValueError("Invalid refined brightness reference.")
            refined_base.update(brightness_ratio=1., feasible=refined_base["valid"],
                                brightness_reference_photons=refined_ref)
            validation["baseline_refined"] = refined_base
            checks["baseline_grid_agreement"] = "pass" if abs(fine_base["D_NL"]-refined_base["D_NL"]) <= sm.TOLERANCE else "fail"
            refined = self.measure(best["point"], "selected refined", fine=True, refined=True, validation=True)
            # Exactly the original protocol: use the unmasked reference on the matching refined grid.
            refined["brightness_ratio"] = refined["diagnostics"]["total_photons"] / refined_ref
            refined["feasible"] = bool(refined["valid"] and refined["brightness_ratio"] >= sm.FLOOR)
            refined["brightness_reference_photons"] = refined_ref
            validation["selected_refined"] = refined
            self.event("refined_brightness_normalization", reference_photons=refined_ref,
                       baseline_evaluation=refined_base["evaluation"], selected_evaluation=refined["evaluation"],
                       selected_brightness=refined["brightness_ratio"])
            checks["selected_grid_agreement"] = "pass" if abs(fine["D_NL"]-refined["D_NL"]) <= sm.TOLERANCE else "fail"
            checks["selected_fine_and_refined_brightness"] = "pass" if fine["feasible"] and refined["feasible"] else "fail"
            checks["finite_defined_objective"] = "pass" if all(r["valid"] for r in (fine_base, fine, refined_base, refined)) else "fail"
            validation["status"] = "pass" if all(v == "pass" for v in checks.values()) else "fail"
            validation["verified_improvement"] = bool(validation["status"] == "pass" and
                fine["D_NL"] < fine_base["D_NL"]-sm.TOLERANCE and
                refined["D_NL"] < refined_base["D_NL"]-sm.TOLERANCE)
        except Exception as error:
            validation.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        finally:
            self.event("validation", **validation)
            self.checkpoint()

    def run(self):
        unity = {"factors": [1.]*sm.BANDS}
        best = base = fine_base = None
        try:
            fine_base = self.measure(unity, "unmasked baseline fine", fine=True)
            base = self.measure(unity, "unmasked baseline")
            if not base["feasible"]:
                raise ValueError("Unmasked search baseline is not feasible.")
            self.state["history"].append(base)
            best = base
            initial = self.measure(sm.random_point(self.physics.seed), "seeded random mask")
            self.state["history"].append(initial)
            self.state["initial_point"] = initial["point"]
            current = self.restore(initial)
            self.state["restored_start"] = current
            if current["D_NL"] < best["D_NL"]:
                best = current
            for iteration in range(1, self.limits.max_iterations+1):
                print(f"Gradient {iteration}/{self.limits.max_iterations}: 100 finite-step slopes", flush=True)
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
            if best is not None:
                self.state["best_search"] = best
                self.validate(base, fine_base, best)
            self.evaluator.close()
            self.state["status"] = "completed" if best is not None else "failed"
            self.state["result_label"] = ("numerically validated and feasible" if
                self.state.get("validation", {}).get("status") == "pass" else "preliminary")
            self.checkpoint()
        return self.state


def load_run(directory):
    directory = Path(directory).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    return SimpleNamespace(directory=directory.resolve(), result=json.loads((directory / "state.json").read_text()), prior=None)


def run_experiment(*, seed=20261113, max_iterations=2, max_evaluations=450,
                   max_minutes=45., validation_minutes=6., load_directory=None):
    if load_directory is not None:
        run = load_run(load_directory)
        print("Loaded saved gradient-descent measurements:", run.directory)
        return run
    physics = sm.MaskConfig(power_kW=100., seed=seed)
    limits = DescentConfig(max_iterations=max_iterations, max_evaluations=max_evaluations,
                          max_wall_seconds=60*max_minutes, validation_reserve_seconds=60*validation_minutes)
    directory = ROOT / "output" / datetime.now(timezone.utc).strftime("%Y-%m-%d_gd_spectral_mask_%H%M%S_%fZ")
    directory.mkdir(parents=True)
    started = time.monotonic()
    wall_started = time.time()
    simulator = sm.MaskEvaluator(directory / "gnlse", physics)
    simulator.max_evaluations = limits.max_evaluations
    campaign = Campaign(directory, physics, limits, simulator)
    campaign.started = started
    campaign.wall_started = wall_started
    sources = [SOURCE_NOTEBOOK, Path(__file__), Path(sm.__file__), simulator.worker,
               simulator.worker.with_name("dio_worker.jl")]
    (directory / "sources").mkdir()
    hashes = {}
    for path in sources:
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        (directory / "sources" / path.name).write_bytes(data)
    save(directory / "run.json", {"physics": asdict(physics), "limits": asdict(limits),
         "input": sm.input_description(physics), "initial_point": sm.random_point(seed),
         "evaluator": simulator.identity, "source_sha256": hashes,
         "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                      "python": platform.python_version(), "cpu_count": os.cpu_count()},
         "gradient_descent_executed": True, "AI_calls": 0,
         "evaluation_definition": "one matched linear/nonlinear pair; two propagations",
         "restoration": "project(s_seed + t*(1-s_seed)), t=0.1,...,1; first measured feasible point",
         "selection": "best baseline/restored start/accepted descent iterate; probes and rejected proposals do not replace the iterate",
         "runtime_boundary": "controller setup through search, validation and worker shutdown; excludes notebook rendering",
         "clock_policy": "calendar wall-time deadline, with monotonic active runtime recorded separately"})
    print("Saving gradient descent to:", directory, flush=True)
    from .spectral_mask_resources import ResourceMonitor
    monitor = ResourceMonitor(directory, started=started).start()
    try:
        result = campaign.run()
    finally:
        monitor.stop()
    print("Stop:", result["stop_reason"], flush=True)
    print("Validation:", result.get("validation", {}).get("status", "not_run"), flush=True)
    print(f"Measured {result['evaluation_attempts']} pairs in {result['active_runtime_seconds']/60:.2f} active min "
          f"({result['calendar_elapsed_seconds']/60:.2f} calendar min); no AI calls.", flush=True)
    return SimpleNamespace(directory=directory, result=result, prior=None)


if __name__ == "__main__":
    run_experiment()
