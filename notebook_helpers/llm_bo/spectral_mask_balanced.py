"""Balanced six-level joint sampling followed by bounded, feasible local search.

The optical worker is unchanged. Its legacy all-mode D_NL is retained under an
explicit total-field name; both optimization scores are recomputed from LP01.
"""
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import threading
import time

import numpy as np

from . import spectral_mask as sm
from .spectral_mask_gd import BudgetReached, save
from .spectral_mask_cost_monitor import CostMonitor
from .spectral_mask_resources import ResourceMonitor

OBJECTIVES = ("D_NL", "D_IO")
NOTEBOOK_NAME = "spectral_mask_balanced_joint_local.ipynb"


def balanced_masks(seed=42, samples=300, bands=100):
    """Give each factor exactly samples/6 copies of every allowed amplitude."""
    if samples <= 0 or samples % 6 or bands <= 0:
        raise ValueError("Use a positive multiple of six samples and positive band count.")
    rng = np.random.default_rng(seed)
    ids = np.column_stack([rng.permutation(np.repeat(np.arange(6), samples // 6))
                           for _ in range(bands)])
    counts = np.array([(ids == k).sum(axis=0) for k in range(6)])
    assert ids.shape == (samples, bands) and np.all(counts == samples // 6)
    return ids / 5


def coarse_point(factors):
    """Validate the six-level grid without silently projecting arbitrary inputs."""
    a = np.asarray(factors, dtype=float)
    if (a.shape != (100,) or not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1))
            or not np.allclose(5*a, np.rint(5*a), rtol=0, atol=1e-10)):
        raise ValueError("Exactly 100 factors on {0, 0.2, 0.4, 0.6, 0.8, 1} are required.")
    return {"factors": (np.rint(5*a) / 5).tolist()}


def distortion(a, b):
    """Compute normalized complex overlap distortion without time alignment."""
    x, y = np.asarray(a).ravel(), np.asarray(b).ravel()
    nx, ny = float(np.sum(abs(x)**2)), float(np.sum(abs(y)**2))
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and nx > 0 and ny > 0):
        raise ValueError("Undefined overlap for zero-norm or nonfinite fields.")
    fidelity = float(abs(np.sum(x.conj()*y))**2 / (nx*ny))
    if not -1e-12 <= fidelity <= 1+1e-12:
        raise ValueError("Invalid overlap normalization.")
    return 1 - float(np.clip(fidelity, 0, 1))


def field_scores(path):
    """Score LP01 input/output and linear/nonlinear overlaps from saved fields."""
    with np.load(path) as fields:
        a, linear, nonlinear = (fields[k] for k in ("input", "linear", "nonlinear"))
        return {"D_NL": distortion(linear[:, 0, :], nonlinear[:, 0, :]),
                "D_IO": distortion(a[:, 0, :], nonlinear[:, 0, :]),
                "D_NL_total_field_diagnostic": distortion(linear, nonlinear),
                "D_IO_total_field_diagnostic": distortion(a, nonlinear)}


def perturb(factors, rng):
    """Change 2–10 distinct bands by one amplitude level, respecting the bounds."""
    ids = np.rint(5*np.asarray(factors)).astype(int)
    bands = rng.choice(100, size=int(rng.integers(2, 11)), replace=False)
    signs = rng.choice([-1, 1], len(bands))
    signs[ids[bands] == 0] = 1
    signs[ids[bands] == 5] = -1
    ids[bands] += signs
    return coarse_point(ids/5)


def accepts(current, candidate, objective, tolerance=1e-10):
    """Accept only a valid, brightness-feasible improvement in the chosen score."""
    return bool(candidate["feasible"] and candidate[objective] < current[objective]-tolerance)


class Telemetry:
    """Sample process-tree CPU/RSS and device-wide GPU activity from run start."""
    def __init__(self, directory, started):
        self.directory = directory
        self.cost = CostMonitor(directory, os.getpid(), interval=2.)
        self.gpu = ResourceMonitor(directory, started=started, interval=5.)
        self.stop_event = threading.Event()

    def start(self):
        self.cost.sample()
        self.gpu.start()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while not self.stop_event.wait(2):
            try:
                self.cost.sample()
            except Exception as error:
                with (self.directory/"telemetry_errors.log").open("a") as handle:
                    handle.write(f"{type(error).__name__}: {error}\n")

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        self.cost.sample()
        cost = self.cost.summarize()
        cost["memory_basis"] = "Sampled process-tree RSS from campaign start; shared pages may be counted more than once and between-sample peaks are unknown."
        cost["run_coverage"] = "Telemetry started before the first baseline solve, including compilation, search, validation and worker shutdown."
        save(self.directory/"cost_summary.json", cost)
        self.gpu.stop()


class Campaign:
    """Own the frozen budgets, paired evaluator, evidence ledger and local chains."""
    def __init__(self, directory, seed=42, max_minutes=90.):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started, self.wall_started = time.monotonic(), time.time()
        self.physics = sm.MaskConfig(power_kW=100., seed=seed)
        self.limits = {"joint_masks": 300, "local_proposals_per_objective": 150,
                       "max_search_pairs": 600, "max_total_pairs": 607,
                       "max_wall_seconds": 60*max_minutes, "validation_reserve_seconds": 900.,
                       "max_solver_seconds": 900., "local_chains": 3,
                       "minimum_improvement": 1e-10, "retries": 0}
        if max_minutes <= 15:
            raise ValueError("The wall allowance must exceed the 15-minute validation reserve.")
        self.evaluator = sm.MaskEvaluator(self.directory/"gnlse", self.physics)
        self.evaluator.max_evaluations = self.limits["max_total_pairs"]
        self.masks = balanced_masks(seed)
        self.references = {}
        self.state = {"status": "running", "stage": "setup", "seed": seed,
                      "method": "balanced discrete joint sampling + local refinement",
                      "metric_scope": "LP01; no fitted time shift", "limits": self.limits,
                      "config": asdict(self.physics), "observations": [], "joint": [],
                      "local": {}, "best_search": {}, "validation": {}, "cache_hits": 0,
                      "search_pairs": 0, "AI_calls": 0, "application_usefulness": "not_specified"}
        np.save(self.directory/"balanced_masks.npy", self.masks)
        np.savetxt(self.directory/"balanced_masks.csv", self.masks, delimiter=",", fmt="%.1f",
                   header=",".join(f"factor_{i+1:03d}" for i in range(100)), comments="")
        counts = np.array([(np.rint(self.masks*5) == k).sum(axis=0) for k in range(6)])
        save(self.directory/"design_checks.json", {
            "shape": list(self.masks.shape), "levels": (np.arange(6)/5).tolist(),
            "counts_by_level_and_factor": counts.tolist(), "all_counts_equal_50": bool(np.all(counts == 50)),
            "unique_masks": len(np.unique(self.masks, axis=0)),
            "interpretation": "Exact marginal balance of the 300 proposed masks; no claim of pairwise balance or global coverage. Feasible-subset and adaptive-trial balance is not guaranteed."})
        self.checkpoint()

    def elapsed(self):
        return time.monotonic() - self.started

    def wall_elapsed(self):
        return time.time() - self.wall_started

    def event(self, kind, **data):
        row = {"kind": kind, "elapsed_seconds": self.elapsed(),
               "calendar_seconds": self.wall_elapsed(), **data}
        with (self.directory/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False)+"\n")

    def checkpoint(self):
        self.state.update(active_runtime_seconds=self.elapsed(), calendar_elapsed_seconds=self.wall_elapsed(),
                          evaluation_attempts=self.evaluator.attempts)
        save(self.directory/"state.json", self.state)

    def measure(self, point, role, *, fine=False, refined=False, search=False, validation=False, **context):
        point = coarse_point(point["factors"])
        grid = self.physics.grid(refined)
        dz = self.physics.check_dz_m if fine else self.physics.search_dz_m
        key = (sm.point_key(point), grid["Nt"], grid["time_window_ps"], dz)
        cached = key in self.evaluator.cache
        deadline = self.limits["max_wall_seconds"] - (0 if validation else self.limits["validation_reserve_seconds"])
        remaining = deadline-self.wall_elapsed()
        if remaining <= 1:
            raise BudgetReached("wall-time allowance exhausted")
        if not cached and search and self.state["search_pairs"] >= self.limits["max_search_pairs"]:
            raise BudgetReached("600 search-pair allowance exhausted")
        self.evaluator.config = replace(self.physics, solver_timeout_seconds=min(remaining, 900.))
        self.event("objective_request", point=point, role=role, cached=cached, resolution={**grid, "dz_m": dz}, **context)
        request_start = self.elapsed()
        done = threading.Event()
        def watch_deadline():
            while not done.wait(.25):
                if self.wall_elapsed() >= deadline:
                    self.evaluator.responses.put("CALENDAR_DEADLINE_REACHED")
                    return
        watchdog = threading.Thread(target=watch_deadline, daemon=True)
        if not cached:
            if search:
                self.state["search_pairs"] += 1
            watchdog.start()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                measured = self.evaluator.measure(point, fine=fine, refined=refined)
            scores = field_scores(measured["artifacts"]["fields"])
            if abs(scores["D_NL_total_field_diagnostic"] - measured["D_NL"]) > 1e-10:
                raise ValueError("Saved-field total-field overlap disagrees with the optical worker.")
            ref_key = "refined" if refined else "main"
            photons = measured["metrics"]["total_photons"]
            if ref_key not in self.references:
                if not fine or point["factors"] != [1.]*100:
                    raise ValueError("Measure the matched unmasked fine reference first.")
                if not measured["valid"] or not np.isfinite(photons) or photons <= 0:
                    raise ValueError("Invalid brightness reference.")
                self.references[ref_key] = photons
            ratio = photons/self.references[ref_key]
            row = {"evaluation": measured["evaluation"], "role": role, "point": point, **scores,
                   "brightness_ratio": ratio, "brightness_reference_photons": self.references[ref_key],
                   "valid": measured["valid"], "feasible": bool(measured["valid"] and ratio >= sm.FLOOR),
                   "resolution": measured["resolution"], "artifacts": measured["artifacts"],
                   "diagnostics": measured["metrics"], "cached": cached, "search": search,
                   "elapsed_seconds": self.elapsed(), "calendar_seconds": self.wall_elapsed(),
                   "request_seconds": self.elapsed()-request_start, **context}
            self.state["observations"].append(row)
            self.state["cache_hits"] += int(cached)
            save(Path(measured["artifacts"]["fields"]).parent/"lp01_scores.json", scores)
            self.event("objective_result", **row)
            print(f"Pair {row['evaluation']:03d} {role}: LP01 D_NL={row['D_NL']:.7f}, "
                  f"D_IO={row['D_IO']:.7f}, brightness={ratio:.3%}, feasible={row['feasible']} "
                  f"[{self.elapsed()/60:.1f} min]{' cache' if cached else ''}", flush=True)
            return row
        except BaseException as error:
            self.event("objective_failure", role=role, point=point, error=f"{type(error).__name__}: {error}", **context)
            raise
        finally:
            done.set()
            if watchdog.is_alive():
                watchdog.join(timeout=1)
            self.checkpoint()

    def baseline(self):
        unity = {"factors": [1.]*100}
        self.state["stage"] = "baseline"
        self.state["baseline_fine"] = self.measure(unity, "unmasked baseline fine", fine=True)
        self.state["baseline"] = self.measure(unity, "unmasked baseline search")
        if not self.state["baseline"]["feasible"]:
            raise ValueError("Unmasked search baseline failed feasibility.")
        for objective in OBJECTIVES:
            self.state["best_search"][objective] = self.state["baseline"]
        self.checkpoint()

    def sample(self, index, mask):
        self.state["stage"] = "joint sampling"
        row = self.measure(coarse_point(mask), "balanced joint", search=True, sample=index+1)
        self.state["joint"].append(row)
        for objective in OBJECTIVES:
            if accepts(self.state["best_search"][objective], row, objective):
                self.state["best_search"][objective] = row
                self.event("joint_incumbent", objective=objective, sample=index+1, evaluation=row["evaluation"])
        self.checkpoint()
        return row

    def start_local(self, objective):
        self.state["stage"] = f"local {objective}"
        self.state["local"][objective] = {"proposals": [], "restoration": [], "accepted": [], "chains": []}
        data = self.state["local"][objective]
        pool = [self.state["baseline"], *self.state["joint"]]
        feasible = sorted((r for r in pool if r["feasible"]), key=lambda r: r[objective])
        starts = []
        for row in feasible:
            a = np.asarray(row["point"]["factors"])
            if all(np.sqrt(np.mean((a-np.asarray(s["point"]["factors"]))**2)) >= .15 for s in starts):
                starts.append(row)
            if len(starts) == 3:
                break
        # Restore only when fewer than three diverse feasible starts were found.
        # Each lift changes field-amplitude factors by +0.2, with a ceiling of 1.
        candidates = sorted((r for r in self.state["joint"] if not r["feasible"]),
                            key=lambda r: (-r["brightness_ratio"], r[objective]))
        for seed in candidates:
            if len(starts) >= 3 or len(data["restoration"]) >= 15:
                break
            for lift in range(1, 6):
                ids = np.minimum(5, np.rint(5*np.asarray(seed["point"]["factors"]))+lift)
                row = self.measure(coarse_point(ids/5), "feasibility restoration", search=True,
                                   objective=objective, seed_evaluation=seed["evaluation"], lift=lift)
                data["restoration"].append(row)
                self.event("restoration_decision", objective=objective, evaluation=row["evaluation"],
                           feasible=row["feasible"], lift=lift)
                if row["feasible"]:
                    if sm.point_key(row["point"]) not in {sm.point_key(s["point"]) for s in starts}:
                        starts.append(row)
                    if accepts(self.state["best_search"][objective], row, objective):
                        self.state["best_search"][objective] = row
                    break
        data["starts"] = starts
        data["chains"] = list(starts)
        data["restoration_explanation"] = "When fewer than three diverse feasible starts exist, lift the brightest infeasible sampled masks toward unity by +0.2 per band until measured feasible. Count all these trials within this objective's 150-proposal allowance; do not relabel them as balanced samples."
        self.checkpoint()
        return data["chains"]

    def local_step(self, objective, iteration, rng):
        data = self.state["local"][objective]
        chain = iteration % len(data["chains"])
        current = data["chains"][chain]
        point = perturb(current["point"]["factors"], rng)
        proposal = self.measure(point, "local proposal", search=True, objective=objective,
                                iteration=iteration+1, chain=chain+1, from_evaluation=current["evaluation"])
        accepted = accepts(current, proposal, objective)
        decision = {"objective": objective, "iteration": iteration+1, "chain": chain+1,
                    "from_evaluation": current["evaluation"], "evaluation": proposal["evaluation"],
                    "accepted": accepted, "reason": "feasible improvement" if accepted else
                    "brightness or validity constraint" if not proposal["feasible"] else "no objective improvement",
                    "elapsed_seconds": self.elapsed(), "point": point,
                    "D_NL": proposal["D_NL"], "D_IO": proposal["D_IO"],
                    "brightness_ratio": proposal["brightness_ratio"]}
        data["proposals"].append(decision)
        if accepted:
            data["chains"][chain] = proposal
            data["accepted"].append(decision)
            if accepts(self.state["best_search"][objective], proposal, objective):
                self.state["best_search"][objective] = proposal
        self.event("step_accepted" if accepted else "step_rejected", **decision)
        self.checkpoint()
        return proposal, accepted

    def validate(self):
        self.state["stage"] = "validation"
        try:
            refined_base = self.measure({"factors": [1.]*100}, "unmasked baseline refined",
                                        fine=True, refined=True, validation=True)
            self.state["baseline_refined"] = refined_base
        except Exception as error:
            self.state["validation_error"] = f"{type(error).__name__}: {error}"
            self.checkpoint()
            return
        for objective in OBJECTIVES:
            check_names = ("baseline_step", "baseline_grid", "selected_step", "selected_grid",
                           "fine_brightness", "refined_brightness", "finite_objectives")
            entry = {"status": "not_run", "checks": {k: "not_run" for k in check_names},
                     "tolerance": sm.TOLERANCE}
            self.state["validation"][objective] = entry
            try:
                best = self.state["best_search"][objective]
                fine = self.measure(best["point"], f"{objective} selected fine", fine=True, validation=True)
                refined = self.measure(best["point"], f"{objective} selected refined", fine=True, refined=True, validation=True)
                entry.update(selected_search=best, selected_fine=fine, selected_refined=refined)
                deltas = {"baseline_step": abs(self.state["baseline"][objective]-self.state["baseline_fine"][objective]),
                          "baseline_grid": abs(self.state["baseline_fine"][objective]-refined_base[objective]),
                          "selected_step": abs(best[objective]-fine[objective]),
                          "selected_grid": abs(fine[objective]-refined[objective])}
                entry["absolute_differences"] = deltas
                entry["checks"].update({k: "pass" if v <= sm.TOLERANCE else "fail" for k, v in deltas.items()})
                entry["checks"].update(fine_brightness="pass" if fine["feasible"] else "fail",
                                       refined_brightness="pass" if refined["feasible"] else "fail",
                                       finite_objectives="pass" if all(r["valid"] for r in (best, fine, refined)) else "fail")
                entry["status"] = "pass" if all(v == "pass" for v in entry["checks"].values()) else "fail"
                entry["result_label"] = "numerically validated and feasible" if entry["status"] == "pass" else "preliminary"
            except Exception as error:
                entry.update(status="incomplete", error=f"{type(error).__name__}: {error}")
            self.event("validation", objective=objective, **entry)
            self.checkpoint()

    def run(self):
        try:
            self.baseline()
            for index, mask in enumerate(self.masks):
                self.sample(index, mask)
            for objective in OBJECTIVES:
                self.start_local(objective)
                used = len(self.state["local"][objective]["restoration"])
                rng = np.random.default_rng(self.state["seed"] + (1001 if objective == "D_NL" else 2001))
                for iteration in range(self.limits["local_proposals_per_objective"]-used):
                    self.local_step(objective, iteration, rng)
            self.state["stop_reason"] = "Completed the 300-mask batch and both 150-proposal local allowances; no convergence or global-optimum claim."
        except Exception as error:
            self.state["stop_reason"] = f"{type(error).__name__}: {error}"
            self.event("search_stopped", reason=self.state["stop_reason"])
        finally:
            if self.state["best_search"]:
                self.validate()
            self.evaluator.close()
            self.state["status"] = "completed" if len(self.state["validation"]) == 2 else "incomplete"
            self.state["stage"] = "finished"
            self.state["result_label"] = "numerically validated and feasible" if (
                len(self.state["validation"]) == 2 and all(v["status"] == "pass" for v in self.state["validation"].values())) else "preliminary"
            self.checkpoint()
        return self.state


def load_run(directory):
    """Load saved observations and configuration without new optical evaluations."""
    path = Path(directory).resolve()
    return SimpleNamespace(directory=path, result=json.loads((path/"state.json").read_text()))


def run_experiment(directory=None, *, seed=42, max_minutes=90.):
    """Execute one separately saved campaign with resource telemetry and frozen sources."""
    if directory is None:
        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_spectral_mask_balanced_%H%M%S")
        directory = sm.ROOT/"output"/stamp
    campaign = Campaign(directory, seed, max_minutes)
    source_dir = campaign.directory/"sources"
    source_dir.mkdir()
    sources = [Path(__file__), Path(sm.__file__), campaign.evaluator.worker,
               campaign.evaluator.worker.with_name("dio_worker.jl"),
               sm.ROOT/"notebook_helpers/ai_augmented_loop/evaluators.py"]
    hashes = {}
    for path in sources:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, source_dir/path.name)
    save(campaign.directory/"run.json", {
        "created_local": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "command": sys.argv, "physics": asdict(campaign.physics), "limits": campaign.limits,
        "input": sm.input_description(campaign.physics), "evaluator": campaign.evaluator.identity,
        "source_sha256": hashes, "seed": seed, "objectives": list(OBJECTIVES), "objective_mode": "LP01",
        "lower_is_better": True, "brightness_floor": .5, "brightness_modes": "all three modes",
        "brightness_reference": "unmasked nonlinear output at 1 mm; matched time grid for refined checks",
        "mask": "100 equal-width angular-frequency bands spanning 99.9% of original Gaussian energy; one shared field-amplitude mask, unchanged tails, no renormalization",
        "method": "300 exactly marginal-balanced masks followed by 150 proposals per objective, including any feasibility restoration; three local chains, 2–10 bands changed by +/-0.2",
        "validation": "5 mm -> 1 mm on Nt=16384, then 1 mm on Nt=65536 with double time window; absolute objective tolerance 2e-4; failure blocks promotion",
        "evaluation_definition": "One matched linear/nonlinear pair; count nonlinear GNLSE solves separately",
        "runtime_boundary": "Campaign setup, baseline, search, validation, and worker shutdown; report/notebook rendering recorded separately",
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "python": platform.python_version(), "logical_cpus": os.cpu_count()},
        "AI_calls": 0, "electricity_and_money": "not measured"})
    telemetry = Telemetry(campaign.directory, campaign.started).start()
    print("RUN_DIRECTORY", campaign.directory, flush=True)
    try:
        campaign.run()
    finally:
        telemetry.stop()
    print("FINISHED", campaign.state["result_label"], campaign.state["stop_reason"], flush=True)
    return load_run(campaign.directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-minutes", type=float, default=90.)
    args = parser.parse_args()
    run_experiment(args.directory, seed=args.seed, max_minutes=args.max_minutes)
