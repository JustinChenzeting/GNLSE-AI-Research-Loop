"""Exhaust all 729 six-band masks without modifying the earlier 100-band runs."""
from dataclasses import asdict, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import contextlib
import hashlib
import io
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from ai_augmented_loop.evaluators import FIBER_SETUP, OpticsEvaluator
from llambo_mvp.llm_client import write_json
from . import spectral_mask as sm
from .spectral_mask_balanced import Campaign as BaseCampaign, Telemetry, field_scores, load_run
from .spectral_mask_gd import BudgetReached, save

BANDS = 6
LEVELS = (0.0, 0.5, 1.0)
OBJECTIVES = ("D_NL", "D_IO")
NOTEBOOK_NAME = "spectral_mask_six_band_exhaustive.ipynb"


def all_masks():
    """Enumerate the Cartesian product once, with the sixth factor changing fastest."""
    return np.array(list(product(LEVELS, repeat=BANDS)), dtype=float)


def six_point(factors):
    """Require six exact field-amplitude factors from the declared three levels."""
    a = np.asarray(factors, dtype=float)
    if a.shape != (BANDS,) or not np.all(np.isfinite(a)) or not np.all(np.isin(a, LEVELS)):
        raise ValueError("Exactly six factors in {0, 0.5, 1} are required; no rounding or clipping.")
    return {"factors": a.tolist()}


def point_key(point):
    """Encode a verified six-band mask using exact half-step integer IDs."""
    return tuple(int(2*v) for v in six_point(point["factors"])["factors"])


def sampled_mask(omega, factors, halfwidth):
    """Apply six equal angular-frequency bins while leaving the spectral tails at one."""
    a = np.asarray(six_point(factors)["factors"])
    omega = np.asarray(omega)
    bins = np.floor((omega+halfwidth)/(2*halfwidth/BANDS)+1e-10).astype(int)
    inside = (bins >= 0) & (bins < BANDS) & (omega >= -halfwidth-1e-10) & (omega < halfwidth-1e-10)
    mask = np.ones(omega.shape)
    mask[inside] = a[bins[inside]]
    return mask, np.where(inside, bins+1, 0)


def input_description(config, refined=False):
    """Describe the six physical bands on the unchanged search or refined time grid."""
    grid = config.grid(refined)
    dt = grid["time_window_ps"]/grid["Nt"]
    t = (np.arange(grid["Nt"])-grid["Nt"]//2)*dt
    omega = np.fft.fftshift(np.fft.fftfreq(len(t), dt))*2*np.pi
    pulse = np.exp(-2*np.log(2)*(t/FIBER_SETUP["pulse_fwhm_ps"])**2)
    energy = abs(np.fft.fftshift(np.fft.ifft(pulse)))**2
    _, bands = sampled_mask(omega, [1.]*BANDS, config.halfwidth)
    counts = np.bincount(bands, minlength=BANDS+1)[1:]
    weights = np.bincount(bands, weights=energy, minlength=BANDS+1)/energy.sum()
    assert counts.min() > 0
    return {"band_edges_detuning_rad_ps": np.linspace(-config.halfwidth, config.halfwidth, BANDS+1).tolist(),
            "band_energy_fractions": weights[1:].tolist(), "outside_band_energy_fraction": float(weights[0]),
            "samples_per_band": counts.tolist(), "time_window_ps": grid["time_window_ps"],
            "Nt": grid["Nt"], "dt_ps": dt, "omega_sample_spacing_rad_ps": 2*np.pi/grid["time_window_ps"]}


class SixBandEvaluator(OpticsEvaluator):
    """Run and cache matched pairs with six bands using the unchanged fiber solver."""
    def __init__(self, directory, config):
        setup = dict(FIBER_SETUP, time_window_ps=config.window_ps, mask_halfwidth_rad_ps=config.halfwidth,
                     allowed_Nt=[config.nt, 4*config.nt], allowed_windows_ps=[config.window_ps, 2*config.window_ps])
        super().__init__(sm.ROOT, setup, objective="D_NL")
        self.worker = Path(__file__).with_name("spectral_mask_six_worker.jl")
        self.config = config
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.attempts, self.cache, self.records = 0, {}, []
        self.max_evaluations = 735
        self.identity.update(launch="six_band_spectral_amplitude_mask", mask_bands=BANDS,
                             allowed_amplitude_factors=list(LEVELS), objective_modes=[1],
                             raw_worker_score="D_NL_total_field", brightness_modes=[1,2,3],
                             brightness_reference="unmasked_equal_amplitude_fine_baseline",
                             power_renormalization="none", mask_halfwidth_rad_ps=config.halfwidth,
                             worker_sha256=hashlib.sha256(self.worker.read_bytes()).hexdigest(),
                             shared_worker_sha256=hashlib.sha256(self.worker.with_name("dio_worker.jl").read_bytes()).hexdigest(),
                             adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_json(self.directory/"evaluator.json", self.identity)

    def measure(self, point, *, fine=False, refined=False):
        point = six_point(point["factors"])
        grid = self.config.grid(refined)
        resolution = dict(grid, dz_m=self.config.check_dz_m if fine else self.config.search_dz_m)
        key = (point_key(point), resolution["Nt"], resolution["time_window_ps"], resolution["dz_m"])
        if key in self.cache:
            return dict(self.cache[key], cached=True)
        if self.attempts >= self.max_evaluations:
            raise BudgetReached("735-pair total allowance exhausted")
        self.attempts += 1
        directory = self.directory/f"evaluation_{self.attempts:03d}"
        directory.mkdir()
        row = {"evaluation": self.attempts, "point": point, "objective": "D_NL_total_field", "status": "started",
               "power_kW_before_mask": self.config.power_kW, "resolution": resolution, "cached": False}
        self.records.append(row)
        write_json(directory/"measurement.json", row)
        started = time.monotonic()
        try:
            value = super().__call__({**point, "power_kW": self.config.power_kW}, resolution, directory,
                                     self.config.solver_timeout_seconds)
            row.update(status="ok", D_NL=value["score"], valid=value["checks"]["valid"],
                       checks=value["checks"], metrics=value["metrics"], artifacts=value["artifacts"])
            self.cache[key] = row
            return row
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["wall_seconds"] = time.monotonic()-started
            write_json(directory/"measurement.json", row)


class SixBandCampaign(BaseCampaign):
    """Own a finite complete enumeration, per-objective rankings and finer checks."""
    def __init__(self, directory, max_minutes=120.):
        if max_minutes <= 15:
            raise ValueError("The time allowance must exceed the 15-minute check reserve.")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started, self.wall_started = time.monotonic(), time.time()
        # Keep the original 100-band run's time window; only the mask partition changes.
        self.physics = sm.MaskConfig(power_kW=100.)
        self.limits = {"sweep_masks": 729, "max_search_pairs": 729, "max_total_pairs": 735,
                       "max_wall_seconds": 60*max_minutes, "validation_reserve_seconds": 900.,
                       "max_solver_seconds": 900., "retries": 0}
        self.evaluator = SixBandEvaluator(self.directory/"gnlse", self.physics)
        self.masks, self.references = all_masks(), {}
        self.state = {"status": "running", "stage": "setup", "method": "exhaustive six-band three-level sweep",
                      "metric_scope": "LP01; no fitted time shift", "limits": self.limits,
                      "config": asdict(self.physics), "observations": [], "sweep": [], "best_search": {},
                      "validation": {}, "cache_hits": 0, "search_pairs": 0, "AI_calls": 0,
                      "application_usefulness": "not_specified"}
        np.save(self.directory/"all_masks.npy", self.masks)
        np.savetxt(self.directory/"all_masks.csv", self.masks, delimiter=",", fmt="%.1f",
                   header=",".join(f"factor_{i+1}" for i in range(BANDS)), comments="")
        counts = np.array([(self.masks == level).sum(axis=0) for level in LEVELS])
        assert self.masks.shape == (729,6) and len(np.unique(self.masks,axis=0)) == 729
        assert np.all(counts == 243)
        save(self.directory/"design_checks.json", {"shape": [729,6], "levels": list(LEVELS),
            "unique_masks": 729, "counts_by_level_and_factor": counts.tolist(),
            "all_counts_equal_243": True, "order": "Cartesian product; band 6 changes fastest",
            "interpretation": "Every combination in the declared finite six-band space is included, including infeasible masks. No adaptive restoration or local refinement is added."})
        self.checkpoint()

    def measure(self, point, role, *, fine=False, refined=False, search=False, validation=False, **context):
        point = six_point(point["factors"])
        grid = self.physics.grid(refined)
        dz = self.physics.check_dz_m if fine else self.physics.search_dz_m
        key = (point_key(point), grid["Nt"], grid["time_window_ps"], dz)
        cached = key in self.evaluator.cache
        deadline = self.limits["max_wall_seconds"] - (0 if validation else self.limits["validation_reserve_seconds"])
        remaining = deadline-self.wall_elapsed()
        if remaining <= 1:
            raise BudgetReached("wall-time allowance exhausted")
        if not cached and search and self.state["search_pairs"] >= self.limits["max_search_pairs"]:
            raise BudgetReached("729 search-pair allowance exhausted")
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
                if not fine or point["factors"] != [1.]*BANDS:
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
        unity = {"factors": [1.]*BANDS}
        self.state["stage"] = "baseline"
        self.state["baseline_fine"] = self.measure(unity, "unmasked baseline fine", fine=True)
        self.state["baseline"] = self.measure(unity, "unmasked baseline search", search=True)
        if not self.state["baseline"]["feasible"]:
            raise ValueError("Unmasked search baseline failed feasibility.")
        for objective in OBJECTIVES:
            self.state["best_search"][objective] = self.state["baseline"]
        self.checkpoint()


    def validate(self):
        self.state["stage"] = "validation"
        try:
            refined_base = self.measure({"factors": [1.]*BANDS}, "unmasked baseline refined",
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


    def sample(self, index, mask):
        """Measure one enumerated mask and update both feasible objective minima."""
        self.state["stage"] = "exhaustive sweep"
        row = self.measure(six_point(mask), "exhaustive sweep", search=True, sample=index+1)
        self.state["sweep"].append(row)
        for objective in OBJECTIVES:
            if row["feasible"] and row[objective] < self.state["best_search"][objective][objective]:
                self.state["best_search"][objective] = row
                self.event("sweep_incumbent", objective=objective, sample=index+1, evaluation=row["evaluation"])
        self.checkpoint()
        return row

    def run(self):
        """Evaluate all combinations, validate both finalists, and preserve incomplete runs."""
        try:
            self.baseline()
            for index, mask in enumerate(self.masks):
                self.sample(index, mask)
            self.state["stop_reason"] = "All 729 combinations evaluated; selected minima are for the declared finite search grid."
        except Exception as error:
            self.state["stop_reason"] = f"{type(error).__name__}: {error}"
            self.event("search_stopped", reason=self.state["stop_reason"])
        finally:
            if self.state["best_search"]:
                self.validate()
            self.evaluator.close()
            complete = len(self.state["sweep"]) == 729 and len(self.state["validation"]) == 2
            self.state["status"] = "completed" if complete else "incomplete"
            self.state["stage"] = "finished"
            self.state["result_label"] = "numerically validated and feasible" if complete and all(
                v["status"] == "pass" for v in self.state["validation"].values()) else "preliminary"
            self.checkpoint()
        return self.state


def run_experiment(directory=None, *, max_minutes=120.):
    """Run the complete finite sweep with frozen sources and CPU/GPU telemetry."""
    if directory is None:
        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_spectral_mask_six_band_%H%M%S_latest")
        directory = sm.ROOT/"output"/stamp
    campaign = SixBandCampaign(directory, max_minutes)
    source_dir = campaign.directory/"sources"
    source_dir.mkdir()
    sources = [Path(__file__), Path(sm.__file__), Path(__file__).with_name("spectral_mask_balanced.py"),
               campaign.evaluator.worker, campaign.evaluator.worker.with_name("dio_worker.jl"),
                sm.ROOT/"notebook_helpers/ai_augmented_loop/evaluators.py",
               Path(__file__).with_name("spectral_mask_cost_monitor.py"),
               Path(__file__).with_name("spectral_mask_resources.py")]
    hashes = {}
    for path in sources:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, source_dir/path.name)
    save(campaign.directory/"run.json", {
        "created_local": datetime.now(ZoneInfo("America/New_York")).isoformat(), "command": sys.argv,
        "physics": asdict(campaign.physics), "limits": campaign.limits,
        "input": input_description(campaign.physics), "refined_input": input_description(campaign.physics, True),
        "evaluator": campaign.evaluator.identity, "source_sha256": hashes, "objectives": list(OBJECTIVES),
        "objective_mode": "LP01", "lower_is_better": True, "brightness_floor": sm.FLOOR,
        "brightness_modes": "all three modes", "brightness_reference": "unmasked NL output at 1 mm; matched time grid for refined checks",
        "mask": "6 equal-width angular-frequency bands across the original 99.9%-energy bandwidth; shared field-amplitude factors {0,0.5,1}, unchanged tails, no renormalization",
        "band_index_direction": "increasing angular-frequency detuning, not increasing wavelength",
        "method": "Exhaustive Cartesian product; 729 unique masks; no local refinement or feasibility restoration",
        "validation": "5 mm -> 1 mm on Nt=16384, then 1 mm on Nt=65536 with double window; absolute target-metric tolerance 2e-4; failures remain preliminary",
        "claim_boundary": "Exact enumeration of the declared search grid; finalist checks alone do not verify the full ranking at finer resolution or the continuous-space optimum",
        "evaluation_definition": "One NL GNLSE plus one matched linear propagation per new pair; cache hits are free",
        "runtime_boundary": "Campaign setup, baselines, complete sweep, finer checks and worker shutdown; reporting separately timed",
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "logical_cpus": os.cpu_count()},
        "AI_optimizer_calls": 0, "electricity_and_money": "not measured"})
    notebook_dir = sm.ROOT/"notebooks"
    save(campaign.directory/"preserved_notebooks.json", {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in notebook_dir.rglob("*spectral_mask*.ipynb") if p.name != NOTEBOOK_NAME})
    save(campaign.directory/"environment.json", {
        "python_executable": sys.executable, "python_version": sys.version,
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "psutil", "pandas", "matplotlib", "nbformat", "nbclient", "nbconvert")},
        "Julia": "1.12.5; package source and Manifest hashes in evaluator identity",
        "threads": "CPU FFTW and BLAS each one thread, as specified by dio_worker.jl"})
    save(campaign.directory/"metric_key_map.json", {
        "state_and_tables": {"D_NL": "LP01 linear vs nonlinear", "D_IO": "LP01 input vs nonlinear"},
        "raw_measurement_D_NL": "D_NL_total_field; retained worker response key, not the optimized LP01 score",
        "total_field_diagnostics": "computed from raw fields; no separate total-field objective ranking"})
    save(campaign.directory/"telemetry_notes.json", {
        "CPU": "Observed process-tree counters; short-lived children may be missed. cost_monitor_CPU_seconds is the in-process controller counter and must not be added again.",
        "memory": "Sampled RSS sum, not a guaranteed continuous peak; shared pages can be counted twice.",
        "GPU": "Device-wide activity includes other applications. Solver uses CPU FFTW/BLAS; no GPU compute.",
        "energy_money": "Not measured"})
    telemetry = None
    print("RUN_DIRECTORY", campaign.directory, flush=True)
    try:
        telemetry = Telemetry(campaign.directory, campaign.started).start()
        campaign.run()
    finally:
        campaign.evaluator.close()
        if telemetry is not None:
            telemetry.stop()
    print("FINISHED", campaign.state["result_label"], campaign.state["stop_reason"], flush=True)
    return load_run(campaign.directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--max-minutes", type=float, default=120.)
    args = parser.parse_args()
    run_experiment(args.directory, max_minutes=args.max_minutes)
