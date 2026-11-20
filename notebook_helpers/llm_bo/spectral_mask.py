"""A bounded 100-band spectral-amplitude-mask experiment using the existing LLMBO protocol."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
import hashlib
import json
import math
import time

import numpy as np

from ai_augmented_loop.evaluators import FIBER_SETUP, OpticsEvaluator
from llambo_mvp.llm_client import connect_llm, object_schema, write_json

ROOT = Path(__file__).resolve().parents[2]
BANDS = 100
FLOOR = .50
TOLERANCE = 2e-4
POINT_SCHEMA = object_schema({"factors": {
    "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
    "minItems": BANDS, "maxItems": BANDS,
}})
PRIOR_SCHEMA = object_schema({
    "coordinates": {"type": "string", "enum": ["raw_mask", "energy_weighted_mask"]},
    "kernel": {"type": "string", "enum": ["matern52", "rbf"]},
    "mean_D_NL": {"type": "number", "minimum": 0, "maximum": 1},
    "std_D_NL": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
    "length_scale": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
})
PREPARE_SCHEMA = object_schema({"prior": PRIOR_SCHEMA, "rationale": {"type": "string"}})
NEXT_SCHEMA = object_schema({
    "action": {"type": "string", "enum": ["evaluate", "stop"]},
    "point": {**POINT_SCHEMA, "type": ["object", "null"]},
    "rationale": {"type": "string"},
})


@dataclass(frozen=True)
class MaskConfig:
    power_kW: float = 100.
    max_trials: int = 12  # Includes the seeded random mask, excludes the unity baseline.
    seed: int = 20261113
    model: str | None = None
    thinking_effort: str | None = None
    occupied_energy_fraction: float = .999
    samples_per_band: int = 4
    nt: int = 16384
    search_dz_m: float = .005
    check_dz_m: float = .001
    min_trials: int = 8
    plateau_patience: int = 6
    minimum_improvement: float = .001
    max_search_seconds: float = 1200.
    ai_timeout_seconds: float = 300.
    solver_timeout_seconds: float = 900.

    def __post_init__(self):
        if self.power_kW not in (100., 81.943):
            raise ValueError("Use 100 or 81.943 kW before masking.")
        if type(self.max_trials) is not int or not 1 <= self.max_trials <= 100:
            raise ValueError("max_trials must be an integer from 1 to 100.")
        if self.nt < 1024 or self.nt & (self.nt-1) or self.samples_per_band < 2:
            raise ValueError("Use a power-of-two Nt and at least two samples per band.")
        if not .99 <= self.occupied_energy_fraction < 1:
            raise ValueError("Declare an occupied fraction from 0.99 up to, but excluding, 1.")
        if self.search_dz_m != .005 or self.check_dz_m != .001:
            raise ValueError("This notebook uses 5 mm search and 1 mm checks.")
        if self.min_trials < 1 or self.plateau_patience < 1 or self.minimum_improvement <= 0:
            raise ValueError("Stopping parameters must be positive.")
        if min(self.ai_timeout_seconds, self.solver_timeout_seconds, self.max_search_seconds) <= 0:
            raise ValueError("Timeouts must be positive.")

    @property
    def halfwidth(self):
        # Spectral intensity of exp(-a*t^2) has angular-frequency sigma sqrt(a).
        sigma = math.sqrt(2*math.log(2)) / FIBER_SETUP["pulse_fwhm_ps"]
        return sigma * NormalDist().inv_cdf((1+self.occupied_energy_fraction)/2)

    @property
    def window_ps(self):
        return math.pi * BANDS * self.samples_per_band / self.halfwidth

    def grid(self, refined=False):
        # Refinement doubles the time window and halves dt, with the physical band unchanged.
        return {"Nt": self.nt*(4 if refined else 1),
                "time_window_ps": self.window_ps*(2 if refined else 1)}


def check_point(point):
    if not isinstance(point, dict) or set(point) != {"factors"}:
        raise ValueError("Supply exactly {'factors': [100 scaling factors]}.")
    values = point["factors"]
    if not isinstance(values, list) or len(values) != BANDS:
        raise ValueError("Exactly 100 factors are required.")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           or abs(100*v-round(100*v)) > 1e-8 for v in values):
        raise ValueError("Every factor must be in {0, 0.01, ..., 1.00}; no clipping or rescaling.")
    return {"factors": [int(round(100*v))/100 for v in values]}


def point_key(point):
    return tuple(int(round(v*100)) for v in check_point(point)["factors"])


def random_point(seed):
    return {"factors": (np.random.default_rng(seed).integers(0, 101, BANDS)/100).tolist()}


def sampled_mask(omega, point, halfwidth):
    factors = np.array(check_point(point)["factors"])
    omega = np.asarray(omega)
    bands = np.floor((omega+halfwidth)/(2*halfwidth/BANDS)+1e-10).astype(int)
    inside = (bands >= 0) & (bands < BANDS) & (omega >= -halfwidth-1e-10) & (omega < halfwidth-1e-10)
    mask = np.ones(len(omega)); mask[inside] = factors[bands[inside]]
    return mask, np.where(inside, bands+1, 0)


def input_description(config):
    grid = config.grid(); dt = grid["time_window_ps"]/grid["Nt"]
    t = (np.arange(grid["Nt"])-grid["Nt"]//2)*dt
    omega = np.fft.fftshift(np.fft.fftfreq(len(t), dt))*2*np.pi
    field = np.exp(-2*np.log(2)*(t/FIBER_SETUP["pulse_fwhm_ps"])**2)
    spectrum = np.fft.fftshift(np.fft.ifft(field))
    energy = abs(spectrum)**2
    _, bands = sampled_mask(omega, {"factors": [1.]*BANDS}, config.halfwidth)
    counts = np.bincount(bands, minlength=BANDS+1)[1:]
    if counts.min() == 0:
        raise ValueError("At least one spectral band is empty on this grid.")
    weights = np.bincount(bands, weights=energy, minlength=BANDS+1)/energy.sum()
    edges = np.linspace(-config.halfwidth, config.halfwidth, BANDS+1)
    return {"band_edges_detuning_rad_ps": edges.tolist(), "band_energy_fractions": weights[1:].tolist(),
            "outside_band_energy_fraction": float(weights[0]), "samples_per_band": counts.tolist(),
            "time_window_ps": grid["time_window_ps"], "Nt": grid["Nt"], "dt_ps": dt,
            "omega_sample_spacing_rad_ps": 2*np.pi/grid["time_window_ps"]}


class MaskPrior:
    """Initialization-only GP; later acquisition decisions are made directly by the LLM."""
    def __init__(self, spec, weights):
        import jsonschema
        jsonschema.validate(spec, PRIOR_SCHEMA)
        self.spec = dict(spec)
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (BANDS,) or not np.all(self.weights >= 0) or self.weights.sum() <= 0:
            raise ValueError("Invalid spectral energy weights.")

    def mean(self, points):
        return np.full(len(points), self.spec["mean_D_NL"])

    def covariance(self, points, other=None):
        def coordinates(items):
            x = np.array([check_point(p)["factors"] for p in items])
            weights = self.weights/self.weights.sum() if self.spec["coordinates"] == "energy_weighted_mask" else np.full(BANDS, 1/BANDS)
            return x*np.sqrt(weights)
        x, y = coordinates(points), coordinates(points if other is None else other)
        r = np.linalg.norm(x[:, None]-y[None, :], axis=2)/self.spec["length_scale"]
        kernel = np.exp(-.5*r*r) if self.spec["kernel"] == "rbf" else (1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)
        return self.spec["std_D_NL"]**2*kernel


class MaskEvaluator(OpticsEvaluator):
    def __init__(self, directory, config, *, resume=False):
        setup = dict(FIBER_SETUP, time_window_ps=config.window_ps, mask_halfwidth_rad_ps=config.halfwidth,
                     allowed_Nt=[config.nt, 4*config.nt], allowed_windows_ps=[config.window_ps, 2*config.window_ps])
        super().__init__(ROOT, setup, objective="D_NL")
        self.worker = Path(__file__).with_name("spectral_mask_worker.jl")
        self.config = config; self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=resume)
        self.attempts = 0; self.cache = {}; self.records = []
        self.max_evaluations = config.max_trials+5
        self.identity.update(launch="100_band_spectral_amplitude_mask", objective_modes=[1,2,3],
                             brightness_modes=[1,2,3], brightness_reference="unmasked_equal_amplitude_fine_baseline",
                             power_renormalization="none", mask_halfwidth_rad_ps=config.halfwidth,
                             worker_sha256=hashlib.sha256(self.worker.read_bytes()).hexdigest(),
                             shared_worker_sha256=hashlib.sha256(self.worker.with_name("dio_worker.jl").read_bytes()).hexdigest(),
                             adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if resume:
            saved = json.loads((self.directory/"evaluator.json").read_text())
            # The Python controller can gain recovery support; the physics must match.
            for key in self.identity.keys()-{"adapter_sha256"}:
                if saved.get(key) != self.identity[key]:
                    raise ValueError(f"Cannot resume after evaluator change: {key}")
            for path in sorted(self.directory.glob("evaluation_*/measurement.json")):
                row = json.loads(path.read_text())
                self.attempts = max(self.attempts, row["evaluation"])
                self.records.append(row)
                if row["status"] == "ok":
                    artifact = Path(row["artifacts"]["fields"])
                    if hashlib.sha256(artifact.read_bytes()).hexdigest() != row["artifacts"]["fields_sha256"]:
                        raise ValueError(f"Saved fields changed: {artifact}")
                    r = row["resolution"]
                    self.cache[(point_key(row["point"]), r["Nt"], r["time_window_ps"], r["dz_m"])] = row
        else:
            write_json(self.directory/"evaluator.json", self.identity)

    def measure(self, point, *, fine=False, refined=False):
        point = check_point(point); grid = self.config.grid(refined)
        resolution = dict(grid, dz_m=self.config.check_dz_m if fine else self.config.search_dz_m)
        key = (point_key(point), resolution["Nt"], resolution["time_window_ps"], resolution["dz_m"])
        if key in self.cache:
            return dict(self.cache[key], cached=True)
        if self.attempts >= self.max_evaluations:
            raise RuntimeError("Spectral-mask evaluation budget exhausted.")
        self.attempts += 1
        directory = self.directory/f"evaluation_{self.attempts:03d}"; directory.mkdir()
        row = {"evaluation": self.attempts, "point": point, "objective": "D_NL", "status": "started",
               "power_kW_before_mask": self.config.power_kW, "resolution": resolution, "cached": False}
        self.records.append(row); write_json(directory/"measurement.json", row)
        started = time.monotonic()
        try:
            value = super().__call__({**point, "power_kW": self.config.power_kW}, resolution, directory,
                                     self.config.solver_timeout_seconds)
            row.update(status="ok", D_NL=value["score"], valid=value["checks"]["valid"],
                       checks=value["checks"], metrics=value["metrics"], artifacts=value["artifacts"])
            self.cache[key] = row
            print(f"  Pair {self.attempts}/{self.max_evaluations}: D_NL={row['D_NL']:.8f}; "
                  f"input energy={row['metrics']['input_energy_retained']:.3f}; "
                  f"Nt={resolution['Nt']}, dz={1000*resolution['dz_m']:g} mm", flush=True)
            return row
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["wall_seconds"] = time.monotonic()-started
            write_json(directory/"measurement.json", row)

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def problem_card(config, description):
    return {
        "task": "Minimize all-three-mode D_NL by shaping the incoming spectrum with 100 amplitude factors.",
        "parameters": {"factors": "Exactly 100 independent factors in [0,1], each an integer multiple of 0.01, ordered low to high angular frequency."},
        "mask": "Multiply complex spectral field amplitude, not power, in each equal-width omega band. "
                "One shared mask for all three modes. No spectral phase mask and NO power/energy renormalization. "
                "Outside the declared occupied band, transmission is fixed at one; those 0.1% tails are not controls.",
        "initialization": "Trial 1 is an immutable seeded uniform random vector on {0,0.01,...,1}. AI chooses subsequent complete vectors.",
        "launch": {"before_mask_peak_kW": config.power_kW, "intensity_fwhm_fs": 200,
                   "modal_amplitudes": [1/np.sqrt(3)]*3, "modal_phases_rad": [0,0,0]},
        "fiber": {k: FIBER_SETUP[k] for k in ("fiber_length_m","carrier_m","core_radius_m","NA","num_modes","beta_order","cp_target_error","cp_seed")},
        "objective": "D_NL = 1 - |<A_linear(L),A_nonlinear(L)>|^2 / (||A_linear||^2 ||A_nonlinear||^2); "
                     "all three modes enter the coherent overlap and both norms, with the same masked launch in both branches. "
                     "The linear branch uses n2=0. No alignment or spectral filtering in the objective. Lower is better.",
        "brightness": "Hard floor: all-mode nonlinear output photons >= 0.50 * unmasked baseline nonlinear output photons "
                      "at the same original power, measured with the 1 mm step. A low score below this floor is infeasible.",
        "input_energy_formula": "Input energy retained = outside_band_energy_fraction + sum_i band_energy_fractions[i]*factors[i]^2. "
                                "This predicts only input energy, NOT measured output brightness or D_NL. "
                                "Uniform random amplitude factors often transmit about one third of input energy and can violate brightness.",
        "spectral_grid": description,
        "numerics": {"search_dz_m": config.search_dz_m, "check_dz_m": config.check_dz_m,
                     "main_grid": config.grid(), "refinement_grid": config.grid(True), "D_NL_tolerance": TOLERANCE},
        "method": "Instantiate a GP prior for initialization only. Later acquisition is LLM judgment on measured history; no posterior fit. "
                  "Smooth or grouped proposals are allowed, but all 100 factors remain independent and must be returned. "
                  "Do not claim to resolve 100-dimensional interactions or reach a global optimum from a small trial budget.",
    }


def observation(measurement, reference_photons, role):
    ratio = measurement["metrics"]["total_photons"]/reference_photons
    return {"evaluation": measurement["evaluation"], "role": role, "point": measurement["point"],
            "D_NL": measurement["D_NL"], "brightness_ratio": ratio,
            "valid": measurement["valid"], "feasible": bool(measurement["valid"] and ratio >= FLOOR),
            "resolution": measurement["resolution"], "diagnostics": measurement["metrics"]}


def choose_next(problem, history, remaining, ask, validation=None):
    feasible = [r for r in history if r["feasible"]]
    best = min(feasible, key=lambda r: r["D_NL"], default=None)
    reply = ask({
        "operation": "choose_next_mask" if remaining else "finish_search", "problem_card": problem,
        "observations": history, "latest_observation": history[-1], "best_feasible": best,
        "remaining_trials": remaining, "final_validation": validation,
        "instruction": "Return one new full 100-factor point or stop. All factors must be on the 0.01 grid. "
                       "Use 1-3 sentences of rationale tied to measured data. Do not repeat a tested mask, alter physics, "
                       "invent measurements, or lower the brightness floor. Explore spectral shape as well as throughput. "
                       "If remaining_trials=0, return action=stop and point=null. Small-budget results do not establish convergence."
    }, NEXT_SCHEMA)
    if reply.get("action") not in ("evaluate", "stop") or not isinstance(reply.get("rationale"), str):
        raise ValueError("Malformed acquisition reply.")
    if reply["action"] == "stop":
        if reply.get("point") is not None: raise ValueError("Stop must use point=null.")
    else:
        if remaining <= 0: raise ValueError("No trials remain.")
        reply["point"] = check_point(reply["point"])
        if any(point_key(reply["point"]) == point_key(r["point"]) for r in history):
            raise ValueError("Repeated mask; no extra solve was run.")
    print("AI decision:", reply["action"], reply["rationale"], flush=True)
    return reply


@dataclass
class MaskRun:
    directory: Path
    result: dict
    prior: MaskPrior


def run_search(config, ask, simulator, directory, *, resume_state=None):
    directory = Path(directory); start = time.monotonic()
    description = input_description(config); problem = problem_card(config, description)
    initial = random_point(config.seed); unity = {"factors": [1.]*BANDS}
    state = resume_state or {"config": asdict(config), "problem": problem, "history": []}
    if state["config"] != asdict(config) or state["problem"] != problem:
        raise ValueError("Resume must retain the original configuration and physics.")
    history = state["history"]
    previous_wall = state.get("total_wall_seconds", 0.)
    previous_search = state.get("search_seconds", 0.)
    # Compatibility with the original pilot checkpoint, saved before recovery support.
    if resume_state and "search_seconds" not in state:
        calibration_ids = {r["evaluation"] for r in state.get("calibration", [])}
        setup_seconds = sum(r.get("wall_seconds", 0.) for r in getattr(simulator, "records", [])
                            if r["evaluation"] in calibration_ids)
        setup_seconds += sum(c.get("seconds", 0.) for c in getattr(ask, "calls", [])
                             if c.get("operation") == "prepare_prior")
        previous_search = max(0., previous_wall-setup_seconds)
    state.update(status="running")
    state.pop("error", None)
    search_started = None
    write_json(directory/"state.json", state)
    try:
        if resume_state:
            plan = json.loads((directory/"initial_plan.json").read_text())
            if plan["initial_point"] != initial:
                raise ValueError("Saved random mask differs from the frozen seed.")
        else:
            plan = ask({"operation": "prepare_prior", "problem_card": problem, "random_initial_point": initial,
                        "instruction": "Choose a GP prior and briefly explain it in 1-3 sentences. The supplied seeded random "
                                       "mask is trial 1 and must not be replaced. The GP is initialization-only."}, PREPARE_SCHEMA)
            write_json(directory/"initial_plan.json", dict(plan, initial_point=initial, seed=config.seed))
        prior = MaskPrior(plan["prior"], description["band_energy_fractions"])
        print("GP prior:", json.dumps(prior.spec, indent=2)); print("AI explanation:", plan["rationale"])
        baseline_fine = simulator.measure(unity, fine=True)
        reference = baseline_fine["metrics"]["total_photons"]
        if not baseline_fine["valid"] or reference <= 0: raise ValueError("Invalid unmasked brightness reference.")
        base = observation(simulator.measure(unity), reference, "unmasked baseline")
        fine_base = observation(baseline_fine, reference, "unmasked baseline fine")
        if not history:
            history.append(base)
        elif history[0] != base:
            raise ValueError("Saved baseline differs from its cached measurement.")
        state["calibration"] = [base, fine_base]
        significant_best = base["D_NL"]; last_improvement = 0; point = initial
        for trial, row in enumerate(history[1:], 1):
            if row["feasible"] and row["D_NL"] < significant_best-config.minimum_improvement:
                significant_best = row["D_NL"]; last_improvement = trial
        stop_reason = state.get("stop_reason", "trial budget reached")
        search_started = time.monotonic()
        while not state.get("search_complete", False):
            completed = len(history)-1
            if completed >= config.max_trials:
                break
            if completed >= config.min_trials and completed-last_improvement >= config.plateau_patience:
                stop_reason = f"no >= {config.minimum_improvement:g} search-score improvement for {config.plateau_patience} trials"; break
            if previous_search+time.monotonic()-search_started >= config.max_search_seconds:
                stop_reason = "search wall-time budget reached (validation retained)"; break
            if completed:
                # Reserve one call for the final interpretation; failures count toward the cap.
                if len(getattr(ask, "calls", [])) >= config.max_trials:
                    stop_reason = "AI attempt budget reached (final interpretation retained)"; break
                try:
                    decision = state.get("pending_decision") or choose_next(problem, history, config.max_trials-completed, ask)
                except Exception as error:
                    state["acquisition_error"] = f"{type(error).__name__}: {error}"
                    stop_reason = "AI acquisition failed; validate the best measured mask"; break
                if decision["action"] == "stop":
                    stop_reason = "AI requested stop: " + decision["rationale"]; break
                point = check_point(decision["point"])
                state["pending_decision"] = decision
                write_json(directory/"state.json", state)
            trial = completed+1
            print(f"\nTrial {trial}/{config.max_trials}", flush=True)
            row = observation(simulator.measure(point), reference, "seeded random mask" if trial == 1 else "AI mask")
            history.append(row)
            print(f"  D_NL={row['D_NL']:.8f}; brightness={row['brightness_ratio']:.3f}; feasible={row['feasible']}", flush=True)
            if row["feasible"] and row["D_NL"] < significant_best-config.minimum_improvement:
                significant_best = row["D_NL"]; last_improvement = trial
            state.pop("pending_decision", None)
            state["search_seconds"] = previous_search+time.monotonic()-search_started
            write_json(directory/"state.json", state)
        search_seconds = previous_search+time.monotonic()-search_started
        best = min((r for r in history if r["feasible"]), key=lambda r: r["D_NL"])
        state.update(search_complete=True, search_seconds=search_seconds, stop_reason=stop_reason, best_search=best)
        write_json(directory/"state.json", state)
        selected = observation(simulator.measure(best["point"], fine=True), reference, "selected fine")
        # The same physical 100-band mask is re-evaluated with a larger time window and finer dt.
        reference_refined = simulator.measure(unity, fine=True, refined=True)
        refined_photons = reference_refined["metrics"]["total_photons"]
        refined_base = observation(reference_refined, refined_photons, "unmasked baseline refined")
        refined_best = observation(simulator.measure(best["point"], fine=True, refined=True), refined_photons, "selected refined")
        checks = {
            "baseline_step_agreement": abs(base["D_NL"]-fine_base["D_NL"]) <= TOLERANCE,
            "selected_step_agreement": abs(best["D_NL"]-selected["D_NL"]) <= TOLERANCE,
            "baseline_grid_agreement": abs(fine_base["D_NL"]-refined_base["D_NL"]) <= TOLERANCE,
            "selected_grid_agreement": abs(selected["D_NL"]-refined_best["D_NL"]) <= TOLERANCE,
            "selected_fine_and_refined_brightness": selected["feasible"] and refined_best["feasible"],
            "finite_defined_objective": all(r["valid"] for r in (fine_base, selected, refined_base, refined_best)),
        }
        validation = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
                      "baseline_fine": fine_base, "selected_fine": selected,
                      "baseline_refined": refined_base, "selected_refined": refined_best,
                      "verified_improvement": bool(all(checks.values()) and
                          selected["D_NL"] < fine_base["D_NL"]-TOLERANCE and
                          refined_best["D_NL"] < refined_base["D_NL"]-TOLERANCE)}
        state.update(status="measured", best_search=best, validation=validation, stop_reason=stop_reason,
                     search_seconds=search_seconds)
        write_json(directory/"state.json", state)
        try:
            if len(getattr(ask, "calls", [])) >= config.max_trials+1:
                raise RuntimeError("No AI attempts remain for final interpretation.")
            state["final_decision"] = dict(choose_next(problem, history, 0, ask, validation), source="LLM")
        except Exception as error:
            state["final_interpretation_error"] = f"{type(error).__name__}: {error}"
            state["final_decision"] = {"action": "stop", "point": None, "source": "controller",
                                       "rationale": "Search ended; measured validation retained without an AI interpretation."}
        state["status"] = "completed"
        return state, prior
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if search_started is not None and not state.get("search_complete", False):
            state["search_seconds"] = previous_search+time.monotonic()-search_started
        state.update(evaluation_attempts=simulator.attempts, ai_attempts=len(getattr(ask,"calls",[])),
                     total_wall_seconds=previous_wall+time.monotonic()-start)
        write_json(directory/"state.json", state)


def run_experiment(*, power_kW=100., max_trials=12, seed=20261113, model=None, thinking_effort=None,
                   resume_directory=None):
    if resume_directory is not None:
        return resume_experiment(resume_directory, expected={"power_kW": power_kW, "max_trials": max_trials,
                                 "seed": seed, "model": model, "thinking_effort": thinking_effort})
    config = MaskConfig(power_kW=power_kW, max_trials=max_trials, seed=seed, model=model, thinking_effort=thinking_effort,
                        min_trials=10 if max_trials > 12 else 8, plateau_patience=10 if max_trials > 12 else 6,
                        max_search_seconds=max(1200., 120.*max_trials) if max_trials > 12 else 1200.)
    directory = ROOT/"output"/datetime.now(timezone.utc).strftime("%Y-%m-%d_llm_bo_spectral_mask_%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/"run.json", {"config": asdict(config), "grid": config.grid(), "refinement_grid": config.grid(True),
                                     "max_evaluation_pairs": max_trials+5, "max_ai_calls": max_trials+1,
                                     "gradient_descent_executed": False, "application_usefulness": "not_specified"})
    ask = connect_llm(directory/"ai", max_calls=max_trials+1, model=model, reasoning_effort=thinking_effort,
                      timeout_seconds=config.ai_timeout_seconds)
    print("Saving this run to:", directory, flush=True)
    print(f"100 bands across +/-{config.halfwidth:.6f} rad/ps; T={config.window_ps:.4f} ps, Nt={config.nt}.", flush=True)
    with MaskEvaluator(directory/"gnlse", config) as simulator:
        result, prior = run_search(config, ask, simulator, directory)
    print("Stop:", result["stop_reason"])
    print("Numerical checks:", result["validation"]["status"])
    print(f"Used {result['evaluation_attempts']} matched linear/nonlinear pairs and {result['ai_attempts']} AI calls.")
    return MaskRun(directory, result, prior)


def resume_experiment(directory, *, expected=None):
    """Continue a saved checkpoint within its original budgets, or load a completed run."""
    directory = Path(directory).resolve()
    state = json.loads((directory/"state.json").read_text())
    config = MaskConfig(**state["config"])
    if expected and any(state["config"].get(k) != v for k, v in expected.items()):
        raise ValueError("Notebook settings differ from the saved run; use its original settings.")
    print("Saving this run to:", directory, flush=True)
    if state["status"] == "completed":
        plan = json.loads((directory/"initial_plan.json").read_text())
        print("Loaded completed measurements; no new AI calls or propagations.")
        return MaskRun(directory, state, MaskPrior(plan["prior"], input_description(config)["band_energy_fractions"]))
    recovery = directory/"recovery"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    recovery.mkdir(parents=True, exist_ok=False)
    write_json(recovery/"state_before.json", state)
    (recovery/"spectral_mask.py").write_bytes(Path(__file__).read_bytes())
    calls = [json.loads(p.read_text()) for p in sorted((directory/"ai").glob("call_*/metadata.json"))]
    if [c["call"] for c in calls] != list(range(1, len(calls)+1)):
        raise ValueError("AI attempt history is incomplete; refusing to reset its budget.")
    ask = connect_llm(directory/"ai", max_calls=config.max_trials+1, model=config.model,
                      reasoning_effort=config.thinking_effort, timeout_seconds=config.ai_timeout_seconds)
    ask.calls.extend(calls)
    print(f"Resuming {len(state['history'])-1} measured trials; {len(calls)} AI attempts already used.", flush=True)
    with MaskEvaluator(directory/"gnlse", config, resume=True) as simulator:
        result, prior = run_search(config, ask, simulator, directory, resume_state=state)
    print("Stop:", result["stop_reason"])
    print("Numerical checks:", result["validation"]["status"])
    print(f"Used {result['evaluation_attempts']} matched linear/nonlinear pairs and {result['ai_attempts']} AI attempts.")
    return MaskRun(directory, result, prior)


def show_results(run):
    """Plot actual mask factors, throughput, pulses and refinement from saved fields."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    result, directory = run.result, run.directory
    history = result["history"]; validation = result["validation"]
    table = pd.DataFrame([{
        "observation": i, "role": r["role"], "D_NL": r["D_NL"], "brightness": r["brightness_ratio"],
        "input_energy_retained": r["diagnostics"]["input_energy_retained"],
        "input_peak_kW": r["diagnostics"]["input_peak_kW"],
        "input_time_edge_fraction": r["diagnostics"]["input_time_edge_fraction"], "feasible": r["feasible"],
    } for i, r in enumerate(history)])
    display(table); table.to_csv(directory/"history.csv", index=False)
    check_table = pd.DataFrame.from_dict(validation["checks"], orient="index", columns=["passed"])
    display(check_table)
    fine_table = pd.DataFrame([{
        "case": name, "D_NL": validation[name]["D_NL"], "brightness": validation[name]["brightness_ratio"],
        **validation[name]["resolution"],
    } for name in ("baseline_fine", "selected_fine", "baseline_refined", "selected_refined")])
    display(fine_table); fine_table.to_csv(directory/"grid_checks.csv", index=False)

    initial = history[1]["point"]["factors"]; best = result["best_search"]["point"]["factors"]
    selected = validation["selected_fine"]
    field_path = directory/"gnlse"/f"evaluation_{selected['evaluation']:03d}"/"fields.npz"
    with np.load(field_path) as data:
        t = data["time_ps"].copy(); omega = data["omega_absolute_rad_ps"].copy()
        detuning = data["omega_detuning_rad_ps"].copy()
        carrier = float(np.mean(omega-detuning))
        edges = data["band_edges_detuning_rad_ps"].copy()+carrier
        fields = {key: data[key].copy() for key in ("original_input", "input", "linear", "nonlinear")}
    mask_table = pd.DataFrame({"band": np.arange(1, BANDS+1), "omega_low_rad_ps": edges[:-1],
                               "omega_high_rad_ps": edges[1:], "random_factor": initial, "selected_factor": best})
    mask_table.to_csv(directory/"mask_factors.csv", index=False)
    np.save(directory/"selected_factors.npy", np.asarray(best))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(table.observation, table.D_NL, "o-", label="Measured, 5 mm")
    axes[0, 0].plot(table.observation, table.D_NL.where(table.feasible, np.inf).cummin(), "--", label="Best feasible")
    axes[0, 0].set(xlabel="Observation (0 = unmasked)", ylabel="All-mode D_NL"); axes[0, 0].legend()
    axes[0, 1].plot(table.observation, table.brightness, "o-", label="Output photon ratio")
    axes[0, 1].plot(table.observation, table.input_energy_retained, "s--", label="Input energy retained")
    axes[0, 1].axhline(FLOOR, color="black", linestyle=":", label="Brightness minimum")
    axes[0, 1].set(xlabel="Observation", ylabel="Fraction of unmasked reference"); axes[0, 1].legend()
    axes[1, 0].stairs(initial, edges, color="#aaa", label="Seeded random mask")
    axes[1, 0].stairs(best, edges, color="#366b9e", linewidth=2, label="Selected mask")
    axes[1, 0].set(xlabel="Angular frequency ω (rad/ps)", ylabel="Field amplitude factor", ylim=(-.03, 1.05))
    axes[1, 0].legend()
    for label, key in (("Unmasked", "original_input"), ("Selected masked input", "input")):
        spec = (abs(np.fft.fftshift(np.fft.ifft(fields[key], axis=0), axes=0))**2).sum(axis=(1,2))
        axes[1, 1].plot(omega, spec, label=label)
    axes[1, 1].set(xlabel="Angular frequency ω (rad/ps)", ylabel="Input power per frequency bin (W)",
                   xlim=(edges[0]-2, edges[-1]+2)); axes[1, 1].legend()
    fig.suptitle(f"100 spectral factors | no renormalization | numerical checks: {validation['status']}")
    fig.savefig(directory/"mask_results.png", dpi=150); fig.savefig(directory/"mask_results.pdf"); plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    branches = (("Masked input", "input"), ("Matched linear output", "linear"), ("Nonlinear output", "nonlinear"))
    for label, key in branches:
        power = (abs(fields[key])**2).sum(axis=(1,2))/1000
        spec = (abs(np.fft.fftshift(np.fft.ifft(fields[key], axis=0), axes=0))**2).sum(axis=(1,2))
        axes[0].plot(t, power, label=label)
        axes[1].semilogy(t, np.maximum(power, 1e-14), label=label)
        axes[2].plot(omega, spec, label=label)
    axes[0].set(xlabel="Time (ps)", ylabel="All-mode power (kW)", xlim=(-2,2), title="Pulse centre")
    axes[1].set(xlabel="Time (ps)", ylabel="All-mode power (kW)", title="Full window and mask-induced tails")
    axes[2].set(xlabel="Angular frequency ω (rad/ps)", ylabel="Power per frequency bin (W)",
                xlim=(edges[0]-10,edges[-1]+10), title="Output spectrum")
    for ax in axes: ax.legend(fontsize=8)
    fig.savefig(directory/"pulse_results.png", dpi=150); fig.savefig(directory/"pulse_results.pdf"); plt.show()
    print("Selected 100 factors:", directory/"mask_factors.csv")
    print("Validated improvement:", validation["verified_improvement"])
    if validation["status"] != "pass":
        print("PRELIMINARY: failed numerical checks:", ", ".join(k for k,v in validation["checks"].items() if not v))
    print("Attenuation and pulse broadening are allowed controls here; a lower D_NL alone is not an application-performance claim.")


def timing_report(run):
    """Finite-difference estimates only: never runs a derivative or a descent step."""
    records = [json.loads(p.read_text()) for p in sorted((run.directory/"gnlse").glob("evaluation_*/measurement.json"))]
    records = [r for r in records if r["status"] == "ok"]
    calls = [json.loads(p.read_text()) for p in sorted((run.directory/"ai").glob("call_*/metadata.json"))]
    config = MaskConfig(**run.result["config"])
    search = [r for r in records if r["resolution"]["dz_m"] == config.search_dz_m]
    fine = [r for r in records if r["resolution"]["dz_m"] == config.check_dz_m and r["resolution"]["Nt"] == config.nt]
    refined = [r for r in records if r["resolution"]["Nt"] == 4*config.nt]
    pair = float(np.median([r["metrics"]["propagation_seconds"] for r in search]))
    fine_pair = float(np.median([r["metrics"]["propagation_seconds"] for r in fine]))
    refined_pair = float(np.median([r["metrics"]["propagation_seconds"] for r in refined]))
    ai_call = float(np.median([c["seconds"] for c in calls]))
    overhead = sum(max(0, r["wall_seconds"]-r["metrics"]["propagation_seconds"]) for r in records)
    search_io = float(np.median([max(0, r["wall_seconds"]-r["metrics"]["propagation_seconds"]) for r in search]))
    def projected_overhead(search_pairs):
        # Keep observed startup/validation overhead, then account for extra serialized results.
        return overhead+max(0, search_pairs-len(search))*search_io
    validation_cost = 2*fine_pair+2*refined_pair
    actual_propagation = sum(r["metrics"]["propagation_seconds"] for r in records)
    actual_ai = sum(c["seconds"] for c in calls)
    rows = [{"method": "LLMBO pilot (measured)", "trials_or_steps": len(run.result["history"])-1,
             "evaluation_pairs": len(records), "AI_calls": len(calls),
             "propagation_minutes": actual_propagation/60, "AI_minutes": actual_ai/60,
             "total_minutes": run.result["total_wall_seconds"]/60, "basis": "measured"}]
    for trials in (30, 60):
        physics = (trials+1)*pair+validation_cost
        ai = (trials+1)*ai_call
        rows.append({"method": f"LLMBO cap {trials} (estimated)", "trials_or_steps": trials,
                     "evaluation_pairs": trials+5, "AI_calls": trials+1,
                     "propagation_minutes": physics/60, "AI_minutes": ai/60,
                     "total_minutes": (physics+ai+projected_overhead(trials+1))/60,
                     "basis": "projection; one random trial included"})
    for difference, probes, steps in (("central", 200, 1), ("central", 200, 10), ("central", 200, 20), ("forward", 100, 20)):
        search_pairs = 1+steps*(probes+1)
        physics = search_pairs*pair+validation_cost
        rows.append({"method": f"Finite-difference GD, {difference} (estimated)", "trials_or_steps": steps,
                     "evaluation_pairs": search_pairs+4, "AI_calls": 0,
                     "propagation_minutes": physics/60, "AI_minutes": 0.,
                     "total_minutes": (physics+projected_overhead(search_pairs))/60,
                     "basis": f"{probes} probes + 1 accepted trial per step; same final checks"})
    failed = [name for name, passed in run.result["validation"]["checks"].items() if not passed]
    return {
        "gradient_descent_executed": False, "table": rows,
        "measured_search_pair_seconds": pair,
        "measured_search_pair_range_seconds": [min(r["metrics"]["propagation_seconds"] for r in search), max(r["metrics"]["propagation_seconds"] for r in search)],
        "search_pair_samples": len(search), "measured_fine_pair_seconds": fine_pair,
        "measured_refined_pair_seconds": refined_pair, "median_AI_call_seconds": ai_call,
        "AI_attempt_status_counts": {status: sum(c["status"] == status for c in calls) for status in ("completed", "failed")},
        "AI_attempt_range_seconds": [min(c["seconds"] for c in calls), max(c["seconds"] for c in calls)],
        "failed_AI_seconds": sum(c["seconds"] for c in calls if c["status"] == "failed"),
        "median_search_transport_and_io_seconds": search_io,
        "setup_transport_and_io_seconds": overhead, "actual_propagation_seconds": actual_propagation,
        "actual_AI_seconds": actual_ai,
        "assumptions": [
            "One objective evaluation is one matched linear/nonlinear propagation pair.",
            "Central differences use 2*100=200 probes per interior gradient; forward/one-sided differences need about 100.",
            "Finite-difference step is 0.01, followed by projection to the [0,1] grid; these are finite-step slopes, not exact derivatives.",
            "One trial evaluation per descent step is optimistic; backtracking adds further pairs. The current score is reused.",
            "The budget includes an unmasked baseline and the same fine/refined final checks, with no cache savings assumed in projections.",
            "Timings include measured LLM overhead, setup/transport/I/O separately; projections assume the same hardware, single thread, settings and output policy.",
            "Evaluations are serial. Projected overhead retains observed startup/check costs and adds median search I/O time for each additional objective pair.",
            "Measured experiment time includes failed AI attempts and both execution segments, excluding the idle pause, kernel startup and report generation.",
            "AI projections use the median recorded attempt duration, including failed attempts in the sample; latency and failure rates can change.",
            "Only finite-difference descent is estimated. A separately implemented adjoint/autodiff method could have a very different cost.",
            "Different budgets do not establish equal solution quality or an optimizer speedup. No gradient descent was executed.",
        ],
        "recommendation": {"first_cap_trials": 30, "hard_cap_trials": 60, "minimum_trials_before_plateau_stop": 10,
                           "patience_trials": 10, "minimum_absolute_D_NL_improvement": .001,
                           "rule": "Start with a 30-trial cap (including the seeded random mask). Stop after at least 10 trials if the last 10 "
                                   "produce no feasible D_NL improvement of 0.001. Consider a separate extension up to 60 only if progress continues "
                                   "and the finer-grid checks pass. These are practical budgets, not convergence guarantees for 100 dimensions.",
                           "numerical_gate": "Resolve failed refinement checks before extending the search." if failed else "Numerical checks passed for the selected pilot mask.",
                           "failed_checks": failed},
    }


def show_timing_comparison(run):
    import pandas as pd
    from IPython.display import display
    report = timing_report(run)
    table = pd.DataFrame(report["table"])
    display(table)
    table.to_csv(run.directory/"timing_comparison.csv", index=False)
    write_json(run.directory/"timing_comparison.json", report)
    print(f"Measured warm objective pair: {report['measured_search_pair_seconds']:.2f} s "
          f"from {report['search_pair_samples']} search measurements. Median AI call: {report['median_AI_call_seconds']:.2f} s.")
    print(f"AI attempts: {report['AI_attempt_status_counts']}; failed requests used {report['failed_AI_seconds']/60:.2f} min. "
          "Reported experiment time excludes the pause between sessions.")
    print("100 parameters: central differences need about 200 objective pairs per gradient, before line search.")
    print(report["recommendation"]["rule"])
    print(report["recommendation"]["numerical_gate"])
    print("No gradient descent was run. Estimates include final checks and AI overhead; they do not establish equal optimization quality.")
    return table
