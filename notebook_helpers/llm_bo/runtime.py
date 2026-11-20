"""Small adapters for the D_IO/D_NL notebooks; notebook.py contains the search loop."""
from pathlib import Path
import hashlib
import math
import time

import numpy as np

from llambo_mvp.llm_client import connect_llm, object_schema, write_json
from ai_augmented_loop.evaluators import FIBER_SETUP, OpticsEvaluator


NAMES = ("c1", "c2", "c3")
PHASE_NAMES = ("phi2_rad", "phi3_rad")
POINT_SCHEMA = object_schema({k: {"type": "number", "minimum": 0, "maximum": 100}
                              for k in NAMES})
PHASE_POINT_SCHEMA = object_schema({
    **POINT_SCHEMA["properties"],
    **{k: {"type": "number", "minimum": -math.pi, "maximum": math.pi}
       for k in PHASE_NAMES},
})
PRIOR_SCHEMA = object_schema({
    "coordinates": {"type": "string", "enum": ["amplitude_fraction", "power_fraction"]},
    "kernel": {"type": "string", "enum": ["matern52", "rbf"]},
    "mean_D_IO": {"type": "number", "minimum": 0, "maximum": 1},
    "std_D_IO": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
    "length_scale": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
    "exchange_symmetric_LP11": {"type": "boolean"},
})
PREPARE_SCHEMA = object_schema({
    "prior": PRIOR_SCHEMA, "initial_point": POINT_SCHEMA,
    "rationale": {"type": "string"},
})
NEXT_SCHEMA = object_schema({
    "action": {"type": "string", "enum": ["evaluate", "stop"]},
    "point": {**POINT_SCHEMA, "type": ["object", "null"]},
    "rationale": {"type": "string"},
})
PHASE_PREPARE_SCHEMA = object_schema({
    "prior": PRIOR_SCHEMA, "initial_point": PHASE_POINT_SCHEMA,
    "rationale": {"type": "string"},
})
PHASE_NEXT_SCHEMA = object_schema({
    "action": {"type": "string", "enum": ["evaluate", "stop"]},
    "point": {**PHASE_POINT_SCHEMA, "type": ["object", "null"]},
    "rationale": {"type": "string"},
})

OBJECTIVES = {
    "D_IO": {"modes": [1], "label": "LP01", "photons": "LP01_photons"},
    "D_NL": {"modes": [1, 2, 3], "label": "All-mode", "photons": "total_photons"},
}


def planning_schemas(*, phases=False, objective="D_IO"):
    """Keep objective names consistent in the prior and structured model replies."""
    if objective not in OBJECTIVES:
        raise ValueError("Choose D_IO (LP01 input/output) or D_NL (all-mode linear/nonlinear).")
    prior = object_schema({k.replace("D_IO", objective): v for k, v in PRIOR_SCHEMA["properties"].items()})
    point = PHASE_POINT_SCHEMA if phases else POINT_SCHEMA
    prepare = object_schema({"prior": prior, "initial_point": point, "rationale": {"type": "string"}})
    next_point = object_schema({
        "action": {"type": "string", "enum": ["evaluate", "stop"]},
        "point": {**point, "type": ["object", "null"]}, "rationale": {"type": "string"},
    })
    return prior, prepare, next_point


def check_point(point, *, phases=False):
    """Validate the simplex and optional relative phases; canonicalize periodic aliases."""
    names = NAMES + PHASE_NAMES if phases else NAMES
    if not isinstance(point, dict) or set(point) != set(names):
        raise ValueError(f"Supply exactly {', '.join(names)}.")
    values = [point[k] for k in NAMES]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100
           for v in values):
        raise ValueError("Every coefficient must be a finite number in [0, 100].")
    if not math.isclose(sum(values), 100, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Amplitude coefficients must sum to 100.")
    checked = dict(zip(NAMES, map(float, values)))
    if phases:
        for name, amplitude in zip(PHASE_NAMES, NAMES[1:]):
            value = point[name]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not -math.pi <= value <= math.pi):
                raise ValueError(f"{name} must be finite and in [-pi, pi] radians.")
            # +pi and -pi denote the same launch; an absent mode has no phase.
            checked[name] = ((float(value) + math.pi) % (2*math.pi) - math.pi
                             if checked[amplitude] > 0 else 0.0)
    return checked


def same_point(a, b, *, phases=False):
    a, b = check_point(a, phases=phases), check_point(b, phases=phases)
    amplitudes_equal = all(abs(a[k] - b[k]) <= 1e-6 for k in NAMES)
    return amplitudes_equal and (not phases or all(
        abs((a[k] - b[k] + math.pi) % (2*math.pi) - math.pi) <= 1e-6
        for k in PHASE_NAMES))


class GPPrior:
    """An executable constant-mean GP prior, used only during initialization.

    Symmetry averages k(Tx, Ty) over both independent LP11 swaps. This is a
    positive-semidefinite invariant kernel; it is not an empirical symmetry test.
    No posterior fitting or acquisition optimizer is hidden here.
    """
    def __init__(self, spec, *, phases=False, objective="D_IO"):
        schema, _, _ = planning_schemas(phases=phases, objective=objective)
        self.mean_key, self.std_key = f"mean_{objective}", f"std_{objective}"
        if not isinstance(spec, dict) or set(spec) != set(schema["properties"]):
            raise ValueError("Incomplete GP prior specification.")
        for key in ("coordinates", "kernel"):
            if spec[key] not in PRIOR_SCHEMA["properties"][key]["enum"]:
                raise ValueError(f"Unknown prior {key}.")
        for key, limit in ((self.mean_key, 1), (self.std_key, 1), ("length_scale", 10)):
            v = spec[key]
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= limit:
                raise ValueError(f"Invalid prior {key}.")
        if spec[self.std_key] == 0 or spec["length_scale"] == 0:
            raise ValueError("Prior scale parameters must be positive.")
        if type(spec["exchange_symmetric_LP11"]) is not bool:
            raise ValueError("Prior symmetry must be boolean.")
        self.spec = dict(spec)
        self.phases = phases

    def mean(self, points):
        return np.full(len(points), self.spec[self.mean_key])

    def covariance(self, points, other=None):
        def coordinates(items):
            points = [check_point(p, phases=self.phases) for p in items]
            x = np.array([[p[k] for k in NAMES] for p in points], dtype=float).reshape(-1, 3) / 100
            if self.spec["coordinates"] == "power_fraction":
                x = x*x / (x*x).sum(axis=1, keepdims=True)
            if not self.phases:
                return x
            phi = np.array([[p[k] for k in PHASE_NAMES] for p in points], dtype=float).reshape(-1, 2)
            # Continuous across +/-pi; phase distance vanishes for an absent mode.
            return np.column_stack((x[:, 0], x[:, 1]*np.cos(phi[:, 0]),
                                    x[:, 1]*np.sin(phi[:, 0]), x[:, 2]*np.cos(phi[:, 1]),
                                    x[:, 2]*np.sin(phi[:, 1])))
        x, y = coordinates(points), coordinates(points if other is None else other)
        identity = tuple(range(5 if self.phases else 3))
        exchange = (0, 3, 4, 1, 2) if self.phases else (0, 2, 1)
        swaps = (identity, exchange) if self.spec["exchange_symmetric_LP11"] else (identity,)
        result = np.zeros((len(x), len(y)))
        for a in swaps:
            for b in swaps:
                r = np.linalg.norm(x[:, a][:, None, :] - y[:, b][None, :, :], axis=2) / self.spec["length_scale"]
                k = np.exp(-0.5*r*r) if self.spec["kernel"] == "rbf" else (1 + np.sqrt(5)*r + 5*r*r/3)*np.exp(-np.sqrt(5)*r)
                result += self.spec[self.std_key]**2 * k / len(swaps)**2
        return result


class GNLSE(OpticsEvaluator):
    """Persistent Julia transport for real or phase-controlled three-mode launches.

    A measurement runs one nonlinear propagation, plus a matched linear solve for D_NL.
    Baseline and fine checks are
    explicit calls in the search helper, included in max_evaluations. Saved repeats
    cost no solves; an interrupted/failed solve still consumes one attempt.
    """
    def __init__(self, workspace, directory, *, max_evaluations, timeout_seconds=300,
                 phases=False, objective="D_IO"):
        if objective not in OBJECTIVES:
            raise ValueError("Unknown objective.")
        super().__init__(workspace, FIBER_SETUP, objective=objective)
        self.phases = phases
        self.names = NAMES + PHASE_NAMES if phases else NAMES
        self.worker = Path(__file__).with_name("dio_worker.jl").resolve()
        self.identity.update(launch=("three_complex_amplitudes_fixed_total_power" if phases
                                     else "three_amplitudes_fixed_total_power"),
                             objective_modes=OBJECTIVES[objective]["modes"],
                             brightness_modes=OBJECTIVES[objective]["modes"],
                             brightness_reference="explicit_equal_amplitude_1mm",
                             worker_sha256=hashlib.sha256(self.worker.read_bytes()).hexdigest(),
                             adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if type(max_evaluations) is not int or max_evaluations < 1:
            raise ValueError("max_evaluations must be a positive integer.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite.")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.max_evaluations = max_evaluations
        self.timeout_seconds = timeout_seconds
        self.attempts = 0
        self.cache = {}
        write_json(self.directory / "evaluator.json", self.identity)

    def measure(self, point, power_kW, dz_m):
        point = check_point(point, phases=self.phases)
        if power_kW not in (100.0, 81.943) or dz_m not in (0.005, 0.001):
            raise ValueError("Use the source notebook's power cases and propagation steps.")
        key = (power_kW, dz_m, *(round(point[k], 9) for k in self.names))
        if key in self.cache:
            return dict(self.cache[key], cached=True)
        if self.attempts >= self.max_evaluations:
            raise RuntimeError("GNLSE evaluation budget exhausted.")
        self.attempts += 1
        directory = self.directory / f"evaluation_{self.attempts:03d}"
        directory.mkdir()
        record = {"evaluation": self.attempts, "point": point, "power_kW": power_kW,
                  "dz_m": dz_m, "Nt": 2048, "status": "started", "objective": self.objective}
        write_json(directory / "measurement.json", record)
        started = time.monotonic()
        try:
            result = super().__call__({**point, "power_kW": power_kW},
                                      {"Nt": 2048, "dz_m": dz_m}, directory,
                                      self.timeout_seconds)
            metrics = result["metrics"]
            record.update(status="ok",
                          objective_defined=result["checks"]["objective_defined"],
                          valid=result["checks"]["valid"], metrics=metrics,
                          artifacts=result["artifacts"], cached=False)
            record[self.objective] = result["score"]
            # Strict JSON rejects any nonfinite diagnostics as well as scores.
            write_json(directory / "measurement.json", record)
            self.cache[key] = record
            print(f"  GNLSE {self.attempts}/{self.max_evaluations}: "
                  f"{list(point.values())}, {self.objective}={record[self.objective]:.8f}, "
                  f"dz={1000*dz_m:g} mm", flush=True)
            return record
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["wall_seconds"] = round(time.monotonic() - started, 3)
            write_json(directory / "measurement.json", record)

    @property
    def linear_solves(self):
        return sum(row["metrics"].get("linear_solves", 0) for row in self.cache.values())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
