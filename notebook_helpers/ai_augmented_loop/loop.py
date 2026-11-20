"""A small, simulator-independent, resumable experiment loop (Python 3.11+).

The planner proposes data, never executable code. Evaluators own the science.
SQLite is the authoritative history; optimizer checkpoints follow every step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import math
import random
import sqlite3
import time
import uuid


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj):
    return hashlib.sha256(encode(obj).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass
class Control:
    name: str
    lower: float
    upper: float
    scale: float = 1.0
    fd_step: float = 0.005       # in normalized optimizer coordinates
    max_change: float = 0.1     # in displayed physical units
    expansion: float = 0.25    # in displayed physical units; zero disables expansion
    domain_lower: float | None = None
    periodic: bool = False
    transform: str = "linear"  # sqrt is useful for modal powers near zero
    unit: str = ""

    def __post_init__(self):
        if self.transform not in ("linear", "sqrt"):
            raise ValueError("Unknown control transform")
        if not all(finite(v) for v in (self.lower, self.upper, self.scale, self.fd_step,
                                      self.max_change, self.expansion)):
            raise ValueError("Control settings must be finite")
        if not self.lower < self.upper or min(self.scale, self.fd_step, self.max_change) <= 0:
            raise ValueError("Invalid control bounds/scale/step")
        if self.expansion < 0 or (self.domain_lower is not None and self.lower < self.domain_lower):
            raise ValueError("Invalid expansion or physical lower bound")
        if self.transform == "sqrt" and (self.lower < 0 or self.domain_lower is None or self.domain_lower < 0):
            raise ValueError("sqrt controls need a nonnegative physical domain")

    def q(self, value):
        return math.sqrt(value / self.scale) if self.transform == "sqrt" else value / self.scale

    def physical(self, value):
        return self.scale * max(0., value)**2 if self.transform == "sqrt" else self.scale * value

    def wrap(self, value):
        return self.lower + (value - self.lower) % (self.upper - self.lower) if self.periodic else value

    def difference(self, a, b):
        d = a - b
        if self.periodic:
            period = self.upper - self.lower
            d = (d + period / 2) % period - period / 2
        return d


@dataclass
class Spec:
    name: str
    objective: str
    controls: list[Control]
    baseline: dict[str, float]
    evaluator_identity: dict[str, Any]
    search: dict[str, Any]
    refinements: dict[str, dict[str, Any]]
    brightness_fraction: float | None  # None explicitly means no brightness preference
    brightness_policy: str = "soft"
    brightness_metric: str = "brightness_ratio"
    objective_tolerance: float = 2e-4
    gradient_atol: float = 1e-5
    gradient_rtol: float = 0.01
    gradient_tolerance: float = 1e-4
    learning_rate: float = 0.1
    armijo: float = 1e-4
    backtracks: int = 12
    robustness: dict[str, float] = field(default_factory=dict)
    knowledge: str = ""

    def __post_init__(self):
        if len({c.name for c in self.controls}) != len(self.controls) or not self.controls:
            raise ValueError("Controls must have distinct names")
        if set(self.baseline) != {c.name for c in self.controls}:
            raise ValueError("Baseline must specify all controls")
        if self.brightness_policy not in ("hard", "soft"):
            raise ValueError("Brightness policy must be hard or soft")
        if self.brightness_fraction is not None and (not finite(self.brightness_fraction) or self.brightness_fraction < 0):
            raise ValueError("Choose a nonnegative brightness fraction, or explicitly None")
        if not self.refinements:
            raise ValueError("Supply at least one numerical refinement profile")
        if any(k not in self.baseline or not finite(v) or v < 0 for k, v in self.robustness.items()):
            raise ValueError("Robustness tolerances must match controls and be nonnegative")


@dataclass
class Decision:
    action: str
    parameters: dict[str, float] | None = None
    run_id: int | None = None
    evaluations: int = 1
    iterations: int = 1
    control: str | None = None
    direction: str | None = None
    rationale: str = ""
    evidence_ids: list[int] = field(default_factory=list)
    knowledge_basis: str = ""
    expected_outcome: str = ""
    threshold_name: str | None = None
    threshold_value: float | None = None


class AllowanceReached(RuntimeError):
    pass


class InvalidEvaluation(RuntimeError):
    pass


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS campaigns(
 id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, spec TEXT NOT NULL, state TEXT NOT NULL,
 created REAL NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS allowances(
 id INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id),
 evaluations INTEGER NOT NULL, seconds REAL NOT NULL, planner_calls INTEGER NOT NULL,
 reason TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS actions(
 id INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), decision TEXT NOT NULL,
 status TEXT NOT NULL, result TEXT, seconds REAL NOT NULL DEFAULT 0, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations(
 id INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), action INTEGER,
 cache_key TEXT NOT NULL, parameters TEXT NOT NULL, resolution TEXT NOT NULL,
 status TEXT NOT NULL, score REAL, metrics TEXT, checks TEXT, artifacts TEXT,
 error TEXT, seconds REAL NOT NULL DEFAULT 0, created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS evaluations_cache ON evaluations(cache_key,status);
CREATE TABLE IF NOT EXISTS evaluation_uses(
 id INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), action INTEGER,
 evaluation INTEGER NOT NULL REFERENCES evaluations(id), purpose TEXT NOT NULL,
 cache_hit INTEGER NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS checks(
 id INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), action INTEGER,
 parameters TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, details TEXT NOT NULL,
 created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS optimizer_runs(
 id INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), state TEXT NOT NULL,
 status TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS accepted_steps(
 id INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), run INTEGER NOT NULL,
 iteration INTEGER NOT NULL, evaluation INTEGER NOT NULL, parameters TEXT NOT NULL,
 score REAL NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS proposals(
 id INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), status TEXT NOT NULL,
 context TEXT NOT NULL, decision TEXT, metadata TEXT, error TEXT,
 seconds REAL NOT NULL DEFAULT 0, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS criteria_revisions(
 id INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), action INTEGER,
 version INTEGER NOT NULL, name TEXT NOT NULL, old_value TEXT NOT NULL, new_value TEXT NOT NULL,
 reason TEXT NOT NULL, created REAL NOT NULL);
"""


class ExperimentLoop:
    def __init__(self, database: Path, spec: Spec, evaluator: Callable, *, campaign_id=None,
                 artifact_root: Path | None = None):
        self.path = Path(database).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = Path(artifact_root or self.path.parent / (self.path.stem+"_artifacts")).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.spec, self.evaluator = spec, evaluator
        self.id = campaign_id or uuid.uuid4().hex[:12]
        self.action_id, self.action_cap, self.action_used = None, None, 0
        self._clock_start = None
        self.signature = digest({"spec": asdict(spec), "engine": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        row = self.db.execute("SELECT * FROM campaigns WHERE id=?", (self.id,)).fetchone()
        if row:
            if row["fingerprint"] != self.signature:
                raise ValueError("Campaign settings or engine changed. Start a new campaign; existing history is preserved.")
            self.state = json.loads(row["state"])
        else:
            self.state = {"bounds": {c.name: [c.lower, c.upper] for c in spec.controls},
                          "status": "ready", "range_changes": [], "thresholds": {}, "criteria_version": 0}
            self.db.execute("INSERT INTO campaigns VALUES(?,?,?,?,?,?)",
                            (self.id, self.signature, encode(asdict(spec)), encode(self.state), time.time(), time.time()))
            self.db.commit()

    def close(self):
        self.db.close()

    def _save(self, status=None):
        if status:
            self.state["status"] = status
        self.db.execute("UPDATE campaigns SET state=?,updated=? WHERE id=?",
                        (encode(self.state), time.time(), self.id))
        self.db.commit()

    def grant(self, evaluations: int, seconds: float, planner_calls: int, reason="user allocation"):
        """Explicit user allocation. The planner cannot call this method."""
        if (type(evaluations) is not int or type(planner_calls) is not int or
                min(evaluations, seconds, planner_calls) < 0 or not finite(seconds)):
            raise ValueError("Allowances must be finite and nonnegative")
        self.db.execute("INSERT INTO allowances(campaign,evaluations,seconds,planner_calls,reason,created) VALUES(?,?,?,?,?,?)",
                        (self.id, evaluations, seconds, planner_calls, reason, time.time()))
        self.db.commit()
        self._save("ready")

    def budget(self):
        row = self.db.execute("SELECT COALESCE(SUM(evaluations),0),COALESCE(SUM(seconds),0),COALESCE(SUM(planner_calls),0) FROM allowances WHERE campaign=?", (self.id,)).fetchone()
        used = self.db.execute("SELECT COUNT(*) FROM evaluations WHERE campaign=?", (self.id,)).fetchone()[0]
        calls = self.db.execute("SELECT COUNT(*) FROM proposals WHERE campaign=?", (self.id,)).fetchone()[0]
        elapsed = sum(self.db.execute(f"SELECT COALESCE(SUM(seconds),0) FROM {t} WHERE campaign=?", (self.id,)).fetchone()[0]
                      for t in ("actions", "proposals"))
        elapsed += 0 if self._clock_start is None else time.monotonic() - self._clock_start
        return {"evaluations_used": used, "evaluations_left": max(0, row[0]-used),
                "seconds_used": elapsed, "seconds_left": max(0., row[1]-elapsed),
                "planner_calls_used": calls, "planner_calls_left": max(0, row[2]-calls)}

    def _room(self):
        b = self.budget()
        if b["evaluations_left"] <= 0 or b["seconds_left"] <= 0:
            raise AllowanceReached("campaign allowance reached")
        if self.action_cap is not None and self.action_used >= self.action_cap:
            raise AllowanceReached("action evaluation allocation reached")

    def canonical(self, parameters, *, outside=False):
        if not isinstance(parameters, dict) or set(parameters) != set(self.spec.baseline):
            raise ValueError("Specify exactly the configured controls")
        result = {}
        for c in self.spec.controls:
            v = parameters[c.name]
            if not finite(v):
                raise ValueError(f"{c.name} must be finite")
            v = float(c.wrap(v))
            if c.domain_lower is not None and v < c.domain_lower:
                raise ValueError(f"{c.name} is outside its physical domain")
            lo, hi = self.state["bounds"][c.name]
            if not outside and not c.periodic and not lo-1e-12 <= v <= hi+1e-12:
                raise ValueError(f"{c.name} is outside the current search range; use expand_range")
            result[c.name] = 0.0 if v == 0 else v
        return result

    def _cache_key(self, p, resolution):
        # Acceptance preferences/range changes do not alter the physical evaluation.
        return digest({"evaluator": self.spec.evaluator_identity, "objective": self.spec.objective,
                       "parameters": p, "resolution": resolution})

    def evaluate(self, parameters, resolution=None, purpose="candidate", *, outside=False):
        p = self.canonical(parameters, outside=outside)
        if self.budget()["seconds_left"] <= 0:
            raise AllowanceReached("campaign time allowance reached")
        resolution = dict(resolution or self.spec.search)
        key = self._cache_key(p, resolution)
        cached = self.db.execute("SELECT * FROM evaluations WHERE cache_key=? AND status='complete' ORDER BY id DESC LIMIT 1", (key,)).fetchone()
        if cached and self._artifacts_present(json.loads(cached["artifacts"] or "{}")):
            row = self._evaluation(cached)
            self._use(row["id"], purpose, True)
            return row
        self._room()
        cur = self.db.execute("INSERT INTO evaluations(campaign,action,cache_key,parameters,resolution,status,created) VALUES(?,?,?,?,?,'running',?)",
                              (self.id, self.action_id, key, encode(p), encode(resolution), time.time()))
        eid = cur.lastrowid
        self.db.commit()  # reserve the budget before calling the simulator
        self.action_used += 1
        directory = self.artifact_root / f"evaluation_{eid:06d}"
        directory.mkdir(exist_ok=True)
        started = time.monotonic()
        try:
            result = self.evaluator(p, resolution, directory, self.budget()["seconds_left"])
            score, metrics = result["score"], result.get("metrics", {})
            checks = dict(result.get("checks", {}))
            checks["finite_score"] = finite(score)
            if not finite(score) or not self._all_numbers_finite(metrics):
                raise InvalidEvaluation("Nonfinite objective or metrics")
            if checks.get("valid", True) is not True:
                raise InvalidEvaluation("Evaluator reported an invalid physical calculation")
            self.db.execute("UPDATE evaluations SET status='complete',score=?,metrics=?,checks=?,artifacts=?,seconds=? WHERE id=?",
                            (score, encode(metrics), encode(checks), encode(result.get("artifacts", {})), time.monotonic()-started, eid))
            self.db.commit()
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "invalid"
            self.db.execute("UPDATE evaluations SET status=?,error=?,seconds=? WHERE id=?",
                            (status, f"{type(exc).__name__}: {exc}", time.monotonic()-started, eid))
            self.db.commit()
            self._use(eid, purpose, False)
            if status == "interrupted":
                raise
            raise InvalidEvaluation(f"Evaluation {eid}: {exc}") from exc
        self._use(eid, purpose, False)
        return self._evaluation(self.db.execute("SELECT * FROM evaluations WHERE id=?", (eid,)).fetchone())

    @staticmethod
    def _artifacts_present(artifacts):
        path = artifacts.get("fields")
        if not path:
            return True
        path = Path(path)
        if not path.is_file():
            return False
        expected = artifacts.get("fields_sha256")
        return not expected or hashlib.sha256(path.read_bytes()).hexdigest() == expected

    @staticmethod
    def _all_numbers_finite(obj):
        if isinstance(obj, dict):
            return all(ExperimentLoop._all_numbers_finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(ExperimentLoop._all_numbers_finite(v) for v in obj)
        return not isinstance(obj, float) or math.isfinite(obj)

    def _use(self, eid, purpose, cached):
        self.db.execute("INSERT INTO evaluation_uses(campaign,action,evaluation,purpose,cache_hit,created) VALUES(?,?,?,?,?,?)",
                        (self.id, self.action_id, eid, purpose, int(cached), time.time()))
        self.db.commit()

    @staticmethod
    def _evaluation(row):
        result = dict(row)
        for k in ("parameters", "resolution", "metrics", "checks", "artifacts"):
            result[k] = json.loads(result[k]) if result.get(k) else {}
        return result

    def usable(self, row):
        ratio = row["metrics"].get(self.spec.brightness_metric)
        if self.threshold("brightness_fraction") is None:
            return True
        if not finite(ratio):
            return False
        return self.spec.brightness_policy == "soft" or ratio >= self.threshold("brightness_fraction")

    ADAPTIVE_THRESHOLDS = ("brightness_fraction", "objective_tolerance", "gradient_atol", "gradient_rtol", "gradient_tolerance")

    def threshold(self, name):
        return self.state.get("thresholds", {}).get(name, getattr(self.spec, name))

    def update_threshold(self, name, value, reason):
        """User-authorized adaptive criteria, versioned without rewriting observations."""
        if name not in self.ADAPTIVE_THRESHOLDS:
            raise ValueError("Unknown adaptive threshold")
        if not (name == "brightness_fraction" and value is None) and (not finite(value) or value < 0):
            raise ValueError("Thresholds must be finite and nonnegative")
        if not reason.strip():
            raise ValueError("Explain the threshold change")
        old = self.threshold(name)
        version = self.state.get("criteria_version", 0)+1
        self.state.setdefault("thresholds", {})[name] = value
        self.state["criteria_version"] = version
        self.db.execute("INSERT INTO criteria_revisions(campaign,action,version,name,old_value,new_value,reason,created) VALUES(?,?,?,?,?,?,?,?)",
                        (self.id, self.action_id, version, name, encode(old), encode(value), reason, time.time()))
        # Reassess existing measurements under the new criteria; retain earlier check rows.
        previous = self.db.execute("SELECT * FROM checks WHERE campaign=? ORDER BY id", (self.id,)).fetchall()
        latest = {(r["parameters"], r["name"]): r for r in previous}
        for (point, check_name), row in latest.items():
            details = json.loads(row["details"])
            details["reassesses_check_id"] = row["id"]
            if check_name == "gradient_accuracy":
                gradients = details["gradients"]
                fine = list(gradients.values())[-1]
                # Explicit fine gradient is saved because JSON's sorted keys are not profile order.
                fine = details.get("finest_gradient", fine)
                errors = details["component_errors"]
                thresholds = [self.threshold("gradient_atol")+self.threshold("gradient_rtol")*max(abs(g[i]) for g in gradients.values()) for i in range(len(fine))]
                passed = all(e <= t for e, t in zip(errors, thresholds))
                details.update(thresholds=thresholds, stationary_with_margin=passed and
                               details["projected_norm"]+details["uncertainty_margin"] < self.threshold("gradient_tolerance"))
            elif check_name == "objective_resolution":
                details["tolerance"] = self.threshold("objective_tolerance")
                passed = details["difference"] <= details["tolerance"]
            else:
                continue
            self._check(json.loads(point), check_name, passed, details)
        self.db.execute("UPDATE optimizer_runs SET status='criteria changed; reassess',updated=? WHERE campaign=?",
                        (time.time(), self.id))
        self._save()
        return {"status": "criteria updated", "version": version, "name": name, "old": old,
                "new": value, "reason": reason, "reassessed_checks": len(latest)}

    def _check(self, p, name, passed, details):
        p = self.canonical(p, outside=True)
        details = {**details, "criteria_version": self.state.get("criteria_version", 0)}
        self.db.execute("INSERT INTO checks(campaign,action,parameters,name,status,details,created) VALUES(?,?,?,?,?,?,?)",
                        (self.id, self.action_id, encode(p), name, "pass" if passed else "fail", encode(details), time.time()))
        self.db.commit()
        return {"status": "pass" if passed else "fail", **details}

    def gradient(self, p, resolution=None, *, outside=False):
        p = self.canonical(p, outside=outside)
        result = []
        for c in self.spec.controls:
            q, h = c.q(p[c.name]), c.fd_step
            lo, hi = self.state["bounds"][c.name]
            qlo, qhi = q-h, q+h
            if not c.periodic:
                qlo, qhi = max(c.q(lo), qlo), min(c.q(hi), qhi)
            if qhi <= qlo:
                raise InvalidEvaluation("No finite-difference interval")
            left, right = dict(p), dict(p)
            left[c.name], right[c.name] = c.wrap(c.physical(qlo)), c.wrap(c.physical(qhi))
            a = self.evaluate(left, resolution, "gradient_minus")
            b = self.evaluate(right, resolution, "gradient_plus")
            result.append((b["score"]-a["score"])/(qhi-qlo))
        return result

    def projected_gradient(self, p, g):
        result = []
        for c, value in zip(self.spec.controls, g):
            q = c.q(p[c.name])
            lo, hi = self.state["bounds"][c.name]
            result.append(value if c.periodic else q-min(c.q(hi), max(c.q(lo), q-value)))
        return result

    def check_gradient(self, p):
        gradients = {"search": self.gradient(p)}
        for name, profile in self.spec.refinements.items():
            gradients[name] = self.gradient(p, profile)
        fine = list(gradients.values())[-1]
        errors = [max(abs(g[i]-fine[i]) for g in gradients.values()) for i in range(len(fine))]
        thresholds = [self.threshold("gradient_atol") + self.threshold("gradient_rtol") * max(abs(g[i]) for g in gradients.values()) for i in range(len(fine))]
        passed = all(e <= t for e, t in zip(errors, thresholds))
        pg = max(abs(v) for v in self.projected_gradient(p, fine))
        details = {"gradients": gradients, "finest_gradient": fine, "component_errors": errors, "thresholds": thresholds,
                   "projected_norm": pg, "uncertainty_margin": max(errors),
                   "stationary_with_margin": passed and pg+max(errors) < self.threshold("gradient_tolerance")}
        return self._check(p, "gradient_accuracy", passed, details)

    def validate_candidate(self, p):
        # Carry over the reference notebook's baseline/selected finer-dz check.
        profile = next(iter(self.spec.refinements.values()))
        records = {}
        for label, point in (("baseline", self.spec.baseline), ("candidate", p)):
            coarse = self.evaluate(point, purpose=f"{label}_search")
            fine = self.evaluate(point, profile, f"{label}_fine")
            error = abs(coarse["score"]-fine["score"])
            records[label] = self._check(point, "objective_resolution", error <= self.threshold("objective_tolerance"),
                                        {"difference": error, "tolerance": self.threshold("objective_tolerance"),
                                         "coarse_id": coarse["id"], "fine_id": fine["id"]})
        return records

    def _run_state(self, run_id):
        row = self.db.execute("SELECT * FROM optimizer_runs WHERE id=? AND campaign=?", (run_id, self.id)).fetchone()
        if not row:
            raise ValueError("Unknown optimizer run in this campaign")
        return json.loads(row["state"])

    def _save_run(self, run_id, state, status):
        self.db.execute("UPDATE optimizer_runs SET state=?,status=?,updated=? WHERE id=? AND campaign=?",
                        (encode(state), status, time.time(), run_id, self.id))
        self.db.commit()

    def _outward(self, p, g):
        for c, slope in zip(self.spec.controls, g):
            if c.periodic or c.expansion <= 0:
                continue
            lo, hi = self.state["bounds"][c.name]
            near = c.max_change * 0.05
            if p[c.name] >= hi-near and slope < -self.threshold("gradient_atol"):
                return c, "upper"
            if p[c.name] <= lo+near and slope > self.threshold("gradient_atol") and (c.domain_lower is None or lo > c.domain_lower):
                return c, "lower"
        return None

    def expand(self, p, name=None, direction=None):
        checked = self.check_gradient(p)
        if checked["status"] != "pass":
            return {"status": "accuracy limited", "gradient_check": checked}
        outward = self._outward(p, checked["finest_gradient"])
        if not outward:
            return {"status": "range adequate locally"}
        c, side = outward
        if name is not None and (name != c.name or direction != side):
            return {"status": "requested expansion unsupported", "supported_control": c.name, "direction": side}
        lo, hi = self.state["bounds"][c.name]
        new = [lo, hi+c.expansion] if side == "upper" else [max(c.domain_lower if c.domain_lower is not None else -math.inf, lo-c.expansion), hi]
        trial = dict(p)
        trial[c.name] = min(new[1], hi+min(c.max_change, c.expansion)) if side == "upper" else max(new[0], lo-min(c.max_change, c.expansion))
        profile = list(self.spec.refinements.values())[-1]
        old = self.evaluate(p, profile, "expansion_reference")
        tested = self.evaluate(trial, profile, "expansion_probe", outside=True)
        if not self.usable(tested) or tested["score"] >= old["score"]:
            return {"status": "outward probe did not improve", "evaluation_id": tested["id"]}
        change = {"control": c.name, "old": [lo, hi], "new": new, "evidence": [old["id"], tested["id"]]}
        self.state["bounds"][c.name] = new
        self.state["range_changes"].append(change)
        self._save()
        return {"status": "range expanded", **change}

    def descend(self, decision):
        if decision.run_id is None:
            p = self.canonical(decision.parameters or self.spec.baseline)
            current = self.evaluate(p, purpose="descent_start")
            state = {"parameters": p, "iteration": 0, "evaluation_id": current["id"]}
            run_id = self.db.execute("INSERT INTO optimizer_runs(campaign,state,status,updated) VALUES(?,?,?,?)",
                                     (self.id, encode(state), "running", time.time())).lastrowid
            self.db.execute("INSERT INTO accepted_steps(campaign,run,iteration,evaluation,parameters,score,created) VALUES(?,?,?,?,?,?,?)",
                            (self.id, run_id, 0, current["id"], encode(p), current["score"], time.time()))
            self.db.commit()
        else:
            run_id, state = decision.run_id, self._run_state(decision.run_id)
        status = "iteration allocation reached"
        try:
            for _ in range(decision.iterations):
                p = state["parameters"]
                current = self.evaluate(p, purpose="descent_current")
                if not self.usable(current):
                    status = "brightness requirement not met"; break
                g = self.gradient(p)
                unresolved_boundary = False
                if self._outward(p, g):
                    expansion = self.expand(p)
                    if expansion["status"] == "accuracy limited":
                        status = "accuracy limited"; break
                    unresolved_boundary = expansion["status"] != "range expanded"
                pg = max(abs(v) for v in self.projected_gradient(p, g))
                if pg < self.threshold("gradient_tolerance"):
                    if unresolved_boundary:
                        status = "no acceptable step"; break
                    checked = self.check_gradient(p)
                    # Boundary evidence is evaluated before this stationarity claim.
                    status = "locally converged" if checked["stationary_with_margin"] else "accuracy limited"
                    break
                accepted = False
                for attempt in range(self.spec.backtracks):
                    factor = self.spec.learning_rate / 2**attempt
                    trial, displacement = {}, []
                    for c, slope in zip(self.spec.controls, g):
                        value = c.physical(c.q(p[c.name])-factor*slope)
                        delta = max(-c.max_change, min(c.max_change, value-p[c.name]))
                        lo, hi = self.state["bounds"][c.name]
                        value = c.wrap(p[c.name]+delta) if c.periodic else min(hi, max(lo, p[c.name]+delta))
                        trial[c.name] = value
                        displacement.append(c.difference(value, p[c.name])/c.scale if c.periodic else c.q(value)-c.q(p[c.name]))
                    if max(abs(v) for v in displacement) < 1e-12:
                        break
                    try:
                        row = self.evaluate(trial, purpose="line_search")
                    except InvalidEvaluation:
                        continue
                    directional = sum(a*b for a, b in zip(g, displacement))
                    if self.usable(row) and row["score"] < current["score"] and row["score"] <= current["score"]+self.spec.armijo*directional:
                        state.update(parameters=trial, iteration=state["iteration"]+1, evaluation_id=row["id"])
                        self.db.execute("INSERT INTO accepted_steps(campaign,run,iteration,evaluation,parameters,score,created) VALUES(?,?,?,?,?,?,?)",
                                        (self.id, run_id, state["iteration"], row["id"], encode(trial), row["score"], time.time()))
                        self._save_run(run_id, state, "running")
                        accepted = True
                        break
                if not accepted:
                    status = "no acceptable step"; break
        except AllowanceReached:
            status = "budget reached"
        except BaseException:
            self._save_run(run_id, state, "interrupted")
            raise
        self._save_run(run_id, state, status)
        return {"status": status, "run_id": run_id, **state}

    def robustness(self, p, samples):
        if not self.spec.robustness or not any(self.spec.robustness.values()):
            return {"status": "equipment precision not specified"}
        rng = random.Random(digest(p))
        profile = list(self.spec.refinements.values())[-1]
        center = self.evaluate(p, profile, "robustness_center")
        rows = []
        for _ in range(samples):
            point = dict(p)
            for c in self.spec.controls:
                v = point[c.name] + rng.uniform(-1, 1)*self.spec.robustness.get(c.name, 0.)
                if c.domain_lower is not None:
                    v = max(c.domain_lower, v)
                point[c.name] = c.wrap(v)
            rows.append(self.evaluate(point, profile, "combined_input_perturbation", outside=True))
        return {"status": "robustness sampled", "center": center["id"],
                "evaluation_ids": [r["id"] for r in rows],
                "observed_score_range": [min(r["score"] for r in [center]+rows), max(r["score"] for r in [center]+rows)],
                "note": "Uniform bounded test perturbations, not a measured equipment-error distribution."}

    def execute(self, decision: Decision):
        allowed = {"evaluate", "descend", "validate_gradient", "validate_candidate", "expand_range", "robustness", "update_threshold", "pause"}
        if decision.action not in allowed:
            raise ValueError("Unknown planner action")
        if type(decision.evaluations) is not int or type(decision.iterations) is not int or min(decision.evaluations, decision.iterations) < 0:
            raise ValueError("Allocations must be nonnegative integers")
        if decision.action not in ("pause", "update_threshold") and decision.evaluations == 0:
            raise ValueError("Allocate at least one evaluation")
        self.action_id = self.db.execute("INSERT INTO actions(campaign,decision,status,created) VALUES(?,?,'running',?)",
                                         (self.id, encode(asdict(decision)), time.time())).lastrowid
        self.db.commit()
        self.action_cap, self.action_used = decision.evaluations, 0
        self._clock_start = time.monotonic()
        result = {}
        try:
            p = decision.parameters or (self._run_state(decision.run_id)["parameters"] if decision.run_id else self.spec.baseline)
            if decision.action == "pause":
                result = {"status": "planner checkpoint", "reason": decision.rationale}
            elif decision.action == "update_threshold":
                result = self.update_threshold(decision.threshold_name, decision.threshold_value, decision.rationale)
            elif decision.action == "evaluate":
                row = self.evaluate(p)
                result = {"status": "evaluated", "evaluation_id": row["id"], "score": row["score"], "usable": self.usable(row)}
            elif decision.action == "descend":
                result = self.descend(decision)
            elif decision.action == "validate_gradient":
                result = {"status": "gradient checked", "check": self.check_gradient(p)}
            elif decision.action == "validate_candidate":
                result = {"status": "candidate checked", "checks": self.validate_candidate(p)}
            elif decision.action == "expand_range":
                result = self.expand(p, decision.control, decision.direction)
            elif decision.action == "robustness":
                result = self.robustness(p, decision.iterations)
        except AllowanceReached as exc:
            result = {"status": "budget reached", "reason": str(exc)}
        except Exception as exc:
            result = {"status": "action failed", "error": f"{type(exc).__name__}: {exc}"}
        except BaseException:
            result = {"status": "interrupted"}
            raise
        finally:
            elapsed = time.monotonic()-self._clock_start
            result["new_evaluations"] = self.action_used
            self.db.execute("UPDATE actions SET status=?,result=?,seconds=? WHERE id=?",
                            (result.get("status", "interrupted"), encode(result), elapsed, self.action_id))
            self._clock_start = None
            self._save(result.get("status", "interrupted"))
            self.action_id, self.action_cap = None, None
        return result

    def rows(self):
        rows = self.db.execute("SELECT DISTINCT e.* FROM evaluations e JOIN evaluation_uses u ON e.id=u.evaluation WHERE u.campaign=? ORDER BY e.id", (self.id,)).fetchall()
        result = []
        baseline_check = self.db.execute("SELECT status FROM checks WHERE campaign=? AND parameters=? AND name='objective_resolution' ORDER BY id DESC LIMIT 1",
                                         (self.id, encode(self.canonical(self.spec.baseline)))).fetchone()
        for row in rows:
            r = self._evaluation(row)
            ratio = r["metrics"].get(self.spec.brightness_metric)
            r["brightness_status"] = "not specified" if self.threshold("brightness_fraction") is None else (
                "pass" if finite(ratio) and ratio >= self.threshold("brightness_fraction") else "below target")
            checks = self.db.execute("SELECT name,status,details FROM checks WHERE campaign=? AND parameters=? ORDER BY id", (self.id, encode(r["parameters"]))).fetchall()
            latest = {x["name"]: {"status": x["status"], "details": json.loads(x["details"])} for x in checks}
            adapter_ok = all(v is True for v in r["checks"].values())
            required = ("objective_resolution", "gradient_accuracy")
            r["numerical_status"] = "invalid" if r["status"] != "complete" else (
                "validated" if adapter_ok and baseline_check and baseline_check["status"] == "pass" and
                all(latest.get(k, {}).get("status") == "pass" for k in required) else "provisional")
            r["validation"] = latest
            r["criteria_version"] = self.state.get("criteria_version", 0)
            result.append(r)
        return result

    def context(self):
        rows = self.rows()
        complete = [r for r in rows if r["status"] == "complete"]
        best = sorted(complete, key=lambda r: r["score"])[:12]
        recent = rows[-16:]
        near = [r for r in complete if r["brightness_status"] == "below target" or r["numerical_status"] == "provisional"][-8:]
        chosen = {r["id"]: r for r in best+recent+near}
        fields = ("id", "parameters", "resolution", "score", "metrics", "checks", "numerical_status", "brightness_status", "error", "validation")
        runs = [dict(r) for r in self.db.execute("SELECT * FROM optimizer_runs WHERE campaign=? ORDER BY id", (self.id,))]
        for r in runs:
            r["state"] = json.loads(r["state"])
        return {"campaign": self.id, "spec": asdict(self.spec), "state": self.state,
                "effective_thresholds": {k: self.threshold(k) for k in self.ADAPTIVE_THRESHOLDS},
                "budget": self.budget(), "total_recorded_evaluations": len(rows),
                "history": [{k: r.get(k) for k in fields} for r in chosen.values()],
                "optimizer_runs": runs,
                "cost_hint": {"one_gradient_max_new_evaluations": 2*len(self.spec.controls),
                              "gradient_check_max_new_evaluations": 2*len(self.spec.controls)*(1+len(self.spec.refinements))},
                "recent_actions": [dict(r) for r in self.db.execute("SELECT id,decision,status,result FROM actions WHERE campaign=? ORDER BY id DESC LIMIT 8", (self.id,))]}

    def plan(self, planner):
        budget = self.budget()
        if budget["planner_calls_left"] <= 0 or budget["seconds_left"] <= 0:
            raise AllowanceReached("planner/time allowance reached")
        context = self.context()
        pid = self.db.execute("INSERT INTO proposals(campaign,status,context,created) VALUES(?,'running',?,?)",
                              (self.id, encode(context), time.time())).lastrowid
        self.db.commit()
        started = time.monotonic()
        try:
            decision, metadata = planner(context, self.artifact_root/f"proposal_{pid:05d}", budget["seconds_left"])
            if not isinstance(decision, Decision):
                raise TypeError("Planner must return Decision")
            known = {r["id"] for r in self.rows()}
            if any(type(i) is not int or i not in known for i in decision.evidence_ids):
                raise ValueError("Planner cited an unknown evaluation ID")
            self.db.execute("UPDATE proposals SET status='complete',decision=?,metadata=?,seconds=? WHERE id=?",
                            (encode(asdict(decision)), encode(metadata), time.monotonic()-started, pid))
            self.db.commit()
            return decision
        except BaseException as exc:
            self.db.execute("UPDATE proposals SET status='failed',error=?,seconds=? WHERE id=?",
                            (f"{type(exc).__name__}: {exc}", time.monotonic()-started, pid))
            self.db.commit()
            raise

    def run(self, planner, max_decisions=5, on_event=print):
        """A finite decision batch. Calling again resumes the same database campaign."""
        results = []
        for _ in range(max_decisions):
            if self.budget()["evaluations_left"] <= 0 or self.budget()["seconds_left"] <= 0:
                self._save("budget reached"); break
            try:
                decision = self.plan(planner)
                on_event({"AI proposal": asdict(decision)})
                result = self.execute(decision)
                on_event({"measured result": result, "remaining": self.budget()})
                results.append(result)
                if decision.action == "pause":
                    break
            except AllowanceReached:
                self._save("budget reached"); break
            except KeyboardInterrupt:
                self._save("interrupted"); raise
            except Exception as exc:
                self._save("planner error")
                on_event({"status": "planner error", "error": str(exc)})
                break
        return results

    def recover_interrupted(self):
        """Call only after confirming no other process owns this campaign."""
        for table in ("evaluations", "actions", "proposals"):
            # A hard kill loses the final timer update. Conservatively account for the
            # elapsed interval; an explicit new allowance can resume an expired one.
            self.db.execute(f"UPDATE {table} SET status='interrupted',seconds=MAX(seconds,?-created) WHERE campaign=? AND status='running'", (time.time(), self.id))
        self.db.commit()
        self._save("ready")
