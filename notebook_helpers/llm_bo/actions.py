"""Eight-action D_IO notebook interface, backed by the existing experiment engine."""
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import random

from ai_augmented_loop.loop import (
    AllowanceReached, Control, Decision, ExperimentLoop, Spec, digest,
)
from ai_augmented_loop.evaluators import FIBER_SETUP, OpticsEvaluator
from .notebook import ROOT, problem_card, prepare_ai
from .runtime import GPPrior, NAMES, check_point, connect_llm, object_schema, write_json

ACTIONS = ("evaluate", "descend", "validate_candidate", "validate_gradient",
           "expand_range", "robustness", "update_threshold", "pause")
PHYSICAL_NAMES = (*NAMES, "power_kW")
SEARCH = {"Nt": 2048, "dz_m": .005, "adaptive_power_search": True}
FINE = {"Nt": 2048, "dz_m": .001, "adaptive_power_search": True}
FINE_TIME = {"Nt": 4096, "dz_m": .001, "adaptive_power_search": True}


def to_coordinates(parameters):
    """Two independent bounded coordinates cover the entire amplitude simplex."""
    if not isinstance(parameters, dict) or set(parameters) != set(PHYSICAL_NAMES):
        raise ValueError("Supply c1, c2, c3, and power_kW.")
    c = check_point({k: parameters[k] for k in NAMES})
    power = parameters["power_kW"]
    if type(power) not in (int, float) or not math.isfinite(power) or power <= 0:
        raise ValueError("power_kW must be finite and positive.")
    rest = c["c2"] + c["c3"]
    return {"amplitude_fraction": c["c1"] / 100,
            "lp11_balance": c["c2"] / rest if rest else .5, "power_kW": float(power)}


def to_parameters(coordinates):
    u, v = coordinates["amplitude_fraction"], coordinates["lp11_balance"]
    return {"c1": 100*u, "c2": 100*(1-u)*v, "c3": 100*(1-u)*(1-v),
            "power_kW": coordinates["power_kW"]}


def decision_schema():
    parameter_fields = {k: {"type": "number", "minimum": 0, "maximum": 100} for k in NAMES}
    parameter_fields["power_kW"] = {"type": "number", "exclusiveMinimum": 0}
    return object_schema({
        "action": {"type": "string", "enum": list(ACTIONS)},
        "parameters": {**object_schema(parameter_fields), "type": ["object", "null"]},
        "run_id": {"type": ["integer", "null"], "minimum": 1},
        "evaluations": {"type": "integer", "minimum": 0},
        "iterations": {"type": "integer", "minimum": 0},
        "evidence_ids": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "rationale": {"type": "string"}, "knowledge_basis": {"type": "string"},
        "expected_outcome": {"type": "string"},
        "control": {"type": ["string", "null"], "enum": ["power_kW", None]},
        "direction": {"type": ["string", "null"], "enum": ["upper", "lower", None]},
        "threshold_name": {"type": ["string", "null"],
                           "enum": [*ExperimentLoop.ADAPTIVE_THRESHOLDS, None]},
        "threshold_value": {"type": ["number", "null"], "minimum": 0},
    })


def parse_decision(payload):
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(Decision)}:
        raise ValueError("Return all decision fields, with null for unused values.")
    data = deepcopy(payload)
    if data["action"] not in ACTIONS:
        raise ValueError("Unknown action.")
    for key in ("evaluations", "iterations"):
        if type(data[key]) is not int or data[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    if data["run_id"] is not None and (type(data["run_id"]) is not int or data["run_id"] < 1):
        raise ValueError("run_id must be null or a positive saved trajectory ID.")
    if not isinstance(data["evidence_ids"], list) or any(type(i) is not int or i < 1 for i in data["evidence_ids"]):
        raise ValueError("evidence_ids must contain saved evaluation numbers.")
    for key in ("rationale", "knowledge_basis", "expected_outcome"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"Supply a brief {key}.")
    if data["parameters"] is not None:
        if data["run_id"] is not None:
            raise ValueError("For a resumed trajectory, use parameters=null.")
        data["parameters"] = to_coordinates(data["parameters"])
    if data["action"] not in ("pause", "update_threshold"):
        if data["evaluations"] < 1:
            raise ValueError("Allocate at least one evaluation; cache hits remain free.")
        if data["parameters"] is None and data["run_id"] is None:
            raise ValueError("Supply a point or a saved trajectory ID.")
    if data["action"] in ("descend", "robustness") and data["iterations"] < 1:
        raise ValueError("Allocate at least one iteration.")
    if data["action"] == "expand_range":
        if data["control"] != "power_kW" or data["direction"] not in ("lower", "upper"):
            raise ValueError("Only power_kW can expand, with direction lower or upper.")
    elif data["control"] is not None or data["direction"] is not None:
        raise ValueError("control and direction apply only to expand_range.")
    if data["action"] == "update_threshold":
        if data["threshold_name"] not in ExperimentLoop.ADAPTIVE_THRESHOLDS:
            raise ValueError("Unknown threshold.")
        value = data["threshold_value"]
        if value is None:
            if data["threshold_name"] != "brightness_fraction":
                raise ValueError("Only the brightness preference may be disabled with null.")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Threshold must be finite and nonnegative.")
    elif data["threshold_name"] is not None or data["threshold_value"] is not None:
        raise ValueError("Threshold fields apply only to update_threshold.")
    return Decision(**data)


def public_decision(decision):
    data = asdict(decision)
    if data["parameters"] is not None:
        data["parameters"] = to_parameters(data["parameters"])
    return data


class AmplitudeEvaluator(OpticsEvaluator):
    """One budgeted request is exactly one nonlinear propagation."""
    def __init__(self, directory, power_kW=100., timeout_seconds=300):
        super().__init__(ROOT, FIBER_SETUP)
        self.worker = Path(__file__).with_name("dio_worker.jl")
        self.reference_file = Path(directory) / "brightness_reference.json"
        self.reference = json.loads(self.reference_file.read_text()) if self.reference_file.exists() else None
        self.power_kW, self.timeout_seconds = power_kW, timeout_seconds
        self.identity.update(
            launch="zero_phase_amplitudes_and_adaptive_power",
            brightness_reference="fixed_initial_power_equal_amplitude_1mm",
            reference_power_kW=power_kW,
            worker_sha256=hashlib.sha256(self.worker.read_bytes()).hexdigest(),
            action_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )

    def __call__(self, coordinates, resolution, directory, timeout_seconds):
        point = to_parameters(coordinates)
        result = super().__call__(point, resolution, directory, min(timeout_seconds, self.timeout_seconds))
        if self.reference is None:
            baseline = (math.isclose(point["power_kW"], self.power_kW)
                        and all(math.isclose(point[k], 100/3, abs_tol=1e-9) for k in NAMES)
                        and resolution == FINE)
            if not baseline or not result["checks"]["valid"] or result["metrics"]["LP01_photons"] <= 0:
                raise ValueError("Initialize the valid equal-amplitude 1 mm brightness reference first.")
            self.reference = {"photons": result["metrics"]["LP01_photons"],
                              "power_kW": self.power_kW, "fields": result["artifacts"]["fields"]}
            write_json(self.reference_file, self.reference)
        result["metrics"].update(
            brightness_ratio=result["metrics"]["LP01_photons"] / self.reference["photons"],
            launch_parameters=point,
        )
        write_json(Path(directory) / "measurement.json", {"parameters": point, "resolution": resolution, **result})
        return result


class AmplitudeLoop(ExperimentLoop):
    def canonical(self, parameters, *, outside=False):
        result = super().canonical(parameters, outside=outside)
        if any(not 0 <= result[k] <= 1 for k in ("amplitude_fraction", "lp11_balance")):
            raise ValueError("Amplitude coordinates must stay within the physical simplex.")
        return result

    def validate_candidate(self, p):
        records = {}
        for label, point in (("baseline", self.spec.baseline), ("candidate", p)):
            # Fine baseline comes first so its photons define every recorded ratio.
            fine = self.evaluate(point, FINE, f"{label}_fine")
            coarse = self.evaluate(point, purpose=f"{label}_search")
            error = abs(coarse["score"] - fine["score"])
            records[label] = self._check(point, "objective_resolution", error <= self.threshold("objective_tolerance"),
                {"difference": error, "tolerance": self.threshold("objective_tolerance"),
                 "coarse_id": coarse["id"], "fine_id": fine["id"]})
        return records

    def robustness(self, p, samples):
        rng = random.Random(digest(p))
        center = self.evaluate(p, FINE_TIME, "robustness_center")
        rows = []
        for _ in range(samples):
            point = {c.name: p[c.name] + rng.uniform(-1, 1)*self.spec.robustness[c.name]
                     for c in self.spec.controls}
            for key in ("amplitude_fraction", "lp11_balance"):
                point[key] = min(1., max(0., point[key]))
            point["power_kW"] = max(.001, point["power_kW"])
            rows.append(self.evaluate(point, FINE_TIME, "combined_input_perturbation", outside=True))
        return {"status": "robustness sampled", "center": center["id"],
                "evaluation_ids": [r["id"] for r in rows],
                "observed_score_range": [min(r["score"] for r in [center]+rows), max(r["score"] for r in [center]+rows)],
                "brightness_passes": [self.usable(r) for r in rows],
                "note": "Declared bounded sensitivity probes; not measured equipment uncertainties. "
                        "Probes may cross a search boundary, but do not expand its bounds."}


def make_spec(evaluator, power_kW, power_range_kW):
    lo, hi = power_range_kW
    if not (0 < lo < hi and lo <= power_kW <= hi):
        raise ValueError("Initial power must lie in a positive, ordered power range.")
    return Spec(
        name="D_IO amplitudes and adaptive power", objective="Minimize unaligned complex LP01 D_IO",
        controls=[Control("amplitude_fraction", 0, 1, fd_step=.005, max_change=.1, expansion=0, domain_lower=0),
                  Control("lp11_balance", 0, 1, fd_step=.005, max_change=.1, expansion=0, domain_lower=0),
                  Control("power_kW", lo, hi, scale=100, fd_step=.005, max_change=10, expansion=25,
                          domain_lower=.001, unit="kW")],
        baseline={"amplitude_fraction": 1/3, "lp11_balance": .5, "power_kW": power_kW},
        evaluator_identity=evaluator.identity, search=SEARCH,
        refinements={"fine_dz": FINE, "fine_time": FINE_TIME},
        brightness_fraction=.5, brightness_policy="hard",
        robustness={"amplitude_fraction": .005, "lp11_balance": .005, "power_kW": 1.},
        knowledge="Three zero-phase LP modes in the reference fiber. The two LP11 orientations are "
                  "degenerate. Total power may vary; the brightness denominator stays fixed at the "
                  "initial equal-amplitude 1 mm baseline. Lower power can improve D_IO by weakening "
                  "nonlinearity, so power changes must be reported with brightness. An amplitude "
                  "fraction u and split v map to c=(100u,100(1-u)v,100(1-u)(1-v)).",
    )


class ActionStudy:
    def __init__(self, directory, experiment, evaluator, config, connection_factory=connect_llm):
        self.directory, self.experiment, self.evaluator = Path(directory), experiment, evaluator
        self.config, self.connection_factory = config, connection_factory

    def _connection(self, directory, seconds):
        return self.connection_factory(directory / "ai", max_calls=1, model=self.config["model"],
            reasoning_effort=self.config["thinking_effort"],
            timeout_seconds=min(seconds, self.config["timeout_seconds"]))

    def prepare_ai(self):
        path = self.directory / "initial_plan.json"
        if path.exists():
            return json.loads(path.read_text())
        # Recover an already committed prior if interruption preceded its JSON export.
        for row in self.experiment.db.execute(
                "SELECT metadata FROM proposals WHERE campaign=? AND status='complete' ORDER BY id",
                (self.experiment.id,)):
            saved = json.loads(row["metadata"] or "{}")
            if "initial_plan" in saved:
                write_json(path, saved["initial_plan"])
                return saved["initial_plan"]
        def planner(context, directory, seconds):
            ask = self._connection(directory, seconds)
            problem = problem_card(100.)
            problem["power_kW"] = self.config["power_kW"]
            problem["future_actions"] = list(ACTIONS)
            problem["power_search"] = {
                "initial_range_kW": self.config["power_range_kW"],
                "initial_point_power_kW": self.config["power_kW"],
                "note": "Choose the initial mixture at the initial power; later actions may vary power. "
                        "This GP describes the mixture only and is not fitted after initialization.",
            }
            plan, _ = prepare_ai(problem, ask)
            plan["initial_point"]["power_kW"] = self.config["power_kW"]
            decision = Decision("evaluate", to_coordinates(plan["initial_point"]), evaluations=1,
                rationale=plan["rationale"], knowledge_basis="Initial GP mixture prior and supplied fiber context.",
                expected_outcome="Measure D_IO and brightness at this previously untested mixture.")
            # The proposal and its prior share one durable record before any simulations.
            return decision, {"initial_plan": plan, "model": self.config["model"],
                              "thinking_effort": self.config["thinking_effort"]}
        self.experiment.plan(planner)
        row = self.experiment.db.execute(
            "SELECT metadata FROM proposals WHERE campaign=? AND status='complete' ORDER BY id DESC LIMIT 1",
            (self.experiment.id,)).fetchone()
        plan = json.loads(row["metadata"])["initial_plan"]
        write_json(path, plan)
        return plan

    def initialize(self):
        if self.experiment.state.get("baseline_ready"):
            return
        result = self.experiment.execute(Decision("validate_candidate", self.experiment.spec.baseline,
            evaluations=2, rationale="Establish the original 5 mm/1 mm baseline and fixed brightness reference."))
        checks = result.get("checks", {})
        if result["status"] != "candidate checked" or not all(v["status"] == "pass" for v in checks.values()):
            self.save()
            raise ValueError(f"Baseline initialization did not pass: {result}")
        self.experiment.state["baseline_ready"] = True
        self.experiment._save()
        self.save()

    def evaluate_initial(self):
        if self.experiment.state.get("initial_evaluation_id"):
            return self.experiment.state["initial_evaluation_id"]
        plan = self.prepare_ai()
        self.initialize()
        result = self.experiment.execute(Decision("evaluate", to_coordinates(plan["initial_point"]),
            evaluations=1, rationale=plan["rationale"], knowledge_basis="Initial GP prior.",
            expected_outcome="Measure the initial mixture before selecting subsequent actions."))
        if result["status"] != "evaluated":
            self.save()
            raise ValueError(f"Initial evaluation did not complete: {result}")
        self.experiment.state["initial_evaluation_id"] = result["evaluation_id"]
        self.experiment._save()
        self.save()
        return result["evaluation_id"]

    def _context(self):
        context = self.experiment.context()
        for row in context["history"]:
            row["parameters"] = to_parameters(row["parameters"])
        for run in context["optimizer_runs"]:
            run["state"]["parameters"] = to_parameters(run["state"]["parameters"])
        return {
            "operation": "choose_next_action", "problem": context["spec"]["objective"],
            "physics": self.experiment.spec.knowledge,
            "parameters": {"c1_c2_c3": "Each in [0,100], sum 100; zero phases.",
                           "power_kW": context["state"]["bounds"]["power_kW"]},
            "baseline": to_parameters(self.experiment.spec.baseline),
            "brightness_reference": "Fixed equal-amplitude 1 mm reference at initial power, not renormalized after power changes.",
            "observations": context["history"], "optimizer_runs": context["optimizer_runs"],
            "budget": context["budget"], "effective_thresholds": context["effective_thresholds"],
            "criteria_version": context["state"]["criteria_version"],
            "range_changes": context["state"]["range_changes"],
            "recent_actions": context["recent_actions"],
            "allowed_actions": list(ACTIONS),
            "cost_hint": {**context["cost_hint"], "candidate_check_max_new_evaluations": 4,
                          "one_descent_step_typically": "6 gradient probes plus line search; may need refinement at boundaries.",
                          "range_expansion_max_new_evaluations": 20,
                          "robustness": "One Nt=4096 center plus iterations perturbations; cache hits are free."},
            "robustness_perturbations": "Uniform +/-0.005 in LP01 amplitude fraction and LP11 split, +/-1 kW; sensitivity probes only.",
            "instruction": "Choose one allowed action and return every decision field. Use measured evidence IDs. "
                "Balance learning with lower feasible D_IO; expected_outcome is a prediction, not a measurement. "
                "Use a short rationale, a physical hypothesis in knowledge_basis, and a testable expected_outcome. "
                "Provide all three amplitudes and power_kW for a new point; run_id resumes a saved descent "
                "with parameters=null. evaluations limits new simulations, iterations limits descent steps or "
                "robustness samples. Keep allocations within the remaining grant, accounting for gradient costs. "
                "Only power_kW may expand (25 kW per supported expansion); use a boundary point and upper/lower. "
                "Python requires accurate outward slope and a refined improving probe. Nonpositive power and "
                "amplitude constraints cannot be relaxed. update_threshold uses threshold_name and threshold_value; "
                "explain the scientific tradeoff, not merely passing a failed check. Revisions preserve original "
                "measurements and criteria. null brightness_fraction disables that preference. Looser tolerances "
                "do not improve numerical precision. validate_candidate checks 5 mm/1 mm; validate_gradient "
                "also checks Nt=4096. A pause is resumable and makes no convergence claim. With no new simulations "
                "left, choose pause. Cache hits do not use simulation allowance. You cannot increase the global budgets.",
        }

    def choose_next(self):
        def planner(context, directory, seconds):
            request = self._context()
            ask = self._connection(directory, seconds)
            decision = parse_decision(ask(request, decision_schema()))
            if decision.parameters is not None:
                self.experiment.canonical(decision.parameters)
            if decision.run_id is not None:
                self.experiment._run_state(decision.run_id)
            return decision, {"model": self.config["model"], "thinking_effort": self.config["thinking_effort"]}
        decision = public_decision(self.experiment.plan(planner))
        print(json.dumps(decision, indent=2))
        self.save()
        return decision

    def execute(self, decision):
        parsed = parse_decision(decision)
        known = {r["id"] for r in self.experiment.rows()}
        if any(i not in known for i in parsed.evidence_ids):
            raise ValueError("Unknown evidence ID in this campaign.")
        result = self.experiment.execute(parsed)
        print("Measured action result:", json.dumps(result, indent=2))
        self.save()
        return result

    def run(self, decisions=3):
        if type(decisions) is not int or decisions < 1:
            raise ValueError("decisions must be a positive integer.")
        try:
            self.prepare_ai()
            self.evaluate_initial()
            for _ in range(decisions):
                decision = self.choose_next()
                self.execute(decision)
                if decision["action"] == "pause":
                    self.experiment._save("paused")
                    break
            else:
                budget = self.experiment.budget()
                exhausted = any(budget[k] <= 0 for k in ("evaluations_left", "planner_calls_left", "seconds_left"))
                self.experiment._save("budget reached" if exhausted else "decision batch reached")
        except AllowanceReached as error:
            self.experiment._save("budget reached")
            print(str(error))
        except BaseException:
            self.experiment._save("interrupted or failed; inspect saved records")
            raise
        finally:
            self.evaluator.close()
            self.save()
        self.remaining()

    def save(self):
        rows = self.experiment.rows()
        public_rows = deepcopy(rows)
        for row in public_rows:
            row["parameters"] = to_parameters(row["parameters"])
        actions = []
        for row in self.experiment.db.execute("SELECT * FROM actions WHERE campaign=? ORDER BY id", (self.experiment.id,)):
            record = dict(row)
            record["decision"] = public_decision(Decision(**json.loads(record["decision"])))
            record["result"] = json.loads(record["result"]) if record["result"] else None
            actions.append(record)
        write_json(self.directory / "history.json", public_rows)
        write_json(self.directory / "actions.json", actions)
        write_json(self.directory / "state.json", {"campaign_id": self.experiment.id,
            **self.experiment.state, "budget": self.experiment.budget()})

    def remaining(self):
        print("Status:", self.experiment.state["status"])
        print("Remaining:", json.dumps(self.experiment.budget(), indent=2))
        print("Resume directory:", self.directory)

    def results(self):
        import pandas as pd
        from IPython.display import display
        self.save()
        rows = json.loads((self.directory / "history.json").read_text())
        display(pd.DataFrame([{"id": r["id"], **r["parameters"], "D_IO": r["score"],
            "brightness": r["metrics"].get("brightness_ratio"), "brightness_status": r["brightness_status"],
            "Nt": r["resolution"].get("Nt"), "dz_m": r["resolution"].get("dz_m"),
            "numerical_status": r["numerical_status"]} for r in rows]))
        self.remaining()

    def allow_more(self, *, tests=0, ai_calls=0, minutes=0):
        self.experiment.grant(tests, 60*minutes, ai_calls, "explicit notebook allocation")
        self.save()

    def close(self):
        self.evaluator.close()
        self.experiment.close()


def start_experiment(*, power_kW=100., power_range_kW=(81.943, 100.), max_tests=6,
                     max_ai_calls=4, max_minutes=10., model=None, thinking_effort=None,
                     timeout_seconds=300, directory=None):
    """Create a separate resumable campaign; no AI calls or simulations yet."""
    if (type(max_tests) is not int or max_tests < 3 or type(max_ai_calls) is not int or max_ai_calls < 1
            or not math.isfinite(max_minutes) or max_minutes <= 0
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Use at least three tests, one AI call, and positive time limits.")
    directory = Path(directory).expanduser() if directory else ROOT / "output" / datetime.now(timezone.utc).strftime("%Y-%m-%d_llm_actions_%H%M%S_%fZ")
    if not directory.is_absolute():
        directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=False)
    evaluator = AmplitudeEvaluator(directory, power_kW, timeout_seconds)
    spec = make_spec(evaluator, power_kW, power_range_kW)
    experiment = AmplitudeLoop(directory / "experiments.sqlite", spec, evaluator,
                               artifact_root=directory / "artifacts")
    experiment.grant(max_tests, 60*max_minutes, max_ai_calls)
    config = {"campaign_id": experiment.id, "power_kW": power_kW, "power_range_kW": list(power_range_kW),
              "model": model, "thinking_effort": thinking_effort, "timeout_seconds": timeout_seconds,
              "spec": asdict(spec)}
    write_json(directory / "experiment.json", config)
    print("Saving this action campaign to:", directory)
    return ActionStudy(directory, experiment, evaluator, config)


def resume_experiment(directory):
    """Reopen a paused campaign with its original model, criteria, and remaining budget."""
    directory = Path(directory).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve()
    config = json.loads((directory / "experiment.json").read_text())
    evaluator = AmplitudeEvaluator(directory, config["power_kW"], config["timeout_seconds"])
    data = deepcopy(config["spec"])
    if evaluator.identity != data["evaluator_identity"]:
        raise ValueError("Evaluator changed; start a new campaign. Saved data remain intact.")
    data["controls"] = [Control(**c) for c in data["controls"]]
    experiment = AmplitudeLoop(directory / "experiments.sqlite", Spec(**data), evaluator,
                               campaign_id=config["campaign_id"], artifact_root=directory / "artifacts")
    if any(experiment.db.execute(f"SELECT 1 FROM {table} WHERE campaign=? AND status='running'", (experiment.id,)).fetchone()
           for table in ("actions", "proposals", "evaluations")):
        experiment.close()
        raise RuntimeError("Campaign has an unfinished operation. Confirm its old runner has stopped before recovery.")
    return ActionStudy(directory, experiment, evaluator, config)
