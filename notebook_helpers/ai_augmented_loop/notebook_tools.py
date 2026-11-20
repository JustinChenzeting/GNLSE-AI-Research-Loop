"""Small notebook conveniences; the experiment engine does not depend on Jupyter."""
from dataclasses import asdict
from pathlib import Path
from contextlib import redirect_stdout
import html
import io
import json
import math
import sqlite3

from loop import Control, Spec, ExperimentLoop, Decision
from presets import analytic_demo, optics_preset


def ask_preferences(example):
    # Backwards-compatible entry point; the simple notebook supplies visible settings.
    while True:
        raw = input("Brightness target in percent [50], or 'none': ").strip().lower() or "50"
        try:
            fraction = None if raw == "none" else float(raw)/100
            if fraction is not None and (not math.isfinite(fraction) or fraction < 0):
                raise ValueError
            break
        except ValueError:
            print("Enter a nonnegative number, such as 50, or 'none'.")
    print("Using a soft target, 80 tests, 60 minutes and 10 Codex calls.")
    return {"brightness_fraction": fraction, "brightness_policy": "soft",
            "evaluations": 80, "seconds": 3600, "planner_calls": 10, "robustness": {}}


def open_example(database, workspace, example="dio_power", *, resume="latest", preferences=None):
    """Resume stored settings without asking again; new campaigns prompt for preferences."""
    database = Path(database)
    name = "analytic demonstration" if example == "demo" else example
    saved = None
    if database.is_file() and resume:
        with sqlite3.connect(database) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM campaigns ORDER BY updated DESC").fetchall()
            for row in rows:
                if (resume == "latest" and json.loads(row["spec"])["name"] == name) or resume == row["id"]:
                    saved = dict(row); break
        if resume != "latest" and not saved:
            raise ValueError("Requested campaign ID does not exist")
    if saved:
        data = json.loads(saved["spec"])
        prefs = {"brightness_fraction": data["brightness_fraction"], "brightness_policy": data["brightness_policy"],
                 "robustness": data["robustness"]}
    else:
        prefs = preferences if preferences is not None else ask_preferences(example)
    if example == "demo":
        spec, evaluator = analytic_demo(prefs["brightness_fraction"], prefs["brightness_policy"])
        spec.robustness = prefs.get("robustness", {})
    else:
        spec, evaluator = optics_preset(workspace, kind=example, brightness_fraction=prefs["brightness_fraction"],
                                       brightness_policy=prefs["brightness_policy"], robustness=prefs.get("robustness", {}))
    if saved:
        if data["evaluator_identity"] != spec.evaluator_identity:
            raise ValueError("Evaluator changed since this campaign. Set RESUME=None to start a new campaign; previous results remain in the database.")
        data["controls"] = [Control(**c) for c in data["controls"]]
        spec = Spec(**data)
    loop = ExperimentLoop(database, spec, evaluator, campaign_id=saved["id"] if saved else None)
    if not saved:
        loop.grant(prefs["evaluations"], prefs["seconds"], prefs["planner_calls"], "initial user allocation")
    print(("Resumed" if saved else "Created"), "campaign", loop.id)
    print("Objective:", spec.objective)
    print("Search bounds:", loop.state["bounds"])
    print("Current criteria:", {k: loop.threshold(k) for k in loop.ADAPTIVE_THRESHOLDS})
    print("Remaining allowance:", loop.budget())
    return loop, evaluator


def show_table(rows, columns=None):
    from IPython.display import HTML, display
    rows = list(rows)
    if not rows:
        print("No records yet."); return
    columns = columns or list(rows[0])
    def text(value):
        if isinstance(value, float): return f"{value:.7g}"
        if isinstance(value, (dict, list)): return json.dumps(value)
        return str(value)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join("<tr>"+"".join(f"<td>{html.escape(text(r.get(c,'')))}</td>" for c in columns)+"</tr>" for r in rows)
    display(HTML('<div style="overflow:auto"><table style="text-align:left;border-collapse:collapse">'+
                 "<thead><tr>"+head+"</tr></thead><tbody>"+body+"</tbody></table></div>"))


def show_history(loop, limit=20):
    rows = []
    for row in loop.rows()[-limit:]:
        rows.append({"ID": row["id"], "controls": row["parameters"], "score": row["score"],
                     "resolution": row["resolution"], "brightness ratio": row["metrics"].get(loop.spec.brightness_metric),
                     "brightness": row["brightness_status"], "numerics": row["numerical_status"],
                     "criteria version": row["criteria_version"]})
    show_table(rows)


def show_event(event):
    from IPython.display import HTML, display
    key = next(iter(event))
    data = event[key]
    if key == "AI proposal":
        print(f"Codex → {data['action']} | allocation: {data['evaluations']} evaluations, {data['iterations']} iterations")
        print(data["rationale"])
        if data.get("knowledge_basis"): print("Knowledge / hypothesis:", data["knowledge_basis"])
    else:
        print(json.dumps(event, indent=2))
    display(HTML("<details><summary>Full decision / measurements</summary><pre>"+
                 html.escape(json.dumps(event, indent=2))+"</pre></details>"))


def plot_progress(loop, output_directory=None):
    import matplotlib.pyplot as plt
    rows = [r for r in loop.rows() if r["score"] is not None]
    if not rows:
        print("No objective values to plot yet."); return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for refined, marker, label in [(False, "o", "Main simulation"), (True, "x", "Finer simulation")]:
        group = [r for r in rows if (r["resolution"] != loop.spec.search) == refined]
        if group:
            axes[0].scatter([r["id"] for r in group], [r["score"] for r in group], marker=marker, label=label)
    axes[0].set_xlabel("Saved test number"); axes[0].set_ylabel("Score (lower is better)")
    axes[0].set_title("Scores of the tested launches"); axes[0].legend()
    control = loop.spec.controls[0]
    search = [r for r in rows if r["resolution"] == loop.spec.search]
    axes[1].scatter([r["parameters"][control.name] for r in search], [r["score"] for r in search],
                    color="#999999", s=16, alpha=.5, label="Tested launches")
    runs = loop.db.execute("SELECT id FROM optimizer_runs WHERE campaign=?", (loop.id,)).fetchall()
    for run in runs:
        steps = loop.db.execute("SELECT * FROM accepted_steps WHERE run=? ORDER BY iteration,id", (run["id"],)).fetchall()
        x = [json.loads(r["parameters"])[control.name] for r in steps]
        y = [r["score"] for r in steps]
        axes[1].plot(x, y, "o-", label=f"Accepted path {run['id']}")
        if steps:
            axes[1].scatter(x[0], y[0], marker="s", color="#366b9e", zorder=4)
            axes[1].scatter(x[-1], y[-1], marker="*", s=100, color="#cf7c32", zorder=4)
        for i, r in enumerate(steps):
            axes[1].annotate(str(r["iteration"]), (x[i], y[i]), xytext=(4, 5), textcoords="offset points", fontsize=8)
            if i:
                axes[1].annotate("", xy=(x[i],y[i]), xytext=(x[i-1],y[i-1]), arrowprops={"arrowstyle":"->", "color":"#555555", "lw":.8})
    axes[1].set_xlabel(f"{CONTROL_NAMES.get(control.name, control.name)} ({control.unit or 'dimensionless'})")
    axes[1].set_ylabel("Score in the main simulation")
    axes[1].set_title("Accepted improvements (step numbers shown)"); axes[1].legend()
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    if output_directory:
        target = Path(output_directory); target.mkdir(exist_ok=True, parents=True)
        fig.savefig(target/"optimization_progress.png", dpi=160)
    return fig


CONTROL_NAMES = {"power_kW": "Input peak power", "P2_kW": "Mode 2 power", "P3_kW": "Mode 3 power",
                 "phi2_rad": "Mode 2 phase", "phi3_rad": "Mode 3 phase", "phase": "Phase", "x": "x"}
THRESHOLD_NAMES = {"brightness_fraction": "Brightness target", "objective_tolerance": "Allowed score difference",
                   "gradient_atol": "Absolute slope agreement", "gradient_rtol": "Relative slope agreement",
                   "gradient_tolerance": "Small-slope stopping threshold"}


def _percent(value):
    return "not set" if value is None else f"{100*value:.3g}%"


def _controls_text(loop, values):
    parts = []
    for c in loop.spec.controls:
        if c.name not in values:
            continue
        value, unit = values[c.name], c.unit
        if unit == "rad":
            value, unit = math.degrees(value), "°"
        parts.append(f"{CONTROL_NAMES.get(c.name, c.name)}: {value:.5g}{' ' if unit else ''}{unit}")
    return "; ".join(parts)


def show_lab_event(event):
    """Show the scientific action and outcome without exposing nested JSON."""
    if "retry_in_seconds" in event:
        print(f"Codex is busy. Retrying in {event['retry_in_seconds']:g} seconds "
              f"({event['ai_calls_left']} AI call attempts remain).", flush=True)
        return
    if "AI proposal" in event:
        d = event["AI proposal"]
        names = {"evaluate": "Test a launch", "descend": "Try gradient descent",
                 "validate_candidate": "Check a promising result", "validate_gradient": "Check the slopes",
                 "expand_range": "Test a wider search range", "robustness": "Test small input errors",
                 "update_threshold": "Revise a threshold", "pause": "Pause for now"}
        print(f"Codex: {names.get(d['action'], d['action'])}")
        if d["action"] not in ("pause", "update_threshold"):
            print(f"Allowance: up to {d['evaluations']} new tests and {d['iterations']} steps or samples.")
        print("Reason:", d["rationale"])
        if d.get("knowledge_basis"):
            print("Idea being tested:", d["knowledge_basis"])
        if d.get("evidence_ids"):
            print("Uses saved tests:", ", ".join(map(str, d["evidence_ids"])))
        if d.get("expected_outcome"):
            print("Prediction:", d["expected_outcome"])
        return
    result = event.get("measured result", event)
    print("Outcome:", result.get("status", "saved"))
    if "score" in result:
        print(f"Measured score: {result['score']:.7g}")
    if "new_evaluations" in result:
        print("New tests used:", result["new_evaluations"])
    if "iteration" in result:
        print("Accepted steps on this path:", result["iteration"])
    if "check" in result:
        print("Slope check:", result["check"].get("status", "not run"))
    for label in ("error", "reason"):
        if result.get(label):
            print(label.capitalize()+":", result[label])


class NotebookStudy:
    """Readable lab commands around the unchanged ExperimentLoop."""
    def __init__(self, experiment, evaluator, planner=None):
        self.experiment, self.evaluator, self.planner = experiment, evaluator, planner
        self.figure_folder = experiment.path.parent/"figures"/experiment.id

    def _refresh_saved_state(self):
        """An idle notebook should use the latest state saved by this study."""
        loop = self.experiment
        if loop.action_id is not None or loop._clock_start is not None:
            return  # Keep the current action's working state while it is executing.
        row = loop.db.execute("SELECT state FROM campaigns WHERE id=?", (loop.id,)).fetchone()
        if row is None:
            raise RuntimeError("This study is no longer in the history database.")
        loop.state = json.loads(row["state"])

    def _activity_ids(self):
        loop = self.experiment
        return {name: loop.db.execute(f"SELECT MAX(id) FROM {name} WHERE campaign=?", (loop.id,)).fetchone()[0]
                for name in ("actions", "proposals")}

    def _remember_stop(self, requested, error=None):
        """Save the batch's stopping reason separately from its last action's outcome."""
        loop, completed = self.experiment, len(self.last_results)
        status = loop.state["status"]
        last = self.last_results[-1] if self.last_results else {}
        if error is not None:
            reason = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error"
            explanation = "The cell was interrupted; progress is saved." if reason == "interrupted" else str(error)
        elif last.get("status") == "planner checkpoint":
            reason = "planner_pause"
            explanation = "Codex requested a pause. " + last.get("reason", "")
        elif completed >= requested:
            reason = "batch_complete"
            noun = "decision" if requested == 1 else "decisions"
            explanation = f"Completed the {requested} AI {noun} requested for this batch."
        elif status == "budget reached":
            reason = "budget_reached"
            budget = loop.budget()
            exhausted = [label for key, label in (("evaluations_left", "simulation tests"),
                         ("seconds_left", "time"), ("planner_calls_left", "AI call attempts")) if budget[key] <= 0]
            explanation = "Used up the allowance for " + ", ".join(exhausted or ["this action"]) + "."
        elif status == "planner error":
            reason = "planner_error"
            row = loop.db.execute("SELECT error FROM proposals WHERE campaign=? ORDER BY id DESC LIMIT 1",
                                  (loop.id,)).fetchone()
            explanation = "The AI call failed. " + ((row["error"] or "") if row else "")
        else:
            reason, explanation = status, f"The loop stopped with status: {status}."
        loop.state["last_notebook_batch"] = {
            "reason": reason, "explanation": explanation.strip(), "requested_decisions": requested,
            "completed_decisions": completed, "activity_ids": self._activity_ids(),
        }
        loop._save()

    def remaining(self):
        self._refresh_saved_state()
        loop = self.experiment
        saved = loop.state.get("last_notebook_batch")
        if saved and saved.get("activity_ids") == self._activity_ids():
            print("Why the last batch stopped:", saved["explanation"])
            if saved["reason"] == "batch_complete":
                print("That is the batch limit; it does not establish a minimum.")
        elif loop.state["status"] == "planner error":
            row = loop.db.execute("SELECT error FROM proposals WHERE campaign=? ORDER BY id DESC LIMIT 1",
                                  (loop.id,)).fetchone()
            print("Why it stopped: The AI call failed.", (row["error"] or "") if row else "")
        elif loop.state["status"] in ("interrupted", "waiting for Codex"):
            print("Status:", loop.state["status"]+". Progress is saved.")
        else:
            print("Status:", "Ready." if loop.state["status"] == "ready" else
                  "No batch stopping reason was recorded for the latest work.")
        action = loop.db.execute("SELECT status,result FROM actions WHERE campaign=? ORDER BY id DESC LIMIT 1",
                                 (loop.id,)).fetchone()
        if action:
            names = {"candidate checked": "Finished the finer-step score check.",
                     "gradient checked": "Finished checking the numerical slopes.",
                     "iteration allocation reached": "Used the gradient-descent steps allocated for that action.",
                     "locally converged": "The checked local slopes were small.",
                     "accuracy limited": "Numerical accuracy limited confidence in the slopes.",
                     "no acceptable step": "No tried change met the acceptance rules.",
                     "evaluated": "Measured the requested launch.",
                     "planner checkpoint": "Codex requested a pause."}
            result = json.loads(action["result"] or "{}")
            print("Last action:", names.get(action["status"], action["status"])+
                  (" " + result["reason"] if action["status"] == "budget reached" and result.get("reason") else ""))
        b = loop.budget()
        print(f"Remaining: {b['evaluations_left']} new tests, {b['seconds_left']/60:.1f} minutes, "
              f"{b['planner_calls_left']} AI call attempts.")
        if all(b[k] > 0 for k in ("evaluations_left", "seconds_left", "planner_calls_left")):
            if loop.state["status"] == "planner error":
                print("Check the reported AI error before continuing.")
            else:
                print("To continue: study.run(decisions=3).")
        else:
            print("To continue, add the exhausted allowance with study.allow_more(...).")

    def prepare(self):
        self._refresh_saved_state()
        if self.planner is None:
            from codex_planner import CodexPlanner
            self.planner = CodexPlanner()
        login = self.planner.status()
        if not login["ready"]:
            raise RuntimeError("Please run 'codex login' in the VS Code terminal, then rerun this cell.")
        print("Codex is ready.")
        if not self.experiment.rows():
            result = self.experiment.execute(Decision("evaluate", self.experiment.spec.baseline,
                evaluations=1, rationale="Measure the starting pulse."))
            show_lab_event({"measured result": result})
        else:
            print("Using saved measurements; the starting test is not repeated.")
        self.remaining()

    def run(self, decisions=3):
        from planner_runner import run_with_capacity_retries
        if type(decisions) is not int or decisions < 0:
            raise ValueError("The number of decisions must be a nonnegative whole number.")
        if decisions == 0:
            self.remaining()
            return
        self._refresh_saved_state()
        if self.planner is None:
            self.prepare()
        self.last_results = []

        def report(event):
            if "measured result" in event:
                self.last_results.append(event["measured result"])
            show_lab_event(event)

        try:
            run_with_capacity_retries(self.experiment, self.planner,
                                      max_decisions=decisions, on_event=report)
        except BaseException as exc:
            self._remember_stop(decisions, exc)
            raise
        self._remember_stop(decisions)
        self.remaining()

    def results(self, recent=20):
        self._refresh_saved_state()
        loop = self.experiment
        label = "D_IO" if "D_IO" in loop.spec.objective else "D_NL" if "D_NL" in loop.spec.objective else "Loss"
        rows = []
        for r in loop.rows()[-recent:]:
            resolution = r["resolution"]
            grid = []
            if "Nt" in resolution:
                grid.append(f"{resolution['Nt']:,} time points")
            if "dz_m" in resolution:
                grid.append(f"{1000*resolution['dz_m']:g} mm steps")
            status = {"validated": "Meets current checks", "provisional": "Needs checking", "invalid": "Invalid calculation"}
            rows.append({"Test": r["id"], "Launch": _controls_text(loop, r["parameters"]), label: r["score"],
                         "LP01 photons / reference" if label != "Loss" else "Useful output / reference":
                             _percent(r["metrics"].get(loop.spec.brightness_metric)),
                         "Brightness target": r["brightness_status"], "Accuracy": status[r["numerical_status"]],
                         "Simulation detail": "; ".join(grid) or "configured resolution"})
        show_table(rows)
        print("Lower scores are better. 'Needs checking' results are kept for follow-up.")
        plot_progress(loop, self.figure_folder)

    def changes(self):
        self._refresh_saved_state()
        loop = self.experiment
        revisions = loop.db.execute("SELECT version,name,old_value,new_value,reason FROM criteria_revisions WHERE campaign=? ORDER BY id", (loop.id,)).fetchall()
        if revisions:
            rows = []
            for r in revisions:
                old, new = json.loads(r["old_value"]), json.loads(r["new_value"])
                rows.append({"Version": r["version"], "Threshold": THRESHOLD_NAMES.get(r["name"], r["name"]),
                             "Before": _percent(old) if r["name"] == "brightness_fraction" else old,
                             "After": _percent(new) if r["name"] == "brightness_fraction" else new,
                             "Reason": r["reason"]})
            show_table(rows)
        else:
            print("No thresholds have changed yet.")
        bounds = loop.state["range_changes"]
        if bounds:
            show_table([{"Control": CONTROL_NAMES.get(r["control"], r["control"]),
                         "Before": f"{r['old'][0]:g} to {r['old'][1]:g}",
                         "After": f"{r['new'][0]:g} to {r['new'][1]:g}",
                         "Unit": next(c.unit for c in loop.spec.controls if c.name == r["control"])} for r in bounds])
        else:
            print("The search ranges have not expanded yet.")

    def pulses(self):
        from evaluators import plot_optical_output
        self._refresh_saved_state()
        rows = [r for r in self.experiment.rows() if r["status"] == "complete" and r["artifacts"].get("fields")]
        if not rows:
            print("There are no saved optical fields to plot yet.")
            return
        search = [r for r in rows if r["resolution"] == self.experiment.spec.search]
        selected = {rows[0]["id"]: rows[0]}
        if search:
            best = min(search, key=lambda r: r["score"])
            selected[best["id"]] = best
        for r in selected.values():
            print(f"Test {r['id']}: {_controls_text(self.experiment, r['parameters'])}")
            print(f"Numerics: {r['numerical_status']}; brightness: {r['brightness_status']}; "
                  f"threshold version: {r['criteria_version']}")
            plot_optical_output(r, self.figure_folder)

    def allow_more(self, tests=40, minutes=30, ai_calls=5):
        self._refresh_saved_state()
        self.experiment.grant(tests, minutes*60, ai_calls, reason="User continued the study")
        print(f"Added {tests} tests, {minutes:g} minutes and {ai_calls} Codex decisions.")
        self.remaining()

    def finish(self):
        if hasattr(self.evaluator, "close"):
            self.evaluator.close()
        print("Results are saved. You can continue this study later.")


def start_study(example="dio_power", *, brightness_percent=50, max_tests=80,
                fresh=False, max_minutes=60, max_ai_calls=10, brightness_required=False,
                power_error_kW=None, phase_error_degrees=None, workspace=None, database=None):
    """Open a study without input prompts, JSON, database paths or manual run IDs."""
    workspace = Path(workspace).resolve() if workspace else Path(__file__).resolve().parents[2]
    database = Path(database) if database else workspace/"output/ai_augmented_loop/experiments.sqlite"
    if brightness_percent is not None and (not isinstance(brightness_percent, (float, int)) or
                                           not math.isfinite(brightness_percent) or brightness_percent < 0):
        raise ValueError("Brightness target must be a nonnegative percentage, or None.")
    if type(max_tests) is not int or type(max_ai_calls) is not int or max_tests < 0 or max_ai_calls < 0:
        raise ValueError("Test and Codex allowances must be nonnegative whole numbers.")
    if not math.isfinite(max_minutes) or max_minutes < 0:
        raise ValueError("Minutes must be finite and nonnegative.")
    robustness = {}
    for value in (power_error_kW, phase_error_degrees):
        if value is not None and (not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
            raise ValueError("Input uncertainties must be nonnegative numbers.")
    if power_error_kW is not None:
        if example == "dio_power":
            robustness["power_kW"] = float(power_error_kW)
        elif example.startswith("lp01_auxiliary"):
            robustness.update(P2_kW=float(power_error_kW), P3_kW=float(power_error_kW))
        else:
            raise ValueError("This example has no power control measured in kW.")
    if phase_error_degrees is not None:
        if not example.startswith("lp01_auxiliary"):
            raise ValueError("This example has no optical phase controls.")
        robustness.update(phi2_rad=math.radians(phase_error_degrees), phi3_rad=math.radians(phase_error_degrees))
    preferences = {"brightness_fraction": None if brightness_percent is None else brightness_percent/100,
                   "brightness_policy": "hard" if brightness_required else "soft",
                   "evaluations": max_tests, "seconds": max_minutes*60, "planner_calls": max_ai_calls,
                   "robustness": robustness}
    # The original API stays available for detailed programmatic use.
    with redirect_stdout(io.StringIO()) as captured:
        loop, evaluator = open_example(database, workspace, example, resume=None if fresh else "latest", preferences=preferences)
    resumed = captured.getvalue().startswith("Resumed")
    study = NotebookStudy(loop, evaluator)
    print(("Continuing saved study" if resumed else "Starting a new study")+f": {example}")
    for c in loop.spec.controls:
        lo, hi = loop.state["bounds"][c.name]
        unit = c.unit
        if unit == "rad":
            lo, hi, unit = math.degrees(lo), math.degrees(hi), "degrees"
        print(f"{CONTROL_NAMES.get(c.name,c.name)}: {lo:g} to {hi:g} {unit}".rstrip())
    fraction = loop.threshold("brightness_fraction")
    policy = "required minimum" if loop.spec.brightness_policy == "hard" else "preferred target; lower results are kept"
    print(f"Brightness: {_percent(fraction)} of the reference ({policy}).")
    if resumed:
        print("Saved settings and remaining allowances are kept. Use fresh=True for new starting settings.")
    study.remaining()
    print("History saved automatically in:", loop.path)
    return study
