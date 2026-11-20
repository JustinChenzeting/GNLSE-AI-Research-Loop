"""Notebook-facing helpers for bounded LLM-guided D_IO and D_NL searches."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

from .runtime import (
    GNLSE, GPPrior, NAMES, PHASE_NAMES, OBJECTIVES, planning_schemas,
    check_point, same_point, connect_llm, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
EQUAL = dict(zip(NAMES, [100/3]*3))
BRIGHTNESS_FLOOR = 0.50
SEARCH_DZ, CHECK_DZ = 0.005, 0.001
D_IO_TOLERANCE = 2e-4
SCORE_TOLERANCE = D_IO_TOLERANCE


def problem_card(power_kW, *, phases=False, objective="D_IO"):
    if objective not in OBJECTIVES:
        raise ValueError("Choose D_IO or D_NL.")
    if power_kW not in (100.0, 81.943):
        raise ValueError("Choose 100.0 or 81.943 kW from the reference notebook.")
    problem = {
        "score_name": objective,
        "objective_modes": OBJECTIVES[objective]["modes"],
        "brightness_modes": OBJECTIVES[objective]["modes"],
        "task": "Minimize LP01 D_IO by choosing three nonnegative real launch amplitudes.",
        "power_kW": power_kW,
        "parameters": {k: {"bounds": [0, 100], "unit": "relative amplitude"} for k in NAMES},
        "constraint": "c1+c2+c3=100; modal power fractions = c_i^2/sum(c_j^2).",
        "modes": ["LP01", "LP11-cos", "LP11-sin"],
        "launch": "200 fs intensity-FWHM Gaussian; zero relative phases; fixed total peak power.",
        "fiber": "Silica step-index, core radius 52.5 um, NA=0.1, length=2.86 m, carrier=1379.2 nm; "
                 "3 scalar LP modes, beta order 5, package Raman and self-steepening defaults.",
        "objective": "D_IO = 1 - squared normalized complex overlap of input and nonlinear output LP01, "
                     "same time samples, no alignment or spectral filter. Lower is better; range [0,1]. "
                     "Undefined LP01 overlap is marked invalid and assigned a diagnostic penalty 1.",
        "brightness": "Hard minimum: LP01 photons / equal-amplitude 1mm reference photons >= 0.50. "
                      "Invalid or too-dim points are recorded but cannot be selected as best.",
        "numerics": {"Nt": 2048, "time_window_ps": 10, "search_dz_m": SEARCH_DZ,
                     "check_dz_m": CHECK_DZ, "D_IO_step_tolerance": D_IO_TOLERANCE,
                     "cp_target_relative_error": 1e-3, "cp_seed": int(0x4d4d474e)},
        "initial_reference": EQUAL,
        "prior_context": "The two LP11 orientations are degenerate in an ideal circular fiber. "
                         "Exchange symmetry is a plausible prior assumption, not an imposed search constraint. "
                         "The simplex has two independent dimensions. No historical optima are supplied.",
    }
    if phases:
        import math
        problem.update(
            control_space="phase_amplitude",
            task="Minimize LP01 D_IO by choosing three launch amplitudes and two relative phases.",
            parameters={**problem["parameters"], **{
                k: {"bounds": [-math.pi, math.pi], "unit": "radian (periodic)"}
                for k in PHASE_NAMES}},
            constraint="c1+c2+c3=100; a_m=c_m*exp(i*phi_m)/sqrt(sum(c_j^2)); "
                       "phi1=0. Modal power fractions = c_i^2/sum(c_j^2). "
                       "Phases wrap into [-pi,pi); absent-mode phases are canonicalized to zero.",
            launch="200 fs intensity-FWHM Gaussian; two variable relative phases; fixed total peak power.",
            initial_reference={**EQUAL, **dict.fromkeys(PHASE_NAMES, 0.0)},
            prior_context="Five stored coordinates describe four independent physical controls: "
                          "two on the amplitude simplex and two periodic phases. The GP embeds "
                          "amplitude fractions (or power fractions) as "
                          "[w1, w2*cos(phi2), w2*sin(phi2), w3*cos(phi3), w3*sin(phi3)]. "
                          "This is periodic and ignores absent-mode phase. One length scale applies "
                          "to this embedding. LP11 exchange, if enabled in the prior, swaps BOTH "
                          "amplitude and phase together; it is a physical assumption, not a search "
                          "constraint. No historical optima are supplied.",
        )
    if objective == "D_NL":
        problem["task"] = problem["task"].replace("LP01 D_IO", "all-three-mode D_NL")
        problem["objective"] = (
            "D_NL = 1 - |sum_m integral(conj(A_linear,m)*A_nonlinear,m) dt|^2 / "
            "(sum_m integral |A_linear,m|^2 dt * sum_m integral |A_nonlinear,m|^2 dt), "
            "with all three modes in the numerator and both norm factors. Each candidate "
            "uses matched linear (n2=0) and nonlinear output fields from the same launch, "
            "fiber, time grid and propagation step. No time alignment or spectral filter. "
            "Lower is better; range [0,1]. Undefined overlap is invalid with diagnostic penalty 1."
        )
        problem["brightness"] = (
            "Hard minimum: all-mode nonlinear output photons / equal-amplitude, zero-phase "
            "1 mm nonlinear baseline photons at the same total power >= 0.50. LP01 photons "
            "are diagnostic only. Invalid or too-dim points cannot be selected as best."
        )
        problem["numerics"]["D_NL_step_tolerance"] = problem["numerics"].pop("D_IO_step_tolerance")
    return problem


def prepare_ai(problem, ask):
    phases = problem.get("control_space") == "phase_amplitude"
    objective = problem.get("score_name", "D_IO")
    _, schema, _ = planning_schemas(phases=phases, objective=objective)
    plan = ask({
        "operation": "prepare_prior_and_initial_point",
        "problem_card": problem,
        "instruction": "Choose a GP prior and one informative starting point in the declared control space, "
                       "different from initial_reference. Include both relative phases when they are controls. "
                       "Refine the prior using physical context if helpful. Provide a brief 1-3 sentence "
                       "rationale summarizing assumptions and the choice, not private deliberations. "
                       "The prior is only for initialization; later acquisition decisions use measurements."
    }, schema)
    prior = GPPrior(plan["prior"], phases=phases, objective=objective)
    plan["initial_point"] = check_point(plan["initial_point"], phases=phases)
    if same_point(plan["initial_point"], problem["initial_reference"], phases=phases):
        raise ValueError("AI's initial point repeats the equal-amplitude reference.")
    print("GP prior:", json.dumps(prior.spec, indent=2))
    print("AI explanation:", plan["rationale"])
    print("Initial point:", plan["initial_point"])
    return plan, prior


def observation(measurement, reference_photons, role):
    objective = measurement.get("objective", "D_IO")
    ratio = measurement["metrics"][OBJECTIVES[objective]["photons"]] / reference_photons
    return {
        "evaluation": measurement["evaluation"], "role": role,
        "point": measurement["point"], "power_kW": measurement["power_kW"],
        "objective": objective, objective: measurement[objective], "brightness_ratio": ratio,
        "objective_defined": measurement["objective_defined"], "valid": measurement["valid"],
        "feasible": bool(measurement["valid"] and ratio >= BRIGHTNESS_FLOOR),
        "resolution": {"Nt": measurement["Nt"], "dz_m": measurement["dz_m"]},
        "diagnostics": measurement["metrics"],
    }


def evaluate(point, simulator, power_kW, reference_photons, role="AI trial", *, phases=False):
    measured = simulator.measure(check_point(point, phases=phases), power_kW, SEARCH_DZ)
    row = observation(measured, reference_photons, role)
    metric = row["objective"]
    print(f"  Measured {metric}={row[metric]:.8f}; brightness={row['brightness_ratio']:.3f}; "
          f"feasible={row['feasible']}")
    return row


def choose_next(problem, history, calibration, remaining, ask, validation=None):
    phases = problem.get("control_space") == "phase_amplitude"
    metric = problem.get("score_name", "D_IO")
    _, _, schema = planning_schemas(phases=phases, objective=metric)
    feasible = [r for r in history if r["feasible"]]
    best = min(feasible, key=lambda r: r[metric]) if feasible else None
    scores = [r[metric] for r in feasible]
    target = max(0.0, min(scores) - 0.01*(max(scores)-min(scores))) if scores else None
    decision = ask({
        "operation": "choose_next_point" if remaining else "finish_search",
        "problem_card": problem,
        "reference_measurements": calibration,
        "observations": history,
        "latest_observation": history[-1],
        "best_feasible": best,
        f"target_{metric}": target,
        "remaining_trials": remaining,
        "final_step_check": validation,
        "instruction": "Act as the acquisition decision: balance learning about unexplored mixtures "
                       f"with improving feasible measured {metric}. Return exactly one new point or stop. "
                       "Do not repeat an observed point, change physics/bounds/brightness, or invent data. "
                       f"Use a short 1-3 sentence rationale tied to observations. Low {metric} below the brightness "
                       "floor is infeasible. An invalid score is not a measurement of distortion. "
                       "With zero trials remaining you MUST return action=stop and point=null. "
                       "Stopping on a small budget does not establish convergence or a global optimum."
    }, schema)
    if decision.get("action") not in ("evaluate", "stop") or not isinstance(decision.get("rationale"), str):
        raise ValueError("Malformed acquisition decision.")
    if decision["action"] == "stop":
        if decision["point"] is not None:
            raise ValueError("A stop decision must have point=null.")
    else:
        if remaining <= 0:
            raise ValueError("AI requested a point after the trial budget was exhausted.")
        decision["point"] = check_point(decision["point"], phases=phases)
        if any(same_point(decision["point"], r["point"], phases=phases) for r in history):
            raise ValueError("AI repeated an observed point; no extra solve was run.")
    print("AI decision:", decision["action"], decision["rationale"], flush=True)
    return decision


def run_search(problem, ask, simulator, run_dir, max_trials):
    if type(max_trials) is not int or max_trials < 1:
        raise ValueError("max_trials must be a positive integer.")
    history, calibration = [], []
    phases = problem.get("control_space") == "phase_amplitude"
    metric = problem.get("score_name", "D_IO")
    reference = problem["initial_reference"]
    state = {"status": "running", "problem": problem, "history": history}
    write_json(run_dir / "state.json", state)
    try:
        # 1. AI chooses the prior and its first point before seeing any run data.
        plan, prior = prepare_ai(problem, ask)
        write_json(run_dir / "initial_plan.json", plan)

        # 2. Establish the source notebook's baseline and brightness denominator.
        power = problem["power_kW"]
        baseline_fine = simulator.measure(reference, power, CHECK_DZ)
        reference_photons = baseline_fine["metrics"][OBJECTIVES[metric]["photons"]]
        if not baseline_fine["valid"] or reference_photons <= 0:
            raise ValueError("Invalid brightness reference.")
        baseline = evaluate(reference, simulator, power, reference_photons, "baseline", phases=phases)
        calibration = [baseline, observation(baseline_fine, reference_photons, "baseline fine")]
        history.append(baseline)
        state["calibration"] = calibration
        write_json(run_dir / "state.json", state)

        # 3–4. Exactly one measured trial, then a fresh acquisition decision.
        point = plan["initial_point"]
        stop_reason = "trial budget reached"
        for trial in range(1, max_trials + 1):
            print(f"\nTrial {trial}/{max_trials}", flush=True)
            history.append(evaluate(point, simulator, power, reference_photons, phases=phases))
            write_json(run_dir / "state.json", state)
            if trial == max_trials:
                break
            decision = choose_next(problem, history, calibration, max_trials-trial, ask)
            if decision["action"] == "stop":
                stop_reason = "AI requested stop: " + decision["rationale"]
                break
            point = decision["point"]

        best = min((r for r in history if r["feasible"]), key=lambda r: r[metric], default=None)
        validation = {"status": "not_run", "reason": "No feasible search point."}
        if best is not None:
            selected_fine = simulator.measure(best["point"], power, CHECK_DZ)
            fine = observation(selected_fine, reference_photons, "selected fine")
            checks = {
                "baseline_step_agreement": abs(baseline[metric]-baseline_fine[metric]) <= SCORE_TOLERANCE,
                "selected_step_agreement": abs(best[metric]-fine[metric]) <= SCORE_TOLERANCE,
                "selected_fine_valid": fine["valid"],
                "selected_fine_brightness": fine["feasible"],
            }
            validation = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
                          "selected_fine": fine, f"baseline_fine_{metric}": baseline_fine[metric],
                          "verified_improvement": bool(all(checks.values()) and
                              fine[metric] < baseline_fine[metric] - SCORE_TOLERANCE)}
        state.update(status="measured", stop_reason=stop_reason, best_search=best,
                     validation=validation, gnlse_attempts=simulator.attempts)
        write_json(run_dir / "state.json", state)
        # Send the final measurement and fine check too; remaining=0 forbids another solve.
        state["final_decision"] = choose_next(problem, history, calibration, 0, ask, validation)
        state["status"] = "completed"
        return state, prior
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state.update(gnlse_attempts=simulator.attempts,
                     linear_solves=getattr(simulator, "linear_solves", 0),
                     ai_attempts=len(getattr(ask, "calls", [])))
        write_json(run_dir / "state.json", state)


@dataclass
class Run:
    """A completed search, its initialization prior, and saved artifact directory."""
    directory: Path
    result: dict
    prior: GPPrior


def run_experiment(*, power_kW=100.0, max_trials=3, model=None,
                   thinking_effort=None, timeout_seconds=300, phases=False, objective="D_IO"):
    """Prepare AI, run the bounded search, and close the simulator on every exit."""
    if type(max_trials) is not int or max_trials < 1:
        raise ValueError("max_trials must be a positive integer.")
    problem = problem_card(power_kW, phases=phases, objective=objective)
    prefix = "llm_bo_phase_amplitude" if phases else "llm_bo"
    if objective == "D_NL":
        prefix = prefix.replace("llm_bo", "llm_bo_dnl")
    run_dir = ROOT / "output" / datetime.now(timezone.utc).strftime(f"%Y-%m-%d_{prefix}_%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    if objective == "D_NL":
        reference = "step_index_phase_amplitude_optimization" if phases else "step_index_dnl_three_coefficient_optimization"
    else:
        reference = "dio_phase_amplitude_optimization" if phases else "dio_three_coefficient_optimization"
    source = next((ROOT / "notebooks").rglob(reference + ".ipynb"))
    write_json(run_dir / "run.json", {
        "model": model, "thinking_effort": thinking_effort, "power_kW": power_kW,
        "max_trials": max_trials, "max_nonlinear_solves": max_trials+3,
        "max_linear_solves": max_trials+3 if objective == "D_NL" else 0,
        "objective": objective, "objective_modes": OBJECTIVES[objective]["modes"],
        "brightness_modes": OBJECTIVES[objective]["modes"],
        "max_ai_calls": max_trials+1, "per_call_timeout_seconds": timeout_seconds,
        "control_space": "phase_amplitude" if phases else "amplitude",
        "source_notebook": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    })
    ask = connect_llm(run_dir / "ai", max_calls=max_trials+1, model=model,
                      reasoning_effort=thinking_effort, timeout_seconds=timeout_seconds)
    print(f"Codex: {model}, thinking effort: {thinking_effort}")
    print("Saving this run to:", run_dir)
    with GNLSE(ROOT, run_dir / "gnlse", max_evaluations=max_trials+3,
               timeout_seconds=timeout_seconds, phases=phases, objective=objective) as simulator:
        result, prior = run_search(problem, ask, simulator, run_dir, max_trials)
    print("\nStop:", result["stop_reason"])
    print("1 mm checks:", result["validation"]["status"])
    linear_note = f", {result['linear_solves']} matched linear solves" if objective == "D_NL" else ""
    print(f"Used {result['gnlse_attempts']} nonlinear solves{linear_note} and {result['ai_attempts']} AI calls.")
    return Run(run_dir, result, prior)


def show_results(run):
    """Display and save the measurement table and plots using saved fields only."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    result, run_dir = run.result, run.directory
    power_kW = result["problem"]["power_kW"]
    metric = result["problem"].get("score_name", "D_IO")
    mode_label = OBJECTIVES[metric]["label"]
    modes = np.array(OBJECTIVES[metric]["modes"])-1
    table = pd.DataFrame([{
        "role": row["role"], **row["point"], metric: row[metric],
        "brightness": row["brightness_ratio"], "feasible": row["feasible"]
    } for row in result["history"]])
    display(table)
    table.to_csv(run_dir / "history.csv", index=False)
    display(pd.DataFrame([result["validation"].get("checks", {})]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot(table.index, table[metric], "o-", label="Measured (5 mm)")
    feasible_scores = table[metric].where(table.feasible, np.inf).cummin().replace(np.inf, np.nan)
    axes[0].plot(table.index, feasible_scores, "--", label="Best feasible")
    axes[0].set(xlabel="Observation (0 = baseline)", ylabel=f"{mode_label} {metric}")
    axes[0].legend()

    if result["best_search"] is not None:
        fine = result["validation"]["selected_fine"]
        directory = run_dir / "gnlse" / f"evaluation_{fine['evaluation']:03d}"
        with np.load(directory / "fields.npz") as data:
            t, wl = data["time_ps"], data["wavelength_nm"]
            good = np.flatnonzero(np.isfinite(wl) & (wl > 0))
            good = good[np.argsort(wl[good])]
            branches = [("Input", "input"), ("Nonlinear output", "nonlinear")]
            if metric == "D_NL":
                branches.insert(1, ("Matched linear output", "linear"))
            for label, key in branches:
                field = data[key][:, modes, :]
                axes[1].plot(t, (abs(field)**2).sum(axis=(1, 2))/1000, label=label)
                spectrum = (abs(np.fft.fftshift(np.fft.ifft(field, axis=0), axes=0))**2).sum(axis=(1, 2))
                axes[2].plot(wl[good], spectrum[good], label=label)
        axes[1].set(xlabel="Time (ps)", ylabel=f"{mode_label} power (kW)", xlim=(-1, 1))
        axes[2].set(xlabel="Wavelength (nm)", ylabel="Power per frequency bin (W)", xlim=(1200, 1600))
        axes[1].legend(); axes[2].legend()
        fig.suptitle(f"{power_kW:g} kW | selected 1 mm {metric}={fine[metric]:.6f} | "
                     f"brightness={fine['brightness_ratio']:.3f} | checks: {result['validation']['status']}")
    fig.savefig(run_dir / "results.png", dpi=150)
    plt.show()
    if result["problem"].get("control_space") == "phase_amplitude":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        amplitudes = table[list(NAMES)].to_numpy()
        fractions = amplitudes**2 / (amplitudes**2).sum(axis=1, keepdims=True)
        styles = (dict(marker="o", linestyle="-"),
                  dict(marker="s", linestyle="-", markerfacecolor="none", markersize=8),
                  dict(marker="^", linestyle="--", markersize=4))
        for i, name in enumerate(NAMES):
            axes[0].plot(table.index, amplitudes[:, i], label=name, **styles[i])
            axes[1].plot(table.index, fractions[:, i], label=("LP01", "LP11-cos", "LP11-sin")[i], **styles[i])
        for i, name in enumerate(PHASE_NAMES):
            # Separate trial settings, not a continuous path across the periodic boundary.
            axes[2].plot(table.index, np.rad2deg(table[name]), linestyle="none",
                         marker=("o", "x")[i], markerfacecolor="none", markersize=7,
                         label=rf"$\phi_{i+2}$")
        for ax, ylabel in zip(axes, ("Launch amplitude coefficient", "Input power fraction",
                                    "Relative phase (degrees)")):
            ax.set(xlabel="Observation (0 = baseline)", ylabel=ylabel)
            ax.set_xticks(table.index)
            ax.legend()
        axes[2].set_ylim(-185, 185)
        fig.savefig(run_dir / "controls.png", dpi=150)
        plt.show()
    print("Artifacts:", run_dir)
