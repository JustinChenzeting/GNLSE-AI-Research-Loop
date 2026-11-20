"""Compare saved phase/amplitude results without running extra propagations."""
from pathlib import Path
import hashlib
import json
import math
import tomllib

from .notebook import ROOT, BRIGHTNESS_FLOOR, SEARCH_DZ, CHECK_DZ, D_IO_TOLERANCE
from .runtime import NAMES, PHASE_NAMES, OBJECTIVES, check_point, write_json


REFERENCE_NOTEBOOK = next((ROOT / "notebooks").rglob("dio_phase_amplitude_optimization.ipynb"))
DNL_REFERENCE_NOTEBOOK = next((ROOT / "notebooks").rglob("step_index_phase_amplitude_optimization.ipynb"))


def previous_result(power_kW, reference_notebook=None, *, previous_directory=None, objective="D_IO"):
    """Read the exact saved run named by the reference notebook's metadata."""
    import pandas as pd

    if objective not in OBJECTIVES:
        raise ValueError("Unknown objective.")
    is_dnl = objective == "D_NL"
    source = Path(reference_notebook or (DNL_REFERENCE_NOTEBOOK if is_dnl else REFERENCE_NOTEBOOK)).resolve()
    notebook = json.loads(source.read_text())
    location = previous_directory or notebook.get("metadata", {}).get("execution_artifacts" if is_dnl else "dio_run_directory")
    if not location:
        raise ValueError("Reference notebook has no saved run. Supply previous_directory explicitly.")
    directory = Path(location).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve()
    case = directory / f"{power_kW:.3f}kw".replace(".", "p")
    settings_path, summary_path = directory / "settings.toml", case / "summary.toml"
    for path in (settings_path, summary_path, case / "comparisons.csv", case / "evaluations.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing historical artifact: {path}")
    settings = tomllib.loads(settings_path.read_text())
    summary = tomllib.loads(summary_path.read_text())
    comparisons = pd.read_csv(case / "comparisons.csv")
    baseline = comparisons.loc[comparisons.launch == "equal_zero_phase_baseline"].iloc[0]
    selected = comparisons.loc[comparisons.launch == "phase_amplitude_selected"].iloc[0]
    evaluations = pd.read_csv(case / "evaluations.csv")
    reference_photons = float(evaluations.loc[evaluations.id == baseline.evaluation_id,
                                            "photons" if is_dnl else "photons_lp01"].iloc[0])
    # Normalize the historical table's spelling without changing saved source files.
    if is_dnl:
        baseline = baseline.rename({"all_mode_DNL": "D_NL"})
        selected = selected.rename({"all_mode_DNL": "D_NL"})
        summary = {k.replace("DNL", "D_NL"): v for k, v in summary.items()}
        summary["essential_checks_passed"] = summary["numerical_checks_passed"]
    point = check_point({
        **dict(zip(NAMES, summary["selected_coefficients"])),
        **dict(zip(PHASE_NAMES, summary["selected_phases_rad"][1:])),
    }, phases=True)
    if (not math.isclose(summary["power_W"], power_kW*1000, abs_tol=1e-8)
            or not math.isclose(float(selected[objective]), summary[f"selected_{objective}_fine"], abs_tol=1e-12)
            or not math.isclose(float(baseline[objective]), summary[f"baseline_{objective}_fine"], abs_tol=1e-12)):
        raise ValueError("Historical summary and comparison records disagree.")
    return dict(source=source, directory=directory, case=case, settings=settings, summary=summary,
                baseline=baseline, selected=selected, point=point, reference_photons=reference_photons)


def comparison_data(run, reference_notebook=None, *, previous_directory=None):
    """Return comparison rows and explicit checks of the shared numerical setup."""
    result = run.result
    metric = result["problem"].get("score_name", "D_IO")
    spec = OBJECTIVES[metric]
    if result["problem"].get("control_space") != "phase_amplitude":
        raise ValueError("Use a phase-and-amplitude run for this comparison.")
    fine = result.get("validation", {}).get("selected_fine")
    if fine is None:
        raise ValueError("The current run has no selected 1 mm result to compare.")
    power = result["problem"]["power_kW"]
    old = previous_result(power, reference_notebook, previous_directory=previous_directory, objective=metric)
    settings, summary = old["settings"], old["summary"]
    baseline = next(r for r in result["calibration"] if r["role"] == "baseline fine")
    identity = json.loads((run.directory / "gnlse/evaluator.json").read_text())
    physical_keys = ("time_window_ps", "pulse_fwhm_ps", "fiber_length_m", "carrier_m",
                     "core_radius_m", "NA", "num_modes", "cp_target_error", "cp_seed")
    checks = {
        "fixed_power": all(math.isclose(float(v), power*1000, rel_tol=1e-12) for v in
                           (summary["power_W"], old["baseline"].power_W, old["selected"].power_W)),
        "fiber_pulse_and_CP_settings": all(math.isclose(identity["setup"][k], settings[k], rel_tol=1e-12)
                                          for k in physical_keys),
        "objective_and_propagated_modes": (settings["objective_modes"] == spec["modes"]
                                           and settings.get("propagated_modes", [1, 2, 3]) == [1, 2, 3]
                                           and settings["objective_filter"] == "none"
                                           and identity["objective"] == metric),
        "amplitudes_and_periodic_phases": (settings["coefficient_bounds"] == [0., 100.]
                                          and settings["start_phases_rad"] == [0., 0., 0.]
                                          and settings["phase_range_rad"] == [-math.pi, math.pi]),
        "same_time_and_step_grids": (settings["Nt"] == fine["resolution"]["Nt"] == 2048
                                     and settings["search_dz_m"] == SEARCH_DZ
                                     and settings["validation_dz_m"] == CHECK_DZ
                                     and fine["resolution"]["dz_m"] == CHECK_DZ
                                     and old["baseline"].dz_m == old["selected"].dz_m == CHECK_DZ
                                     and old["baseline"].nt == old["selected"].nt == 2048),
        "same_brightness_floor": settings["minimum_brightness_fraction"] == BRIGHTNESS_FLOOR,
        "same_brightness_modes": settings["brightness_definition"].startswith(
            "all-mode" if metric == "D_NL" else "LP01"),
        "same_package_manifest": identity["manifest_sha256"] == settings["manifest_sha256"],
        f"baseline_{metric}_agreement": abs(baseline[metric]-summary[f"baseline_{metric}_fine"]) <= D_IO_TOLERANCE,
        "baseline_photon_agreement": math.isclose(baseline["diagnostics"][spec["photons"]],
                                                  old["reference_photons"], rel_tol=1e-4),
    }
    def row(label, point, score, brightness, optimization, validation, stop, valid):
        c = [point[k] for k in NAMES]
        return {
            "method": label, "power_kW": power, f"{metric}_1mm": float(score),
            "brightness_ratio": float(brightness), **point,
            **{f"P{i+1}_fraction": v*v/sum(x*x for x in c) for i, v in enumerate(c)},
            "optimization_solves": optimization, "fine_check_solves": validation,
            "total_nonlinear_solves": optimization + validation,
            **({"total_linear_solves": optimization + validation} if metric == "D_NL" else {}),
            "checks_passed": bool(valid), "stop_reason": stop,
        }
    current_measurements = [json.loads(p.read_text()) for p in
                            sorted((run.directory / "gnlse").glob("evaluation_*/measurement.json"))]
    current_search_solves = sum(m["dz_m"] == SEARCH_DZ for m in current_measurements)
    current_check_solves = sum(m["dz_m"] == CHECK_DZ for m in current_measurements)
    rows = [
        row("Equal amplitudes, zero phases", baseline["point"], baseline[metric], 1., 0, 0,
            "reference (cost included in each search)", baseline["valid"]),
        row("Previous gradient descent", old["point"], summary[f"selected_{metric}_fine"],
            summary["brightness_ratio"], summary["new_optimization_evaluations"],
            summary["new_check_evaluations"], summary["stop_reason"], summary["essential_checks_passed"]),
        row("LLM phase + amplitude", fine["point"], fine[metric], fine["brightness_ratio"],
            current_search_solves, current_check_solves, result["stop_reason"],
            result["validation"]["status"] == "pass"),
    ]
    difference = float(fine[metric] - summary[f"selected_{metric}_fine"])
    matched = bool(all(checks.values()))
    feasible = bool(fine["feasible"] and summary["brightness_ratio"] >= BRIGHTNESS_FLOOR)
    validated = result["validation"]["status"] == "pass" and summary["essential_checks_passed"]
    if not (matched and feasible):
        conclusion = "Results are shown descriptively; setup or feasibility checks prevent a verified ranking."
    elif abs(difference) <= D_IO_TOLERANCE:
        conclusion = f"The selected {metric} values differ by no more than the 0.0002 step-size tolerance."
    elif difference < 0:
        conclusion = f"The LLM-selected launch has lower 1 mm {metric} than the saved gradient-descent launch."
    else:
        conclusion = f"The saved gradient-descent launch has lower 1 mm {metric} than the LLM-selected launch."
    if not validated:
        conclusion += " This is a comparison on the saved grids; failed validation checks prevent a fully validated ranking."
    report = {
        "power_kW": power, "objective": metric, "objective_modes": spec["modes"],
        "brightness_modes": spec["modes"], "checks": {k: bool(v) for k, v in checks.items()},
        "matched_setup": matched, f"LLM_minus_previous_{metric}": difference,
        f"relative_{metric}_reduction_vs_previous": -difference/summary[f"selected_{metric}_fine"],
        "conclusion": conclusion,
        "previous_validation_checks": summary["checks"],
        "previous_validation_passed": bool(summary["essential_checks_passed"]),
        "current_step_checks_passed": result["validation"]["status"] == "pass",
        "interpretation": "Unequal evaluation budgets and one run per method; this is a selected-launch "
                          "comparison, not an equal-budget optimizer benchmark or evidence of a global optimum.",
        "current_run": str(run.directory), "previous_run": str(old["directory"]),
        "reference_notebook": str(old["source"]),
        "reference_notebook_sha256": hashlib.sha256(old["source"].read_bytes()).hexdigest(),
        "previous_summary_sha256": hashlib.sha256((old["case"] / "summary.toml").read_bytes()).hexdigest(),
    }
    return rows, report, old


def compare_to_previous(run, reference_notebook=None, *, previous_directory=None):
    """Display fine-grid scores, controls, budgets, outputs and the selected input."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import display

    rows, report, old = comparison_data(run, reference_notebook, previous_directory=previous_directory)
    metric = report["objective"]
    mode_label = OBJECTIVES[metric]["label"]
    modes = np.array(OBJECTIVES[metric]["modes"])-1
    table = pd.DataFrame(rows)
    target = run.directory / "comparison_previous_phase_amplitude"
    target.mkdir(exist_ok=True)
    table.to_csv(target / "results.csv", index=False)
    write_json(target / "comparison.json", report)
    display(table[["method", f"{metric}_1mm", "brightness_ratio", "total_nonlinear_solves",
                   *(["total_linear_solves"] if metric == "D_NL" else []), "checks_passed", "stop_reason"]])
    display(table[["method", *NAMES, *PHASE_NAMES, "P1_fraction", "P2_fraction", "P3_fraction"]])
    display(pd.DataFrame([report["checks"]]).T.rename(columns={0: "matched"}))
    if metric == "D_NL":
        display(pd.DataFrame.from_dict(report["previous_validation_checks"], orient="index",
                                       columns=["previous validation"]))
        print("The current run checks 5 mm versus 1 mm at Nt=2048; it does not repeat the older run's gradient, dense-CP or Nt=4096 checks.")

    labels = ["Equal / zero phase", "Previous gradient", "LLM phase + amplitude"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, column, ylabel in zip(axes, (f"{metric}_1mm", "brightness_ratio"),
                                 (f"{mode_label} {metric} (1 mm; lower is better)", f"{mode_label} brightness / matched baseline")):
        bars = ax.bar(labels, table[column], color=["#a8b6c6", "#cf7c32", "#366b9e"])
        ax.bar_label(bars, fmt="%.6f", padding=3, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.2*max(table[column]))
    axes[1].axhline(BRIGHTNESS_FLOOR, linestyle="--", color="#555555", label="50% minimum")
    axes[1].legend()
    fig.suptitle(f"{report['power_kW']:g} kW | previous and current selected launches")
    fig.savefig(target / "scores.png", dpi=150)
    fig.savefig(target / "scores.pdf")
    plt.show()

    fine = run.result["validation"]["selected_fine"]
    new_fields = run.directory / "gnlse" / f"evaluation_{fine['evaluation']:03d}" / "fields.npz"
    field_paths = [("Equal / zero phase", old["case"] / "baseline_fields.npz"),
                   ("Previous gradient", old["case"] / "selected_fields.npz"),
                   ("LLM phase + amplitude", new_fields)]
    if all(path.is_file() for _, path in field_paths):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        curves = [(label, path, "nonlinear", {}) for label, path in field_paths]
        curves.append(("Input (LLM launch)", new_fields, "input",
                       {"color": "black", "linestyle": "--", "linewidth": 1.7}))
        for label, path, field_key, style in curves:
            with np.load(path) as data:
                field, wl = data[field_key][:, modes, :], data["wavelength_nm"]
                indices = np.flatnonzero(np.isfinite(wl) & (wl > 0))
                indices = indices[np.argsort(wl[indices])]
                axes[0].plot(data["time_ps"], (abs(field)**2).sum(axis=(1, 2))/1000,
                             label=label, **style)
                spectrum = (abs(np.fft.fftshift(np.fft.ifft(field, axis=0), axes=0))**2).sum(axis=(1, 2))
                axes[1].plot(wl[indices], spectrum[indices], label=label, **style)
        axes[0].set(xlabel="Time (ps)", ylabel=f"{mode_label} power (kW)", xlim=(-1, 1))
        axes[1].set(xlabel="Wavelength (nm)", ylabel="Power per frequency bin (W)", xlim=(1200, 1600))
        for ax in axes:
            ax.legend()
        fig.savefig(target / "output_fields.png", dpi=150)
        fig.savefig(target / "output_fields.pdf")
        plt.show()
    else:
        print("Some historical field files are absent; the scalar comparison is still available.")
    print(report["conclusion"])
    print(f"LLM minus previous {metric}: {report[f'LLM_minus_previous_{metric}']:+.8f} (negative favors LLM).")
    print(report["interpretation"])
    print("Historical source:", old["case"])
    print("Saved comparison:", target)
    return table
