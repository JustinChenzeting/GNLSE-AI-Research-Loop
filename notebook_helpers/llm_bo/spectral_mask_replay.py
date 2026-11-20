"""Interactive replay of saved masks and fields with a diagnostic, prefix-only GP.

This module never calls the LLM, propagator, or an optimizer. The actual campaign
used its GP only for initialization. Posterior bands here are a retrospective
visualization, not recorded LLM confidence or evidence of numerical convergence.
"""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import html
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .spectral_mask import BANDS, FLOOR, MaskPrior, MaskRun, check_point, point_key

TEMPLATE = Path(__file__).with_name("spectral_mask_replay.html")
DISPLAY_NOISE_STD = 2e-4
BRANCHES = ("input", "linear", "nonlinear")


def load_saved_replay(directory):
    """Read a completed run for presentation only; never resume a campaign."""
    directory = Path(directory).resolve()
    result = json.loads((directory/"state.json").read_text())
    if result.get("status") != "completed":
        raise ValueError("Replay requires a completed saved run; it never resumes simulations.")
    plan = json.loads((directory/"initial_plan.json").read_text())
    prior = MaskPrior(plan["prior"], result["problem"]["spectral_grid"]["band_energy_fractions"])
    return MaskRun(directory, result, prior)


def mask_path(points, samples_per_segment=32):
    """A fixed path through complete masks, snapped to the actual 0.01 lattice.

    The x coordinate is a mask index, never a one-dimensional physical control.
    Geometry may include later masks; no later measured objective enters the GP.
    """
    if not points or samples_per_segment < 2:
        raise ValueError("Need masks and at least two samples per segment.")
    controls = np.array([check_point(p)["factors"] for p in points])
    x = np.linspace(0, len(points)-1, max(1, (len(points)-1)*samples_per_segment+1))
    left = np.minimum(x.astype(int), max(0, len(points)-2))
    right = np.minimum(left+1, len(points)-1)
    fraction = x-left
    values = np.round(100*((1-fraction[:, None])*controls[left]+fraction[:, None]*controls[right]))/100
    return x, [{"factors": row.tolist()} for row in values]


def posterior(prior, query, observed, scores, noise_std=DISPLAY_NOISE_STD):
    """Fixed-hyperparameter Gaussian conditioning; latent mean/std, no fitting."""
    if len(observed) != len(scores) or noise_std <= 0:
        raise ValueError("Invalid observations or display regularizer.")
    mean = prior.mean(query).astype(float)
    variance = np.full(len(query), prior.spec["std_D_NL"]**2)
    if not observed:
        return mean, np.sqrt(variance)
    scores = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Only finite measured scores may condition the display GP.")
    covariance = prior.covariance(observed)+noise_std**2*np.eye(len(observed))
    factor, lower = cho_factor(covariance, lower=True, check_finite=True)
    cross = prior.covariance(observed, query)
    mean += cross.T@cho_solve((factor, lower), scores-prior.mean(observed))
    solved = solve_triangular(factor, cross, lower=lower)
    variance -= np.sum(solved*solved, axis=0)
    if np.min(variance) < -1e-10:
        raise ArithmeticError("Invalid negative GP posterior variance.")
    return mean, np.sqrt(np.maximum(variance, 0))


def prefix_predictions(prior, points, scores, samples_per_segment=32):
    """State n sees exactly the first n measurements, including the baseline."""
    x, query = mask_path(points, samples_per_segment)
    states = []
    for count in range(len(points)+1):
        mean, std = posterior(prior, query, points[:count], scores[:count])
        node_mean, node_std = posterior(prior, points, points[:count], scores[:count])
        states.append({"observation_count": count, "mean": mean.tolist(), "std": std.tolist(),
                       "node_mean": node_mean.tolist(), "node_std": node_std.tolist(),
                       "mean_path_std": float(np.mean(std))})
    return x, states


def envelope_indices(curves, buckets=450):
    """Common display samples retain extrema of all branches; original files stay intact."""
    curves = np.asarray(curves)
    count = curves.shape[1]
    keep = {0, count-1}
    for start, end in zip(np.linspace(0, count, buckets+1, dtype=int)[:-1],
                          np.linspace(0, count, buckets+1, dtype=int)[1:]):
        if start == end:
            continue
        for row in curves:
            keep.add(start+int(np.argmin(row[start:end])))
            keep.add(start+int(np.argmax(row[start:end])))
    return np.array(sorted(keep), dtype=int)


def _list(values):
    # Display serialization only; calculations and source fields retain full precision.
    return np.asarray(values, dtype=float).round(10).tolist()


def build_replay_data(run):
    directory = Path(run.directory).resolve()
    result = run.result
    history = result["history"]
    if not history or not all(row["valid"] for row in history):
        raise ValueError("Replay currently requires finite, valid saved observations.")
    points = [check_point(row["point"]) for row in history]
    scores = [row["D_NL"] for row in history]
    plan = json.loads((directory/"initial_plan.json").read_text())
    prior = MaskPrior(plan["prior"], result["problem"]["spectral_grid"]["band_energy_fractions"])
    x, gp = prefix_predictions(prior, points, scores)
    rationales = {}
    for path in sorted((directory/"ai").glob("call_*/metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata["status"] == "completed":
            answer = json.loads((path.parent/"answer.json").read_text())
            if answer.get("action") == "evaluate":
                rationales[point_key(answer["point"])] = answer["rationale"]
    frames, sources = [], []
    reference = result["calibration"][1]["diagnostics"]["total_photons"]
    for index, row in enumerate(history):
        measurement = directory/"gnlse"/f"evaluation_{row['evaluation']:03d}"/"measurement.json"
        record = json.loads(measurement.read_text())
        fields = measurement.with_name("fields.npz")
        digest = hashlib.sha256(fields.read_bytes()).hexdigest()
        if digest != record["artifacts"]["fields_sha256"]:
            raise ValueError(f"Saved fields changed: {fields}")
        if record["point"] != points[index] or record["D_NL"] != scores[index]:
            raise ValueError("History and saved measurement disagree.")
        with np.load(fields, allow_pickle=False) as data:
            t = data["time_ps"]
            omega = data["omega_detuning_rad_ps"]
            power = np.array([np.sum(abs(data[key])**2, axis=(1, 2))/1000 for key in BRANCHES])
            spectra = np.array([np.sum(abs(np.fft.fftshift(np.fft.ifft(data[key], axis=0), axes=0))**2,
                                       axis=(1, 2)) for key in BRANCHES])
            # Verify every plotted score from all three saved modes.
            a, b = data["linear"], data["nonlinear"]
            score = 1-abs(np.vdot(a, b))**2/(np.vdot(a, a).real*np.vdot(b, b).real)
            if abs(score-row["D_NL"]) > 1e-12:
                raise ValueError("Saved fields do not reproduce all-mode D_NL.")
            centre = np.flatnonzero(abs(t) <= 2)
            full = envelope_indices(power)
            halfwidth = float(data["band_edges_detuning_rad_ps"][-1])
            spectral = np.flatnonzero(abs(omega) <= halfwidth+15)
            before, after = gp[index], gp[index+1]
            frames.append({
                "iteration": index, "evaluation": row["evaluation"], "role": row["role"],
                "D_NL": row["D_NL"], "brightness": row["brightness_ratio"], "feasible": row["feasible"],
                "factors": points[index]["factors"], "input_peak_kW": float(power[0].max()),
                "input_energy_retained": row["diagnostics"]["input_energy_retained"],
                "before_mean": before["node_mean"][index], "before_std": before["node_std"][index],
                "after_mean": after["node_mean"][index], "after_std": after["node_std"][index],
                "best_feasible": min(r["D_NL"] for r in history[:index+1] if r["feasible"]),
                "time_centre_ps": _list(t[centre]), "power_centre_kW": _list(power[:, centre]),
                "time_full_ps": _list(t[full]), "power_full_kW": _list(power[:, full]),
                "omega_detuning_rad_ps": _list(omega[spectral]), "spectra_W": _list(spectra[:, spectral]),
                "band_edges_rad_ps": _list(data["band_edges_detuning_rad_ps"]),
                "resolution": record["resolution"],
                "rationale": ("Unmasked reference: all 100 factors are one." if index == 0 else
                              "Seeded uniform random mask, drawn before the first simulation." if index == 1 else
                              rationales.get(point_key(points[index]), "No recorded rationale for this mask.")),
            })
        if not np.isclose(row["brightness_ratio"], record["metrics"]["total_photons"]/reference, rtol=1e-12):
            raise ValueError("Brightness reference differs from the saved unmasked baseline.")
        sources.append({"evaluation": row["evaluation"], "fields": str(fields), "sha256": digest})
    low = min(min(np.array(p["mean"])-1.96*np.array(p["std"])) for p in gp)
    high = max(max(np.array(p["mean"])+1.96*np.array(p["std"])) for p in gp)
    return {"schema_version": 1, "run_name": directory.name, "frames": frames,
            "path_x": x.tolist(), "gp": gp, "gp_y_range": [float(min(0, low)-.04), float(max(1, high)+.04)],
            "prior": plan["prior"], "display_noise_std": DISPLAY_NOISE_STD,
            "brightness_floor": FLOOR, "validation": result["validation"]["status"],
            "failed_checks": [k for k, v in result["validation"]["checks"].items() if not v],
            "sources": sources,
            "method": "Retrospective GP reconstruction; fixed original prior, prefix-only observations. "
                      "Not LLM uncertainty or the acquisition model used in the saved campaign.",
            "path_definition": "A fixed piecewise-linear path through all recorded 100-factor masks, snapped to 0.01. "
                               "The horizontal coordinate is a display index. Future scores are never used before their iteration.",
            "pulse_definition": "Saved 5 mm search-grid fields; summed power across all three modes. "
                                "No renormalization, time alignment, or new propagation. Full-window display retains extrema."}


def export_replay(run, output_directory=None):
    """Write a self-contained HTML viewer and its provenance; no network dependencies."""
    target = Path(output_directory) if output_directory else Path(run.directory)/"interactive_replay"
    target.mkdir(parents=True, exist_ok=True)
    data = build_replay_data(run)
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    document = TEMPLATE.read_text().replace("__REPLAY_DATA__", encoded)
    html_path = target/"spectral_mask_replay.html"
    html_path.write_text(document)
    (target/"replay_data.json").write_text(encoded+"\n")
    (target/"provenance.json").write_text(json.dumps({
        "method": data["method"], "path": data["path_definition"], "pulse_definition": data["pulse_definition"],
        "sources": data["sources"], "prior": data["prior"], "display_noise_std": DISPLAY_NOISE_STD,
        "gradient_descent_executed": False, "new_simulations": 0, "new_AI_calls": 0,
        "numerical_validation": data["validation"], "failed_checks": data["failed_checks"],
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
    }, indent=2)+"\n")
    return html_path


def show_replay(run):
    """Embed the offline viewer in a trusted notebook and save a standalone copy."""
    from IPython.display import HTML, display
    path = export_replay(run)
    document = html.escape(path.read_text(), quote=True)
    display(HTML(f'<iframe title="Spectral-mask iteration replay" srcdoc="{document}" '
                 'sandbox="allow-scripts allow-downloads" style="width:100%;height:1110px;border:0;border-radius:6px"></iframe>'))
    print("Standalone interactive viewer:", path)
    print("Slider: baseline 0 through the final search trial. Curves use saved fields; uncertainty is a diagnostic GP reconstruction.")
    return path
