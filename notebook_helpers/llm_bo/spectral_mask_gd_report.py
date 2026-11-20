"""Read-only analysis of GD measurements and a frozen saved LLMBO checkpoint."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np

from . import spectral_mask as sm
from .spectral_mask_gd import save


def _records(directory):
    return [json.loads(p.read_text()) for p in sorted((Path(directory)/"gnlse").glob("evaluation_*/measurement.json"))]


def _events(directory):
    return [json.loads(line) for line in (Path(directory)/"events.jsonl").read_text().splitlines()]


def snapshot_reference(run, reference_directory):
    """Copy metadata into the new run; never write into the LLMBO directory."""
    destination = run.directory / "llm_reference_snapshot"
    index_path = destination / "index.json"
    if index_path.exists():
        return destination
    source = Path(reference_directory).resolve()
    paths = [source / name for name in ("state.json", "run.json", "initial_plan.json", "gnlse/evaluator.json")]
    paths += sorted((source / "gnlse").glob("evaluation_*/measurement.json"))
    paths += sorted((source / "ai").glob("call_*/metadata.json"))
    index = []
    for path in paths:
        if not path.exists():
            continue
        data = path.read_bytes()
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        index.append({"source": str(path), "copy": str(relative), "sha256": hashlib.sha256(data).hexdigest()})
    save(index_path, {"reference_directory": str(source), "files": index,
                     "note": "Read-only snapshot; incomplete or unfinished operations are not filled in."})
    return destination


def _compatibility(run, snapshot):
    original = json.loads((snapshot/"state.json").read_text())
    new = json.loads((run.directory/"run.json").read_text())
    old_identity = json.loads((snapshot/"gnlse/evaluator.json").read_text())
    fields = ("power_kW", "seed", "occupied_energy_fraction", "samples_per_band", "nt", "search_dz_m", "check_dz_m")
    checks = {k: new["physics"][k] == original["config"][k] for k in fields}
    for key in ("setup", "worker_sha256", "shared_worker_sha256", "source_sha256", "manifest_sha256",
                "objective_modes", "brightness_modes", "power_renormalization"):
        checks[key] = new["evaluator"].get(key) == old_identity.get(key)
    old_initial = json.loads((snapshot/"initial_plan.json").read_text())["initial_point"]
    checks["identical_seeded_start"] = new["initial_point"] == old_initial
    return checks


def _llm_timeline(state, records, calls):
    """Reconstruct serial observation times from measured components, not file mtimes."""
    by_id = {r["evaluation"]: r for r in records if r["status"] == "ok"}
    elapsed = sum(c.get("seconds", 0) for c in calls if c.get("operation") == "prepare_prior")
    acquisitions = iter(c for c in calls if c.get("operation") == "choose_next_mask")
    last_id, rows = 0, []
    for i, observation in enumerate(state.get("history", [])):
        if i >= 2:
            for call in acquisitions:
                if "seconds" not in call:
                    return rows  # Do not invent the elapsed time of an unfinished call.
                elapsed += call["seconds"]
                if call.get("status") == "completed":
                    break
            else:
                return rows
        record = by_id.get(observation["evaluation"])
        if record is None:
            break
        for evaluation in range(last_id+1, observation["evaluation"]+1):
            if evaluation not in by_id:
                return rows
            elapsed += by_id[evaluation]["wall_seconds"]
        last_id = observation["evaluation"]
        rows.append({**observation, "elapsed_seconds": elapsed, "evaluation_pairs": last_id,
                     "time_basis": "reconstructed estimate from measured AI and objective-pair durations"})
    return rows


def comparison_data(run, reference_directory):
    snapshot = snapshot_reference(run, reference_directory)
    old = json.loads((snapshot/"state.json").read_text())
    previous, measured = _records(snapshot), _records(run.directory)
    calls = [json.loads(p.read_text()) for p in sorted((snapshot/"ai").glob("call_*/metadata.json"))]
    rows = []
    for name, result, records, model_calls in (("Projected FD GD", run.result, measured, []),
                                              ("LLMBO", old, previous, calls)):
        successful = [r for r in records if r["status"] == "ok"]
        best = result.get("best_search") or min((r for r in result.get("history", []) if r["feasible"]), key=lambda r:r["D_NL"])
        v = result.get("validation", {})
        fine, refined = v.get("selected_fine", {}), v.get("selected_refined", {})
        propagation = sum(r.get("metrics", {}).get("propagation_seconds", 0.) for r in successful)
        ai = sum(c.get("seconds", 0.) for c in model_calls)
        recorded_total = result.get("total_wall_seconds")
        current_checkpoint = result.get("evaluation_attempts") == len(records)
        total = recorded_total if current_checkpoint else None
        rows.append({"method": name, "run_status": result["status"], "search_D_NL": best["D_NL"],
            "search_brightness": best["brightness_ratio"], "fine_D_NL": fine.get("D_NL"),
            "fine_brightness": fine.get("brightness_ratio"), "refined_D_NL": refined.get("D_NL"),
            "refined_brightness": refined.get("brightness_ratio"), "validation": v.get("status", "not_run"),
            "objective_pairs_attempted": len(records), "objective_pairs_completed": len(successful),
            "linear_GNLSE_completed": sum(r.get("metrics",{}).get("linear_solves",0) for r in successful),
            "nonlinear_GNLSE_completed": sum(r.get("metrics",{}).get("nonlinear_solves",0) for r in successful),
            "propagations_completed": sum(r.get("metrics",{}).get("linear_solves",0)+r.get("metrics",{}).get("nonlinear_solves",0) for r in successful),
            "AI_attempts": len(model_calls), "AI_calls_without_duration": sum("seconds" not in c for c in model_calls),
            "propagation_seconds": propagation, "AI_seconds_including_failed_calls": ai,
            "other_seconds": None if total is None else total-propagation-ai,
            "total_wall_seconds": total, "last_recorded_wall_seconds": recorded_total,
            "runtime_basis": ("unavailable: runtime checkpoint predates current evaluations" if not current_checkpoint else
                              "measured active runtime" if result["status"] == "completed" else "recorded active runtime at incomplete checkpoint")})
    events = _events(run.directory)
    report = {"table": rows, "compatibility": _compatibility(run, snapshot),
              "GD_accepted_steps": len(run.result["accepted_steps"]),
              "GD_gradient_probe_requests": sum(e["kind"]=="objective_request" and e.get("role")=="gradient probe" for e in events),
              "GD_rejected_proposals": sum(e["kind"]=="step_rejected" for e in events),
              "GD_cache_hits": run.result["cache_hits"], "LLM_snapshot": str(snapshot),
              "LLM_timeline": _llm_timeline(old, previous, calls),
              "notes": ["Each objective pair contains one matched linear and one nonlinear propagation.",
                        "GD totals include baseline, restoration, gradient probes, rejected proposals, final checks and worker shutdown.",
                        "LLM overhead includes prior/acquisition calls and failed calls with recorded durations; unfinished calls have unknown cost.",
                        "GD observation times are measured directly. LLM observation times are reconstructed estimates from measured component durations; controller gaps are not included.",
                        "No equal-quality speedup is inferred. Budgets and attained scores differ; failed or absent refinement checks leave results preliminary.",
                        "This is one shared seed on the same local solver, not a controlled repeated hardware benchmark; concurrent machine load was not controlled. Runtime excludes notebook plotting for both methods."]}
    save(run.directory/"measured_comparison.json", report)
    audit = run.directory/"timing_audit.json"
    if audit.exists():
        report["timing_audit"] = json.loads(audit.read_text())
        save(run.directory/"measured_comparison.json", report)
    return report


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":12, "axes.labelsize":12, "axes.titlesize":12, "xtick.labelsize":12,
        "ytick.labelsize":12, "legend.fontsize":12, "font.weight":"normal", "axes.labelweight":"normal",
        "axes.titleweight":"normal", "pdf.fonttype":42, "svg.fonttype":"none"})


def _finish(fig, stem):
    """Square panels, vector output, and previews for final visual review."""
    from matplotlib.text import Text
    from PIL import Image
    styles = {"Masked input":("#666666",":"), "Matched linear output":("#CC8844","--"),
              "Nonlinear output":("#4477AA","-"), "Unmasked":("#666666",":"),
              "Selected masked input":("#4477AA","-"), "Measured, 5 mm":("#4477AA","-"),
              "Best feasible":("#CC8844","--"), "Output photon ratio":("#4477AA","-"),
              "Input energy retained":("#CC8844","--"), "Brightness minimum":("#666666",":"),
              "Brightness floor":("#666666",":")}
    for i, ax in enumerate(fig.axes):
        ax.set_box_aspect(1)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.text(.025,.97,f"({chr(97+i)})",transform=ax.transAxes,va="top")
        for line in ax.lines:
            if line.get_label() in styles:
                color, linestyle = styles[line.get_label()]
                line.set_color(color); line.set_linestyle(linestyle)
        legend = ax.get_legend()
        if legend:
            for handle, label in zip(legend.legend_handles, legend.texts):
                if label.get_text() in styles and hasattr(handle,"set_linestyle"):
                    color, linestyle = styles[label.get_text()]
                    handle.set_color(color); handle.set_linestyle(linestyle)
    for item in fig.findobj(Text):
        item.set_fontsize(12); item.set_fontweight("normal"); item.set_color("black")
    for suffix in ("png","pdf","svg"):
        fig.savefig(Path(stem).with_suffix('.'+suffix),dpi=160)
    with Image.open(Path(stem).with_suffix('.png')) as original:
        original.convert('L').save(str(stem)+'_grayscale.png')
        original.resize((original.width//4,original.height//4)).save(str(stem)+'_quarter.png')


def descent_route_data(run):
    """Accepted iterates only; probes and rejected proposals never advance the route."""
    start = run.result.get("restored_start")
    if start is None:
        return []
    route = [{**start, "iteration": 0, "stage": "Feasible start"}]
    route.extend({**step, "stage": f"Accepted step {step['iteration']}"}
                 for step in run.result.get("accepted_steps", []))
    return route


def show_descent_route(run):
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display, Markdown
    _style()
    route = descent_route_data(run)
    display(Markdown(f"**Saved stopping reason:** {run.result['stop_reason']}. "
                     f"The configured limit was **{run.result['descent_config']['max_iterations']} iterations**; "
                     f"{len(run.result['accepted_steps'])} steps were accepted. "
                     "This stopping reason does not establish convergence."))
    if not route:
        print("No feasible descent start was saved.")
        return
    rows = [{"iterate": r["iteration"], "stage": r["stage"], "evaluation": r["evaluation"],
             "D_NL": r["D_NL"], "brightness": r["brightness_ratio"],
             "active elapsed (min)": r["elapsed_seconds"]/60} for r in route]
    table = pd.DataFrame(rows)
    display(table.round(6))
    table.to_csv(run.directory/"descent_route.csv", index=False)
    pd.DataFrame({f"iterate_{r['iteration']}": r["point"]["factors"] for r in route},
                 index=np.arange(1, sm.BANDS+1)).rename_axis("band").to_csv(run.directory/"descent_route_factors.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), layout="constrained")
    for ax, key, ylabel in zip(axes, ("D_NL", "brightness"), ("All-mode D_NL", "Output photon ratio")):
        ax.plot(table.iterate, table[key], "o-", color="#4477AA", linewidth=1.8)
        ax.set(xlabel="Accepted GD step (0 = feasible start)", ylabel=ylabel,
               xticks=table.iterate, xlim=(-.25, table.iterate.max()+.25))
        span = max(float(table[key].max()-table[key].min()), .005)
        low = min(float(table[key].min()), sm.FLOOR) if key == "brightness" else float(table[key].min())
        ax.set_ylim(low-.3*span, float(table[key].max())+.6*span)
        for x, y in zip(table.iterate, table[key]):
            interior = x not in (table.iterate.min(), table.iterate.max())
            ax.annotate(f"{y:.6f}", (x,y), xytext=(7 if interior else 0, 9),
                        textcoords="offset points", ha="left" if interior else "center")
        ax.ticklabel_format(axis="y", useOffset=False)
    axes[0].set_title("Accepted objective route")
    axes[1].set_title("Accepted brightness route")
    axes[1].axhline(sm.FLOOR, color="#666666", linestyle=":")
    axes[1].annotate("50% floor", (0, sm.FLOOR), xytext=(0,-14), textcoords="offset points")
    _finish(fig, run.directory/"descent_route")
    display(fig); plt.close(fig)

    rejected = [e for e in _events(run.directory) if e["kind"] == "step_rejected"]
    measured_rejected = [e for e in rejected if "brightness_ratio" in e]
    below_floor = sum(e["brightness_ratio"] < sm.FLOOR for e in measured_rejected)
    seed = next((r for r in run.result["history"] if r["role"] == "seeded random mask"), None)
    restoration_note = ""
    if seed and not seed["feasible"]:
        restoration_note = (f"The seed had **{100*seed['brightness_ratio']:.2f}% brightness**. "
                            f"Restoration raised D_NL from {seed['D_NL']:.6f} to {route[0]['D_NL']:.6f} "
                            "to obtain the feasible start, before GD began. ")
    display(Markdown(
        f"**Why the earlier trace bounced:** it joined restoration and trial proposals to the accepted steps. "
        + restoration_note +
        f"Here, **{below_floor} of {len(measured_rejected)} measured rejected proposals** were below the 50% brightness floor. "
        "A smaller D_NL at an infeasible proposal cannot replace the current mask. "
        "The step direction follows D_NL slopes; brightness is checked after measuring each proposal. "
        "Backtracking halves the step from the same current mask until a feasible improvement is found. "
        "Each new iteration starts again with a proposed maximum factor change of 0.10. "
        "Restoration can increase D_NL to recover brightness; the accepted GD route above is separate."))
    if len(rejected) > len(measured_rejected):
        print(f"{len(rejected)-len(measured_rejected)} additional projected proposal(s) reproduced the current mask; no new simulation was needed.")


def _plot_trial_history(axes, run):
    """Unconnected trial markers distinguish measurements from the accepted path."""
    from matplotlib.ticker import MaxNLocator
    history = run.result["history"]
    positions = {r["evaluation"]: i for i, r in enumerate(history)}
    route = descent_route_data(run)
    accepted_ids = {r["evaluation"] for r in route}
    rejected_ids = {e["evaluation"] for e in _events(run.directory)
                    if e["kind"] == "step_rejected" and "evaluation" in e}
    other = [r for r in history if r["evaluation"] not in accepted_ids | rejected_ids]
    rejected = [r for r in history if r["evaluation"] in rejected_ids]
    for ax in axes:
        ax.clear()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for rows, label, marker, color, linestyle in (
            (other, "Initialization / restoration", "s", "#888888", "none"),
            (rejected, "Rejected proposal", "x", "#CC8844", "none"),
            (route, "Accepted GD route", "o", "#4477AA", "-")):
        axes[0].plot([positions[r["evaluation"]] for r in rows], [r["D_NL"] for r in rows],
                     marker=marker, color=color, linestyle=linestyle, label=label)
    axes[0].set(xlabel="Recorded proposal (probes omitted)", ylabel="All-mode D_NL",
                title="Trials and accepted route")
    x = np.arange(len(history))
    axes[1].plot(x, [r["brightness_ratio"] for r in history], "o", color="#4477AA", markersize=4,
                 label="Measured photon ratio")
    axes[1].plot(x, [r["diagnostics"]["input_energy_retained"] for r in history], "s", color="#CC8844",
                 markersize=3, markerfacecolor="none", label="Measured input energy")
    axes[1].plot([positions[r["evaluation"]] for r in route], [r["brightness_ratio"] for r in route],
                 "o-", color="#4477AA", label="Accepted photon ratio")
    axes[1].axhline(sm.FLOOR, color="#666666", linestyle=":", label="Brightness floor")
    axes[1].set(xlabel="Recorded proposal (probes omitted)", ylabel="Fraction of unmasked reference")


def show_results(run):
    """Retain the original mask, throughput, pulse-centre, tail and spectrum plots."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display
    _style()
    validation = run.result.get("validation", {})
    if not all(name in validation for name in ("baseline_fine","selected_fine","baseline_refined","selected_refined")):
        display(pd.DataFrame(run.result.get("history", [])))
        print("PRELIMINARY: final validation is incomplete. Saved fields and events remain available.")
        return
    adapted = deepcopy(run.result)
    adapted["validation"]["checks"] = {k:v=="pass" for k,v in validation["checks"].items()}
    # Collect the existing plotting helper's figures, then apply consistent export styling.
    original_show = plt.show
    figures = []
    plt.show = lambda *a, **k: figures.append(plt.gcf())
    try:
        sm.show_results(SimpleNamespace(directory=run.directory, result=adapted))
    finally:
        plt.show = original_show
    for fig, name in zip(figures, ("mask_results", "pulse_results")):
        if name == "mask_results":
            _plot_trial_history(fig.axes[:2], run)
            fig.axes[0].set_ylim(.25,.94)
            fig.axes[1].set_ylim(.35,1.3)
            fig.axes[2].set_ylim(-.03,1.4)
            fig.axes[2].set_yticks([0,.25,.5,.75,1.])
            # Do not draw artificial zero-transmission edges on a finite-band mask.
            for patch in list(fig.axes[2].patches):
                if hasattr(patch,"get_data"):
                    data = patch.get_data()
                    properties = dict(color=patch.get_edgecolor(),linewidth=patch.get_linewidth(),
                                      label=patch.get_label())
                    patch.remove()
                    fig.axes[2].stairs(data.values,data.edges,baseline=None,**properties)
            fig.axes[3].set_ylim(0,3.2)
        else:
            fig.axes[0].set_ylim(-1,66)
            fig.axes[1].set_ylim(top=100)
            fig.axes[2].set_ylim(-.03,2.)
        for ax in fig.axes:
            ax.legend(loc="upper right",bbox_to_anchor=(.98,.94),frameon=False)
        if name == "pulse_results":
            # The same three branches appear in every panel; one shared key suffices.
            for ax in fig.axes[1:]:
                ax.get_legend().remove()
        _finish(fig, run.directory/name)
        display(fig)
        plt.close(fig)
    display(pd.DataFrame.from_dict(validation["checks"], orient="index", columns=["check status"]))
    records = _records(run.directory)
    pd.DataFrame([{k:r.get(k) for k in ("evaluation","status","D_NL","wall_seconds")}
                  | {"linear_seconds":r.get("metrics",{}).get("linear_seconds"),
                     "nonlinear_seconds":r.get("metrics",{}).get("nonlinear_seconds"),
                     "Nt":r["resolution"]["Nt"],"dz_m":r["resolution"]["dz_m"]} for r in records]).to_csv(run.directory/"objective_pairs.csv",index=False)
    events = _events(run.directory)
    pd.DataFrame([{k:e.get(k) for k in ("kind","elapsed_seconds","role","evaluation","iteration","band","side",
                    "D_NL","brightness_ratio","feasible","cached","backtrack","alpha","reason")}
                  for e in events]).to_csv(run.directory/"event_summary.csv",index=False)
    print("Finite-difference probes and all proposal decisions:", run.directory/"events.jsonl")
    print("Result:", run.result["result_label"], "|", run.result["stop_reason"])


def show_comparison(run, reference_directory):
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display, Markdown
    _style()
    report = comparison_data(run, reference_directory)
    table = pd.DataFrame(report["table"])
    table.to_csv(run.directory/"measured_comparison.csv",index=False)
    display(table[["method","search_D_NL","search_brightness","fine_D_NL","refined_D_NL","validation"]].round(6))
    costs = table[["method","objective_pairs_completed","propagation_seconds","AI_seconds_including_failed_calls",
                   "total_wall_seconds","runtime_basis"]].copy()
    for column in ("propagation_seconds","AI_seconds_including_failed_calls","total_wall_seconds"):
        costs[column] = costs[column]/60
    costs.columns = ["method","objective pairs","propagation (min)","AI including failures (min)","active total (min)","runtime basis"]
    display(costs.round(3))
    if "timing_audit" in report:
        audit = report["timing_audit"]
        display(pd.DataFrame([{"GD active timer (min)":audit["recorded_active_seconds"]/60,
            "calendar timestamp span (min)":audit["calendar_timestamp_span_seconds"]/60,
            "gap absent from active timer (min)":audit["timestamp_gap_not_in_active_timer_seconds"]/60}]).round(3))
        print(audit["calendar_basis"])
        print(audit["pilot_limit_note"])
    propagation_counts = [{"method": row["method"], "objective pairs":row["objective_pairs_completed"],
                           "linear GNLSE":row["linear_GNLSE_completed"], "nonlinear GNLSE":row["nonlinear_GNLSE_completed"],
                           "GNLSE propagations (linear + nonlinear)":row["propagations_completed"]}
                          for row in report["table"]]
    display(pd.DataFrame(propagation_counts))
    resource_path = run.directory/"resource_summary.json"
    if resource_path.exists():
        resources = json.loads(resource_path.read_text())
        display(pd.DataFrame([{"backend":resources["solver_backend"],"GPU":resources.get("gpu_model"),
            "solver GPU use":resources["solver_GPU_compute"],
            "system GPU sample mean (%)":resources["device_utilization_percent_mean"],
            "system GPU maximum (%)":resources["device_utilization_percent_max"],
            "telemetry starts (min)":resources["coverage_start_seconds"]/60,
            "samples":resources["measured_samples"]}]).round(2))
        print("GPU telemetry is system-wide and includes other applications. The GNLSE solver uses the CPU; its GPU-use estimate is 0%, inferred from the backend, not measured per process.")
        print(resources["note"])
        if "timing_audit" in report:
            audit = report["timing_audit"]
            print("GPU samples (UTC):",audit["GPU_first_sample_utc"],"to",audit["GPU_last_sample_utc"])
            print(audit["GPU_coverage"])
        pd.DataFrame([json.loads(line) for line in (run.directory/"resource_usage.jsonl").read_text().splitlines()]).to_csv(run.directory/"resource_usage.csv",index=False)
    display(pd.DataFrame([{"quantity":"Accepted GD steps","value":report["GD_accepted_steps"]},
        {"quantity":"Gradient probe requests","value":report["GD_gradient_probe_requests"]},
        {"quantity":"Rejected line-search proposals","value":report["GD_rejected_proposals"]},
        {"quantity":"Cache hits","value":report["GD_cache_hits"]}]))
    mismatches = [k for k,v in report["compatibility"].items() if not v]
    print("Frozen physics and seed comparison:", "match" if not mismatches else "MISMATCH: "+", ".join(mismatches))
    old = report["table"][1]
    llm_label = "LLMBO" if old["run_status"] == "completed" else "LLMBO checkpoint"
    if old["run_status"] != "completed":
        print("LLMBO is an incomplete saved checkpoint. Missing fine/refined results and unfinished-call costs are not estimated as completed work.")
    events = _events(run.directory)
    measurements = [e for e in events if e["kind"]=="objective_result" and not e["cached"]]
    accepted_ids = {s["evaluation"] for s in run.result["accepted_steps"]}
    accepted_ids.add(run.result["history"][0]["evaluation"])
    accepted_ids.update(r["evaluation"] for r in run.result["history"]
                        if r["role"] in ("seeded random mask", "feasibility restoration"))
    if "restored_start" in run.result:
        accepted_ids.add(run.result["restored_start"]["evaluation"])
    gd = [e for e in measurements if e["evaluation"] in accepted_ids and e["resolution"]["dz_m"]==.005]
    llm = report["LLM_timeline"]
    fig, axes = plt.subplots(1,3,figsize=(16,5.5),layout="constrained",sharex=True)
    timeline_rows = []
    for name, data, color, linestyle, marker, total in (
            ("FD GD",gd,"#4477AA","-","o",report["table"][0]["total_wall_seconds"]),
            (llm_label,llm,"#CC8844","--","s",old["total_wall_seconds"])):
        feasible = [r for r in data if r["feasible"]]
        t = [r["elapsed_seconds"]/60 for r in feasible]
        best = np.minimum.accumulate([r["D_NL"] for r in feasible]).tolist()
        if t:
            end = total/60 if total is not None else t[-1]
            axes[0].step(t+[end],best+[best[-1]],where="post",color=color,linestyle=linestyle,label=name)
            axes[0].plot(t,best,color=color,linestyle="none",marker=marker,markersize=4)
        axes[1].plot([r["elapsed_seconds"]/60 for r in data],[r["brightness_ratio"] for r in data],
                     color=color,linestyle=linestyle,marker=marker,markersize=4,label=name)
        for row in data:
            timeline_rows.append({"method":name,"evaluation":row["evaluation"],"elapsed_seconds":row["elapsed_seconds"],
                "D_NL":row["D_NL"],"brightness_ratio":row["brightness_ratio"],"feasible":row["feasible"],
                "time_basis":"directly measured" if name=="FD GD" else row["time_basis"]})
    axes[1].axhline(sm.FLOOR,color="#666666",linestyle=":",label="Brightness floor")
    axes[2].step([r["elapsed_seconds"]/60 for r in measurements],range(1,len(measurements)+1),where="post",color="#4477AA",label="FD GD")
    if llm:
        end = old["total_wall_seconds"]/60 if old["total_wall_seconds"] is not None else llm[-1]["elapsed_seconds"]/60
        axes[2].step([r["elapsed_seconds"]/60 for r in llm]+[end],
                     [r["evaluation_pairs"] for r in llm]+[old["objective_pairs_completed"]],where="post",color="#CC8844",linestyle="--",label=llm_label)
    for ax, ylabel in zip(axes,("Best feasible iterate D_NL","Output photon ratio","Completed objective pairs")):
        ax.set(xlabel="Active elapsed time (min)",ylabel=ylabel,xlim=(0,None))
        ax.legend(loc="upper right",frameon=False)
    axes[0].set_ylim(0,1.18*max(r["D_NL"] for r in gd+llm if r["feasible"]))
    axes[1].set_ylim(0,1.18*max(r["brightness_ratio"] for r in gd+llm))
    axes[2].set_ylim(bottom=0)
    axes[2].legend(loc="upper left",bbox_to_anchor=(.015,.9),frameon=False)
    _finish(fig,run.directory/"measured_convergence")
    display(fig);plt.close(fig)
    pd.DataFrame(timeline_rows).to_csv(run.directory/"convergence.csv",index=False)
    print("GD traces include the seeded start, restoration and accepted iterates; only feasible points enter the best-score curve. Gradient probes do not replace the iterate.")
    print("GD elapsed times are directly measured. Dashed LLMBO times are reconstructed estimates from recorded AI and objective-pair durations.")
    for note in report["notes"]:
        print(note)
    gd_result,llm_result = report["table"]
    if gd_result["fine_D_NL"] is not None and llm_result["fine_D_NL"] is not None:
        display(Markdown(
            f"**Measured pilot:** after feasibility restoration, GD accepted {report['GD_accepted_steps']} steps. "
            f"At 1 mm its D_NL is **{gd_result['fine_D_NL']:.6f}** and brightness is "
            f"**{gd_result['fine_brightness']:.6f}**; LLMBO gives **{llm_result['fine_D_NL']:.6f}** "
            f"and **{llm_result['fine_brightness']:.6f}**. GD used **{gd_result['objective_pairs_completed']} pairs** "
            f"versus **{llm_result['objective_pairs_completed']}** for LLMBO. "
            "These attained qualities differ, so the timing comparison does not establish an equal-quality speedup. "
            f"Numerical-check status: GD **{gd_result['validation']}**, LLMBO **{llm_result['validation']}**; "
            "any failed checks leave the corresponding result preliminary."))
    return table


def show_pulse_comparison(run, reference_directory):
    """Show LLAMBO beside GD, with each mask's input and both propagation branches."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display, Markdown
    _style()
    snapshot = snapshot_reference(run, reference_directory)
    old = json.loads((snapshot/"state.json").read_text())
    cases, sources = {}, []
    for method, state, directory in (("BO", old, snapshot), ("GD", run.result, run.directory)):
        selected = state.get("validation", {}).get("selected_fine")
        if selected is None:
            print(f"Pulse comparison unavailable: {method} has no saved fine-grid selected result.")
            return
        record = next(r for r in _records(directory) if r["evaluation"] == selected["evaluation"])
        path = Path(record["artifacts"]["fields"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["artifacts"]["fields_sha256"]:
            raise ValueError(f"Saved {method} field hash differs from its measurement record: {path}")
        with np.load(path) as fields:
            cases[method] = {key: fields[key].copy() for key in
                            ("time_ps", "omega_absolute_rad_ps", "omega_detuning_rad_ps", "wavelength_nm",
                             "band_edges_detuning_rad_ps", "input", "linear", "nonlinear", "Nt", "time_window_ps", "dz_m")}
        sources.append({"method": method, "evaluation": selected["evaluation"], "fields": str(path),
                        "sha256": digest, "resolution": selected["resolution"],
                        "D_NL": selected["D_NL"], "brightness_ratio": selected["brightness_ratio"],
                        "validation": state["validation"]["status"]})
    for key in ("time_ps", "omega_absolute_rad_ps", "wavelength_nm", "Nt", "time_window_ps", "dz_m"):
        if not np.array_equal(cases["BO"][key], cases["GD"][key]):
            raise ValueError(f"The selected BO/GD fields do not share the same {key}; compare matched grids first.")

    reference = cases["GD"]
    time = reference["time_ps"]
    wavelength = reference["wavelength_nm"]
    spectral_order = np.flatnonzero(np.isfinite(wavelength) & (wavelength > 0))
    spectral_order = spectral_order[np.argsort(wavelength[spectral_order])]
    time_table, spectrum_table = {"time_ps": time}, {"wavelength_nm": wavelength[spectral_order]}
    curves = {}
    max_power, max_spectrum, min_power = 0., 0., np.inf
    for method in ("BO", "GD"):
        for key in ("input", "linear", "nonlinear"):
            field = cases[method][key]
            power = (abs(field)**2).sum(axis=(1,2))/1000
            spectrum = (abs(np.fft.fftshift(np.fft.ifft(field, axis=0), axes=0))**2).sum(axis=(1,2))
            curves[method, key] = power, spectrum[spectral_order]
            time_table[f"{method}_{key}_kW"] = power
            spectrum_table[f"{method}_{key}_W_per_frequency_bin"] = spectrum[spectral_order]
            max_power = max(max_power, float(power.max()))
            max_spectrum = max(max_spectrum, float(spectrum.max()))
            min_power = min(min_power, float(power[power > 0].min()))

    carrier = float(np.mean(reference["omega_absolute_rad_ps"]-reference["omega_detuning_rad_ps"]))
    frequency_window = carrier + reference["band_edges_detuning_rad_ps"][[0,-1]] + np.array([-10,10])
    wavelength_window = 2*np.pi*299792.458/frequency_window[::-1]
    tail_floor = 10**np.floor(np.log10(max(min_power, 1e-14)))
    styles = (("input", "Masked input", "#666666", ":"),
              ("linear", "Matched linear output", "#CC8844", "--"),
              ("nonlinear", "Nonlinear output", "#4477AA", "-"))
    views = (
        ("time", "Pulse centre", "Time (ps)", "All-mode power (kW)", (-2, 2), (0, 1.55*max_power)),
        ("wavelength", "Spectrum", "Wavelength (nm)", "Power per frequency bin (W)",
         wavelength_window, (0, 1.55*max_spectrum)),
        ("tails", "Full time window", "Time (ps)", "All-mode power (kW)",
         (time[0], time[-1]), (tail_floor, max_power*50)),
    )
    exports = []
    for view, title, xlabel, ylabel, xlim, ylim in views:
        fig, axes = plt.subplots(1, 2, figsize=(11, 6), layout="constrained", sharex=True, sharey=True)
        for ax, method, source in zip(axes, ("BO", "GD"), sources):
            for key, label, color, linestyle in styles:
                power, spectrum = curves[method, key]
                x = wavelength[spectral_order] if view == "wavelength" else time
                y = spectrum if view == "wavelength" else power
                if view == "tails":
                    y = np.maximum(y, 1e-14)
                ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.5)
            if view == "tails":
                ax.set_yscale("log")
            name = "LLAMBO" if method == "BO" else "GD"
            ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                   title=f"{name} · {title}\n"
                         rf"$D_{{\mathrm{{NL}}}}$ = {source['D_NL']:.6f} · brightness = {source['brightness_ratio']:.2%}")
            ax.tick_params(labelleft=True)
            if view == "tails":
                ax.legend(loc="upper left", bbox_to_anchor=(.04,.94), frameon=False)
            else:
                ax.legend(loc="upper right", bbox_to_anchor=(.98,.94), frameon=False)
        stem = run.directory/f"bo_gd_pulses_{view}"
        with plt.rc_context({"savefig.bbox": "tight", "savefig.pad_inches": .08}):
            _finish(fig, stem)
        # Export each method's graph as well as the side-by-side notebook figure.
        renderer = fig.canvas.get_renderer()
        for ax, method in zip(axes, ("llambo", "gd")):
            bounds = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(.05)
            for suffix in ("png", "pdf", "svg"):
                fig.savefig(run.directory/f"{method}_pulses_{view}.{suffix}", dpi=160, bbox_inches=bounds)
        exports.append(str(stem.with_suffix(".png")))
        display(fig); plt.close(fig)
    pd.DataFrame(time_table).to_csv(run.directory/"bo_gd_pulses_time.csv", index=False)
    pd.DataFrame(spectrum_table).to_csv(run.directory/"bo_gd_pulses_wavelength.csv", index=False)
    save(run.directory/"bo_gd_pulses_sources.json", {
        "sources": sources, "new_GNLSE_propagations": 0,
        "mode_aggregation": "Sum of power over all three modes.",
        "input": "Each method's selected masked launch, without renormalization.",
        "outputs": "Matched linear and nonlinear outputs from the same selected fine propagation check as the input.",
        "layout": "LLAMBO left, GD right; shared axes for each view.",
        "figures": exports,
        "time_alignment": "None; original saved time coordinates.",
        "spectrum": "fftshift(ifft(field)); squared amplitudes summed over modes. Frequency-bin power plotted versus wavelength, not a density per nm."})
    evaluations = "; ".join(f"{'LLAMBO' if r['method'] == 'BO' else r['method']} evaluation {r['evaluation']}" for r in sources)
    display(Markdown(
        f"**Saved pulse comparison:** {evaluations}, both at **dz = {float(reference['dz_m'])*1000:g} mm**, "
        f"Nt = {int(reference['Nt'])}. Each input is that method's **selected shaped pulse**; "
        "the dashed curve is its **matched linear output** and the solid curve its **nonlinear output**. "
        "LLAMBO appears on the left and GD on the right, on identical axes. Powers sum all three modes, with no "
        "renormalization or temporal alignment. The wavelength panels retain power per equal frequency bin, "
        "not power per nm. The input and matched linear spectra overlap on this scale. "
        "These pulse shapes do not display the spectral phase used by D_NL. "
        "Failed refinement checks still leave the optimization results preliminary."))
