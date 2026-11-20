"""Measured simulation, process, storage and telemetry costs of a GD continuation."""
from datetime import datetime, timezone
import json

from .spectral_mask_gd import save
from .spectral_mask_gd_report import _events, _records, _style, _finish


def file_bytes(directory):
    files = [p for p in directory.rglob("*") if p.is_file() and not p.is_symlink()]
    return {"logical_bytes":sum(p.stat().st_size for p in files),
            "allocated_bytes":sum(p.stat().st_blocks*512 for p in files), "files":len(files)}


def show_run_costs(run):
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display, Markdown
    _style()
    state = run.result
    inherited = state["continuation"]["inherited_objective_pairs"]
    records = [r for r in _records(run.directory) if r["evaluation"] > inherited]
    events = [e for e in _events(run.directory) if e.get("segment") == "continuation"]
    results = {e["evaluation"]:e for e in events if e["kind"]=="objective_result" and not e["cached"]}
    rows = []
    for record in records:
        metrics = record.get("metrics",{})
        role = results.get(record["evaluation"],{}).get("role","failed or unfinished pair")
        rows.append({"evaluation":record["evaluation"],"role":role,"status":record["status"],
            "objective_pairs":1, "linear_GNLSE":metrics.get("linear_solves",0),
            "nonlinear_GNLSE":metrics.get("nonlinear_solves",0), "pair_seconds":record.get("wall_seconds",0.),
            "linear_seconds":metrics.get("linear_seconds",0.),"nonlinear_seconds":metrics.get("nonlinear_seconds",0.)})
    ledger = pd.DataFrame(rows)
    ledger.to_csv(run.directory/"new_evaluation_costs.csv",index=False)
    if rows:
        phase = ledger.groupby("role",sort=False).sum(numeric_only=True).drop(columns="evaluation")
        display(phase.round(3))
        phase.to_csv(run.directory/"cost_by_evaluation_role.csv")
    pair_seconds = sum(r["pair_seconds"] for r in rows)
    linear = sum(r["linear_seconds"] for r in rows)
    nonlinear = sum(r["nonlinear_seconds"] for r in rows)
    active = state["continuation_active_seconds"]
    clocks = {"additional_active_seconds":active,"additional_calendar_seconds":state["continuation_calendar_seconds"],
        "matched_pair_seconds":pair_seconds,"linear_solver_seconds":linear,"nonlinear_solver_seconds":nonlinear,
        "pair_setup_and_IO_seconds":pair_seconds-linear-nonlinear,
        "controller_and_other_seconds":active-pair_seconds,
        "cumulative_GD_active_seconds":state["total_wall_seconds"]}
    display(pd.DataFrame([{"component":k.replace("_seconds",""),"measured minutes":v/60}
                          for k,v in clocks.items()]).round(4))
    cost_path = run.directory/"cost_summary.json"
    cpu = json.loads(cost_path.read_text()) if cost_path.exists() else {"status":"not_measured"}
    gpu = json.loads((run.directory/"resource_summary.json").read_text())
    storage = file_bytes(run.directory)
    new_storage = [file_bytes(run.directory/"gnlse"/f"evaluation_{r['evaluation']:03d}") for r in records]
    storage["new_evaluation_artifact_bytes"] = sum(r["logical_bytes"] for r in new_storage)
    storage["measurement_note"] = "Snapshot when this reporting cell runs; excludes symlink targets and later notebook/HTML exports. Logical and allocated sizes differ; inherited raw fields are shared with the parent."
    report = {"reported_utc":datetime.now(timezone.utc).isoformat(),"runtime":clocks,
        "new_objective_pairs_attempted":len(records),
        "new_objective_pairs_completed":sum(r["status"]=="ok" for r in records),
        "new_objective_pairs_failed":sum(r["status"]=="failed" for r in records),
        "new_linear_GNLSE":sum(r["linear_GNLSE"] for r in rows),
        "new_nonlinear_GNLSE":sum(r["nonlinear_GNLSE"] for r in rows),
        "new_cache_hits":sum(e["kind"]=="objective_result" and e.get("cached",False) for e in events),
        "optimizer_AI_calls":0,"optimizer_AI_overhead_seconds":0.,
        "CPU_and_memory":cpu,"GPU":gpu,"storage":storage,
        "energy_kWh":None,"electricity_cost":None,
        "unmeasured":"Electrical energy, electricity tariff/cost, process I/O bytes and utilization/memory history before CPU sampling began.",
        "timing_note":"Solver timers may include first-call compilation; setup/compilation and file I/O are not independently profiled. CPU seconds measure processor work and are not additive to elapsed time."}
    save(run.directory/"run_costs.json",report)
    gpu_value = ("unavailable" if gpu["device_utilization_percent_mean"] is None else
                 f"{gpu['device_utilization_percent_mean']:.2f} / {gpu['device_utilization_percent_max']}")
    resource_rows = [
        {"quantity":"New objective pairs", "value":len(records), "basis":"measured; failed attempts included"},
        {"quantity":"New GNLSE propagations", "value":report["new_linear_GNLSE"]+report["new_nonlinear_GNLSE"],"basis":"recorded completed linear + nonlinear solves"},
        {"quantity":"Cache hits", "value":report["new_cache_hits"],"basis":"no new GNLSE pair"},
        {"quantity":"Optimizer AI calls / AI overhead (s)","value":"0 / 0","basis":"no LLM called by GD"},
        {"quantity":"New evaluation artifacts (GiB)","value":storage["new_evaluation_artifact_bytes"]/2**30,"basis":"measured logical bytes; excludes inherited raw fields"},
        {"quantity":"Branch files including metadata/reports (GiB)","value":storage["logical_bytes"]/2**30,"basis":"measured at this cell; symlink targets excluded"},
        {"quantity":"System GPU mean / maximum (%)","value":gpu_value,"basis":"system-wide samples; includes other applications"},
        {"quantity":"Solver GPU compute","value":"not used","basis":"CPU backend; code-path inference"},
        {"quantity":"Energy (kWh) / electricity cost","value":"not measured","basis":"no power measurement or tariff"},
    ]
    if "observed_process_CPU_seconds" in cpu:
        resource_rows[4:4] = [
            {"quantity":"Observed process CPU time (s)","value":cpu["observed_process_CPU_seconds"],"basis":"cumulative OS counters, including before telemetry; possible lower bound"},
            {"quantity":"Observed CPU core-hours","value":cpu["observed_CPU_core_hours"],"basis":"CPU seconds / 3600"},
            {"quantity":"Sampled CPU mean (% of one core)","value":cpu["sampled_CPU_percent_one_core_100_time_weighted_mean"],"basis":"100% = one logical CPU; measured sampling interval only"},
            {"quantity":"Sampled CPU mean (% of machine)","value":cpu["sampled_CPU_percent_machine_capacity_time_weighted_mean"],"basis":f"normalized by {cpu['logical_cpus']} logical CPUs"},
            {"quantity":"Sampled peak process RSS (GiB)","value":cpu["sampled_peak_process_tree_RSS_bytes"]/2**30,"basis":"earlier peaks unknown; shared pages may be counted twice"},
            {"quantity":"CPU cost monitor's own CPU time (s)","value":cpu["cost_monitor_CPU_seconds"],"basis":"separate monitoring process; not added to solver CPU"},
        ]
    resource_table = pd.DataFrame(resource_rows)
    resource_table["value"] = resource_table["value"].map(lambda x:f"{x:.4f}" if isinstance(x,float) else x)
    display(resource_table)
    resource_table.to_csv(run.directory/"resource_costs.csv",index=False)
    if "observed_process_CPU_seconds" in cpu:
        samples = pd.read_json(run.directory/"cost_usage.jsonl",lines=True)
        samples.drop(columns=["processes","errors"]).to_csv(run.directory/"cost_usage.csv",index=False)
        device = pd.read_json(run.directory/"resource_usage.jsonl",lines=True)
        fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout="constrained",sharex=True)
        start = cpu["first_sample_seconds_since_controller_start"]/60
        for ax in axes[:2]:
            ax.axvspan(0,start,color="#EEEEEE",label="Not sampled")
            ax.legend(loc="lower left",frameon=False)
        axes[0].plot(samples.seconds_since_controller_start/60,samples.CPU_percent_one_core_100,color="#4477AA")
        axes[0].set(ylabel="Process CPU (%; 100 = one core)",title="CPU utilization",ylim=(0,max(110,samples.CPU_percent_one_core_100.max()*1.12)))
        axes[1].plot(samples.seconds_since_controller_start/60,samples.process_tree_RSS_bytes/2**30,color="#4477AA")
        axes[1].set(ylabel="Process RSS (GiB)",title="Sampled resident memory",ylim=(0,samples.process_tree_RSS_bytes.max()/2**30*1.15))
        if "device_utilization_percent" in device:
            axes[2].plot(device.elapsed_seconds/60,device.device_utilization_percent,color="#CC8844")
        axes[2].set(ylabel="Device GPU utilization (%)",title="System-wide GPU",ylim=(0,105))
        for ax in axes:ax.set(xlabel="Continuation elapsed time (min)",xlim=(0,state["continuation_calendar_seconds"]/60))
        _finish(fig,run.directory/"resource_costs")
        display(fig);plt.close(fig)
        display(Markdown(
            f"CPU utilization and memory sampling began **{start:.2f} minutes** into the continuation "
            f"({cpu['first_sample_utc']}). The cumulative CPU counters include earlier work by these live "
            "processes, but earlier utilization intervals and memory peaks are unavailable. "
            "The shaded portions are unsampled. GPU telemetry began with the continuation and includes other applications."))
    print(report["timing_note"])
    print("Full cost ledger:",run.directory/"run_costs.json")
    return report
