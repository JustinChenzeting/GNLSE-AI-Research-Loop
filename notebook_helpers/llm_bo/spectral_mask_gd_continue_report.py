"""Additional and cumulative costs of a saved GD continuation."""
import json
from pathlib import Path

from .spectral_mask_gd import save
from .spectral_mask_gd_report import _records


def show_continuation_summary(run):
    import pandas as pd
    from IPython.display import display, Markdown
    result = run.result
    continuation = result["continuation"]
    records = _records(run.directory)
    parent = json.loads((run.directory/"parent_snapshot/state.json").read_text())
    inherited = continuation["inherited_objective_pairs"]
    old = [r for r in records if r["evaluation"] <= inherited]
    new = [r for r in records if r["evaluation"] > inherited]
    audit = run.directory/"parent_snapshot/timing_audit.json"
    parent_calendar = json.loads(audit.read_text())["calendar_timestamp_span_seconds"] if audit.exists() else parent.get("calendar_elapsed_seconds")
    new_calendar = result["continuation_calendar_seconds"]
    rows = []
    for label, measurements, steps, active, calendar in (
        ("Preserved pilot", old, len(parent["accepted_steps"]), parent["total_wall_seconds"], parent_calendar),
        ("This continuation", new, len(result["accepted_steps"])-len(parent["accepted_steps"]),
         result["continuation_active_seconds"], new_calendar),
        ("Combined GD", records, len(result["accepted_steps"]), result["total_wall_seconds"],
         None if parent_calendar is None else parent_calendar+new_calendar)):
        rows.append({"segment": label, "accepted steps": steps, "objective pairs": len(measurements),
            "linear GNLSE": sum(r.get("metrics",{}).get("linear_solves",0) for r in measurements),
            "nonlinear GNLSE": sum(r.get("metrics",{}).get("nonlinear_solves",0) for r in measurements),
            "active minutes": active/60, "calendar minutes": None if calendar is None else calendar/60})
    table = pd.DataFrame(rows)
    display(table.round(3))
    table.to_csv(run.directory/"continuation_costs.csv", index=False)
    events = [json.loads(line) for line in (run.directory/"events.jsonl").read_text().splitlines()]
    new_events = [e for e in events if e.get("segment") == "continuation"]
    summary = {"costs": rows, "parent_iteration": continuation["parent_iteration"],
               "total_iteration_limit": continuation["total_iteration_limit"],
               "last_attempted_iteration": result.get("last_attempted_iteration"),
               "new_cache_hits": sum(e["kind"]=="objective_result" and e.get("cached",False) for e in new_events),
               "new_rejected_proposals": sum(e["kind"]=="step_rejected" for e in new_events),
               "start_D_NL": continuation["start_D_NL"], "end_D_NL": result["best_search"]["D_NL"],
               "start_brightness": continuation["start_brightness"],
               "end_brightness": result["best_search"]["brightness_ratio"],
               "stop_reason": result["stop_reason"],
               "validation": result.get("validation",{}).get("status","not_run")}
    save(run.directory/"continuation_summary.json", summary)
    display(Markdown(
        f"Continued from accepted step **{continuation['parent_iteration']}**, evaluation "
        f"**{continuation['start_evaluation']}**. Search-grid D_NL: "
        f"**{summary['start_D_NL']:.6f} → {summary['end_D_NL']:.6f}**; brightness: "
        f"**{100*summary['start_brightness']:.4f}% → {100*summary['end_brightness']:.4f}%**. "
        f"New cache hits: **{summary['new_cache_hits']}**. Stop: **{summary['stop_reason']}**."))
    last_iteration = result.get("last_attempted_iteration")
    decisions = [e for e in new_events if e.get("iteration") == last_iteration
                 and e["kind"] in ("step_accepted", "step_rejected")]
    if decisions:
        display(pd.DataFrame([{"iteration": e["iteration"], "backtrack": e["backtrack"],
            "evaluation": e.get("evaluation"), "D_NL": e.get("D_NL"),
            "brightness": e.get("brightness_ratio"), "decision": e["reason"]} for e in decisions]).round(6))
    if result["stop_reason"] == "backtracking reached the factor-grid resolution":
        display(Markdown(
            "**Why this continuation stopped early:** the line search rejected the tested moves, "
            "then the next smaller step rounded back to the current mask. That final proposal "
            "needed no GNLSE solve. The iteration cap was not reached. This is a stall along "
            "the tested projected direction; constrained optimality remains untested."))
    print("Calendar totals add the two sessions and exclude the gap between them.")
    return summary
