"""Read-only process-tree CPU/memory telemetry for one active optical campaign."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import time

import psutil

from .spectral_mask_gd import save


class CostMonitor:
    def __init__(self, directory, pid, interval=2.):
        self.directory, self.pid, self.interval = Path(directory), pid, interval
        self.root = psutil.Process(pid)
        self.root_created = self.root.create_time()
        self.registry, self.previous = {}, None
        self.started = time.time()
        self.logical_cpus = psutil.cpu_count()
        psutil.cpu_percent(interval=None)

    def sample(self):
        now = time.time()
        processes, errors = [], []
        root_alive = self.root.is_running() and self.root.create_time() == self.root_created
        targets = [self.root, *self.root.children(recursive=True)] if root_alive else []
        reaped_seconds = 0.
        root_seconds = 0.
        live_child_seconds = 0.
        for process in targets:
            try:
                cpu, memory = process.cpu_times(), process.memory_info()
                created = process.create_time()
                seconds = cpu.user + cpu.system
                row = {"pid": process.pid, "name": process.name(), "created_unix": created,
                       "user_cpu_seconds": cpu.user, "system_cpu_seconds": cpu.system,
                       "cpu_seconds": seconds, "rss_bytes": memory.rss, "vms_bytes": memory.vms,
                       "threads": process.num_threads()}
                processes.append(row)
                key = f"{process.pid}:{created}"
                self.registry[key] = row
                if process.pid == self.pid:
                    root_seconds = seconds
                    reaped_seconds = getattr(cpu,"children_user",0.)+getattr(cpu,"children_system",0.)
                else:
                    live_child_seconds += seconds
            except (psutil.Error, OSError) as error:
                errors.append({"pid":process.pid,"error":type(error).__name__})
        # Retain the last counters of exited children. Reaped-child counters, when
        # supported by this OS, can recover work between their last sample and exit.
        observed = sum(p["cpu_seconds"] for p in self.registry.values())
        observed = max(observed, root_seconds+reaped_seconds+live_child_seconds)
        utilization = None
        if self.previous:
            dt = now-self.previous["timestamp_unix"]
            if dt > 0:
                utilization = max(0.,100*(observed-self.previous["observed_cpu_seconds"])/dt)
        own = psutil.Process().cpu_times()
        row = {"timestamp_utc":datetime.fromtimestamp(now,timezone.utc).isoformat(), "timestamp_unix":now,
               "seconds_since_controller_start":now-self.root_created,
               "seconds_since_monitor_start":now-self.started, "controller_alive":root_alive,
               "observed_cpu_seconds":observed, "CPU_percent_one_core_100":utilization,
               "CPU_percent_machine_capacity":None if utilization is None else utilization/self.logical_cpus,
               "process_tree_RSS_bytes":sum(p["rss_bytes"] for p in processes),
               "system_CPU_percent":psutil.cpu_percent(interval=None),
               "system_memory_percent":psutil.virtual_memory().percent,
               "cost_monitor_CPU_seconds":own.user+own.system, "processes":processes, "errors":errors}
        with (self.directory/"cost_usage.jsonl").open("a") as handle:
            handle.write(json.dumps(row,allow_nan=False)+"\n")
        self.previous = row
        return row

    def summarize(self):
        rows = [json.loads(line) for line in (self.directory/"cost_usage.jsonl").read_text().splitlines()]
        first, last = rows[0], rows[-1]
        span = last["timestamp_unix"]-first["timestamp_unix"]
        rate = None if span <= 0 else 100*(last["observed_cpu_seconds"]-first["observed_cpu_seconds"])/span
        values = [r["CPU_percent_one_core_100"] for r in rows if r["CPU_percent_one_core_100"] is not None]
        summary = {"controller_pid":self.pid, "logical_cpus":self.logical_cpus,
                   "physical_cpus":psutil.cpu_count(logical=False), "sample_interval_seconds":self.interval,
                   "samples":len(rows), "first_sample_utc":first["timestamp_utc"],
                   "last_sample_utc":last["timestamp_utc"],
                   "first_sample_seconds_since_controller_start":first["seconds_since_controller_start"],
                   "observed_process_CPU_seconds":last["observed_cpu_seconds"],
                   "observed_CPU_core_hours":last["observed_cpu_seconds"]/3600,
                   "CPU_seconds_observed_at_monitor_start":first["observed_cpu_seconds"],
                   "sampled_CPU_percent_one_core_100_time_weighted_mean":rate,
                   "sampled_CPU_percent_machine_capacity_time_weighted_mean":None if rate is None else rate/self.logical_cpus,
                   "sampled_CPU_percent_one_core_100_max":max(values) if values else None,
                   "sampled_peak_process_tree_RSS_bytes":max(r["process_tree_RSS_bytes"] for r in rows),
                   "cost_monitor_CPU_seconds":last["cost_monitor_CPU_seconds"],
                   "process_counters":list(self.registry.values()),
                   "CPU_counter_basis":"OS cumulative user+system counters of the controller and descendants, retained after exit; includes their work before telemetry began. Treat as observed cumulative CPU time, possibly a lower bound because exits and short-lived children can occur between samples.",
                   "CPU_utilization_basis":"Change in cumulative process CPU seconds divided by elapsed calendar seconds. 100% means one logical CPU; machine capacity divides by logical CPU count. Interval measurements begin with telemetry, not campaign start.",
                   "memory_basis":"Sampled sum of process-tree resident memory. Sampling began partway through the campaign; earlier peaks are unknown. Shared pages can be counted more than once.",
                   "disk_IO_bytes":"unavailable from this macOS process interface",
                   "energy_kWh":None, "energy_basis":"not measured; CPU utilization is not electrical power",
                   "electricity_cost":None, "electricity_cost_basis":"energy use and tariff not measured"}
        save(self.directory/"cost_summary.json",summary)
        return summary

    def run(self):
        print(f"Recording CPU and memory for controller {self.pid} in {self.directory}",flush=True)
        try:
            while True:
                row = self.sample()
                self.summarize()
                state = json.loads((self.directory/"state.json").read_text())
                if not row["controller_alive"] or state["status"] in ("completed","failed"):
                    break
                time.sleep(self.interval)
        finally:
            summary = self.summarize()
            print(json.dumps({k:summary[k] for k in ("samples","observed_process_CPU_seconds",
                  "sampled_CPU_percent_one_core_100_time_weighted_mean","sampled_peak_process_tree_RSS_bytes")}),flush=True)
        return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--pid",type=int,required=True)
    parser.add_argument("--interval",type=float,default=2.)
    args=parser.parse_args()
    if args.interval <= 0:parser.error("interval must be positive")
    CostMonitor(args.directory,args.pid,args.interval).run()
