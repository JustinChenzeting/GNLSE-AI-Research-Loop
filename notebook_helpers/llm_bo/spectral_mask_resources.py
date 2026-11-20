"""Lightweight macOS GPU telemetry, kept separate from the optical evaluator."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import statistics
import subprocess
import threading
import time


def gpu_sample():
    """Device-wide counters include all applications; they are not process attribution."""
    raw = subprocess.check_output(["/usr/sbin/ioreg","-r","-c","IOAccelerator","-d","1","-l","-w","0"],
                                  text=True, timeout=5, stderr=subprocess.PIPE)
    row = {}
    model = re.search(r'"model"\s*=\s*"([^"]+)"',raw)
    if model:
        row["gpu_model"] = model.group(1)
    for source, target in (("Device Utilization %","device_utilization_percent"),
                           ("Renderer Utilization %","renderer_utilization_percent"),
                           ("Tiler Utilization %","tiler_utilization_percent"),
                           ("In use system memory","gpu_system_memory_bytes")):
        match = re.search('"'+re.escape(source)+r'"\s*=\s*(\d+)',raw)
        if match:
            row[target] = int(match.group(1))
    if "device_utilization_percent" not in row:
        raise RuntimeError("Device utilization counter unavailable.")
    return row


class ResourceMonitor:
    def __init__(self, directory, *, started=None, interval=5.):
        self.directory, self.started, self.interval = Path(directory), started, interval
        self.stopping = threading.Event()
        self.samples = []

    def elapsed(self):
        if self.started is not None:
            return time.monotonic()-self.started
        path = self.directory/"state.json"
        state = json.loads(path.read_text())
        return state["total_wall_seconds"]+max(0,time.time()-path.stat().st_mtime)

    def sample(self):
        row = {"timestamp_utc":datetime.now(timezone.utc).isoformat(), "elapsed_seconds":self.elapsed()}
        try:
            row.update(status="measured", **gpu_sample())
        except Exception as error:
            row.update(status="unavailable", error=f"{type(error).__name__}: {error}")
        self.samples.append(row)
        with (self.directory/"resource_usage.jsonl").open("a") as handle:
            handle.write(json.dumps(row)+"\n")

    def summarize(self):
        from .spectral_mask_gd import save
        path = self.directory/"resource_usage.jsonl"
        all_rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        values = [r["device_utilization_percent"] for r in all_rows if r["status"]=="measured"]
        summary = {"solver_backend":"CPU", "solver_GPU_compute":"not used",
            "backend_basis":"spectral_mask_worker.jl calls solve_mmgnlse with default backend=:cpu; CPU FFTW and BLAS each use one thread",
            "solver_GPU_utilization_percent_estimate":0., "estimate_basis":"code-path inference, not per-process GPU telemetry",
            "telemetry_scope":"system-wide Apple GPU; includes display and other applications",
            "sample_interval_seconds":self.interval, "samples":len(all_rows), "measured_samples":len(values),
            "coverage_start_seconds":all_rows[0]["elapsed_seconds"] if all_rows else None,
            "coverage_end_seconds":all_rows[-1]["elapsed_seconds"] if all_rows else None,
            "elapsed_basis":"direct monotonic clock" if self.started is not None else "latest campaign checkpoint plus time since its write",
            "gpu_model":next((r.get("gpu_model") for r in all_rows if r.get("gpu_model")),None),
            "device_utilization_percent_mean":statistics.mean(values) if values else None,
            "device_utilization_percent_max":max(values) if values else None,
            "device_utilization_percent_min":min(values) if values else None,
            "note":"Telemetry started when requested during the current pilot; earlier GPU activity is not reconstructed. Future runs start the monitor with the campaign." if self.started is None else "Telemetry begins with this campaign."}
        save(self.directory/"resource_summary.json",summary)
        return summary

    def _run(self):
        while not self.stopping.is_set():
            self.sample()
            self.summarize()
            self.stopping.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stopping.set()
        self.thread.join(timeout=7)
        return self.summarize()


if __name__ == "__main__":
    import sys
    directory = Path(sys.argv[1]).resolve()
    monitor = ResourceMonitor(directory).start()
    print("Recording system-wide GPU telemetry to:",directory/"resource_usage.jsonl",flush=True)
    try:
        while True:
            state = json.loads((directory/"state.json").read_text())
            if state["status"] in ("completed","failed") or monitor.elapsed() > state["descent_config"]["max_wall_seconds"]+15:
                break
            time.sleep(5)
    finally:
        result = monitor.stop()
        print(json.dumps(result,indent=2),flush=True)
