"""Evaluator examples. Replace this interface to use a different simulator."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time
import tomllib


class AnalyticDemo:
    """A labelled two-control mathematical test, NOT a fiber simulation."""
    identity = {"kind": "analytic demonstration", "version": 1}

    def __call__(self, p, resolution, directory, timeout_seconds):
        x, phase = p["x"], p["phase"]
        # A smooth valley whose amplitude optimum lies outside the initial range.
        score = 0.08 + 0.10*(x-1.35)**2 + 0.025*(1-math.cos(phase-0.4))
        # Deterministic discretization term to exercise numerical refinement checks.
        score += resolution["dz"]**4 * 1e-3 * x
        return {"score": score, "metrics": {"brightness_ratio": 0.95-0.04*x,
                "example_only": True}, "checks": {"valid": True}}


def write_toml(path, data):
    """The worker protocol uses only scalars/arrays and nested string-keyed tables."""
    lines = []
    def value(x):
        if isinstance(x, bool): return "true" if x else "false"
        if isinstance(x, str): return json.dumps(x)
        if isinstance(x, (list, tuple)): return "["+", ".join(value(v) for v in x)+"]"
        if isinstance(x, (int, float)) and math.isfinite(x): return repr(x)
        raise ValueError(f"Unsupported TOML protocol value: {type(x).__name__}")
    def table(d, prefix=()):
        if prefix: lines.append("["+".".join(json.dumps(v) for v in prefix)+"]")
        for key, v in d.items():
            if not isinstance(v, dict) and v is not None:
                lines.append(json.dumps(key)+" = "+value(v))
        for key, v in d.items():
            if isinstance(v, dict):
                lines.append(""); table(v, prefix+(key,))
    table(data)
    Path(path).write_text("\n".join(lines)+"\n")


class OpticsEvaluator:
    """Persistent Julia worker; no notebook execution and no core package edits."""
    def __init__(self, workspace, setup, *, launch="power", objective="D_IO",
                 brightness_reference="baseline_nonlinear", julia=None):
        self.workspace = Path(workspace).resolve()
        self.project = self.workspace/"FastDifferentiableMMNLO"
        self.worker = Path(__file__).with_name("optics_worker.jl").resolve()
        self.setup, self.launch, self.objective = dict(setup), launch, objective
        self.brightness_reference = brightness_reference
        self.julia = str(julia or os.environ.get("JULIA") or "julia")
        if objective not in ("D_IO", "D_NL") or launch not in ("power", "auxiliary"):
            raise ValueError("Unknown optics preset")
        if brightness_reference not in ("baseline_nonlinear", "matched_linear"):
            raise ValueError("Unknown brightness reference")
        h = hashlib.sha256()
        for p in sorted((self.project/"src").rglob("*.jl")):
            h.update(str(p.relative_to(self.project)).encode()); h.update(p.read_bytes())
        self.identity = {"kind": "MMGNLSE", "setup": self.setup, "launch": launch,
                         "objective": objective, "brightness_reference": brightness_reference,
                         "worker_sha256": hashlib.sha256(self.worker.read_bytes()).hexdigest(),
                         "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "source_sha256": h.hexdigest(),
                         "manifest_sha256": hashlib.sha256((self.project/"Manifest.toml").read_bytes()).hexdigest()}
        self.process = None
        self.responses = queue.Queue()
        self.log = None

    def _start(self, directory):
        environment = os.environ.copy()
        self.log = (Path(directory).parent/"julia_worker.log").open("a")
        self.process = subprocess.Popen([self.julia, f"--project={self.project}", "--startup-file=no", str(self.worker)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, bufsize=1, env=environment, start_new_session=True)
        def listen():
            for line in self.process.stdout:
                if line.startswith("RESULT\t"):
                    self.responses.put(line.strip().split("\t", 1)[1])
                else:
                    self.log.write(line); self.log.flush()
            self.responses.put(None)
        self.reader = threading.Thread(target=listen, daemon=True)
        self.reader.start()

    def __call__(self, parameters, resolution, directory, timeout_seconds):
        directory = Path(directory).resolve()
        request, response = directory/"request.toml", directory/"response.toml"
        write_toml(request, {"setup": self.setup, "parameters": parameters, "resolution": resolution,
                            "launch": self.launch, "objective": self.objective,
                            "brightness_reference": self.brightness_reference,
                            "artifact_directory": str(directory), "response_path": str(response)})
        if self.process is None or self.process.poll() is not None:
            self.responses = queue.Queue()
            self._start(directory)
        try:
            self.process.stdin.write(str(request)+"\n"); self.process.stdin.flush()
            answer = self.responses.get(timeout=max(1, timeout_seconds))
            if answer != str(response) or not response.is_file():
                raise RuntimeError("Julia worker exited before returning this evaluation")
        except BaseException:
            self.close()
            raise
        result = tomllib.loads(response.read_text())
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Julia evaluation failed"))
        artifacts = {"fields": str(directory/"fields.npz"), "request": str(request), "response": str(response)}
        artifacts["fields_sha256"] = hashlib.sha256((directory/"fields.npz").read_bytes()).hexdigest()
        return {"score": result["score"], "metrics": result["metrics"], "checks": result["checks"], "artifacts": artifacts}

    def close(self):
        if self.process and self.process.poll() is None:
            import signal
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL); self.process.wait()
        if hasattr(self, "reader"):
            self.reader.join(timeout=2)
        if self.log and not self.log.closed:
            self.log.close()
        self.process = None


FIBER_SETUP = {
    "time_window_ps": 10.0, "pulse_fwhm_ps": 0.2, "fiber_length_m": 2.86,
    "carrier_m": 1379.2e-9, "core_radius_m": 52.5e-6, "NA": 0.1, "num_modes": 3,
    "beta_order": 5, "cp_target_error": 1e-3, "cp_seed": int(0x4d4d474e),
    "reference_power_W": 100_000.0, "fixed_mode1_W": 100_000.0,
    "brightness_reference_dz_m": 0.001,
}


def plot_optical_output(evaluation, output_directory=None):
    """Paired time/spectrum figures from saved fields; never reruns propagation."""
    import numpy as np
    import matplotlib.pyplot as plt
    data = np.load(evaluation["artifacts"]["fields"])
    fields = [("Input", data["input"], "#777777", ":"),
              ("Linear output", data["linear"], "#cf7c32", "--"),
              ("Nonlinear output", data["nonlinear"], "#366b9e", "-")]
    t, wavelength = data["time_ps"], data["wavelength_nm"]
    good = np.where(np.isfinite(wavelength) & (wavelength > 0))[0]
    good = good[np.argsort(wavelength[good])]
    figures = []
    for kind in ("time", "spectrum"):
        fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
        curves = []
        x = t if kind == "time" else wavelength[good]
        for label, field, color, style in fields:
            y = np.abs(field[:, 0, 0])**2/1000 if kind == "time" else np.abs(np.fft.fftshift(np.fft.ifft(field, axis=0), axes=0)[:, 0, 0])[good]**2
            curves.append(y)
            ax.plot(x, y, label=label, color=color, linestyle=style)
        support = np.flatnonzero(np.max(np.array([y/max(y.max(), 1e-100) for y in curves]), axis=0) > 1e-5)
        if len(support) > 1:
            left, right = x[support[0]], x[support[-1]]
            ax.set_xlim(left-.1*(right-left), right+.1*(right-left))
        ax.set_xlabel("Time (ps)" if kind == "time" else "Wavelength (nm)")
        ax.set_ylabel("LP01 power (kW)" if kind == "time" else "Power per frequency bin (W)")
        ax.set_title(f"Evaluation {evaluation['id']}: score {evaluation['score']:.6f}")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        if output_directory:
            target = Path(output_directory); target.mkdir(parents=True, exist_ok=True)
            fig.savefig(target/f"evaluation_{evaluation['id']}_{kind}.png", dpi=160)
        figures.append(fig)
    return figures
