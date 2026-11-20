"""Time and wavelength comparison from the original saved 100 kW runs."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
def compare_saved_waveforms(*, llm_directory, previous_case, output_directory, show=True):
    """Plot actual LP01 output powers; retain absolute amplitudes and original time axes."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    llm_directory, previous_case = Path(llm_directory), Path(previous_case)
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    state = json.loads((llm_directory / "state.json").read_text())
    fine = state["validation"]["selected_fine"]
    files = [
        ("Baseline", previous_case / "baseline_fields.npz", "#666666", ":"),
        ("Previous GD", previous_case / "selected_fields.npz", "#CC8844", "--"),
        ("LLM selected", llm_directory / "gnlse" / f"evaluation_{fine['evaluation']:03d}" / "fields.npz", "#4477AA", "-"),
    ]
    sources, extracted = [], []
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
             "legend.fontsize": 12, "axes.titlesize": 12, "font.weight": "normal",
             "axes.labelweight": "normal", "pdf.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
        reference_time = None
        for label, path, color, linestyle in files:
            with np.load(path, allow_pickle=False) as data:
                time, wavelength = data["time_ps"], data["wavelength_nm"]
                field = data["nonlinear"][:, 0, 0]
                if not np.isclose(float(data["power_W"]), 100000) or not np.isclose(float(data["dz_m"]), .001):
                    raise ValueError("Comparison requires saved 100 kW, 1 mm fields.")
                if reference_time is not None and not np.array_equal(time, reference_time):
                    raise ValueError("Saved field time grids differ.")
                reference_time = time.copy()
                power = np.abs(field)**2 / 1000
                spectrum = np.abs(np.fft.fftshift(np.fft.ifft(field)))**2
                if not np.isclose(spectrum.sum(), np.mean(np.abs(field)**2), rtol=1e-12):
                    raise ValueError("Spectrum normalization failed Parseval's identity.")
                good = np.flatnonzero(np.isfinite(wavelength) & (wavelength > 0))
                good = good[np.argsort(wavelength[good])]
                axes[0].plot(time, power, color=color, linestyle=linestyle, marker="", linewidth=1.5, label=label)
                axes[1].plot(wavelength[good], spectrum[good], color=color, linestyle=linestyle, marker="", linewidth=1.5, label=label)
                for i in range(len(time)):
                    extracted.append({"case": label, "sample": i, "time_ps": float(time[i]),
                        "power_kW": float(power[i]), "wavelength_nm": float(wavelength[i]),
                        "power_per_frequency_bin_W": float(spectrum[i])})
                sources.append({"case": label, "fields": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        axes[0].set(xlabel="Time (ps)", ylabel="LP01 output power (kW)", xlim=(-1, 1))
        axes[1].set(xlabel="Wavelength (nm)", ylabel="Power per frequency bin (W)", xlim=(1200, 1600))
        for letter, ax in zip(("a", "b"), axes):
            ax.set_box_aspect(1)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(top=False, right=False)
            ax.text(.025, .96, f"({letter})", va="top", transform=ax.transAxes)
            ax.legend(loc="upper left", bbox_to_anchor=(.015, .89), frameon=False)
        axes[1].set_xticks([1250, 1350, 1450, 1550])
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(target / f"waveforms.{suffix}", dpi=180)
        if show:
            plt.show()
    with (target / "waveforms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=extracted[0])
        writer.writeheader()
        writer.writerows(extracted)
    (target / "waveforms_sources.json").write_text(json.dumps({
        "sources": sources, "mode": "LP01", "resolution": "Nt=2048, dz=1 mm",
        "normalization": "Absolute time-domain power and frequency-bin power; no peak normalization or time alignment. "
                         "The spectral horizontal axis is wavelength; the ordinate is not power per nm.",
    }, indent=2) + "\n")
    return fig
