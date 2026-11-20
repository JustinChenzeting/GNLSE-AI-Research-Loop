"""Example specifications. The loop itself contains no optics-specific controls."""
from pathlib import Path
import math
from loop import Control, Spec
from evaluators import AnalyticDemo, OpticsEvaluator, FIBER_SETUP


def analytic_demo(brightness_fraction=0.9, brightness_policy="soft"):
    evaluator = AnalyticDemo()
    spec = Spec(
        name="analytic demonstration", objective="illustrative smooth loss; not optical data",
        controls=[Control("x", 0., 1., max_change=.1, expansion=.25, domain_lower=0.),
                  Control("phase", -math.pi, math.pi, fd_step=.01, max_change=.1, expansion=0,
                          periodic=True, unit="rad")],
        baseline={"x": .6, "phase": 0.}, evaluator_identity=evaluator.identity,
        search={"dz": .1, "Nt": 64}, refinements={"dz": {"dz": .05, "Nt": 64},
                                                        "time": {"dz": .05, "Nt": 128}},
        brightness_fraction=brightness_fraction, brightness_policy=brightness_policy,
        learning_rate=1.0, robustness={"x": .02, "phase": .02},
        knowledge="This is a labelled mathematical demonstration with a smooth loss. Test observations; no fiber claims.")
    return spec, evaluator


def optics_preset(workspace, *, kind="dio_power", brightness_fraction, brightness_policy="soft", robustness=None):
    if kind == "dio_power":
        evaluator = OpticsEvaluator(workspace, FIBER_SETUP, launch="power", objective="D_IO",
                                    brightness_reference="baseline_nonlinear")
        controls = [Control("power_kW", 50., 150., scale=100., fd_step=.002, max_change=10.,
                            expansion=25., domain_lower=1e-6, unit="kW")]
        baseline = {"power_kW": 100.}
        objective = "LP01 D_IO = 1 - input/nonlinear-output complex fidelity; unchanged time samples"
        knowledge = "Matches dio_power_optimization.ipynb: 200 fs LP01 input, three propagated modes, 2.86 m silica step-index fiber. D_IO includes linear dispersion; reducing nonlinear interaction alone need not minimize D_IO. Brightness reference is the 100 kW nonlinear baseline at dz=1 mm, with matching Nt. It is not the matched linear reference."
    elif kind in ("lp01_auxiliary_dnl", "lp01_auxiliary_dio"):
        obj = "D_NL" if kind.endswith("dnl") else "D_IO"
        evaluator = OpticsEvaluator(workspace, FIBER_SETUP, launch="auxiliary", objective=obj,
                                    brightness_reference="matched_linear")
        controls = [Control(name, 0., 100., scale=100., transform="sqrt", fd_step=.005,
                            max_change=10., expansion=25., domain_lower=0., unit="kW") for name in ("P2_kW", "P3_kW")]
        controls += [Control(name, -math.pi, math.pi, fd_step=.01, max_change=.1, expansion=0.,
                             periodic=True, unit="rad") for name in ("phi2_rad", "phi3_rad")]
        baseline = {"P2_kW": 0., "P3_kW": 0., "phi2_rad": 0., "phi3_rad": 0.}
        objective = "LP01 "+obj+" = 1 - complex field fidelity; no alignment"
        knowledge = "Mode 1 is fixed at 100 kW and phase zero. Each auxiliary power starts in 0–100 kW; upper bounds are adaptive. Optimize normalized field amplitudes sqrt(P/100 kW) internally and phase in radians. A zero-power mode has an irrelevant phase: explore nonzero helper powers at different phases before inferring a minimum there. Brightness uses the matched linear LP01 reference."
    else:
        raise ValueError("Unknown optics example")
    spec = Spec(name=kind, objective=objective, controls=controls, baseline=baseline,
                evaluator_identity=evaluator.identity, search={"Nt": 2048, "dz_m": .01},
                refinements={"dz": {"Nt": 2048, "dz_m": .001},
                             "time": {"Nt": 4096, "dz_m": .001}},
                brightness_fraction=brightness_fraction, brightness_policy=brightness_policy,
                robustness=robustness or {}, knowledge=knowledge)
    return spec, evaluator
