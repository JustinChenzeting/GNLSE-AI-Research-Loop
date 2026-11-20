"""Polynomial validation error and the proposal loop for the LLAMBO notebook."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from llm_client import check_point, check_points, measured_examples, object_schema, points_schema, write_json


def fit_curve(degree, x_train, y_train):
    """Fit one polynomial; the ridge penalty is fixed, not optimized."""
    design = np.polynomial.polynomial.polyvander(x_train, degree)
    penalty = 1e-4 * np.diag([0] + [1] * degree)
    return np.linalg.solve(design.T @ design + penalty, design.T @ y_train)


def evaluate(parameters, x_train, y_train, x_valid, y_valid):
    """Run the real experiment and return its validation error (lower is better)."""
    coefficients = fit_curve(parameters["degree"], x_train, y_train)
    predicted = np.polynomial.polynomial.polyval(x_valid, coefficients)
    return float(np.mean((predicted - y_valid) ** 2))

def suggest_starting_points(problem, ask_llm, count=3):
    """Suggest distinct initial settings from the problem description."""
    request = {
        "operation": "warm_start",
        "instruction": f"Suggest exactly {count} distinct, promising starting settings.",
        "problem": problem,
    }
    reply = ask_llm(request, points_schema(problem))
    return check_points(problem, reply, count)

def suggest_candidates(problem, history, ask_llm, count=4, alpha=0.01):
    """Propose new settings aiming slightly below the best observed score."""
    scores = [row["score"] for row in history]
    target = min(scores) - alpha * (max(scores) - min(scores))
    request = {
        "operation": "propose_candidates",
        "instruction": (
            f"Suggest exactly {count} distinct settings aiming for the target score. "
            "Do not repeat any already measured setting. Keep every value in range."
        ),
        "problem": problem,
        "observations": measured_examples(history),
        "target_score": target,
    }
    reply = ask_llm(request, points_schema(problem))
    return check_points(problem, reply, count, history)

def predict_scores(problem, history, candidates, ask_llm, rng, repeats=3):
    """Estimate candidate means and uncertainty from repeated LLM predictions."""
    if repeats < 2:
        raise ValueError("Use at least two predictions to estimate uncertainty.")
    ids = [str(index) for index in range(len(candidates))]
    schema = object_schema({key: {"type": "number"} for key in ids})
    predictions = []
    for sample in range(repeats):
        shuffled = [history[index] for index in rng.permutation(len(history))]
        request = {
            "operation": "predict_scores",
            "instruction": "Predict the score for each candidate ID. Return one number per ID.",
            "problem": problem,
            "observations": measured_examples(shuffled),
            "candidates": dict(zip(ids, candidates)),
            "prediction_sample": sample + 1,
        }
        reply = ask_llm(request, schema)
        if not isinstance(reply, dict) or set(reply) != set(ids):
            raise ValueError("Prediction IDs do not match the candidates.")
        values = [reply[key] for key in ids]
        if any(type(value) not in (int, float) or not np.isfinite(value) for value in values):
            raise ValueError("Every predicted score must be a finite number.")
        predictions.append(values)
    return np.mean(predictions, axis=0), np.std(predictions, axis=0, ddof=1)

def expected_improvement(mean, std, best_score):
    """Expected improvement for minimization under a Gaussian approximation."""
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if mean.shape != std.shape or not np.isfinite([mean, std]).all() or np.any(std < 0):
        raise ValueError("Means and standard deviations must be finite and match in shape.")
    improvement = best_score - mean
    z = np.divide(improvement, std, out=np.zeros_like(mean), where=std > 0)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.where(std > 0, np.maximum(ei, 0), np.maximum(improvement, 0))

def run_optimization(problem, evaluate, ask_llm, run_dir, *,
                     n_initial=3, n_steps=2, n_candidates=4, repeats=3, seed=7):
    """Warm-start, propose, predict, select, measure, and save."""
    counts = (n_initial, n_steps, n_candidates, repeats)
    if any(type(count) is not int for count in counts):
        raise ValueError("Experiment and prediction counts must be integers.")
    if n_initial < 2 or n_steps < 0 or n_candidates < 1 or repeats < 2:
        raise ValueError("Use >=2 starting points, >=0 steps, >=1 candidate, >=2 predictions.")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "history.json").exists():
        raise FileExistsError("Choose a fresh run directory to preserve earlier results.")
    rng = np.random.default_rng(seed)
    history = []

    def measure(point, source):
        point = check_point(problem, point)
        started = time.monotonic()
        score = evaluate(dict(point))
        if isinstance(score, (bool, np.bool_)) or not np.isscalar(score) or not np.isfinite(score):
            raise ValueError("The evaluator must return a finite scalar score.")
        row = {"trial": len(history) + 1, "parameters": point,
               "score": float(score), "source": source,
               "seconds": time.monotonic() - started}
        history.append(row)
        write_json(run_dir / "history.json", history)
        print(f"Trial {row['trial']}: {point} -> {score:.5f}", flush=True)

    initial = suggest_starting_points(problem, ask_llm, n_initial)
    for point in initial:
        measure(point, "warm_start")

    for step in range(n_steps):
        candidates = suggest_candidates(problem, history, ask_llm, n_candidates)
        mean, std = predict_scores(problem, history, candidates, ask_llm, rng, repeats)
        ei = expected_improvement(mean, std, min(row["score"] for row in history))
        chosen = int(np.argmax(ei))
        write_json(run_dir / f"step_{step + 1:02d}.json", {
            "candidates": candidates, "predicted_mean": mean.tolist(),
            "predicted_std": std.tolist(), "expected_improvement": ei.tolist(),
            "chosen_index": chosen,
        })
        measure(candidates[chosen], "llambo")
    return pd.DataFrame([
        {"trial": row["trial"], **row["parameters"], "score": row["score"], "source": row["source"]}
        for row in history
    ])
