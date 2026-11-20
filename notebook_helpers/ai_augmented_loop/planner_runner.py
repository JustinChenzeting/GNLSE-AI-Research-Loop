"""Retry temporary Codex capacity failures while preserving the existing experiment."""
import json
import time

from codex_planner import CodexCapacityError


def _wait_for_retry(loop, proposal_id, seconds):
    """Charge waiting time to the failed attempt, including notebook interruptions."""
    row = loop.db.execute("SELECT metadata FROM proposals WHERE id=? AND campaign=?",
                          (proposal_id, loop.id)).fetchone()
    metadata = json.loads(row["metadata"] or "{}")
    metadata["capacity_retry"] = {"wait_seconds": seconds, "status": "waiting"}
    # Reserve the whole wait first, so a killed kernel cannot erase its cost.
    loop.db.execute("UPDATE proposals SET seconds=seconds+?,metadata=? WHERE id=?",
                    (seconds, json.dumps(metadata), proposal_id))
    loop.db.commit()
    loop._save("waiting for Codex")
    started, completed = time.monotonic(), False
    try:
        time.sleep(seconds)
        completed = True
    finally:
        elapsed = time.monotonic()-started
        metadata["capacity_retry"] = {"wait_seconds": elapsed,
                                      "status": "complete" if completed else "interrupted"}
        loop.db.execute("UPDATE proposals SET seconds=seconds-?+?,metadata=? WHERE id=?",
                        (seconds, elapsed, json.dumps(metadata), proposal_id))
        loop.db.commit()
        loop._save("ready" if completed else "interrupted")


def run_with_capacity_retries(loop, planner, max_decisions=3, on_event=print):
    """Count each call attempt, but count only executed decisions toward the batch."""
    if type(max_decisions) is not int or max_decisions < 0:
        raise ValueError("The number of decisions must be a nonnegative whole number.")
    results, delay = [], 15.0
    while len(results) < max_decisions:
        capacity_failure, proposal_id = None, None

        def attempt(context, directory, seconds):
            nonlocal capacity_failure, proposal_id
            proposal_id = loop.db.execute(
                "SELECT id FROM proposals WHERE campaign=? AND status='running' ORDER BY id DESC LIMIT 1",
                (loop.id,)).fetchone()["id"]
            try:
                return planner(context, directory, seconds)
            except CodexCapacityError as exc:
                capacity_failure = exc
                raise

        def report(event):
            if capacity_failure is not None and event.get("status") == "planner error":
                return  # Replace the temporary failure message with a retry notice below.
            on_event(event)

        batch = loop.run(attempt, max_decisions=1, on_event=report)
        if capacity_failure is not None:
            budget = loop.budget()
            if budget["planner_calls_left"] <= 0 or budget["seconds_left"] <= 0:
                loop._save("budget reached")
                on_event({"status": "budget reached", "reason":
                          "Codex is still busy; the AI-call or time allowance is used up. Progress is saved."})
                break
            wait = min(delay, budget["seconds_left"])
            on_event({"status": "Codex temporarily busy", "retry_in_seconds": wait,
                      "ai_calls_left": budget["planner_calls_left"]})
            _wait_for_retry(loop, proposal_id, wait)
            delay = min(60.0, delay*2)
            continue
        results.extend(batch)
        delay = 15.0
        if not batch or batch[-1].get("status") == "planner checkpoint":
            break
    return results
