from __future__ import annotations

import concurrent.futures
from pathlib import Path

from .types import MetricPoint

DEFAULT_METRIC_KEYS = [
    "Environment/Cumulative Reward",
    "Environment/Episode Length",
    "Losses/Policy Loss",
    "Losses/Value Loss",
    "Policy/Learning Rate",
]

# Timeout for EventAccumulator.Reload() — prevents hanging on large/live event files
_RELOAD_TIMEOUT = 30.0


def read_metrics(
    results_dir: Path,
    run_id: str,
    behavior_name: str | None = None,
    metric_keys: list[str] | None = None,
    last_n: int | None = None,
) -> dict[str, list[MetricPoint]]:
    """Read TensorBoard scalars from event files for a given run."""
    run_dir = results_dir / run_id
    if not run_dir.is_dir():
        return {}

    # Find the right subdirectory
    if behavior_name:
        event_dir = run_dir / behavior_name
    else:
        # Auto-detect: find first directory with event files
        event_dir = _find_event_dir(run_dir)

    if event_dir is None:
        return {}

    return _read_from_event_dir(event_dir, metric_keys, last_n)


def _read_from_event_dir(
    event_dir: Path,
    metric_keys: list[str] | None = None,
    last_n: int | None = None,
) -> dict[str, list[MetricPoint]]:
    """Read metrics with a timeout to prevent hanging on large event files."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    acc = EventAccumulator(str(event_dir))

    # Reload in a thread with timeout — this is what hangs on large/live files
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(acc.Reload)
        try:
            future.result(timeout=_RELOAD_TIMEOUT)
        except concurrent.futures.TimeoutError:
            return {}

    keys = metric_keys or DEFAULT_METRIC_KEYS
    result: dict[str, list[MetricPoint]] = {}

    available_tags = acc.Tags().get("scalars", [])

    for key in keys:
        if key not in available_tags:
            continue
        events = acc.Scalars(key)
        points = [
            MetricPoint(step=e.step, value=e.value, wall_time=e.wall_time)
            for e in events
        ]
        if last_n is not None:
            points = points[-last_n:]
        result[key] = points

    return result


def list_behaviors(results_dir: Path, run_id: str) -> list[str]:
    """List behavior names that have TensorBoard event files."""
    run_dir = results_dir / run_id
    if not run_dir.is_dir():
        return []
    behaviors = []
    for sub in run_dir.iterdir():
        if sub.is_dir() and sub.name != "run_logs":
            # Check if it has event files
            if any(sub.glob("events.out.tfevents.*")):
                behaviors.append(sub.name)
    return behaviors


def _find_event_dir(run_dir: Path) -> Path | None:
    """Find the first subdirectory containing TensorBoard event files."""
    for sub in run_dir.iterdir():
        if sub.is_dir() and sub.name != "run_logs":
            if any(sub.glob("events.out.tfevents.*")):
                return sub
    return None
