from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import metrics_reader
from .run_registry import RunRegistry
from .types import RunStatus


def _check_run_exists(registry: RunRegistry, run_id: str) -> dict[str, Any] | None:
    """Return an error dict if the run doesn't exist, else None."""
    info = registry.get(run_id)
    if not info:
        return {"error": f"Run '{run_id}' not found."}
    return None


def _check_run_died(registry: RunRegistry, run_id: str) -> dict[str, Any] | None:
    """Return an error dict if the run stopped/failed/completed unexpectedly, else None."""
    info = registry.get(run_id)
    if not info:
        return {"error": f"Run '{run_id}' not found."}
    if info.status == RunStatus.FAILED:
        lines = list(info.log_buffer)[-20:]
        return {
            "error": f"Run '{run_id}' failed (return code: {info.return_code}).",
            "status": info.status.value,
            "last_logs": lines,
        }
    if info.status == RunStatus.STOPPED:
        return {
            "error": f"Run '{run_id}' was stopped externally.",
            "status": info.status.value,
        }
    return None


def _timeout_result(run_id: str, timeout: float, what: str) -> dict[str, Any]:
    return {
        "error": f"Timeout after {timeout}s waiting for {what}.",
        "run_id": run_id,
        "timed_out": True,
    }


def wait_for_ready(
    registry: RunRegistry,
    run_id: str,
    is_editor_mode: bool,
    timeout: float = 120.0,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """Block until mlagents-learn is ready.

    Editor mode (no env_path): waits for 'Listening on port' — the signal to press Play.
    Batch mode (with env_path): waits for 'Connected to Unity environment' — fully connected.
    """
    err = _check_run_exists(registry, run_id)
    if err:
        return err

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        info = registry.get(run_id)
        if not info:
            return {"error": f"Run '{run_id}' not found."}

        if info.status != RunStatus.RUNNING:
            died = _check_run_died(registry, run_id)
            if died:
                return died

        lines = list(info.log_buffer)

        if is_editor_mode:
            # Editor mode: ready when "Listening on port" appears
            for line in lines:
                if "Listening on port" in line:
                    return {
                        "run_id": run_id,
                        "ready": True,
                        "waiting_for_unity": True,
                        "message": line.strip(),
                        "status": info.status.value,
                    }
        else:
            # Batch mode: ready when Unity executable connects
            for line in lines:
                if "Connected to Unity environment" in line:
                    return {
                        "run_id": run_id,
                        "ready": True,
                        "connected": True,
                        "message": line.strip(),
                        "status": info.status.value,
                    }

        time.sleep(poll_interval)

    info = registry.get(run_id)
    what = "Unity Editor listening" if is_editor_mode else "Unity connection"
    result = _timeout_result(run_id, timeout, what)
    if info:
        result["status"] = info.status.value
        result["last_logs"] = list(info.log_buffer)[-10:]
    return result


def wait_for_first_metrics(
    registry: RunRegistry,
    results_dir: Path,
    run_id: str,
    timeout: float = 600.0,
    poll_interval: float = 10.0,
) -> dict[str, Any]:
    """Block until the run produces at least one TensorBoard scalar data point."""
    err = _check_run_exists(registry, run_id)
    if err:
        return err

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        died = _check_run_died(registry, run_id)
        if died:
            return died

        metrics = metrics_reader.read_metrics(
            results_dir,
            run_id,
            metric_keys=["Environment/Cumulative Reward"],
            last_n=1,
        )
        reward_points = metrics.get("Environment/Cumulative Reward", [])
        if reward_points:
            p = reward_points[0]
            return {
                "run_id": run_id,
                "ready": True,
                "first_step": p.step,
                "first_reward": round(p.value, 4),
                "behaviors": metrics_reader.list_behaviors(results_dir, run_id),
            }

        time.sleep(poll_interval)

    return _timeout_result(run_id, timeout, "first metrics")


def wait_for_step(
    registry: RunRegistry,
    results_dir: Path,
    run_id: str,
    target_step: int,
    timeout: float = 3600.0,
    poll_interval: float = 15.0,
) -> dict[str, Any]:
    """Block until training reaches the target step count."""
    err = _check_run_exists(registry, run_id)
    if err:
        return err

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        # Check if run completed (which is fine — it reached the end)
        info = registry.get(run_id)
        if info and info.status == RunStatus.COMPLETED:
            # Still read final metrics before returning
            pass
        elif info and info.status in (RunStatus.FAILED, RunStatus.STOPPED):
            died = _check_run_died(registry, run_id)
            if died:
                died["target_step"] = target_step
                return died

        metrics = metrics_reader.read_metrics(
            results_dir,
            run_id,
            metric_keys=["Environment/Cumulative Reward"],
            last_n=5,
        )
        reward_points = metrics.get("Environment/Cumulative Reward", [])
        if reward_points:
            current_step = reward_points[-1].step
            if current_step >= target_step:
                return {
                    "run_id": run_id,
                    "reached": True,
                    "target_step": target_step,
                    "current_step": current_step,
                    "current_reward": round(reward_points[-1].value, 4),
                    "reward_trend": [
                        {"step": p.step, "value": round(p.value, 4)}
                        for p in reward_points
                    ],
                    "status": info.status.value if info else "unknown",
                }

        # If completed but never reached the step, report that
        if info and info.status == RunStatus.COMPLETED:
            current_step = reward_points[-1].step if reward_points else 0
            return {
                "run_id": run_id,
                "reached": False,
                "target_step": target_step,
                "current_step": current_step,
                "message": "Training completed before reaching target step.",
                "status": "completed",
            }

        time.sleep(poll_interval)

    result = _timeout_result(run_id, timeout, f"step {target_step}")
    result["target_step"] = target_step
    return result


def wait_for_reward(
    registry: RunRegistry,
    results_dir: Path,
    run_id: str,
    target_reward: float,
    timeout: float = 3600.0,
    poll_interval: float = 15.0,
) -> dict[str, Any]:
    """Block until mean cumulative reward reaches the target threshold."""
    err = _check_run_exists(registry, run_id)
    if err:
        return err

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        info = registry.get(run_id)
        if info and info.status == RunStatus.COMPLETED:
            pass  # Check metrics one last time
        elif info and info.status in (RunStatus.FAILED, RunStatus.STOPPED):
            died = _check_run_died(registry, run_id)
            if died:
                died["target_reward"] = target_reward
                return died

        metrics = metrics_reader.read_metrics(
            results_dir,
            run_id,
            metric_keys=["Environment/Cumulative Reward"],
            last_n=5,
        )
        reward_points = metrics.get("Environment/Cumulative Reward", [])
        if reward_points:
            current_reward = reward_points[-1].value
            if current_reward >= target_reward:
                return {
                    "run_id": run_id,
                    "reached": True,
                    "target_reward": target_reward,
                    "current_reward": round(current_reward, 4),
                    "current_step": reward_points[-1].step,
                    "reward_trend": [
                        {"step": p.step, "value": round(p.value, 4)}
                        for p in reward_points
                    ],
                    "status": info.status.value if info else "unknown",
                }

        if info and info.status == RunStatus.COMPLETED:
            current_reward = reward_points[-1].value if reward_points else 0.0
            return {
                "run_id": run_id,
                "reached": False,
                "target_reward": target_reward,
                "current_reward": round(current_reward, 4),
                "message": "Training completed without reaching target reward.",
                "status": "completed",
            }

        time.sleep(poll_interval)

    result = _timeout_result(run_id, timeout, f"reward {target_reward}")
    result["target_reward"] = target_reward
    return result


def wait_for_completion(
    registry: RunRegistry,
    results_dir: Path,
    run_id: str,
    timeout: float = 7200.0,
    poll_interval: float = 10.0,
) -> dict[str, Any]:
    """Block until the training run finishes (completed, failed, or stopped)."""
    err = _check_run_exists(registry, run_id)
    if err:
        return err

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        info = registry.get(run_id)
        if not info:
            return {"error": f"Run '{run_id}' not found."}

        if info.status != RunStatus.RUNNING:
            # Run finished — gather final info
            result: dict[str, Any] = {
                "run_id": run_id,
                "completed": True,
                "status": info.status.value,
                "return_code": info.return_code,
            }

            # Attach final metrics
            metrics = metrics_reader.read_metrics(
                results_dir,
                run_id,
                metric_keys=["Environment/Cumulative Reward"],
                last_n=5,
            )
            reward_points = metrics.get("Environment/Cumulative Reward", [])
            if reward_points:
                result["final_reward"] = round(reward_points[-1].value, 4)
                result["final_step"] = reward_points[-1].step

            # Attach last few log lines
            result["last_logs"] = list(info.log_buffer)[-15:]

            return result

        time.sleep(poll_interval)

    info = registry.get(run_id)
    result = _timeout_result(run_id, timeout, "completion")
    if info:
        result["status"] = info.status.value
        # Include progress so far
        metrics = metrics_reader.read_metrics(
            results_dir,
            run_id,
            metric_keys=["Environment/Cumulative Reward"],
            last_n=1,
        )
        reward_points = metrics.get("Environment/Cumulative Reward", [])
        if reward_points:
            result["current_step"] = reward_points[-1].step
            result["current_reward"] = round(reward_points[-1].value, 4)
    return result


def wait_for_checkpoint(
    registry: RunRegistry,
    run_id: str,
    timeout: float = 600.0,
    poll_interval: float = 10.0,
) -> dict[str, Any]:
    """Block until a new .onnx checkpoint file appears on disk."""
    info = registry.get(run_id)
    if not info:
        return {"error": f"Run '{run_id}' not found."}
    if not info.results_dir:
        return {"error": f"No results directory for run '{run_id}'."}

    # Snapshot existing .onnx files
    existing_onnx = set(str(p) for p in info.results_dir.rglob("*.onnx"))

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        died = _check_run_died(registry, run_id)
        if died:
            return died

        info = registry.get(run_id)
        if not info or not info.results_dir:
            return {"error": f"Run '{run_id}' not found."}

        current_onnx = set(str(p) for p in info.results_dir.rglob("*.onnx"))
        new_files = current_onnx - existing_onnx
        if new_files:
            new_list = []
            for f in sorted(new_files):
                p = Path(f)
                new_list.append(
                    {
                        "path": f,
                        "behavior": p.parent.name,
                        "size_mb": f"{p.stat().st_size / (1024 * 1024):.2f}",
                    }
                )
            return {
                "run_id": run_id,
                "new_checkpoint": True,
                "files": new_list,
                "total_checkpoints": len(current_onnx),
                "status": info.status.value,
            }

        time.sleep(poll_interval)

    return _timeout_result(run_id, timeout, "new checkpoint")
