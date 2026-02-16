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


def _run_status_info(registry: RunRegistry, run_id: str) -> str:
    """Get the current run status as a string."""
    info = registry.get(run_id)
    return info.status.value if info else "unknown"


def _timeout_result(run_id: str, timeout: float, what: str) -> dict[str, Any]:
    return {
        "error": f"Timeout after {timeout}s waiting for {what}.",
        "run_id": run_id,
        "timed_out": True,
    }


def _get_current_progress(
    results_dir: Path,
    run_id: str,
    last_n: int = 5,
) -> tuple[list[Any], int, float]:
    """Read current reward points. Returns (points, current_step, current_reward)."""
    metrics = metrics_reader.read_metrics(
        results_dir,
        run_id,
        metric_keys=["Environment/Cumulative Reward"],
        last_n=last_n,
    )
    reward_points = metrics.get("Environment/Cumulative Reward", [])
    if reward_points:
        return reward_points, reward_points[-1].step, reward_points[-1].value
    return [], 0, 0.0


# ── Short-lived blockers (always block — used during startup) ──────────


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


# ── Long-lived blocker (used for chaining runs) ───────────────────────


def wait_for_completion(
    registry: RunRegistry,
    results_dir: Path,
    run_id: str,
    timeout: float = 14400.0,
    poll_interval: float = 60.0,
) -> dict[str, Any]:
    """Block until a training run finishes (completed, failed, or stopped).

    Designed for automated chaining: start skill A → wait_for_completion → start skill B.
    Polls the process status (not TensorBoard) so it detects completion immediately.
    Default timeout is 4 hours.
    """
    err = _check_run_exists(registry, run_id)
    if err:
        return err

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        info = registry.get(run_id)
        if not info:
            return {"error": f"Run '{run_id}' disappeared from registry."}

        if info.status != RunStatus.RUNNING:
            # Training finished — gather final info
            result: dict[str, Any] = {
                "run_id": run_id,
                "completed": info.status == RunStatus.COMPLETED,
                "status": info.status.value,
                "return_code": info.return_code,
            }

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
                result["reward_trend"] = [
                    {"step": p.step, "value": round(p.value, 4)} for p in reward_points
                ]

            result["last_logs"] = list(info.log_buffer)[-15:]
            return result

        time.sleep(poll_interval)

    # Timeout — return current progress
    info = registry.get(run_id)
    result_timeout = _timeout_result(run_id, timeout, "training completion")
    if info:
        result_timeout["status"] = info.status.value
        reward_points, current_step, current_reward = _get_current_progress(
            results_dir, run_id
        )
        result_timeout["current_step"] = current_step
        result_timeout["current_reward"] = round(current_reward, 4)
    return result_timeout


# ── Instant checks (never block) ──────────────────────────────────────


def check_step(
    registry: RunRegistry,
    results_dir: Path,
    run_id: str,
    target_step: int,
) -> dict[str, Any]:
    """Check once if training reached the target step count. Returns instantly."""
    err = _check_run_exists(registry, run_id)
    if err:
        return err

    died = _check_run_died(registry, run_id)
    if died:
        died["target_step"] = target_step
        return died

    reward_points, current_step, current_reward = _get_current_progress(
        results_dir, run_id
    )

    return {
        "run_id": run_id,
        "reached": current_step >= target_step if current_step > 0 else False,
        "target_step": target_step,
        "current_step": current_step,
        "current_reward": round(current_reward, 4),
        "reward_trend": [
            {"step": p.step, "value": round(p.value, 4)} for p in reward_points
        ],
        "status": _run_status_info(registry, run_id),
    }


def check_reward(
    registry: RunRegistry,
    results_dir: Path,
    run_id: str,
    target_reward: float,
) -> dict[str, Any]:
    """Check once if mean reward reached the target threshold. Returns instantly."""
    err = _check_run_exists(registry, run_id)
    if err:
        return err

    died = _check_run_died(registry, run_id)
    if died:
        died["target_reward"] = target_reward
        return died

    reward_points, current_step, current_reward = _get_current_progress(
        results_dir, run_id
    )

    return {
        "run_id": run_id,
        "reached": current_reward >= target_reward if reward_points else False,
        "target_reward": target_reward,
        "current_step": current_step,
        "current_reward": round(current_reward, 4),
        "reward_trend": [
            {"step": p.step, "value": round(p.value, 4)} for p in reward_points
        ],
        "status": _run_status_info(registry, run_id),
    }


def check_completion(
    registry: RunRegistry,
    results_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """Check once if training has finished. Returns instantly."""
    err = _check_run_exists(registry, run_id)
    if err:
        return err

    info = registry.get(run_id)
    if not info:
        return {"error": f"Run '{run_id}' not found."}

    if info.status != RunStatus.RUNNING:
        # Done — return final info
        result: dict[str, Any] = {
            "run_id": run_id,
            "completed": True,
            "status": info.status.value,
            "return_code": info.return_code,
        }
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
        result["last_logs"] = list(info.log_buffer)[-15:]
        return result

    # Still running — return current progress
    reward_points, current_step, current_reward = _get_current_progress(
        results_dir, run_id
    )
    return {
        "run_id": run_id,
        "completed": False,
        "status": info.status.value,
        "current_step": current_step,
        "current_reward": round(current_reward, 4),
        "reward_trend": [
            {"step": p.step, "value": round(p.value, 4)} for p in reward_points
        ],
    }


def check_checkpoint(
    registry: RunRegistry,
    run_id: str,
    known_checkpoints: list[str] | None = None,
) -> dict[str, Any]:
    """Check once if new .onnx checkpoint files appeared on disk. Returns instantly."""
    info = registry.get(run_id)
    if not info:
        return {"error": f"Run '{run_id}' not found."}
    if not info.results_dir:
        return {"error": f"No results directory for run '{run_id}'."}

    existing_onnx = set(known_checkpoints) if known_checkpoints else set()
    current_onnx = set(str(p) for p in info.results_dir.rglob("*.onnx"))
    new_files = current_onnx - existing_onnx

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
        "new_checkpoint": len(new_list) > 0,
        "new_files": new_list,
        "total_checkpoints": len(current_onnx),
        "all_checkpoints": sorted(current_onnx),
        "status": info.status.value,
    }
