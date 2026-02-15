from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import config_manager, metrics_reader, waiters
from .process_manager import ProcessManager
from .run_registry import RunRegistry
from .types import RunStatus

# Configuration from environment
PROJECT_ROOT = Path(os.environ.get("MLAGENTS_PROJECT_ROOT", ".")).resolve()
RESULTS_DIR = PROJECT_ROOT / os.environ.get("MLAGENTS_RESULTS_DIR", "results")
CONFIG_DIR = PROJECT_ROOT / os.environ.get("MLAGENTS_CONFIG_DIR", "config")
CONDA_ENV = os.environ.get("MLAGENTS_CONDA_ENV")
CONDA_PATH = os.environ.get("MLAGENTS_CONDA_PATH")

# Initialize components
registry = RunRegistry(RESULTS_DIR)
process_mgr = ProcessManager(
    registry,
    RESULTS_DIR,
    PROJECT_ROOT,
    conda_env=CONDA_ENV,
    conda_path=CONDA_PATH,
)

mcp = FastMCP(
    "mlagents-training",
    instructions=(
        "MCP server for controlling Unity ML-Agents training. "
        "Start/stop training runs, monitor metrics, compare experiments, "
        "and manage training configs."
    ),
)


# ── Training Control ─────────────────────────────────────────────────────


def _launch(
    config_path: str,
    run_id: str,
    env_path: str | None = None,
    resume: bool = False,
    num_envs: int = 1,
    no_graphics: bool = False,
    torch_device: str | None = None,
    seed: int = -1,
    time_scale: float = 20.0,
    base_port: int | None = None,
    wait: bool = True,
    wait_timeout: int = 120,
) -> dict[str, Any]:
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = CONFIG_DIR / config_path

    existing = registry.get(run_id)
    if existing and existing.status == RunStatus.RUNNING:
        return {
            "error": f"Run '{run_id}' is already running. Stop it first or use a different run_id."
        }

    run_info = process_mgr.start(
        config_path=str(resolved_config),
        run_id=run_id,
        env_path=env_path,
        resume=resume,
        force=not resume,
        num_envs=num_envs,
        no_graphics=no_graphics,
        torch_device=torch_device,
        seed=seed,
        time_scale=time_scale,
        base_port=base_port,
    )

    result = {
        "run_id": run_info.run_id,
        "status": run_info.status.value,
        "pid": run_info.pid,
        "base_port": run_info.base_port,
        "config_path": run_info.config_path,
        "results_dir": str(run_info.results_dir),
    }

    if wait:
        is_editor_mode = env_path is None
        wait_result = waiters.wait_for_ready(
            registry,
            run_id,
            is_editor_mode,
            timeout=float(wait_timeout),
        )
        result.update(wait_result)

    return result


@mcp.tool()
def force_training(
    config_path: str,
    run_id: str,
    env_path: str | None = None,
    num_envs: int = 1,
    no_graphics: bool = False,
    torch_device: str | None = None,
    seed: int = -1,
    time_scale: float = 20.0,
    base_port: int | None = None,
    wait: bool = True,
    wait_timeout: int = 120,
) -> dict[str, Any]:
    """Launch a new mlagents-learn training run as a background process. Always overwrites previous results for the same run_id. To continue from a checkpoint, use resume_training instead.

    By default, blocks until ready: in editor mode (no env_path), waits until mlagents-learn says 'Listening on port... press Play'. In batch mode (with env_path), waits until the executable connects.

    Args:
        config_path: Path to the YAML training config file.
        run_id: Unique identifier for this training run.
        env_path: Path to a built Unity environment executable. Omit to use the Unity Editor.
        num_envs: Number of parallel Unity environment instances.
        no_graphics: Disable graphics rendering for faster training.
        torch_device: PyTorch device (e.g. 'cuda', 'cpu', 'cuda:0').
        seed: Random seed (-1 for random).
        time_scale: Unity time scale multiplier.
        base_port: Base port for Unity communication (auto-assigned if omitted).
        wait: Block until ready (default true). Editor mode: waits for 'press Play'. Batch mode: waits for connection.
        wait_timeout: Max seconds to wait when wait=true.
    """
    return _launch(
        config_path=config_path,
        run_id=run_id,
        env_path=env_path,
        resume=False,
        num_envs=num_envs,
        no_graphics=no_graphics,
        torch_device=torch_device,
        seed=seed,
        time_scale=time_scale,
        base_port=base_port,
        wait=wait,
        wait_timeout=wait_timeout,
    )


@mcp.tool()
def stop_training(run_id: str, timeout: int = 30) -> dict[str, Any]:
    """Gracefully stop a running training run (sends SIGINT so the model is saved).

    Args:
        run_id: The run to stop.
        timeout: Seconds to wait for graceful shutdown before force-killing.
    """
    info = registry.get(run_id)
    if not info:
        return {"error": f"Run '{run_id}' not found."}
    if info.status != RunStatus.RUNNING:
        return {
            "error": f"Run '{run_id}' is not running (status: {info.status.value})."
        }

    success = process_mgr.stop(run_id, timeout=float(timeout))
    info = registry.get(run_id)
    return {
        "run_id": run_id,
        "stopped": success,
        "status": info.status.value if info else "unknown",
        "return_code": info.return_code if info else None,
    }


@mcp.tool()
def resume_training(
    run_id: str,
    config_path: str | None = None,
    num_envs: int = 1,
    no_graphics: bool = False,
    torch_device: str | None = None,
    time_scale: float = 20.0,
    wait: bool = True,
    wait_timeout: int = 120,
) -> dict[str, Any]:
    """Resume a previously stopped/completed training run.

    If config_path is not provided, reads the saved configuration.yaml from the previous run.
    By default, blocks until ready (same as force_training).

    Args:
        run_id: The run_id to resume.
        config_path: Config file path (auto-detected from previous run if omitted).
        num_envs: Number of parallel Unity environment instances.
        no_graphics: Disable graphics rendering.
        torch_device: PyTorch device.
        time_scale: Unity time scale multiplier.
        wait: Block until ready (default true). Editor mode: waits for 'press Play'. Batch mode: waits for connection.
        wait_timeout: Max seconds to wait when wait=true.
    """
    if config_path is None:
        # Try to read from previous run's saved config
        saved_config = RESULTS_DIR / run_id / "configuration.yaml"
        if saved_config.exists():
            config_path = str(saved_config)
        else:
            prev = registry.get(run_id)
            if prev and prev.config_path:
                config_path = prev.config_path
            else:
                return {
                    "error": f"No config found for run '{run_id}'. Provide config_path explicitly."
                }

    prev = registry.get(run_id)
    env_path = prev.env_path if prev else None

    return _launch(
        config_path=config_path,
        run_id=run_id,
        env_path=env_path,
        resume=True,
        num_envs=num_envs,
        no_graphics=no_graphics,
        torch_device=torch_device,
        time_scale=time_scale,
        wait=wait,
        wait_timeout=wait_timeout,
    )


# ── Monitoring ────────────────────────────────────────────────────────────


@mcp.tool()
def get_run_status(
    run_id: str, last_n_rewards: int = 10, last_n_checkpoints: int = 5
) -> dict[str, Any]:
    """Get detailed status of a training run including step progress, reward trend, and checkpoints.

    Args:
        run_id: The run to query.
        last_n_rewards: Number of recent reward data points to include in the trend.
        last_n_checkpoints: Number of recent checkpoints to include.
    """
    info = registry.get(run_id)
    if not info:
        return {"error": f"Run '{run_id}' not found."}

    result: dict[str, Any] = {
        "run_id": run_id,
        "status": info.status.value,
        "pid": info.pid,
        "config_path": info.config_path,
        "results_dir": str(info.results_dir),
        "return_code": info.return_code,
    }

    # Get latest metrics for reward trend
    metrics = metrics_reader.read_metrics(
        RESULTS_DIR,
        run_id,
        metric_keys=["Environment/Cumulative Reward"],
        last_n=last_n_rewards,
    )
    reward_points = metrics.get("Environment/Cumulative Reward", [])
    if reward_points:
        result["latest_reward"] = reward_points[-1].value
        result["latest_step"] = reward_points[-1].step
        result["reward_trend"] = [
            {"step": p.step, "value": round(p.value, 3)} for p in reward_points
        ]

    # Get checkpoints
    checkpoints = registry.get_checkpoints(run_id)
    if checkpoints:
        result["checkpoints"] = [asdict(cp) for cp in checkpoints[-last_n_checkpoints:]]
        result["total_checkpoints"] = len(checkpoints)

    # Get behaviors
    behaviors = metrics_reader.list_behaviors(RESULTS_DIR, run_id)
    if behaviors:
        result["behaviors"] = behaviors

    return result


@mcp.tool()
def get_metrics(
    run_id: str,
    behavior_name: str | None = None,
    metric_keys: list[str] | None = None,
    last_n: int = 20,
) -> dict[str, Any]:
    """Read TensorBoard scalar metrics from a training run.

    Args:
        run_id: The run to query.
        behavior_name: Specific behavior to read (auto-detected if omitted).
        metric_keys: Specific metric keys to read (defaults to reward, losses, LR).
        last_n: Number of most recent data points per metric (default 20). Use -1 for all.
    """
    effective_last_n = None if last_n == -1 else last_n
    metrics = metrics_reader.read_metrics(
        RESULTS_DIR, run_id, behavior_name, metric_keys, effective_last_n
    )
    if not metrics:
        return {
            "run_id": run_id,
            "metrics": {},
            "note": "No metrics found. Training may not have produced data yet.",
        }

    return {
        "run_id": run_id,
        "metrics": {
            key: [{"step": p.step, "value": round(p.value, 6)} for p in points]
            for key, points in metrics.items()
        },
    }


@mcp.tool()
def get_training_logs(run_id: str, last_n_lines: int = 50) -> dict[str, Any]:
    """Get recent stdout/stderr output from an active training run.

    Args:
        run_id: The run to query.
        last_n_lines: Number of most recent log lines to return.
    """
    info = registry.get(run_id)
    if not info:
        return {"error": f"Run '{run_id}' not found."}

    lines = list(info.log_buffer)[-last_n_lines:]
    return {
        "run_id": run_id,
        "status": info.status.value,
        "log_lines": lines,
        "total_buffered": len(info.log_buffer),
    }


@mcp.tool()
def list_runs(status_filter: str | None = None, last_n: int = 20) -> dict[str, Any]:
    """List known training runs with their status.

    Args:
        status_filter: Filter by status: 'running', 'completed', 'failed', or 'stopped'.
        last_n: Number of most recent runs to return (default 20). Use -1 for all.
    """
    filter_enum = None
    if status_filter:
        try:
            filter_enum = RunStatus(status_filter)
        except ValueError:
            return {
                "error": f"Invalid status filter '{status_filter}'. Use: running, completed, failed, stopped."
            }

    runs = registry.list_runs(filter_enum)
    total = len(runs)
    if last_n != -1:
        runs = runs[-last_n:]
    return {
        "runs": [
            {
                "run_id": r.run_id,
                "status": r.status.value,
                "pid": r.pid,
                "config_path": r.config_path,
                "env_path": r.env_path,
            }
            for r in runs
        ],
        "total": total,
    }


# ── Comparison & Export ───────────────────────────────────────────────────


@mcp.tool()
def compare_runs(
    run_ids: list[str], metric_key: str, last_n: int = 20
) -> dict[str, Any]:
    """Compare a specific metric across multiple training runs (useful for hyperparameter tuning).

    Args:
        run_ids: List of run IDs to compare.
        metric_key: The TensorBoard metric key to compare (e.g. 'Environment/Cumulative Reward').
        last_n: Number of most recent data points per run to include in the trend. Use -1 for all.
    """
    effective_last_n = None if last_n == -1 else last_n
    comparison: dict[str, Any] = {}
    for rid in run_ids:
        metrics = metrics_reader.read_metrics(
            RESULTS_DIR, rid, metric_keys=[metric_key]
        )
        all_points = metrics.get(metric_key, [])
        if all_points:
            all_values = [p.value for p in all_points]
            points = all_points[-effective_last_n:] if effective_last_n else all_points
            comparison[rid] = {
                "final_value": round(all_values[-1], 4),
                "max_value": round(max(all_values), 4),
                "min_value": round(min(all_values), 4),
                "num_points": len(all_values),
                "final_step": all_points[-1].step,
                "trend": [{"step": p.step, "value": round(p.value, 4)} for p in points],
            }
        else:
            comparison[rid] = {"error": "No data for this metric."}

    return {"metric_key": metric_key, "comparison": comparison}


@mcp.tool()
def export_model(
    run_id: str, behavior_name: str | None = None, last_n: int = 5
) -> dict[str, Any]:
    """Locate .onnx model files and checkpoints for a training run.

    Args:
        run_id: The run to query.
        behavior_name: Specific behavior (searches all if omitted).
        last_n: Number of most recent models/checkpoints to return. Use -1 for all.
    """
    run_dir = RESULTS_DIR / run_id
    if not run_dir.is_dir():
        return {"error": f"Run directory not found: {run_dir}"}

    models: list[dict[str, str]] = []
    search_dir = run_dir / behavior_name if behavior_name else run_dir
    for onnx in sorted(search_dir.rglob("*.onnx")):
        models.append(
            {
                "path": str(onnx),
                "behavior": onnx.parent.name,
                "size_mb": f"{onnx.stat().st_size / (1024 * 1024):.2f}",
            }
        )

    checkpoints = registry.get_checkpoints(run_id)

    total_models = len(models)
    total_checkpoints = len(checkpoints)
    if last_n != -1:
        models = models[-last_n:]
        checkpoints = checkpoints[-last_n:]

    return {
        "run_id": run_id,
        "models": models,
        "total_models": total_models,
        "checkpoints": [asdict(cp) for cp in checkpoints],
        "total_checkpoints": total_checkpoints,
    }


# ── Configuration ─────────────────────────────────────────────────────────


@mcp.tool()
def get_config(config_path: str) -> dict[str, Any]:
    """Read a YAML training configuration file.

    Args:
        config_path: Path to the config file (relative to config dir or absolute).
    """
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = CONFIG_DIR / config_path

    if not resolved.exists():
        # Try listing available configs
        available = config_manager.list_configs(CONFIG_DIR)
        return {
            "error": f"Config file not found: {resolved}",
            "available_configs": available,
        }

    data = config_manager.read_config(resolved)
    return {"config_path": str(resolved), "config": data}


@mcp.tool()
def update_config(config_path: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge updates into an existing YAML config file. Only specified keys are changed.

    Args:
        config_path: Path to the config file (relative to config dir or absolute).
        updates: Dictionary of updates to deep-merge into the config.
    """
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = CONFIG_DIR / config_path

    if not resolved.exists():
        return {"error": f"Config file not found: {resolved}"}

    merged = config_manager.update_config(resolved, updates)
    return {"config_path": str(resolved), "updated_config": merged}


# ── Blocking Wait Tools ────────────────────────────────────────────────


@mcp.tool()
def wait_for_first_metrics(
    run_id: str,
    timeout: int = 600,
    poll_interval: int = 10,
) -> dict[str, Any]:
    """Block until the training run produces its first TensorBoard metric data point. Use this to know when training has actually started producing results.

    Args:
        run_id: The run to watch.
        timeout: Max seconds to wait.
        poll_interval: Seconds between checks.
    """
    return waiters.wait_for_first_metrics(
        registry,
        RESULTS_DIR,
        run_id,
        float(timeout),
        float(poll_interval),
    )


@mcp.tool()
def wait_for_step(
    run_id: str,
    target_step: int,
    timeout: int = 3600,
    poll_interval: int = 15,
) -> dict[str, Any]:
    """Block until training reaches the target step count. Returns current metrics at that point. Useful for checking progress at specific milestones without polling.

    Args:
        run_id: The run to watch.
        target_step: The step count to wait for.
        timeout: Max seconds to wait.
        poll_interval: Seconds between checks.
    """
    return waiters.wait_for_step(
        registry,
        RESULTS_DIR,
        run_id,
        target_step,
        float(timeout),
        float(poll_interval),
    )


@mcp.tool()
def wait_for_reward(
    run_id: str,
    target_reward: float,
    timeout: int = 3600,
    poll_interval: int = 15,
) -> dict[str, Any]:
    """Block until mean cumulative reward reaches the target threshold. Returns reward trend at that point. Useful for stopping training once a performance target is hit.

    Args:
        run_id: The run to watch.
        target_reward: The reward threshold to wait for.
        timeout: Max seconds to wait.
        poll_interval: Seconds between checks.
    """
    return waiters.wait_for_reward(
        registry,
        RESULTS_DIR,
        run_id,
        target_reward,
        float(timeout),
        float(poll_interval),
    )


@mcp.tool()
def wait_for_completion(
    run_id: str,
    timeout: int = 7200,
    poll_interval: int = 10,
) -> dict[str, Any]:
    """Block until the training run finishes (completed, failed, or stopped). Returns final status, metrics, and last log lines. Use this instead of polling get_run_status in a loop.

    Args:
        run_id: The run to watch.
        timeout: Max seconds to wait (default 2 hours).
        poll_interval: Seconds between checks.
    """
    return waiters.wait_for_completion(
        registry,
        RESULTS_DIR,
        run_id,
        float(timeout),
        float(poll_interval),
    )


@mcp.tool()
def wait_for_checkpoint(
    run_id: str,
    timeout: int = 600,
    poll_interval: int = 10,
) -> dict[str, Any]:
    """Block until a new .onnx model checkpoint file appears on disk. Snapshots existing files at call time and waits for a new one. Useful for knowing when a model is ready to export.

    Args:
        run_id: The run to watch.
        timeout: Max seconds to wait.
        poll_interval: Seconds between checks.
    """
    return waiters.wait_for_checkpoint(
        registry,
        run_id,
        float(timeout),
        float(poll_interval),
    )


def main() -> None:
    mcp.run(transport="stdio")
