from __future__ import annotations

import json
import threading
from pathlib import Path

from .types import CheckpointInfo, RunInfo, RunStatus


class RunRegistry:
    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir
        self._runs: dict[str, RunInfo] = {}
        self._lock = threading.Lock()
        self._scan_historical()

    def _scan_historical(self) -> None:
        if not self._results_dir.is_dir():
            return
        for run_dir in self._results_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            if run_id in self._runs:
                continue
            status = self._detect_status(run_dir)
            config_path = ""
            config_file = run_dir / "configuration.yaml"
            if config_file.exists():
                config_path = str(config_file)
            self._runs[run_id] = RunInfo(
                run_id=run_id,
                config_path=config_path,
                status=status,
                base_port=0,
                results_dir=run_dir,
            )

    def _detect_status(self, run_dir: Path) -> RunStatus:
        status_file = run_dir / "run_logs" / "training_status.json"
        if status_file.exists():
            try:
                data = json.loads(status_file.read_text())
                # If training_status.json exists with metadata, training completed
                if "metadata" in data:
                    return RunStatus.COMPLETED
            except (json.JSONDecodeError, KeyError):
                pass
        # If directory exists but no status file, assume it failed or was stopped
        return RunStatus.STOPPED

    def register(self, run_info: RunInfo) -> None:
        with self._lock:
            self._runs[run_info.run_id] = run_info

    def get(self, run_id: str) -> RunInfo | None:
        with self._lock:
            return self._runs.get(run_id)

    def update_status(
        self, run_id: str, status: RunStatus, return_code: int | None = None
    ) -> None:
        with self._lock:
            info = self._runs.get(run_id)
            if info:
                info.status = status
                if return_code is not None:
                    info.return_code = return_code

    def list_runs(self, status_filter: RunStatus | None = None) -> list[RunInfo]:
        with self._lock:
            runs = list(self._runs.values())
        if status_filter:
            runs = [r for r in runs if r.status == status_filter]
        return runs

    def get_active_ports(self) -> list[tuple[int, int]]:
        """Return (base_port, num_envs) for all running runs."""
        with self._lock:
            return [
                (r.base_port, r.num_envs)
                for r in self._runs.values()
                if r.status == RunStatus.RUNNING
            ]

    def get_checkpoints(self, run_id: str) -> list[CheckpointInfo]:
        info = self.get(run_id)
        if not info or not info.results_dir:
            return []
        checkpoints = []
        for onnx_file in info.results_dir.rglob("*.onnx"):
            checkpoints.append(
                CheckpointInfo(
                    file_path=str(onnx_file),
                    steps=0,
                )
            )
        # Try reading training_status.json for step counts
        status_file = info.results_dir / "run_logs" / "training_status.json"
        if status_file.exists():
            try:
                data = json.loads(status_file.read_text())
                for behavior_name, behavior_data in data.items():
                    if behavior_name == "metadata":
                        continue
                    if (
                        isinstance(behavior_data, dict)
                        and "checkpoints" in behavior_data
                    ):
                        for cp in behavior_data["checkpoints"]:
                            if isinstance(cp, dict):
                                checkpoints.append(
                                    CheckpointInfo(
                                        file_path=cp.get("file_path", ""),
                                        steps=cp.get("steps", 0),
                                        reward=cp.get("reward"),
                                    )
                                )
            except (json.JSONDecodeError, KeyError):
                pass
        return checkpoints
