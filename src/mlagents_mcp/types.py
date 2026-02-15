from __future__ import annotations

import enum
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


class RunStatus(enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class MetricPoint:
    step: int
    value: float
    wall_time: float


@dataclass
class CheckpointInfo:
    file_path: str
    steps: int
    reward: float | None = None


@dataclass
class RunInfo:
    run_id: str
    config_path: str
    status: RunStatus
    base_port: int
    num_envs: int = 1
    pid: int | None = None
    process: subprocess.Popen | None = None
    reader_thread: threading.Thread | None = None
    log_buffer: deque[str] = field(default_factory=lambda: deque(maxlen=2000))
    results_dir: Path | None = None
    env_path: str | None = None
    return_code: int | None = None
