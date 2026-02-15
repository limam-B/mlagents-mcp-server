from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path

from .run_registry import RunRegistry
from .types import RunInfo, RunStatus

DEFAULT_BASE_PORT = 5005
CONDA_SETUP = (
    "source {conda_path}/etc/profile.d/conda.sh && conda activate {conda_env} && "
)


def _find_child_pids(parent_pid: int) -> list[int]:
    """Find all direct child PIDs of a process using /proc."""
    children = []
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                # Format: pid (comm) state ppid ...
                parts = stat.split(") ", 1)
                if len(parts) < 2:
                    continue
                fields = parts[1].split()
                ppid = int(fields[1])  # ppid is 4th field, index 1 after split
                if ppid == parent_pid:
                    children.append(int(entry.name))
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        pass
    return children


def _find_mlagents_pid(shell_pid: int) -> int | None:
    """Walk the process tree from the shell to find the mlagents-learn process."""
    visited = set()
    queue = [shell_pid]
    while queue:
        pid = queue.pop(0)
        if pid in visited:
            continue
        visited.add(pid)
        # Check if this PID is mlagents-learn
        try:
            cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            if b"mlagents-learn" in cmdline:
                return pid
        except OSError:
            continue
        queue.extend(_find_child_pids(pid))
    return None


class ProcessManager:
    def __init__(
        self,
        registry: RunRegistry,
        results_dir: Path,
        project_root: Path,
        conda_env: str | None = None,
        conda_path: str | None = None,
    ) -> None:
        self._registry = registry
        self._results_dir = results_dir
        self._project_root = project_root
        self._conda_env = conda_env
        self._conda_path = conda_path or str(Path.home() / "miniconda3")

    def allocate_port(self, num_envs: int) -> int:
        """Find a base port that doesn't conflict with any active run."""
        active = self._registry.get_active_ports()
        port = DEFAULT_BASE_PORT
        while True:
            conflict = False
            for active_base, active_num in active:
                # Each run uses ports [base_port, base_port + num_envs)
                if not (
                    port + num_envs <= active_base or port >= active_base + active_num
                ):
                    conflict = True
                    break
            if not conflict:
                return port
            port += num_envs

    def start(
        self,
        config_path: str,
        run_id: str,
        env_path: str | None = None,
        resume: bool = False,
        force: bool = False,
        num_envs: int = 1,
        no_graphics: bool = True,
        torch_device: str | None = None,
        seed: int = -1,
        time_scale: float = 20.0,
        base_port: int | None = None,
    ) -> RunInfo:
        if base_port is None:
            base_port = self.allocate_port(num_envs)

        args = [
            "mlagents-learn",
            config_path,
            f"--run-id={run_id}",
            f"--base-port={base_port}",
            f"--num-envs={num_envs}",
            f"--time-scale={time_scale}",
            f"--results-dir={self._results_dir}",
        ]

        if env_path:
            args.append(f"--env={env_path}")
        if resume:
            args.append("--resume")
        if force:
            args.append("--force")
        if no_graphics:
            args.append("--no-graphics")
        if torch_device:
            args.append(f"--torch-device={torch_device}")
        if seed != -1:
            args.append(f"--seed={seed}")

        cmd_str = " ".join(args)
        if self._conda_env:
            prefix = CONDA_SETUP.format(
                conda_path=self._conda_path,
                conda_env=self._conda_env,
            )
            cmd_str = prefix + cmd_str

        proc = subprocess.Popen(
            cmd_str,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(self._project_root),
            text=True,
            # setsid so we can signal the whole process group for cleanup
            preexec_fn=os.setsid,
        )

        run_info = RunInfo(
            run_id=run_id,
            config_path=config_path,
            status=RunStatus.RUNNING,
            base_port=base_port,
            num_envs=num_envs,
            pid=proc.pid,
            process=proc,
            results_dir=self._results_dir / run_id,
            env_path=env_path,
        )

        reader = threading.Thread(
            target=self._reader_loop,
            args=(run_info,),
            daemon=True,
        )
        run_info.reader_thread = reader
        self._registry.register(run_info)
        reader.start()

        return run_info

    def _reader_loop(self, run_info: RunInfo) -> None:
        proc = run_info.process
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                run_info.log_buffer.append(line.rstrip("\n"))
        except ValueError:
            pass  # stdout closed
        proc.wait()
        rc = proc.returncode
        if rc == 0:
            status = RunStatus.COMPLETED
        elif rc == -signal.SIGINT or rc == -signal.SIGTERM:
            status = RunStatus.STOPPED
        else:
            status = RunStatus.FAILED
        self._registry.update_status(run_info.run_id, status, rc)

    def stop(self, run_id: str, timeout: float = 30.0) -> bool:
        info = self._registry.get(run_id)
        if not info or not info.process or info.status != RunStatus.RUNNING:
            return False

        proc = info.process

        # Find the actual mlagents-learn PID (proc.pid is the bash shell)
        ml_pid = _find_mlagents_pid(proc.pid)

        # Step 1: Send SIGINT to mlagents-learn directly (it handles graceful shutdown)
        target_pid = ml_pid or proc.pid
        try:
            os.kill(target_pid, signal.SIGINT)
        except ProcessLookupError:
            self._registry.update_status(run_id, RunStatus.STOPPED, proc.returncode)
            return True

        try:
            proc.wait(timeout=timeout)
            self._registry.update_status(run_id, RunStatus.STOPPED, proc.returncode)
            return True
        except subprocess.TimeoutExpired:
            pass

        # Step 2: SIGINT didn't finish in time — SIGTERM the process group
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        try:
            proc.wait(timeout=5)
            self._registry.update_status(run_id, RunStatus.STOPPED, proc.returncode)
            return True
        except subprocess.TimeoutExpired:
            pass

        # Step 3: Still stuck — SIGKILL the entire process group
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

        self._registry.update_status(run_id, RunStatus.STOPPED, proc.returncode)
        return True
