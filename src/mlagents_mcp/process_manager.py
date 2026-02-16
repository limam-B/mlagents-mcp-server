from __future__ import annotations

import concurrent.futures
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

# Timeout for /proc reads — prevents hanging on D-state processes
_PROC_READ_TIMEOUT = 3.0


def _safe_read_cmdline(pid: int) -> bytes | None:
    """Read /proc/PID/cmdline with a timeout. Returns None if blocked or missing."""

    def _read() -> bytes:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_read)
        try:
            return future.result(timeout=_PROC_READ_TIMEOUT)
        except (concurrent.futures.TimeoutError, OSError, FileNotFoundError):
            return None


def _get_all_descendants(root_pid: int) -> list[int]:
    """Get all descendant PIDs by scanning /proc for ppid matches.

    Uses /proc/PID/stat (which reads the ppid field) instead of cmdline,
    so it won't block on D-state processes.
    """
    # Build a full parent→children map in one pass
    children_map: dict[int, list[int]] = {}
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                parts = stat.split(") ", 1)
                if len(parts) < 2:
                    continue
                fields = parts[1].split()
                ppid = int(fields[1])
                pid = int(entry.name)
                children_map.setdefault(ppid, []).append(pid)
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        pass

    # BFS from root
    descendants = []
    queue = [root_pid]
    visited = {root_pid}
    while queue:
        current = queue.pop(0)
        for child in children_map.get(current, []):
            if child not in visited:
                visited.add(child)
                descendants.append(child)
                queue.append(child)
    return descendants


def _find_mlagents_pid(shell_pid: int) -> int | None:
    """Find mlagents-learn PID among descendants. Uses safe reads with timeout."""
    descendants = _get_all_descendants(shell_pid)
    for pid in descendants:
        cmdline = _safe_read_cmdline(pid)
        if cmdline and b"mlagents-learn" in cmdline:
            return pid
    return None


def _find_stale_processes(
    registry_pids: set[int],
) -> list[dict[str, int | str]]:
    """Find orphaned mlagents-learn and Unity build processes not tracked by the registry."""
    stale = []
    ml_patterns = [b"mlagents-learn", b"mlagents.trainers"]
    unity_patterns = [b".x86_64", b"LinuxPlayer", b"UnityPlayer"]

    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in registry_pids:
                continue
            cmdline = _safe_read_cmdline(pid)
            if not cmdline:
                continue

            for pattern in ml_patterns:
                if pattern in cmdline:
                    if b"mlagents-mcp" in cmdline:
                        break
                    cmd_str = cmdline.replace(b"\x00", b" ").decode(errors="replace")
                    stale.append(
                        {"pid": pid, "type": "mlagents-learn", "cmd": cmd_str[:200]}
                    )
                    break

            for pattern in unity_patterns:
                if pattern in cmdline:
                    cmd_str = cmdline.replace(b"\x00", b" ").decode(errors="replace")
                    stale.append(
                        {"pid": pid, "type": "unity-build", "cmd": cmd_str[:200]}
                    )
                    break

    except OSError:
        pass
    return stale


def _kill_tree(root_pid: int, sig: int) -> list[int]:
    """Send a signal to all descendants of root_pid, then to root_pid itself.

    Returns list of PIDs that were successfully signaled.
    """
    descendants = _get_all_descendants(root_pid)
    signaled = []
    # Kill children first (bottom-up would be ideal, but order matters less for SIGKILL)
    for pid in descendants:
        try:
            os.kill(pid, sig)
            signaled.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    # Then kill root
    try:
        os.kill(root_pid, sig)
        signaled.append(root_pid)
    except (ProcessLookupError, PermissionError):
        pass
    return signaled


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

        # If stop was requested, always mark as STOPPED regardless of exit code.
        # mlagents-learn exits with rc=0 on SIGINT (graceful shutdown), which
        # would otherwise be misclassified as COMPLETED.
        if run_info.stop_requested:
            status = RunStatus.STOPPED
        elif rc == 0:
            status = RunStatus.COMPLETED
        elif rc == -signal.SIGINT or rc == -signal.SIGTERM:
            status = RunStatus.STOPPED
        else:
            status = RunStatus.FAILED
        self._registry.update_status(run_info.run_id, status, rc)

    def cleanup_stale(self) -> dict:
        """Find and kill orphaned mlagents-learn and Unity build processes."""
        tracked_pids: set[int] = set()
        for run_info in self._registry.list_runs():
            if run_info.process and run_info.status == RunStatus.RUNNING:
                tracked_pids.add(run_info.pid)
                tracked_pids.update(_get_all_descendants(run_info.pid))

        stale = _find_stale_processes(tracked_pids)
        killed = []
        failed = []

        for proc_info in stale:
            pid = proc_info["pid"]
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(proc_info)
            except ProcessLookupError:
                killed.append(proc_info)
            except PermissionError:
                failed.append({**proc_info, "reason": "permission denied"})

        return {
            "killed": killed,
            "failed": failed,
            "total_found": len(stale),
            "total_killed": len(killed),
        }

    def stop(self, run_id: str, timeout: float = 30.0) -> bool:
        info = self._registry.get(run_id)
        if not info or not info.process or info.status != RunStatus.RUNNING:
            return False

        proc = info.process

        # Mark as stop-requested BEFORE sending signal, so _reader_loop
        # knows this was user-initiated (not natural completion with rc=0)
        info.stop_requested = True

        # Step 1: Try graceful SIGINT to mlagents-learn (with timeout on PID lookup)
        ml_pid = _find_mlagents_pid(proc.pid)
        target_pid = ml_pid or proc.pid
        try:
            os.kill(target_pid, signal.SIGINT)
        except ProcessLookupError:
            self._registry.update_status(run_id, RunStatus.STOPPED, proc.returncode)
            return True

        # Wait for graceful shutdown (shorter timeout for batch: 15s)
        grace_timeout = min(timeout, 15.0)
        try:
            proc.wait(timeout=grace_timeout)
            self._registry.update_status(run_id, RunStatus.STOPPED, proc.returncode)
            return True
        except subprocess.TimeoutExpired:
            pass

        # Step 2: SIGINT didn't work — kill the ENTIRE process tree with SIGTERM
        _kill_tree(proc.pid, signal.SIGTERM)

        try:
            proc.wait(timeout=5)
            self._registry.update_status(run_id, RunStatus.STOPPED, proc.returncode)
            return True
        except subprocess.TimeoutExpired:
            pass

        # Step 3: SIGTERM didn't work — SIGKILL the entire tree
        _kill_tree(proc.pid, signal.SIGKILL)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

        self._registry.update_status(run_id, RunStatus.STOPPED, proc.returncode)
        return True
