from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

from experiments.recorders.resource_monitor import ResourceMonitor


class ProcessSupervisor:
    """Own exactly the process groups launched for one real experiment run."""
    def __init__(self, popen=subprocess.Popen, health_timeout_s=120.0, shutdown_grace_s=5.0):
        self._popen = popen
        self.health_timeout_s = float(health_timeout_s)
        self.shutdown_grace_s = float(shutdown_grace_s)
        self._processes = []
        self._handles = []

    def _launch(self, name, command, log_dir, cwd):
        log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
        stdout = (log_dir / f"{name}.stdout.log").open("ab")
        stderr = (log_dir / f"{name}.stderr.log").open("ab")
        self._handles.extend((stdout, stderr))
        process = self._popen(command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True)
        self._processes.append(process)
        return process

    def _wait_health(self, url):
        deadline = time.monotonic() + self.health_timeout_s
        last_error = None
        while time.monotonic() < deadline:
            try:
                with urlopen(url, timeout=2.0) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if payload.get("ok"):
                    return payload
            except Exception as error:
                last_error = error
            time.sleep(0.25)
        raise TimeoutError(f"NavDP health check timed out: {last_error!r}")

    def run_pair(self, server_command, eval_command, run_dir, cwd, port=8888, timeout_s=1800.0):
        resource_monitor = ResourceMonitor(run_dir, lambda: self._processes)
        error = None
        try:
            self._launch("navdp_server", server_command, Path(run_dir) / "logs", cwd)
            resource_monitor.start()
            self._wait_health(f"http://127.0.0.1:{port}/health")
            evaluation = self._launch("isaac_eval", eval_command, Path(run_dir) / "logs", cwd)
            return_code = evaluation.wait(timeout=timeout_s)
            if return_code != 0:
                raise RuntimeError(f"Isaac eval exited with code {return_code}")
            return return_code
        except BaseException as caught:
            error = caught
            raise
        finally:
            self.close()
            resource_monitor.stop(status="failed" if error is not None else "finished", error=error)

    def _signal(self, process, sig):
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), sig)

    def close(self):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            live = [process for process in self._processes if process.poll() is None]
            if not live:
                break
            for process in live:
                try: self._signal(process, sig)
                except ProcessLookupError: pass
            if sig != signal.SIGKILL:
                deadline = time.monotonic() + self.shutdown_grace_s
                while time.monotonic() < deadline and any(process.poll() is None for process in live):
                    time.sleep(0.05)
        for handle in self._handles:
            handle.close()
        self._handles.clear(); self._processes.clear()
