from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import psutil

from experiments.recorders.run_recorder import atomic_json


def _sha256(path):
    path = Path(path)
    if not path.is_file(): return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _command_output(command):
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_inventory(probe_external=True):
    if not probe_external: return {"status":"not_probed_dry_run", "devices":[]}
    output = _command_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if not output: return {"status":"unavailable", "devices":[]}
    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            devices.append({"index":int(fields[0]), "name":fields[1], "uuid":fields[2], "memory_total_mib":int(fields[3]), "driver_version":fields[4]})
    return {"status":"available", "devices":devices}


def _git_snapshot(repo_root, probe_external):
    if probe_external:
        head = _command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
        status = _command_output(["git", "-C", str(repo_root), "status", "--porcelain"])
        return {"git_head":head, "dirty":bool(status), "status_porcelain":status or "", "status":"probed"}
    head_file = Path(repo_root) / ".git" / "HEAD"
    head = head_file.read_text(encoding="utf-8").strip() if head_file.exists() else None
    if head and head.startswith("ref: "):
        ref_file = Path(repo_root) / ".git" / head[5:]
        head = ref_file.read_text(encoding="utf-8").strip() if ref_file.exists() else head
    return {"git_head":head, "dirty":None, "status_porcelain":"", "status":"not_probed_dry_run"}


def build_run_manifest(repo_root, run_config, evaluation_command, server_command, probe_external=True):
    repo_root = Path(repo_root).resolve()
    checkpoint = Path(server_command[server_command.index("--checkpoint") + 1])
    repository = _git_snapshot(repo_root, probe_external)
    memory = psutil.virtual_memory()
    if probe_external:
        host_identity = {"hostname":platform.node(), "platform":platform.platform(), "cpu":platform.processor() or platform.machine()}
    else:
        uname = os.uname()
        host_identity = {"hostname":uname.nodename, "platform":f"{uname.sysname}-{uname.release}-{uname.machine}", "cpu":uname.machine}
    return {
        "schema_version":1, "captured_at_utc":datetime.now(timezone.utc).isoformat(),
        "run_identity":{key:run_config.get(key) for key in ("suite_id","experiment_id","run_id","variant","scene_id","seed")},
        "commands":{"evaluation":list(evaluation_command), "navdp_server":list(server_command)},
        "checkpoint":{"path":str(checkpoint), "sha256":_sha256(checkpoint)},
        "repository":{"root":str(repo_root), **repository},
        "environment":{"orchestrator_python":sys.version, "executable":sys.executable, "conda_prefix":os.environ.get("CONDA_PREFIX"), "eval_conda_env":"isaaclab", "server_conda_env":"navdp"},
        "host":{**host_identity, "logical_cpu_count":os.cpu_count(), "memory_total_bytes":int(memory.total), "gpu":_gpu_inventory(probe_external)},
        "effective_parameters":run_config.get("effective_parameters", {}),
        "parameter_receipt":run_config.get("parameter_receipt", {}),
    }


def write_run_manifest(path, repo_root, run_config, evaluation_command, server_command, probe_external=True):
    payload = build_run_manifest(repo_root, run_config, evaluation_command, server_command, probe_external=probe_external)
    atomic_json(Path(path), payload)
    return payload
