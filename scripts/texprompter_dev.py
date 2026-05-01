#!/usr/bin/env python3
"""Interactive dev launcher: SSH tunnel (key-only), MLflow server, pipeline or eval.

See README «Dev startup scripts». SSH key authentication only — no password storage.
"""
from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

# Repo root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_RZ_SSH_HOST = "194.95.108.135"
DEFAULT_OLLAMA_PORT = "11434"

_ssh_proc: subprocess.Popen | None = None
_mlflow_proc: subprocess.Popen | None = None


def _prepend_pythonpath() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_dotenv_into_os() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ENV_PATH)
    load_dotenv()


def upsert_dotenv(updates: dict[str, str]) -> None:
    """Drop existing lines for keys in ``updates``, append fresh KEY=value pairs."""
    keys = set(updates.keys())
    lines_out: list[str] = []
    if ENV_PATH.exists():
        for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
            key = raw.split("=", 1)[0].strip() if "=" in raw else None
            if key and key in keys:
                continue
            lines_out.append(raw)
    for k in sorted(updates.keys()):
        lines_out.append(f"{k}={updates[k]}")
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def ensure_env_keys() -> tuple[str, str, str, str]:
    """Ensure RZ_KENNUNG and RZ_SSH_HOST exist (persist defaults). Returns host, user, local_port, remote_port."""
    _load_dotenv_into_os()
    updates: dict[str, str] = {}

    rz_user = (os.getenv("RZ_KENNUNG") or "").strip()
    if not rz_user:
        rz_user = input("RZ-Kennung (SSH username): ").strip()
        if not rz_user:
            print("[dev] RZ-Kennung is required.", file=sys.stderr)
            sys.exit(1)
        updates["RZ_KENNUNG"] = rz_user

    host = (os.getenv("RZ_SSH_HOST") or "").strip()
    if not host:
        host = DEFAULT_RZ_SSH_HOST
        updates["RZ_SSH_HOST"] = host

    local_port = (os.getenv("RZ_SSH_LOCAL_OLLAMA_PORT") or DEFAULT_OLLAMA_PORT).strip()
    remote_port = (os.getenv("RZ_SSH_REMOTE_OLLAMA_PORT") or DEFAULT_OLLAMA_PORT).strip()

    if updates:
        upsert_dotenv(updates)
        _load_dotenv_into_os()

    host = (os.getenv("RZ_SSH_HOST") or host).strip()
    rz_user = (os.getenv("RZ_KENNUNG") or rz_user).strip()
    local_port = (os.getenv("RZ_SSH_LOCAL_OLLAMA_PORT") or local_port).strip()
    remote_port = (os.getenv("RZ_SSH_REMOTE_OLLAMA_PORT") or remote_port).strip()

    return host, rz_user, local_port, remote_port


def maybe_ssh_copy_id(host: str, rz_user: str) -> None:
    ans = input("Copy SSH public key now via ssh-copy-id? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        return
    if sys.platform == "win32" and not shutil.which("ssh-copy-id"):
        print(
            "[dev] ssh-copy-id not found. Use WSL/Git Bash, or manually append your "
            ".ssh/*.pub to ~/.ssh/authorized_keys on the server.",
            file=sys.stderr,
        )
        return
    if not shutil.which("ssh-copy-id"):
        print(
            "[dev] ssh-copy-id not on PATH (install openssh-client / OpenSSH). Skipping.",
            file=sys.stderr,
        )
        return
    target = f"{rz_user}@{host}"
    print(f"[dev] Running: ssh-copy-id {target}")
    r = subprocess.run(["ssh-copy-id", target])
    if r.returncode != 0:
        print("[dev] ssh-copy-id exited non-zero; fix keys manually before tunnel.", file=sys.stderr)


def start_ssh_tunnel(host: str, rz_user: str, local_port: str, remote_port: str) -> subprocess.Popen:
    spec = f"{local_port}:localhost:{remote_port}"
    target = f"{rz_user}@{host}"
    cmd = [
        "ssh",
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        spec,
        target,
    ]
    print(f"[dev] SSH tunnel: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    except FileNotFoundError:
        print("[dev] ssh executable not found.", file=sys.stderr)
        sys.exit(1)
    import time

    time.sleep(1.2)
    if proc.poll() is not None:
        print(
            "[dev] SSH tunnel exited immediately. Ensure your public key is authorized on "
            f"{target} (ssh-copy-id or manual ~/.ssh/authorized_keys).",
            file=sys.stderr,
        )
        sys.exit(1)
    return proc


def start_mlflow_server() -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        "sqlite:///./mlflow.db",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--workers",
        "1",
    ]
    print(f"[dev] MLflow: {' '.join(cmd)} (cwd={PROJECT_ROOT})")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("[dev] mlflow module not found; activate conda env and pip install mlflow deps.", file=sys.stderr)
        sys.exit(1)
    import time

    time.sleep(1.0)
    if proc.poll() is not None:
        print("[dev] MLflow server exited immediately.", file=sys.stderr)
        sys.exit(1)
    print("[dev] MLflow UI: http://127.0.0.1:5000")
    return proc


def cleanup_procs(*_args: object) -> None:
    global _ssh_proc, _mlflow_proc
    for proc in (_ssh_proc, _mlflow_proc):
        if proc is None or proc.poll() is not None:
            continue
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def register_cleanup() -> None:
    atexit.register(cleanup_procs)

    def _sig(_signum: int, _frame: object | None) -> None:
        cleanup_procs()
        sys.exit(128 + _signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig)
        except (ValueError, OSError):
            pass


def list_existing_evaluation_csvs_relative() -> list[str]:
    _prepend_pythonpath()
    from agents.shared import get_data_dir
    from evaluation.datasets import SEED_CSVS, VERSATILE_PRODUCTION_RELDIR

    data_dir = get_data_dir()
    rel: list[str] = list(SEED_CSVS)
    versatile_dir = data_dir / VERSATILE_PRODUCTION_RELDIR
    if versatile_dir.is_dir():
        extras = sorted(p.relative_to(data_dir).as_posix() for p in versatile_dir.glob("*.csv"))
        seen = set(rel)
        for extra in extras:
            if extra not in seen:
                rel.append(extra)
                seen.add(extra)
    existing = [r for r in rel if (data_dir / r).exists()]
    if not existing:
        print("[dev] No evaluation CSVs found under data/.", file=sys.stderr)
        sys.exit(1)
    return existing


def prompt_live_debug() -> bool:
    a = input("Enable live LLM token / pipeline stream output? [y/N]: ").strip().lower()
    return a in ("y", "yes")


def prompt_preview_rows() -> int:
    raw = input("Preview rows for pipeline CSV preview [5]: ").strip()
    if not raw:
        return 5
    try:
        n = int(raw)
        return n if n > 0 else 5
    except ValueError:
        return 5


def apply_stream_env(enable: bool) -> None:
    if enable:
        os.environ["OLLAMA_STREAM_STDOUT"] = "1"
    else:
        os.environ.pop("OLLAMA_STREAM_STDOUT", None)


def subprocess_env_for_menu_stream(stream: bool) -> dict[str, str]:
    """Env passed to pipeline/eval subprocesses.

    Syncs the parent process, copies current environ, then pins ``OLLAMA_STREAM_STDOUT``
    so (1) a stale copy-before-mutation bug cannot leak ``=1`` into the child and
    (2) ``agents.shared``'s import-time ``load_dotenv`` cannot resurrect ``=1`` from
    ``.env`` when the user declined streaming (dotenv skips keys already set).
    """
    apply_stream_env(stream)
    env = os.environ.copy()
    env["OLLAMA_STREAM_STDOUT"] = "1" if stream else "0"
    return env


def run_pipeline_choice(csv_rel: str, stream: bool, preview_rows: int) -> None:
    _prepend_pythonpath()
    cmd = [
        sys.executable,
        "-m",
        "orchestrator.pipeline",
        csv_rel,
        "--preview-rows",
        str(preview_rows),
    ]
    if stream:
        cmd.append("--stream-pipeline-output")
    env = subprocess_env_for_menu_stream(stream)
    print(f"[dev] Running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)


def run_eval_choice(stream: bool) -> None:
    _prepend_pythonpath()
    env = subprocess_env_for_menu_stream(stream)
    cmd = [sys.executable, "-m", "evaluation.run_eval", "--with-judge"]
    print(f"[dev] Running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main_menu(stream: bool) -> None:
    print("\nWhat do you want to do?")
    print("  1) Run pipeline (pick CSV)")
    print("  2) Run evaluation with LLM judge (all datasets in seed harness)")
    choice = input("Choice [1-2]: ").strip()

    if choice == "2":
        run_eval_choice(stream)
        return
    if choice != "1":
        print("[dev] Invalid choice.", file=sys.stderr)
        sys.exit(1)

    csvs = list_existing_evaluation_csvs_relative()
    print("\nCSV paths (relative to data/):")
    for i, rel in enumerate(csvs, start=1):
        print(f"  {i}) {rel}")
    idx_raw = input(f"Select CSV [1-{len(csvs)}]: ").strip()
    try:
        idx = int(idx_raw)
    except ValueError:
        print("[dev] Invalid index.", file=sys.stderr)
        sys.exit(1)
    if idx < 1 or idx > len(csvs):
        print("[dev] Out of range.", file=sys.stderr)
        sys.exit(1)
    preview = prompt_preview_rows()
    run_pipeline_choice(csvs[idx - 1], stream, preview)


def main() -> int:
    global _ssh_proc, _mlflow_proc
    os.chdir(PROJECT_ROOT)
    _prepend_pythonpath()
    _load_dotenv_into_os()

    print("[dev] Project:", PROJECT_ROOT)
    print(
        "[dev] CSV folder spelling is data/versatile_producion_system (repo typo — matches evaluation.datasets)."
    )

    host, rz_user, local_port, remote_port = ensure_env_keys()
    maybe_ssh_copy_id(host, rz_user)

    register_cleanup()
    _ssh_proc = start_ssh_tunnel(host, rz_user, local_port, remote_port)
    _mlflow_proc = start_mlflow_server()

    stream = prompt_live_debug()
    main_menu(stream)

    # Optional: leave servers running until user Ctrl+C — menu subprocess already exited
    cleanup_procs()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cleanup_procs()
        raise SystemExit(130)
