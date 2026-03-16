from __future__ import annotations

import os
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=os.path.dirname(os.path.realpath(__file__)))


def main() -> int:
    req = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    if not os.path.isfile(req):
        print("[IAT] requirements.txt not found, skipping dependency install.")
        return 0

    # Use the current interpreter to avoid mismatching ComfyUI's Python.
    cmd = [sys.executable, "-m", "pip", "install", "-r", req]
    print(f"[IAT] Installing dependencies: {' '.join(cmd)}")
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

