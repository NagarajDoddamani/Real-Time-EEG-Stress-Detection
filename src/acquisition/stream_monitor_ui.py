"""Compatibility launcher for the combined Electron EEG desktop application."""
from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    app_dir = project_root / "apps" / "device2-dashboard"
    try:
        return subprocess.call(["npm", "start"], cwd=app_dir)
    except FileNotFoundError as exc:
        raise SystemExit("npm was not found. Install Node.js, then run `npm start` from apps/device2-dashboard.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
