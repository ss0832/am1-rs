#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-resolved profile of the main code paths, via the ``AM1_TIMING`` environment variable.

    python tools/profile.py                    # 150-atom cluster, energy and gradient
    python tools/profile.py --atoms 400
    python tools/profile.py --modes energy frequencies

**Measure before optimizing.** The last round of performance work on this crate began from an
assumption carried over from a sister crate and spent effort in the wrong place; the real hot
spot was somewhere nobody had guessed, and cost 24× on its own once found. The numbers this
prints are the whole point — do not skip to editing code.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def binary_path() -> Path:
    name = "am1_rs_cli.exe" if sys.platform == "win32" else "am1_rs_cli"
    exe = ROOT / "target" / "release" / name
    if exe.exists():
        return exe
    found = shutil.which("am1_rs_cli")
    if found:
        return Path(found)
    raise SystemExit(f"release binary not found at {exe} — run: cargo build --release")


def make_structure(atoms: int, shape: str) -> Path:
    """Generate a cluster or chain of roughly ``atoms`` atoms into a temporary file."""
    path = Path(tempfile.gettempdir()) / f"am1_profile_{shape}_{atoms}.xyz"
    if not path.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "make_water_cluster.py"),
             shape, str(atoms), str(path)],
            check=True,
            capture_output=True,
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atoms", type=int, default=150)
    parser.add_argument("--shape", default="cluster", choices=["cluster", "chain"])
    parser.add_argument("--modes", nargs="+", default=["energy", "gradient"])
    args = parser.parse_args()

    exe = binary_path()
    xyz = make_structure(args.atoms, args.shape)
    print(f"structure: {xyz}")

    env = dict(os.environ, AM1_TIMING="1")
    for mode in args.modes:
        print(f"\n=== {mode} ===")
        result = subprocess.run([str(exe), mode, str(xyz)],
                                capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"FAILED (exit {result.returncode})")
            print(result.stderr.strip())
            continue
        # The timing report goes to stderr; the per-atom output goes to stdout and is noise here.
        for line in result.stderr.splitlines():
            if re.search(r"timing|\d+\.\d+ s\s+\d+\.\d+ %", line):
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
