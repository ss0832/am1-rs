#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the time actually goes in a divide-and-conquer run.

Builds water clusters of increasing size, runs the CLI's divide-and-conquer path under
``AM1_TIMING=1``, and prints the per-phase breakdown next to the total. The point is to find the
bottleneck by measuring it rather than by reasoning about asymptotics: an ``O(N^2)`` term with a
small prefactor can sit well below an ``O(N)`` term with a large one across the whole range
anybody actually runs.

    python tools/profile_dc.py [--sizes 64 128 256 512]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANG = 1.0


def water_cluster(n_waters: int, spacing: float = 4.0) -> str:
    """`n_waters` molecules on a cubic grid, spaced far enough not to clash.

    4.0 Å is not arbitrary: at 3.1 Å with arbitrary orientations the hydrogens of neighbouring
    molecules land 1.2-1.4 Å apart, which is a broken structure rather than a hard test case,
    and it produces SCF failures that read like a method defect.
    """
    side = 1
    while side ** 3 < n_waters:
        side += 1
    geom = [
        (8, (0.0, 0.0, 0.0)),
        (1, (0.9614, 0.0, 0.0)),
        (1, (-0.2246, 0.9348, 0.0)),
    ]
    lines = []
    made = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if made >= n_waters:
                    break
                ox, oy, oz = i * spacing, j * spacing, k * spacing
                for z, (x, y, zc) in geom:
                    sym = {8: "O", 1: "H"}[z]
                    lines.append(f"{sym} {x + ox:.6f} {y + oy:.6f} {zc + oz:.6f}")
                made += 1
    return f"{len(lines)}\nwater cluster, {n_waters} molecules\n" + "\n".join(lines) + "\n"


PHASE = re.compile(r"^\s*(\S+)\s+([0-9.]+)\s*(?:ms|s)\b", re.MULTILINE)


def run(path: Path, mode: str) -> tuple[float, dict[str, float]]:
    env = dict(os.environ, AM1_TIMING="1")
    exe = ROOT / "target" / "release" / ("am1_rs_cli.exe" if os.name == "nt" else "am1_rs_cli")
    if not exe.exists():
        sys.exit(f"{exe} not found — run: cargo build --release")
    import time

    start = time.perf_counter()
    proc = subprocess.run(
        [str(exe), mode, str(path)],
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(f"{mode} failed on {path}")
    phases = {}
    for name, value in PHASE.findall(proc.stderr):
        try:
            phases[name] = float(value)
        except ValueError:
            pass
    return elapsed, phases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[27, 64, 125, 216])
    ap.add_argument("--mode", default="energy")
    args = ap.parse_args()

    print(f"{'waters':>7} {'atoms':>7} {'wall (s)':>10}   phases (largest first)")
    for n in args.sizes:
        with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as fh:
            fh.write(water_cluster(n))
            path = Path(fh.name)
        try:
            elapsed, phases = run(path, args.mode)
        finally:
            path.unlink(missing_ok=True)
        top = sorted(phases.items(), key=lambda kv: -kv[1])[:6]
        summary = "  ".join(f"{k}={v:.0f}" for k, v in top)
        print(f"{n:7d} {n * 3:7d} {elapsed:10.2f}   {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
