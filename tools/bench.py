#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wall-clock and peak-memory scaling benchmark for the ``am1_rs_cli`` release binary.

    python tools/bench.py                          # energy + gradient over the standard ladder
    python tools/bench.py --modes energy           # one mode
    python tools/bench.py --shapes chain           # chain only
    python tools/bench.py --json bench_before.json

Generate the structures first::

    python tools/make_water_cluster.py cluster 801 examples/bench/water_cluster_801.xyz

The fitted exponent is a least-squares slope of log(time) against log(atoms) over **all** sizes
measured. A two-point slope between the largest pair is more sensitive to whatever else the
machine was doing than to the algorithm: one run of the divide-and-conquer benchmark produced an
apparent exponent of 0.90 for an ``O(N³)`` diagonalization that way. Treat the number as
indicative and give the machine to the benchmark.

Peak memory needs ``psutil``; without it the column reads ``-`` and everything else still works.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - optional
    psutil = None

ROOT = Path(__file__).resolve().parent.parent


def binary_path() -> Path:
    """The release CLI, whatever this platform calls it."""
    name = "am1_rs_cli.exe" if sys.platform == "win32" else "am1_rs_cli"
    exe = ROOT / "target" / "release" / name
    if exe.exists():
        return exe
    found = shutil.which("am1_rs_cli")
    if found:
        return Path(found)
    raise SystemExit(f"release binary not found at {exe} — run: cargo build --release")


def run_timed(exe: Path, args: list[str]) -> dict:
    """Run to completion, returning wall seconds and peak resident memory in MB.

    Output is drained by ``communicate`` rather than read after the fact: the CLI prints a line
    per atom, which is well past a pipe's default buffer at these sizes, and a full pipe
    deadlocks the child.

    Memory is sampled sparsely — every 100 ms — for a reason worth keeping. An earlier
    PowerShell version of this script polled tightly with a 15 ms sleep, and because Windows'
    sleep granularity is ~15.6 ms the polling itself inflated the wall time it was supposed to be
    measuring, by around 60 %: a 10.5 s run read as 16.8 s.
    """
    start = time.perf_counter()
    process = subprocess.Popen(
        [str(exe), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    peak_bytes = 0
    if psutil is not None:
        try:
            handle = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak_bytes = max(peak_bytes, handle.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(0.1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - start
    return {
        "seconds": round(elapsed, 3),
        "peak_mb": round(peak_bytes / (1024 * 1024), 1) if peak_bytes else None,
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def fitted_exponent(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of log(y) against log(x). ``None`` if there is not enough to fit."""
    usable = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(usable) < 2:
        return None
    n = len(usable)
    mean_x = sum(math.log(x) for x, _ in usable) / n
    mean_y = sum(math.log(y) for _, y in usable) / n
    num = sum((math.log(x) - mean_x) * (math.log(y) - mean_y) for x, y in usable)
    den = sum((math.log(x) - mean_x) ** 2 for x, _ in usable)
    return num / den if den > 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=["energy", "gradient"])
    parser.add_argument("--shapes", nargs="+", default=["cluster", "chain"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[102, 201, 399, 801])
    parser.add_argument("--json", default=None, help="write the results here")
    args = parser.parse_args()

    exe = binary_path()
    if psutil is None:
        print("note: psutil not installed — peak memory will not be reported", file=sys.stderr)

    results = []
    for shape in args.shapes:
        for mode in args.modes:
            print(f"\n=== {shape} / {mode} ===")
            print(f"{'atoms':>7}  {'seconds':>10}  {'peak MB':>10}  {'ms/atom':>10}")
            series: list[tuple[float, float]] = []
            for n in args.sizes:
                xyz = ROOT / "examples" / "bench" / f"water_{shape}_{n}.xyz"
                if not xyz.exists():
                    print(f"{n:>7}  (missing {xyz})")
                    continue
                r = run_timed(exe, [mode, str(xyz)])
                if r["exit_code"] != 0:
                    print(f"{n:>7}  FAILED (exit {r['exit_code']})")
                    print(r["stderr"].strip())
                    continue
                peak = f"{r['peak_mb']:.1f}" if r["peak_mb"] is not None else "-"
                print(f"{n:>7}  {r['seconds']:>10.3f}  {peak:>10}  "
                      f"{1000.0 * r['seconds'] / n:>10.3f}")
                results.append({"shape": shape, "mode": mode, "atoms": n,
                                "seconds": r["seconds"], "peak_mb": r["peak_mb"]})
                series.append((float(n), r["seconds"]))

            exponent = fitted_exponent(series)
            if exponent is not None:
                print(f"  fitted exponent over {len(series)} sizes: N^{exponent:.2f}  "
                      f"(indicative only — see the module docstring)")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
