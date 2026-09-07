# SPDX-License-Identifier: GPL-3.0-or-later

"""**Every mode against every combination of the options that apply to it.**

`tests/test_cli.py` runs each mode once, with no flags. That is enough to catch a mode that does
not start and not much else: the CLI's failure mode is a *combination* — a periodic `energy`, an
`optimize` that writes a file, an unrestricted `frequencies` — where one branch reads a key the
native layer does not return and raises `KeyError` after the calculation has already run. Nothing
short of actually running the combination finds that, because the branch is only reached there.

Two properties are checked, and they are different:

1. **Every combination exits cleanly.** Not necessarily zero — asking for `ir` on a crystal is a
   user error and must be refused — but never with a traceback, and never with a bare `KeyError`.
2. **The two front ends produce the same bytes.** The packaging claims the pip-installed Python
   CLI mirrors the Rust one "same modes, same flags, same output"; this is where that claim is
   tested rather than asserted. It skips when the Rust binary has not been built.

The first runs the Python CLI **in process** — several hundred invocations at ~0.3 s of
interpreter startup each is a test nobody runs. The second is a subprocess comparison over a
representative subset, which is where the cost has to be paid.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("am1_rs")

from am1_rs.__main__ import main as python_cli_main  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- structures

WATER = """3
water
O   0.00000000   0.00000000   0.11730000
H   0.00000000   0.75720000  -0.46920000
H   0.00000000  -0.75720000  -0.46920000
"""

METHYL = """4
methyl radical
C   0.00000000   0.00000000   0.00000000
H   1.07900000   0.00000000   0.00000000
H  -0.53950000   0.93440000   0.00000000
H  -0.53950000  -0.93440000   0.00000000
"""

# A one-dimensional H2 chain, carrying its cell on the comment line. The bond is AM1's own
# 0.6766 A, so the structure is not up the repulsive wall; see `tests/pbc_phonon.rs`.
CHAIN = """2
H2 chain Lattice="60.0 0.0 0.0 0.0 60.0 0.0 0.0 0.0 3.0" pbc="F F T"
H   0.00000000   0.00000000   0.00000000
H   0.00000000   0.00000000   0.67660000
"""


@pytest.fixture(scope="module")
def structures(tmp_path_factory) -> dict[str, str]:
    directory = tmp_path_factory.mktemp("cli-matrix")
    out = {}
    for name, text in (("water", WATER), ("methyl", METHYL), ("chain", CHAIN)):
        path = directory / f"{name}.xyz"
        path.write_text(text, encoding="utf-8")
        out[name] = str(path)
    out["_dir"] = str(directory)
    return out


# --------------------------------------------------------------------------- the matrix

MODES = [
    "energy",
    "gradient",
    "optimize",
    "frequencies",
    "phonons",
    "charges",
    "orbitals",
    "ir",
    "molden",
]

#: Options that apply to any mode. Each entry is a complete flag group.
GENERAL = [
    [],
    ["--method", "rm1"],
    ["--rhf"],
    ["--uhf"],
    ["--reference", "auto"],
    ["--field", "0.0", "0.0", "0.002"],
    ["--charge", "0"],
]

#: Options that only mean something to one mode, keyed by it.
PER_MODE = {
    # Divide-and-conquer, on the three modes that accept it. `--dc-core 2` forces a real
    # partition out of a three-atom water rather than the degenerate one-subsystem case, which is
    # what a default core size gives on anything this small.
    "energy": [[], ["--dc"], ["--dc-core", "2", "--dc-buffer", "4.0"]],
    "gradient": [[], ["--dc-core", "2"]],
    "optimize": [[], ["--opt-output", "OUT.xyz"], ["--dc-core", "2"]],
    "molden": [
        [],
        ["--molden-output", "OUT.molden"],
        ["--molden-basis", "sto"],
        ["--molden-basis", "gto", "--molden-primitives", "3"],
        ["--molden-orthogonal"],
    ],
    "charges": [[], ["--mulliken"], ["--mol2-output", "OUT.mol2"]],
    "orbitals": [[], ["--orbital-coefficients"]],
}

#: How each structure is made periodic (or not) on the command line.
PERIODIC = {
    "molecular": [],
    "cell-flag": ["--cell", "8.0", "--kpts", "1", "1", "1"],
    "per-axis-z": ["--cell", "8.0", "--no-pbc", "--pbc-z", "--kpts", "1", "1", "2"],
    "per-axis-xy": ["--cell", "8.0", "--no-pbc", "--pbc-x", "--pbc-y", "--kpts", "2", "2", "1"],
    "pbc-token": ["--cell", "8.0", "--pbc", "xyz", "--kpts", "1", "1", "1"],
    "dropped": ["--cell", "8.0", "--no-pbc"],
}

#: Modes with no periodic implementation. They must *say so*, not ignore the cell.
MOLECULAR_ONLY = {"charges", "ir", "molden"}

#: Modes that need a cell, and must say so without one.
PERIODIC_ONLY = {"phonons"}

#: Refusals that are correct answers rather than defects.
#:
#: The matrix is a *crash* hunt: a traceback, a bare `KeyError`, or a mode that silently does the
#: wrong thing. A calculation that declines because the physics does not support it is the CLI
#: working. Each pattern is listed explicitly, so a **new** kind of failure still fails the test
#: rather than being swept in with these.
ACCEPTABLE_REFUSALS = (
    # An SCF that will not converge is a property of the system, not of the front end. An
    # open-shell radical in a periodic cell under RM1 is one such: it is a hard case and it is
    # allowed to say so.
    "did not converge",
    "diverged",
    # A mode asked for something the structure cannot supply.
    "molecular only",
    "need a periodic cell",
    "not periodic",
    # The reference itself is unusable, so the response has no meaning; see `check_cphf`.
    "not positive definite",
    # A uniform field along a periodic axis has no bounded `F.R`, so the perturbation is not
    # lattice-periodic and the run is refused. This entry is new in 0.2.3 and the reason it was
    # not needed before is worth recording: the Python CLI's `pbc_common` simply omitted
    # `electric_field`, so a periodic `--field` was accepted and **ignored** there while the Rust
    # front end applied it and refused. The matrix passed because one of the two front ends was
    # not doing the thing being tested.
    "along a periodic direction",
)

#: Modes that cost a Hessian, an optimization or a supercell. The point of the matrix is to reach
#: every *branch*, not to multiply out every flag against the slowest code in the crate, so these
#: get one structure and a short flag list. Without the split the matrix is forty minutes, which
#: is a test nobody runs, which is the same as not having one.
EXPENSIVE = {"gradient", "optimize", "frequencies", "ir", "phonons"}
EXPENSIVE_GENERAL = [[], ["--method", "rm1"], ["--field", "0.0", "0.0", "0.002"]]


def _substitute(flags: list[str], directory: str) -> list[str]:
    """Point every output-file placeholder at the temporary directory."""
    return [
        os.path.join(directory, f) if f.startswith("OUT.") else f for f in flags
    ]


def commands(structures: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Every (label, argv) the matrix covers."""
    directory = structures["_dir"]
    out: list[tuple[str, list[str]]] = []
    for mode in MODES:
        expensive = mode in EXPENSIVE
        structures_here = ("water", "chain") if expensive else ("water", "methyl", "chain")
        for structure in structures_here:
            spin = ["--multiplicity", "2", "--uhf"] if structure == "methyl" else []
            for periodic_name, periodic in PERIODIC.items():
                if expensive and periodic_name in ("pbc-token", "per-axis-xy"):
                    continue
                # The chain carries its own cell; adding `--cell` on top is covered by the
                # `cell-flag` entry, and repeating it for every other entry only re-runs the
                # same thing.
                if structure == "chain" and periodic_name not in ("molecular", "dropped"):
                    continue
                if structure != "chain" and periodic_name == "dropped":
                    continue
                for general in EXPENSIVE_GENERAL if expensive else GENERAL:
                    # `--rhf` and an open-shell doublet are contradictory, and the CLI is right
                    # to refuse; that refusal is covered once in `test_refusals_agree` rather
                    # than several hundred times here.
                    if spin and ("--rhf" in general or "--reference" in general):
                        continue
                    for specific in PER_MODE.get(mode, [[]]):
                        argv = (
                            [mode, structures[structure]]
                            + periodic
                            + general
                            + _substitute(specific, directory)
                            + spin
                        )
                        label = (
                            f"{mode}/{structure}/{periodic_name}/"
                            f"{'+'.join(general) or 'plain'}/"
                            f"{'+'.join(specific) or 'plain'}"
                        )
                        out.append((label, argv))
    return out


def run_in_process(argv: list[str]) -> tuple[int, str, str]:
    """The Python CLI without an interpreter start, so the matrix is affordable."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = python_cli_main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def test_every_combination_exits_cleanly(structures) -> None:
    """No traceback, no bare exception name, from any combination.

    A `KeyError` reaches the user as `am1-rs: 'electronic_ev'` — a one-word message with no verb,
    which is what a missing dict key looks like once the CLI's blanket `except Exception` has
    caught it. That is the signature this looks for, and it is how the periodic `energy` mode was
    found to be reading a key `pbc_point` did not return.
    """
    failures = []
    for label, argv in commands(structures):
        try:
            code, out, err = run_in_process(argv)
        except BaseException as exc:  # noqa: BLE001 - a crash is the thing being tested
            failures.append(f"{label}: raised {type(exc).__name__}: {exc}")
            continue
        if "Traceback" in err:
            failures.append(f"{label}: printed a traceback\n{err}")
            continue
        mode = argv[0]
        periodic = _is_periodic(argv)
        if mode in MOLECULAR_ONLY and periodic:
            if code == 0:
                failures.append(f"{label}: a molecular-only mode accepted a periodic cell")
            elif "molecular only" not in err:
                failures.append(f"{label}: refused, but not for the stated reason: {err.strip()}")
            continue
        if mode in PERIODIC_ONLY and not periodic:
            if code == 0:
                failures.append(f"{label}: a periodic-only mode accepted a molecule")
            elif "need a periodic cell" not in err:
                failures.append(f"{label}: refused, but not for the stated reason: {err.strip()}")
            continue
        if code != 0:
            # A refusal the physics justifies is the CLI working; anything else is not.
            if not any(reason in err for reason in ACCEPTABLE_REFUSALS):
                failures.append(f"{label}: exited {code}: {err.strip()[:200]}")
            elif not err.strip():
                failures.append(f"{label}: exited {code} without saying why")
            continue
        if not out.strip():
            failures.append(f"{label}: printed nothing")
        # The message a missing key produces: the whole line is a quoted identifier.
        for line in err.splitlines():
            body = line.removeprefix("am1-rs: ").strip()
            if body.startswith("'") and body.endswith("'") and " " not in body:
                failures.append(f"{label}: looks like a KeyError: {line}")
    assert not failures, (
        f"{len(failures)} of {len(commands(structures))} combinations failed:\n"
        + "\n".join(failures[:40])
    )


def _is_periodic(argv: list[str]) -> bool:
    """Whether this command line ends up with a cell — the CLI's own resolution order."""
    if "--no-pbc" in argv and not any(f in argv for f in ("--pbc-x", "--pbc-y", "--pbc-z")):
        after = argv[argv.index("--no-pbc") + 1 :]
        if not any(f.startswith("--pbc") for f in after):
            return False
    if "--cell" in argv:
        return True
    # The chain carries a Lattice= on its comment line.
    return any(a.endswith("chain.xyz") for a in argv)


# --------------------------------------------------------------- parity with the Rust CLI


def rust_cli() -> pathlib.Path | None:
    """The most recently built `am1_rs_cli`, whichever profile and target directory it is in.

    `CARGO_TARGET_DIR` is honoured because it is set whenever the build is moved off the source
    tree — which on Windows is the standard workaround for the linker's file locking. Without it
    this test skips silently on exactly the machine doing the development, which is the worst
    place for a parity check to be absent.
    """
    roots = [ROOT / "target"]
    target = os.environ.get("CARGO_TARGET_DIR")
    if target:
        roots.insert(0, pathlib.Path(target))
    candidates = [
        root / profile / name
        for root in roots
        for profile in ("release", "fast", "debug")
        for name in ("am1_rs_cli", "am1_rs_cli.exe")
    ]
    existing = [c for c in candidates if c.exists()]
    if not existing:
        return None
    newest = max(existing, key=lambda p: p.stat().st_mtime)
    # A binary older than its sources is worse than no binary — see the same guard in
    # `tests/test_cli.py`, which a stale `./target` turned into four bogus parity failures.
    sources = [
        p.stat().st_mtime
        for d in (ROOT / "src", ROOT / "Cargo.toml")
        for p in ([d] if d.is_file() else d.rglob("*.rs"))
    ]
    return None if sources and newest.stat().st_mtime < max(sources) else newest


def _subprocess_python(argv: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, "-m", "am1_rs", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        env=env,
    )


def _subprocess_rust(binary: pathlib.Path, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary), *argv], capture_output=True, text=True, encoding="utf-8", cwd=ROOT
    )


#: One command per mode per interesting branch — enough to cover every printing routine on both
#: sides without paying for the whole matrix twice.
def parity_commands(structures) -> list[tuple[str, list[str]]]:
    d = structures["_dir"]
    water, methyl, chain = structures["water"], structures["methyl"], structures["chain"]
    uhf = ["--multiplicity", "2", "--uhf"]
    return [
        ("energy", ["energy", water]),
        ("energy/field", ["energy", water, "--field", "0.0", "0.0", "0.005"]),
        ("energy/rm1", ["energy", water, "--method", "rm1"]),
        ("energy/uhf", ["energy", methyl] + uhf),
        ("energy/periodic", ["energy", chain, "--kpts", "1", "1", "4"]),
        ("energy/per-axis", ["energy", water, "--cell", "8.0", "--no-pbc", "--pbc-z"]),
        ("gradient", ["gradient", water]),
        ("gradient/periodic", ["gradient", chain, "--kpts", "1", "1", "4"]),
        ("optimize", ["optimize", water]),
        ("optimize/file", ["optimize", water, "--opt-output", os.path.join(d, "o.xyz")]),
        ("optimize/periodic", ["optimize", chain, "--kpts", "1", "1", "4"]),
        (
            "optimize/relax-cell",
            ["optimize", chain, "--kpts", "1", "1", "4", "--relax-cell"],
        ),
        # Divide-and-conquer. `--dc-core 2` forces a real partition out of a three-atom water,
        # so the subsystem line in the output is not the degenerate "1 subsystem" case; the
        # defaults are shared across five layers (see `tests/dc_optimize.rs`) and a drift in any
        # of them shows up here as two front ends printing different numbers.
        ("energy/dc", ["energy", water, "--dc"]),
        ("energy/dc-core", ["energy", water, "--dc-core", "2"]),
        ("energy/dc-buffer", ["energy", water, "--dc", "--dc-buffer", "4.0"]),
        ("gradient/dc", ["gradient", water, "--dc-core", "2"]),
        ("optimize/dc", ["optimize", water, "--dc-core", "2"]),
        ("frequencies", ["frequencies", water]),
        ("frequencies/uhf", ["frequencies", methyl] + uhf),
        ("frequencies/periodic", ["frequencies", chain, "--kpts", "1", "1", "4"]),
        ("phonons", ["phonons", chain, "--supercell", "1", "1", "2", "--kpts", "1", "1", "2"]),
        ("orbitals", ["orbitals", water]),
        ("orbitals/coefficients", ["orbitals", water, "--orbital-coefficients"]),
        ("orbitals/uhf", ["orbitals", methyl, "--orbital-coefficients"] + uhf),
        ("ir", ["ir", water]),
        ("charges", ["charges", water]),
        ("charges/mulliken", ["charges", water, "--mulliken"]),
        ("charges/mol2", ["charges", water, "--mol2-output", os.path.join(d, "c.mol2")]),
        ("molden", ["molden", water]),
        ("molden/sto", ["molden", water, "--molden-basis", "sto"]),
        ("molden/orthogonal", ["molden", water, "--molden-orthogonal"]),
        ("molden/primitives", ["molden", water, "--molden-primitives", "4"]),
        ("molden/uhf", ["molden", methyl] + uhf),
        ("refusal/ir-periodic", ["ir", chain]),
        ("refusal/bad-pbc", ["energy", water, "--pbc", "q"]),
        ("refusal/bad-cell", ["energy", water, "--cell", "1.0", "2.0"]),
        ("refusal/unknown-mode", ["nonsense", water]),
        ("refusal/unknown-flag", ["energy", water, "--not-a-flag"]),
    ]


def test_the_two_front_ends_agree(structures) -> None:
    binary = rust_cli()
    if binary is None:
        pytest.skip("the Rust CLI has not been built (cargo build --bin am1_rs_cli)")
    failures = []
    for label, argv in parity_commands(structures):
        mine = _subprocess_python(argv)
        theirs = _subprocess_rust(binary, argv)
        if (mine.returncode == 0) != (theirs.returncode == 0):
            failures.append(
                f"{label}: python exited {mine.returncode}, rust {theirs.returncode}\n"
                f"  python stderr: {mine.stderr.strip()[:200]}\n"
                f"  rust stderr:   {theirs.stderr.strip()[:200]}"
            )
            continue
        if mine.stdout != theirs.stdout:
            failures.append(
                f"{label}: stdout differs\n--- rust ---\n{theirs.stdout}\n"
                f"--- python ---\n{mine.stdout}"
            )
    assert not failures, "\n\n".join(failures[:6])


def test_refusals_agree(structures) -> None:
    """A contradiction has to be refused by both, with a non-zero exit and no traceback."""
    water, methyl = structures["water"], structures["methyl"]
    for argv in (
        ["energy", methyl, "--multiplicity", "2", "--rhf"],
        ["energy", water, "--multiplicity", "2"],
        ["energy", water, "--method", "nonsense"],
        ["energy", water, "--molden-primitives", "99"],
        ["phonons", water],
    ):
        code, _, err = run_in_process(argv)
        assert code != 0, f"{argv} should have been refused"
        assert "Traceback" not in err, err
        assert err.strip(), f"{argv} was refused without saying why"
