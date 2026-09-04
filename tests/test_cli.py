# SPDX-License-Identifier: GPL-3.0-or-later

"""The console script `pip install` puts on PATH.

This exists because the CLI shipped broken once and nothing noticed. `am1_rs/__main__.py` reads
keys out of the dicts the native bindings return, and those are plain dictionaries — a key that
does not exist raises `KeyError` at the moment it is printed, not at import, and not in any test
that only imports the library. Three of the five modes were reading names the bindings never
emitted (`total_ev`, `max_gradient`, `forces`, `positions`, `steps`), which meant `am1-rs energy`
exited non-zero on a water molecule while every other test in the suite passed.

So each mode is actually run, end to end, through the same entry point a user gets.

The second test is the stronger one: the packaging says the Python CLI mirrors the Rust one
"same modes, same flags, same output", and that claim is checked by diffing them rather than
asserted in a README. It skips when the Rust binary has not been built, since a wheel-only
environment has no way to produce one.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import sysconfig

import pytest

pytest.importorskip("am1_rs")

ROOT = pathlib.Path(__file__).resolve().parent.parent
WATER = ROOT / "examples" / "water.xyz"
MODES = [
    "energy",
    "gradient",
    "optimize",
    "frequencies",
    "charges",
    "orbitals",
    "ir",
    "molden",
]


def _run(
    args: list[str], io_encoding: str | None = None
) -> subprocess.CompletedProcess:
    """Run a CLI and decode its output as **UTF-8**, whatever the machine's locale is.

    Two separate encodings are in play and conflating them hid a real bug for a whole release.

    *Decoding*, here in the parent: `text=True` alone would use
    `locale.getpreferredencoding()`, so on a Japanese Windows box (cp932) reading UTF-8 output
    raises `UnicodeDecodeError` in subprocess's reader thread — a failure that has nothing to do
    with the CLIs. Hence the explicit `encoding="utf-8"`.

    *Encoding*, in the child: `PYTHONIOENCODING` is *removed* from its environment unless a test
    asks for one. It was set on the development machine, which silently gave every child a UTF-8
    stdout and made the parity tests pass while `am1-rs energy` was in fact dying half-way
    through its output on any ordinary cp932 or `C`-locale machine. The suite has to reproduce
    the user's environment, not the developer's.
    """
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    if io_encoding is not None:
        env["PYTHONIOENCODING"] = io_encoding
    return subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", cwd=ROOT, env=env
    )


def run_python_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke the module the console script points at, in this interpreter."""
    return _run([sys.executable, "-m", "am1_rs", *args])


def console_script() -> pathlib.Path | None:
    """The `am1-rs` executable `pip install` puts on PATH, if this env has one.

    This is a *different* entry point from `python -m am1_rs`: pip generates a launcher that
    imports `am1_rs.__main__:main` per `[project.scripts]` in pyproject.toml. Exercising it is
    the only way to check that what a user gets from `pip install am1-rs-python` actually runs —
    the module form can work while the packaged script is missing or misdeclared.
    """
    # `sysconfig` knows where this interpreter installs scripts; guessing does not. A venv puts
    # them beside `python.exe`, a system Python on Windows puts them in a sibling `Scripts/`, and
    # a `pip install --user` puts them somewhere else again. Miss any of those and this test
    # skips exactly where it matters — which it did, silently, until the user scheme was added.
    schemes = {sysconfig.get_default_scheme()}
    try:
        schemes.add(sysconfig.get_preferred_scheme("user"))
    except (AttributeError, KeyError, OSError):
        pass
    candidates = []
    for scheme in schemes:
        try:
            path = sysconfig.get_path("scripts", scheme)
        except (KeyError, OSError):
            continue
        if path:
            candidates.append(pathlib.Path(path))
    exe = pathlib.Path(sys.executable).parent
    candidates += [exe, exe / "Scripts", exe / "bin"]
    for directory in candidates:
        for name in ("am1-rs.exe", "am1-rs"):
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def rust_cli() -> pathlib.Path | None:
    """The most recently built `am1_rs_cli`, whichever profile it came from.

    By modification time rather than by a fixed profile order: a stale `release` binary left over
    from an earlier version would otherwise shadow the `fast` one being iterated on, and the
    diff below would compare today's Python CLI against last week's Rust one — which reads as a
    parity failure rather than as a stale build.
    """
    candidates = [
        ROOT / "target" / profile / name
        for profile in ("release", "fast", "debug")
        for name in ("am1_rs_cli", "am1_rs_cli.exe")
    ]
    existing = [c for c in candidates if c.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_runs(mode: str) -> None:
    result = run_python_cli(mode, str(WATER))
    assert result.returncode == 0, (
        f"`am1-rs {mode}` exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert result.stdout.strip(), f"`am1-rs {mode}` printed nothing"


def test_energy_reports_the_known_heat_of_formation() -> None:
    # Not just "it ran": the number has to be the AM1 one for water, so a mode that printed a
    # plausible-looking but wrong quantity would still fail here.
    result = run_python_cli("energy", str(WATER))
    assert result.returncode == 0, result.stderr
    line = next(l for l in result.stdout.splitlines() if "heat of formation" in l)
    kcal = float(line.split("(")[1].split()[0])
    assert kcal == pytest.approx(-59.216, abs=0.01), line


@pytest.mark.parametrize("mode", MODES)
def test_output_is_pure_ascii(mode: str) -> None:
    """Both front ends must print ASCII, so that no locale can fail to render it.

    This is the property that actually travels. `pip install` puts this CLI on machines whose
    stdout encoding is cp932, cp1252 or plain ASCII, and Python encodes `print` with that codec:
    a single `·` in the dipole line raised `UnicodeEncodeError` after six lines had already been
    written, so `am1-rs energy` exited 1 with a truncated report. The Rust binary never raised —
    it writes UTF-8 regardless — but rendered mojibake on the same console, so the two front ends
    did not agree either.

    Asserting the bytes, rather than only that the command exits 0, is deliberate: the CLI now
    forces its streams to UTF-8, which would mask a reintroduced non-ASCII character behind a
    clean exit while still printing mojibake for the user.
    """
    binary = rust_cli()
    runners = [("python", run_python_cli)]
    if binary is not None:
        runners.append(("rust", lambda *a: _run([str(binary), *a])))
    for name, run in runners:
        result = run(mode, str(WATER))
        assert result.returncode == 0, result.stderr
        for stream, text in (("stdout", result.stdout), ("stderr", result.stderr)):
            if text.isascii():
                continue
            bad = sorted({c for c in text if not c.isascii()})
            offending = next(l for l in text.splitlines() if not l.isascii())
            raise AssertionError(
                f"the {name} CLI's {stream} for mode `{mode}` is not ASCII: "
                f"{bad} in {offending!r}"
            )


def test_it_runs_when_the_locale_cannot_encode_anything(tmp_path) -> None:
    """`am1-rs energy` under an ASCII stdout — a `C`-locale container, the harshest real case.

    The regression this pins is not hypothetical: it was masked on the development machine by a
    `PYTHONIOENCODING=utf-8` that happened to be set in the shell, so the whole parity suite
    passed against a CLI that was broken everywhere else. `_run` now strips that variable; this
    test goes further and forces the least capable codec there is.
    """
    result = _run(
        [sys.executable, "-m", "am1_rs", "energy", str(WATER)], io_encoding="ascii"
    )
    assert result.returncode == 0, (
        f"exited {result.returncode} under an ASCII stdout\n{result.stderr}"
    )
    assert "UnicodeEncodeError" not in result.stderr, result.stderr
    assert "dipole" in result.stdout, result.stdout


@pytest.mark.parametrize("mode", MODES)
def test_the_pip_installed_console_script_runs(mode: str) -> None:
    """`pip install am1-rs-python` must give a working `am1-rs` command, in every mode.

    A wheel cannot ship the Rust `am1_rs_cli` binary alongside the extension module, so this
    script *is* the CLI for everyone who installs from PyPI. It is also the entry point that the
    locale bug broke: `am1-rs energy` exited 1 part-way through its output on any machine whose
    stdout could not encode a middle dot, which is most of them outside a UTF-8 locale.
    """
    script = console_script()
    if script is None:
        pytest.skip("no console script in this environment (pip install the wheel to test it)")
    result = _run([str(script), mode, str(WATER)])
    assert result.returncode == 0, (
        f"`am1-rs {mode}` exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert result.stdout.strip(), f"`am1-rs {mode}` printed nothing"
    # And it must agree with `python -m am1_rs`, since they are the same code reached two ways.
    assert result.stdout == run_python_cli(mode, str(WATER)).stdout


def test_a_bad_path_fails_cleanly() -> None:
    # A missing file is a user error, not a crash: it should be a message and a non-zero exit,
    # with no traceback.
    result = run_python_cli("energy", str(ROOT / "no-such-file.xyz"))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr, result.stderr


@pytest.mark.parametrize("mode", MODES)
def test_it_matches_the_rust_cli(mode: str) -> None:
    binary = rust_cli()
    if binary is None:
        pytest.skip("the Rust CLI has not been built (cargo build --bin am1_rs_cli)")
    mine = run_python_cli(mode, str(WATER))
    theirs = _run([str(binary), mode, str(WATER)])
    assert mine.returncode == theirs.returncode == 0
    assert mine.stdout == theirs.stdout, (
        f"the Python and Rust CLIs disagree on `{mode}`\n"
        f"--- rust ---\n{theirs.stdout}\n--- python ---\n{mine.stdout}"
    )


def test_the_field_flag_matches_between_the_two_clis() -> None:
    """`--field` takes three values, so it is the flag most likely to be parsed differently."""
    binary = rust_cli()
    if binary is None:
        pytest.skip("the Rust CLI has not been built (cargo build --bin am1_rs_cli)")
    args = ["energy", str(WATER), "--field", "0.0", "0.0", "0.005"]
    mine = run_python_cli(*args)
    theirs = _run([str(binary), *args])
    assert mine.returncode == theirs.returncode == 0, (mine.stderr, theirs.stderr)
    assert mine.stdout == theirs.stdout

    # And the field actually did something — otherwise the two could agree on ignoring it.
    bare = run_python_cli("energy", str(WATER))
    assert bare.stdout != mine.stdout


def test_molden_output_goes_to_a_file_when_asked(tmp_path) -> None:
    out = tmp_path / "wavefunction.molden"
    result = run_python_cli("molden", str(WATER), "--molden-output", str(out))
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    for section in ("[Molden Format]", "[Atoms] Angs", "[STO]", "[MO]"):
        assert section in text, f"missing {section}"


def test_the_unit_constants_are_the_crate_s_own() -> None:
    # The CLI converts to atomic units using `native.constants()`. If those ever became CODATA
    # values the output would drift from MOPAC's by a few hundredths of a kcal/mol with nothing
    # failing, so the deliberately non-CODATA choice is pinned here.
    from am1_rs import native

    c = native.constants()
    assert c["hartree_to_ev"] == pytest.approx(27.21, abs=1e-12)
    assert c["bohr_to_angstrom"] == pytest.approx(0.529167, abs=1e-12)
    assert c["ev_to_kcal"] == pytest.approx(23.061, abs=1e-12)
    assert c["ev_to_hartree"] == pytest.approx(1.0 / 27.21, rel=1e-15)
