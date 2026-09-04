# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line interface, installed as ``am1-rs`` by ``pip install``.

The Rust crate ships its own ``am1_rs_cli`` binary, which is the one to use when you build from
source with cargo. A wheel cannot carry that binary alongside the extension module, so this
mirrors it on top of the same native bindings: identical modes, identical flags, identical
output. Two front ends, one engine — nothing here re-implements any physics.

Run ``am1-rs`` with no arguments for usage, or ``python -m am1_rs``.
"""

from __future__ import annotations

import sys
from typing import Sequence

from . import native

USAGE = """am1-rs - AM1/RM1 semiempirical calculations

USAGE:
  am1-rs <mode> <file.xyz> [options]

MODES:
  energy      single point: heat of formation, charges, dipole, HOMO/LUMO
  gradient    energy + forces (Hartree/Bohr)
  optimize    L-BFGS geometry optimization
  frequencies harmonic vibrational frequencies (cm^-1)
  charges     AM1-BCC partial charges for AMBER (--mulliken for raw AM1)
  orbitals    orbital energies and occupations (both spins for UHF)
  ir          infrared spectrum: atomic polar tensor and km/mol intensities
  molden      wavefunction in Molden format (stdout, or --molden-output)

OPTIONS:
  --method M            NDDO parameterization: am1|rm1 (default am1)
  --charge Q            total molecular charge (default 0)
  --multiplicity M      spin multiplicity 2S+1 (default 1; M>1 requires UHF)
  --reference REF       SCF reference: auto|rhf|uhf (default auto)
  --rhf | --uhf         shortcuts for --reference rhf / uhf
  --field FX FY FZ      uniform electric field, atomic units (Hartree per e*Bohr)
  --opt-output FILE     write optimized geometry (XYZ)
  --molden-output FILE  write the Molden wavefunction to a file instead of stdout
  --mulliken            charges mode: use raw AM1 Mulliken charges
"""

# Keep every printed string ASCII. Not a style rule: USAGE and the result lines are written to a
# stream whose encoding comes from the user's locale, and a cp932 or C locale cannot encode an em
# dash or a middle dot at all. `_use_utf8_streams` stops that from raising, but a console on such
# a locale still renders UTF-8 as mojibake, so the fix that works everywhere is not to emit it.
# Units are spelled `e*a0`, `cm^-1`, `eV/A`. `src/bin/am1_rs.rs` must stay output-identical.

# Enough of the periodic table to read and write ordinary XYZ files.
_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
    "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm "
    "Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()
_NUMBER = {s.lower(): i for i, s in enumerate(_SYMBOLS)}


class CliError(Exception):
    """A problem with the command line or the input file, reported without a traceback."""


def read_xyz(path: str) -> tuple[list[int], list[list[float]]]:
    """Read a plain XYZ file into atomic numbers and Ångström coordinates."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc
    if len(lines) < 3:
        raise CliError(f"{path} is too short to be an XYZ file")
    try:
        count = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise CliError(f"{path}: the first line must be an atom count") from exc
    numbers: list[int] = []
    positions: list[list[float]] = []
    for lineno, line in enumerate(lines[2 : 2 + count], start=3):
        fields = line.split()
        if len(fields) < 4:
            raise CliError(f"{path}:{lineno}: expected 'symbol x y z'")
        symbol = fields[0]
        z = _NUMBER.get(symbol.lower())
        if z is None:
            # Some writers put the atomic number in the symbol column.
            try:
                z = int(symbol)
            except ValueError as exc:
                raise CliError(f"{path}:{lineno}: unknown element '{symbol}'") from exc
        numbers.append(z)
        positions.append([float(v) for v in fields[1:4]])
    if len(numbers) != count:
        raise CliError(f"{path}: header says {count} atoms, found {len(numbers)}")
    return numbers, positions


def _to_xyz(numbers: Sequence[int], positions, comment: str = "") -> str:
    """The Rust CLI's ``to_xyz``, field for field."""
    lines = [f"{len(numbers)}", comment]
    for z, p in zip(numbers, positions):
        lines.append(f"{_SYMBOLS[z]:<2} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}")
    return "\n".join(lines) + "\n"


def _unsigned_zero(value: float, decimals: int) -> float:
    """A value that rounds to zero at ``decimals`` places, without a sign.

    Mirrors ``unsigned_zero`` in ``src/bin/am1_rs.rs``. ``-0.0`` and ``0.0`` are the same number
    at any printed precision but different *text*, and ``tests/test_cli.py`` diffs this CLI's text
    against the Rust one's. The two take different routes to the same eigenvalue, so a quantity
    that is zero -- a rigid-body frequency, a symmetry-forbidden polar tensor element -- can land
    on either side of it and make the two disagree about a number they both agree is zero.
    """
    return 0.0 if abs(value) < 0.5 * 10.0**-decimals else value

def _rust_exponent(value: float, digits: int = 6) -> str:
    """Rust's ``{:.6e}``, which differs from Python's in the exponent.

    Python pads the exponent to two digits and keeps a sign (``4.311773e-03``); Rust writes it
    plainly (``4.311773e-3``). The numbers are the same, but the CLI output is meant to match the
    Rust one line for line, and a diff over these files is how that is checked.
    """
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _print_orbitals(energies, n_occ: int, spin: str) -> None:
    """One spin channel's orbital energies, with the frontier marked.

    Byte-identical to the Rust CLI's `print_orbitals`; `tests/test_cli.py` diffs the two.
    """
    tag = f" [{spin}]" if spin else ""
    for i, e in enumerate(energies):
        if i + 1 == n_occ:
            marker = "  <- HOMO"
        elif i == n_occ:
            marker = "  <- LUMO"
        else:
            marker = ""
        occ = 2.0 if i < n_occ else 0.0
        print(f"  {i + 1:>4}  {e:>14.8f}  occ {occ:.1f}{marker}{tag}")


def _print_charges(numbers: Sequence[int], charges) -> None:
    """The Rust CLI's ``print_charges``."""
    print("Mulliken charges (e):")
    for z, q in zip(numbers, charges):
        print(f"  {_SYMBOLS[z]:<2}  {q:+.5f}")


def _print_energy(numbers: Sequence[int], r: dict) -> None:
    """The Rust CLI's ``print_energy``, in the same atomic units and the same layout.

    The conversions come from ``native.constants()`` rather than being written down here. The
    crate uses MOPAC7's ``ev = 27.21`` and ``a0 = 0.529167``, not CODATA, deliberately — a second
    copy of those numbers on this side is a copy that can drift, and the symptom would be a heat
    of formation a few hundredths of a kcal/mol away from the Rust CLI's with nothing failing.
    """
    from . import native

    c = native.constants()
    ev_to_hartree = c["ev_to_hartree"]
    debye_to_au = 1.0 / c["au_dipole_to_debye"]
    hf_hartree = r["heat_of_formation_kcal"] * c["kcal_to_ev"] * ev_to_hartree

    tag = " (UHF)" if r.get("unrestricted") else ""
    # An `optimize` result uses `iterations` for the optimizer's steps, so the SCF count is
    # reported separately there; everywhere else `iterations` is the SCF's own.
    iterations = r.get("scf_iterations", r.get("iterations"))
    print(f"SCF converged in {iterations} iterations{tag}")
    print(f"total energy      : {r['energy_ev'] * ev_to_hartree:16.8f} Hartree")
    print(f"  electronic      : {r['electronic_ev'] * ev_to_hartree:16.8f} Hartree")
    print(f"  core repulsion  : {r['core_ev'] * ev_to_hartree:16.8f} Hartree")
    print(
        f"heat of formation : {hf_hartree:16.8f} Hartree   "
        f"({r['heat_of_formation_kcal']:.6f} kcal/mol)"
    )
    homo, lumo = r.get("homo_ev"), r.get("lumo_ev")
    if homo is not None and lumo is not None:
        h, l = homo * ev_to_hartree, lumo * ev_to_hartree
        print(f"HOMO / LUMO       : {h:.6f} / {l:.6f} Hartree  (gap {l - h:.6f})")
    dx, dy, dz = r["dipole_debye"]
    print(
        f"dipole            : {r['dipole_magnitude'] * debye_to_au:.6f} e*a0  "
        f"({dx * debye_to_au:.6f}, {dy * debye_to_au:.6f}, {dz * debye_to_au:.6f})"
    )
    _print_charges(numbers, r["charges"])


def _parse(argv: Sequence[str]) -> dict:
    if len(argv) < 2:
        raise CliError("")
    opts = {
        "mode": argv[0],
        "path": argv[1],
        "method": "am1",
        "charge": 0.0,
        "multiplicity": 1,
        "reference": "auto",
        "opt_output": None,
        "molden_output": None,
        "field": None,
        "mulliken": False,
    }
    i = 2
    while i < len(argv):
        flag = argv[i]

        def value(name: str) -> str:
            nonlocal i
            i += 1
            if i >= len(argv):
                raise CliError(f"{name} needs a value")
            return argv[i]

        if flag == "--method":
            opts["method"] = value("--method")
        elif flag == "--charge":
            opts["charge"] = float(value("--charge"))
        elif flag in ("--multiplicity", "--spin-multiplicity"):
            opts["multiplicity"] = int(float(value("--multiplicity")))
        elif flag in ("--reference", "--ref"):
            opts["reference"] = value("--reference")
        elif flag == "--rhf":
            opts["reference"] = "rhf"
        elif flag == "--uhf":
            opts["reference"] = "uhf"
        elif flag == "--field":
            # Three values, atomic units — the same convention as `am1_rs.native`.
            opts["field"] = [float(value("--field")) for _ in range(3)]
        elif flag == "--opt-output":
            opts["opt_output"] = value("--opt-output")
        elif flag == "--molden-output":
            opts["molden_output"] = value("--molden-output")
        elif flag == "--mulliken":
            opts["mulliken"] = True
        elif flag in ("-h", "--help"):
            raise CliError("")
        else:
            raise CliError(f"unknown option '{flag}'")
        i += 1
    return opts


def _use_utf8_streams() -> None:
    """Make stdout and stderr UTF-8, whatever the locale says.

    Python encodes ``print`` output with the locale encoding, which is ``cp932`` on a Japanese
    Windows and plain ASCII under the ``C``/``POSIX`` locale that minimal Docker images ship
    with. Any character outside that set then raises ``UnicodeEncodeError`` *mid-run*, after
    part of the output has already been written. The Rust front end has no such failure mode —
    it writes UTF-8 bytes unconditionally — so without this the two CLIs would not agree on
    every platform, which is the one property they are supposed to have.

    The routine output is deliberately ASCII (see the note on :data:`USAGE`), so this is the
    second line of defence: an error message raised by the native layer can contain anything.
    ``errors="backslashreplace"`` means even a stream that cannot be reconfigured at all still
    degrades to an escape rather than an exception.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):  # already detached, or not a real text stream
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _use_utf8_streams()
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        opts = _parse(argv)
    except CliError as exc:
        if str(exc):
            print(f"am1-rs: {exc}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 1

    try:
        numbers, positions = read_xyz(opts["path"])
        common = dict(
            charge=opts["charge"],
            multiplicity=opts["multiplicity"],
            reference=opts["reference"],
            method=opts["method"],
            electric_field=opts["field"],
        )
        mode = opts["mode"]

        if mode == "energy":
            r = native.single_point(numbers, positions, **common)
            _print_energy(numbers, r)

        elif mode == "gradient":
            r = native.gradient(numbers, positions, **common)
            _print_energy(numbers, r)
            # Forces, not the gradient: the force is minus the energy derivative, and the Rust
            # CLI prints forces under this heading.
            print("\nforces (Hartree/Bohr):")
            for z, g in zip(numbers, r["gradient_hartree_per_bohr"]):
                print(f"  {_SYMBOLS[z]:<2}  {-g[0]:14.8f} {-g[1]:14.8f} {-g[2]:14.8f}")
            grad = _rust_exponent(r["max_gradient_hartree_per_bohr"])
            print(f"max |grad| = {grad} Hartree/Bohr")

        elif mode == "optimize":
            r = native.optimize(numbers, positions, **common)
            state = "converged" if r["converged"] else "did NOT converge"
            print(f"optimization {state} in {r['iterations']} steps")
            _print_energy(numbers, r)
            xyz = _to_xyz(numbers, r["positions_angstrom"], "am1-rs optimized")
            if opts["opt_output"]:
                with open(opts["opt_output"], "w", encoding="utf-8") as handle:
                    handle.write(xyz)
                print(f"\noptimized geometry written to {opts['opt_output']}")
            else:
                print(f"\noptimized geometry (Angstrom):\n{xyz}")

        elif mode == "frequencies":
            r = native.frequencies(numbers, positions, **common)
            print("harmonic vibrational frequencies (cm^-1):")
            for i, nu in enumerate(r["frequencies_cm"], start=1):
                flag = "  (translation/rotation)" if abs(nu) < 50.0 else ""
                print(f"  {i:>3}  {_unsigned_zero(nu, 1):>10.1f}{flag}")
            print("\n(compute at an optimized geometry for meaningful frequencies)")

        elif mode == "orbitals":
            r = native.orbitals(numbers, positions, **common)
            print(f"orbital energies (Hartree), {r['n_occupied']} occupied:")
            _print_orbitals(
                r["energies_hartree"],
                r["n_occupied"],
                "alpha" if r["unrestricted"] else "",
            )
            if r["unrestricted"]:
                print(f"\nbeta channel, {r['beta_n_occupied']} occupied:")
                _print_orbitals(r["beta_energies_hartree"], r["beta_n_occupied"], "beta")

        elif mode == "ir":
            r = native.ir_spectrum(numbers, positions, **common)
            print("atomic polar tensor d(mu_a)/d(R_b) (e), rows x/y/z, columns 3*atom+axis:")
            for row in r["dipole_derivatives"]:
                print("  " + " ".join(f"{_unsigned_zero(v, 5):9.5f}" for v in row))
            print("\ninfrared spectrum:")
            print("  mode   freq (cm^-1)   intensity (km/mol)   rigid-body")
            for k, nu in enumerate(r["frequencies_cm"], start=1):
                print(
                    f"  {k:>4}  {_unsigned_zero(nu, 2):>13.2f}"
                    f"  {r['intensities_km_per_mol'][k - 1]:>19.4f}"
                    f"  {r['translation_rotation_overlap'][k - 1]:>10.3f}"
                )
            print("\n(rigid-body near 1 is a translation or rotation, not a vibration)")

        elif mode == "molden":
            text = native.molden(numbers, positions, **common)
            if opts["molden_output"]:
                with open(opts["molden_output"], "w", encoding="utf-8") as handle:
                    handle.write(text)
                print(f"molden wavefunction written to {opts['molden_output']}")
            else:
                print(text, end="")

        elif mode == "charges":
            if opts["mulliken"]:
                r = native.single_point(numbers, positions, **common)
                _print_charges(numbers, r["charges"])
            else:
                r = native.am1_bcc(
                    numbers,
                    positions,
                    charge=opts["charge"],
                    multiplicity=opts["multiplicity"],
                )
                for warning in r.get("warnings", []):
                    print(f"warning: {warning}", file=sys.stderr)
                print("AM1-BCC charges (e):")
                for z, q, t in zip(numbers, r["charges"], r["atom_types"]):
                    print(f"  {_SYMBOLS[z]:<2}  {q:+.5f}   [type {t}]")
                print(f"sum = {sum(r['charges']):+.5f} e")

        else:
            raise CliError(f"unknown mode '{mode}'")

    except CliError as exc:
        print(f"am1-rs: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # native errors carry their own message
        print(f"am1-rs: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
