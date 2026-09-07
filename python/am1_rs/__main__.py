# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line interface, installed as ``am1-rs`` by ``pip install``.

The Rust crate ships its own ``am1_rs_cli`` binary, which is the one to use when you build from
source with cargo. A wheel cannot carry that binary alongside the extension module, so this
mirrors it on top of the same native bindings: identical modes, identical flags, identical
output. Two front ends, one engine — nothing here re-implements any physics.

``tests/test_cli_matrix.py`` runs **every** mode against **every** combination of the options
that apply to it and diffs the two front ends byte for byte. That test exists because the
alternative — keeping two hand-written front ends in step by reading them — is what left the
pip-installed CLI broken in three of five modes before 0.2.2.

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
  gradient    energy + forces (Hartree/Bohr), and stress under a cell
  optimize    L-BFGS geometry optimization (periodic when a cell is given)
  frequencies harmonic vibrational frequencies (cm^-1); phonons at q=0 under a cell
  phonons     phonon frequencies from a supercell (needs a cell)
  charges     AM1-BCC partial charges for AMBER (--mulliken for raw AM1)
  orbitals    orbital energies, occupations and optionally MO coefficients
  ir          infrared spectrum: atomic polar tensor and km/mol intensities
  molden      wavefunction in Molden format (stdout, or --molden-output)

OPTIONS:
  --method M            NDDO parameterization: am1|rm1 (default am1)
  --charge Q            total molecular charge (default 0)
  --multiplicity M      spin multiplicity 2S+1 (default 1; M>1 requires UHF)
  --reference REF       SCF reference: auto|rhf|uhf (default auto)
  --rhf | --uhf         shortcuts for --reference rhf / uhf (force restricted/unrestricted)
  --field FX FY FZ      uniform electric field, atomic units (Hartree per e*Bohr)
  --opt-output FILE     write optimized geometry (extended XYZ, with the cell)
  --mol2-output FILE    write AM1-BCC charges as a mol2 file
  --molden-output FILE  write the Molden wavefunction to a file instead of stdout
  --molden-basis B      Molden basis section: gto|sto (default gto)
  --molden-primitives N Gaussians per shell for gto (default 6, max 10)
  --molden-orthogonal   write raw NDDO coefficients instead of S^-1/2 C
  --orbital-coefficients  orbitals mode: also print the MO coefficient matrix
  --mulliken            charges mode: use raw AM1 Mulliken charges

PERIODIC OPTIONS (a cell also comes from a Lattice="..." XYZ comment line):
  --cell ...            1, 3, 6 or 9 numbers: a | a b c | a b c alpha beta gamma |
                        ax ay az bx by bz cx cy cz  (Angstrom, degrees)
  --pbc AXES            periodic axes: x, y, z, xy, xyz, none (combinations allowed)
  --pbc-x --pbc-y --pbc-z   make one axis periodic; they accumulate
  --no-pbc              drop the cell and run the contents of it as a molecule
  --kpts NX NY NZ       Monkhorst-Pack k-mesh for the SCF (default 1 1 1)
  --smearing EV         Fermi-Dirac electronic temperature (default 0)
  --max-scf N           SCF iteration limit
  --supercell NX NY NZ  phonons mode: supercell for the force constants (default 2 2 2)
  --qpoints QX QY QZ .. phonons mode: explicit q points, fractional, three numbers each
  --qpath QX QY QZ ..   phonons mode: corners of a straight-line q path, fractional
  --qpath-points N      points per segment of --qpath (default 20)
  --relax-cell          optimize mode: relax the lattice against the stress too
  --pressure P          target pressure with --relax-cell, Hartree/Bohr^d (default 0)
DIVIDE-AND-CONQUER (energy, gradient and optimize modes):
  --dc                  linear-scaling divide-and-conquer SCF instead of the full one
  --dc-core N           target atoms per core region (default 12; implies --dc)
  --dc-buffer R         buffer radius in Angstrom (default 5.82; implies --dc)
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

_MAX_NGAUSS = 10


class CliError(Exception):
    """A problem with the command line or the input file, reported without a traceback."""


def _extxyz_field(comment: str, key: str) -> str | None:
    """``key="..."`` or ``key=token`` in an extended-XYZ comment line, case-insensitively.

    Mirrors ``extxyz_field`` in ``src/system.rs``. Without it the Python front end would ignore
    a lattice the Rust one reads, so the same file would be a crystal to one CLI and a molecule
    to the other — which is exactly the kind of divergence the two front ends exist to not have.
    """
    lower = comment.lower()
    needle = key.lower() + "="
    start = 0
    while True:
        at = lower.find(needle, start)
        if at < 0:
            return None
        # Must start a token, so `Lattice=` does not match inside `SuperLattice=`.
        if at == 0 or comment[at - 1].isspace():
            rest = comment[at + len(needle) :]
            if rest.startswith('"'):
                end = rest.find('"', 1)
                return rest[1:end] if end > 0 else None
            token = rest.split()
            return token[0] if token else None
        start = at + 1


def read_xyz(path: str) -> tuple[list[int], list[list[float]], list[list[float]] | None, list[bool]]:
    """Read an XYZ file into atomic numbers, Ångström coordinates, and an optional cell.

    The comment line is read as extended XYZ: ``Lattice="ax ay az bx by bz cx cy cz"`` (Ångström)
    gives the cell and ``pbc="T T F"`` which axes are periodic, defaulting to all three when a
    lattice is present. A comment line without ``Lattice=`` is ignored exactly as before.
    """
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

    comment = lines[1] if len(lines) > 1 else ""
    cell: list[list[float]] | None = None
    pbc = [False, False, False]
    lattice_text = _extxyz_field(comment, "Lattice")
    if lattice_text is not None:
        values = [float(v) for v in lattice_text.replace(",", " ").split()]
        if len(values) != 9:
            raise CliError(f"{path}: Lattice= needs nine numbers, got {len(values)}")
        cell = [values[0:3], values[3:6], values[6:9]]
        pbc = [True, True, True]
        pbc_text = _extxyz_field(comment, "pbc")
        if pbc_text is not None:
            flags = pbc_text.replace(",", " ").split()
            if len(flags) != 3:
                raise CliError(f"{path}: pbc= needs three entries, got {len(flags)}")
            pbc = [f.strip().upper() in ("T", "TRUE", "1") for f in flags]

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
    return numbers, positions, cell, pbc


def _to_xyz(numbers: Sequence[int], positions, comment: str = "", cell=None, pbc=None) -> str:
    """The Rust CLI's ``to_xyz``, field for field, extended XYZ when there is a cell."""
    if cell is not None:
        flat = " ".join(f"{v:.8f}" for row in cell for v in row)
        flags = " ".join("T" if p else "F" for p in (pbc or [True, True, True]))
        comment = f'{comment} Lattice="{flat}" pbc="{flags}"'
    lines = [f"{len(numbers)}", comment]
    for z, p in zip(numbers, positions):
        lines.append(f"{_SYMBOLS[z]:<2} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}")
    return "\n".join(lines) + "\n"


def _unsigned_zero(value: float, decimals: int) -> float:
    """A value that rounds to zero at ``decimals`` places, without a sign.

    Mirrors ``unsigned_zero`` in ``src/bin/am1_rs.rs``. ``-0.0`` and ``0.0`` are the same number
    at any printed precision but different *text*, and ``tests/test_cli.py`` diffs this CLI's text
    against the Rust one's. The two take different routes to the same eigenvalue, so a quantity
    that is zero -- a residual rigid-body frequency, a symmetry-forbidden polar tensor element --
    can land on either side of it and make the two disagree about a number they both agree is zero.
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


def _print_pbc_energy(numbers: Sequence[int], r: dict) -> None:
    """The Rust CLI's ``print_pbc_energy``."""
    ev_to_hartree = native.constants()["ev_to_hartree"]
    warning = r.get("charged_cell_warning")
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    tag = ", UHF" if r.get("unrestricted") else ""
    # An `optimize` result uses `iterations` for the optimizer's steps, so the SCF count is
    # reported separately there; everywhere else `iterations` is the SCF's own. Same rule as
    # `_print_energy`, and getting it wrong made the periodic optimize report "3 iterations"
    # where the Rust front end reported the SCF's 16.
    iterations = r.get("scf_iterations", r.get("iterations"))
    print(f"SCF converged in {iterations} iterations ({r['k_points']} k-points{tag})")
    print(f"total energy      : {r['energy_ev'] * ev_to_hartree:16.8f} Hartree")
    print(f"  electronic      : {r['electronic_ev'] * ev_to_hartree:16.8f} Hartree")
    print(f"  core repulsion  : {r['core_ev'] * ev_to_hartree:16.8f} Hartree")
    print(f"Fermi energy      : {r['fermi_energy_ev'] * ev_to_hartree:16.8f} Hartree")
    print(f"max image overlap : {r['max_image_overlap']:16.8f}   (NDDO assumes 0)")
    _print_charges(numbers, r["charges"])


def _print_dc_energy(numbers: Sequence[int], r: dict, mode: str) -> None:
    """Byte-for-byte the Rust `print_dc_energy`.

    The subsystem count and the largest subsystem's basis go first because they are what say
    whether the partition did anything: one subsystem holding every atom is a full SCF with extra
    steps, and a user who passed ``--dc`` should be able to see that rather than infer it from
    the timing.
    """
    c = native.constants()
    ev_to_hartree = c["ev_to_hartree"]
    kcal_to_ev = c["kcal_to_ev"]
    hf = _field(r, "heat_of_formation_kcal", mode)
    state = "converged" if _field(r, "converged", mode) else "did NOT converge"
    tag = " (UHF)" if r.get("unrestricted") else ""
    print(
        f"divide-and-conquer SCF {state} in "
        f"{_field(r, 'iterations', mode)} iterations{tag}"
    )
    print(
        f"subsystems        : {_field(r, 'subsystems', mode)} "
        f"(largest {_field(r, 'largest_subsystem_aos', mode)} AOs)"
    )
    print(f"total energy      : {_field(r, 'energy_ev', mode) * ev_to_hartree:16.8f} Hartree")
    print(f"  electronic      : {_field(r, 'electronic_ev', mode) * ev_to_hartree:16.8f} Hartree")
    print(f"  core repulsion  : {_field(r, 'core_ev', mode) * ev_to_hartree:16.8f} Hartree")
    print(
        f"heat of formation : {hf * kcal_to_ev * ev_to_hartree:16.8f} Hartree   "
        f"({hf:.6f} kcal/mol)"
    )
    print(
        f"Fermi energy      : {_field(r, 'fermi_energy_ev', mode) * ev_to_hartree:16.8f} Hartree"
    )
    print(
        f"HOMO-LUMO gap     : {_field(r, 'homo_lumo_gap_ev', mode) * ev_to_hartree:16.8f} Hartree"
    )
    _print_charges(numbers, _field(r, "charges", mode))
    warning = r.get("small_gap_warning")
    if warning:
        print(f"\nwarning: {warning}")


def _print_forces(numbers: Sequence[int], forces_hartree_per_bohr, max_grad_hartree) -> None:
    print("\nforces (Hartree/Bohr):")
    for z, f in zip(numbers, forces_hartree_per_bohr):
        print(f"  {_SYMBOLS[z]:<2}  {f[0]:14.8f} {f[1]:14.8f} {f[2]:14.8f}")
    print(f"max |grad| = {_rust_exponent(max_grad_hartree)} Hartree/Bohr")


def _print_stress(voigt_hartree, pressure_hartree) -> None:
    row = " ".join(f"{s:14.8f}" for s in voigt_hartree)
    print(f"stress (Hartree/Bohr^d, Voigt xx yy zz yz xz xy):\n  {row}")
    print(f"pressure          : {pressure_hartree:16.8f} Hartree/Bohr^d")


def _print_cell(cell, pbc) -> None:
    print(f"cell (Angstrom), periodic {_pbc_text(pbc)}:")
    for v in cell:
        print(f"  {v[0]:14.8f} {v[1]:14.8f} {v[2]:14.8f}")


def _pbc_text(pbc) -> str:
    text = "".join(name for name, on in zip("xyz", pbc) if on)
    return text or "none"


def _print_asr(before: float, after: float) -> None:
    print(f"\nacoustic sum rule violation: {_rust_exponent(before, 3)} -> "
          f"{_rust_exponent(after, 3)} eV/Bohr^2 (imposed)")


def _parse_pbc(text: str) -> list[bool]:
    cleaned = text.strip().lower()
    if cleaned in ("none", "false", "f"):
        return [False, False, False]
    if cleaned in ("all", "true", "t"):
        return [True, True, True]
    out = [False, False, False]
    for ch in cleaned:
        if ch == "x":
            out[0] = True
        elif ch == "y":
            out[1] = True
        elif ch == "z":
            out[2] = True
        elif ch in (",", " ", "+"):
            pass
        else:
            raise CliError(
                f"invalid --pbc value: '{ch}' in '{text}' (expected x, y, z, none, or a "
                "combination such as xy)"
            )
    return out


def _cell_from(values: list[float]) -> list[list[float]]:
    """``--cell`` with 1, 3, 6 or 9 numbers; the Rust CLI's ``cell_from``, in Angstrom."""
    import math

    if len(values) == 1:
        a = values[0]
        return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    if len(values) == 3:
        return [[values[0], 0.0, 0.0], [0.0, values[1], 0.0], [0.0, 0.0, values[2]]]
    if len(values) == 6:
        a, b, c, alpha, beta, gamma = values
        rad = math.pi / 180.0
        alpha, beta, gamma = alpha * rad, beta * rad, gamma * rad
        if abs(math.sin(gamma)) < 1.0e-12:
            raise CliError("cell angle gamma cannot be 0 or 180 degrees")
        cx = c * math.cos(beta)
        cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
        cz2 = c * c - cx * cx - cy * cy
        if cz2 <= 0.0:
            raise CliError(
                f"cell angles alpha={values[3]}, beta={values[4]}, gamma={values[5]} do not "
                "describe a realizable triclinic lattice"
            )
        return [
            [a, 0.0, 0.0],
            [b * math.cos(gamma), b * math.sin(gamma), 0.0],
            [cx, cy, math.sqrt(cz2)],
        ]
    if len(values) == 9:
        return [values[0:3], values[3:6], values[6:9]]
    raise CliError(
        "--cell takes 1, 3, 6 or 9 numbers (cube; a b c; a b c alpha beta gamma; or three "
        f"lattice vectors), got {len(values)}"
    )


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
        "mol2_output": None,
        "molden_output": None,
        "molden_basis": "gto",
        "molden_primitives": 6,
        "molden_deorthogonalize": True,
        "orbital_coefficients": False,
        "field": None,
        "mulliken": False,
        "cell": None,
        "pbc": None,
        "kpts": [1, 1, 1],
        "supercell": [2, 2, 2],
        "supercell_explicit": False,
        "qpoints": None,
        "qpath": None,
        "qpath_points": 20,
        "smearing": 0.0,
        "max_scf": None,
        "relax_cell": False,
        "pressure": 0.0,
        "dc": False,
        "dc_core": 12,
        # None keeps the library default (12 Bohr). Given in Angstrom, like every other length
        # on this command line, and converted at the boundary -- see the Rust `dc_options_from`.
        "dc_buffer": None,
    }
    i = 2

    def value(name: str) -> str:
        nonlocal i
        i += 1
        if i >= len(argv):
            raise CliError(f"{name} needs a value")
        return argv[i]

    def merge_pbc(add: list[bool]) -> list[bool]:
        base = opts["pbc"] or [False, False, False]
        return [base[k] or add[k] for k in range(3)]

    while i < len(argv):
        flag = argv[i]
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
        elif flag == "--mol2-output":
            opts["mol2_output"] = value("--mol2-output")
        elif flag == "--molden-output":
            opts["molden_output"] = value("--molden-output")
        elif flag == "--molden-basis":
            basis = value("--molden-basis").strip().lower()
            if basis in ("gto", "gaussian"):
                opts["molden_basis"] = "gto"
            elif basis in ("sto", "slater"):
                opts["molden_basis"] = "sto"
            else:
                raise CliError(f"invalid --molden-basis value: {basis} (expected gto|sto)")
        elif flag == "--molden-primitives":
            n = int(float(value("--molden-primitives")))
            if not 1 <= n <= _MAX_NGAUSS:
                raise CliError(f"--molden-primitives must be between 1 and {_MAX_NGAUSS}")
            opts["molden_primitives"] = n
        elif flag == "--molden-orthogonal":
            opts["molden_deorthogonalize"] = False
        elif flag == "--orbital-coefficients":
            opts["orbital_coefficients"] = True
        elif flag == "--mulliken":
            opts["mulliken"] = True
        elif flag == "--cell":
            values: list[float] = []
            while i + 1 < len(argv):
                try:
                    values.append(float(argv[i + 1]))
                except ValueError:
                    break
                i += 1
            opts["cell"] = _cell_from(values)
        elif flag == "--pbc":
            opts["pbc"] = merge_pbc(_parse_pbc(value("--pbc")))
        elif flag == "--pbc-x":
            opts["pbc"] = merge_pbc([True, False, False])
        elif flag == "--pbc-y":
            opts["pbc"] = merge_pbc([False, True, False])
        elif flag == "--pbc-z":
            opts["pbc"] = merge_pbc([False, False, True])
        elif flag == "--no-pbc":
            opts["pbc"] = [False, False, False]
        elif flag == "--kpts":
            opts["kpts"] = [int(float(value("--kpts"))) for _ in range(3)]
        elif flag == "--supercell":
            opts["supercell"] = [int(float(value("--supercell"))) for _ in range(3)]
            opts["supercell_explicit"] = True
        elif flag in ("--qpoints", "--qpath"):
            # Variadic like `--cell`: consume numbers until something that is not one, then
            # group into (x, y, z) triples. A count that is not a multiple of three is a typo.
            numbers: list[float] = []
            while i + 1 < len(argv):
                try:
                    numbers.append(float(argv[i + 1]))
                except ValueError:
                    break
                i += 1
            if not numbers or len(numbers) % 3 != 0:
                raise CliError(
                    f"{flag} takes a multiple of three numbers (x y z per point, fractional), "
                    f"got {len(numbers)}"
                )
            triples = [numbers[k : k + 3] for k in range(0, len(numbers), 3)]
            opts["qpoints" if flag == "--qpoints" else "qpath"] = triples
        elif flag == "--qpath-points":
            opts["qpath_points"] = int(float(value("--qpath-points")))
        elif flag == "--smearing":
            opts["smearing"] = float(value("--smearing"))
        elif flag == "--max-scf":
            opts["max_scf"] = int(float(value("--max-scf")))
        elif flag == "--dc":
            opts["dc"] = True
        elif flag == "--dc-core":
            opts["dc"] = True
            opts["dc_core"] = int(float(value("--dc-core")))
        elif flag == "--dc-buffer":
            opts["dc"] = True
            opts["dc_buffer"] = float(value("--dc-buffer"))
        elif flag == "--relax-cell":
            opts["relax_cell"] = True
        elif flag == "--pressure":
            opts["pressure"] = float(value("--pressure"))
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


def _field(result: dict, key: str, mode: str):
    """A result field, or a `CliError` naming what is missing.

    A native entry point that stops returning a key the CLI prints is a packaging or a version
    mismatch, not a bug in the calculation — and a bare ``KeyError`` traceback says neither.
    Every read of a native result goes through here so that failure mode is a one-line message
    naming the key and the mode, with a non-zero exit status, instead of a stack trace after the
    numbers have already been computed. ``tests/test_result_keys.py`` asserts that every key
    each mode reads is actually produced, so this should never fire.
    """
    try:
        return result[key]
    except KeyError:
        raise CliError(
            f"the native module returned no '{key}' for mode '{mode}'; the extension module and "
            "the Python package are probably from different versions of am1-rs "
            f"(am1_rs {getattr(native, '__version__', '')} at {native.__file__})"
        ) from None


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
        numbers, positions, cell, pbc = read_xyz(opts["path"])
        mode = opts["mode"]

        # A cell on the command line replaces one from the file; --pbc replaces the axes of
        # whichever cell is in play. Neither silently invents the other.
        if opts["cell"] is not None:
            cell = opts["cell"]
            pbc = [True, True, True]
        if opts["pbc"] is not None:
            if cell is None and any(opts["pbc"]):
                raise CliError(
                    "--pbc asks for a periodic direction but there is no cell; give one with "
                    '--cell or a Lattice="..." comment line in the XYZ file'
                )
            pbc = opts["pbc"]
        if cell is not None and not any(pbc):
            cell = None
        periodic = cell is not None

        c = native.constants()
        ev_to_hartree = c["ev_to_hartree"]
        bohr = c["bohr_to_angstrom"]

        common = dict(
            charge=opts["charge"],
            multiplicity=opts["multiplicity"],
            reference=opts["reference"],
            method=opts["method"],
            electric_field=opts["field"],
        )
        pbc_common = dict(
            cell=cell,
            pbc=pbc,
            kpts=tuple(opts["kpts"]),
            charge=opts["charge"],
            multiplicity=opts["multiplicity"],
            method=opts["method"],
            smearing_ev=opts["smearing"],
            # `--field` and `--uhf` reach the periodic path too. They did not until 0.2.3: this
            # dict simply omitted them, so a periodic --field was accepted and ignored here
            # while the Rust front end applied it (and, for a field along a periodic axis,
            # correctly refused the run). Silently dropping a flag is the worst of the three
            # possible behaviours, and the CLI matrix found it by running `--dc` — which does not
            # drop the field — against the same combinations.
            electric_field=opts["field"],
            unrestricted=opts["reference"] == "uhf",
        )
        if opts["max_scf"] is not None:
            pbc_common["max_scf"] = opts["max_scf"]

        dc_common = dict(
            charge=opts["charge"],
            multiplicity=opts["multiplicity"],
            reference=opts["reference"],
            method=opts["method"],
            electric_field=opts["field"],
            core_size=opts["dc_core"],
        )
        # Only when there is one. The binding refuses a lattice without periodicity flags and
        # the reverse, and a molecule has neither.
        if periodic:
            dc_common["cell"] = cell
            dc_common["pbc"] = pbc
        if opts["dc_buffer"] is not None:
            # Angstrom on the command line, Bohr in the binding -- the same conversion the Rust
            # `dc_options_from` does, and the reason both are written down.
            dc_common["buffer_radius"] = opts["dc_buffer"] * c["angstrom_to_bohr"]
        if opts["max_scf"] is not None:
            dc_common["max_scf"] = opts["max_scf"]
        # Refused, not ignored: `--dc` on `frequencies` would return full-SCF frequencies and the
        # only sign the flag did nothing would be the run being no faster, which is the
        # observation `--dc` exists to avoid having to make. Same message as the Rust front end.
        if opts["dc"] and mode not in ("energy", "gradient", "optimize"):
            raise CliError(
                f"--dc has no effect in `{mode}` mode; it applies to energy, gradient and "
                "optimize. (There is no divide-and-conquer second derivative: the Hessian needs "
                "the coupled response of the whole system, which is what the partition does not "
                "have.)"
            )

        def reject_cell(name: str) -> None:
            if periodic:
                raise CliError(
                    f"`{name}` is molecular only and the structure has a periodic cell; remove "
                    "the cell (--no-pbc) to run it on the contents of one cell"
                )

        # eV/A (the periodic surface's unit) -> Hartree/Bohr (this CLI's).
        def to_hartree_per_bohr(v):
            return v * bohr * ev_to_hartree

        if mode == "energy":
            if opts["dc"]:
                r = native.divide_conquer(numbers, positions, forces=False, **dc_common)
                _print_dc_energy(numbers, r, mode)
            elif periodic:
                r = native.pbc_point(numbers, positions, **pbc_common)
                _print_pbc_energy(numbers, r)
            else:
                r = native.single_point(numbers, positions, **common)
                _print_energy(numbers, r)

        elif mode == "gradient":
            if opts["dc"]:
                r = native.divide_conquer(numbers, positions, forces=True, **dc_common)
                _print_dc_energy(numbers, r, mode)
                # `forces_ev_per_angstrom` is already minus the gradient; the Rust front end
                # negates its gradient to print the same numbers.
                forces = [
                    [to_hartree_per_bohr(v) for v in f]
                    for f in _field(r, "forces_ev_per_angstrom", mode)
                ]
                max_grad = max(abs(v) for f in forces for v in f)
                _print_forces(numbers, forces, max_grad)
            elif periodic:
                r = native.pbc_point(numbers, positions, **pbc_common)
                _print_pbc_energy(numbers, r)
                forces = [
                    [to_hartree_per_bohr(v) for v in f]
                    for f in _field(r, "forces_ev_per_angstrom", mode)
                ]
                # The crate's own max, not one recomputed here: the eV/Å round trip moves the
                # last bit, which shows in a `{:.6e}` field and nowhere else.
                max_grad = _field(r, "max_gradient_ev_per_bohr", mode) * ev_to_hartree
                _print_forces(numbers, forces, max_grad)
                n_periodic = sum(1 for p in pbc if p)
                # The crate's own eV/Bohr^d, not the ASE-facing eV/A^d: converting here would
                # mean redoing a (Bohr per A)^d with d = the periodic count, in a second
                # language. That is exactly how the two front ends came to disagree.
                voigt = [s * ev_to_hartree for s in _field(r, "stress_voigt_ev_per_bohr", mode)]
                pressure = -sum(voigt[:3]) / n_periodic if n_periodic else 0.0
                _print_stress(voigt, pressure)
            else:
                r = native.gradient(numbers, positions, **common)
                _print_energy(numbers, r)
                # Forces, not the gradient: the force is minus the energy derivative.
                forces = [[-v for v in g] for g in _field(r, "gradient_hartree_per_bohr", mode)]
                _print_forces(
                    numbers, forces, _field(r, "max_gradient_hartree_per_bohr", mode)
                )

        elif mode == "optimize":
            if opts["dc"]:
                r = native.divide_conquer(
                    numbers, positions, forces=True, optimize=True, **dc_common
                )
                state = "converged" if _field(r, "opt_converged", mode) else "did NOT converge"
                print(f"optimization {state} in {_field(r, 'opt_iterations', mode)} steps")
                _print_dc_energy(numbers, r, mode)
                max_force = _field(r, "max_force_ev_per_bohr", mode) * ev_to_hartree
                print(f"max |force| = {_rust_exponent(max_force)} Hartree/Bohr")
                xyz = _to_xyz(
                    numbers,
                    _field(r, "positions_angstrom", mode),
                    "am1-rs optimized (divide-and-conquer)",
                )
            elif periodic:
                r = native.pbc_optimize(
                    numbers,
                    positions,
                    relax_cell=opts["relax_cell"],
                    pressure=opts["pressure"],
                    **pbc_common,
                )
                state = "converged" if _field(r, "converged", mode) else "did NOT converge"
                print(f"optimization {state} in {_field(r, 'iterations', mode)} steps")
                _print_pbc_energy(numbers, r)
                max_force = _field(r, "max_force_ev_per_bohr", mode) * ev_to_hartree
                print(f"max |force| = {_rust_exponent(max_force)} Hartree/Bohr")
                n_periodic = sum(1 for p in pbc if p)
                # The crate's own eV/Bohr^d, not the ASE-facing eV/A^d: converting here would
                # mean redoing a (Bohr per A)^d with d = the periodic count, in a second
                # language. That is exactly how the two front ends came to disagree.
                voigt = [s * ev_to_hartree for s in _field(r, "stress_voigt_ev_per_bohr", mode)]
                pressure = -sum(voigt[:3]) / n_periodic if n_periodic else 0.0
                _print_stress(voigt, pressure)
                out_cell = _field(r, "cell_angstrom", mode)
                out_pbc = _field(r, "pbc", mode)
                _print_cell(out_cell, out_pbc)
                xyz = _to_xyz(
                    numbers,
                    _field(r, "positions_angstrom", mode),
                    "am1-rs optimized",
                    out_cell,
                    out_pbc,
                )
            else:
                r = native.optimize(numbers, positions, **common)
                state = "converged" if _field(r, "converged", mode) else "did NOT converge"
                print(f"optimization {state} in {_field(r, 'iterations', mode)} steps")
                _print_energy(numbers, r)
                xyz = _to_xyz(
                    numbers, _field(r, "positions_angstrom", mode), "am1-rs optimized"
                )
            if opts["opt_output"]:
                with open(opts["opt_output"], "w", encoding="utf-8") as handle:
                    handle.write(xyz)
                print(f"\noptimized geometry written to {opts['opt_output']}")
            else:
                print(f"\noptimized geometry (Angstrom):\n{xyz}")

        elif mode == "frequencies":
            if periodic:
                r = native.phonons(
                    numbers,
                    positions,
                    cell,
                    pbc,
                    supercell=(1, 1, 1),
                    q_points=[[0.0, 0.0, 0.0]],
                    charge=opts["charge"],
                    multiplicity=opts["multiplicity"],
                    method=opts["method"],
                )
                freqs = _field(r, "frequencies_cm", mode)[0]
                print(f"phonon frequencies at q = 0 (cm^-1), {len(freqs)} modes:")
                for i, nu in enumerate(freqs, start=1):
                    print(f"  {i:>3}  {_unsigned_zero(nu, 1):>10.1f}")
                _print_asr(
                    _field(r, "acoustic_sum_rule_error_before", mode),
                    _field(r, "acoustic_sum_rule_error", mode),
                )
            else:
                r = native.frequencies(numbers, positions, **common)
                freqs = _field(r, "frequencies_cm", mode)
                print(
                    f"harmonic vibrational frequencies (cm^-1), {len(freqs)} modes "
                    f"({_field(r, 'rigid_body_count', mode)} rigid-body removed):"
                )
                for i, nu in enumerate(freqs, start=1):
                    print(f"  {i:>3}  {_unsigned_zero(nu, 1):>10.1f}")
                residual = max(
                    (abs(f) for f in _field(r, "rigid_body_frequencies_cm", mode)), default=0.0
                )
                print(
                    f"\nrigid-body residual: {_unsigned_zero(residual, 1):>10.1f} cm^-1 "
                    "(0 at a stationary point)"
                )
                print("(compute at an optimized geometry for meaningful frequencies)")

        elif mode == "phonons":
            if not periodic:
                raise CliError(
                    'phonons need a periodic cell; give one with --cell or a Lattice="..." '
                    "comment line, or use `frequencies` for a molecule"
                )
            # A repeat along a non-periodic direction is meaningless, and the default `2 2 2`
            # would otherwise refuse every slab and every chain — the systems this crate handles
            # best. Clamping is what makes the default usable; an explicit repeat on a
            # non-periodic axis is still an error, because that is a mistake worth naming.
            # Mirrors `src/bin/am1_rs.rs`.
            requested = [max(1, n) for n in opts["supercell"]]
            for axis, periodic_axis in enumerate(pbc):
                if not periodic_axis:
                    if opts["supercell_explicit"] and requested[axis] > 1:
                        raise CliError(
                            f"--supercell asks for {requested[axis]} repeats along axis {axis}, "
                            "which is not periodic"
                        )
                    requested[axis] = 1
            n1, n2, n3 = requested
            if opts["qpoints"] is not None:
                q_points = opts["qpoints"]
                note = "these are the q points you asked for"
            elif opts["qpath"] is not None:
                # `am1_rs::pbc::q_path`, point for point: each segment contributes
                # `points_per_segment` samples from its start (inclusive) and the final corner
                # is appended once, so consecutive segments do not repeat their shared corner.
                per = max(1, opts["qpath_points"])
                corners = opts["qpath"]
                q_points = []
                for a, b in zip(corners[:-1], corners[1:]):
                    for step in range(per):
                        t = step / per
                        q_points.append([a[k] + t * (b[k] - a[k]) for k in range(3)])
                q_points.append(list(corners[-1]))
                note = (
                    f"a straight-line path through {len(corners)} corners, {per} points per "
                    "segment; only the q commensurate with the supercell are exact"
                )
            else:
                # The same enumeration, in the same order, as `ForceConstants::commensurate_q`.
                q_points = [
                    [i / n1, j / n2, k / n3]
                    for i in range(n1)
                    for j in range(n2)
                    for k in range(n3)
                ]
                note = "these are the q the supercell represents exactly"
            r = native.phonons(
                numbers,
                positions,
                cell,
                pbc,
                supercell=(n1, n2, n3),
                q_points=q_points,
                charge=opts["charge"],
                multiplicity=opts["multiplicity"],
                method=opts["method"],
            )
            print(f"phonon frequencies (cm^-1) from a {n1}x{n2}x{n3} supercell:")
            for q, freqs in zip(q_points, _field(r, "frequencies_cm", mode)):
                row = " ".join(f"{_unsigned_zero(f, 1):>10.1f}" for f in freqs)
                print(f"  q = ({q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f}) {row}")
            _print_asr(
                _field(r, "acoustic_sum_rule_error_before", mode),
                _field(r, "acoustic_sum_rule_error", mode),
            )
            print(f"({note})")

        elif mode == "orbitals":
            r = native.orbitals(numbers, positions, **common)
            n_occ = _field(r, "n_occupied", mode)
            unrestricted = _field(r, "unrestricted", mode)
            print(f"orbital energies (Hartree), {n_occ} occupied:")
            _print_orbitals(
                _field(r, "energies_hartree", mode), n_occ, "alpha" if unrestricted else ""
            )
            if unrestricted:
                beta_occ = _field(r, "beta_n_occupied", mode)
                print(f"\nbeta channel, {beta_occ} occupied:")
                _print_orbitals(_field(r, "beta_energies_hartree", mode), beta_occ, "beta")

            print("\nfrontier orbitals:")

            def frontier(name: str, ev):
                if ev is not None:
                    print(f"  {name:<12} {ev * ev_to_hartree:14.8f} Hartree  {ev:12.6f} eV")

            homo, lumo = r.get("homo_ev"), r.get("lumo_ev")
            frontier("HOMO", homo)
            frontier("LUMO", lumo)
            if homo is not None and lumo is not None:
                frontier("gap", lumo - homo)
            if unrestricted:
                frontier("HOMO [beta]", r.get("homo_beta_ev"))
                frontier("LUMO [beta]", r.get("lumo_beta_ev"))

            if opts["orbital_coefficients"]:
                labels = [
                    f"{_SYMBOLS[numbers[atom]]}{atom + 1}:{shell}"
                    for atom, shell in _field(r, "ao_labels", mode)
                ]

                def print_coefficients(matrix) -> None:
                    header = " ".join(f"{k + 1:>10}" for k in range(len(matrix[0])))
                    print(f"  {'AO':<10} {header}")
                    for label, row in zip(labels, matrix):
                        cells = " ".join(f"{_unsigned_zero(v, 5):>10.5f}" for v in row)
                        print(f"  {label:<10} {cells}")

                print("\nMO coefficients (rows are AOs, columns are orbitals) [alpha]:")
                print_coefficients(_field(r, "coefficients", mode))
                if unrestricted:
                    print("\nMO coefficients [beta]:")
                    print_coefficients(_field(r, "beta_coefficients", mode))

        elif mode == "ir":
            reject_cell("ir")
            r = native.ir_spectrum(numbers, positions, **common)
            print("atomic polar tensor d(mu_a)/d(R_b) (e), rows x/y/z, columns 3*atom+axis:")
            for row in _field(r, "dipole_derivatives", mode):
                print("  " + " ".join(f"{_unsigned_zero(v, 5):9.5f}" for v in row))
            print("\ninfrared spectrum:")
            print("  mode   freq (cm^-1)   intensity (km/mol)   rigid-body")
            for k, nu in enumerate(_field(r, "frequencies_cm", mode), start=1):
                print(
                    f"  {k:>4}  {_unsigned_zero(nu, 2):>13.2f}"
                    f"  {r['intensities_km_per_mol'][k - 1]:>19.4f}"
                    f"  {r['translation_rotation_overlap'][k - 1]:>10.3f}"
                )
            print(
                f"\n({_field(r, 'rigid_body_count', mode)} rigid-body modes were projected out; "
                "the column above measures what is left of them and must be 0)"
            )

        elif mode == "molden":
            reject_cell("molden")
            text = native.molden(
                numbers,
                positions,
                basis=opts["molden_basis"],
                primitives=opts["molden_primitives"],
                deorthogonalize=opts["molden_deorthogonalize"],
                **common,
            )
            if opts["molden_output"]:
                with open(opts["molden_output"], "w", encoding="utf-8") as handle:
                    handle.write(text)
                print(f"molden wavefunction written to {opts['molden_output']}")
            else:
                print(text, end="")

        elif mode == "charges":
            reject_cell("charges")
            if opts["mulliken"]:
                r = native.single_point(numbers, positions, **common)
                _print_charges(numbers, _field(r, "charges", mode))
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
                if opts["mol2_output"]:
                    # The text comes from the crate, not from here: a second hand-written mol2
                    # writer would be a second thing to keep byte-identical.
                    with open(opts["mol2_output"], "w", encoding="utf-8") as handle:
                        handle.write(_field(r, "mol2", mode))
                    print(f"\nmol2 written to {opts['mol2_output']}")

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
