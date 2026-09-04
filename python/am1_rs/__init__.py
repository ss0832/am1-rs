# SPDX-License-Identifier: GPL-3.0-or-later
"""am1-rs: AM1-family semiempirical quantum chemistry (AM1, RM1).

The native API (:mod:`am1_rs.native`) works in atomic units (Hartree, Bohr) for molecules and
in eV/Å for periodic systems. The ASE calculator (:class:`am1_rs.ase.AM1`) uses ASE's eV/Å
convention throughout and covers both.
"""

from __future__ import annotations

from . import native

__all__ = [
    "native",
    "single_point",
    "gradient",
    "pbc_point",
    "divide_conquer",
    "phonons",
    "optimize",
    "frequencies",
    "hessian",
    "am1_bcc",
    "orbitals",
    "molden",
    "ir_spectrum",
    "dipole_derivatives",
    "orbital_response",
    "pbc_hessian",
    "born_charges",
    "dielectric",
    "dielectric_with_extent",
    "dfpt",
    "lo_to_frequencies",
    "__version__",
]

single_point = native.single_point
gradient = native.gradient
pbc_point = native.pbc_point
divide_conquer = native.divide_conquer
phonons = native.phonons
optimize = native.optimize
frequencies = native.frequencies
hessian = native.hessian
am1_bcc = native.am1_bcc
orbitals = native.orbitals
molden = native.molden
ir_spectrum = native.ir_spectrum
dipole_derivatives = native.dipole_derivatives
orbital_response = native.orbital_response
pbc_hessian = native.pbc_hessian
born_charges = native.born_charges
dielectric = native.dielectric
dielectric_with_extent = native.dielectric_with_extent
dfpt = native.dfpt
lo_to_frequencies = native.lo_to_frequencies

# Single source of truth: the version comes from the installed distribution metadata, which
# maturin fills in from Cargo.toml. Keeping a literal here as well is how the three copies got
# out of step in the first place.
try:  # pragma: no cover - trivial fallback
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        __version__ = _version("am1-rs-python")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
except ImportError:  # pragma: no cover
    __version__ = "0.0.0+unknown"
