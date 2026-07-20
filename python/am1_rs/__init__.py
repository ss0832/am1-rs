# SPDX-License-Identifier: GPL-3.0-or-later
"""am1-rs: AM1 semiempirical quantum chemistry.

The native API (:mod:`am1_rs.native`) works in atomic units (Hartree, Bohr). The ASE
calculator (:class:`am1_rs.ase.AM1`) uses ASE's eV / Angstrom convention.
"""

from . import native

__all__ = [
    "native",
    "single_point",
    "gradient",
    "optimize",
    "frequencies",
    "hessian",
    "am1_bcc",
]

single_point = native.single_point
gradient = native.gradient
optimize = native.optimize
frequencies = native.frequencies
hessian = native.hessian
am1_bcc = native.am1_bcc

__version__ = "0.1.3"
