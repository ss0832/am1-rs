# SPDX-License-Identifier: GPL-3.0-or-later
"""Native AM1 API (atomic units: Hartree, Bohr).

Thin wrapper over the compiled ``am1_rs._native`` extension. Input coordinates are
Ångström; energies are returned in Hartree (and, for convenience, the heat of formation
in kcal/mol). This is the raw model surface; the ASE calculator layer converts to eV/Å.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _native


def _as_lists(numbers, positions):
    numbers = [int(z) for z in np.asarray(numbers).reshape(-1)]
    positions = np.asarray(positions, dtype=float).reshape(len(numbers), 3).tolist()
    return numbers, positions


def single_point(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto") -> dict:
    """AM1 single-point energy and properties.

    Parameters
    ----------
    numbers : sequence of int
        Atomic numbers, one per atom.
    positions : array-like, shape (N, 3)
        Cartesian coordinates in **Ångström**.
    charge : float
        Total molecular charge (electrons = Σ Z_valence − charge).
    multiplicity : int
        Spin multiplicity (2S+1); fixes the α/β electron counts.
    reference : str
        SCF reference, independent of the multiplicity: ``"auto"`` (default — RHF for a
        closed-shell singlet, UHF for an open shell), ``"rhf"`` (force restricted; requires a
        closed shell), or ``"uhf"`` (force unrestricted, even for a singlet).

    Returns
    -------
    dict with keys:
        ``energy_hartree`` (float) — total AM1 energy in Hartree;
        ``energy_ev`` (float) — the same energy in eV (AM1's native unit);
        ``heat_of_formation_kcal`` (float) — AM1 heat of formation, kcal/mol;
        ``electronic_ev``, ``core_ev`` (float) — electronic and core–core parts, eV;
        ``charges`` (list[float]) — Mulliken net atomic charges, e;
        ``dipole_debye`` (list[float]) — dipole vector [x, y, z] in Debye;
        ``homo_ev``, ``lumo_ev`` (float or None) — frontier orbital energies, eV;
        ``converged`` (bool) — whether the SCF converged.
    """
    n, p = _as_lists(numbers, positions)
    return _native.single_point(n, p, float(charge), int(multiplicity), reference)


def gradient(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto") -> dict:
    """AM1 energy and analytic nuclear gradient.

    Coordinates in **Ångström**; see :func:`single_point` for ``charge``/``multiplicity``/``reference``.

    Returns
    -------
    dict with keys:
        ``energy_hartree`` (float), ``energy_ev`` (float) — total energy;
        ``heat_of_formation_kcal`` (float);
        ``gradient_hartree_per_bohr`` (list[[float, float, float]]) — dE/dR in atomic units;
        ``gradient_ev_per_angstrom`` (list[[float, float, float]]) — dE/dR in eV/Å (forces = −gradient).
    """
    n, p = _as_lists(numbers, positions)
    return _native.gradient(n, p, float(charge), int(multiplicity), reference)


def optimize(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto") -> dict:
    """L-BFGS geometry optimization on the analytic AM1 gradient.

    Coordinates in **Ångström**; see :func:`single_point` for ``charge``/``multiplicity``/``reference``.

    Returns
    -------
    dict with keys:
        ``positions_angstrom`` (list[[float, float, float]]) — optimized geometry, Å;
        ``energy_hartree`` (float), ``heat_of_formation_kcal`` (float) — at the optimized geometry;
        ``converged`` (bool), ``iterations`` (int).
    """
    n, p = _as_lists(numbers, positions)
    return _native.optimize(n, p, float(charge), int(multiplicity), reference)


def hessian(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto") -> dict:
    """AM1 analytic (CPHF) Cartesian Hessian at the given geometry.

    The Hessian is the matrix of second derivatives ∂²E/∂R_i∂R_j. It is meaningful at any
    geometry; evaluate at a **stationary point** (optimize first) for a physical force-constant
    matrix. Coordinates in **Ångström**; see :func:`single_point` for ``charge``/``multiplicity``/``reference``.

    Returns
    -------
    dict with keys:
        ``hessian_hartree_per_bohr2`` (list[list[float]]) — the ``3N × 3N`` Hessian in **atomic
        units** (Hartree/Bohr²), the native surface's convention;
        ``hessian_ev_per_angstrom2`` (list[list[float]]) — the same matrix in eV/Å²;
        ``ndof`` (int) — number of Cartesian degrees of freedom (``3N``).

    Row/column index ``3*i + k`` is atom ``i``, axis ``k`` (0=x, 1=y, 2=z), in input atom order.
    """
    n, p = _as_lists(numbers, positions)
    return _native.hessian(n, p, float(charge), int(multiplicity), reference)


def am1_bcc(numbers: Sequence[int], positions, charge: float = 0.0) -> dict:
    """AM1-BCC partial charges for AMBER (Jakalian *et al.* 2000/2002).

    Runs the AM1 SCF for Mulliken charges, perceives the molecular graph, assigns the
    antechamber BCC atom types, and applies the exact ``BCCPARM.DAT`` bond charge corrections.
    Coordinates in **Ångström**.

    Returns
    -------
    dict with keys:
        ``charges`` (list[float]) — AM1-BCC net atomic charges, e (Σ = ``charge``);
        ``mulliken`` (list[float]) — the underlying AM1 Mulliken charges before corrections;
        ``atom_types`` (list[str]) — antechamber BCC atom-type codes (``"11"`` … ``"91"``).
    """
    n, p = _as_lists(numbers, positions)
    return _native.am1_bcc(n, p, float(charge))


def frequencies(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto") -> dict:
    """Harmonic vibrational frequencies from the analytic (CPHF) Hessian.

    Evaluate at a **stationary point** (optimize first) for physically meaningful modes.
    Coordinates in **Ångström**; see :func:`single_point` for ``charge``/``multiplicity``/``reference``.

    Returns
    -------
    dict with keys:
        ``frequencies_cm`` (list[float]) — harmonic frequencies in cm⁻¹, ascending; negative
        values denote imaginary modes (saddle point or non-stationary geometry);
        ``eigenvalues`` (list[float]) — mass-weighted Hessian eigenvalues, eV/(Å²·amu).
    """
    n, p = _as_lists(numbers, positions)
    return _native.frequencies(n, p, float(charge), int(multiplicity), reference)
