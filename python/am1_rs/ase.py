# SPDX-License-Identifier: GPL-3.0-or-later
"""ASE calculator for AM1.

Uses ASE's convention throughout: energies in **eV**, forces in **eV/Å**, positions in
**Å**. Internally it calls the native (atomic-unit) API and converts at this boundary.
"""

from __future__ import annotations

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The AM1 ASE calculator requires ASE. Install with `pip install am1-rs-python[ase]`."
    ) from exc

from . import native


class AM1(Calculator):
    """AM1 semiempirical calculator (ASE units: eV, eV/Å, Å).

    Parameters
    ----------
    charge : float
        Total molecular charge.
    multiplicity : int
        Spin multiplicity (2S+1); fixes the α/β electron counts.
    reference : str
        SCF reference, independent of the multiplicity: ``"auto"`` (default — RHF for a
        closed-shell singlet, UHF for an open shell), ``"rhf"`` (force restricted; requires a
        closed shell), or ``"uhf"`` (force unrestricted, even for a singlet).

    Charge, multiplicity and reference may be supplied **either** here at construction (the
    default for every structure this calculator evaluates) **or** per structure at calculation
    time via ``atoms.info``: ``atoms.info["charge"]`` / ``atoms.info["multiplicity"]`` /
    ``atoms.info["reference"]`` override the constructor values for that structure. Changing any
    of them re-triggers the calculation (cached results are invalidated).

    Populates ``results`` with ``energy`` (eV), ``free_energy`` (eV, = energy), ``forces``
    (eV/Å), ``charges`` (Mulliken, e), ``dipole`` (e·Å), and ``heat_of_formation_kcal``.
    Use :meth:`get_hessian` for the analytic Cartesian Hessian (eV/Å²).
    """

    implemented_properties = ["energy", "free_energy", "forces", "charges", "dipole"]

    def __init__(self, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", **kwargs):
        super().__init__(**kwargs)
        self.charge = float(charge)
        self.multiplicity = int(multiplicity)
        self.reference = str(reference)
        # Last (charge, multiplicity, reference) actually used, for cache invalidation.
        self._last_state = None

    def _resolve_state(self, atoms):
        """Effective ``(charge, multiplicity, reference)`` for ``atoms``.

        Per-structure ``atoms.info`` entries (``"charge"``, ``"multiplicity"``, ``"reference"``)
        take precedence over the values passed to the constructor, so either entry point works.
        """
        charge = self.charge
        multiplicity = self.multiplicity
        reference = self.reference
        info = getattr(atoms, "info", None) if atoms is not None else None
        if info:
            if "charge" in info:
                charge = float(info["charge"])
            if "multiplicity" in info:
                multiplicity = int(info["multiplicity"])
            if "reference" in info:
                reference = str(info["reference"])
        return float(charge), int(multiplicity), str(reference)

    def check_state(self, atoms, tol=1e-15):
        """Force a recompute when charge/multiplicity/reference change even if geometry does not."""
        system_changes = super().check_state(atoms, tol=tol)
        state = self._resolve_state(atoms)
        if (
            self._last_state is not None
            and state != self._last_state
            and "charge" not in system_changes
        ):
            system_changes = system_changes + ["charge"]
        return system_changes

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        charge, multiplicity, reference = self._resolve_state(self.atoms)
        self._last_state = (charge, multiplicity, reference)
        numbers = self.atoms.get_atomic_numbers()
        positions = self.atoms.get_positions()  # Å

        if "forces" in properties:
            g = native.gradient(numbers, positions, charge, multiplicity, reference)
            # AM1 energies are natively eV; use them directly (no Hartree round-trip).
            energy_ev = g["energy_ev"]
            grad = np.asarray(g["gradient_ev_per_angstrom"], dtype=float)  # eV/Å
            self.results["forces"] = -grad
        else:
            energy_ev = None

        sp = native.single_point(numbers, positions, charge, multiplicity, reference)
        if energy_ev is None:
            energy_ev = sp["energy_ev"]

        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev
        self.results["charges"] = np.asarray(sp["charges"], dtype=float)
        self.results["dipole"] = np.asarray(sp["dipole_debye"], dtype=float) * 0.2081943  # Debye → e·Å
        self.results["heat_of_formation_kcal"] = sp["heat_of_formation_kcal"]

    def get_hessian(self, atoms=None):
        """Analytic (CPHF) Cartesian Hessian in **eV/Å²** (ASE convention).

        Returns the ``3N × 3N`` matrix of second energy derivatives as a NumPy array; row/column
        ``3*i + k`` is atom ``i``, axis ``k`` (0=x, 1=y, 2=z). Charge, multiplicity and reference
        are resolved exactly as for the other properties (constructor value, or an ``atoms.info``
        override). Evaluate at a relaxed geometry for a physically meaningful force-constant
        matrix.
        """
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("get_hessian requires an Atoms object")
        charge, multiplicity, reference = self._resolve_state(atoms)
        h = native.hessian(
            atoms.get_atomic_numbers(), atoms.get_positions(), charge, multiplicity, reference
        )
        return np.asarray(h["hessian_ev_per_angstrom2"], dtype=float)


__all__ = ["AM1"]
