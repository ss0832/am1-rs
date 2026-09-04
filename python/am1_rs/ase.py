# SPDX-License-Identifier: GPL-3.0-or-later
"""ASE calculator for the AM1 family (AM1, RM1).

Uses ASE's convention throughout: energies in **eV**, forces in **eV/Å**, positions in
**Å**, stress in **eV/Å³**. Internally it calls the native API and converts at this boundary.

Periodic and molecular systems go through the same class. Which one you get is decided by
``atoms.pbc`` — the same rule ASE itself uses — so a structure read from a CIF is periodic and
one read from an XYZ is not, without a separate calculator or a flag to remember.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    from ase.calculators.calculator import (
        Calculator,
        PropertyNotImplementedError,
        all_changes,
    )
    from ase.units import Debye
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The AM1 ASE calculator requires ASE. Install with `pip install am1-rs-python[ase]`."
    ) from exc

from . import native


class AM1(Calculator):
    """AM1/RM1 semiempirical calculator (ASE units: eV, eV/Å, Å, eV/Å³).

    Parameters
    ----------
    charge : float
        Total charge. For a periodic cell this is the charge **per cell**.
    multiplicity : int
        Spin multiplicity (2S+1); fixes the α/β electron counts.
    reference : str
        SCF reference, independent of the multiplicity: ``"auto"`` (default — RHF for a
        closed-shell singlet, UHF for an open shell), ``"rhf"`` (force restricted; requires a
        closed shell), or ``"uhf"`` (force unrestricted, even for a singlet).
    method : str
        ``"am1"`` (default) or ``"rm1"``.
    kpts : tuple of three int, or int, or None
        Monkhorst–Pack mesh for periodic systems. Entries along a non-periodic direction are
        ignored, so ``kpts=(8, 8, 8)`` on a chain is an 8-point mesh, not 512. Ignored entirely
        for a non-periodic structure.
    smearing : float
        Fermi–Dirac smearing width kT in **eV**, for periodic systems. Leave at 0 for a gapped
        system; a metal will not converge without a finite width.
    realspace_cutoff, exchange_cutoff : float
        Periodic real-space sum and exchange-taper cutoffs in **Bohr**. See ``docs/pbc.md``;
        ``exchange_cutoff`` in particular is a documented approximation, not a knob whose
        default is always right.
    e_tol, p_tol, max_scf, mixing : float / int
        Periodic SCF controls. The defaults suit dynamics; tighten ``e_tol``/``p_tol`` before
        taking a finite difference of the energy, because the SCF's own convergence error does
        not cancel between two displaced geometries.
    divide_conquer : bool
        Solve with divide-and-conquer instead of one global diagonalization. For large molecules
        only — the crossover against the full SCF is a few hundred atoms — and molecular systems
        only for now. ``core_size`` and ``buffer_radius`` (Bohr) control it; increase
        ``buffer_radius`` until the property you care about stops moving. A ``RuntimeWarning``
        is raised if the system turns out to have too small a gap for the method's assumption
        that the density matrix decays with distance. See :func:`am1_rs.native.divide_conquer`.

    Charge, multiplicity, reference and method may be supplied **either** here at construction
    (the default for every structure this calculator evaluates) **or** per structure at
    calculation time via ``atoms.info``: ``atoms.info["charge"]`` /
    ``atoms.info["multiplicity"]`` / ``atoms.info["reference"]`` / ``atoms.info["method"]``
    override the constructor values for that structure. Changing any of them re-triggers the
    calculation (cached results are invalidated).

    Populates ``results`` with ``energy`` (eV), ``free_energy`` (eV), ``forces`` (eV/Å),
    ``charges`` (Mulliken, e) and, for a molecule, ``dipole`` (e·Å). Periodic structures also
    get ``stress`` (Voigt 6-vector). Use :meth:`get_hessian` for the analytic Cartesian Hessian
    (eV/Å², molecular only).

    Notes
    -----
    ``free_energy`` is E − TS, and equals ``energy`` whenever ``smearing`` is zero. ASE's
    optimizers and molecular-dynamics drivers use ``free_energy`` because that — not E — is the
    quantity whose derivative the forces are.

    For a slab or a chain the stress is reported per unit **area** or **length** (eV/Å², eV/Å)
    rather than per volume, because a non-periodic direction has no extent to divide by. The
    components touching a non-periodic axis are exactly zero. This means the Voigt vector is
    *not* an ASE 3D stress for those systems: use it for analysis, and do not hand it to a
    variable-cell driver, which assumes 3D.
    """

    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "charges",
        "dipole",
        "stress",
    ]

    default_parameters = {
        "charge": 0.0,
        "multiplicity": 1,
        "reference": "auto",
        "method": "am1",
        "kpts": (1, 1, 1),
        "smearing": 0.0,
        "realspace_cutoff": 40.0,
        "exchange_cutoff": 20.0,
        "e_tol": 1.0e-8,
        "p_tol": 1.0e-7,
        "max_scf": 300,
        "mixing": 0.3,
        "divide_conquer": False,
        "core_size": 12,
        "buffer_radius": 11.0,
        "gap_warn_ev": 0.5,
        "multipole_cutoff": None,
        # Uniform external electric field in ASE units, V/Å. Molecular only.
        "field": None,
    }

    # ------------------------------------------------------------------------------- units
    #: 1 Bohr in Ångström, as *this model* defines it. The crate deliberately uses MOPAC7's
    #: ``a0 = 0.529167`` rather than CODATA, so the field conversion has to use the same one or a
    #: field set in V/Å would not be the field the model applies.
    _BOHR = native.constants()["bohr_to_angstrom"]
    _HARTREE = native.constants()["hartree_to_ev"]

    def _field_au(self, atoms=None):
        """The external field in atomic units, which is what the native surface takes.

        ASE's convention for a field is **V/Å**, i.e. eV per (e·Å). The native layer takes
        Hartree per (e·Bohr). Converting here, once, is what keeps ``field=0.01`` meaning the same
        physical field on both surfaces.
        """
        field = self.parameters.get("field")
        info = getattr(atoms, "info", None) if atoms is not None else None
        if info and "field" in info:
            field = info["field"]
        if field is None:
            return None
        f = np.asarray(field, dtype=float).reshape(-1)
        if f.size != 3:
            raise ValueError(f"field must have three components, got {f.size}")
        # eV/(e·Å) → eV/(e·Bohr) → Hartree/(e·Bohr).
        return (f * self._BOHR / self._HARTREE).tolist()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Last (charge, multiplicity, reference, method) actually used, for cache invalidation
        # against per-structure `atoms.info` overrides, which ASE's own machinery cannot see.
        self._last_state = None
        self._warned_charged_cell = False
        # The lazy `get_*` results, keyed on the geometry and the resolved state. Deliberately
        # *not* `self.results`: see `_cached`.
        self._extra_cache: dict = {}
        # Set by `set_atoms`, which ASE calls from `atoms.calc = calc`.
        self._bound_atoms = None

    def set_atoms(self, atoms):
        """ASE calls this from ``atoms.calc = calc``; that is the only reason it exists.

        Kept in its **own** attribute rather than assigned to ``self.atoms``. ASE's
        ``Calculator.calculate`` deliberately stores a *copy* there, because ``check_state``
        compares it against the live object to decide whether the geometry moved — parking a live
        reference in ``self.atoms`` would make every comparison a comparison of an object with
        itself, and the standard properties would never invalidate.
        """
        self._bound_atoms = atoms

    # ------------------------------------------------------------------ parameter resolution
    @property
    def charge(self) -> float:
        return float(self.parameters["charge"])

    @property
    def multiplicity(self) -> int:
        return int(self.parameters["multiplicity"])

    @property
    def reference(self) -> str:
        return str(self.parameters["reference"])

    @property
    def method(self) -> str:
        return str(self.parameters["method"])

    def _resolve_state(self, atoms):
        """Effective ``(charge, multiplicity, reference, method, field)`` for ``atoms``.

        Per-structure ``atoms.info`` entries take precedence over the values passed to the
        constructor, so either entry point works. The field is included because it changes the
        answer exactly as the charge does, and leaving it out of the state would let a cached
        result survive a field change.
        """
        p = self.parameters
        charge = p["charge"]
        multiplicity = p["multiplicity"]
        reference = p["reference"]
        method = p["method"]
        info = getattr(atoms, "info", None) if atoms is not None else None
        if info:
            if "charge" in info:
                charge = info["charge"]
            if "multiplicity" in info:
                multiplicity = info["multiplicity"]
            if "reference" in info:
                reference = info["reference"]
            if "method" in info:
                method = info["method"]
        field = self._field_au(atoms)
        return (
            float(charge),
            int(multiplicity),
            str(reference),
            str(method),
            None if field is None else tuple(field),
        )

    def _kpts(self):
        kpts = self.parameters["kpts"]
        if kpts is None:
            return (1, 1, 1)
        if np.isscalar(kpts):
            return (int(kpts),) * 3
        kpts = tuple(int(k) for k in kpts)
        if len(kpts) != 3:
            raise ValueError(f"kpts must be three integers, got {kpts!r}")
        return kpts

    def check_state(self, atoms, tol=1e-15):
        """Force a recompute when an ``atoms.info`` override changes even if geometry does not.

        Constructor-level parameters are handled by ASE (``Calculator.set`` resets the cache),
        and geometry, cell and ``pbc`` by ``compare_atoms``. Only the ``atoms.info`` overrides
        are invisible to both.
        """
        system_changes = super().check_state(atoms, tol=tol)
        state = self._resolve_state(atoms)
        if (
            self._last_state is not None
            and state != self._last_state
            and "charge" not in system_changes
        ):
            system_changes = system_changes + ["charge"]
        return system_changes

    # ------------------------------------------------------------------------------ calculate
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        state = self._resolve_state(self.atoms)
        self._last_state = state
        if self.parameters["divide_conquer"]:
            # Periodic too, since 0.2.1: the subsystem buffers are built from the image-aware
            # pair list, so a buffer can wrap through the cell boundary.
            self._calculate_divide_conquer(state, properties)
        elif self.atoms.pbc.any():
            self._calculate_periodic(state)
        else:
            self._calculate_molecular(state, properties)

    def _calculate_divide_conquer(self, state, properties):
        charge, multiplicity, reference, method, field = state
        periodic = bool(self.atoms.pbc.any())
        r = native.divide_conquer(
            self.atoms.get_atomic_numbers(),
            self.atoms.get_positions(),
            charge=charge,
            multiplicity=multiplicity,
            reference=reference,
            method=method,
            core_size=self.parameters["core_size"],
            buffer_radius=self.parameters["buffer_radius"],
            smearing_ev=self.parameters["smearing"] or 0.05,
            e_tol=self.parameters["e_tol"],
            p_tol=self.parameters["p_tol"],
            max_scf=self.parameters["max_scf"],
            mixing=self.parameters["mixing"],
            gap_warn_ev=self.parameters["gap_warn_ev"],
            forces="forces" in properties,
            multipole_cutoff=self.parameters["multipole_cutoff"],
            electric_field=field,
            cell=np.asarray(self.atoms.get_cell(), dtype=float) if periodic else None,
            pbc=self.atoms.pbc if periodic else None,
            realspace_cutoff=self.parameters["realspace_cutoff"],
            exchange_cutoff=self.parameters["exchange_cutoff"],
        )
        self.results["energy"] = r["energy_ev"]
        self.results["free_energy"] = r["free_energy_ev"]
        if "forces_ev_per_angstrom" in r:
            self.results["forces"] = np.asarray(r["forces_ev_per_angstrom"], dtype=float)
        self.results["charges"] = np.asarray(r["charges"], dtype=float)
        self.results["heat_of_formation_kcal"] = r["heat_of_formation_kcal"]
        self.results["fermi_energy"] = r["fermi_energy_ev"]
        self.results["subsystems"] = r["subsystems"]
        self.results["largest_subsystem_aos"] = r["largest_subsystem_aos"]
        self.results["homo_lumo_gap"] = r["homo_lumo_gap_ev"]
        if r["small_gap_warning"]:
            warnings.warn(r["small_gap_warning"], RuntimeWarning, stacklevel=2)

    def _calculate_molecular(self, state, properties):
        charge, multiplicity, reference, method, field = state
        numbers = self.atoms.get_atomic_numbers()
        positions = self.atoms.get_positions()  # Å

        # One SCF either way: the gradient path returns the SCF properties too, so asking for
        # forces does not cost a second pass over the same geometry.
        if "forces" in properties:
            r = native.gradient(
                numbers, positions, charge, multiplicity, reference, method, field
            )
            self.results["forces"] = -np.asarray(
                r["gradient_ev_per_angstrom"], dtype=float
            )
        else:
            r = native.single_point(
                numbers, positions, charge, multiplicity, reference, method, field
            )

        energy_ev = r["energy_ev"]
        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev
        self.results["charges"] = np.asarray(r["charges"], dtype=float)
        self.results["dipole"] = np.asarray(r["dipole_debye"], dtype=float) * Debye
        self.results["heat_of_formation_kcal"] = r.get("heat_of_formation_kcal")

    def _calculate_periodic(self, state):
        charge, multiplicity, reference, method, field = state
        # A field is passed through, not refused. It has to be orthogonal to every lattice vector
        # — normal to a slab, transverse to a chain — and the native side rejects a component
        # along a periodic direction with a message naming it. Through 0.2.1 this refused any
        # field under any cell, which threw the well-defined cases out with the ill-defined one.
        atoms = self.atoms
        cell = np.asarray(atoms.get_cell(), dtype=float)
        for axis in range(3):
            if not atoms.pbc[axis] and np.linalg.norm(cell[axis]) < 1e-8:
                raise ValueError(
                    f"lattice vector {axis} has zero length. A non-periodic direction still "
                    "needs a vector: give it a length large enough to separate the images "
                    "(e.g. atoms.cell[2] = [0, 0, 30] for a slab)."
                )

        r = native.pbc_point(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            cell,
            atoms.pbc,
            kpts=self._kpts(),
            charge=charge,
            multiplicity=multiplicity,
            unrestricted=(reference.strip().lower() in ("uhf", "u", "unrestricted")),
            smearing_ev=self.parameters["smearing"],
            realspace_cutoff=self.parameters["realspace_cutoff"],
            exchange_cutoff=self.parameters["exchange_cutoff"],
            method=method,
            e_tol=self.parameters["e_tol"],
            p_tol=self.parameters["p_tol"],
            max_scf=self.parameters["max_scf"],
            mixing=self.parameters["mixing"],
            electric_field=field,
        )

        self.results["energy"] = r["energy_ev"]
        # E - TS is what the forces differentiate; with no smearing the two coincide.
        self.results["free_energy"] = r["free_energy_ev"]
        self.results["forces"] = np.asarray(r["forces_ev_per_angstrom"], dtype=float)
        self.results["stress"] = np.asarray(r["stress_voigt"], dtype=float)
        self.results["charges"] = np.asarray(r["charges"], dtype=float)
        self.results["fermi_energy"] = r["fermi_energy_ev"]
        self.results["entropy"] = r["entropy_ev"]
        self.results["k_points"] = r["k_points"]
        self.results["iterations"] = r["iterations"]
        self.results["max_image_overlap"] = r["max_image_overlap"]
        if r["charged_cell_warning"] and not self._warned_charged_cell:
            # Once per calculator, not once per step: this fires from inside molecular dynamics,
            # and a warning repeated every step is a warning nobody reads.
            self._warned_charged_cell = True
            warnings.warn(r["charged_cell_warning"], RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------------------ extras
    def get_stress(self, atoms=None):
        """Stress as an ASE Voigt 6-vector.

        Raises ``PropertyNotImplementedError`` for a non-periodic structure, which is ASE's
        convention: a molecule in free space has no stress, and returning zeros would let a
        variable-cell optimizer run happily on nothing.
        """
        if atoms is None:
            atoms = self.atoms
        if atoms is None or not atoms.pbc.any():
            raise PropertyNotImplementedError(
                "stress is only defined for a periodic structure; set atoms.pbc and atoms.cell"
            )
        return self.get_property("stress", atoms)

    def get_dipole_moment(self, atoms=None):
        """Dipole moment in e·Å. Not defined under periodic boundary conditions."""
        if atoms is None:
            atoms = self.atoms
        if atoms is not None and atoms.pbc.any():
            raise PropertyNotImplementedError(
                "the dipole of a periodic cell is not defined by the charge distribution "
                "alone (it depends on the choice of unit cell); use the Berry phase "
                "polarization instead"
            )
        return self.get_property("dipole", atoms)

    def get_hessian(self, atoms=None):
        """Analytic (CPHF) Cartesian Hessian in **eV/Å²** (ASE convention).

        Returns the ``3N × 3N`` matrix of second energy derivatives as a NumPy array; row/column
        ``3*i + k`` is atom ``i``, axis ``k`` (0=x, 1=y, 2=z). Charge, multiplicity, reference
        and method are resolved exactly as for the other properties (constructor value, or an
        ``atoms.info`` override). Evaluate at a relaxed geometry for a physically meaningful
        force-constant matrix.

        For a periodic structure this returns the **k-point** force constants at ``q = 0``, in the
        same ``3N × 3N`` Cartesian form, using ``kpts``.
        """
        atoms = self._require_atoms(atoms, "get_hessian")
        charge, multiplicity, _, method, _ = self._resolve_state(atoms)
        if atoms.pbc.any():
            h = self._cached(
                "pbc_hessian",
                atoms,
                lambda: native.pbc_hessian(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    np.asarray(atoms.get_cell(), dtype=float),
                    atoms.pbc,
                    kpts=self._kpts(),
                    charge=charge,
                    multiplicity=multiplicity,
                    method=method,
                    realspace_cutoff=self.parameters["realspace_cutoff"],
                    exchange_cutoff=self.parameters["exchange_cutoff"],
                ),
            )
        else:
            # Shares the solve with get_frequencies / get_ir_spectrum / get_dipole_derivatives.
            h = self._vibrational(atoms)
        return np.asarray(h["hessian_ev_per_angstrom2"], dtype=float)

    # ------------------------------------------------------------------------ parity surface
    #
    # Everything below mirrors `am1_rs.native`, in ASE units. The redundancy is deliberate: the
    # two surfaces are meant to expose the same features so that choosing one is a choice of unit
    # convention and not a choice of capability.
    #
    # All of these are **explicit methods, never computed by `calculate()`**. A Hessian, a phonon
    # band or an infrared spectrum costs orders of magnitude more than an energy, and a property
    # that appeared merely because something touched `atoms.calc.results` would be a trap.
    #
    # They memoize into `self._extra_cache`, keyed on the geometry, the resolved state and their
    # own arguments — *not* into `self.results`. Through 0.2.1 they did use `self.results` and the
    # invalidation did not work: `results` is cleared by `Calculator.get_property`, which these
    # methods never go through, so moving the atoms and asking again returned the old geometry's
    # answer. See `_cached`.

    def _require_atoms(self, atoms, who):
        if atoms is None:
            # `self.atoms` is only populated once ASE has run a property through `calculate`, and
            # `_bound_atoms` only once the calculator has been attached to an `Atoms`.
            atoms = self.atoms if self.atoms is not None else self._bound_atoms
        if atoms is None:
            raise ValueError(
                f"{who} requires an Atoms object: pass one explicitly, e.g. "
                f"calc.{who}(atoms=atoms), or evaluate a property first "
                "(atoms.get_potential_energy()) so the calculator learns its system."
            )
        return atoms

    def _molecular_only(self, atoms, who):
        if atoms.pbc.any():
            raise PropertyNotImplementedError(
                f"{who} is implemented for molecular systems; clear atoms.pbc, or use the "
                "periodic equivalents (get_phonons, get_born_charges, get_dielectric_tensor)."
            )

    @staticmethod
    def _geometry_key(atoms):
        """An exact key for one geometry: species, positions, cell and periodicity, as bytes.

        Bytes rather than values so the comparison is bit-for-bit — a geometry differing in the
        last bit misses, which is the conservative direction for a cache.
        """
        return (
            atoms.get_atomic_numbers().tobytes(),
            np.ascontiguousarray(atoms.get_positions(), dtype=float).tobytes(),
            np.ascontiguousarray(atoms.cell[:], dtype=float).tobytes(),
            tuple(bool(p) for p in atoms.pbc),
        )

    def _cached(self, key, atoms, build):
        """Run `build` once per (geometry, state, arguments) and memoize it under `key`.

        # Why this does not live in `self.results`

        It used to, and it was **stale**. `self.results` is cleared by `Calculator.get_property`,
        which calls `check_state` and then `reset()` — but the lazy methods below never go through
        `get_property`. So a caller who moved the atoms and asked for a frequency again, without an
        energy call in between to trigger that machinery, got the *previous geometry's* answer:

            f1 = atoms.calc.get_frequencies(atoms)
            atoms.positions += 0.5
            f2 = atoms.calc.get_frequencies(atoms)   # was f1

        Nothing announced it, and the docs claimed the opposite. `tests/test_lazy_cache.py`
        asserts the whole family against it now.

        The fix is to key on the geometry and the resolved state directly rather than to rely on
        being cleared by something else. `_geometry_key` covers species, positions, cell and pbc;
        `_resolve_state` covers charge, multiplicity, reference, method and the field, including
        the `atoms.info` overrides ASE's own comparison cannot see.
        """
        full = (key, self._geometry_key(atoms), self._resolve_state(atoms))
        hit = self._extra_cache.get(key)
        if hit is not None and hit[0] == full:
            return hit[1]
        value = build()
        self._extra_cache[key] = (full, value)
        return value

    def reset(self):
        """Drop the lazy cache along with `results`, so `Calculator.reset()` means what it says."""
        super().reset()
        self._extra_cache = {}

    @staticmethod
    def _arg_key(name, *args):
        """A cache key that includes the arguments, for methods whose result depends on them.

        `get_phonons(supercell=(2,1,1))` and `get_phonons(supercell=(4,1,1))` are different
        calculations at the same geometry, so caching them under one key would hand back the
        first answer for the second question. These four methods were left uncached entirely for
        that reason, which meant re-running a CPHF or a DFPT solve on every call instead.

        `repr` on an ndarray is not stable enough to key on (it elides large arrays), so
        array-like arguments are converted to nested lists first.
        """
        parts = []
        for a in args:
            if a is None or isinstance(a, (str, bool, int, float)):
                parts.append(repr(a))
            else:
                parts.append(repr(np.asarray(a).tolist()))
        return f"{name}:" + "|".join(parts)

    # ------------------------------------------------------------------- one solve, many answers
    def _vibrational(self, atoms, want_response=False, want_density=False):
        """The Hessian, the normal modes, the atomic polar tensor and the intensities, together.

        All four are contractions of one CPHF solve, and asking for them separately ran that solve
        once per question — up to five times for a caller that wanted the spectrum, the Hessian and
        the orbital response. See :func:`am1_rs.native.vibrations`.

        The two large sections are keyed separately rather than always fetched, because `u_ov` and
        the response densities are `O(ndof · n_ov)` and `O(ndof · nao²)`.
        """
        charge, multiplicity, reference, method, field = self._resolve_state(atoms)
        return self._cached(
            self._arg_key("vibrational", want_response, want_density),
            atoms,
            lambda: native.vibrations(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                charge,
                multiplicity,
                reference,
                method,
                field,
                orbital_response=want_response,
                response_density=want_density,
            ),
        )

    def get_frequencies(self, atoms=None):
        """Harmonic vibrational frequencies in cm⁻¹, from the analytic (CPHF) Hessian.

        Evaluate at a relaxed geometry. Negative values denote imaginary modes.
        """
        atoms = self._require_atoms(atoms, "get_frequencies")
        self._molecular_only(atoms, "get_frequencies")
        return np.asarray(self._vibrational(atoms)["frequencies_cm"], dtype=float)

    def get_ir_spectrum(self, atoms=None):
        """Infrared spectrum: frequencies (cm⁻¹), intensities (km/mol) and the raw tensor.

        **This runs an analytic Hessian**, so it costs orders of magnitude more than an energy.
        Returns the whole dict from :func:`am1_rs.native.ir_spectrum`, with
        ``dipole_derivatives`` the ``3 × 3N`` atomic polar tensor in units of ``e``.
        """
        atoms = self._require_atoms(atoms, "get_ir_spectrum")
        self._molecular_only(atoms, "get_ir_spectrum")
        return self._vibrational(atoms)

    def get_dipole_derivatives(self, atoms=None):
        """The atomic polar tensor ``∂μ_α/∂R_{a,β}`` as a ``3 × 3N`` array, in units of ``e``.

        Same cost as :meth:`get_ir_spectrum` — it is where most of that cost goes.
        """
        atoms = self._require_atoms(atoms, "get_dipole_derivatives")
        self._molecular_only(atoms, "get_dipole_derivatives")
        r = self._vibrational(atoms)
        return np.asarray(r["dipole_derivatives"], dtype=float)

    def get_orbitals(self, atoms=None):
        """Orbital energies (eV), coefficients and occupations.

        Returns the dict from :func:`am1_rs.native.orbitals`. Energies are given in both Hartree
        and eV; ASE has no convention for orbital coefficients, so those are passed through as
        they are — rows are atomic orbitals, columns molecular orbitals.
        """
        atoms = self._require_atoms(atoms, "get_orbitals")
        self._molecular_only(atoms, "get_orbitals")
        charge, multiplicity, reference, method, field = self._resolve_state(atoms)
        return self._cached(
            "orbitals",
            atoms,
            lambda: native.orbitals(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                charge,
                multiplicity,
                reference,
                method,
                field,
            ),
        )

    def get_orbital_response(self, atoms=None, response_density=False):
        """First-order orbital response ``U^j_{ai}`` from the CPHF equations.

        **Runs an analytic Hessian.** The Hessian comes back in the same dict, so asking for both
        costs one calculation.
        """
        atoms = self._require_atoms(atoms, "get_orbital_response")
        self._molecular_only(atoms, "get_orbital_response")
        return self._vibrational(atoms, want_response=True, want_density=response_density)

    def write_molden(self, path, atoms=None):
        """Write the wavefunction to `path` in Molden format.

        See :func:`am1_rs.native.molden` for the caveat that travels with the file: NDDO assumes
        an orthonormal AO basis, so the coefficients are in an implicitly orthogonalized basis
        while the Slater functions listed are the raw ones.
        """
        atoms = self._require_atoms(atoms, "write_molden")
        self._molecular_only(atoms, "write_molden")
        charge, multiplicity, reference, method, field = self._resolve_state(atoms)
        text = native.molden(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            charge,
            multiplicity,
            reference,
            method,
            field,
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def get_am1_bcc_charges(self, atoms=None):
        """AM1-BCC partial charges for AMBER, in ``e``.

        Fixed to the AM1 parameterization by the increments themselves — they were fitted to AM1
        Mulliken charges, so applying them to an RM1 density would not be AM1-BCC. **Check the
        returned warnings**: a non-empty list means some bond kept its raw Mulliken charge, which
        is a difference of tenths of an electron rather than a rounding.
        """
        atoms = self._require_atoms(atoms, "get_am1_bcc_charges")
        self._molecular_only(atoms, "get_am1_bcc_charges")
        charge, multiplicity, _, _, _ = self._resolve_state(atoms)
        r = self._cached(
            "am1_bcc",
            atoms,
            lambda: native.am1_bcc(
                atoms.get_atomic_numbers(), atoms.get_positions(), charge, multiplicity
            ),
        )
        return r

    def optimize(self, atoms=None, apply=True):
        """L-BFGS geometry optimization on the analytic gradient.

        Returns the optimized positions in Å. With ``apply`` (the default) the positions are
        written back into `atoms`, which is what ASE users expect from an in-place relaxation;
        pass ``apply=False`` to leave the structure alone.

        ASE has its own optimizers, and for anything beyond a plain relaxation they are the
        better choice — this is here so the ASE surface exposes what the native one does.
        """
        atoms = self._require_atoms(atoms, "optimize")
        self._molecular_only(atoms, "optimize")
        charge, multiplicity, reference, method, field = self._resolve_state(atoms)
        r = native.optimize(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            charge,
            multiplicity,
            reference,
            method,
            field,
        )
        positions = np.asarray(r["positions_angstrom"], dtype=float)
        if apply:
            atoms.set_positions(positions)
            # `reset()`, not `results.clear()`: the latter left `self.atoms` holding the
            # pre-optimization geometry and left the lazy cache untouched.
            self.reset()
        return positions

    # ---------------------------------------------------------------------- periodic response

    def get_phonons(self, atoms=None, supercell=(2, 2, 2), q_points=None):
        """Phonon frequencies (cm⁻¹) from supercell force constants.

        ``supercell`` is the convergence knob and controls two things at once: how far ``Φ(T)``
        is resolved, and — since Γ on an ``n``-fold supercell is the primitive cell at ``n``
        k-points — the k-sampling underneath. Cost grows as the supercell's atom count cubed.
        """
        atoms = self._require_atoms(atoms, "get_phonons")
        if not atoms.pbc.any():
            raise PropertyNotImplementedError(
                "phonons need a periodic cell; use get_frequencies() for a molecule"
            )
        charge, multiplicity, _, method, _ = self._resolve_state(atoms)
        return self._cached(
            self._arg_key("phonons", supercell, q_points),
            atoms,
            lambda: native.phonons(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                supercell=supercell,
                q_points=q_points,
                charge=charge,
                multiplicity=multiplicity,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
            ),
        )

    def get_dfpt_frequencies(
        self,
        q_points,
        atoms=None,
        long_range="auto",
        cpscf_tol=1.0e-10,
        cpscf_max_iter=200,
        cpscf_mixing=0.7,
    ):
        """Phonons at **arbitrary** ``q`` by density-functional perturbation theory.

        No supercell: the response couples ``k`` to ``k + q`` on the primitive cell directly.
        ``q_points`` are fractional coordinates of the primitive reciprocal lattice.

        On a 3D cell this is the **full** ``D(q)``, long-range monopole channel included, so its
        ``q → 0`` limit is direction dependent — which is the physics. Do **not** also apply
        :meth:`get_lo_to_frequencies`, which exists to give that same physics to the supercell
        route :meth:`get_phonons`. Use one or the other, not both.
        """
        atoms = self._require_atoms(atoms, "get_dfpt_frequencies")
        if not atoms.pbc.any():
            raise PropertyNotImplementedError("DFPT needs a periodic cell")
        charge, multiplicity, _, method, _ = self._resolve_state(atoms)
        return self._cached(
            self._arg_key(
                "dfpt", q_points, long_range, cpscf_tol, cpscf_max_iter, cpscf_mixing
            ),
            atoms,
            lambda: native.dfpt(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                q_points,
                kpts=self._kpts(),
                charge=charge,
                multiplicity=multiplicity,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
                long_range=long_range,
                cpscf_tol=cpscf_tol,
                cpscf_max_iter=cpscf_max_iter,
                cpscf_mixing=cpscf_mixing,
            ),
        )

    def get_lo_to_frequencies(
        self,
        direction=(1.0, 0.0, 0.0),
        q_points=None,
        supercell=(2, 2, 2),
        atoms=None,
        enforce_acoustic_sum_rule=True,
    ):
        """Supercell phonons in cm⁻¹ with the **LO–TO splitting** restored.

        **Three-dimensional cells only**: ``4π(q·Z*)²/(Ω q·ε_∞·q)`` is the 3D form and ``Ω`` has
        to be a volume, so a chain or a slab is refused rather than given the wrong units.

        :meth:`get_phonons` alone returns the transverse branches at ``Γ`` and misses the
        longitudinal shift, because a truncated real-space ``Φ(T)`` cannot carry the
        dipole–dipole tail. ``direction`` is required because the ``q → 0`` limit is direction
        dependent — that is what the splitting is.

        Returns the same dict the native API does, including ``frequencies_cm_no_lo_to`` so the
        size of the shift is visible. Not combinable with :meth:`get_dfpt_frequencies`, which
        already carries the long-range channel.
        """
        atoms = self._require_atoms(atoms, "get_lo_to_frequencies")
        if not atoms.pbc.all():
            raise PropertyNotImplementedError(
                "LO-TO splitting is three-dimensional: it needs a cell periodic along all three "
                "axes, because the non-analytic term divides by the cell volume"
            )
        charge, multiplicity, _, method, _ = self._resolve_state(atoms)
        return self._cached(
            self._arg_key(
                "lo_to", direction, q_points, supercell, enforce_acoustic_sum_rule
            ),
            atoms,
            lambda: native.lo_to_frequencies(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                supercell=supercell,
                direction=direction,
                q_points=q_points,
                kpts=self._kpts(),
                charge=charge,
                multiplicity=multiplicity,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
                enforce_acoustic_sum_rule=enforce_acoustic_sum_rule,
            ),
        )

    def get_born_charges(self, atoms=None):
        """Born effective charges ``Z*_{a,αβ}`` as an ``(nat, 3, 3)`` array, in ``e``."""
        atoms = self._require_atoms(atoms, "get_born_charges")
        if not atoms.pbc.any():
            raise PropertyNotImplementedError(
                "Born charges need a periodic cell; the molecular equivalent is "
                "get_dipole_derivatives()"
            )
        charge, multiplicity, _, method, _ = self._resolve_state(atoms)
        r = self._cached(
            "born_charges",
            atoms,
            lambda: native.born_charges(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                kpts=self._kpts(),
                charge=charge,
                multiplicity=multiplicity,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
            ),
        )
        return np.asarray(r["born_charges"], dtype=float)

    def get_dielectric_tensor(self, atoms=None):
        """Electronic dielectric tensor ``ε_∞`` as a 3×3 array.

        **Three-dimensional cells only**: ``ε_∞ = 1 + 4πα/Ω`` needs ``Ω`` to be a volume, and a
        chain or a slab is refused rather than divided by a length. The clamped-ion
        polarizability is in the same dict under ``polarizability_bohr3``.

        For a slab or a chain, :meth:`get_dielectric_tensor_with_extent` does the conversion once
        you say how thick the material is.
        """
        atoms = self._require_atoms(atoms, "get_dielectric_tensor")
        if not atoms.pbc.all():
            raise PropertyNotImplementedError(
                "the electronic dielectric tensor is three-dimensional; a chain or a slab needs "
                "a thickness, which the cell does not fix -- pass one to "
                "get_dielectric_tensor_with_extent(), or use get_polarizability() / "
                "get_dielectric_function(), which need no such choice"
            )
        charge, multiplicity, _, method, _ = self._resolve_state(atoms)
        r = self._cached(
            "dielectric",
            atoms,
            lambda: native.dielectric(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                kpts=self._kpts(),
                charge=charge,
                multiplicity=multiplicity,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
            ),
        )
        return np.asarray(r["epsilon_infinity"], dtype=float)

    def get_dielectric_tensor_with_extent(
        self, slab_thickness=None, wire_cross_section=None, atoms=None, full=False
    ):
        """``ε_∞`` for a **slab or a chain**, given the extent you assign the material.

        Parameters
        ----------
        slab_thickness : float, optional
            Thickness in **Angstrom**, for a cell periodic in two directions.
        wire_cross_section : float, optional
            Cross-sectional area in **Angstrom²**, for a cell periodic in one. The section is
            taken to be circular, which is what fixes the transverse depolarization factor at 1/2.
        full : bool
            Return the whole dict rather than just the tensor.

        Exactly one extent is required and there is no default: a supercell says where the atoms
        are, not where the material stops, and every choice here changes ``eps``. What does *not*
        depend on the choice comes back in the dict (``full=True``) as ``sheet_susceptibility``
        and ``inverse_sheet_response``, both in Bohr, and for a slab ``rytova_keldysh_length``.
        Those are scalars averaged over the two-dimensional half; the same identities hold per
        direction against the returned tensor, which is where to look if the direction matters.

        The conversion carries the depolarization factor of the assumed body, so the out-of-plane
        law is ``1/(1 - 4*pi*chi)`` and not ``1 + 4*pi*chi``: the polarizability this crate
        computes is the response to the *external* field, with the depolarizing field of the
        induced charges already inside it.
        """
        atoms = self._require_atoms(atoms, "get_dielectric_tensor_with_extent")
        n_periodic = int(np.count_nonzero(atoms.pbc))
        if n_periodic == 3:
            raise ValueError(
                "this cell is periodic in three dimensions, where the volume is the cell's own "
                "and nothing has to be assigned -- use get_dielectric_tensor()"
            )
        if n_periodic == 0:
            raise ValueError("this cell has no periodic direction; use get_polarizability()")
        # Angstrom in, Bohr out: a thickness is a length and a cross-section is an area, so they
        # convert with different powers. Getting that wrong is a silent factor of ~0.28.
        per_bohr = 1.0 / self._BOHR
        d = None if slab_thickness is None else float(slab_thickness) * per_bohr
        s = None if wire_cross_section is None else float(wire_cross_section) * per_bohr**2
        charge, multiplicity, _, method, _ = self._resolve_state(atoms)
        r = self._cached(
            f"dielectric_extent:{d!r}:{s!r}",
            atoms,
            lambda: native.dielectric_with_extent(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                slab_thickness=d,
                wire_cross_section=s,
                kpts=self._kpts(),
                charge=charge,
                multiplicity=multiplicity,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
            ),
        )
        if full:
            return r
        return np.asarray(r["epsilon_infinity"], dtype=float)

    def get_polarizability(self, atoms=None):
        """Clamped-ion polarizability as a 3x3 array, Bohr^3.

        Works for a chain and a slab as well as a crystal -- it is a *response*, and a response is
        well defined whatever the cell is periodic in. ``get_dielectric_tensor`` is the one that is
        three-dimensional, because ``eps_inf = 1 + 4*pi*alpha/V`` needs ``V`` to be a volume.

        Divide by the cell measure yourself if you want a susceptibility, and mind what it is:
        dimensionless in 3D, a **length** for a slab, an **area** for a chain.
        """
        atoms = self._require_atoms(atoms, "get_polarizability")
        charge, multiplicity, _, method, _ = self._resolve_state(atoms)
        r = self._cached(
            "polarizability",
            atoms,
            lambda: native.polarizability(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                kpts=self._kpts(),
                charge=charge,
                multiplicity=multiplicity,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
            ),
        )
        return np.asarray(r["polarizability_bohr3"], dtype=float)
    def get_dielectric_function(self, q, atoms=None, chain_radius=None):
        """Macroscopic longitudinal dielectric function ``eps(q)`` along ``q``.

        ``q`` is a Cartesian wavevector in **inverse Angstrom** here, matching the rest of the ASE
        surface; the native layer takes inverse Bohr and the conversion happens at this boundary.

        Three dimensions gives a constant and ``get_dielectric_tensor`` returns it directly. A slab
        and a chain do **not**: ``eps(q) -> 1`` at long wavelength, because a sheet or a wire does
        not screen a field whose wavelength exceeds its own extent -- the same fact as a slab
        having no LO-TO splitting at Gamma.

        ``chain_radius`` (in Angstrom) is required for a chain and ignored otherwise: the 1D
        Coulomb kernel is a logarithm and has no value without a reference length.
        """
        atoms = self._require_atoms(atoms, "get_dielectric_function")
        if not atoms.pbc.any():
            raise PropertyNotImplementedError(
                "a molecule has no dielectric function; use the polarizability"
            )
        charge, multiplicity, _, method, _ = self._resolve_state(atoms)
        c = native.constants()
        # A wavevector is an inverse length, so it converts the other way from a position.
        q_bohr = (np.asarray(q, dtype=float).reshape(-1) * c["bohr_to_angstrom"]).tolist()
        radius = None if chain_radius is None else float(chain_radius) / c["bohr_to_angstrom"]
        key = f"dielectric_function:{q_bohr!r}:{radius!r}"
        return self._cached(
            key,
            atoms,
            lambda: native.dielectric_function(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                q_bohr,
                chain_radius=radius,
                kpts=self._kpts(),
                charge=charge,
                multiplicity=multiplicity,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
            ),
        )
    def get_polarization(self, atoms=None, strings: int = 8):
        """Berry-phase polarization as a 3-vector, ``e/Bohr²``.

        **Three-dimensional, restricted cells only.** ``strings`` is the number of k points per
        Berry-phase string, resampled independently of ``kpts``.

        Defined **modulo the quantum** ``e a_α/Ω``: only differences between two states on a
        common branch are physical, so comparing two geometries means reducing both to one branch
        rather than subtracting these numbers directly.

        The full dictionary -- electronic and ionic halves, the phases in turns, the quantum -- is
        available from ``am1_rs.native.polarization``.
        """
        atoms = self._require_atoms(atoms, "get_polarization")
        if not atoms.pbc.all():
            raise PropertyNotImplementedError(
                "the Berry-phase polarization is three-dimensional; a chain or a slab is "
                "polarized along its periodic directions only, which this does not separate out"
            )
        charge, _, _, method, _ = self._resolve_state(atoms)
        r = self._cached(
            f"polarization:{strings}",
            atoms,
            lambda: native.polarization(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                kpts=self._kpts(),
                strings=strings,
                charge=charge,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
            ),
        )
        return np.asarray(r["polarization"], dtype=float)

    def get_finite_field(self, field, atoms=None):
        """Solve in a finite field **along** a periodic direction, by the Berry-phase enthalpy.

        ``field`` is in **V/Å**, matching ``AM1(field=...)`` and the rest of the ASE surface; the
        native layer takes atomic units and the conversion happens here.

        ``F·R`` is unbounded along a periodic direction, so this minimizes ``E - Ω E·P`` with ``P``
        the Berry phase. For a field **orthogonal** to every lattice vector -- normal to a slab,
        transverse to a chain -- pass ``AM1(field=...)`` instead and call ``get_potential_energy``:
        that case is an ordinary calculation and needs none of this machinery.

        Returns the native dictionary, which carries ``enthalpy_ev`` (the quantity minimized),
        ``energy_ev``, and the polarization it converged to.
        """
        atoms = self._require_atoms(atoms, "get_finite_field")
        if not atoms.pbc.all():
            raise PropertyNotImplementedError(
                "the Berry-phase finite field is three-dimensional; for a field along a "
                "non-periodic direction use AM1(field=...), which needs none of this"
            )
        charge, _, _, method, _ = self._resolve_state(atoms)
        f = np.asarray(field, dtype=float).reshape(-1)
        if f.size != 3:
            raise ValueError(f"field must have three components, got {f.size}")
        # V/Å to Hartree per e·Bohr, through the crate's own constants rather than a second copy.
        c = native.constants()
        au = f * c["bohr_to_angstrom"] / c["hartree_to_ev"]
        key = f"finite_field:{au[0]!r}:{au[1]!r}:{au[2]!r}"
        return self._cached(
            key,
            atoms,
            lambda: native.finite_field(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                atoms.pbc,
                au.tolist(),
                kpts=self._kpts(),
                charge=charge,
                method=method,
                realspace_cutoff=self.parameters["realspace_cutoff"],
                exchange_cutoff=self.parameters["exchange_cutoff"],
            ),
        )


__all__ = ["AM1"]
