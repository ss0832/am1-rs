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


def constants() -> dict:
    """The model's unit conversions, as the Rust crate defines them.

    Deliberately **not** CODATA: the crate uses MOPAC7's ``ev = 27.21`` and ``a0 = 0.529167``
    throughout, which is what makes its heats of formation comparable with MOPAC's. Anything on
    this side that converts between the crate's units should take the factor from here rather
    than write it down, so the two cannot drift apart.

    Keys: ``hartree_to_ev``, ``ev_to_hartree``, ``angstrom_to_bohr``, ``bohr_to_angstrom``,
    ``ev_to_kcal``, ``kcal_to_ev``, ``au_dipole_to_debye``.
    """
    return _native.constants()


def _as_lists(numbers, positions):
    numbers = [int(z) for z in np.asarray(numbers).reshape(-1)]
    positions = np.asarray(positions, dtype=float).reshape(len(numbers), 3).tolist()
    return numbers, positions


def _field(electric_field):
    """Normalize an external field argument to a plain 3-list, or ``None``.

    The field is in **atomic units** (Hartree per e·Bohr) throughout this module, matching every
    other quantity here. The ASE layer takes V/Å and converts at its own boundary.
    """
    if electric_field is None:
        return None
    f = np.asarray(electric_field, dtype=float).reshape(-1)
    if f.size != 3:
        raise ValueError(f"electric_field must have three components, got {f.size}")
    return f.tolist()


def single_point(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None) -> dict:
    """Single-point energy and properties.

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
    method : str
        NDDO parameterization: ``"am1"`` (default) or ``"rm1"``. RM1 shares AM1's functional
        form, so only the parameters differ; it covers H, C, N, O, P, S, F, Cl, Br, I. Asking
        for an element the chosen method does not parameterize is an error naming both.
    electric_field : array-like of three floats, or None
        Uniform external electric field in **atomic units** (Hartree per e·Bohr). The energy
        becomes ``E = E₀ − μ·F`` with the model's own dipole. **Molecules only**: a periodic cell
        with a field is an error, because ``F·R`` is unbounded along a periodic direction.

    Returns
    -------
    dict with keys:
        ``energy_hartree`` (float) — total energy in Hartree;
        ``energy_ev`` (float) — the same energy in eV (the method's native unit);
        ``heat_of_formation_kcal`` (float) — heat of formation, kcal/mol;
        ``electronic_ev``, ``core_ev`` (float) — electronic and core–core parts, eV;
        ``field_nuclear_ev`` (float) — the nuclear half of ``−μ·F``; the electronic half is
        already inside ``electronic_ev``. Zero without a field;
        ``charges`` (list[float]) — Mulliken net atomic charges, e;
        ``dipole_debye`` (list[float]) — dipole vector [x, y, z] in Debye;
        ``homo_ev``, ``lumo_ev`` (float or None) — frontier orbital energies, eV;
        ``homo_beta_ev``, ``lumo_beta_ev`` (float or None) — the β channel, unrestricted only;
        ``converged`` (bool) — whether the SCF converged;
        ``method`` (str) — the parameterization actually used.
    """
    n, p = _as_lists(numbers, positions)
    return _native.single_point(
        n, p, float(charge), int(multiplicity), reference, method, _field(electric_field)
    )


def gradient(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None) -> dict:
    """Energy and analytic nuclear gradient.

    Coordinates in **Ångström**; see :func:`single_point` for the keyword arguments.

    The SCF properties come back too (``charges``, ``dipole_debye``), because computing the
    gradient already converged an SCF: a caller that wants forces *and* charges at one geometry
    should not pay for two.

    Returns
    -------
    dict with keys:
        ``energy_hartree`` (float), ``energy_ev`` (float) — total energy;
        ``heat_of_formation_kcal`` (float);
        ``gradient_hartree_per_bohr`` (list[[float, float, float]]) — dE/dR in atomic units;
        ``gradient_ev_per_angstrom`` (list[[float, float, float]]) — dE/dR in eV/Å (forces = −gradient);
        ``charges`` (list[float]), ``dipole_debye`` (list[float]), ``converged`` (bool), ``method`` (str).

    With ``electric_field`` the force on atom ``a`` gains ``+Q_a F``; see :func:`single_point`.
    """
    n, p = _as_lists(numbers, positions)
    return _native.gradient(
        n, p, float(charge), int(multiplicity), reference, method, _field(electric_field)
    )


def pbc_point(
    numbers: Sequence[int],
    positions,
    cell,
    pbc,
    kpts=(1, 1, 1),
    charge: float = 0.0,
    multiplicity: int = 1,
    unrestricted: bool = False,
    smearing_ev: float = 0.0,
    realspace_cutoff: float = 40.0,
    exchange_cutoff: float = 20.0,
    method: str = "am1",
    e_tol: float = 1.0e-8,
    p_tol: float = 1.0e-7,
    max_scf: int = 300,
    mixing: float = 0.3,
    electric_field=None,
) -> dict:
    """Periodic single point: energy, analytic forces and analytic stress.

    Unlike the molecular functions in this module this one returns **eV / Å** directly. A
    periodic result is consumed by ASE, and a stress tensor converted twice is an easy way to
    be quietly wrong.

    Parameters
    ----------
    cell : array-like, shape (3, 3)
        The three lattice vectors as **rows**, in Ångström (ASE's ``atoms.cell`` convention).
    pbc : sequence of three bool
        Which lattice directions are periodic. One entry point covers a chain
        (``[True, False, False]``), a slab (``[True, True, False]``) and a crystal
        (``[True, True, True]``); the non-periodic rows still have to be given as vectors, and
        they should be long enough to separate the images.
    kpts : tuple of three int
        Monkhorst–Pack mesh. Entries along a non-periodic direction are ignored (folded to 1).
    smearing_ev : float
        Fermi–Dirac smearing width kT, eV. Zero means sharp aufbau filling, which is what a
        gapped system wants; a metal needs a finite width to converge at all.
    realspace_cutoff : float
        Largest lattice translation |T| included in the real-space sums, **Bohr**.
    electric_field : sequence of three float, optional
        Uniform external field in **atomic units** (Hartree per e·Bohr), matching the molecular
        functions in this module rather than ASE's V/Å. It must be **orthogonal to every lattice
        vector** — normal to a slab, transverse to a chain: ``F·R`` shifts by ``F·T`` under
        translation by ``T``, so the perturbation is lattice-periodic exactly when ``F·T = 0``. A
        component along a periodic direction raises, naming itself.
    exchange_cutoff : float
        Distance, in **Bohr**, over which the two-centre exchange is tapered off. See
        ``docs/pbc.md``: NDDO exchange decays only as fast as the density matrix it contracts,
        which at Γ does not decay at all, so this truncation is a documented approximation
        rather than a convergence parameter you can ignore.
    e_tol, p_tol : float
        SCF convergence thresholds on the energy change (eV) and the density RMS change. The
        defaults are chosen for dynamics, where the geometry moves every step. **Tighten them
        for a finite difference**: differentiating an energy that is only converged to 1e-8
        with a step of 1e-4 leaves the SCF's own convergence error in the answer, and it does
        not cancel between the two displaced points because they converge differently.
    max_scf : int
        Iteration limit. Not converging raises rather than returning an unconverged result.
    mixing : float
        Linear mixing fraction for the real-space density between iterations.

    Returns
    -------
    dict with keys:
        ``energy_ev`` (float) — total energy per cell, eV;
        ``free_energy_ev`` (float) — E − TS, the quantity whose derivative is the force;
        ``forces_ev_per_angstrom`` (list[[float, float, float]]);
        ``stress_voigt`` (list[float]) — the six Voigt components ``[xx, yy, zz, yz, xz, xy]``,
        in eV/Å³ for a 3D cell (ASE's convention), eV/Å² for a slab and eV/Å for a chain,
        since the periodic measure is an area or a length there;
        ``stress_matrix`` (list[list[float]]) — the same tensor in full 3×3 form;
        ``charges`` (list[float]) — Mulliken charges, e;
        ``fermi_energy_ev``, ``entropy_ev`` (float);
        ``k_points`` (int) — number of k-points actually sampled after folding;
        ``iterations`` (int), ``n_periodic`` (int);
        ``max_image_overlap`` (float) — the largest |S_μν(T)| over non-zero translations. NDDO
        *assumes* an orthonormal AO basis, so this is the size of the assumption's own error;
        above roughly 0.4 the cell is too small for the model to be meaningful.
    """
    n, p = _as_lists(numbers, positions)
    cell = np.asarray(cell, dtype=float).reshape(3, 3).tolist()
    pbc = [bool(x) for x in np.asarray(pbc).reshape(-1)]
    if len(pbc) == 1:
        pbc = pbc * 3
    kpts = tuple(int(k) for k in kpts)
    return _native.pbc_point(
        n,
        p,
        cell,
        pbc,
        kpts,
        float(charge),
        int(multiplicity),
        bool(unrestricted),
        float(smearing_ev),
        float(realspace_cutoff),
        float(exchange_cutoff),
        method,
        float(e_tol),
        float(p_tol),
        int(max_scf),
        float(mixing),
        _field(electric_field),
    )


def divide_conquer(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "am1",
    core_size: int = 12,
    buffer_radius: float = 11.0,
    smearing_ev: float = 0.05,
    e_tol: float = 1.0e-7,
    p_tol: float = 1.0e-6,
    max_scf: int = 300,
    mixing: float = 0.4,
    gap_warn_ev: float = 0.5,
    forces: bool = True,
    multipole_cutoff: float | None = None,
    electric_field=None,
    cell=None,
    pbc=None,
    realspace_cutoff: float = 40.0,
    exchange_cutoff: float = 20.0,
) -> dict:
    """Divide-and-conquer SCF for large systems. Returns eV / Å.

    The atoms are split into disjoint **core** regions, each padded with a **buffer** of nearby
    atoms, and each resulting subsystem is diagonalized on its own. Subsystem size stops growing
    with the molecule, so the diagonalization cost becomes linear in the number of atoms rather
    than cubic. The subsystems share electrons through a single chemical potential, so charge
    still flows between them.

    Parameters
    ----------
    core_size : int
        Target atoms per core region. Smaller cores mean more, smaller diagonalizations, but the
        buffer around each one is then a larger fraction of the work.
    buffer_radius : float
        In **Bohr**. The method's one physical parameter: the distance beyond which the density
        matrix is taken to vanish. Increasing it drives the answer monotonically to the full SCF,
        and a buffer covering the whole molecule reproduces it exactly. Increase it until the
        property you care about stops moving.
    smearing_ev : float
        Fermi–Dirac width kT, eV, for the common chemical potential. Zero selects sharp aufbau
        filling. A small non-zero value is the default and is not merely defensive: each level
        enters the electron count weighted by the fraction of it that lives on the core, and
        those fractions are not integers, so with sharp filling the frontier would have to be
        resolved by sort order — a discontinuous function of geometry.
    gap_warn_ev : float
        A ``small_gap_warning`` is returned when the smallest subsystem gap falls below this.
        Divide-and-conquer assumes the density matrix decays with distance, which is a property
        of gapped systems; in a metal it does not, and neither the accuracy nor the linear
        scaling survives.
    forces : bool
        Whether to compute the gradient as well.
    multipole_cutoff : float or None
        Separation in **Bohr** beyond which a pair's electrostatics is treated as an atomic
        monopole instead of the full multipole block. ``None`` (default) keeps every pair exact,
        which is what every validated number in this crate was produced against; setting it is an
        explicit accuracy-for-speed trade whose error falls as ``(d/R)²``. See ``tests/farfield.rs``,
        which measures both sides of it.

    Returns
    -------
    dict with keys:
        ``energy_ev``, ``energy_hartree``, ``free_energy_ev``, ``heat_of_formation_kcal``;
        ``forces_ev_per_angstrom`` (when ``forces``); ``charges`` (Mulliken, e);
        ``fermi_energy_ev``, ``fermi_energies_ev`` (one per spin channel), ``entropy_ev``;
        ``homo_lumo_gap_ev``, ``small_gap_warning`` (str or None);
        ``subsystems``, ``largest_subsystem_aos``, ``diagonalization_work`` (``Σ n_α³``),
        ``coulomb_work``, ``exchange_work``, ``retained_density_blocks``,
        ``diis_pattern_elements`` and ``dense_triangle_elements`` (the DIIS history's memory, and
        what it would have been dense — linear against quadratic), ``iterations``,
        ``unrestricted``, ``method``.

    Notes
    -----
    **What is linear and what is not.** The diagonalization, the two-centre exchange and the DIIS
    history are linear in the number of atoms; the two-centre **Coulomb** sum is not, and stays
    quadratic. NDDO's two-centre integrals decay as ``1/R``, so they cannot be dropped by distance
    without changing the answer — ``multipole_cutoff`` simplifies their *shape* rather than
    dropping them, which lowers the prefactor without touching the exponent. The counters above
    are returned so this can be checked rather than believed. See ``docs/divide-conquer.md``.

    The gradient is the Hellmann–Feynman gradient at the divide-and-conquer density. That is the
    exact derivative for a variational density; the divide-and-conquer density is not
    variational, so it is exact only in the limit of a buffer covering the system. The residual
    shrinks with ``buffer_radius`` alongside the energy error.

    **Periodic cells.** Pass ``cell`` (3×3, Å) and ``pbc`` together to run under periodic boundary
    conditions: the subsystem buffers are then built from the image-aware pair list, so an atom's
    buffer can wrap through the cell boundary. ``exchange_cutoff`` matters there and not for a
    molecule — at Γ the two-centre exchange integral decays only as ``1/R`` while the density
    matrix does not decay at all, so the image sum needs the taper. ``tests/dc_periodic.rs``
    checks convergence to the full periodic SCF as ``buffer_radius`` grows.
    """
    n, p = _as_lists(numbers, positions)
    if (cell is None) != (pbc is None):
        raise ValueError(
            "cell and pbc must be given together: a lattice without a periodicity flag, or the "
            "reverse, is ambiguous"
        )
    if cell is not None:
        cell, pbc = _cell_and_pbc(cell, pbc)
    return _native.divide_conquer(
        n,
        p,
        float(charge),
        int(multiplicity),
        reference,
        method,
        int(core_size),
        float(buffer_radius),
        float(smearing_ev),
        float(e_tol),
        float(p_tol),
        int(max_scf),
        float(mixing),
        float(gap_warn_ev),
        bool(forces),
        None if multipole_cutoff is None else float(multipole_cutoff),
        _field(electric_field),
        cell,
        pbc,
        float(realspace_cutoff),
        float(exchange_cutoff),
    )


def phonons(
    numbers: Sequence[int],
    positions,
    cell,
    pbc,
    supercell=(2, 2, 2),
    q_points=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "am1",
    realspace_cutoff: float = 40.0,
    exchange_cutoff: float = 20.0,
    e_tol: float = 1.0e-10,
    p_tol: float = 1.0e-9,
    max_scf: int = 500,
    enforce_acoustic_sum_rule: bool = True,
) -> dict:
    """Phonon frequencies of a periodic system, via supercell force constants.

    The Γ-point Hessian gives ``Σ_T Φ(0,T)`` — the force constants *summed* over lattice
    translations — which is what ``q = 0`` needs and useless anywhere else, since
    ``D(q) = Σ_T Φ(0,T) e^{iq·T}`` needs them resolved. So ``Φ(T)`` is read off the Γ Hessian of
    a supercell, where the force constant between an atom in the home cell and one in cell ``T``
    is simply a matrix element.

    Parameters
    ----------
    supercell : tuple of three int
        How many primitive cells along each axis. **This is the convergence knob**, and it
        controls two things at once: how far ``Φ(T)`` is resolved before truncation, and — since
        Γ on an ``n``-fold supercell is the primitive cell at ``n`` k-points — the k-sampling of
        the electronic structure underneath. Increase it until the frequencies you care about
        stop moving. Cost grows as the supercell's atom count cubed.
    q_points : list of three-component fractional coordinates, or None
        Where to evaluate. ``None`` gives Γ. The returned ``commensurate_q`` lists the points
        this supercell represents **exactly**; anywhere else is an interpolation of a truncated
        ``Φ(T)``.
    enforce_acoustic_sum_rule : bool
        Force the three acoustic modes to exactly zero at Γ by correcting the on-site block.
        This is a correction, not a refinement — it moves the truncation error into the on-site
        term rather than removing it. ``acoustic_sum_rule_error_before`` is the honest measure of
        how much was wrong.

    Returns
    -------
    dict with keys:
        ``q_points`` — the fractional coordinates evaluated;
        ``frequencies_cm`` — one list of ``3·nat`` frequencies per q point, ascending, cm⁻¹;
        negative values denote imaginary modes (the structure is not a minimum along them);
        ``commensurate_q`` — the q this supercell is exact at;
        ``acoustic_sum_rule_error_before``, ``acoustic_sum_rule_error`` — eV/Bohr²;
        ``supercell``, ``method``.

    Notes
    -----
    **This function carries no LO–TO splitting.** The non-analytic correction
    ``D_NA ∝ (q·Z*_a)(q·Z*_b)/(q·ε_∞·q)`` cannot come from a truncated real-space ``Φ(T)``, so
    longitudinal and transverse optical branches come out degenerate as ``q → 0``, and in a polar
    crystal they are not. Acoustic branches and non-polar systems are unaffected.

    It is available: :func:`lo_to_frequencies` adds the term from :func:`born_charges` and
    :func:`dielectric`, both of which this version *does* compute. (An earlier version of this
    note said it did not — that was already wrong in 0.2.0.) See ``docs/pbc.md``.
    """
    n, p = _as_lists(numbers, positions)
    cell = np.asarray(cell, dtype=float).reshape(3, 3).tolist()
    pbc = [bool(x) for x in np.asarray(pbc).reshape(-1)]
    if len(pbc) == 1:
        pbc = pbc * 3
    supercell = tuple(int(s) for s in supercell)
    if q_points is not None:
        q_points = np.asarray(q_points, dtype=float).reshape(-1, 3).tolist()
    return _native.phonons(
        n,
        p,
        cell,
        pbc,
        supercell,
        q_points,
        float(charge),
        int(multiplicity),
        method,
        float(realspace_cutoff),
        float(exchange_cutoff),
        float(e_tol),
        float(p_tol),
        int(max_scf),
        bool(enforce_acoustic_sum_rule),
    )


def optimize(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None) -> dict:
    """L-BFGS geometry optimization on the analytic gradient.

    Coordinates in **Ångström**; see :func:`single_point` for the keyword arguments.

    Returns
    -------
    dict with keys:
        ``positions_angstrom`` (list[[float, float, float]]) — optimized geometry, Å;
        ``energy_hartree`` (float), ``heat_of_formation_kcal`` (float) — at the optimized geometry;
        ``converged`` (bool), ``iterations`` (int).
    """
    n, p = _as_lists(numbers, positions)
    return _native.optimize(
        n, p, float(charge), int(multiplicity), reference, method, _field(electric_field)
    )


def orbitals(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None) -> dict:
    """Orbital energies, coefficients and occupations — the wavefunction as numbers.

    Coordinates in **Ångström**; see :func:`single_point` for the keyword arguments.

    Returns
    -------
    dict with keys:
        ``energies_hartree``, ``energies_ev`` (list[float]) — orbital energies, ascending;
        ``coefficients`` (list[list[float]]) — rows are atomic orbitals, **columns are molecular
        orbitals**, matching ``energies``;
        ``n_occupied`` (int), ``homo_ev``, ``lumo_ev`` (float or None);
        ``ao_labels`` (list[(int, str)]) — the atom index and shell label (``"s"``, ``"px"``,
        ``"py"``, ``"pz"``) of each row, so a coefficient can be identified without rebuilding
        the basis;
        ``unrestricted`` (bool), and for an unrestricted run ``beta_energies_hartree``,
        ``beta_energies_ev``, ``beta_coefficients``, ``beta_n_occupied``, ``homo_beta_ev``,
        ``lumo_beta_ev``.

    Notes
    -----
    NDDO **assumes** an orthonormal AO basis, so these coefficients live in an implicitly
    orthogonalized basis rather than in the raw Slater functions. See :func:`molden`.
    """
    n, p = _as_lists(numbers, positions)
    return _native.orbitals(
        n, p, float(charge), int(multiplicity), reference, method, _field(electric_field)
    )


def molden(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None) -> str:
    """The wavefunction as a **Molden**-format string.

    Coordinates in **Ångström**; see :func:`single_point` for the keyword arguments. Write the
    returned text to a file and open it in a viewer::

        text = am1_rs.native.molden(numbers, positions)
        with open("water.molden", "w", encoding="utf-8") as fh:
            fh.write(text)

    The explicit ``encoding`` is not decoration: ``open`` defaults to the locale's codec, which
    is ``cp932`` on a Japanese Windows and ASCII under a ``C`` locale, so a file written without
    it is not portable between machines.

    Notes
    -----
    The AM1 valence basis is genuinely Slater-type, so the ``[STO]`` section describes it exactly
    and no Gaussian expansion is invented. But NDDO **assumes** an orthonormal AO basis — its
    working equations have no overlap matrix — so the ``[MO]`` coefficients are in an implicitly
    orthogonalized basis while the ``[STO]`` functions are the raw, non-orthogonal ones. Orbital
    shapes, nodes and symmetry are faithful; amplitudes in the bonding region are approximate.
    The same caveat is written into the file itself.
    """
    n, p = _as_lists(numbers, positions)
    return _native.molden(
        n, p, float(charge), int(multiplicity), reference, method, _field(electric_field)
    )


def ir_spectrum(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None) -> dict:
    """Infrared spectrum: the atomic polar tensor and the mode-resolved intensities.

    **Expensive** — this solves the CPHF equations, i.e. it costs an analytic Hessian. It is a
    separate call rather than part of a single point for exactly that reason. Evaluate at a
    **stationary point**: the intensities are defined against harmonic normal modes.

    Returns
    -------
    dict with keys:
        ``dipole_derivatives`` (list[list[float]]) — the raw **atomic polar tensor**
        ``∂μ_α/∂R_{a,β}``, 3 rows by 3N columns, in units of ``e``. Column ``3a + β`` is atom
        ``a``, axis ``β``. This is the molecular counterpart of the Born effective charges;
        ``frequencies_cm`` (list[float]) — harmonic frequencies, cm⁻¹, ascending;
        ``intensities_km_per_mol`` (list[float]) — one per mode;
        ``mode_dipole_derivatives`` (list[list[float]]) — the dense per-mode tensor
        ``∂μ_α/∂Q_k``, 3 by 3N, in D·Å⁻¹·amu^(−1/2). The intensity discards the transition
        dipole's *direction*; this keeps it, which is what a polarized measurement sees;
        ``modes`` (list[list[float]]) — mass-weighted normal modes, columns are modes;
        ``translation_rotation_overlap`` (list[float]) — each mode's overlap with the rigid-body
        subspace, 0…1. Use it to tell vibrations from translations and rotations: a linear
        molecule has five rigid-body modes, not six, and this reports that rather than assuming
        ``3N − 6``;
        ``vibrational_modes`` (list[int]) — indices whose overlap is below 0.5.

    Notes
    -----
    The tensor satisfies ``Σ_a ∂μ_α/∂R_{a,β} = q δ_αβ`` — translating the molecule moves its net
    charge and nothing else. ``tests/ir.rs`` asserts that, and checks the tensor two further
    independent ways.
    """
    n, p = _as_lists(numbers, positions)
    return _native.ir_spectrum(
        n, p, float(charge), int(multiplicity), reference, method, _field(electric_field)
    )


def vibrations(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None, hessian: bool = True, frequencies: bool = True, ir: bool = True, orbital_response: bool = False, response_density: bool = False) -> dict:
    """The whole vibrational group from **one** SCF and one CPHF solve.

    :func:`hessian`, :func:`frequencies`, :func:`ir_spectrum`, :func:`dipole_derivatives` and
    :func:`orbital_response` each run the full analytic-Hessian solve and then keep a different
    contraction of it. Asking for several — an infrared spectrum *and* the Hessian it came from,
    which is the ordinary case — used to cost one CPHF per question. They are all contractions of
    the same response, so this returns them together for the price of one.

    Each section is opt-in, because two of them are large: ``orbital_response`` is
    ``O(ndof · n_occ · n_vir)`` and ``response_density`` is ``O(ndof · nao²)``, the biggest array
    in the calculation.

    Parameters
    ----------
    hessian, frequencies, ir, orbital_response, response_density : bool
        Which sections to return. ``ir`` implies ``frequencies``: the spectrum carries the normal
        modes, so asking for both costs one vibrational analysis, not two.

    Returns
    -------
    dict
        The union of the keys the individual functions return, restricted to the sections asked
        for: ``hessian_hartree_per_bohr2`` / ``hessian_ev_per_angstrom2`` /
        ``hessian_ev_per_bohr2``; ``frequencies_cm`` / ``eigenvalues`` / ``modes`` /
        ``cartesian_displacements`` / ``translation_rotation_overlap``; ``dipole_derivatives`` /
        ``intensities_km_per_mol`` / ``mode_dipole_derivatives`` / ``vibrational_modes``;
        ``u_ov`` / ``g_ov`` / ``n_occupied`` / ``n_virtual`` (and the ``beta_*`` counterparts for
        an unrestricted run); ``response_density``. Always ``ndof``, ``method`` and
        ``cphf_iterations``.

    Notes
    -----
    The values are the same ones the individual functions return — the same solve, contracted the
    same ways — so this is a cost change and not a numerical one.
    """
    n, p = _as_lists(numbers, positions)
    return _native.vibrations(
        n,
        p,
        float(charge),
        int(multiplicity),
        reference,
        method,
        _field(electric_field),
        bool(hessian),
        bool(frequencies),
        bool(ir),
        bool(orbital_response),
        bool(response_density),
    )


def dipole_derivatives(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None) -> dict:
    """The atomic polar tensor ``∂μ_α/∂R_{a,β}`` alone, in units of ``e``.

    Same cost as :func:`ir_spectrum` — it is where most of that cost goes. Use this when the
    normal modes are not wanted.

    Returns
    -------
    dict with keys ``dipole_derivatives`` (3 × 3N) and ``ndof``.
    """
    n, p = _as_lists(numbers, positions)
    return _native.dipole_derivatives(
        n, p, float(charge), int(multiplicity), reference, method, _field(electric_field)
    )


def orbital_response(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None, response_density: bool = False) -> dict:
    """First-order orbital response ``U^j_{ai}`` — the CPHF coefficients.

    **Expensive**, and a by-product of the analytic Hessian: the ``U`` returned is the one the
    Hessian already solves for, so nothing is recomputed — but the Hessian does have to run. The
    Hessian itself comes back in the result, so asking for both costs one calculation.

    Parameters
    ----------
    response_density : bool
        Also return the ``3N`` AO-basis first-order densities ``∂P/∂R_j``. Off by default: that
        is the largest array in the calculation, and it is built on demand from ``U``.

    Returns
    -------
    dict with keys:
        ``u_ov`` (list of ``n_virtual × n_occupied`` blocks, one per Cartesian degree of freedom);
        ``g_ov`` — the skeleton derivative Fock in the same blocks;
        ``n_occupied``, ``n_virtual``, ``ndof`` (int);
        ``cphf_iterations`` (list[int]) — what each solve actually did;
        ``hessian_ev_per_bohr2`` (list[list[float]]);
        ``response_density`` when asked for; and the ``beta_*`` equivalents for an unrestricted
        run.
    """
    n, p = _as_lists(numbers, positions)
    return _native.orbital_response(
        n,
        p,
        float(charge),
        int(multiplicity),
        reference,
        method,
        _field(electric_field),
        bool(response_density),
    )


def hessian(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None) -> dict:
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
    return _native.hessian(
        n, p, float(charge), int(multiplicity), reference, method, _field(electric_field)
    )


def am1_bcc(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1) -> dict:
    """AM1-BCC partial charges for AMBER (Jakalian *et al.* 2000/2002).

    Runs the AM1 SCF for Mulliken charges, perceives the molecular graph, assigns the
    antechamber BCC atom types, and applies the exact ``BCCPARM.DAT`` bond charge corrections.
    Coordinates in **Ångström**.

    The method is fixed to AM1 by the parameterization itself: the BCC increments were fitted
    to AM1 Mulliken charges, so applying them to an RM1 density would not be AM1-BCC.

    Returns
    -------
    dict with keys:
        ``charges`` (list[float]) — AM1-BCC net atomic charges, e (Σ = ``charge``);
        ``mulliken`` (list[float]) — the underlying AM1 Mulliken charges before corrections;
        ``atom_types`` (list[str]) — antechamber BCC atom-type codes (``"11"`` … ``"91"``;
        ``"0"`` means no type was assignable);
        ``warnings`` (list[str]) — anything the perception or the parameter lookup could not do
        confidently. **Check this.** An empty list means the molecule was fully covered by the
        typing rules; a non-empty one means some bond was left at its raw AM1 Mulliken charge,
        which is a difference of tenths of an electron rather than a rounding.
    """
    n, p = _as_lists(numbers, positions)
    return _native.am1_bcc(n, p, float(charge), int(multiplicity))


def frequencies(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1, reference: str = "auto", method: str = "am1", electric_field=None) -> dict:
    """Harmonic vibrational frequencies from the analytic (CPHF) Hessian.

    Evaluate at a **stationary point** (optimize first) for physically meaningful modes.
    Coordinates in **Ångström**; see :func:`single_point` for ``charge``/``multiplicity``/``reference``.

    Returns
    -------
    dict with keys:
        ``frequencies_cm`` (list[float]) — harmonic frequencies in cm⁻¹, ascending; negative
        values denote imaginary modes (saddle point or non-stationary geometry);
        ``eigenvalues`` (list[float]) — mass-weighted Hessian eigenvalues, eV/(Å²·amu);
        ``modes`` (list[list[float]]) — mass-weighted eigenvectors, **columns** are modes and are
        orthonormal;
        ``cartesian_displacements`` (list[list[float]]) — ``M^(−1/2)`` times those, deliberately
        not renormalized;
        ``translation_rotation_overlap`` (list[float]) — how much of each mode lies in the
        rigid-body subspace, 0…1. A linear molecule has five rigid-body modes and a bent one six;
        this reports which is which from the eigenvectors rather than from a frequency cutoff.
    """
    n, p = _as_lists(numbers, positions)
    return _native.frequencies(
        n, p, float(charge), int(multiplicity), reference, method, _field(electric_field)
    )


def _cell_and_pbc(cell, pbc):
    """Normalize a cell and pbc flags the way every periodic entry point wants them."""
    cell = np.asarray(cell, dtype=float).reshape(3, 3).tolist()
    pbc = [bool(x) for x in np.asarray(pbc).reshape(-1)]
    if len(pbc) == 1:
        pbc = pbc * 3
    return cell, pbc


def pbc_hessian(numbers: Sequence[int], positions, cell, pbc, kpts=(2, 2, 2), charge: float = 0.0, multiplicity: int = 1, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500) -> dict:
    """Analytic force constants at ``q = 0`` with **k-point sampling** — the periodic Hessian.

    Unlike the Γ-only path this does not lean on the exchange taper: sampling ``k`` makes the
    real-space density matrix decay on its own, so the second derivatives are the ones the
    Hamiltonian implies rather than the ones the taper leaves behind. See ``docs/pbc.md``.

    Returns
    -------
    dict with ``hessian_hartree_per_bohr2``, ``hessian_ev_per_angstrom2`` and ``ndof``.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    return _native.pbc_hessian(
        n, p, cell, pbc, tuple(int(k) for k in kpts), float(charge), int(multiplicity), method,
        float(realspace_cutoff), float(exchange_cutoff), float(e_tol), float(p_tol), int(max_scf),
    )


def born_charges(numbers: Sequence[int], positions, cell, pbc, kpts=(2, 2, 2), charge: float = 0.0, multiplicity: int = 1, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500) -> dict:
    """Born effective charges ``Z*_{a,αβ}``, in units of ``e``.

    ``Z*`` is how much dipole appears when an atom moves — the crystal's counterpart of
    :func:`dipole_derivatives`. It is well defined under periodic boundary conditions even though
    the absolute dipole is not, because charge is conserved and the origin dependence cancels
    term by term.

    Returns
    -------
    dict with keys:
        ``born_charges`` (list of 3×3 tensors, one per atom);
        ``acoustic_sum_rule_error`` (3×3) — ``Σ_a Z*_a``, which must vanish: translating the
        crystal produces no dipole. Reported rather than left to be recomputed.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    return _native.born_charges(
        n, p, cell, pbc, tuple(int(k) for k in kpts), float(charge), int(multiplicity), method,
        float(realspace_cutoff), float(exchange_cutoff), float(e_tol), float(p_tol), int(max_scf),
    )


def polarizability(numbers: Sequence[int], positions, cell, pbc, kpts=(2, 2, 2), charge: float = 0.0, multiplicity: int = 1, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500) -> dict:
    """Clamped-ion polarizability ``alpha`` (Bohr^3), in **any** periodic dimensionality.

    The same ``alpha`` :func:`dielectric` returns, without the ``eps_inf`` conversion -- which is
    why this one works for a chain and a slab and that one does not: ``eps_inf = 1 + 4*pi*alpha/V``
    needs ``V`` to be a volume.

    Returns
    -------
    dict with ``polarizability_bohr3`` (3x3), ``measure`` (the cell's volume, area or length) and
    ``n_periodic``. Divide by ``measure`` yourself, and mind what the result is: dimensionless in
    3D, a **length** for a slab, an **area** for a chain. Turning a slab's into a dielectric
    constant needs a thickness, which is a choice about the material and not something a supercell
    fixes.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    return _native.polarizability(
        n, p, cell, pbc, tuple(int(k) for k in kpts), float(charge), int(multiplicity), method,
        float(realspace_cutoff), float(exchange_cutoff), float(e_tol), float(p_tol), int(max_scf),
    )

def dielectric_function(numbers: Sequence[int], positions, cell, pbc, q, chain_radius=None, kpts=(2, 2, 2), charge: float = 0.0, multiplicity: int = 1, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500) -> float:
    """Macroscopic longitudinal dielectric function ``eps(q)``, in any dimensionality.

    ``q`` is a Cartesian wavevector in **inverse Bohr** and must lie in the periodic subspace.

    Three dimensions gives a constant -- the familiar ``eps_infinity``, which ``dielectric``
    returns directly. A slab and a chain do **not**: ``eps(q) -> 1`` at long wavelength, because a
    sheet or a wire does not screen a field whose wavelength exceeds its own extent. That is the
    same fact as a slab having no LO-TO splitting at Gamma, and it is why
    ``eps = 1 + 4*pi*alpha/Omega`` cannot be evaluated there rather than a gap in this code.

    Parameters
    ----------
    chain_radius : float, optional
        Transverse radius in **Bohr**, required for a chain and ignored otherwise. The 1D Coulomb
        kernel is ``2 K0(|q| rho)``, a logarithm at small ``q``, and has no value without one.
        There is no natural choice, so it is required rather than guessed.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 3:
        raise ValueError(f"q must have three components, got {q.size}")
    return _native.dielectric_function(
        n, p, cell, pbc, q.tolist(),
        None if chain_radius is None else float(chain_radius),
        tuple(int(k) for k in kpts), float(charge), int(multiplicity), method,
        float(realspace_cutoff), float(exchange_cutoff), float(e_tol), float(p_tol), int(max_scf),
    )

def polarization(numbers: Sequence[int], positions, cell, pbc, kpts=(2, 2, 2), strings: int = 8, charge: float = 0.0, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500) -> dict:
    """Berry-phase polarization (King-Smith--Vanderbilt), ``e/Bohr²``.

    **Three-dimensional, restricted cells only.** ``strings`` is the number of k points per
    Berry-phase string, resampled independently of ``kpts``: the two transverse directions use
    ``kpts``, the string's own direction uses ``strings``.

    Returns
    -------
    dict with ``polarization``, ``electronic``, ``ionic`` (each a 3-vector), ``phase_turns``
    (three Berry phases in turns), ``quantum`` (the three ``e a_α/Ω`` vectors) and
    ``string_length``.

    Notes
    -----
    Defined **modulo the quantum**: only differences between two states on a common branch are
    physical, which is what ``BerryPolarization::difference`` does on the Rust side.

    The phase in this atom-centred minimal basis tracks the charge **centres** -- the
    ``e^{-i b·τ}`` factor sits on each orbital's own atom -- and carries no on-site ``s``--``p``
    moment. Against the CPHF Born charges that is a 0.207 e gap on HF and 7.5e-13 e on a
    hydrogen-only cell, where the moment cannot exist. See ``docs/pbc.md``.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    return _native.polarization(
        n, p, cell, pbc, tuple(int(k) for k in kpts), int(strings), float(charge), method,
        float(realspace_cutoff), float(exchange_cutoff), float(e_tol), float(p_tol), int(max_scf),
    )


def finite_field(numbers: Sequence[int], positions, cell, pbc, field, kpts=(4, 4, 4), charge: float = 0.0, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500, max_outer: int = 60, outer_tol: float = 1.0e-8, outer_mixing: float = 0.5) -> dict:
    """A finite electric field **along** a periodic direction, by the Berry-phase enthalpy.

    ``field`` is in **atomic units** (Hartree per e·Bohr), like every other field in this module.

    ``F·R`` is unbounded along a periodic direction, so this minimizes the electric enthalpy
    ``E - Ω E·P`` with ``P`` the Berry phase instead. The field term couples neighbouring k points,
    so the k points are solved together and an outer loop refreshes the operator.

    For a field **orthogonal** to every lattice vector -- normal to a slab, transverse to a chain
    -- use ``pbc_point(..., electric_field=...)``: that is an ordinary ``F·R`` calculation and
    needs none of this.

    Three-dimensional, restricted, no smearing, and at least three k points along any direction the
    field has a component in.

    Returns
    -------
    dict with ``energy_ev``, ``enthalpy_ev`` (the quantity minimized), ``polarization``,
    ``electronic``, ``ionic``, ``phase_turns``, ``charges``, ``outer_iterations`` and
    ``scf_iterations``.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    return _native.finite_field(
        n, p, cell, pbc, _field(field), tuple(int(k) for k in kpts), float(charge), method,
        float(realspace_cutoff), float(exchange_cutoff), float(e_tol), float(p_tol), int(max_scf),
        int(max_outer), float(outer_tol), float(outer_mixing),
    )


def dielectric(numbers: Sequence[int], positions, cell, pbc, kpts=(2, 2, 2), charge: float = 0.0, multiplicity: int = 1, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500) -> dict:
    """Clamped-ion polarizability ``α`` (Bohr³) and electronic dielectric tensor ``ε_∞``.

    **Three-dimensional cells only.** ``ε_∞ = 1 + 4πα/Ω`` needs ``Ω`` to be a volume; a chain has
    only a length and a slab only an area, and those are refused rather than silently divided by.
    Since 0.2.2 :func:`dielectric_with_extent` handles those, once the caller says how thick the
    material is — and it carries the depolarization factor the low-dimensional case needs.

    Returns
    -------
    dict with ``polarizability_bohr3`` and ``epsilon_infinity``, each a 3×3.

    Notes
    -----
    This is a clamped-ion tight-binding response to the model's own dipole operator, **not** a
    Berry-phase polarization. The position operator it uses is not a well-defined periodic
    operator; what makes the result meaningful anyway is that the response conserves charge, so
    the origin dependence cancels — measured at 1e-14 relative, not argued.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    return _native.dielectric(
        n, p, cell, pbc, tuple(int(k) for k in kpts), float(charge), int(multiplicity), method,
        float(realspace_cutoff), float(exchange_cutoff), float(e_tol), float(p_tol), int(max_scf),
    )


def dielectric_with_extent(numbers: Sequence[int], positions, cell, pbc, slab_thickness=None, wire_cross_section=None, kpts=(2, 2, 2), charge: float = 0.0, multiplicity: int = 1, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500) -> dict:
    """``eps_infinity`` for a **slab or a chain**, given the extent you assign the material.

    Parameters
    ----------
    slab_thickness : float, optional
        Bohr, for a cell periodic in two directions. The assigned volume per cell is ``A * d``.
    wire_cross_section : float, optional
        Bohr², for a cell periodic in one. The assigned volume per cell is ``L * S``, and the
        section is taken to be **circular** — which is what fixes the transverse depolarization
        factor at ``1/2``.

    Exactly one is required and neither has a default. A supercell says where the atoms are, not
    where the material stops, and every choice here changes ``eps``.

    Notes
    -----
    The conversion is not a division. ``alpha`` is the response to the **external** field — the
    depolarizing field the induced charges make is already inside it — so::

        eps = 1 + 4*pi*chi / (1 - 4*pi*N*chi),    chi = alpha / (measure * extent)

    with ``N`` the depolarization factor of the assumed body: 0 in a slab's plane and along a
    wire's axis, 1 along a slab normal, 1/2 transverse to a wire. In three dimensions tin-foil
    boundary conditions remove the macroscopic depolarizing field, ``N`` is 0 everywhere, and this
    reduces to :func:`dielectric`'s ``1 + 4*pi*alpha/Omega``.

    Returns
    -------
    dict
        ``polarizability_bohr3`` and ``epsilon_infinity`` (3×3 each), the ``extent``, ``measure``,
        ``n_periodic`` and ``axis`` that produced it, and the two combinations that do **not**
        depend on the choice: ``sheet_susceptibility`` = ``(eps_par - 1) * extent`` and
        ``inverse_sheet_response`` = ``(1 - 1/eps_perp) * extent``. For a slab
        ``rytova_keldysh_length`` is half the first, which is the layer's intrinsic screening
        length. ``axis_mixing`` reports how much of ``alpha`` couples the distinguished axis to its
        complement — the part the split drops, zero whenever that axis is principal.

        Those two are **scalars**, and ``alpha_par`` in them is the *mean* over the
        two-dimensional half — the plane for a slab, the transverse pair for a wire. The identities
        hold per direction against the returned tensor and coincide with these scalars only when
        the response is isotropic there, which a real slab's rarely is. Take the tensor when the
        direction matters and these when quoting one number.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    return _native.dielectric_with_extent(
        n, p, cell, pbc,
        None if slab_thickness is None else float(slab_thickness),
        None if wire_cross_section is None else float(wire_cross_section),
        tuple(int(k) for k in kpts), float(charge), int(multiplicity), method,
        float(realspace_cutoff), float(exchange_cutoff), float(e_tol), float(p_tol), int(max_scf),
    )


def dfpt(numbers: Sequence[int], positions, cell, pbc, q_points, kpts=(2, 2, 2), charge: float = 0.0, multiplicity: int = 1, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500, long_range: str = "auto", cpscf_tol: float = 1.0e-10, cpscf_max_iter: int = 200, cpscf_mixing: float = 0.7) -> dict:
    """Phonons at **arbitrary** ``q`` by density-functional perturbation theory.

    No supercell. The response is solved on the primitive cell directly, with the perturbation
    coupling ``k`` to ``k + q`` — so a phonon at any ``q`` costs a primitive cell rather than a
    supercell large enough to represent that ``q``.

    Parameters
    ----------
    q_points : array-like, shape (M, 3)
        **Fractional** coordinates of the primitive reciprocal lattice. A component along a
        non-periodic axis is an error: no lattice translation carries that phase.
    kpts : tuple of three int
        Brillouin-zone sampling for both the response **and** the ground state it is built on.
        The two are deliberately the same set: the coupled-perturbed equations assume the
        zeroth-order state satisfies the SCF condition, so a response sampled more finely than
        its own density would be the response of a different functional.
    long_range : str
        ``"auto"`` (default) includes the long-range monopole term for a 3D cell and leaves it
        out for a chain or a slab, where it is not implemented; ``"require"`` makes its absence
        an error; ``"off"`` excludes it everywhere.
    cpscf_tol, cpscf_max_iter, cpscf_mixing : float / int / float
        The coupled-perturbed self-consistent solve. It is a **linearly mixed fixed point**, not
        the conjugate-gradient solver the molecular CPHF uses, so on a dense polar 3D cell it can
        need more than the default 200 iterations to reach 1e-10 — a converged-but-slow case
        looks like ``worst residual=5e-9`` in the error. Loosen ``cpscf_tol`` to 1e-8, or raise
        ``cpscf_max_iter``, rather than reading the failure as a defect in the geometry.

    Returns
    -------
    dict with ``q_points``, ``frequencies_cm`` (one list of ``3·nat`` values per q) and ``method``.

    Notes
    -----
    On a 3D cell this is the **full** ``D(q)``, long-range monopole channel included, so its
    ``q → 0`` limit is direction dependent — which is the physics. Do **not** also apply
    :func:`lo_to_frequencies`, whose job is to give that same physics to the *supercell* route
    (:func:`phonons`), where a truncated ``Φ(T)`` structurally cannot carry it. Use one or the
    other. See ``docs/pbc.md``.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    q_points = np.asarray(q_points, dtype=float).reshape(-1, 3).tolist()
    return _native.dfpt(
        n, p, cell, pbc, q_points, tuple(int(k) for k in kpts), float(charge),
        int(multiplicity), method, float(realspace_cutoff), float(exchange_cutoff),
        float(e_tol), float(p_tol), int(max_scf), long_range,
        float(cpscf_tol), int(cpscf_max_iter), float(cpscf_mixing),
    )


def lo_to_frequencies(numbers: Sequence[int], positions, cell, pbc, supercell=(2, 2, 2), direction=(1.0, 0.0, 0.0), q_points=None, kpts=(2, 2, 2), charge: float = 0.0, multiplicity: int = 1, method: str = "am1", realspace_cutoff: float = 40.0, exchange_cutoff: float = 20.0, e_tol: float = 1.0e-11, p_tol: float = 1.0e-10, max_scf: int = 500, enforce_acoustic_sum_rule: bool = True) -> dict:
    """Supercell phonons with the **LO–TO splitting** restored, from ``Z*`` and ``ε_∞``.

    **Three-dimensional cells only**: the non-analytic term ``4π(q·Z*)²/(Ω q·ε_∞·q)`` is the 3D
    form and ``Ω`` has to be a volume. A chain or a slab is an error, not a number in the wrong
    units.

    :func:`phonons` alone gives the transverse branches at ``Γ`` and misses the longitudinal
    shift, because a truncated real-space ``Φ(T)`` cannot carry the dipole–dipole tail. This adds
    that piece analytically.

    Parameters
    ----------
    direction : tuple of three float
        The unit vector along which the ``q → 0`` limit is taken. It is required because the
        limit *is* direction dependent — that is what LO–TO splitting means.

    Returns
    -------
    dict with ``frequencies_cm`` (split), ``frequencies_cm_no_lo_to`` (what :func:`phonons`
    gives), ``born_charges``, ``dielectric``, ``direction``, ``supercell`` and ``method``, so the
    size of the shift is visible rather than asserted.

    Notes
    -----
    Do **not** combine this with :func:`dfpt`, which already carries the long-range monopole
    channel inside ``D(q)``; applying both counts it twice.
    """
    n, p = _as_lists(numbers, positions)
    cell, pbc = _cell_and_pbc(cell, pbc)
    if q_points is not None:
        q_points = np.asarray(q_points, dtype=float).reshape(-1, 3).tolist()
    return _native.lo_to_frequencies(
        n, p, cell, pbc, tuple(int(s) for s in supercell),
        tuple(float(d) for d in direction), q_points, tuple(int(k) for k in kpts),
        float(charge), int(multiplicity), method, float(realspace_cutoff),
        float(exchange_cutoff), float(e_tol), float(p_tol), int(max_scf),
        bool(enforce_acoustic_sum_rule),
    )
