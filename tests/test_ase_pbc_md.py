# SPDX-License-Identifier: GPL-3.0-or-later
"""Periodic boundary conditions, exercised through ASE by running real molecular dynamics.

A finite-difference agreement test says the analytic gradient matches the analytic energy.
It does not say the periodic code is *usable* — that the ASE calculator reads ``atoms.pbc``
and ``atoms.cell``, that the stress arrives in the layout and units a barostat expects, that
the SCF converges from geometry to geometry as atoms move, or that the cell degrees of freedom
work at all. Running an ensemble is what tests those, because every one of them is load-bearing
for a hundred consecutive steps and a single wrong sign stops the run or blows it up.

The three ensembles here are chosen for what each can prove:

* **NVE** is the sharpest test of the *forces*. Energy is conserved only if the force really is
  the negative gradient of the reported energy. A dropped image pair, a missing derivative of
  the exchange taper, a sign error in the Bloch phase — none of these break a single-point
  energy, and all of them make the total energy walk away. Run for 1D, 2D and 3D.
* **NPT** additionally exercises the **stress** and the cell degrees of freedom, which nothing
  else here touches. It is the reason this file exists.
* **NVT** on a chain and a slab checks that a partially periodic cell survives dynamics and
  that no stress leaks into a direction that has no periodicity to carry it.

Everything is small: these are correctness tests, not production runs. Timesteps are chosen for
unconstrained O–H stretches (period ≈ 9 fs), where 1 fs is already too coarse to integrate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms, units  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402  (used by the 1D/2D NVT test)
from ase.md.npt import NPT  # noqa: E402
from ase.md.nptberendsen import NPTBerendsen  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402

try:
    from am1_rs.ase import AM1
except ImportError:  # pragma: no cover - the extension is not built
    pytest.skip("am1_rs native extension not built", allow_module_level=True)


# --------------------------------------------------------------------------------- systems
#
# Water everywhere: it is parameterized, it has a large gap (so the SCF converges without
# smearing and the exchange taper has a decaying density matrix to truncate), and it has a
# genuinely stiff internal mode, which is what makes energy conservation a real test rather
# than a test that nothing moved.

_OH = 0.9584
_WATER = np.array([[0.0, 0.0, 0.0], [_OH, 0.0, 0.0], [-0.2400, 0.9279, 0.0]])


def _water(offset=(0.0, 0.0, 0.0)) -> np.ndarray:
    return _WATER + np.asarray(offset, dtype=float)


def chain_1d() -> Atoms:
    """A hydrogen-bonded water chain along x. Periodic in one direction only."""
    return Atoms(
        "OH2",
        positions=_water(),
        cell=[[3.2, 0.0, 0.0], [0.0, 24.0, 0.0], [0.0, 0.0, 24.0]],
        pbc=[True, False, False],
    )


def sheet_2d() -> Atoms:
    """A water sheet in the xy plane. Periodic in two directions."""
    return Atoms(
        "OH2",
        positions=_water(),
        cell=[[3.4, 0.0, 0.0], [0.0, 3.4, 0.0], [0.0, 0.0, 24.0]],
        pbc=[True, True, False],
    )


def crystal_3d(length: float = 6.0) -> Atoms:
    """Two waters in a cubic cell, offset so no force vanishes by symmetry.

    The cell is diagonal, hence upper triangular, which ``ase.md.npt.NPT`` requires.
    """
    return Atoms(
        "OH2OH2",
        positions=np.vstack([_water((0.2, 0.1, 0.0)), _water((0.3, 0.4, 3.1))]),
        cell=np.eye(3) * length,
        pbc=True,
    )


def water_box(n_side: int = 2, spacing: float = 4.0) -> Atoms:
    """``n_side³`` waters on a cubic lattice, each given a different orientation.

    Spacing is 4.0 Å rather than the 2.76 Å of ice for the reason set out in
    ``tests/scaling.rs``: these orientations are pseudo-random, so at ice spacing hydrogens end
    up pointing at each other and the structure is clashed. At 4.0 Å the worst contact is a
    normal van der Waals one, and a zero-pressure barostat then has somewhere to go — the cell
    starts at 64 Å³ per molecule against liquid water's 30, so watching it contract is itself a
    physical check rather than just a stability one.
    """
    symbols, positions = [], []
    for index in range(n_side**3):
        i, j, k = index // (n_side * n_side), (index // n_side) % n_side, index % n_side
        centre = np.array([i, j, k], dtype=float) * spacing
        # A deterministic pseudo-random orientation from the lattice index (golden angle).
        t = index * 2.3999632
        s, c = np.sin(t), np.cos(t)
        u = np.array([c, s, 0.35])
        u /= np.linalg.norm(u)
        w = np.cross(u, [0.0, 0.0, 1.0])
        w /= np.linalg.norm(w)
        symbols += ["O", "H", "H"]
        positions += [
            centre,
            centre + 0.9584 * u,
            centre + 0.9584 * (-0.2440 * u + 0.9698 * w),
        ]
    atoms = Atoms(symbols, positions=np.array(positions),
                  cell=np.eye(3) * (n_side * spacing), pbc=True)
    return atoms


def _geometry_health(atoms):
    """Intramolecular bond lengths and angles, and the closest intermolecular contact.

    The decisive check that a trajectory is physical rather than merely finite. A run that has
    gone wrong shows up here long before the energy does: molecules dissociate (a bond stretches
    past ~1.2 Å), collapse (below ~0.8 Å), or atoms from different molecules pass through each
    other. None of that moves a "is the energy finite" assertion at all.
    """
    n_molecules = len(atoms) // 3
    bonds, angles = [], []
    for m in range(n_molecules):
        o, h1, h2 = 3 * m, 3 * m + 1, 3 * m + 2
        bonds.append(atoms.get_distance(o, h1, mic=True))
        bonds.append(atoms.get_distance(o, h2, mic=True))
        angles.append(atoms.get_angle(h1, o, h2, mic=True))

    # Closest contact between atoms of *different* molecules.
    distances = atoms.get_all_distances(mic=True)
    molecule_of = np.arange(len(atoms)) // 3
    intermolecular = molecule_of[:, None] != molecule_of[None, :]
    closest = float(np.where(intermolecular, distances, np.inf).min())
    return np.array(bonds), np.array(angles), closest


# k-meshes and cutoffs per system. The exchange cutoff is an approximation, not a convergence
# parameter (see docs/pbc.md), so it is stated explicitly rather than left to the default.
CALC_1D = dict(kpts=(4, 1, 1), exchange_cutoff=12.0)
CALC_2D = dict(kpts=(3, 3, 1), exchange_cutoff=10.0)
CALC_3D = dict(kpts=(1, 1, 1), exchange_cutoff=10.0)

# Tolerances for anything that *differentiates* the energy numerically, or compares two
# different routes to the same number. The defaults above are dynamics settings: an SCF
# converged to 1e-8 eV is plenty when the geometry moves 0.01 Å per step and the error is
# common to consecutive steps, and useless when the whole quantity being measured is the
# difference between two geometries 1e-4 apart. The convergence error does not cancel between
# them, because they converge along different paths.
TIGHT = dict(e_tol=1.0e-12, p_tol=1.0e-11, max_scf=2000)


def _thermalized(atoms: Atoms, calc_kwargs: dict, temperature_K: float, seed: int) -> Atoms:
    atoms = atoms.copy()
    atoms.calc = AM1(**calc_kwargs)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed), force_temp=True
    )
    return atoms


# ------------------------------------------------------------------------------ NVE / forces
@pytest.mark.parametrize(
    "name, builder, calc_kwargs",
    [
        ("1D chain", chain_1d, CALC_1D),
        ("2D sheet", sheet_2d, CALC_2D),
        ("3D crystal", crystal_3d, CALC_3D),
    ],
)
def test_nve_conserves_energy(name, builder, calc_kwargs):
    """Total energy is conserved under velocity Verlet, in every dimensionality.

    The tolerance is expressed **relative to the energy actually exchanged** between kinetic
    and potential during the run, not as an absolute number. That matters: an absolute bound
    can be met by a trajectory where nothing happens, whereas this one can only be met by a
    force that really is the gradient of the energy. If the periodic force were missing a term,
    the drift would be a sizeable fraction of the swing rather than a percent of it.
    """
    atoms = _thermalized(builder(), calc_kwargs, temperature_K=200, seed=7)
    dyn = VelocityVerlet(atoms, timestep=0.25 * units.fs)
    dyn.run(5)  # let the initial velocities settle into the potential

    total, potential = [], []

    def record():
        e_pot = atoms.get_potential_energy()
        total.append(e_pot + atoms.get_kinetic_energy())
        potential.append(e_pot)

    record()
    dyn.attach(record, interval=1)
    dyn.run(40)

    total = np.asarray(total)
    swing = float(np.ptp(np.asarray(potential)))
    drift = abs(total[-1] - total[0])
    spread = float(np.ptp(total))

    print(
        f"\n    {name}: drift {drift * 1e3:+.3f} meV, spread {spread * 1e3:.3f} meV, "
        f"potential swing {swing * 1e3:.1f} meV "
        f"({drift / len(atoms) * 1e3:.4f} meV/atom)"
    )
    assert swing > 1.0e-3, "nothing moved, so conservation was not tested"
    assert drift < 0.05 * swing, (
        f"{name}: energy drifted {drift * 1e3:.3f} meV against a potential swing of "
        f"{swing * 1e3:.1f} meV -- the periodic force is not the gradient of the energy"
    )
    assert spread < 0.20 * swing


# ------------------------------------------------------------------------------- NPT / stress
def test_npt_is_stable_and_physically_reasonable():
    """A real NPT ensemble, checked for **divergence and physical validity**, not just survival.

    This is the acceptance test for the periodic work, and it is the only thing here that moves
    the **cell** — so it is the only thing that consumes the analytic stress the way a user
    would, including the requirement that it arrive as a Voigt 6-vector in eV/Å³ in ASE's
    component order. A stress with the shear components transposed, or off by a factor of the
    cell volume, passes every finite-difference test in the Rust suite and destroys this
    trajectory.

    "It ran without raising" is a weak claim, so it is not the one being made. What is asserted:

    * **Not diverging.** The Nosé–Hoover conserved quantity oscillates by construction, so a raw
      drift bound says little. The test compares the fitted linear trend over the whole
      production window against the quantity's own oscillation amplitude: a genuine runaway is a
      trend comparable to or larger than the fluctuation, and that is what is excluded.
    * **Molecules survive as molecules.** Every O–H bond and H–O–H angle, at every recorded step,
      inside a physical range. A trajectory going wrong dissociates or collapses molecules long
      before its energy stops being finite, and no energy-based assertion notices.
    * **Matter does not pass through itself.** The closest intermolecular contact stays above a
      floor at every step.
    * **The thermostat works.** The mean temperature over the production window sits near the
      target, within the fluctuation a system this small actually has — `σ_T/T = √(2/3N)`, which
      is 14 % for 24 atoms, so the band is wide on purpose rather than by carelessness.
    * **The barostat does something physical.** Starting at 64 Å³ per molecule against liquid
      water's 30, a zero-pressure barostat must *contract* the cell, and the density must stay
      inside a range that is recognisably condensed matter.
    """
    atoms = _thermalized(water_box(2), CALC_3D, temperature_K=250, seed=5)
    n_molecules = len(atoms) // 3
    volume_0 = atoms.get_volume()

    dyn = NPT(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=250,
        externalstress=0.0,
        ttime=25 * units.fs,
        pfactor=(100 * units.fs) ** 2 * 10 * units.GPa,
    )
    # Equilibrate. The conserved quantity is only meaningful once the thermostat and barostat
    # variables have spun up, so nothing before this point is measured.
    dyn.run(30)

    conserved, volumes, temperatures = [], [], []
    bond_min, bond_max, angle_min, angle_max, contact_min = [], [], [], [], []

    def record():
        conserved.append(dyn.get_gibbs_free_energy())
        volumes.append(atoms.get_volume())
        temperatures.append(atoms.get_temperature())
        bonds, angles, closest = _geometry_health(atoms)
        bond_min.append(bonds.min())
        bond_max.append(bonds.max())
        angle_min.append(angles.min())
        angle_max.append(angles.max())
        contact_min.append(closest)

    record()
    dyn.attach(record, interval=1)
    dyn.run(80)

    conserved = np.asarray(conserved)
    volumes = np.asarray(volumes)
    temperatures = np.asarray(temperatures)
    steps = np.arange(len(conserved))

    # --- divergence ---------------------------------------------------------------------
    slope = float(np.polyfit(steps, conserved, 1)[0])       # eV per step
    total_trend = abs(slope) * len(steps)                   # eV over the production window
    oscillation = float(conserved.std())
    density = n_molecules * 18.015 / (volumes * 0.6022142)  # g/cm³

    print(
        f"\n    NPT on {n_molecules} waters ({len(atoms)} atoms), {len(steps)} production steps:"
        f"\n      conserved quantity : trend {slope * 1e3:+.3f} meV/step, "
        f"total {total_trend * 1e3:+.1f} meV vs σ = {oscillation * 1e3:.1f} meV"
        f"\n      temperature        : {temperatures.mean():.0f} ± {temperatures.std():.0f} K "
        f"(target 250)"
        f"\n      volume             : {volume_0:.1f} → {volumes[-1]:.1f} Å³ "
        f"({volumes.min():.1f}–{volumes.max():.1f})"
        f"\n      density            : {density.mean():.3f} g/cm³"
        f"\n      O–H bond           : {min(bond_min):.3f}–{max(bond_max):.3f} Å"
        f"\n      H–O–H angle        : {min(angle_min):.1f}–{max(angle_max):.1f}°"
        f"\n      closest contact    : {min(contact_min):.3f} Å"
    )

    assert np.all(np.isfinite(conserved)), "the conserved quantity went non-finite"
    # The real divergence test: a runaway is a trend comparable to the natural fluctuation.
    # Nosé–Hoover conserves only on average, so demanding zero trend would be wrong.
    assert total_trend < 2.0 * oscillation, (
        f"the conserved quantity is diverging: it trends {total_trend * 1e3:.1f} meV over the "
        f"window against an oscillation amplitude of only {oscillation * 1e3:.1f} meV"
    )

    # --- the molecules are still molecules -----------------------------------------------
    assert min(bond_min) > 0.85, f"an O–H bond collapsed to {min(bond_min):.3f} Å"
    assert max(bond_max) < 1.20, f"an O–H bond dissociated to {max(bond_max):.3f} Å"
    assert min(angle_min) > 90.0, f"an H–O–H angle closed to {min(angle_min):.1f}°"
    assert max(angle_max) < 130.0, f"an H–O–H angle opened to {max(angle_max):.1f}°"
    assert min(contact_min) > 1.2, (
        f"atoms from different molecules came within {min(contact_min):.3f} Å — the trajectory "
        "is passing matter through itself"
    )

    # --- the thermostat and barostat are doing their jobs --------------------------------
    # σ_T/T = sqrt(2/3N) for N atoms; at 24 atoms that is 14 %, so allow ~3σ plus slack.
    expected_sigma = 250.0 * np.sqrt(2.0 / (3.0 * len(atoms)))
    assert abs(temperatures.mean() - 250.0) < 4.0 * expected_sigma, (
        f"mean temperature {temperatures.mean():.0f} K is far from the 250 K target "
        f"(expected fluctuation ±{expected_sigma:.0f} K)"
    )
    # Starting well below liquid density, a zero-pressure barostat must contract the cell.
    assert volumes[-1] < volume_0, (
        f"a zero-pressure barostat should contract a cell at {volume_0 / n_molecules:.0f} Å³ per "
        f"molecule; it went {volume_0:.1f} → {volumes[-1]:.1f} Å³"
    )
    # And the result has to remain recognisably condensed matter rather than collapsing or boiling.
    assert 0.2 < density.mean() < 3.0, (
        f"mean density {density.mean():.3f} g/cm³ is not a physical condensed phase"
    )
    # The cell must stay a cell: no near-degenerate lattice vector.
    lengths = np.linalg.norm(np.asarray(atoms.get_cell()), axis=1)
    assert lengths.min() > 2.0, f"a lattice vector collapsed to {lengths.min():.2f} Å"


def test_drift_is_integrator_error_not_force_error():
    """The decisive divergence test: halve the timestep and watch the drift collapse.

    A trajectory always drifts a little, and staring at a single number cannot tell you whether
    that is the integrator or the forces. The two have completely different signatures:

    * **Integrator error** falls as `dt²` for velocity Verlet.
    * **A force inconsistent with the energy** does not fall at all. Halving `dt` doubles the
      number of steps and each step commits the same relative error, so the drift per unit
      *time* is unchanged.

    So this measures the same physical interval at four step sizes. Measured: 11.62, 3.18, 0.84
    and 0.22 meV per atom per picosecond at 1.0, 0.5, 0.25 and 0.125 fs — ratios of 3.66, 3.77
    and 3.91, converging on the 4 that `dt²` demands. That settles the question. It is also why
    the tests above use 0.25–0.5 fs: an unconstrained O–H stretch has a period near 9 fs, and
    1 fs does not resolve it.

    # Why this is NVE and not NPT

    It used to run under `NPT` and fit `get_gibbs_free_energy()`. That measurement is not valid,
    and its invalidity is visible in the result: 24.97, 1.73, 9.27 meV/atom/ps at 1.0, 0.5 and
    0.25 fs — *non-monotone*, so no exponent describes it and the premise of the test cannot be
    tested. The reason is that the Nosé–Hoover/Parrinello–Rahman conserved quantity contains
    thermostat and barostat terms which oscillate with their own time constants — 25 fs and
    100 fs here — while the fit window is 12 fs. A straight line through a fifth of an
    oscillation measures its phase, not a secular drift, and the phase moves with `dt` for
    reasons that have nothing to do with the integrator's order.

    Lengthening the window is not available: this cell is two waters in 6 Å, and an NPT
    trajectory here reaches a geometry whose SCF does not converge before 300 fs.

    That the residual was *not* numerical noise was checked separately — tightening the SCF from
    `1e-8` to `1e-12` moved the three numbers by less than 0.01 % — so the old test was not
    failing because of convergence error either. It was measuring the wrong thing.

    Under NVE the conserved quantity is the plain total energy: no auxiliary oscillator, and the
    `dt²` signature is clean. NPT's own correctness is covered by the tests around this one —
    that it runs, and that the cell responds to pressure in the right direction, which is what
    catches a sign error in the stress.
    """
    physical_fs = 12.0
    results = {}
    for dt in (1.0, 0.5, 0.25, 0.125):
        atoms = _thermalized(crystal_3d(), CALC_3D, temperature_K=250, seed=5)
        dyn = VelocityVerlet(atoms, timestep=dt * units.fs)
        dyn.run(int(round(10.0 / dt)))  # same equilibration time at every step size
        conserved = [atoms.get_total_energy()]
        dyn.attach(lambda: conserved.append(atoms.get_total_energy()), interval=1)
        dyn.run(int(round(physical_fs / dt)))

        conserved = np.asarray(conserved)
        slope = float(np.polyfit(np.arange(len(conserved)), conserved, 1)[0])
        total = abs(slope) * len(conserved)
        # meV per atom per picosecond — the standard figure of merit, and independent of dt.
        results[dt] = total / len(atoms) / (physical_fs / 1000.0) * 1e3
        print(f"\n    dt = {dt:5.3f} fs: drift {results[dt]:8.4f} meV/atom/ps")

    # Each halving must improve things by at least 2.5x. A dt-independent error — the signature
    # of a force that is not the gradient of the energy — gives 1; `dt²` gives 4. The bar sits
    # between them rather than at 4, because the measured ratios approach 4 from below and
    # asserting the asymptote itself would be asserting that the leading term is the only one.
    for coarse, fine in ((1.0, 0.5), (0.5, 0.25), (0.25, 0.125)):
        ratio = results[coarse] / results[fine]
        assert ratio > 2.5, (
            f"halving the timestep from {coarse} to {fine} fs improved the drift by only "
            f"{ratio:.2f}x ({results[coarse]:.4f} → {results[fine]:.4f} meV/atom/ps). `dt²` "
            "gives 4x; a force that is not the gradient of the energy gives 1x."
        )
    # And the smallest step must reach a genuinely good figure, not merely a better one.
    assert results[0.125] < 0.5, (
        f"even at 0.125 fs the drift is {results[0.125]:.4f} meV/atom/ps, which is too large "
        "for the trajectory to be trusted over a production run"
    )


def test_npt_volume_responds_to_external_pressure():
    """Squeezing harder must give a smaller cell.

    A sign error in the stress is the classic way to get a barostat that runs and produces
    nonsense, so the ordering is checked directly rather than inferred from a single run.
    """
    volumes = {}
    for pressure_gpa in (-3.0, 0.0, 3.0):
        atoms = _thermalized(crystal_3d(), CALC_3D, temperature_K=200, seed=5)
        dyn = NPTBerendsen(
            atoms,
            timestep=1.0 * units.fs,
            temperature_K=200,
            taut=50 * units.fs,
            pressure_au=pressure_gpa * units.GPa,
            taup=200 * units.fs,
            compressibility_au=5.0e-3 / units.GPa,
        )
        dyn.run(25)
        volumes[pressure_gpa] = atoms.get_volume()
        print(f"\n    P = {pressure_gpa:+5.1f} GPa -> V = {atoms.get_volume():.4f} Å³")

    assert volumes[3.0] < volumes[0.0] < volumes[-3.0], (
        "the cell did not respond monotonically to the applied pressure: "
        f"{volumes} -- the sign of the stress is probably wrong"
    )


def test_stress_matches_a_strain_finite_difference_through_ase():
    """The stress ASE receives is dE/dε per unit volume, checked by straining the cell.

    The Rust suite already checks this against the internal energy. Repeating it *through the
    ASE boundary* is not redundant: it is where the unit conversion, the Voigt packing and the
    row-vs-column convention of ``atoms.cell`` live, and those are exactly the things a barostat
    would silently misuse.
    """
    calc_kwargs = {**CALC_3D, **TIGHT}
    atoms = crystal_3d()
    atoms.calc = AM1(**calc_kwargs)
    stress = atoms.get_stress(voigt=False)
    volume = atoms.get_volume()

    cell_0 = np.asarray(atoms.get_cell(), dtype=float)
    positions_0 = atoms.get_positions()

    h = 2.0e-5
    worst = 0.0
    # Voigt order: xx, yy, zz, yz, xz, xy.
    for i, j in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        energies = []
        for sign in (+1.0, -1.0):
            strain = np.eye(3)
            strain[i, j] += 0.5 * sign * h
            strain[j, i] += 0.5 * sign * h
            strained = atoms.copy()
            # `atoms.cell` holds the lattice vectors as ROWS, so a right-multiply by the
            # (symmetric) strain applies it to each vector.
            strained.set_cell(cell_0 @ strain, scale_atoms=False)
            strained.set_positions(positions_0 @ strain)
            strained.calc = AM1(**calc_kwargs)
            energies.append(strained.get_potential_energy())
        finite_difference = (energies[0] - energies[1]) / (2.0 * h) / volume
        worst = max(worst, abs(finite_difference - stress[i, j]))
        print(
            f"\n    ({i},{j})  analytic {stress[i, j]:+.9f}   "
            f"finite difference {finite_difference:+.9f} eV/Å³"
        )

    assert worst < 1.0e-7, f"stress disagrees with the strain derivative by {worst:.2e} eV/Å³"


# --------------------------------------------------------------------------- partial periodicity
@pytest.mark.parametrize(
    "name, builder, calc_kwargs, periodic_axes",
    [
        ("1D chain", chain_1d, CALC_1D, (0,)),
        ("2D sheet", sheet_2d, CALC_2D, (0, 1)),
    ],
)
def test_nvt_on_a_chain_and_a_slab(name, builder, calc_kwargs, periodic_axes):
    """A partially periodic cell survives dynamics, and carries no stress where it has no extent.

    Parrinello–Rahman cell dynamics assumes three periodic directions, so a chain and a slab get
    a thermostat instead of a barostat; the stress is checked directly rather than through a
    cell that cannot move.
    """
    atoms = _thermalized(builder(), calc_kwargs, temperature_K=200, seed=3)
    dyn = Langevin(
        atoms,
        timestep=0.25 * units.fs,
        temperature_K=200,
        friction=0.01 / units.fs,
        rng=np.random.default_rng(3),
    )
    dyn.run(25)

    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_temperature() < 5000.0

    stress = atoms.get_stress(voigt=False)
    non_periodic = [axis for axis in range(3) if axis not in periodic_axes]
    for i in range(3):
        for j in range(3):
            if i in non_periodic or j in non_periodic:
                assert stress[i, j] == 0.0, (
                    f"{name}: stress ({i},{j}) = {stress[i, j]:.3e} on an axis with no "
                    "periodicity -- there is no extent there to divide by"
                )
    on_axis = stress[periodic_axes[0], periodic_axes[0]]
    print(f"\n    {name}: stress along the periodic axis = {on_axis:+.6f}")
    assert np.isfinite(on_axis)


def test_molecular_structure_has_no_stress():
    """``atoms.pbc`` decides, and a molecule in free space must refuse rather than return zeros.

    Returning a zero stress would let a variable-cell optimizer run happily on a system that
    has no cell, which is worse than an error.
    """
    from ase.calculators.calculator import PropertyNotImplementedError

    atoms = Atoms("OH2", positions=_water())
    atoms.calc = AM1()
    assert np.isfinite(atoms.get_potential_energy())
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()


# ------------------------------------------------------------------------------- charged cell
def test_a_charged_cell_is_usable_for_everything_except_its_absolute_energy():
    """A cell with a net formal charge: what holds, and the warning about what does not.

    The absolute energy of a charged periodic cell is **not** converged here — without Ewald
    summation there is no compensating background and the monopole lattice sum diverges with the
    cutoff (see ``docs/pbc.md``, and ``tests/pbc_charged.rs`` which measures the divergence).
    So this test deliberately does not assert anything about the energy's value.

    What it does assert is everything that *is* well defined: the charge is conserved, the
    warning is raised, and — the substantive one — the forces are consistent with whatever
    energy is being reported, which is what lets dynamics run at all. Energy conservation under
    NVE tests that consistency without needing the energy to be physically converged.
    """
    atoms = crystal_3d()
    with pytest.warns(RuntimeWarning, match="net charge"):
        atoms.calc = AM1(charge=1.0, multiplicity=2, **CALC_3D)
        atoms.get_potential_energy()

    charges = atoms.calc.results["charges"]
    print(f"\n    charged cell: E = {atoms.get_potential_energy():.4f} eV (not converged; "
          f"see docs/pbc.md), Σq = {charges.sum():+.8f} e")
    assert abs(charges.sum() - 1.0) < 1.0e-8, "Mulliken charges must sum to the formal charge"

    # Force/energy consistency on the charged cell, by the same NVE argument as for the neutral
    # one. The energy being cutoff-dependent does not excuse the forces from being its gradient.
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=200, rng=np.random.default_rng(9), force_temp=True
    )
    dyn = VelocityVerlet(atoms, timestep=0.25 * units.fs)
    dyn.run(5)
    total, potential = [], []

    def record():
        e_pot = atoms.get_potential_energy()
        total.append(e_pot + atoms.get_kinetic_energy())
        potential.append(e_pot)

    record()
    dyn.attach(record, interval=1)
    dyn.run(30)

    swing = float(np.ptp(np.asarray(potential)))
    drift = abs(total[-1] - total[0])
    print(f"    charged NVE: drift {drift * 1e3:+.3f} meV against a swing of {swing * 1e3:.1f} meV")
    assert swing > 1.0e-3
    assert drift < 0.05 * swing, (
        f"the charged-cell forces are not the gradient of the charged-cell energy: "
        f"drift {drift * 1e3:.3f} meV against a swing of {swing * 1e3:.1f} meV"
    )


def test_rm1_runs_periodic_dynamics_too():
    """RM1 shares AM1's code path entirely, so periodic dynamics should just work for it.

    Worth one explicit run rather than trusting the sharing: the method travels with the
    parameter set, and a binding that dropped the argument somewhere in the periodic path would
    silently give AM1 answers under an RM1 label. The energies differing is what rules that out.
    """
    am1 = crystal_3d()
    am1.calc = AM1(**CALC_3D)
    rm1 = crystal_3d()
    rm1.calc = AM1(method="rm1", **CALC_3D)

    e_am1 = am1.get_potential_energy()
    e_rm1 = rm1.get_potential_energy()
    print(f"\n    periodic cell: AM1 {e_am1:.4f} eV, RM1 {e_rm1:.4f} eV")
    assert abs(e_am1 - e_rm1) > 0.5, "RM1 should not reproduce AM1's energy; is `method` reaching the solver?"

    MaxwellBoltzmannDistribution(
        rm1, temperature_K=200, rng=np.random.default_rng(2), force_temp=True
    )
    dyn = VelocityVerlet(rm1, timestep=0.25 * units.fs)
    dyn.run(5)
    total, potential = [], []

    def record():
        e_pot = rm1.get_potential_energy()
        total.append(e_pot + rm1.get_kinetic_energy())
        potential.append(e_pot)

    record()
    dyn.attach(record, interval=1)
    dyn.run(25)

    swing = float(np.ptp(np.asarray(potential)))
    drift = abs(total[-1] - total[0])
    print(f"    RM1 periodic NVE: drift {drift * 1e3:+.3f} meV, swing {swing * 1e3:.1f} meV")
    assert swing > 1.0e-3
    assert drift < 0.05 * swing


def test_a_neutral_cell_raises_no_charge_warning():
    """The complement: the warning must not fire when there is nothing to warn about."""
    atoms = crystal_3d()
    atoms.calc = AM1(**CALC_3D)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        atoms.get_potential_energy()


# ---------------------------------------------------------------------------- physics, via ASE
def test_a_k_mesh_agrees_with_the_equivalent_supercell():
    """Band folding, checked at the ASE boundary.

    An n-point mesh on the primitive cell and Γ on the n-fold supercell describe the same
    infinite chain, so the energy **per primitive cell** must agree. This is the sharp test of
    the Bloch phases, and doing it here also confirms that ``atoms.repeat`` — how a user would
    actually build the supercell — lands in the same physics.

    The real-space cutoff has to be far larger than for an ordinary calculation, and the reason
    is worth stating: it truncates on the lattice translation |T|, so the *same* cutoff admits
    different sets of physical pairs in the two cells. At the 40 Bohr default the primitive
    cell reaches ±6 cells symmetrically while the 3× supercell reaches −6…+8, and the two
    disagree by 7e-5 eV for that reason alone — a truncation artefact that has nothing to do
    with the phases this test is trying to check. At 320 Bohr both are converged and they agree
    to 3e-9 eV.
    """
    converged_sum = dict(exchange_cutoff=12.0, realspace_cutoff=320.0, **TIGHT)

    primitive = chain_1d()
    primitive.calc = AM1(kpts=(3, 1, 1), **converged_sum)
    per_cell_mesh = primitive.get_potential_energy()

    supercell = chain_1d().repeat((3, 1, 1))
    supercell.calc = AM1(kpts=(1, 1, 1), **converged_sum)
    per_cell_folded = supercell.get_potential_energy() / 3.0

    print(
        f"\n    3 k-points on the primitive cell: {per_cell_mesh:.9f} eV;  "
        f"Γ on the 3× supercell: {per_cell_folded:.9f} eV per cell;  "
        f"Δ = {per_cell_mesh - per_cell_folded:+.2e}"
    )
    assert abs(per_cell_mesh - per_cell_folded) < 1.0e-7

def test_a_field_transverse_to_a_chain_is_allowed_and_along_it_is_not():
    """An external field under a cell, through the ASE boundary.

    Refused for any cell through 0.2.1. The rule is on the *direction*: `F.R` shifts by `F.T`
    under translation by `T`, so the perturbation is lattice-periodic exactly when `F.T = 0` for
    every lattice vector. A chain along x may sit in a field along y or z; a field along x is
    still an error, and the message says which component is the problem.

    Through ASE the field is in V/A and reaches the native side converted with the crate's own
    Bohr radius, so this also checks that the periodic branch did not grow its own conversion.
    """
    atoms = chain_1d()
    atoms.calc = AM1(**CALC_1D)
    e0 = atoms.get_potential_energy()

    transverse = chain_1d()
    transverse.calc = AM1(field=[0.0, 0.15, 0.0], **CALC_1D)
    e_plus = transverse.get_potential_energy()
    forces = transverse.get_forces()

    minus = chain_1d()
    minus.calc = AM1(field=[0.0, -0.15, 0.0], **CALC_1D)
    e_minus = minus.get_potential_energy()

    mean = 0.5 * (e_plus + e_minus)
    print(
        f"    chain in a transverse field: E(0) = {e0:.9f}, E(+F) = {e_plus:.9f}, "
        f"E(-F) = {e_minus:.9f} eV, mean - E(0) = {mean - e0:.3e}"
    )
    # The linear term cancels in the mean; what is left is the polarizability, which lowers the
    # energy for any stable system. One sign alone can rise, and does, because the cell is polar.
    assert mean < e0, f"the second-order response must lower the energy: {mean} vs {e0}"
    assert np.all(np.isfinite(forces))

    along = chain_1d()
    along.calc = AM1(field=[0.15, 0.0, 0.0], **CALC_1D)
    with pytest.raises(Exception) as excinfo:
        along.get_potential_energy()
    message = str(excinfo.value)
    print(f"    along the chain: {message.splitlines()[0][:110]}")
    assert "along a periodic direction" in message, message

def _h2_crystal(a: float = 5.0) -> Atoms:
    """Two hydrogens in a cubic cell: s orbitals only, so the Berry phase and the CPHF response
    are computing the same object."""
    return Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.80, 0.15, 0.05]],
        cell=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
        pbc=True,
    )


def test_berry_polarization_and_the_finite_field_are_reachable_from_ase():
    """The two Berry-phase capabilities, through the ASE surface.

    Both are new in 0.2.2 and were Rust-only until this: `get_polarization` and
    `get_finite_field`. The physics is validated in Rust (`tests/pbc_finite_field.rs` compares the
    finite-field polarizability against the CPHF one); what this checks is that the ASE layer
    reaches them, converts the field units, and returns what it says it does.
    """
    atoms = _h2_crystal()
    calc = AM1(kpts=(4, 4, 4), realspace_cutoff=30.0, exchange_cutoff=12.0, smearing=0.0)
    atoms.calc = calc
    calc.set_atoms(atoms)

    p = calc.get_polarization()
    assert p.shape == (3,)
    assert np.all(np.isfinite(p))
    # Neutral H2: the electronic centre sits on the bond midpoint and so does the ionic one, so
    # the total polarization is zero on this branch. That it comes out zero rather than merely
    # finite is what says the two halves carry the sign convention consistently.
    assert np.abs(p).max() < 1e-10, p

    r = calc.get_finite_field([0.0, 0.0, 0.05])
    print(
        f"    finite field: enthalpy {r['enthalpy_ev']:.9f} eV, energy {r['energy_ev']:.9f} eV, "
        f"{r['outer_iterations']} outer iterations"
    )
    # The enthalpy is what is minimized, so it sits below the energy of the same state whenever
    # the field has polarized it at all.
    assert r["enthalpy_ev"] < r["energy_ev"]
    assert r["outer_iterations"] >= 1
    assert len(r["polarization"]) == 3

    # A field in V/A must reach the native layer in atomic units. The same field expressed both
    # ways has to give the same enthalpy.
    from am1_rs import native

    c = native.constants()
    au = 0.05 * c["bohr_to_angstrom"] / c["hartree_to_ev"]
    direct = native.finite_field(
        atoms.get_atomic_numbers(),
        atoms.get_positions(),
        np.asarray(atoms.get_cell(), dtype=float),
        atoms.pbc,
        [0.0, 0.0, au],
        kpts=(4, 4, 4),
        realspace_cutoff=30.0,
        exchange_cutoff=12.0,
    )
    assert abs(direct["enthalpy_ev"] - r["enthalpy_ev"]) < 1e-9


def test_the_berry_phase_route_refuses_a_low_dimensional_cell():
    """A chain is polarized along its periodic direction only, which the module does not separate
    out; it says so rather than returning a three-dimensional number."""
    atoms = chain_1d()
    calc = AM1(**CALC_1D)
    atoms.calc = calc
    calc.set_atoms(atoms)
    with pytest.raises(Exception) as excinfo:
        calc.get_polarization()
    assert "three-dimensional" in str(excinfo.value)

def test_the_polarizability_works_in_reduced_dimensionality():
    """`alpha` for a chain and a slab, where `eps_inf` is refused.

    `eps_inf = 1 + 4*pi*alpha/V` needs `V` to be a volume, so `get_dielectric_tensor` refuses a
    chain or a slab -- but `alpha` is a *response*, well defined whatever the cell is periodic in,
    and it was refused along with the conversion until 0.2.2. This checks that the two are now
    separate and that the refusal points at the one that works.
    """
    for name, builder, calc_kwargs in (("chain", chain_1d, CALC_1D), ("slab", sheet_2d, CALC_2D)):
        atoms = builder()
        calc = AM1(smearing=0.0, **calc_kwargs)
        atoms.calc = calc
        calc.set_atoms(atoms)

        alpha = calc.get_polarizability()
        assert alpha.shape == (3, 3)
        assert np.all(np.isfinite(alpha))
        assert np.all(np.diag(alpha) > 0.0), (name, alpha)
        assert np.allclose(alpha, alpha.T, atol=1e-9 * max(abs(np.diag(alpha)).max(), 1.0))
        print(f"    {name}: alpha diagonal = {np.diag(alpha)}")

        with pytest.raises(Exception) as excinfo:
            calc.get_dielectric_tensor()
        assert "three-dimensional" in str(excinfo.value)
        # And the refusal names both ways forward rather than stopping there.
        assert "get_dielectric_tensor_with_extent" in str(excinfo.value)


def test_the_extent_crosses_the_ase_boundary_in_angstrom():
    """A thickness is a length and a cross-section is an area, so they convert with different
    powers of the Bohr radius.

    The ASE surface is Angstrom throughout and the native one is Bohr, and this is the only place
    in the crate where the *same argument name* would carry different powers depending on which
    variant it is. Getting it wrong is a silent factor of 0.53 or 0.28, both of which return a
    perfectly plausible dielectric constant -- so it is checked against the native call rather
    than trusted.
    """
    from am1_rs import native

    bohr = 1.0 / native.constants()["bohr_to_angstrom"]

    atoms = sheet_2d()
    calc = AM1(smearing=0.0, **CALC_2D)
    atoms.calc = calc
    calc.set_atoms(atoms)
    d_ang = 3.0
    via_ase = calc.get_dielectric_tensor_with_extent(slab_thickness=d_ang)
    direct = native.dielectric_with_extent(
        atoms.get_atomic_numbers(),
        atoms.get_positions(),
        np.asarray(atoms.get_cell(), dtype=float),
        atoms.pbc,
        slab_thickness=d_ang * bohr,
        kpts=CALC_2D["kpts"],
        exchange_cutoff=CALC_2D["exchange_cutoff"],
        realspace_cutoff=AM1.default_parameters["realspace_cutoff"],
    )
    assert np.allclose(via_ase, np.asarray(direct["epsilon_infinity"]), rtol=1e-12)
    assert direct["extent"] == pytest.approx(d_ang * bohr)

    atoms = chain_1d()
    calc = AM1(smearing=0.0, **CALC_1D)
    atoms.calc = calc
    calc.set_atoms(atoms)
    s_ang2 = 12.0
    full = calc.get_dielectric_tensor_with_extent(wire_cross_section=s_ang2, full=True)
    # An **area**: two powers of the conversion, not one.
    assert full["extent"] == pytest.approx(s_ang2 * bohr**2)
    assert full["n_periodic"] == 1
    eps = np.asarray(full["epsilon_infinity"])
    assert np.all(np.diag(eps) >= 1.0)
