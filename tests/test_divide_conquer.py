# SPDX-License-Identifier: GPL-3.0-or-later
"""Divide-and-conquer through the Python and ASE surfaces.

The formulation is tested in ``tests/divide_conquer.rs``. What is tested here is that it is
reachable and correct *through the bindings* — that the buffer radius actually reaches the
solver, that forces come back in ASE's sign and units, and that the warnings surface as Python
warnings rather than being swallowed at the boundary.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

am1_rs = pytest.importorskip("am1_rs")


def water_chain(n: int, spacing: float = 2.9):
    """``n`` waters in a line, alternately offset so nothing is degenerate by symmetry."""
    numbers, positions = [], []
    for i in range(n):
        base = np.array([i * spacing, 0.35 if i % 2 else 0.0, 0.0])
        numbers += [8, 1, 1]
        positions += [
            base,
            base + [0.9584, 0.0, 0.0],
            base + [-0.2400, 0.9279, 0.0],
        ]
    return numbers, np.array(positions)


# ---------------------------------------------------------------------------- native surface
def test_a_buffer_covering_the_molecule_reproduces_the_full_scf():
    numbers, positions = water_chain(4)
    full = am1_rs.single_point(numbers, positions)
    dc = am1_rs.divide_conquer(
        numbers, positions, buffer_radius=500.0, core_size=3, smearing_ev=0.0,
        e_tol=1e-11, p_tol=1e-10, max_scf=600,
    )
    delta = abs(dc["energy_ev"] - full["energy_ev"])
    print(f"\n    full {full['energy_ev']:.9f} eV, DC {dc['energy_ev']:.9f} eV, Δ = {delta:.2e}")
    assert delta < 1e-7
    assert np.allclose(dc["charges"], full["charges"], atol=1e-7)


def test_the_buffer_radius_reaches_the_solver():
    # A parameter that silently did not arrive would leave every result identical. This is the
    # cheapest way to notice a binding that drops an argument.
    numbers, positions = water_chain(6)
    energies = {
        r: am1_rs.divide_conquer(numbers, positions, buffer_radius=r, core_size=6)["energy_ev"]
        for r in (6.0, 12.0, 30.0)
    }
    print("\n   ", {k: round(v, 6) for k, v in energies.items()})
    assert len(set(energies.values())) == 3, "the buffer radius did not change the answer"

    reference = am1_rs.single_point(numbers, positions)["energy_ev"]
    errors = [abs(energies[r] - reference) for r in (6.0, 12.0, 30.0)]
    assert errors[0] > errors[1] > errors[2], f"error should fall with the buffer: {errors}"


def test_the_scaling_counters_come_back():
    numbers, positions = water_chain(16)
    r = am1_rs.divide_conquer(numbers, positions, buffer_radius=11.0, core_size=6)
    n_atoms = len(numbers)
    print(
        f"\n    {r['subsystems']} subsystems, largest {r['largest_subsystem_aos']} AOs, "
        f"Σn³/atom {r['diagonalization_work'] / n_atoms:.0f}, "
        f"coulomb/atom {r['coulomb_work'] / n_atoms:.1f}, "
        f"exchange/atom {r['exchange_work'] / n_atoms:.2f}"
    )
    assert r["subsystems"] > 1
    assert r["largest_subsystem_aos"] < 6 * 16, "subsystems should be smaller than the whole"
    assert r["exchange_work"] < r["coulomb_work"], (
        "the truncated density should make the exchange cheaper than the Coulomb sum"
    )
    assert r["homo_lumo_gap_ev"] > 0.0
    assert r["small_gap_warning"] is None


def test_open_shell_divide_and_conquer_matches_the_full_uhf():
    numbers, positions = water_chain(3)
    full = am1_rs.single_point(numbers, positions, charge=1.0, multiplicity=2)
    dc = am1_rs.divide_conquer(
        numbers, positions, charge=1.0, multiplicity=2, buffer_radius=500.0, core_size=3,
        smearing_ev=0.0, e_tol=1e-11, p_tol=1e-10, max_scf=600,
    )
    delta = abs(dc["energy_ev"] - full["energy_ev"])
    print(f"\n    open shell: full {full['energy_ev']:.9f}, DC {dc['energy_ev']:.9f}, Δ = {delta:.2e}")
    assert delta < 1e-6
    assert dc["unrestricted"]
    assert len(dc["fermi_energies_ev"]) == 2, "UHF needs one chemical potential per spin channel"
    assert abs(sum(dc["charges"]) - 1.0) < 1e-8


# -------------------------------------------------------------------------------- ASE surface
def test_ase_divide_conquer_matches_the_native_call():
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from am1_rs.ase import AM1

    numbers, positions = water_chain(6)
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.calc = AM1(divide_conquer=True, core_size=6, buffer_radius=12.0)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # The binding test proper: ASE must reach the same solver with the same arguments.
    native = am1_rs.divide_conquer(
        numbers, positions, core_size=6, buffer_radius=12.0, smearing_ev=0.05
    )
    assert abs(energy - native["energy_ev"]) < 1e-9
    assert np.allclose(forces, native["forces_ev_per_angstrom"], atol=1e-12)

    # Separately: the sign and units are ASE's. Checked against the full calculator at a buffer
    # large enough that the divide-and-conquer error is far below the tolerance -- at 12 Bohr it
    # is ~3e-4 eV/Å, which is the method working as documented, not a binding problem, so
    # comparing there would be testing the approximation rather than the wiring.
    plain = Atoms(numbers=numbers, positions=positions)
    plain.calc = AM1()
    converged = Atoms(numbers=numbers, positions=positions)
    converged.calc = AM1(divide_conquer=True, core_size=6, buffer_radius=60.0, smearing=0.0)
    worst = np.abs(converged.get_forces() - plain.get_forces()).max()
    print(f"    max |F_DC(60 Bohr) − F_full| = {worst:.2e} eV/Å")
    assert worst < 1e-6, (
        f"at a 60 Bohr buffer the forces should match the full ones; worst {worst:.2e} eV/Å"
    )


def test_ase_divide_conquer_runs_under_a_periodic_cell():
    """Periodic divide-and-conquer through ASE, which until 0.2.1 raised instead.

    The Rust API had accepted a lattice since 0.2.0 — the subsystem buffers are built from the
    image-aware pair list, so a buffer wraps through the cell boundary — but neither Python
    surface passed the cell through, and the ASE error said the buffers were "not wired up yet".

    The assertion that matters is not that it runs but that it ran *periodically*: the same atoms
    with `pbc` cleared must give a different energy, or the cell was being dropped on the floor
    and the molecular answer returned for a periodic system.
    """
    pytest.importorskip("ase")
    from ase import Atoms
    from am1_rs.ase import AM1

    numbers, positions = water_chain(2)
    cell = np.diag([9.0, 12.0, 12.0])

    periodic = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    periodic.calc = AM1(divide_conquer=True, buffer_radius=14.0)
    e_periodic = periodic.get_potential_energy()
    assert np.isfinite(e_periodic)

    isolated = Atoms(numbers=numbers, positions=positions)
    isolated.calc = AM1(divide_conquer=True, buffer_radius=14.0)
    e_isolated = isolated.get_potential_energy()

    print(f"    DC periodic {e_periodic:.6f} eV, isolated {e_isolated:.6f} eV")
    assert abs(e_periodic - e_isolated) > 1e-6, (
        "the periodic and isolated energies are identical, so the cell was ignored"
    )


def test_divide_and_conquer_molecular_dynamics_conserves_energy():
    """NVE under divide-and-conquer, which is the sharpest check its gradient can be given.

    The divide-and-conquer density is **not** variational, so the Hellmann–Feynman gradient is
    not automatically the derivative of the divide-and-conquer energy — the term that vanishes
    for a stationary density does not vanish here. Comparing against the full SCF gradient (as
    ``tests/divide_conquer.rs`` does) measures how close the two methods are; this measures
    something different and more practical: whether the force and the energy *this method
    reports* are consistent with each other, which is what decides whether dynamics runs.

    A finite buffer leaves a real residual, so the tolerance is looser than for the full SCF —
    but it is expressed against the energy actually exchanged during the run, so it cannot be
    met by a trajectory where nothing happens.
    """
    ase = pytest.importorskip("ase")
    from ase import Atoms, units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from am1_rs.ase import AM1

    numbers, positions = water_chain(5)
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.calc = AM1(divide_conquer=True, core_size=6, buffer_radius=20.0, smearing=0.0)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=200, rng=np.random.default_rng(4), force_temp=True
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

    total = np.asarray(total)
    swing = float(np.ptp(np.asarray(potential)))
    drift = abs(total[-1] - total[0])
    print(
        f"\n    DC NVE: drift {drift * 1e3:+.3f} meV against a potential swing of "
        f"{swing * 1e3:.1f} meV ({drift / len(numbers) * 1e3:.4f} meV/atom)"
    )
    assert swing > 1.0e-3, "nothing moved, so consistency was not tested"
    assert drift < 0.10 * swing, (
        f"the divide-and-conquer force and energy are inconsistent: drift {drift * 1e3:.3f} meV "
        f"against a swing of {swing * 1e3:.1f} meV"
    )


def test_ase_small_gap_warning_surfaces_as_a_python_warning():
    """The warning has to cross the binding, not be swallowed at it.

    The threshold is moved across a measured gap rather than a metallic system being
    constructed — see ``tests/divide_conquer.rs`` for why AM1 does not offer one to use here.
    Both directions are checked, because a warning that always fires is as useless as one that
    never does.
    """
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from am1_rs.ase import AM1

    numbers, positions = water_chain(4)
    gap = am1_rs.divide_conquer(numbers, positions, buffer_radius=12.0)["homo_lumo_gap_ev"]
    print(f"\n    measured gap {gap:.3f} eV")

    quiet = Atoms(numbers=numbers, positions=positions)
    quiet.calc = AM1(divide_conquer=True, buffer_radius=12.0, gap_warn_ev=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        quiet.get_potential_energy()

    loud = Atoms(numbers=numbers, positions=positions)
    loud.calc = AM1(divide_conquer=True, buffer_radius=12.0, gap_warn_ev=gap + 1.0)
    with pytest.warns(RuntimeWarning, match="gap"):
        loud.get_potential_energy()
