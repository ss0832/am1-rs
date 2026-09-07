# SPDX-License-Identifier: GPL-3.0-or-later

"""**The keys the native entry points promise, written down and checked.**

Every native function returns a plain `dict`. Nothing in Python declares what is in it, so a key
that stops being emitted — renamed, moved to another entry point, dropped in a refactor — is
found by the *first caller that reads it*, at the moment it prints, after the calculation has
already run. The user sees `am1-rs: 'electronic_ev'`: one word, no verb, no clue.

That has now happened twice in this project's history: three of five CLI modes read names the
bindings never emitted before 0.2.2, and the periodic `energy` mode read `electronic_ev` from
`pbc_point`, which did not emit it, before 0.2.3.

So the contract is a list rather than a convention. Each entry below names the keys a *consumer*
depends on — `am1_rs/__main__.py`, `am1_rs/ase.py`, or the documented API — and this file asserts
they are all there. Adding a key is free; removing one fails here, at the boundary, instead of in
someone's terminal.
"""

from __future__ import annotations

import pytest

pytest.importorskip("am1_rs")

from am1_rs import native  # noqa: E402

WATER_NUMBERS = [8, 1, 1]
WATER_POSITIONS = [
    [0.0, 0.0, 0.1173],
    [0.0, 0.7572, -0.4692],
    [0.0, -0.7572, -0.4692],
]

# A one-dimensional H2 chain: the cheapest structure with a cell.
CHAIN_NUMBERS = [1, 1]
CHAIN_POSITIONS = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6766]]
CHAIN_CELL = [[60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 3.0]]
CHAIN_PBC = [False, False, True]

#: The molecular energy report, printed by both CLI front ends' `_print_energy`.
ENERGY_KEYS = {
    "energy_ev",
    "energy_hartree",
    "electronic_ev",
    "core_ev",
    "heat_of_formation_kcal",
    "homo_ev",
    "lumo_ev",
    "dipole_debye",
    "dipole_magnitude",
    "charges",
    "iterations",
    "unrestricted",
}

#: The periodic energy report, printed by `_print_pbc_energy`.
PBC_ENERGY_KEYS = {
    "energy_ev",
    "energy_hartree",
    "electronic_ev",
    "core_ev",
    "charges",
    "fermi_energy_ev",
    "entropy_ev",
    "k_points",
    "iterations",
    "unrestricted",
    "max_image_overlap",
    "charged_cell_warning",
}


def _keys(result: dict) -> set[str]:
    assert isinstance(result, dict), f"expected a dict, got {type(result).__name__}"
    return set(result)


def test_single_point_carries_the_energy_report() -> None:
    assert ENERGY_KEYS <= _keys(native.single_point(WATER_NUMBERS, WATER_POSITIONS))


def test_gradient_carries_the_energy_report_and_the_forces() -> None:
    r = native.gradient(WATER_NUMBERS, WATER_POSITIONS)
    assert ENERGY_KEYS - {"iterations"} <= _keys(r)
    assert {
        "gradient_hartree_per_bohr",
        "gradient_ev_per_angstrom",
        "max_gradient_hartree_per_bohr",
    } <= _keys(r)


def test_optimize_carries_the_energy_report_and_the_geometry() -> None:
    r = native.optimize(WATER_NUMBERS, WATER_POSITIONS)
    assert ENERGY_KEYS - {"iterations"} <= _keys(r)
    # `iterations` is the optimizer's step count here, so the SCF's own is reported separately.
    # Both CLIs print the SCF one; reading `iterations` instead reported "3 iterations" for a
    # 16-iteration SCF.
    assert {"positions_angstrom", "converged", "iterations", "scf_iterations"} <= _keys(r)


def test_frequencies_carries_the_modes_and_what_was_projected_out() -> None:
    r = native.frequencies(WATER_NUMBERS, WATER_POSITIONS)
    assert {
        "frequencies_cm",
        "eigenvalues",
        "modes",
        "cartesian_displacements",
        "translation_rotation_overlap",
        "rigid_body_count",
        "rigid_body_frequencies_cm",
    } <= _keys(r)
    # The shapes have to agree with each other, or a consumer indexing one by the other's length
    # walks off the end. Since 0.2.3 there are `3N - rigid_body_count` modes, not `3N`.
    n = len(r["frequencies_cm"])
    assert n == 3 * len(WATER_NUMBERS) - r["rigid_body_count"] == 3
    assert len(r["eigenvalues"]) == n
    assert len(r["translation_rotation_overlap"]) == n
    assert len(r["modes"][0]) == n
    assert len(r["rigid_body_frequencies_cm"]) == r["rigid_body_count"]


def test_ir_spectrum_agrees_with_itself_on_the_mode_count() -> None:
    r = native.ir_spectrum(WATER_NUMBERS, WATER_POSITIONS)
    assert {
        "dipole_derivatives",
        "frequencies_cm",
        "intensities_km_per_mol",
        "mode_dipole_derivatives",
        "translation_rotation_overlap",
        "rigid_body_count",
    } <= _keys(r)
    n = len(r["frequencies_cm"])
    assert len(r["intensities_km_per_mol"]) == n
    assert len(r["translation_rotation_overlap"]) == n
    assert len(r["mode_dipole_derivatives"][0]) == n
    # The polar tensor keeps all 3N columns: it is a property of the geometry, not of the modes.
    assert len(r["dipole_derivatives"][0]) == 3 * len(WATER_NUMBERS)


def test_orbitals_carries_energies_coefficients_and_labels() -> None:
    r = native.orbitals(WATER_NUMBERS, WATER_POSITIONS)
    assert {
        "energies_hartree",
        "energies_ev",
        "coefficients",
        "n_occupied",
        "homo_ev",
        "lumo_ev",
        "unrestricted",
        "ao_labels",
    } <= _keys(r)
    nao = len(r["energies_hartree"])
    assert len(r["coefficients"]) == nao
    assert len(r["coefficients"][0]) == nao
    assert len(r["ao_labels"]) == nao


def test_unrestricted_orbitals_carry_the_beta_channel() -> None:
    r = native.orbitals([6, 1, 1, 1], [[0, 0, 0], [1.079, 0, 0], [-0.5395, 0.9344, 0], [-0.5395, -0.9344, 0]], multiplicity=2, reference="uhf")
    assert r["unrestricted"]
    assert {
        "beta_energies_hartree",
        "beta_energies_ev",
        "beta_coefficients",
        "beta_n_occupied",
        "homo_beta_ev",
        "lumo_beta_ev",
    } <= _keys(r)


def test_am1_bcc_carries_the_charges_types_and_mol2() -> None:
    r = native.am1_bcc(WATER_NUMBERS, WATER_POSITIONS)
    assert {"charges", "mulliken", "atom_types", "warnings", "mol2"} <= _keys(r)
    # The mol2 is rendered by the crate so both CLI front ends write the same bytes.
    assert r["mol2"].startswith("@<TRIPOS>MOLECULE")


def test_hessian_is_the_raw_second_derivative_matrix() -> None:
    r = native.hessian(WATER_NUMBERS, WATER_POSITIONS)
    assert {"hessian_hartree_per_bohr2", "hessian_ev_per_angstrom2", "ndof"} <= _keys(r)
    ndof = r["ndof"]
    assert ndof == 3 * len(WATER_NUMBERS)
    # The full 3N x 3N matrix, with nothing projected out of it: the rigid-body projection added
    # in 0.2.3 belongs to the frequency step and must not reach the Hessian. A projected matrix
    # would be singular in six directions, so a rigid translation is the test.
    h = r["hessian_hartree_per_bohr2"]
    assert len(h) == len(h[0]) == ndof
    translation = [1.0 if k % 3 == 0 else 0.0 for k in range(ndof)]
    curvature = sum(
        translation[i] * h[i][j] * translation[j] for i in range(ndof) for j in range(ndof)
    )
    # Translational invariance makes this zero *analytically*, which is not the point: the point
    # is that the matrix still has the row and column structure of a Cartesian Hessian.
    assert abs(curvature) < 1.0e-6
    assert all(abs(h[i][j] - h[j][i]) < 1.0e-9 for i in range(ndof) for j in range(ndof))


def test_pbc_point_carries_the_periodic_energy_report() -> None:
    r = native.pbc_point(
        CHAIN_NUMBERS, CHAIN_POSITIONS, CHAIN_CELL, CHAIN_PBC, kpts=(1, 1, 4)
    )
    assert PBC_ENERGY_KEYS <= _keys(r)
    assert {"forces_ev_per_angstrom", "stress_voigt", "stress_matrix", "n_periodic"} <= _keys(r)
    # A bound system's chemical potential is below vacuum. Folding `max` from zero — which is
    # what 0.2.2 did — clamps it to exactly 0 for every such system.
    assert r["fermi_energy_ev"] < 0.0, "the Fermi level is clamped at zero again"


def test_pbc_optimize_carries_the_relaxed_structure() -> None:
    r = native.pbc_optimize(
        CHAIN_NUMBERS, CHAIN_POSITIONS, CHAIN_CELL, CHAIN_PBC, kpts=(1, 1, 4), max_iter=5
    )
    assert PBC_ENERGY_KEYS <= _keys(r)
    assert {
        "positions_angstrom",
        "cell_angstrom",
        "pbc",
        "converged",
        "iterations",
        "scf_iterations",
        "stress_voigt",
        "pressure",
        "max_force_ev_per_bohr",
    } <= _keys(r)
    # With `relax_cell` off the cell must come back as it went in — to within the Angstrom to
    # Bohr round trip, which is one ulp and not something the optimizer did.
    flat = [v for row in r["cell_angstrom"] for v in row]
    assert flat == pytest.approx([v for row in CHAIN_CELL for v in row], rel=1e-12, abs=1e-12)
    assert r["pbc"] == CHAIN_PBC


def test_phonons_carries_the_bands_and_the_sum_rule() -> None:
    r = native.phonons(
        CHAIN_NUMBERS,
        CHAIN_POSITIONS,
        CHAIN_CELL,
        CHAIN_PBC,
        supercell=(1, 1, 2),
        q_points=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
    )
    assert {
        "q_points",
        "frequencies_cm",
        "supercell",
        "commensurate_q",
        "acoustic_sum_rule_error_before",
        "acoustic_sum_rule_error",
    } <= _keys(r)
    assert len(r["frequencies_cm"]) == 2
    # Imposing the rule is supposed to make it hold.
    assert r["acoustic_sum_rule_error"] <= r["acoustic_sum_rule_error_before"]


def test_molden_writes_a_gaussian_basis_by_default() -> None:
    # Section headers are matched as **whole lines**. Both files mention the other section by
    # name in their header note, and a substring test would call that a section — which is a
    # mistake a real parser can make too, and is why the writer keeps its note unbracketed at
    # the start of a line.
    def sections(text: str) -> set[str]:
        return {line.strip() for line in text.splitlines() if line.startswith("[")}

    text = native.molden(WATER_NUMBERS, WATER_POSITIONS)
    assert text.startswith("[Molden Format]")
    assert {"[Atoms] Angs", "[GTO]", "[MO]"} <= sections(text)
    assert "[STO]" not in sections(text)
    # And the legacy section is still reachable.
    sto = native.molden(WATER_NUMBERS, WATER_POSITIONS, basis="sto")
    assert "[STO]" in sections(sto)
    assert "[GTO]" not in sections(sto)


def test_constants_carry_every_conversion_the_front_ends_use() -> None:
    c = native.constants()
    assert {
        "hartree_to_ev",
        "ev_to_hartree",
        "angstrom_to_bohr",
        "bohr_to_angstrom",
        "ev_to_kcal",
        "kcal_to_ev",
        "au_dipole_to_debye",
        "ir_intensity_km_per_mol",
        "e_to_debye_per_angstrom",
    } <= set(c)
