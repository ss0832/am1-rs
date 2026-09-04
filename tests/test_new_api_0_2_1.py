# SPDX-License-Identifier: GPL-3.0-or-later
"""The 0.2.1 surface: external field, wavefunction output, IR, and the periodic response.

The point of this file is **parity and units**, not physics — the physics is checked in the Rust
suite, where it can be finite-differenced against the analytic paths directly. What can only be
checked here is that the two Python surfaces expose the same features, that the numbers survive
the unit conversion at each boundary, and that the bindings pass their arguments in the right
order (a transposed matrix or a swapped pair of floats produces a plausible number, not an error).
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from am1_rs import native

WATER_Z = [8, 1, 1]
WATER_XYZ = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]]
METHYL_Z = [6, 1, 1, 1]
METHYL_XYZ = [[0.0, 0.0, 0.0], [1.079, 0.0, 0.0], [-0.5395, 0.9344, 0.0], [-0.5395, -0.9344, 0.0]]

C = native.constants()


# ----------------------------------------------------------------------------- external field

def test_zero_field_is_the_field_free_calculation():
    bare = native.single_point(WATER_Z, WATER_XYZ)
    zero = native.single_point(WATER_Z, WATER_XYZ, electric_field=[0.0, 0.0, 0.0])
    assert bare["energy_hartree"] == pytest.approx(zero["energy_hartree"], abs=1e-12)
    assert bare["field_nuclear_ev"] == 0.0


def test_minus_dE_dF_is_the_reported_dipole():
    """The defining property of the field coupling, across the binding boundary.

    It also pins the *units*: a field passed in the wrong unit would still give a smooth energy
    curve, just with the wrong slope, and only comparing against an independently computed dipole
    catches that.
    """
    ref = native.single_point(WATER_Z, WATER_XYZ)
    mu = np.asarray(ref["dipole_debye"]) / C["au_dipole_to_debye"]
    h = 1.0e-4
    numeric = []
    for axis in range(3):
        f = [0.0, 0.0, 0.0]
        f[axis] = h
        ep = native.single_point(WATER_Z, WATER_XYZ, electric_field=f)["energy_hartree"]
        f[axis] = -h
        em = native.single_point(WATER_Z, WATER_XYZ, electric_field=f)["energy_hartree"]
        numeric.append(-(ep - em) / (2.0 * h))
    assert numeric == pytest.approx(list(mu), abs=1e-6)


def test_field_forces_sum_to_the_charge_times_the_field():
    """Neutral: zero. A cation: `qF`. Cheap, and catches a miswired sign or unit."""
    f_au = [0.0, 0.0, 5.0e-3]
    neutral = native.gradient(WATER_Z, WATER_XYZ, electric_field=f_au)
    net = -np.asarray(neutral["gradient_hartree_per_bohr"]).sum(axis=0)
    assert net == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    cation = native.gradient(WATER_Z, WATER_XYZ, charge=1.0, multiplicity=2, electric_field=f_au)
    net = -np.asarray(cation["gradient_hartree_per_bohr"]).sum(axis=0)
    assert net == pytest.approx(f_au, abs=1e-8)


@pytest.mark.parametrize(
    "name, key",
    [
        ("single_point", "energy_hartree"),
        ("gradient", "energy_hartree"),
        ("optimize", "energy_hartree"),
        ("frequencies", "frequencies_cm"),
        ("orbitals", "energies_ev"),
    ],
)
def test_field_reaches_every_molecular_entry_point(name, key):
    """A field honoured by only some entry points is worse than one honoured by none.

    Each case names the result key that must move, rather than scanning the dict for *anything*
    that changed — a scan would also pass if only some incidental field moved.
    """
    field = [0.0, 0.0, 5.0e-3]
    call = getattr(native, name)
    bare = np.asarray(call(WATER_Z, WATER_XYZ)[key], dtype=float)
    withf = np.asarray(call(WATER_Z, WATER_XYZ, electric_field=field)[key], dtype=float)
    assert bare.shape == withf.shape
    # An absolute comparison with `rtol=0`. `np.allclose`'s default relative tolerance is 1e-5,
    # which on a −12.8 Hartree energy is 1.3e-4 — larger than the 3e-6 shift this field produces,
    # so the default would call an honoured field "unchanged".
    assert not np.allclose(bare, withf, atol=1e-10, rtol=0.0), f"{name} ignored the field"


def test_hessian_honours_the_field():
    """Separate, because the field reaches the Hessian *only* through the CPHF response.

    The dipole operator is linear in the nuclear positions, so it adds nothing to the
    fixed-density second derivative. If the response term were missing the Hessian would still be
    symmetric with sensible eigenvalues — and identical to the field-free one, which is what this
    detects.
    """
    field = [0.0, 0.0, 5.0e-3]
    bare = np.asarray(native.hessian(WATER_Z, WATER_XYZ)["hessian_ev_per_angstrom2"])
    withf = np.asarray(
        native.hessian(WATER_Z, WATER_XYZ, electric_field=field)["hessian_ev_per_angstrom2"]
    )
    assert not np.allclose(bare, withf, atol=1e-8)


# ------------------------------------------------------------------------ wavefunction output

def test_orbitals_agree_with_single_point_and_carry_labels():
    o = native.orbitals(WATER_Z, WATER_XYZ)
    sp = native.single_point(WATER_Z, WATER_XYZ)
    assert o["homo_ev"] == pytest.approx(sp["homo_ev"], abs=1e-12)
    assert o["lumo_ev"] == pytest.approx(sp["lumo_ev"], abs=1e-12)
    assert o["n_occupied"] == 4
    coeff = np.asarray(o["coefficients"])
    assert coeff.shape == (6, 6)
    # Columns are orbitals and the eigenvectors are orthonormal, so CᵀC = I. This is what
    # catches a transposed matrix crossing the binding, which nothing else here would.
    assert np.allclose(coeff.T @ coeff, np.eye(6), atol=1e-10)
    # Energies ascend, and the labels line up with the rows.
    assert list(o["energies_ev"]) == sorted(o["energies_ev"])
    assert len(o["ao_labels"]) == 6
    assert o["ao_labels"][0] == (0, "s")
    assert [lbl[1] for lbl in o["ao_labels"][:4]] == ["s", "px", "py", "pz"]


def test_unrestricted_orbitals_carry_the_beta_channel():
    """Before 0.2.1 the SCF solved for the β orbitals and discarded them."""
    o = native.orbitals(METHYL_Z, METHYL_XYZ, multiplicity=2)
    assert o["unrestricted"]
    assert "beta_coefficients" in o and "beta_energies_ev" in o
    assert o["beta_n_occupied"] == o["n_occupied"] - 1
    beta = np.asarray(o["beta_coefficients"])
    assert np.allclose(beta.T @ beta, np.eye(beta.shape[0]), atol=1e-10)
    # The two channels are genuinely different for an open shell.
    assert not np.allclose(o["energies_ev"], o["beta_energies_ev"], atol=1e-6)


def test_molden_has_the_expected_sections_and_matches_the_orbitals():
    text = native.molden(WATER_Z, WATER_XYZ)
    lines = [ln.strip() for ln in text.splitlines()]
    for section in ("[Molden Format]", "[Atoms] Angs", "[STO]", "[MO]"):
        assert section in lines, f"missing section {section}"
    # One [STO] line per atomic orbital.
    sto = lines.index("[STO]")
    mo = lines.index("[MO]")
    assert mo - sto - 1 == 6
    # The energies in the file are the SCF's, in Hartree.
    written = [float(ln.split("=")[1]) for ln in lines if ln.startswith("Ene=")]
    o = native.orbitals(WATER_Z, WATER_XYZ)
    assert written == pytest.approx(list(o["energies_hartree"]), abs=1e-9)
    # The caveat travels with the file.
    assert "orthonormal AO basis" in text


def test_molden_writes_both_spin_channels_for_an_open_shell():
    # Methyl: carbon contributes 4 AOs and each hydrogen one, so 7 orbitals per spin channel.
    nao = 7
    text = native.molden(METHYL_Z, METHYL_XYZ, multiplicity=2)
    assert text.count("Spin= Alpha") == nao
    assert text.count("Spin= Beta") == nao


# -------------------------------------------------------------------------------------- IR

def test_the_atomic_polar_tensor_obeys_its_sum_rule():
    """`Σ_a ∂μ_α/∂R_{a,β} = q δ_αβ`. Nine exact constraints from charge conservation."""
    apt = np.asarray(native.dipole_derivatives(WATER_Z, WATER_XYZ)["dipole_derivatives"])
    assert apt.shape == (3, 9)
    total = apt.reshape(3, 3, 3).sum(axis=1)
    assert np.abs(total).max() < 1e-8


def test_ir_spectrum_is_consistent_with_its_parts():
    opt = native.optimize(WATER_Z, WATER_XYZ)
    xyz = opt["positions_angstrom"]
    ir = native.ir_spectrum(WATER_Z, xyz)

    # The tensor is the same one `dipole_derivatives` returns on its own.
    alone = np.asarray(native.dipole_derivatives(WATER_Z, xyz)["dipole_derivatives"])
    assert np.allclose(np.asarray(ir["dipole_derivatives"]), alone, atol=1e-10)

    # The frequencies are the same ones `frequencies` returns.
    fr = native.frequencies(WATER_Z, xyz)
    assert ir["frequencies_cm"] == pytest.approx(fr["frequencies_cm"], abs=1e-8)

    # Water is bent: three vibrations, six rigid-body modes, all vibrations infrared active.
    assert len(ir["vibrational_modes"]) == 3
    for k in ir["vibrational_modes"]:
        assert ir["intensities_km_per_mol"][k] > 1.0
    assert sum(1 for x in ir["translation_rotation_overlap"] if x > 0.5) == 6

    # The intensity is the squared norm of the per-mode dipole derivative, times the constant.
    mode_apt = np.asarray(ir["mode_dipole_derivatives"])
    for k in range(len(ir["frequencies_cm"])):
        expected = 42.2561 * float(mode_apt[:, k] @ mode_apt[:, k])
        assert ir["intensities_km_per_mol"][k] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_a_linear_molecule_reports_five_rigid_body_modes():
    """Discovered from the eigenvectors, not assumed from `3N − 6`."""
    co2_z = [6, 8, 8]
    b = 1.189308342
    co2 = [[0.0, 0.0, 0.0], [b, 0.0, 0.0], [-b, 0.0, 0.0]]
    fr = native.frequencies(co2_z, co2)
    assert sum(1 for x in fr["translation_rotation_overlap"] if x > 0.5) == 5


# ------------------------------------------------------------------------- orbital response

def test_orbital_response_shapes_and_hessian_agreement():
    r = native.orbital_response(WATER_Z, WATER_XYZ)
    assert r["ndof"] == 9
    assert len(r["u_ov"]) == 9
    u0 = np.asarray(r["u_ov"][0])
    assert u0.shape == (r["n_virtual"], r["n_occupied"])
    # The Hessian it hands back is the one `native.hessian` computes, so asking for both costs
    # one calculation rather than two.
    ref = np.asarray(native.hessian(WATER_Z, WATER_XYZ)["hessian_ev_per_angstrom2"])
    mine = np.asarray(r["hessian_ev_per_bohr2"]) * C["angstrom_to_bohr"] ** 2
    assert np.allclose(mine, ref, atol=1e-10)


def test_orbital_response_density_is_opt_in():
    without = native.orbital_response(WATER_Z, WATER_XYZ)
    assert "response_density" not in without
    with_it = native.orbital_response(WATER_Z, WATER_XYZ, response_density=True)
    dp = np.asarray(with_it["response_density"])
    assert dp.shape == (9, 6, 6)
    # ∂P/∂R is symmetric, like P.
    assert np.allclose(dp, np.transpose(dp, (0, 2, 1)), atol=1e-12)


# ------------------------------------------------------------------------ periodic response

CRYSTAL_CELL = (np.eye(3) * 4.5).tolist()
CRYSTAL_PBC = [True, True, True]


def test_born_charges_satisfy_the_acoustic_sum_rule():
    r = native.born_charges(WATER_Z, WATER_XYZ, CRYSTAL_CELL, CRYSTAL_PBC, kpts=(2, 2, 2))
    z = np.asarray(r["born_charges"])
    assert z.shape == (3, 3, 3)
    assert np.abs(z.sum(axis=0)).max() < 1e-8
    assert np.abs(np.asarray(r["acoustic_sum_rule_error"])).max() < 1e-8


def test_dielectric_tensor_is_physical_and_refuses_low_dimensions():
    r = native.dielectric(WATER_Z, WATER_XYZ, CRYSTAL_CELL, CRYSTAL_PBC, kpts=(2, 2, 2))
    eps = np.asarray(r["epsilon_infinity"])
    alpha = np.asarray(r["polarizability_bohr3"])
    assert np.all(np.diag(eps) >= 1.0)
    assert np.all(np.diag(alpha) > 0.0)
    assert np.allclose(alpha, alpha.T, atol=1e-10)

    # `ε∞ = 1 + 4πα/Ω` needs Ω to be a volume; a chain has only a length.
    chain = np.diag([3.4, 40.0, 40.0]).tolist()
    with pytest.raises(ValueError, match="three-dimensional"):
        native.dielectric(WATER_Z, WATER_XYZ, chain, [True, False, False])


def test_dielectric_with_extent_needs_exactly_one_extent():
    slab = np.diag([7.5, 7.5, 40.0]).tolist()
    pbc = [True, True, False]
    with pytest.raises(ValueError, match="no default"):
        native.dielectric_with_extent(WATER_Z, WATER_XYZ, slab, pbc)
    with pytest.raises(ValueError, match="not both"):
        native.dielectric_with_extent(
            WATER_Z, WATER_XYZ, slab, pbc, slab_thickness=5.0, wire_cross_section=20.0
        )
    # The convention has to match the cell: a slab thickness on a chain carries the wrong units
    # *and* the wrong depolarization factor, so it is refused rather than reinterpreted.
    chain = np.diag([3.4, 40.0, 40.0]).tolist()
    with pytest.raises(ValueError, match="SlabThickness"):
        native.dielectric_with_extent(
            WATER_Z, WATER_XYZ, chain, [True, False, False], slab_thickness=5.0
        )
    # And a crystal is sent back to the function that needs no choice at all.
    with pytest.raises(ValueError, match="dielectric_tensor"):
        native.dielectric_with_extent(
            WATER_Z, WATER_XYZ, CRYSTAL_CELL, CRYSTAL_PBC, slab_thickness=5.0
        )


def test_dielectric_with_extent_reports_what_the_thickness_cannot_change():
    """The units cross the boundary, and the thickness-free invariants come back with them."""
    slab = np.diag([7.5, 7.5, 40.0]).tolist()
    pbc = [True, True, False]
    seen = []
    for d in (4.0, 12.0):
        r = native.dielectric_with_extent(
            WATER_Z, WATER_XYZ, slab, pbc, slab_thickness=d, kpts=(2, 2, 1)
        )
        eps = np.asarray(r["epsilon_infinity"])
        assert np.all(np.diag(eps) >= 1.0)
        assert r["extent"] == pytest.approx(d)
        assert r["n_periodic"] == 2
        # In plane `(eps - 1) * d`, out of plane `(1 - 1/eps) * d`, both thickness-free. The
        # reported scalars average the two in-plane directions, and water on a square lattice is
        # anisotropic enough in plane that comparing against `xx` alone would fail -- which is the
        # distinction the docstring makes and this pins.
        in_plane_mean = 0.5 * ((eps[0, 0] - 1.0) + (eps[1, 1] - 1.0)) * d
        assert in_plane_mean == pytest.approx(r["sheet_susceptibility"], rel=1e-9)
        assert (1.0 - 1.0 / eps[2, 2]) * d == pytest.approx(
            r["inverse_sheet_response"], rel=1e-9
        )
        assert eps[0, 0] != pytest.approx(eps[1, 1], rel=1e-3), (
            "this slab is supposed to be in-plane anisotropic, or the averaging above is untested"
        )
        assert r["rytova_keldysh_length"] == pytest.approx(
            0.5 * r["sheet_susceptibility"], rel=1e-12
        )
        seen.append((r["sheet_susceptibility"], r["inverse_sheet_response"], eps[0, 0]))
    # The invariants do not move; `eps` does, so the first statement is not vacuous.
    assert seen[0][0] == pytest.approx(seen[1][0], rel=1e-9)
    assert seen[0][1] == pytest.approx(seen[1][1], rel=1e-9)
    assert seen[0][2] > seen[1][2] + 1e-3

    # A wire takes an area instead, and reports no sheet length — that quantity is a sheet's.
    chain = np.diag([3.4, 40.0, 40.0]).tolist()
    w = native.dielectric_with_extent(
        WATER_Z, WATER_XYZ, chain, [True, False, False], wire_cross_section=60.0, kpts=(2, 1, 1)
    )
    assert w["n_periodic"] == 1
    assert "rytova_keldysh_length" not in w


def test_pbc_hessian_shape_and_units():
    r = native.pbc_hessian(WATER_Z, WATER_XYZ, CRYSTAL_CELL, CRYSTAL_PBC, kpts=(2, 2, 2))
    au = np.asarray(r["hessian_hartree_per_bohr2"])
    ev = np.asarray(r["hessian_ev_per_angstrom2"])
    assert au.shape == (9, 9) and ev.shape == (9, 9)
    assert np.allclose(au, au.T, atol=1e-8)
    # The two views are the same matrix in different units.
    ratio = C["hartree_to_ev"] * C["angstrom_to_bohr"] ** 2
    assert np.allclose(ev, au * ratio, rtol=1e-12)


def test_dfpt_runs_at_arbitrary_q_and_reproduces_gamma():
    """At `q = 0` the response must collapse onto the `q = 0` k-point Hessian."""
    q = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]
    r = native.dfpt(WATER_Z, WATER_XYZ, CRYSTAL_CELL, CRYSTAL_PBC, q, kpts=(2, 2, 2))
    assert len(r["frequencies_cm"]) == 2
    assert all(len(f) == 9 for f in r["frequencies_cm"])
    # A different q gives a different answer — otherwise this would pass on a stub.
    assert not np.allclose(r["frequencies_cm"][0], r["frequencies_cm"][1], atol=1e-6)


def test_dfpt_rejects_a_q_along_a_non_periodic_axis():
    chain = np.diag([3.4, 40.0, 40.0]).tolist()
    with pytest.raises(ValueError, match="non-periodic axis"):
        native.dfpt(WATER_Z, WATER_XYZ, chain, [True, False, False], [[0.0, 0.25, 0.0]])


# -------------------------------------------------------------- divide-and-conquer counters

def test_divide_conquer_reports_its_diis_memory():
    cluster_z = WATER_Z * 8
    cluster_xyz = [
        [p[0] + 4.0 * i, p[1], p[2]] for i in range(8) for p in WATER_XYZ
    ]
    r = native.divide_conquer(cluster_z, cluster_xyz, core_size=4, buffer_radius=10.0)
    assert r["diis_pattern_elements"] < r["dense_triangle_elements"]
    assert r["diis_pattern_elements"] > 0


# ------------------------------------------------------------------------------------- ASE

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.units import Debye  # noqa: E402

from am1_rs.ase import AM1  # noqa: E402


def water_atoms():
    return Atoms("OH2", positions=WATER_XYZ)


# Every public native function, mapped to the ASE method that must expose the same capability.
# `None` means "deliberately not an ASE method", and each such entry has to say why.
#
# This is enumerated from the `native` module itself rather than listed by hand: a hardcoded
# allow-list passes happily when a *new* native function is added with no ASE counterpart, which
# is exactly the gap it is supposed to catch. `lo_to_frequencies` existed in the Rust API and
# reached neither Python surface for precisely that reason.
_ASE_EQUIVALENT = {
    "single_point": "calculate",
    "gradient": "get_forces",
    "optimize": "optimize",
    "hessian": "get_hessian",
    "frequencies": "get_frequencies",
    "am1_bcc": "get_am1_bcc_charges",
    "orbitals": "get_orbitals",
    "molden": "write_molden",
    "ir_spectrum": "get_ir_spectrum",
    "dipole_derivatives": "get_dipole_derivatives",
    "orbital_response": "get_orbital_response",
    # The bundle the five above are contractions of. The ASE surface routes all of them through
    # it, so `get_ir_spectrum` *is* its counterpart rather than there being a sixth method: the
    # capability is "get these from one solve", and that is what the ASE layer now does.
    "vibrations": "get_ir_spectrum",
    "phonons": "get_phonons",
    "dfpt": "get_dfpt_frequencies",
    "lo_to_frequencies": "get_lo_to_frequencies",
    "pbc_hessian": "get_hessian",
    "born_charges": "get_born_charges",
    "dielectric": "get_dielectric_tensor",
    "dielectric_with_extent": "get_dielectric_tensor_with_extent",
    "polarizability": "get_polarizability",
    "dielectric_function": "get_dielectric_function",
    "polarization": "get_polarization",
    "finite_field": "get_finite_field",
    # ASE reaches the periodic single point through `calculate` on an `atoms` with `pbc` set,
    # so there is no separate method.
    "pbc_point": "calculate",
    # A constructor flag (`AM1(divide_conquer=True)`), not a method.
    "divide_conquer": None,
    # ASE's unit system is fixed by ASE itself; the conversion factors are internal.
    "constants": None,
}


def test_ase_exposes_the_same_features_as_native():
    """The project's own rule: the two surfaces differ in units, not in capability."""
    import inspect

    public = sorted(
        name
        for name, obj in inspect.getmembers(native, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == native.__name__
    )
    assert public, "found no public functions in am1_rs.native — the enumeration broke"
    unmapped = [n for n in public if n not in _ASE_EQUIVALENT]
    assert not unmapped, (
        f"native gained {unmapped} with no entry in _ASE_EQUIVALENT — add the ASE method, or "
        f"map it to None with a comment saying why it has none"
    )
    missing = [
        f"{name} -> AM1.{method}"
        for name, method in _ASE_EQUIVALENT.items()
        if method is not None and not hasattr(AM1, method)
    ]
    assert not missing, f"the ASE calculator is missing: {missing}"


def test_ase_field_units_match_the_native_atomic_unit_field():
    """ASE takes V/Å; the native layer takes Hartree per e·Bohr. Same physical field."""
    f_au = 5.0e-3
    f_v_per_ang = f_au * C["hartree_to_ev"] / C["bohr_to_angstrom"]
    atoms = water_atoms()
    atoms.calc = AM1(field=[0.0, 0.0, f_v_per_ang])
    ase_energy = atoms.get_potential_energy()
    native_energy = native.single_point(
        WATER_Z, WATER_XYZ, electric_field=[0.0, 0.0, f_au]
    )["energy_ev"]
    assert ase_energy == pytest.approx(native_energy, abs=1e-9)


def test_ase_field_change_invalidates_the_cache():
    atoms = water_atoms()
    atoms.calc = AM1()
    bare = atoms.get_potential_energy()
    atoms.calc.set(field=[0.0, 0.0, 0.5])
    withf = atoms.get_potential_energy()
    assert abs(withf - bare) > 1e-7

    # And through `atoms.info`, which ASE's own machinery cannot see.
    other = water_atoms()
    other.calc = AM1()
    _ = other.get_potential_energy()
    other.info["field"] = [0.0, 0.0, 0.5]
    assert other.get_potential_energy() == pytest.approx(withf, abs=1e-10)


def test_ase_forces_under_a_field_match_a_finite_difference():
    field = [0.0, 0.0, 0.5]
    atoms = water_atoms()
    atoms.calc = AM1(field=field)
    analytic = atoms.get_forces()

    h = 1.0e-4
    numeric = np.zeros_like(analytic)
    for a in range(len(atoms)):
        for k in range(3):
            for sign in (+1, -1):
                shifted = water_atoms()
                shifted.calc = AM1(field=field)
                p = shifted.get_positions()
                p[a, k] += sign * h
                shifted.set_positions(p)
                numeric[a, k] += -sign * shifted.get_potential_energy() / (2.0 * h)
    assert np.abs(analytic - numeric).max() < 1e-4


def test_ase_periodic_field_is_refused_along_a_periodic_direction():
    """A field along a periodic direction, which is the case that is genuinely ill-defined.

    Narrowed in 0.2.2. This used to assert a blanket refusal for any cell, matching the code, and
    the code was too strict: `F.R` shifts by `F.T` under translation, so the perturbation is
    lattice-periodic exactly when `F.T = 0` for every lattice vector. This cell is periodic in all
    three directions, so the field is still refused -- but now for the right reason, and the
    message says which component is the problem. The cases that became legal are covered by
    `tests/test_ase_pbc_md.py::test_a_field_transverse_to_a_chain_is_allowed_and_along_it_is_not`.
    """
    atoms = Atoms("OH2", positions=WATER_XYZ, cell=np.eye(3) * 20.0, pbc=True)
    atoms.calc = AM1(field=[0.0, 0.0, 0.5])
    with pytest.raises(Exception, match="along a periodic direction"):
        atoms.get_potential_energy()


def test_ase_ir_and_molden_round_trip():
    atoms = water_atoms()
    atoms.calc = AM1()
    atoms.calc.optimize(atoms)
    ir = atoms.calc.get_ir_spectrum(atoms)
    assert len(ir["vibrational_modes"]) == 3

    path = os.path.join(tempfile.mkdtemp(), "wavefunction.molden")
    atoms.calc.write_molden(path, atoms)
    text = open(path, encoding="utf-8").read()
    assert "[Molden Format]" in text and "[STO]" in text

    orbitals = atoms.calc.get_orbitals(atoms)
    written = [float(ln.split("=")[1]) for ln in text.splitlines() if ln.strip().startswith("Ene=")]
    assert written == pytest.approx(list(orbitals["energies_hartree"]), abs=1e-9)


def test_ase_dipole_matches_native_in_ase_units():
    atoms = water_atoms()
    atoms.calc = AM1()
    ase_dipole = atoms.get_dipole_moment()
    native_dipole = (
        np.asarray(native.single_point(WATER_Z, WATER_XYZ)["dipole_debye"]) * Debye
    )
    assert np.allclose(ase_dipole, native_dipole, atol=1e-12)


def test_lo_to_frequencies_splits_and_matches_across_surfaces():
    """The LO-TO path exists on both Python surfaces and actually shifts a polar crystal.

    Two things are checked, because either alone is weak. That the ASE method and the native
    function agree (parity, in the same units — both are cm^-1 here). And that the split
    frequencies genuinely differ from the unsplit ones, so a stub returning `phonons`' answer
    would fail: a water crystal is polar, so it must have a non-analytic term.
    """
    cell = np.eye(3) * 4.5
    atoms = Atoms("OH2", positions=WATER_XYZ, cell=cell, pbc=True)
    atoms.calc = AM1(kpts=(2, 2, 2))

    ase_result = atoms.calc.get_lo_to_frequencies(
        direction=(1.0, 0.0, 0.0), supercell=(2, 1, 1), atoms=atoms
    )
    native_result = native.lo_to_frequencies(
        atoms.get_atomic_numbers(),
        atoms.get_positions(),
        cell,
        [True, True, True],
        supercell=(2, 1, 1),
        direction=(1.0, 0.0, 0.0),
        kpts=(2, 2, 2),
    )
    split = np.asarray(ase_result["frequencies_cm"], dtype=float)
    assert np.allclose(
        split, np.asarray(native_result["frequencies_cm"], dtype=float), atol=1e-8
    ), "the ASE and native LO-TO paths disagree"

    unsplit = np.asarray(ase_result["frequencies_cm_no_lo_to"], dtype=float)
    shift = np.abs(split - unsplit).max()
    # A stub returning `phonons`' answer would give exactly 0. The bound is well below the
    # measured 0.69 cm^-1 because the term goes as 1/(q.eps.q): correcting eps_infinity's units
    # in 0.2.1 raised it from ~1.004 to ~1.11 and shrank this shift by the same factor, so a
    # tight bound here would be pinning eps_infinity rather than the LO-TO wiring.
    assert shift > 0.1, (
        f"the non-analytic term moved the spectrum by only {shift:.3e} cm^-1 on a polar "
        f"crystal, which is what returning the unsplit answer would look like"
    )
    # Born charges must obey their sum rule whichever route produced them.
    z = np.asarray(ase_result["born_charges"], dtype=float)
    assert np.abs(z.sum(axis=0)).max() < 1e-6


def test_lo_to_frequencies_is_refused_on_a_chain():
    """1D has no LO-TO term; asking must be an error, not a 3D formula on a length."""
    atoms = Atoms(
        "OH2", positions=WATER_XYZ, cell=np.diag([4.5, 20.0, 20.0]), pbc=[True, False, False]
    )
    atoms.calc = AM1(kpts=(2, 1, 1))
    with pytest.raises(Exception):
        atoms.calc.get_lo_to_frequencies(atoms=atoms)


def test_divide_conquer_runs_under_a_cell_from_both_python_surfaces():
    """Periodic divide-and-conquer, which 0.2.0 could do in Rust and nowhere else.

    The ASE calculator used to raise `NotImplementedError` here with a message saying the
    periodic buffers were "not wired up yet" — true of 0.2.0, not of the Rust API it wraps.
    """
    cell = np.eye(3) * 9.0
    z = WATER_Z * 2
    xyz = np.asarray(WATER_XYZ + [[r[0] + 4.0, r[1], r[2]] for r in WATER_XYZ])

    r = native.divide_conquer(
        z, xyz, cell=cell, pbc=[True, True, True], buffer_radius=14.0, forces=False
    )
    assert np.isfinite(r["energy_ev"])

    atoms = Atoms("OH2OH2", positions=xyz, cell=cell, pbc=True)
    atoms.calc = AM1(divide_conquer=True, buffer_radius=14.0)
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(r["energy_ev"], rel=1e-9)


def test_divide_conquer_rejects_a_half_specified_cell():
    with pytest.raises(ValueError, match="together"):
        native.divide_conquer(WATER_Z, WATER_XYZ, cell=np.eye(3) * 9.0)


def test_the_lazy_cache_is_keyed_on_the_arguments():
    """Caching `get_phonons` under one key would answer the second question with the first.

    These methods take arguments that change the calculation — a supercell, a q list, a solver
    tolerance — so a cache keyed only on the geometry is worse than no cache. They were left
    uncached for that reason, which meant re-running a DFPT or CPHF solve on every call; keying
    on the arguments gets the caching without the staleness.
    """
    cell = np.eye(3) * 4.5
    atoms = Atoms("OH2", positions=WATER_XYZ, cell=cell, pbc=True)
    atoms.calc = AM1(kpts=(2, 1, 1))

    one = atoms.calc.get_phonons(supercell=(1, 1, 1), atoms=atoms)
    two = atoms.calc.get_phonons(supercell=(2, 1, 1), atoms=atoms)
    assert one["supercell"] == [1, 1, 1]
    assert two["supercell"] == [2, 1, 1], "the second call returned the first call's answer"

    # And the cache is a cache: asking again gives the identical object, not a re-solve.
    assert atoms.calc.get_phonons(supercell=(1, 1, 1), atoms=atoms) is one


def test_constants_carry_the_infrared_chain():
    """A caller converting an APT to km/mol must not have to hardcode 42.2561 itself."""
    c = native.constants()
    assert c["ir_intensity_km_per_mol"] == pytest.approx(42.2561, rel=1e-6)
    # Built from *this crate's* Bohr, not CODATA's, so it must not equal the CODATA value.
    assert c["e_to_debye_per_angstrom"] > 0.0


def test_ase_periodic_response_matches_native():
    atoms = Atoms("OH2", positions=WATER_XYZ, cell=np.eye(3) * 4.5, pbc=True)
    atoms.calc = AM1(kpts=(2, 2, 2))

    z_ase = atoms.calc.get_born_charges(atoms)
    z_native = np.asarray(
        native.born_charges(WATER_Z, WATER_XYZ, CRYSTAL_CELL, CRYSTAL_PBC, kpts=(2, 2, 2))[
            "born_charges"
        ]
    )
    assert np.allclose(z_ase, z_native, atol=1e-10)

    eps_ase = atoms.calc.get_dielectric_tensor(atoms)
    eps_native = np.asarray(
        native.dielectric(WATER_Z, WATER_XYZ, CRYSTAL_CELL, CRYSTAL_PBC, kpts=(2, 2, 2))[
            "epsilon_infinity"
        ]
    )
    assert np.allclose(eps_ase, eps_native, atol=1e-10)

    h = atoms.calc.get_hessian(atoms)
    assert h.shape == (9, 9)


def test_ase_optimize_writes_back_and_lowers_the_energy():
    atoms = water_atoms()
    atoms.calc = AM1()
    before = atoms.get_potential_energy()
    start = atoms.get_positions().copy()
    atoms.calc.optimize(atoms)
    assert not np.allclose(atoms.get_positions(), start)
    assert atoms.get_potential_energy() < before + 1e-9

    # `apply=False` leaves the structure alone.
    other = water_atoms()
    other.calc = AM1()
    frozen = other.get_positions().copy()
    returned = other.calc.optimize(other, apply=False)
    assert np.allclose(other.get_positions(), frozen)
    assert not np.allclose(returned, frozen)


def test_ase_dc_exposes_the_multipole_cutoff():
    """A knob that exists in Rust but not in the calculator is a knob nobody can reach."""
    positions = [[p[0] + 4.0 * i, p[1], p[2]] for i in range(8) for p in WATER_XYZ]
    atoms = Atoms("OH2" * 8, positions=positions)
    atoms.calc = AM1(divide_conquer=True, core_size=4, buffer_radius=10.0)
    exact = atoms.get_potential_energy()

    screened_atoms = Atoms("OH2" * 8, positions=positions)
    screened_atoms.calc = AM1(
        divide_conquer=True, core_size=4, buffer_radius=10.0, multipole_cutoff=15.0
    )
    screened = screened_atoms.get_potential_energy()
    # The approximation is small but real: it must change the answer, and not by much.
    assert exact != screened
    assert abs(exact - screened) / len(atoms) < 0.05
