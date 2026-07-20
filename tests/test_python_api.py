# SPDX-License-Identifier: GPL-3.0-or-later
"""Python-layer tests: charge/multiplicity plumbing (native + ASE) and the Hessian API.

Run against a built extension:

    maturin develop            # build am1_rs._native into the active environment
    pytest tests/test_python_api.py

These are not compiled by cargo (it only picks up ``tests/*.rs``) and are not packaged into
the wheel (maturin packages only ``python/am1_rs``).
"""
import math

import numpy as np
import pytest

am1_rs = pytest.importorskip("am1_rs")

# Rust src/data_tables.rs MASS and src/hessian.rs conversion constant.
MASS = {1: 1.0079, 6: 12.011, 8: 15.9994}
SQRT_EV_PER_ANG2_AMU_TO_CM = 521.4709
AM1_EV = 27.21
AM1_A0 = 0.529167

WATER_Z = [8, 1, 1]
WATER_XYZ = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]]


# --------------------------------------------------------------------------- native charge/spin
def test_native_charge_changes_energy():
    neutral = am1_rs.single_point(WATER_Z, WATER_XYZ, charge=0.0, multiplicity=1)
    cation = am1_rs.single_point(WATER_Z, WATER_XYZ, charge=1.0, multiplicity=2)
    assert neutral["converged"] and cation["converged"]
    assert abs(neutral["energy_ev"] - cation["energy_ev"]) > 1.0


def test_native_multiplicity_changes_energy():
    o2 = [8, 8]
    xyz = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]]
    singlet = am1_rs.single_point(o2, xyz, multiplicity=1)
    triplet = am1_rs.single_point(o2, xyz, multiplicity=3)
    assert abs(singlet["energy_ev"] - triplet["energy_ev"]) > 1e-3


def test_native_multiplicity_parity_enforced():
    # 8 valence electrons cannot form a doublet -> must raise (proves multiplicity is used).
    with pytest.raises(Exception):
        am1_rs.single_point(WATER_Z, WATER_XYZ, charge=0.0, multiplicity=2)


# --------------------------------------------------------------------------- native reference
def test_native_reference_uhf_singlet_matches_rhf():
    # A singlet forced to UHF must converge to the RHF energy.
    auto = am1_rs.single_point(WATER_Z, WATER_XYZ)
    rhf = am1_rs.single_point(WATER_Z, WATER_XYZ, reference="rhf")
    uhf = am1_rs.single_point(WATER_Z, WATER_XYZ, reference="uhf")
    assert abs(auto["energy_ev"] - rhf["energy_ev"]) < 1e-9
    assert abs(auto["energy_ev"] - uhf["energy_ev"]) < 1e-6


def test_native_reference_aliases_case_insensitive():
    target = am1_rs.single_point(WATER_Z, WATER_XYZ, reference="uhf")["energy_ev"]
    for alias in ("UHF", "u", "Unrestricted"):
        assert abs(am1_rs.single_point(WATER_Z, WATER_XYZ, reference=alias)["energy_ev"] - target) < 1e-9


def test_native_reference_restricted_rejects_open_shell():
    ch3 = [6, 1, 1, 1]
    xyz = [[0, 0, 0.05], [1.09, 0, 0], [-0.545, 0.944, 0], [-0.545, -0.944, 0]]
    with pytest.raises(Exception):
        am1_rs.single_point(ch3, xyz, charge=0.0, multiplicity=2, reference="rhf")


def test_native_reference_invalid_raises():
    with pytest.raises(Exception):
        am1_rs.single_point(WATER_Z, WATER_XYZ, reference="bogus")


def test_native_hessian_honors_reference():
    Hr = np.asarray(am1_rs.hessian(WATER_Z, WATER_XYZ, reference="rhf")["hessian_ev_per_angstrom2"])
    Hu = np.asarray(am1_rs.hessian(WATER_Z, WATER_XYZ, reference="uhf")["hessian_ev_per_angstrom2"])
    assert np.abs(Hr - Hu).max() < 1e-4


# --------------------------------------------------------------------------- native Hessian
def test_native_hessian_shape_symmetry_units():
    h = am1_rs.hessian(WATER_Z, WATER_XYZ, 0.0, 1)
    Hau = np.asarray(h["hessian_hartree_per_bohr2"])
    Hev = np.asarray(h["hessian_ev_per_angstrom2"])
    assert h["ndof"] == 9
    assert Hau.shape == (9, 9) and Hev.shape == (9, 9)
    assert np.allclose(Hau, Hau.T, atol=1e-8)
    assert np.allclose(Hev, Hev.T, atol=1e-8)
    # eV/Å² = Hartree/Bohr² · (eV/Hartree) · (Bohr/Å)²
    ratio = AM1_EV / (AM1_A0 * AM1_A0)
    nz = np.abs(Hau) > 1e-6
    assert np.allclose(Hev[nz] / Hau[nz], ratio, rtol=1e-6)


def test_native_hessian_reproduces_frequencies():
    h = am1_rs.hessian(WATER_Z, WATER_XYZ, 0.0, 1)
    Hev = np.asarray(h["hessian_ev_per_angstrom2"])
    m = np.array([MASS[z] for z in WATER_Z for _ in range(3)])
    mw = Hev / np.sqrt(np.outer(m, m))
    eig = np.sort(np.linalg.eigvalsh(mw))
    ref = np.sort(np.asarray(am1_rs.frequencies(WATER_Z, WATER_XYZ, 0.0, 1)["eigenvalues"]))
    assert np.abs(eig - ref).max() < 1e-9


def test_native_hessian_matches_finite_difference():
    def grad(pos):
        g = am1_rs.gradient(WATER_Z, pos, 0.0, 1)
        return np.asarray(g["gradient_ev_per_angstrom"]).reshape(-1)

    Hev = np.asarray(am1_rs.hessian(WATER_Z, WATER_XYZ, 0.0, 1)["hessian_ev_per_angstrom2"])
    p0 = np.asarray(WATER_XYZ, float).reshape(-1)
    n = p0.size
    step = 1e-4
    Hfd = np.zeros((n, n))
    for j in range(n):
        pp = p0.copy(); pp[j] += step
        pm = p0.copy(); pm[j] -= step
        Hfd[:, j] = (grad(pp.reshape(-1, 3)) - grad(pm.reshape(-1, 3))) / (2 * step)
    Hfd = 0.5 * (Hfd + Hfd.T)
    assert np.abs(Hev - Hfd).max() < 1e-3


def test_native_hessian_uhf():
    Z = [6, 1, 1, 1]
    xyz = [[0, 0, 0.05], [1.09, 0, 0], [-0.545, 0.944, 0], [-0.545, -0.944, 0]]
    Hev = np.asarray(am1_rs.hessian(Z, xyz, 0.0, 2)["hessian_ev_per_angstrom2"])
    assert Hev.shape == (12, 12)
    assert np.allclose(Hev, Hev.T, atol=1e-7)


# --------------------------------------------------------------------------- ASE layer
ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from am1_rs.ase import AM1  # noqa: E402


def test_ase_charge_multiplicity_at_construction():
    a = Atoms(numbers=WATER_Z, positions=WATER_XYZ)
    a.calc = AM1(charge=0.0, multiplicity=1)
    e0 = a.get_potential_energy()
    b = Atoms(numbers=WATER_Z, positions=WATER_XYZ)
    b.calc = AM1(charge=1.0, multiplicity=2)
    e1 = b.get_potential_energy()
    assert abs(e0 - e1) > 1.0


def test_ase_charge_multiplicity_via_atoms_info():
    calc = AM1()  # neutral singlet defaults
    neutral = Atoms(numbers=WATER_Z, positions=WATER_XYZ); neutral.calc = calc
    e0 = neutral.get_potential_energy()
    cation = Atoms(numbers=WATER_Z, positions=WATER_XYZ)
    cation.info["charge"] = 1.0
    cation.info["multiplicity"] = 2
    cation.calc = calc
    e1 = cation.get_potential_energy()
    assert abs(e0 - e1) > 1.0
    # constructor path must agree with the atoms.info path
    ctor = Atoms(numbers=WATER_Z, positions=WATER_XYZ)
    ctor.calc = AM1(charge=1.0, multiplicity=2)
    assert abs(ctor.get_potential_energy() - e1) < 1e-9


def test_ase_cache_invalidated_on_charge_change():
    atoms = Atoms(numbers=WATER_Z, positions=WATER_XYZ)
    atoms.calc = AM1()
    e0 = atoms.get_potential_energy()
    atoms.info["charge"] = 1.0
    atoms.info["multiplicity"] = 2
    e1 = atoms.get_potential_energy()
    assert abs(e0 - e1) > 1.0


def test_ase_get_hessian_matches_native():
    atoms = Atoms(numbers=WATER_Z, positions=WATER_XYZ)
    atoms.calc = AM1()
    Hase = atoms.calc.get_hessian(atoms)
    Hev = np.asarray(am1_rs.hessian(WATER_Z, WATER_XYZ, 0.0, 1)["hessian_ev_per_angstrom2"])
    assert Hase.shape == (9, 9)
    assert np.allclose(Hase, Hev, atol=1e-8)


def test_ase_reference_constructor_and_atoms_info():
    ctor = Atoms(numbers=WATER_Z, positions=WATER_XYZ); ctor.calc = AM1(reference="uhf")
    e_ctor = ctor.get_potential_energy()
    info = Atoms(numbers=WATER_Z, positions=WATER_XYZ)
    info.info["reference"] = "uhf"
    info.calc = AM1()
    e_info = info.get_potential_energy()
    default = Atoms(numbers=WATER_Z, positions=WATER_XYZ); default.calc = AM1()
    e_default = default.get_potential_energy()
    assert abs(e_ctor - e_info) < 1e-9          # both forced UHF
    assert abs(e_ctor - e_default) < 1e-6        # singlet: UHF energy == RHF energy


def test_ase_reference_change_invalidates_cache():
    atoms = Atoms(numbers=WATER_Z, positions=WATER_XYZ)
    atoms.calc = AM1()
    atoms.get_potential_energy()
    assert atoms.calc._last_state[2] == "auto"
    atoms.info["reference"] = "uhf"
    atoms.get_potential_energy()
    assert atoms.calc._last_state[2] == "uhf"    # recomputed with the new reference
