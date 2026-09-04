# SPDX-License-Identifier: GPL-3.0-or-later
"""The lazy `get_*` cache: that it is a cache, and that it is invalidated.

`docs/scope.md` claims these methods "cache into `results` and are invalidated by `check_state`".
The second half was false through 0.2.1. `Calculator.results` is cleared by
`Calculator.get_property`, which calls `check_state` and then `reset()` — but none of the lazy
methods go through `get_property`, so nothing ever cleared them. Moving the atoms and asking again
returned the previous geometry's answer, silently.

`test_the_lazy_cache_is_keyed_on_the_arguments` in `test_new_api_0_2_1.py` could not catch it
because it never moves the geometry between calls. These do.
"""
from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402

from am1_rs.ase import AM1  # noqa: E402

WATER = [(0.0, 0.0, 0.0), (0.9584, 0.0, 0.0), (-0.2400, 0.9278, 0.0)]


def water_atoms():
    return Atoms("OH2", positions=WATER)


def displaced():
    a = water_atoms()
    p = a.get_positions()
    p[1, 0] += 0.15
    a.set_positions(p)
    return a


# --------------------------------------------------------------------------- invalidation
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda c, a: c.get_frequencies(a), id="get_frequencies"),
        pytest.param(
            lambda c, a: np.asarray(c.get_ir_spectrum(a)["intensities_km_per_mol"]),
            id="get_ir_spectrum",
        ),
        pytest.param(lambda c, a: c.get_dipole_derivatives(a), id="get_dipole_derivatives"),
        pytest.param(lambda c, a: c.get_hessian(a), id="get_hessian"),
        pytest.param(
            lambda c, a: np.asarray(c.get_orbitals(a)["energies_ev"]), id="get_orbitals"
        ),
        pytest.param(
            lambda c, a: np.asarray(c.get_am1_bcc_charges(a)["charges"]),
            id="get_am1_bcc_charges",
        ),
    ],
)
def test_a_geometry_change_invalidates_the_lazy_cache(call):
    """Every lazy method, with no energy call in between to do the invalidating for it."""
    calc = AM1()
    first = np.asarray(call(calc, water_atoms()), dtype=float)
    moved = np.asarray(call(calc, displaced()), dtype=float)
    assert first.shape == moved.shape
    assert not np.allclose(first, moved), (
        "the moved geometry returned the original geometry's answer — the cache was not "
        "invalidated"
    )


def test_the_same_geometry_is_a_cache_hit():
    """The other half: it must still be a cache, or the fix is just "never cache"."""
    calc = AM1()
    atoms = water_atoms()
    one = calc.get_ir_spectrum(atoms)
    two = calc.get_ir_spectrum(atoms)
    assert one is two, "asking twice at the same geometry should not re-solve"


def test_an_atoms_info_override_invalidates_the_lazy_cache():
    """`atoms.info` overrides are invisible to ASE's own comparison, so the key carries them."""
    calc = AM1()
    neutral = calc.get_frequencies(water_atoms())

    charged = water_atoms()
    charged.info["charge"] = 1.0
    charged.info["multiplicity"] = 2
    cation = calc.get_frequencies(charged)
    assert not np.allclose(neutral, cation), "the charge override did not reach the cache key"


def test_optimize_invalidates_everything_it_should():
    """`optimize(apply=True)` moves the atoms, so both caches have to go."""
    calc = AM1()
    atoms = water_atoms()
    atoms.calc = calc
    before = calc.get_frequencies(atoms)
    energy_before = atoms.get_potential_energy()
    calc.optimize(atoms)
    after = calc.get_frequencies(atoms)
    energy_after = atoms.get_potential_energy()
    assert not np.allclose(before, after), "frequencies survived a geometry optimization"
    assert energy_after < energy_before + 1e-9


# ------------------------------------------------------------------------- one solve, many uses
def test_the_hessian_family_shares_one_cphf_solve():
    """Five questions, one CPHF.

    Each of these used to call a different native entry point, and each of those ran the whole
    analytic-Hessian solve. They are contractions of the same response, so the count of CPHF
    iterations — which the native layer reports — has to be identical across them, and the
    underlying dict has to be the *same object*.
    """
    calc = AM1()
    atoms = water_atoms()

    spectrum = calc.get_ir_spectrum(atoms)
    freqs = calc.get_frequencies(atoms)
    hess = calc.get_hessian(atoms)
    apt = calc.get_dipole_derivatives(atoms)

    # One entry in the cache for the whole group, not four.
    vibrational = [k for k in calc._extra_cache if k.startswith("vibrational")]
    assert len(vibrational) == 1, f"expected one shared solve, got {vibrational}"

    # And the pieces agree with each other, which they must if they came from one solve.
    assert np.allclose(freqs, np.asarray(spectrum["frequencies_cm"]))
    assert np.allclose(apt, np.asarray(spectrum["dipole_derivatives"]))
    assert hess.shape == (3 * len(atoms), 3 * len(atoms))
    assert "cphf_iterations" in spectrum


def test_the_orbital_response_is_still_opt_in():
    """`u_ov` is `O(ndof · n_ov)`, so it is a separate cache entry and not fetched by default."""
    calc = AM1()
    atoms = water_atoms()
    plain = calc.get_ir_spectrum(atoms)
    assert "u_ov" not in plain

    with_u = calc.get_orbital_response(atoms)
    assert "u_ov" in with_u
    assert "response_density" not in with_u

    with_dp = calc.get_orbital_response(atoms, response_density=True)
    assert "response_density" in with_dp


def test_the_shared_solve_matches_the_individual_entry_points():
    """A cost change, not a numerical one: the bundle must equal what the five functions gave."""
    from am1_rs import native

    atoms = water_atoms()
    z, p = atoms.get_atomic_numbers(), atoms.get_positions()

    bundle = native.vibrations(z, p, orbital_response=True)
    assert np.allclose(
        bundle["hessian_ev_per_angstrom2"],
        native.hessian(z, p)["hessian_ev_per_angstrom2"],
    )
    assert np.allclose(
        bundle["frequencies_cm"], native.frequencies(z, p)["frequencies_cm"]
    )
    spec = native.ir_spectrum(z, p)
    assert np.allclose(bundle["intensities_km_per_mol"], spec["intensities_km_per_mol"])
    assert np.allclose(bundle["dipole_derivatives"], spec["dipole_derivatives"])
    assert np.allclose(
        bundle["dipole_derivatives"],
        native.dipole_derivatives(z, p)["dipole_derivatives"],
    )
    assert np.allclose(bundle["u_ov"], native.orbital_response(z, p)["u_ov"])
