# SPDX-License-Identifier: GPL-3.0-or-later
"""`phonons` returns the mode vectors, not only the frequencies.

Through 0.2.2 the only way to learn *what a mode does* was to re-run the calculation in a tool
that returns eigenvectors, which is the whole calculation twice for information the
diagonalization already had in hand. Two conventions come back, and the tests below pin the
difference between them, because it is the part that is easy to get silently wrong: the
polarization `e(q)` is orthonormal and belongs in sums over modes, while `e_a/sqrt(m_a)` is what
an atom actually does and belongs in a displaced geometry. Using one where the other is meant is
a mass-weighting error that looks plausible in a picture and is wrong by a factor of
`sqrt(m_H/m_O)` = 4.
"""

import numpy as np
import pytest

from am1_rs import native

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402

from am1_rs.ase import AM1  # noqa: E402

WATER = [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]
CELL = np.eye(3) * 4.5


def _phonons(**kw):
    return native.phonons([8, 1, 1], WATER, CELL, [True, True, True], supercell=(1, 1, 1), **kw)


def test_eigenvectors_are_opt_in():
    """Off by default: `n_q x (3N)^2` complex numbers is a large array to hand back unasked."""
    plain = _phonons()
    assert not [k for k in plain if "polarization" in k or "displacements" in k]

    full = _phonons(eigenvectors=True)
    for key in ("polarization_re", "polarization_im", "displacements_re", "displacements_im"):
        assert key in full

    # Asking for the vectors must not perturb the frequencies -- it is one diagonalization either
    # way, and the flag decides only what is kept from it.
    assert np.allclose(plain["frequencies_cm"], full["frequencies_cm"])


def test_the_polarization_is_orthonormal_and_the_displacement_is_not():
    r = _phonons(eigenvectors=True)
    e = np.asarray(r["polarization_re"]) + 1j * np.asarray(r["polarization_im"])
    u = np.asarray(r["displacements_re"]) + 1j * np.asarray(r["displacements_im"])
    assert e.shape == u.shape == (1, 9, 9), "one q point, 3N x 3N, columns are modes"

    assert np.abs(e[0].conj().T @ e[0] - np.eye(9)).max() < 1e-10

    # u = e / sqrt(m) with one mass per atom, repeated over x, y, z. This is the relation that
    # separates the two, so it is checked against the masses rather than against itself.
    m = np.repeat([15.999, 1.008, 1.008], 3)
    assert np.abs(u[0] - e[0] / np.sqrt(m)[:, None]).max() < 1e-3

    # And so the displacement columns are *not* unit vectors. Every norm sits below 1/sqrt(m) of
    # the lightest atom, since dividing by a mass above 1 amu can only shrink it -- so the thing
    # that distinguishes the two conventions is the *spread*, not the magnitude: a mode carried by
    # the hydrogens keeps almost all of its length and one carried by the oxygen loses most of it.
    # Were the mass weighting lost, all nine would be exactly 1 and the ratio exactly 1.
    norms = np.linalg.norm(u[0], axis=0)
    assert norms.max() < 1.0
    assert norms.max() / norms.min() > 2.0


def test_a_mode_vector_moves_the_atoms_it_should():
    """The highest mode of water is an O-H stretch, so the hydrogens must carry it.

    A frequency alone cannot be checked for *which* motion it describes. This is the assertion
    that the columns are ordered like `frequencies_cm` and that the three components per atom run
    x, y, z -- get either wrong and the amplitude lands on the wrong atom.
    """
    r = _phonons(eigenvectors=True)
    u = np.asarray(r["displacements_re"])[0]
    top = int(np.argmax(r["frequencies_cm"][0]))
    per_atom = np.linalg.norm(u[:, top].reshape(3, 3), axis=1)
    assert per_atom[1:].min() > 4 * per_atom[0], "the light atoms should dominate an O-H stretch"


def test_ase_joins_the_halves_into_complex_arrays():
    """The ASE layer has numpy as a hard dependency, so it returns one complex array, not two."""
    atoms = Atoms("OH2", positions=WATER, cell=CELL, pbc=True)
    atoms.calc = AM1()

    r = atoms.calc.get_phonons(atoms, supercell=(1, 1, 1), eigenvectors=True)
    assert "polarization_re" not in r and "displacements_im" not in r
    for key in ("polarization", "displacements"):
        assert r[key].dtype == np.complex128
        assert r[key].shape == (1, 9, 9)

    native_r = _phonons(eigenvectors=True)
    expect = np.asarray(native_r["polarization_re"]) + 1j * np.asarray(native_r["polarization_im"])
    assert np.abs(r["polarization"] - expect).max() < 1e-12


def test_the_ase_cache_does_not_serve_a_vectorless_answer():
    """`eigenvectors` changes the result, so it has to be part of the cache key.

    Without it the second call here returns the first call's dict and the vectors are simply
    missing -- the same class of bug the argument-keyed cache was introduced to fix.
    """
    atoms = Atoms("OH2", positions=WATER, cell=CELL, pbc=True)
    atoms.calc = AM1()

    plain = atoms.calc.get_phonons(atoms, supercell=(1, 1, 1))
    full = atoms.calc.get_phonons(atoms, supercell=(1, 1, 1), eigenvectors=True)
    assert "polarization" not in plain
    assert "polarization" in full
    assert np.allclose(plain["frequencies_cm"], full["frequencies_cm"])
