# SPDX-License-Identifier: GPL-3.0-or-later
"""A periodic response raises on a partially occupied ground state instead of answering wrongly.

Through 0.2.2 every periodic response path assumed integer occupations and none of them checked.
The CPHF path classifies each band as occupied or virtual and a partially filled one is *neither*,
so it was dropped from the response entirely; DFPT kept every band pair but froze the occupation
difference weighting them. Both produced a number, and nothing in the result said the number was
outside what the equations cover.

The distinction these tests pin is the one a user of a small-gap solid needs: **smearing is not
what is refused**. The judgement is on the converged occupations, so a gapped system smeared well
below its gap passes.
"""

import numpy as np
import pytest

from am1_rs import native

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402

from am1_rs.ase import AM1  # noqa: E402

# Two hydrogens per cell at equal spacing: no dimerization, so the folded band is half filled.
METAL = dict(
    numbers=[1, 1],
    positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
    cell=[[20.0, 0, 0], [0, 20.0, 0], [0, 0, 3.0]],
    pbc=[False, False, True],
)
# Dimerized: a wide gap, and the same atom count.
GAPPED = dict(METAL, positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], cell=[[20.0, 0, 0], [0, 20.0, 0], [0, 0, 4.0]])


def _args(system):
    return (system["numbers"], system["positions"], system["cell"], system["pbc"])


def test_a_half_filled_band_raises_rather_than_answering():
    with pytest.raises(ValueError) as excinfo:
        native.pbc_hessian(*_args(METAL), kpts=(1, 1, 6))

    message = str(excinfo.value)
    # The message has to locate the problem and say what to do, or it is just a refusal.
    assert "integer occupations" in message
    assert "band" in message and "k-point" in message
    assert "not a refusal to use smearing" in message
    assert "finer k-mesh" in message


def test_smearing_on_a_gapped_system_is_accepted():
    """The point of the whole design: this must not be a switch on `smearing_ev`.

    Smearing is how a small-gap or coarsely sampled solid reaches a converged ground state at all.
    A gapped system's conduction band holds about ``exp(-gap/2kT)`` electrons, so at a kT well
    under the gap it is integer to far better than the cut, and the response is accepted.

    ``dfpt`` is the entry point used here because it is the one that takes ``smearing_ev`` — the
    CPHF-side entry points always run the ground state at zero smearing. Measured on real
    structures, cubic BN passes on a 4x4x4 mesh at ``smearing_ev=0.3``, which is the case
    ``docs/pbc.md`` tells a user to reach for; see ``tests/fractional_occupations.rs``.
    """
    for smearing in (0.0, 0.1, 0.3):
        r = native.dfpt(*_args(GAPPED), [[0.0, 0.0, 0.0]], kpts=(1, 1, 4), smearing_ev=smearing)
        assert len(r["frequencies_cm"][0]) == 6, f"refused at smearing_ev={smearing}"


def test_the_escape_hatch_reproduces_the_old_behaviour():
    """`allow_fractional_occupations=True` runs it anyway -- and the number is the wrong one.

    Kept because refusing to produce anything is not always the right trade for a user who knows
    what they are looking at. It is a boolean rather than a tolerance because the choice is
    between an error and a known-wrong answer, not between two accuracies.
    """
    with pytest.raises(ValueError):
        native.pbc_hessian(*_args(METAL), kpts=(1, 1, 6))

    r = native.pbc_hessian(*_args(METAL), kpts=(1, 1, 6), allow_fractional_occupations=True)
    assert r["ndof"] == 6
    assert np.isfinite(np.asarray(r["hessian_ev_per_angstrom2"])).all()


def test_dfpt_refuses_the_same_state():
    """Two response paths, two different failure modes, one diagnosis."""
    with pytest.raises(ValueError) as excinfo:
        native.dfpt(*_args(METAL), [[0.0, 0.0, 0.0]], kpts=(1, 1, 6))
    assert "DFPT" in str(excinfo.value)

    r = native.dfpt(
        *_args(METAL), [[0.0, 0.0, 0.0]], kpts=(1, 1, 6), allow_fractional_occupations=True
    )
    assert len(r["frequencies_cm"][0]) == 6


def test_the_ase_calculator_carries_the_switch():
    """A calculator parameter, not an argument to each `get_*`.

    Being outside the equations is a property of the system, not of the question: a structure that
    needs this needs it for the Hessian, the Born charges and DFPT alike.
    """
    atoms = Atoms(numbers=METAL["numbers"], positions=METAL["positions"],
                  cell=METAL["cell"], pbc=METAL["pbc"])

    atoms.calc = AM1(kpts=(1, 1, 6))
    with pytest.raises(ValueError):
        atoms.calc.get_hessian(atoms)

    atoms.calc = AM1(kpts=(1, 1, 6), allow_fractional_occupations=True)
    h = atoms.calc.get_hessian(atoms)
    assert np.asarray(h).shape == (6, 6)
