#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate water clusters and chains of a requested size, for scaling benchmarks.

Two shapes, because they stress different things:

* ``cluster`` packs molecules on a cubic lattice, so the number of atom pairs within any
  fixed cutoff grows with N. This is the honest test of the pair-integral and electrostatic
  work.
* ``chain`` lays molecules along one axis, so the neighbour count per atom is bounded. A
  linear-scaling method should be linear here first; if it is not linear on a chain it will
  certainly not be linear on a cluster.

Geometry is the AM1 water minimum, rotated by a deterministic pseudo-random sequence so the
structures are reproducible without a seed file, and spaced far enough apart to stay a
physically sensible hydrogen-bonded assembly rather than an overlapping mess.

    python tools/make_water_cluster.py cluster 200 examples/bench/water_200.xyz
"""

import math
import sys

# AM1-optimized water, in Angstrom: r(OH) = 0.9614, angle = 103.5 degrees.
R_OH = 0.9614
THETA = math.radians(103.5)
MONOMER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (R_OH, 0.0, 0.0)),
    ("H", (R_OH * math.cos(THETA), R_OH * math.sin(THETA), 0.0)),
]

# Oxygen-to-oxygen spacing. Ice Ih sits at 2.76 A, but that assumes *ordered* orientations
# where every hydrogen points along a bond. These molecules are deliberately mis-oriented, so
# at that spacing some O-H would point straight into a neighbouring oxygen and produce
# contacts near 1.3 A -- far inside a hydrogen bond, and a strained SCF that would be
# measuring the wrong thing. 3.8 A keeps the closest intermolecular contact above 1.7 A,
# which is a short but physical hydrogen bond.
SPACING = 3.8
# Refuse to emit a structure with any intermolecular contact below this (Angstrom).
MIN_CONTACT = 1.6


def rotation(i):
    """A deterministic, well-spread orientation for molecule ``i``.

    ZYZ Euler angles driven by three mutually incommensurate rates, so successive molecules
    are never aligned and the sequence never repeats over the sizes used here.
    """
    alpha = math.pi * (3.0 - math.sqrt(5.0)) * i  # golden angle
    beta = 0.7 * i
    gamma = 0.9 * i
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    return (
        (ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb),
        (sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb),
        (-sb * cg, sb * sg, cb),
    )


def apply(rot, v):
    return tuple(sum(rot[r][c] * v[c] for c in range(3)) for r in range(3))


def centres(shape, n_mol):
    if shape == "chain":
        return [(i * SPACING, 0.0, 0.0) for i in range(n_mol)]
    side = math.ceil(n_mol ** (1.0 / 3.0))
    out = []
    for i in range(n_mol):
        a = i % side
        b = (i // side) % side
        c = i // (side * side)
        out.append((a * SPACING, b * SPACING, c * SPACING))
    return out


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    shape, natoms, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    if shape not in ("cluster", "chain"):
        print(f"unknown shape {shape!r}; expected 'cluster' or 'chain'")
        return 1
    n_mol = max(1, natoms // 3)

    atoms = []
    for i, centre in enumerate(centres(shape, n_mol)):
        rot = rotation(i)
        for sym, pos in MONOMER:
            x, y, z = apply(rot, pos)
            atoms.append((sym, i, (x + centre[0], y + centre[1], z + centre[2])))

    # A benchmark structure that is physically nonsense measures the wrong thing: a strained
    # SCF converges differently, so check rather than assume.
    closest = min(
        (
            math.dist(a[2], b[2])
            for k, a in enumerate(atoms)
            for b in atoms[k + 1 :]
            if a[1] != b[1]
        ),
        default=float("inf"),
    )
    if closest < MIN_CONTACT:
        print(
            f"refusing to write {path}: closest intermolecular contact is {closest:.3f} A, "
            f"below the {MIN_CONTACT} A floor -- increase SPACING",
            file=sys.stderr,
        )
        return 1

    lines = [f"{s:2s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}" for s, _, p in atoms]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(lines)}\n")
        fh.write(
            f"{shape} of {n_mol} water molecules, AM1 monomer geometry, "
            f"closest intermolecular contact {closest:.3f} A\n"
        )
        fh.write("\n".join(lines) + "\n")
    print(
        f"{path}: {len(lines)} atoms ({n_mol} molecules, {shape}), "
        f"closest intermolecular contact {closest:.3f} A"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
