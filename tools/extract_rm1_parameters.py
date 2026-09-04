#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract the RM1 parameter set from MOPAC's Fortran tabulation into this crate's CSV schema.

RM1 (Rocha, Freire, Simas & Stewart, J. Comput. Chem. 27, 1101 (2006)) has exactly the AM1
functional form -- the same NDDO core, the same core-core Gaussian corrections -- and differs
only in the parameter values. So the whole of this crate's AM1 machinery applies unchanged
once the numbers are loaded, and the only real task is getting the numbers from an
authoritative machine-readable source rather than retyping them from a paper.

That source is MOPAC's `src/models/parameters_for_RM1_C.F90` (Apache-2.0), which is
GPL-compatible in the direction this crate needs. Run:

    python tools/extract_rm1_parameters.py <path-to-parameters_for_RM1_C.F90> \
        src/data/rm1_parameters.csv

Only the main-group elements are emitted. RM1's lanthanides (Z = 57-71) need d and f orbitals
and the sparkle model, neither of which this crate has, and emitting them would produce a
parameter block that loads and then computes nonsense.
"""

import re
import sys
from collections import defaultdict

# RM1's published main-group set (Rocha et al. 2006). Anything else in the file is a
# lanthanide, a sparkle, or the capped bond, all out of scope here.
MAIN_GROUP = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}

# Fortran array name -> CSV column name.
SCALARS = {
    "uss": "U_ss",
    "upp": "U_pp",
    "zs": "zeta_s",
    "zp": "zeta_p",
    "zd": "zeta_d",
    "betas": "beta_s",
    "betap": "beta_p",
    "gss": "g_ss",
    "gsp": "g_sp",
    "gpp": "g_pp",
    "gp2": "g_p2",
    "hsp": "h_sp",
    "alp": "alpha",
}

COLUMNS = (
    ["N", "sym"]
    + [
        "U_ss", "U_pp", "zeta_s", "zeta_p", "zeta_d", "beta_s", "beta_p",
        "g_ss", "g_sp", "g_pp", "g_p2", "h_sp", "alpha",
    ]
    + [f"Gaussian{i}_{c}" for i in (1, 2, 3, 4) for c in ("K", "L", "M")]
)

SCALAR_RE = re.compile(
    r"data\s+(?P<name>[A-Za-z0-9_]+?)RM1\(\s*(?P<z>\d+)\s*\)\s*/\s*(?P<val>[-+0-9.DdEe]+)\s*/",
    re.IGNORECASE,
)
GUESS_RE = re.compile(
    r"data\s+guess(?P<idx>[123])RM1\(\s*(?P<z>\d+)\s*,\s*(?P<k>\d+)\s*\)\s*/\s*(?P<val>[-+0-9.DdEe]+)\s*/",
    re.IGNORECASE,
)


def fortran_float(text):
    return float(text.replace("D", "E").replace("d", "e"))


def parse(path):
    scalars = defaultdict(dict)          # z -> {column: value}
    gaussians = defaultdict(lambda: defaultdict(dict))  # z -> k -> {1|2|3: value}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = GUESS_RE.search(line)
            if m:
                z, k, idx = int(m["z"]), int(m["k"]), int(m["idx"])
                gaussians[z][k][idx] = fortran_float(m["val"])
                continue
            m = SCALAR_RE.search(line)
            if m:
                name = m["name"].lower()
                if name in SCALARS:
                    scalars[int(m["z"])][SCALARS[name]] = fortran_float(m["val"])
    return scalars, gaussians


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, dest = sys.argv[1], sys.argv[2]
    scalars, gaussians = parse(src)

    missing = [z for z in MAIN_GROUP if z not in scalars]
    if missing:
        print(f"error: no RM1 block found for Z = {missing} in {src}", file=sys.stderr)
        return 1

    rows = []
    for z in MAIN_GROUP:
        vals = scalars[z]
        row = [f"{z:4d}", f"{SYMBOLS[z]:>4s}"]
        for col in COLUMNS[2:15]:
            row.append(f"{vals.get(col, 0.0):13.7f}")
        for k in (1, 2, 3, 4):
            g = gaussians[z].get(k, {})
            for idx in (1, 2, 3):
                row.append(f"{g.get(idx, 0.0):13.7f}")
        rows.append(",".join(row))

    header = f"{'N':>4s},{'sym':>4s}," + ",".join(f"{c:>13s}" for c in COLUMNS[2:])
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# RM1 parameters (eV / Bohr^-1 / Angstrom^-1). PROVENANCE:\n"
            "#   Original scientific source: G. B. Rocha, R. O. Freire, A. M. Simas &\n"
            "#   J. J. P. Stewart, \"RM1: A Reparameterization of AM1 for H, C, N, O, P, S, F,\n"
            "#   Cl, Br and I\", J. Comput. Chem. 27, 1101-1111 (2006).\n"
            "#   This machine-readable tabulation was extracted from MOPAC\n"
            "#   (src/models/parameters_for_RM1_C.F90), Copyright 2021 Virginia Polytechnic\n"
            "#   Institute and State University, Apache License 2.0 -- see\n"
            "#   THIRD_PARTY_NOTICES.md and third_party/mopac/LICENSE.\n"
            "#   Extracted by tools/extract_rm1_parameters.py; do not hand-edit.\n"
            "#\n"
            "#   RM1 has the AM1 functional form exactly (same NDDO core, same core-core\n"
            "#   Gaussian corrections), so these load into the same code path as AM1.\n"
            "#   Only RM1's published main-group set is included. Its lanthanides (Z=57-71)\n"
            "#   require d/f orbitals and the sparkle model, which this crate does not have.\n"
            "#\n"
            "# Lines starting with '#' are comments and are skipped by the parser.\n"
        )
        fh.write(header + "\n")
        fh.write("\n".join(rows) + "\n")

    print(f"{dest}: {len(rows)} elements ({', '.join(SYMBOLS[z] for z in MAIN_GROUP)})")
    for z in MAIN_GROUP:
        ng = sum(1 for k in gaussians[z] if any(gaussians[z][k].values()))
        print(f"  Z={z:3d} {SYMBOLS[z]:>2s}  {ng} Gaussian(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
