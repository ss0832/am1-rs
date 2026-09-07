# PySEQM — retained material and provenance

**PySEQM** (LANL) is a PyTorch semiempirical quantum-mechanics package that reproduces MOPAC. It is
the reference implementation several parts of `am1-rs` were **ported from**, and the source of the
machine-readable AM1 parameter table this crate ships.

## Data retained

| file in this repository | upstream name | what it is |
|---|---|---|
| `src/data/am1_parameters.csv` | `seqm/params/parameters_AM1_MOPAC.csv` | the AM1 per-element parameters |

It is **compiled into every binary and wheel** by `include_str!` from `src/params.rs`; it is not
read from disk at run time, so the copy in `src/data/` is the distributed form of this material.
Its header line carries the same provenance note as this file, so the attribution travels with the
data even when the file is read on its own.

The parameter *values* are published scientific facts — Dewar, Zoebisch, Healy & Stewart,
*J. Am. Chem. Soc.* **107**, 3902 (1985) for H, C, N, O, plus the AM1 element-extension papers by
Dewar and co-workers, as consolidated in MOPAC. What came from PySEQM is this particular
machine-readable tabulation of them.

## Code ported

These are ports — the formulas were taken from PySEQM's implementation, not merely from the
literature — and each is named here so the correspondence can be checked:

| this repository | PySEQM |
|---|---|
| `src/params.rs` (closed forms and secant solves for `dd`/`qq`, `ρ0`/`ρ1`/`ρ2`) | `cal_par.py` |
| `src/integrals.rs` (the 22 local-frame two-centre two-electron integrals and their rotation into the molecular frame) | `two_elec_two_center_int*.py` |
| `src/overlap.rs` (analytic Slater diatomic overlap) | `diat_overlap_PM6_SP.py` |
| `src/fock.rs` (MNDO one-centre / two-centre Fock assembly) | `fock.py` |
| `src/repulsion.rs`, `src/scf.rs` (core–core repulsion with the AM1 Gaussians and the N–H/O–H special cases; isolated-atom energies; heat-of-formation assembly) | `energy.py`, `constants.py` |

Everything else in the crate is original. In particular the analytic gradient, the CPHF Hessian,
infrared intensities, AM1-BCC, divide-and-conquer, and the whole periodic-boundary-condition path
have no PySEQM counterpart.

## Where it came from

- Repository: <https://github.com/lanl/PYSEQM>
- G. Zhou, B. Nebgen, N. Lubbers, W. Malone, A. M. N. Niklasson & S. Tretiak, "Graphics Processing
  Unit-Accelerated Semiempirical Born Oppenheimer Molecular Dynamics Using PyTorch,"
  *J. Chem. Theory Comput.* **16**, 4951 (2020).

## Licence

**BSD 3-Clause**, © 2020 Triad National Security, LLC (Los Alamos National Laboratory), produced
under U.S. Government contract 89233218CNA000001. The full text is retained here as `LICENSE`.

Clause 2 of that licence requires the copyright notice to accompany **binary** redistributions, and
a wheel is a binary redistribution. That is why the file lives in this directory rather than being
referenced: `pyproject.toml`'s `license-files = [..., "third_party/*/LICENSE"]` glob ships it inside
the wheel -- along with this README, since 0.2.3. CI checks that every subdirectory of
`third_party/` has both a `LICENSE` and a `README.md` and that both reach the built wheel. The notice is additionally repeated in the header of
`src/data/am1_parameters.csv` and in the module documentation of the ported files.

BSD-3-Clause is GPL-compatible, so the ported material may be combined into this GPL-3.0-or-later
work. See §6 of `THIRD_PARTY_NOTICES.md`.
