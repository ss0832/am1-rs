# Third-party notices and parameter attribution

`am1-rs` reproduces the standard, published **AM1** model. This file records the provenance
of the numerical parameters and of the reference formulas the code was ported from, so that
every parameter's origin is explicit.

## 1. AM1 model parameters (primary scientific sources)

The AM1 per-element parameters — `U_ss, U_pp, ζ_s, ζ_p, β_s, β_p, G_ss, G_sp, G_pp, G_p2,
H_sp, α`, and the AM1 core-core Gaussian `K/L/M` triples — are **published scientific
constants**, not the creation of this project. Their authoritative sources are:

- **H, C, N, O** — M. J. S. Dewar, E. G. Zoebisch, E. F. Healy, J. J. P. Stewart,
  "AM1: A New General Purpose Quantum Mechanical Molecular Model,"
  *J. Am. Chem. Soc.* **107**, 3902–3909 (1985).
- **Halogens and further main-group elements** — the AM1 element-extension papers by Dewar
  and co-workers (F, Cl, Br, I; Al, Si, P, S; Zn, Ge, Hg; …), as consolidated in **MOPAC**
  (J. J. P. Stewart, *Molecular Orbital PACkage*; https://openmopac.github.io/).
- The isolated-atom average-of-configuration coefficients and the experimental atomic heats
  of formation reproduce MOPAC's `calpar.f` / `block.f` tables.

The numerical values are facts drawn from the above literature. The specific machine-readable
tabulation used here (`src/data/am1_parameters.csv`) was obtained from the PySEQM project
(see §2); its header line records this.

## 2. Reference implementation ported: PySEQM (BSD-3-Clause)

The following parts of `am1-rs` were **ported from** the PySEQM reference implementation,
which itself reproduces MOPAC:

- the machine-readable AM1 parameter table `src/data/am1_parameters.csv`
  (from PySEQM `seqm/params/parameters_AM1_MOPAC.csv`);
- the closed forms and secant solves for the charge separations `dd`/`qq` and the additive
  terms `ρ0`/`ρ1`/`ρ2` (`src/params.rs`; PySEQM `cal_par.py`);
- the 22 local-frame two-center two-electron integrals and their rotation into the molecular
  frame (`src/integrals.rs`; PySEQM `two_elec_two_center_int*.py`);
- the analytic Slater diatomic overlap (`src/overlap.rs`; PySEQM `diat_overlap_PM6_SP.py`);
- the MNDO one-center / two-center Fock assembly (`src/fock.rs`; PySEQM `fock.py`);
- the core-core repulsion incl. the AM1 Gaussians and the N–H/O–H special cases, the
  isolated-atom energies and the heat-of-formation assembly
  (`src/repulsion.rs`, `src/scf.rs`; PySEQM `energy.py`, `constants.py`).

PySEQM:
- Repository: https://github.com/lanl/PYSEQM
- G. Zhou, B. Nebgen, N. Lubbers, W. Malone, A. M. N. Niklasson, S. Tretiak,
  "Graphics Processing Unit-Accelerated Semiempirical Born Oppenheimer Molecular Dynamics
  Using PyTorch," *J. Chem. Theory Comput.* **16**, 4951 (2020).
- License: **BSD 3-Clause**, © 2020 Triad National Security, LLC (Los Alamos National
  Laboratory). The full license is retained at `third_party/pyseqm/LICENSE`, as required by
  clause 1 of that license.

## 3. AM1-BCC

The AM1-BCC method is due to A. Jakalian, B. L. Bush, D. B. Jack, C. I. Bayly,
*J. Comput. Chem.* **21**, 132 (2000) and **23**, 1623 (2002).

The **bond-charge-correction parameters** shipped in `src/data/bccparm.dat` are the exact
antechamber `BCCPARM.DAT` file from **AmberTools** (405 corrections). The BCC atom-type scheme
follows antechamber's `ATOMTYPE_BCC.DEF`, retained for reference under
`third_party/antechamber/ATOMTYPE_BCC.DEF`.

- Source: AmberTools / antechamber (D. A. Case *et al.*, *AMBER*; the antechamber tool is
  J. Wang *et al.*, *J. Mol. Graph. Model.* **25**, 247 (2006)), as redistributed in the
  `choderalab/ambermini` package.
- License: **GNU GPL v3** (AmberTools 14), which is compatible with this crate's
  GPL-3.0-or-later license.

**Parity note.** The BCC *parameters* are exact. The atom/bond typing in `src/bcc.rs`
reimplements the common `ATOMTYPE_BCC.DEF` rules from geometry-perceived topology — faithful
for typical organic molecules (e.g. ethanol charges match antechamber closely) but **not**
guaranteed byte-identical to antechamber's full definition-file matching engine and
penalty-based bond-order perception for every edge case (notably the aromatic/delocalized bond
subtypes 6/8/9/10/11). Full byte-exact parity is a documented larger effort.

## 4. Note on GPL compatibility

`am1-rs` is licensed GPL-3.0-or-later. The BSD-3-Clause terms of the ported PySEQM material
are GPL-compatible; the PySEQM copyright notice is retained here and in the relevant source
files, satisfying BSD-3 clause 1. The AM1 parameter values themselves are published
scientific facts and are not subject to copyright.
