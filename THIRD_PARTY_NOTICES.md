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

## 3. RM1 model parameters — MOPAC (Apache-2.0)

RM1 is a reparameterization of AM1 with an identical functional form:

- **Primary scientific source** — G. B. Rocha, R. O. Freire, A. M. Simas, J. J. P. Stewart,
  "RM1: A Reparameterization of AM1 for H, C, N, O, P, S, F, Cl, Br and I,"
  *J. Comput. Chem.* **27**, 1101–1111 (2006).

The machine-readable table shipped in `src/data/rm1_parameters.csv` was extracted from
**MOPAC**'s Fortran tabulation `src/models/parameters_for_RM1_C.F90` by
`tools/extract_rm1_parameters.py`, which is retained in the repository so the extraction is
reproducible and auditable rather than a set of hand-copied numbers.

- Source: MOPAC (J. J. P. Stewart, *Molecular Orbital PACkage*;
  https://github.com/openmopac/mopac), commit `052691223d19935a89f0fe18cd12301bd83e4201`.
- License: **Apache License 2.0**, Copyright 2021 Virginia Polytechnic Institute and State
  University. The full license is retained at `third_party/mopac/LICENSE`. Apache-2.0 is
  compatible with GPL-3.0-or-later in the direction used here (Apache-2.0 material may be
  incorporated into a GPL-3 work).

Only RM1's published main-group set (H, C, N, O, F, P, S, Cl, Br, I) is included. RM1's
lanthanide parameters (Z = 57–71) require d/f orbitals and the sparkle model, neither of which
this crate implements; shipping them would produce a parameter block that loads and then
computes nonsense.

**Constants note.** This crate uses MOPAC7's historical physical constants (`a0 = 0.529167`,
`1 au = 27.21 eV`, `1 eV = 23.061 kcal/mol`) because the AM1 and RM1 parameters were fitted
against them; modern MOPAC defaults to CODATA values. Against MOPAC 22's own reference outputs
for CO₂ this leaves a systematic offset of about **+0.03 kcal/mol** in the heat of formation,
identically for AM1 and RM1, while optimized bond lengths agree to ~1e-4 Å and Koopmans
ionization potentials to ~4e-4 eV. See `tests/mopac_reference.rs`, which measures this.

## 4. AM1-BCC

The AM1-BCC method is due to A. Jakalian, B. L. Bush, D. B. Jack, C. I. Bayly,
*J. Comput. Chem.* **21**, 132 (2000) and **23**, 1623 (2002).

Two antechamber files are used, both verbatim and both compiled into every binary and wheel by
`include_str!`:

- `src/data/bccparm.dat` — the exact `BCCPARM.DAT`, 405 bond charge corrections.
- `third_party/antechamber/ATOMTYPE_BCC.DEF` — the exact atom-type definition file. Since 0.2.2
  this is not merely "retained for reference": `src/bcc/atomtype.rs` **parses and evaluates it**,
  so it is a source input to the build rather than documentation.

- Source: AmberTools / antechamber (D. A. Case *et al.*, *AMBER*; the antechamber tool is
  J. Wang *et al.*, *J. Mol. Graph. Model.* **25**, 247 (2006)), as redistributed in the
  `choderalab/ambermini` package (the AmberTools 14 vintage).
- License: **GNU GPL v3**. The full text is retained at `third_party/antechamber/LICENSE`, and
  the provenance of each file at `third_party/antechamber/README.md`.

  > **Corrected in 0.2.2.** Through 0.2.1 that licence text was **not retained anywhere in the
  > repository**, while MOPAC's and PySEQM's were — so GPL-3 material was compiled into every
  > wheel and every crates.io package without its licence travelling with it. The CI job that
  > checks "third-party licences are inside the wheel" did not catch it because it asserted only
  > that the list was non-empty; it now compares the count against the number of `third_party/`
  > subdirectories, so a fourth bundled work cannot repeat this.

**Parity note (measured).** The BCC *parameter values* are exact, and since 0.2.2 the atom typing
is no longer a transcription either — the definition file is interpreted, so the rules and their
order are antechamber's. What remains a reimplementation is the *perception* underneath: rings,
aromaticity and bond orders from geometry, where antechamber uses its own penalty-based bond-order
assignment. The coverage below is measured against the parameter file rather than estimated.

`BCCPARM.DAT` holds 405 entries across bond types 1, 2, 3, 6, 7, 8, 9, 10 and 11. The typing
emits types 1, 2, 3, 6, 7 and 9 — 339 entries. Of the 66 it does not emit:

- **type 11 (26 entries): every value is exactly 0.0.** These are the same-type-on-both-ends
  pairs, which antisymmetry forces to zero. Reaching them would change nothing.
- **types 8 and 10 (40 entries): byte-identical to type 7** for every one of the 15 and 25 pairs
  they respectively share with it. Verified entry by entry. Reaching them would change nothing.

So the practically unreachable set is empty: everything that can affect a charge is emitted.
Types 6 (nitro, N-oxide) and 9 (carboxylate, phosphate, sulfonate) were the 27 that genuinely
mattered — their corrections differ from the single-bond values by up to 0.2 e — and are now
selected by a chemical rule (a centre atom bearing two or more equivalent terminal O/S atoms, or
a four-coordinate nitrogen with one).

Perception uses the smallest ring through each bond, ring-size-aware aromaticity with a Hückel
4n+2 π count and a planarity test, and a bond-order reference table covering P and S pairs as
well as C/N/O.

**What is still not antechamber.** The matching engine is no longer the gap — the definition file
is interpreted. Two things remain:

- The *perception* is geometry-based (covalent radii, ring search, a Hückel π count, a bond-length
  table) where antechamber uses penalty-based bond-order assignment, so a molecule whose bond
  orders it would assign differently can still be typed differently.
- Two of the file's `23` rules carry a trailing `a1:a2:any` chain constraint. `any` is read as its
  name says — no restriction between the two labelled atoms — so those rules are applied on their
  structural part alone. This is a reading of the syntax rather than a transcription of it, and it
  is recorded in `src/bcc/atomtype.rs` rather than left silent.

The AR1..AR5 sub-classification is *not* a gap and was removed from this list on measurement: every
rule in this file asks for the union `[AR1.AR2]` and none asks for either class alone, so splitting
them would be machinery with no consumer. What the file does need beyond the union is its own
closing note — a five-membered ring fused to a six-membered aromatic one is not aromatic (indole) —
and that is implemented.

Anything the perception cannot do confidently — an element with no BCC atom type, or a typed bond
with no tabulated parameter — is reported in `BccResult::warnings` rather than silently skipped, so
a molecule that returns no warnings is one the rules covered.

## 5. Divide-and-conquer and periodic boundary conditions (methodology)

No third-party code or data is used for these; the formulations are cited so the implementation
can be checked against the literature.

- **Divide-and-conquer** — W. Yang, "Direct calculation of electron density in
  density-functional theory," *Phys. Rev. Lett.* **66**, 1438 (1991); W. Yang & T.-S. Lee,
  "A density-matrix divide-and-conquer approach for electronic structure calculations of large
  molecules," *J. Chem. Phys.* **103**, 5674 (1995). The subsystem projection used for the
  common chemical potential follows T. Akama, M. Kobayashi & H. Nakai, "Implementation of
  divide-and-conquer method including Hartree–Fock exchange interaction,"
  *J. Comput. Chem.* **28**, 2003 (2007).
- **Non-variational divide-and-conquer gradients** — K. Nishizawa, Y. Nishimura, M. Kobayashi,
  S. Irle & H. Nakai, *J. Comput. Chem.* **37**, 1983 (2016). The gradient in this version is
  the fixed-density (Hellmann–Feynman) one; the constraint correction described there is **not**
  implemented, and `docs/divide-conquer.md` measures the residual instead of claiming it away.
- **Ewald summation in reduced dimensionality** — D. E. Parry, "The electrostatic potential in
  the surface region of an ionic crystal," *Surf. Sci.* **49**, 433 (1975), for the 2D case.

  > **Corrected in 0.2.1.** This entry read "Ewald summation is not implemented in 0.2.0". It
  > was: `src/pbc/ewald.rs`, `ewald1d.rs` and `ewald2d.rs` all shipped in 0.2.0, and 0.2.1 adds
  > the phased kernel `EwaldSum::phased_pair_potential` for the `q ≠ 0` DFPT term. The sentence
  > was left over from an earlier draft of that release.
- **Molden file format** — the `[Molden Format]`, `[Atoms]`, `[STO]` and `[MO]` sections are
  written from the format's published description (G. Schaftenaar, *Molden*,
  <https://www.theochem.ru.nl/molden/molden_format.html>). No Molden source was consulted or
  ported; the writer in `src/molden.rs` is original.
- **Infrared intensity conversion** — the `42.2561 km/mol` prefactor for `|∂μ/∂Q|²` in
  `D²·Å⁻²·amu⁻¹` is the standard `N_A π/(3c²) · (4πε₀)⁻¹` grouping tabulated in, among others,
  J. Neugebauer, M. Reiher, C. Kind & B. A. Hess, *J. Comput. Chem.* **23**, 895 (2002). It is a
  unit conversion, not copyrightable expression; it is recorded here so the number has a source.
  `src/ir.rs` derives the `e → D·Å⁻¹` step from **this crate's** Bohr rather than CODATA's.

## 6. Note on GPL compatibility

`am1-rs` is licensed GPL-3.0-or-later. The BSD-3-Clause terms of the ported PySEQM material
are GPL-compatible; the PySEQM copyright notice is retained here and in the relevant source
files, satisfying BSD-3 clause 1. The Apache-2.0 terms of the MOPAC material are compatible
with GPL-3.0-or-later in the direction used here, and MOPAC's license and copyright notice
are retained at `third_party/mopac/LICENSE` and in the header of `src/data/rm1_parameters.csv`.
The AM1 and RM1 parameter values themselves are published scientific facts and are not
subject to copyright.

**On the antechamber material and "or later".** The licence file accompanying the ambermini
redistribution is the plain GPL-3 text and carries no separate "or (at your option) any later
version" grant, so this project does not claim one on Amber's behalf. (conda-forge's `ambertools`
feedstock declares modern AmberTools as `GPL-3.0-or-later`; that is a packager's declaration about
a later release, not a statement about the AmberTools 14 files retained here.) Under the
conservative reading — GPL-3.0 with no "or later" grant — a work combining it must be
distributable under **GPL-3.0**, and `am1-rs`'s own `GPL-3.0-or-later` permits exactly that. So
the crate's SPDX expression stays accurate for the crate's own code, while the *combination* that
is actually shipped should be understood as conveyable under GPL-3.0. Nothing about the packaging
changes; this is recorded so the reading is explicit rather than assumed.
