# am1-rs

A Rust-native implementation of the **AM1** (Austin Model 1) and **RM1** semiempirical NDDO
quantum-chemistry methods. It provides heats of formation, Mulliken and **AM1-BCC** partial
charges (for AMBER), analytic gradients and CPHF Hessians, L-BFGS geometry optimization,
**periodic boundary conditions** with k-points and analytic stress, and a **divide-and-conquer**
solver for large systems — with Rust, Python-native and ASE front ends.

AM1 is a genuine NDDO self-consistent-field method (Roothaan–Hall over a density matrix
with the Dewar–Sabelli–Klopman semiempirical two-electron integrals).

## Scope

[`docs/scope.md`](docs/scope.md) is the canonical matrix, including the known gaps. Summary:

| Capability | Status |
|---|---|
| **AM1** (21 elements) and **RM1** (10 elements) | ✅ same code path, `method="am1"\|"rm1"` |
| RHF/UHF NDDO SCF with **SAD** guess + **A-DIIS→CDIIS** convergence acceleration | ✅ |
| Mulliken charges, dipole, HOMO/LUMO | ✅ |
| Nuclear gradient — closed-form (dual-number AD), **RHF & UHF, all elements** | ✅ matches a full-SCF finite difference to 1e-9–1e-7 eV/Bohr, checked for all 21 elements |
| L-BFGS geometry optimization | ✅ molecules; and **under a cell since 0.2.3**, on the k-point forces and — with `relax_cell` — the analytic stress, at an optional target pressure |
| Analytic Hessian (CPHF/UCPHF) + harmonic frequencies, **RHF & UHF, all elements** | ✅ the Hessian is the raw Cartesian matrix; frequencies come from it with the rigid-body subspace **projected out** since 0.2.3, so `3N − 6` (or `3N − 5`, or `3N − 3`) modes come back and none of them is a translation |
| Open-shell **UHF** (radicals, odd-electron ions, `multiplicity > 1`) | ✅ |
| **Periodic boundary conditions** — 1D / 2D / 3D, Γ and k-points, RHF & UHF | ✅ energy, analytic forces, analytic stress |
| **Divide-and-conquer** — molecular **and periodic**, RHF & UHF, non-neutral | ✅ linear-scaling diagonalization; reachable from Rust, Python and ASE |
| AM1-BCC partial charges + mol2 export | ✅ exact `BCCPARM.DAT` values; ring perception, Hückel aromaticity, delocalized groups; warns on what it cannot type |
| Rust API, Python-native API, ASE `Calculator` | ✅ |
| Ewald summation, 1D / 2D (Parry) / 3D, monopole channel | ✅ |
| **DFPT at arbitrary q**, arbitrary k | ✅ arbitrary mesh or explicit k-list; DIIS-accelerated |
| Periodic Γ **and k-point** analytic Hessian; phonons `Φ(T)→D(q)` | ✅ since 0.2.3 the supercell blocks are filed under their **minimum image** and `Φ(T) = Φ(−T)ᵀ` is imposed alongside the acoustic sum rule, which is what makes the interpolated bands right and the three acoustic modes exactly zero. Validated against measured dispersions: polyethylene 1.5–5.3 %, graphene +11 to +22 % |
| Born charges `Z*`, dielectric tensor `ε_∞`, **LO–TO splitting** | ✅ Born charges in every dimensionality. `ε_∞` as a **constant** is 3D — below three the dielectric response is a *function* of `q` and tends to 1, which `dielectric_function` gives (that is physics, not a gap). LO–TO likewise: only 3D is discontinuous at Γ, so below three there is nothing to add and it is measured rather than argued. All reachable from Rust, Python and ASE |
| Divide-and-conquer under PBC, with **analytic stress** | ✅ |
| **External electric field** — energy, analytic gradient, analytic Hessian | ✅ molecules; and under a cell since 0.2.2 when the field is **orthogonal to every lattice vector** (normal to a slab, transverse to a chain). A component *along* a periodic direction is refused by name: `F·R` is unbounded there |
| **Infrared spectra** — atomic polar tensor (raw 3 × 3N) and km/mol intensities | ✅ |
| **Wavefunction output** — orbital energies/coefficients (both spins), Molden | ✅ `[GTO]` by default since 0.2.3 (a run-time-fitted Gaussian expansion, overlap > 0.99999 with the Slater shell it replaces), with `S^{−1/2}C` so the coefficients belong to the functions the file lists. `[STO]` is exact for this basis and kept behind `--molden-basis sto`, because almost no viewer reads that section |
| **First-order orbital response** `U^j_{ai}` from the CPHF | ✅ lazy; never runs from an energy call |
| Long-range monopole term **inside the DFPT response** | ✅ 3D cells, every `q`, via a phased Ewald sum: independent of the real-space cutoff to 2e-16 at `q = ¼`, against 1e-1 for the truncated sum alone. `D(q)` is then the *full* dynamical matrix — do not also apply `frequencies_with_lo_to`. 1D/2D have no such term and say so |
| **Berry-phase polarization** (King-Smith–Vanderbilt) | ✅ since 0.2.2 — modulo the polarization quantum, exactly translation-invariant, and `Ω ∂P/∂τ` reproduces the CPHF Born charges to 7.5e-13 e where the two formalisms are comparable. 3D |
| **Finite electric field along a periodic direction** | ✅ since 0.2.2 — the Berry-phase electric enthalpy `E − Ω 𝓔·P`, not `F·R`. `α = Ω ∂P/∂𝓔` matches the CPHF polarizability to 0.03–0.47 % where the two compute the same object, converging as `O(1/J²)` |
| **Open-shell (UHF) k-point response** — Hessian, Born charges | ✅ since 0.2.2 — two coupled spin channels. Forcing UHF on a closed shell reproduces the restricted force constants to 8.9e-16 eV/Bohr² |
| `ε_∞` for a chain or a slab | ✅ since 0.2.2 — `dielectric_with_extent`, with the thickness (slab) or cross-section (wire) a **required argument**, and the depolarization factor of the assumed body carried with it. What does not depend on that choice is reported alongside |
| **Periodic SCF convergence** | ✅ since 0.2.2 — Pulay mixing (140 → 22 iterations), a degenerate-level fix in the complex eigensolver (a methane slab could not converge at all before), and the energy evaluated where the functional is stationary. **Since 0.2.3 a fallback pipeline**: the controller reads the residual trace and answers a growing or oscillating one with Kerker preconditioning, a stalled one (ratio > 0.99) with a raised electronic temperature that is then **annealed back**, since smearing moves the fixed point and `converged` must mean converged at the smearing you asked for. `PbcResult::fallback` reports what it did |
| **Phonon eigenvectors** | ✅ since 0.2.3 — `eigenvectors=True` / `ForceConstants::modes`, in two conventions: `polarization` (orthonormal `e(q)`, for sums over modes) and `displacements` (`e_a/√m_a`, for moving a structure along a mode). Through 0.2.2 a mode could not be followed from a phonon result at all |
| **Divide-and-conquer geometry optimization**, and a CLI switch | ✅ since 0.2.3 — `--dc` on `energy`, `gradient` and `optimize`; L-BFGS on the DC gradient, so a structure too large for one full SCF can be relaxed and not only measured |
| SAM1 | ⛔ a different integral engine, not a reparameterization |


The **formulas themselves** are checked piece by piece in `tests/theory_components.rs`, because
an end-to-end identity says a chain is wrong without saying which link. The long-range form of
`(ss|ss)` *recovers* the Klopman–Ohno parameter — 1.9946 Bohr against the table's 1.994724 — and
each multipole channel's decay exponent is measured against the order its expansion demands
(`R^-0.998`, `R^-1.996`, `R^-2.997`). Alongside those: permutation symmetries, rotation
covariance of the whole pair block, overlap parity, `P² = 2P`, `[F,P] = 0` at convergence, and
`E = ½Tr[P(H+F)]`. See `tests/`.

**The ΔHf offset.** The AM1 heat of formation of water comes out **−59.22 kcal/mol** against
MOPAC's −59.24. The 0.03 kcal/mol difference is the deliberate choice of MOPAC7 model constants,
not a model error — AM1 and RM1 show the identical offset, which is what identifies it.

**Corrected in 0.2.1.** 0.2.0 applied the three-dimensional `ε_∞ = 1 + 4πα/Ω` and LO–TO formulas
to 1D chains, where `Ω` was silently a *length*. The "127 cm⁻¹ splitting on a polar chain" it
reported was an artifact; a genuinely 1D-periodic chain has no LO–TO splitting as `q → 0`. Both
now require a fully periodic cell. See [`docs/pbc.md`](docs/pbc.md).


## Build

```
cargo build --release            # library + CLI (am1_rs_cli)
cargo test                       # unit + validation tests
```

Linear algebra uses **[faer](https://faer-rs.github.io/)** (a pure-Rust symmetric
eigensolver and LU — no LAPACK/BLAS) and hot loops are parallelized with **rayon** (gradient
displacements, pair-integral construction, Hessian columns). `pyo3` is pulled in only for the
Python extension (`--features python`).

## Units

The semiempirical block is computed in **eV with distances in Bohr**, using MOPAC7's model
constants (`ev = 27.21`, `a0 = 0.529167 Å`) so published AM1 numbers reproduce. At the API
boundary:

- **Rust-native and Python-native APIs** return **atomic units (Hartree, Bohr)** for the
  molecular entry points.
- `pbc_point` and `divide_conquer` return **eV / Å** directly, because their consumer is ASE and
  converting a stress tensor twice is an easy way to be wrong.
- The **ASE `Calculator`** returns **eV / Å / eV Å⁻³** throughout.
- Heat of formation is additionally reported in **kcal/mol**.

## Parameters

The standard published AM1 parameter set (`U_ss … alpha` plus the AM1 core-core Gaussian
`K/L/M` triples) is embedded from `src/data/am1_parameters.csv`; isolated-atom energies and
experimental atomic heats of formation reproduce MOPAC's `calpar.f`/`block.f`. No external
parameter file is required.

**Provenance / attribution.** The AM1 parameter values are published scientific constants
(Dewar *et al.* 1985 + the element-extension papers, consolidated in MOPAC). The specific
machine-readable table, and the integral/overlap/Fock formulas that `am1-rs` was ported
from, come from the **PySEQM** reference implementation (LANL, **BSD-3-Clause**). Every
parameter's origin is documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), the
retained license at `third_party/pyseqm/LICENSE`, the CSV header, and the module doc-comments
that name the corresponding PySEQM source for each ported piece.

## CLI

```
am1_rs_cli energy    examples/water.xyz            # ΔHf, charges, dipole, HOMO/LUMO
am1_rs_cli gradient  examples/methane.xyz          # energy + gradient (Hartree/Bohr)
am1_rs_cli optimize  examples/water.xyz --opt-output opt.xyz
am1_rs_cli frequencies opt.xyz                             # 3N−6 vibrations (cm^-1)
am1_rs_cli orbitals  opt.xyz --orbital-coefficients        # energies, occupations, MO matrix
am1_rs_cli molden    opt.xyz --molden-output opt.molden    # wavefunction for a viewer
am1_rs_cli charges   examples/ethanol.xyz --mol2-output ethanol.mol2   # AM1-BCC
am1_rs_cli charges   examples/ethanol.xyz --mulliken                   # raw AM1 charges
```

Options: `--method am1|rm1`, `--charge Q`, `--multiplicity M` (`M > 1` requires UHF),
`--reference auto|rhf|uhf` (or the `--rhf` / `--uhf` shortcuts — `--uhf` runs a singlet
unrestricted), `--field FX FY FZ`, `--opt-output`, `--mol2-output`, `--molden-output`,
`--molden-basis gto|sto`, `--orbital-coefficients`.

### Periodic boundary conditions on the command line

A cell comes from an extended-XYZ `Lattice="…"` comment line, or from `--cell` — 1, 3, 6 or 9
numbers (cube; `a b c`; `a b c α β γ`; or three lattice vectors). Which axes are periodic is set
by `--pbc x|y|z|xy|xyz|none`, or **one axis at a time** with `--pbc-x` / `--pbc-y` / `--pbc-z`,
which accumulate. `--no-pbc` drops the cell and runs the contents of it as a molecule.

```
am1_rs_cli energy    slab.xyz --cell 8.0 --pbc-x --pbc-y --kpts 4 4 1
am1_rs_cli gradient  chain.xyz --kpts 1 1 8               # forces and stress
am1_rs_cli optimize  chain.xyz --kpts 1 1 8 --relax-cell  # atoms and lattice
am1_rs_cli frequencies chain.xyz --kpts 1 1 8             # phonons at q = 0
am1_rs_cli phonons   chain.xyz --supercell 1 1 4 --qpath 0 0 0 0 0 0.5
```

`energy`, `gradient`, `optimize` and `frequencies` take the periodic path whenever a cell is
present; `charges`, `ir` and `molden` are molecular-only and say so rather than ignoring it.
`--kpts` is the SCF's Monkhorst–Pack mesh; `phonons` takes `--supercell` and then either
`--qpoints` (a list) or `--qpath` with `--qpath-points`, defaulting to the supercell's
commensurate `q`.

`pip install am1-rs-python` puts the same CLI on `PATH` as **`am1-rs`** (or `python -m am1_rs`,
where the scripts directory is not on `PATH`). A wheel cannot ship the Rust binary alongside the
extension module, so this is a Python front end over the same native bindings — same modes, same
flags, and byte-identical output. `tests/test_cli_matrix.py` checks that by running **every mode
against every combination of the options that apply to it** and diffing the two, rather than
leaving it as a claim.

```bash
am1-rs energy examples/water.xyz
```

### Large systems on the command line

`--dc` runs the linear-scaling divide-and-conquer SCF instead of the full one, on `energy`,
`gradient` and `optimize`. `--dc-core N` sets the target atoms per core region (12) and
`--dc-buffer R` the buffer radius in Ångström (5.82); either implies `--dc`.

```
am1-rs energy   cluster.xyz --dc --dc-buffer 7.0
am1-rs optimize cluster.xyz --dc --opt-output relaxed.xyz
```

A mode `--dc` does not cover **refuses** rather than ignoring the flag: `frequencies --dc` would
return full-SCF frequencies at full-SCF cost, which is the one thing a user reaching for `--dc`
is trying to avoid, and silently doing it is worse than saying so. The buffer radius is the
accuracy knob, and it is not monotonic — see [`docs/divide-conquer.md`](docs/divide-conquer.md).

## Rust API

All Rust entry points are in **atomic units** (Hartree, Bohr) except the AM1 block's native eV
(the `*_ev` fields) and ΔHf in kcal/mol. `Am1Options` carries `charge`, `multiplicity` (2S+1),
`reference` (`Auto`/`Restricted`/`Unrestricted` — restricted vs unrestricted, independent of the
multiplicity), SCF tolerances, and the accelerator choice.

```rust
use am1_rs::{
    Molecule, Am1Parameters, Am1Calculator, Am1Options,
    closed_form_gradient, optimize, vibrational_analysis, am1_bcc_charges,
    OptOptions,
};

let params = Am1Parameters::standard()?;          // embedded AM1 parameter set
let mol = Molecule::from_xyz_file("examples/water.xyz", 0.0)?;   // charge = 0.0
let opts = Am1Options::default();                 // RHF, A-DIIS→CDIIS, SAD guess

// Single point: energy, Mulliken charges, dipole, HOMO/LUMO, ΔHf.
let r = Am1Calculator::with_options(params.clone(), opts.clone()).calculate(&mol)?;
println!("ΔHf = {:.3} kcal/mol   dipole = {:.3} D", r.heat_of_formation_kcal, r.dipole_magnitude);

// Analytic nuclear gradient (eV/Bohr in `.gradient`, forces = −gradient).
let g = closed_form_gradient(&mol, &params, &opts)?;
println!("|grad|max = {:.3e} eV/Bohr", g.max_gradient);

// L-BFGS geometry optimization.
let optd = optimize(&mol, &params, &opts, &OptOptions::default())?;
println!("relaxed ΔHf = {:.3} kcal/mol", optd.scf.heat_of_formation_kcal);

// Analytic Hessian → harmonic frequencies (evaluate at a minimum).
let vib = vibrational_analysis(&optd.molecule, &params, &opts, 1.0e-3)?;
println!("highest mode = {:.1} cm^-1", vib.frequencies_cm.last().copied().unwrap_or(0.0));

// AM1-BCC partial charges for AMBER.
let bcc = am1_bcc_charges(&mol, &params, &opts)?;
println!("BCC charges = {:?}", bcc.charges);
```

## Python / ASE

The distribution is `am1-rs-python` (import module `am1_rs`):

```bash
pip install am1-rs-python
```

Wheels are published for manylinux and musllinux (x86_64, aarch64), macOS (x86_64, arm64) and
Windows x64, so this normally compiles nothing. Where no wheel matches, `pip` falls back to the
sdist and builds the crate under the shipping profile — fat LTO and a single codegen unit, which
was measured at **1.9 GB peak resident and around six minutes** on a warm dependency cache. On a
memory-capped machine that is an out-of-memory failure rather than a slow install. Thin LTO and
more codegen units cut the linker's peak allocation for a few percent of run-time speed, and
cargo's profile environment variables select them without touching the project:

```bash
CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 pip install am1-rs-python
```

For development, build with maturin (`maturin develop --features python`).

**ASE calculator** (eV / Å convention; `multiplicity > 1` selects UHF):

```python
from am1_rs.ase import AM1
from ase.build import molecule

atoms = molecule("H2O")
atoms.calc = AM1(charge=0.0, multiplicity=1)   # ASE convention: eV, Å
print(atoms.get_potential_energy())            # eV
print(atoms.get_forces())                      # eV/Å
print(atoms.get_charges())                     # Mulliken, e
print(atoms.calc.get_hessian(atoms))           # analytic Hessian, eV/Å²
```

Charge, multiplicity and reference may be given at construction (e.g.
`AM1(reference="uhf")`) or per structure at calculation time via `atoms.info["charge"]` /
`atoms.info["multiplicity"]` / `atoms.info["reference"]` (`"auto"`/`"rhf"`/`"uhf"`).

**Native API** — atomic units (Hartree, Bohr); input coordinates in Å. Each call takes
`(numbers, positions, charge=0.0, multiplicity=1, reference="auto")` (`reference` = `"auto"` /
`"rhf"` / `"uhf"`) and returns a dict (keys documented in each function's docstring):

```python
import am1_rs

Z = [8, 1, 1]
xyz = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]]

sp   = am1_rs.single_point(Z, xyz)          # energy_hartree, heat_of_formation_kcal, charges, dipole_debye, homo/lumo_ev…
grad = am1_rs.gradient(Z, xyz)              # gradient_hartree_per_bohr, gradient_ev_per_angstrom
opt  = am1_rs.optimize(Z, xyz)              # positions_angstrom, energy_hartree, converged, iterations
freq = am1_rs.frequencies(Z, opt["positions_angstrom"])   # frequencies_cm (evaluate at a minimum)
hess = am1_rs.hessian(Z, xyz)               # hessian_hartree_per_bohr2, hessian_ev_per_angstrom2, ndof
bcc  = am1_rs.am1_bcc(Z, xyz)               # charges (AM1-BCC), mulliken, atom_types
print(sp["heat_of_formation_kcal"], bcc["charges"])
```

### Periodic systems

Dimensionality follows `atoms.pbc`, exactly as in ASE — one calculator covers chains, slabs and
crystals. Read [`docs/pbc.md`](docs/pbc.md) before using this in anger.

```python
import numpy as np
from ase import Atoms
from am1_rs.ase import AM1

crystal = Atoms("OH2", positions=xyz, cell=np.eye(3) * 6.0, pbc=True)
crystal.calc = AM1(kpts=(2, 2, 2), exchange_cutoff=10.0)
print(crystal.get_potential_energy())   # eV per cell
print(crystal.get_stress())             # Voigt, eV/Å³
```

`tests/test_ase_pbc_md.py` runs real NVE / NPT / NVT ensembles and is the best worked example.

### Large systems: divide-and-conquer

```python
atoms.calc = AM1(divide_conquer=True, core_size=12, buffer_radius=11.0)
```

Increase `buffer_radius` until the property you care about stops moving; a buffer covering the
whole molecule reproduces the full SCF exactly. See
[`docs/divide-conquer.md`](docs/divide-conquer.md).

## AM1-BCC charges for AMBER

`am1_rs_cli charges` runs the AM1 SCF for Mulliken charges, perceives the molecular graph,
assigns antechamber **BCC atom types** (the numeric 11–91 scheme of `ATOMTYPE_BCC.DEF`), and
applies the additive bond charge corrections from the **exact antechamber `BCCPARM.DAT`** (405
parameters, embedded from AmberTools, GPL-3). Ethanol charges match antechamber closely
(O −0.60, hydroxyl-H +0.40, …). The atom/bond typing is faithful for common organic molecules
but not guaranteed byte-identical to antechamber's full typing engine for every edge case —
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Net charge is preserved.

## Documentation

[**`docs/`**](docs/README.md) has an index that routes by question rather than by file. The
documents themselves:

- [`docs/python-api.md`](docs/python-api.md) — complete Python API (native functions + ASE calculator).
- [`docs/rust-api.md`](docs/rust-api.md) — complete Rust API (every public type and function, with examples).
- [`docs/scope.md`](docs/scope.md) — feature matrix, known gaps, units contract. Each entry says
  what was **measured**, not what was intended; it is the right place to check a claim made
  anywhere else.
- [`docs/methods.md`](docs/methods.md) — AM1 vs RM1, element coverage.
- [`docs/pbc.md`](docs/pbc.md) — periodic boundary conditions: setup, conventions, **and the
  limitations, which matter**.
- [`docs/divide-conquer.md`](docs/divide-conquer.md) — the formulation, and a precise statement
  of what became linear.
- [`docs/theory.md`](docs/theory.md) — equations, CPHF, and references.


## References

- **AM1** — M. J. S. Dewar, E. G. Zoebisch, E. F. Healy, J. J. P. Stewart,
  *J. Am. Chem. Soc.* **107**, 3902 (1985).
- **MNDO integrals / core-core** — M. J. S. Dewar, W. Thiel,
  *J. Am. Chem. Soc.* **99**, 4899 (1977).
- **AM1-BCC** — A. Jakalian, B. L. Bush, D. B. Jack, C. I. Bayly,
  *J. Comput. Chem.* **21**, 132 (2000); A. Jakalian, D. B. Jack, C. I. Bayly,
  *J. Comput. Chem.* **23**, 1623 (2002).
- **ASE** — A. Hjorth Larsen *et al.*, "The Atomic Simulation Environment — a Python library
  for working with atoms," *J. Phys.: Condens. Matter* **29**, 273002 (2017).
- Reference integral/rotation formulas cross-checked against the PySEQM implementation
  (Zhou, Nebgen *et al.*), which reproduces MOPAC.

## License

```
Copyright (C) 2024-2026 ss0832

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
```

The full text is in [`LICENSE`](LICENSE), kept verbatim. Bundled third-party material — the
PySEQM-derived parameter tables (BSD-3-Clause), the antechamber `BCCPARM.DAT` and
`ATOMTYPE_BCC.DEF` (GPL-3), and the MOPAC-derived RM1 parameters and per-element reference tables
(Apache-2.0) — is documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), with the
retained licence and a per-work provenance note under `third_party/`.

The linked Rust crates are a separate and larger set — over a hundred of them end up statically
inside `am1_rs._native` and the CLI binary — and their notices are in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), generated from the resolved dependency graph
by `tools/collect_dependency_licenses.py`.

Every one of those files ships inside the wheel, under `dist-info/licenses/`, and in the sdist and
the `.crate`: MIT requires its copyright and permission notice to accompany a copy and a statically
linked binary is one, BSD-3-Clause clause 2 says the same of a binary redistribution, and
Apache-2.0 §4(a)–(c) want the licence text, the retained attribution and the statement of changes.
CI fails the release if any of them is missing, and `tests/attribution.rs` fails the build if the
inputs that determine them drift.

[`gfn1-rs`]: https://github.com/ss0832/gfn1-rs_proto
