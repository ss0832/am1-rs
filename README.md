# am1-rs

A Rust-native implementation of the **AM1** (Austin Model 1) semiempirical NDDO
quantum-chemistry method. It provides
AM1 heats of formation, Mulliken and **AM1-BCC** partial charges (for AMBER), nuclear
gradients, and L-BFGS geometry optimization, with Rust, Python-native and ASE front ends.

AM1 is a genuine NDDO self-consistent-field method (Roothaan–Hall over a density matrix
with the Dewar–Sabelli–Klopman semiempirical two-electron integrals).

## Scope

| Capability | Status |
|---|---|
| RHF/UHF NDDO SCF with **SAD** guess + **A-DIIS→CDIIS** convergence acceleration | ✅ validated |
| AM1 Mulliken charges, dipole, HOMO/LUMO | ✅ |
| Nuclear gradient — fully closed-form (dual-number AD), **RHF & UHF, all elements**, no finite differences | ✅ matches full-SCF FD to ~1e-7 |
| L-BFGS geometry optimization (Rust) | ✅ |
| Analytic Hessian (CPHF/UCPHF orbital Hessian) + harmonic frequencies — **RHF & UHF, all elements**, no finite differences | ✅ rayon-parallel, faer-diagonalized; matches FD Hessian to ~1e-5 |
| Open-shell **UHF** (radicals, odd-electron ions, `multiplicity > 1`) | ✅ |
| AM1-BCC partial charges + mol2 export | ✅ exact antechamber BCCPARM.DAT (405 params) + faithful typing |
| Rust API, Python-native API, ASE `Calculator` | ✅ |
| Elements | full published AM1 main-group set: H, C, N, O, F, Al, Si, P, S, Cl (n ≤ 3, exact analytic overlap) and Zn, Ge, As, Se, Br, Sb, Te, I, Hg (n ≥ 4, general numerical overlap) |
| Periodic boundary conditions (PBC) | ⛔ roadmap (AM1 is molecular) |

**Validation.** Optimized AM1 heat of formation of water is **−59.24 kcal/mol** and dipole
**1.86 D**, matching MOPAC AM1. See `tests/`.


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

- **Rust-native and Python-native APIs** return **atomic units (Hartree, Bohr)**.
- The **ASE `Calculator`** returns **eV / Å** (AM1 energies are natively eV).
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
am1_rs_cli gradient  examples/methane.xyz          # energy + forces (eV/Å)
am1_rs_cli optimize  examples/water.xyz --opt-output opt.xyz
am1_rs_cli frequencies opt.xyz                             # harmonic frequencies (cm^-1)
am1_rs_cli charges   examples/ethanol.xyz --mol2-output ethanol.mol2   # AM1-BCC
am1_rs_cli charges   examples/ethanol.xyz --mulliken                   # raw AM1 charges
```

Options: `--charge Q`, `--multiplicity M` (`M > 1` requires UHF), `--reference auto|rhf|uhf`
(or the `--rhf` / `--uhf` shortcuts — `--uhf` runs a singlet unrestricted), `--opt-output`,
`--mol2-output`.

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

Built with maturin (`maturin develop --features python`). The distribution is
`am1-rs-python` (import module `am1_rs`):

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

## AM1-BCC charges for AMBER

`am1_rs_cli charges` runs the AM1 SCF for Mulliken charges, perceives the molecular graph,
assigns antechamber **BCC atom types** (the numeric 11–91 scheme of `ATOMTYPE_BCC.DEF`), and
applies the additive bond charge corrections from the **exact antechamber `BCCPARM.DAT`** (405
parameters, embedded from AmberTools, GPL-3). Ethanol charges match antechamber closely
(O −0.60, hydroxyl-H +0.40, …). The atom/bond typing is faithful for common organic molecules
but not guaranteed byte-identical to antechamber's full typing engine for every edge case —
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Net charge is preserved.

## Documentation

- [`docs/rust-api.md`](docs/rust-api.md) — complete Rust API (every public type and function, with examples).
- [`docs/python-api.md`](docs/python-api.md) — complete Python API (native functions + ASE calculator).
- [`docs/scope.md`](docs/scope.md) — feature matrix and units contract.
- [`docs/theory.md`](docs/theory.md) — AM1 equations and references.

## Limitations

- Open-shell systems use spin-unrestricted **UHF** (`--multiplicity M`, `M > 1`); no
  spin-contamination annihilation or ROHF yet.
- Gradients use the exact NDDO **Hellmann–Feynman** formula (no Pulay term — the AO basis is
  orthonormal) and are **fully closed-form for every case** — RHF and UHF, all elements. Each
  term (two-electron, core-attraction, overlap radial *and* angular, core-core) is forward-mode
  dual-number AD with **no finite differences**; for heavy elements (`n ≥ 4`) the derivative is
  AD *through* the numerical Slater-overlap quadrature. Open-shell (UHF) uses the spin-resolved
  fixed-density gradient. All match a full-SCF finite difference to ~1e-7.
- The Hessian is **fully analytic for every case** — closed-shell RHF *and* open-shell UHF, all
  elements — with **no finite differences**. The skeleton (fixed-density) second derivative is
  closed-form **second-order** forward-AD ([`Dual2`](src/dual2.rs)) of the integral kernels
  (through the quadrature for `n ≥ 4`); the orbital-relaxation term solves the **CPHF** equations
  against the orbital Hessian (RHF), or the **coupled α/β UCPHF** equations (UHF), in the compact
  MO occupied–virtual subspace. It agrees with an independent finite-difference Hessian to within
  that reference's own truncation error (~1e-5). Harmonic frequencies come from its mass-weighted
  diagonalization. (A finite-difference gradient/Hessian is retained only as a validation gate.)
- Overlaps for valence shells `n ≤ 3` use the exact analytic Slater kernel; `n ≥ 4` (heavy AM1
  elements) use a general numerical Slater overlap (Gauss–Legendre quadrature) whose value and
  derivatives are both consistent (the derivatives are taken analytically through the quadrature),
  validated to reproduce the analytic kernel to ~1e-8.
- AM1-BCC uses the exact antechamber `BCCPARM.DAT` parameters; the atom/bond typing is faithful
  for common organic molecules but not byte-identical to antechamber's full engine for every edge
  case (see `THIRD_PARTY_NOTICES.md`).

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

GPL-3.0-or-later.

[`gfn1-rs`]: https://github.com/ss0832/gfn1-rs_proto
