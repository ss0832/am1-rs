# Rust API

Complete guide to the `am1_rs` crate's public API. Add the dependency and import from the
crate root (all the main types are re-exported there):

```toml
[dependencies]
am1-rs = "0.1"
```

```rust
use am1_rs::{
    // system
    Molecule, Atom, Vec3, symbol_to_z, z_to_symbol,
    // parameters
    Am1Parameters, Am1Element,
    // SCF
    Am1Calculator, Am1Options, Am1Result, ScfReference, ScfAccelerator, run_am1,
    // gradient
    closed_form_gradient, analytic_gradient, numerical_gradient, GradientResult,
    fixed_density_gradient, electronic_gradient_fixed_density, energy_at_fixed_density,
    // hessian / vibrations
    analytic_hessian, numerical_hessian, vibrational_analysis, VibrationalModes,
    // optimizer
    optimize, OptOptions, OptResult,
    // AM1-BCC
    am1_bcc_charges, BccResult,
    // errors / linalg
    Am1Error, Result, Matrix,
};
```

## Units

| Quantity | Unit |
|---|---|
| Input geometry (XYZ) | Ångström (converted to Bohr on load) |
| Positions in `Atom`/`Molecule` | **Bohr** |
| `*_ev` energy fields, orbital energies | **eV** (AM1's native unit) |
| Gradient `.gradient` / forces | **eV/Bohr** |
| Hessian (`analytic_hessian`) | **eV/Bohr²** |
| Heat of formation | **kcal/mol** |
| Dipole | Debye |

To convert eV → Hartree use `am1_rs::constants::EV_TO_HARTREE`; Bohr ↔ Å use
`ANGSTROM_TO_BOHR` / `BOHR_TO_ANGSTROM`.

> **Charge, multiplicity & reference** are taken from [`Am1Options`] (`charge`, `multiplicity`,
> `reference`), not from the `Molecule` fields, when you call `run_am1` / `Am1Calculator::calculate`
> / gradients / Hessian / optimizer. Set them on the options struct. `multiplicity` (2S+1) fixes
> the α/β electron counts; `reference: ScfReference` chooses the spin treatment independently —
> `Auto` (RHF closed shell, UHF open shell), `Restricted` (force RHF; requires a closed shell), or
> `Unrestricted` (force UHF, even for a singlet).

## 1. Building a molecule

```rust
// From an XYZ file or string (coordinates in Å, second arg is total charge):
let mol = Molecule::from_xyz_file("examples/water.xyz", 0.0)?;
let mol = Molecule::from_xyz_str("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n", 0.0)?;

// Programmatically (positions in Bohr):
let mol = Molecule::new(vec![
    Atom { z: 8, position: Vec3::new(0.0, 0.0, 0.0) },
    Atom { z: 1, position: Vec3::new(1.81, 0.0, 0.0) },
])
.with_charge(0.0)
.with_multiplicity(1);

mol.len();                    // number of atoms
symbol_to_z("Cl");            // Some(17)
z_to_symbol(6);               // Some("C")
```

`Molecule { atoms: Vec<Atom>, charge: f64, multiplicity: usize }`;
`Atom { z: u8, position: Vec3 /* Bohr */ }`.

## 2. Parameters

```rust
let params = Am1Parameters::standard()?;      // embedded published AM1 set
let carbon = params.element(6)?;              // &Am1Element (U_ss, zetas, betas, gauss, …)
// Custom/override set from a CSV string in the same column layout:
let params = Am1Parameters::from_csv(std::fs::read_to_string("my_params.csv")?.as_str())?;
```

## 3. Single point

```rust
let opts = Am1Options { charge: 0.0, multiplicity: 1, ..Default::default() };
// Force an unrestricted singlet (e.g. broken-symmetry start):
let uhf = Am1Options { reference: ScfReference::Unrestricted, ..Default::default() };

// Option A — calculator wrapper:
let calc = Am1Calculator::with_options(params.clone(), opts.clone()); // or ::new(params)
let r: Am1Result = calc.calculate(&mol)?;

// Option B — free function:
let r: Am1Result = run_am1(&mol, &params, &opts)?;

println!("ΔHf   = {:.3} kcal/mol", r.heat_of_formation_kcal);
println!("total = {:.4} eV", r.total_ev);
println!("dipole= {:.3} D", r.dipole_magnitude);
println!("q     = {:?}", r.charges);
println!("HOMO/LUMO = {:?}/{:?} eV", r.homo_ev, r.lumo_ev);
```

**`Am1Options`** (with `Default`):

| Field | Meaning | Default |
|---|---|---|
| `charge: f64` | total charge | `0.0` |
| `multiplicity: usize` | spin multiplicity 2S+1 | `1` |
| `reference: ScfReference` | `Auto` / `Restricted` / `Unrestricted` | `Auto` |
| `max_scf: usize` | max SCF iterations | `200` |
| `e_tol: f64` | energy convergence (eV) | `1e-8` |
| `p_tol: f64` | density RMS convergence | `1e-7` |
| `use_diis: bool` | `false` disables acceleration | `true` |
| `accelerator: ScfAccelerator` | `None` / `Cdiis` / `AdiisCdiis` | `AdiisCdiis` |
| `adiis_switch: f64` | commutator norm to switch A-DIIS→CDIIS | `0.1` |

**`Am1Result`** fields: `density`, `spin_density: Option<Matrix>` (UHF), `mo_energies: Vec<f64>`,
`mo_coeff: Matrix`, `n_occ`, `electronic_ev`, `core_ev`, `total_ev`, `heat_of_formation_kcal`,
`charges: Vec<f64>`, `dipole_debye: Vec3`, `dipole_magnitude`, `homo_ev/lumo_ev: Option<f64>`,
`iterations`, `converged: bool`, `unrestricted: bool`.

## 4. Nuclear gradient

```rust
// Primary: fully closed-form (dual-number AD), what the optimizer uses.
let g: GradientResult = closed_form_gradient(&mol, &params, &opts)?;
println!("|grad|max = {:.3e} eV/Bohr", g.max_gradient);
for f in &g.forces { /* eV/Bohr, = −gradient */ }

// Hellmann–Feynman with fixed-density finite-difference electronic term (step in Bohr):
let g = analytic_gradient(&mol, &params, &opts, 5.0e-4)?;
// Full-SCF finite-difference reference (slow; validation only):
let g = numerical_gradient(&mol, &params, &opts, 1.0e-4)?;
```

`GradientResult { scf: Am1Result, energy_ev, gradient: Vec<Vec3>, forces: Vec<Vec3>, max_gradient }`.

Lower-level fixed-density helpers (advanced): `energy_at_fixed_density(&mol, &params, &p)`,
`fixed_density_gradient(&mol, &params, &p)`, `electronic_gradient_fixed_density(&mol, &params, &basis, &p)`.

## 5. Geometry optimization (L-BFGS)

```rust
let res: OptResult = optimize(&mol, &params, &opts, &OptOptions::default())?;
println!("converged = {} in {} steps", res.converged, res.iterations);
println!("ΔHf = {:.3} kcal/mol", res.scf.heat_of_formation_kcal);
let relaxed: &Molecule = &res.molecule;         // optimized geometry (positions in Bohr)
for step in &res.trajectory { /* energy_ev, max_gradient, positions per step */ }
```

**`OptOptions`**: `max_iter` (200), `gtol` (eV/Bohr, `1e-3`), `grad_step` (Bohr, `5e-4`),
`history` (L-BFGS memory, `8`).
**`OptResult`**: `molecule`, `scf: Am1Result`, `converged`, `iterations`, `trajectory: Vec<OptStep>`
where `OptStep { energy_ev, heat_of_formation_kcal, max_gradient, positions: Vec<Vec3> }`.

## 6. Hessian & vibrational frequencies

```rust
// Analytic (CPHF) Hessian, eV/Bohr² (the `step` seeds internal FD only in fallback paths).
let h: Matrix = analytic_hessian(&mol, &params, &opts, 1.0e-3)?;

// Harmonic analysis (mass-weighted, diagonalized). Evaluate at a minimum for real modes.
let vib: VibrationalModes = vibrational_analysis(&res.molecule, &params, &opts, 1.0e-3)?;
for f in &vib.frequencies_cm { /* cm⁻¹, ascending; negatives = imaginary */ }

// Finite-difference Hessian reference:
let h = numerical_hessian(&mol, &params, &opts, 1.0e-3)?;
```

`VibrationalModes { hessian: Matrix, frequencies_cm: Vec<f64>, eigenvalues: Vec<f64> }`.

## 7. AM1-BCC partial charges (AMBER)

```rust
let bcc: BccResult = am1_bcc_charges(&mol, &params, &opts)?;
println!("AM1-BCC charges = {:?}", bcc.charges);     // Σ = total charge
println!("Mulliken (pre)  = {:?}", bcc.mulliken);
println!("BCC atom types  = {:?}", bcc.atom_types);  // e.g. ["31", "91", "91"]

am1_rs::bcc::write_mol2("out.mol2", &mol, &bcc)?;    // Tripos MOL2 with the charges
```

`BccResult { charges: Vec<f64>, atom_types: Vec<String>, mulliken: Vec<f64> }`.

## 8. Errors

Every fallible call returns `am1_rs::Result<T>` (= `Result<T, Am1Error>`). Variants include
`Io`, `Parse`, `InvalidInput`, `MissingElement(u8)`, `MissingParameter(String)`,
`LinearAlgebra(String)`, and `ScfNotConverged { iterations, error }`.

## End-to-end example

```rust
use am1_rs::*;

fn main() -> Result<()> {
    let params = Am1Parameters::standard()?;
    let opts = Am1Options::default();               // RHF, SAD guess, A-DIIS→CDIIS
    let mol = Molecule::from_xyz_file("examples/water.xyz", 0.0)?;

    // optimize, then vibrations at the minimum
    let opt = optimize(&mol, &params, &opts, &OptOptions::default())?;
    let vib = vibrational_analysis(&opt.molecule, &params, &opts, 1.0e-3)?;

    println!("ΔHf = {:.2} kcal/mol", opt.scf.heat_of_formation_kcal);
    println!("frequencies (cm^-1): {:?}",
        vib.frequencies_cm.iter().rev().take(3).collect::<Vec<_>>());

    let bcc = am1_bcc_charges(&opt.molecule, &params, &opts)?;
    println!("AM1-BCC = {:?}", bcc.charges);
    Ok(())
}
```

See also [`rust-api` module docs](https://docs.rs) via `cargo doc --open`, the CLI in
[`README.md`](../README.md), and [`scope.md`](scope.md) for the feature matrix.
