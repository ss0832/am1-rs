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

`VibrationalModes` also carries `modes` (mass-weighted eigenvectors, columns are modes and are
orthonormal), `cartesian_displacements` (`M^{−1/2}L`, deliberately not renormalized) and
`translation_rotation_overlap` — each mode's share of rigid-body motion, so a linear molecule's
five rigid-body modes are discovered from the eigenvectors rather than assumed from `3N − 6`.

`analytic_hessian_with_response` returns the CPHF solution the Hessian was built from instead of
discarding it:

```rust
let r: HessianResponse = analytic_hessian_with_response(&mol, &params, &opts, 1.0e-3)?;
r.hessian;                    // the same matrix `analytic_hessian` returns
r.alpha.u_ov[dof];            // n_vir × n_occ CPHF block for one Cartesian perturbation
r.response_density(dof);      // ∂P/∂R_j in the AO basis, built on demand
r.beta;                       // Some(..) for an unrestricted run
```

The response densities are built on demand rather than stored: all `3N` at once is the largest
array in the calculation and is almost never wanted whole.

## 6b. External electric field

```rust
let opts = Am1Options {
    // eV per (e·Bohr) — the crate's internal unit.
    electric_field: Some(Vec3::new(0.0, 0.0, 0.136)),
    ..Am1Options::default()
};
let g = closed_form_gradient(&mol, &params, &opts)?;   // force on atom a gains +Q_a F
let h = analytic_hessian(&mol, &params, &opts, 1.0e-3)?;
```

`E(F) = E₀ − μ·F`. `src/dipole.rs` holds the operator and the one sign convention every consumer
shares; `Am1Result::field_nuclear_ev` reports the nuclear half, the electronic half already being
inside `electronic_ev`. Under a cell the field must be **orthogonal to every lattice vector** (normal to a slab, transverse to a chain); a component along a periodic direction is an error naming itself, because `F·R` is unbounded there. `PbcOptions::electric_field` is the periodic counterpart. Refused for any cell before 0.2.2.

## 6c. Infrared intensities and the atomic polar tensor

```rust
let apt = am1_rs::ir::dipole_derivatives(&mol, &params, &opts)?;   // 3 × 3N, units of e
let s: IrSpectrum = am1_rs::ir::ir_spectrum(&mol, &params, &opts)?;
s.intensities_km_per_mol;
s.mode_dipole_derivatives;              // dense per-mode tensor, keeps the dipole direction
s.vibrational_bands(0.5);               // (index, cm⁻¹, km/mol) for the non-rigid-body modes
```

Both run an analytic Hessian; `ir_spectrum` reuses the one CPHF solve for the tensor and the
modes. `Σ_a ∂μ_α/∂R_{a,β} = q δ_αβ` is the sum rule that checks it.

## 6d. Wavefunction output

```rust
let scf = run_am1(&mol, &params, &opts)?;
let text = am1_rs::molden::to_molden(&mol, &params, &scf)?;
am1_rs::molden::write_molden("water.molden", &mol, &params, &scf)?;
```

`Am1Result` carries `mo_energies`, `mo_coeff`, `n_occ` and — for an unrestricted run — `beta`,
which UHF used to solve for and discard. See [theory.md](theory.md) for why the coefficients are
in an implicitly orthogonalized basis.

## 7. AM1-BCC partial charges (AMBER)

```rust
let bcc: BccResult = am1_bcc_charges(&mol, &params, &opts)?;
println!("AM1-BCC charges = {:?}", bcc.charges);     // Σ = total charge
println!("Mulliken (pre)  = {:?}", bcc.mulliken);
println!("BCC atom types  = {:?}", bcc.atom_types);  // e.g. ["31", "91", "91"]

am1_rs::bcc::write_mol2("out.mol2", &mol, &bcc)?;    // Tripos MOL2 with the charges
```

`BccResult { charges: Vec<f64>, atom_types: Vec<String>, mulliken: Vec<f64> }`.

## 8. Methods: AM1 and RM1

```rust
use am1_rs::{Am1Parameters, NddoMethod};

let am1 = Am1Parameters::standard()?;                       // AM1, 21 elements
let rm1 = Am1Parameters::for_method(NddoMethod::Rm1)?;      // RM1, 10 elements
```

The method travels with the parameter set, so every downstream path — gradients, Hessians,
periodic boundary conditions, divide-and-conquer — follows from the one argument. See
[methods.md](methods.md).

## 9. Periodic boundary conditions

```rust
use am1_rs::{Lattice, Molecule, Vec3, KMesh, PbcOptions, run_pbc_scf, pbc_energy_and_gradient};

// Lattice vectors are in BOHR here (the Rust surface's unit), not Ångström.
let crystal = mol.with_cell(Lattice::from_vectors(
    Vec3::new(a, 0.0, 0.0),
    Vec3::new(0.0, b, 0.0),
    Vec3::new(0.0, 0.0, c),
    [true, true, false],            // a slab; [true, true, true] for a crystal
)?);

let opts = PbcOptions {
    kmesh: KMesh::MonkhorstPack([4, 4, 1]),
    exchange_cutoff: Some(12.0),
    smearing_ev: 0.0,
    ..PbcOptions::default()
};

let scf = run_pbc_scf(&crystal, &params, &opts)?;
println!("{} eV per cell, {} k-points", scf.total_ev, scf.k_points);
if let Some(w) = &scf.charged_cell_warning { eprintln!("{w}"); }

let (scf, grad) = pbc_energy_and_gradient(&crystal, &params, &opts)?;
println!("stress (Voigt) = {:?}", grad.stress_voigt());   // eV/Bohr^d
```

`PbcResult` carries `total_ev`, `band_energy_ev`, `fermi_energy_ev`, `entropy_ev`, `charges`,
`density` / `spin_density` as `RealSpaceBlocks`, `k_points`, `max_image_overlap` and
`charged_cell_warning`, with `free_energy_ev()` and `extrapolated_energy_ev()`.
`PbcGradient` carries `gradient`, `forces`, `stress`, `stress_voigt()` and `pressure(d)`.

### Periodic response properties

```rust
use am1_rs::pbc::{pbc_hessian, born_charges, dielectric_tensor};
use am1_rs::pbc::{dielectric_tensor_with_extent, ExtentConvention};
use am1_rs::pbc::{frequencies_dfpt_with, DfptOptions, LongRange, KPoint};

// q = 0 force constants with k-point sampling — no exchange taper standing in for physics.
let h = pbc_hessian(&crystal, &params, &opts)?;

let z = born_charges(&crystal, &params, &opts)?;             // Z*, Σ_a Z*_a = 0
let (alpha, epsilon) = dielectric_tensor(&crystal, &params, &opts)?;   // 3D cells only

// Below three dimensions the volume is not the cell's, so it is named rather than assumed. The
// conversion carries the depolarization factor that goes with the body you just declared, so a
// slab's out-of-plane law is `1/(1 − 4πχ)` and not `1 + 4πχ`.
let (alpha, epsilon) = dielectric_tensor_with_extent(
    &slab, &params, &opts, ExtentConvention::SlabThickness(6.0),   // Bohr
)?;

// Phonons at any q, no supercell. On a 3D cell this is the *full* D(q) — the long-range
// monopole channel is inside it — so do NOT also apply `frequencies_with_lo_to`, which exists
// to give that same physics to the supercell route. Use one route or the other, never both.
let q = KPoint { fractional: [0.25, 0.0, 0.0], weight: 1.0 };
let nu = frequencies_dfpt_with(&crystal, &params, &opts, &DfptOptions {
    long_range: LongRange::Auto,
    ..DfptOptions::default()
}, q)?;

// The other route: supercell force constants plus the non-analytic term from `z` and `epsilon`.
// `direction` is the unit vector along which the q -> 0 limit is taken; the limit is direction
// dependent, which is what LO-TO splitting *is*.
use am1_rs::pbc::ForceConstants;
let volume = crystal.cell.unwrap().measure();
let direction = am1_rs::math::Vec3::new(1.0, 0.0, 0.0);
let fc = ForceConstants::from_supercell(&crystal, &params, &scf_opts, [2, 2, 2])?;
let split = fc.frequencies_with_lo_to(q, direction, &z, &epsilon, volume)?;
```

`DfptOptions` also takes an arbitrary `kmesh` or an explicit `kpoints` list — and the ground state
is solved on that same set, because the response equations assume the zeroth order satisfies the
SCF condition. `force_constants_at_q_with` returns a `DfptResult` carrying the band energies,
occupations and (with `keep_response`) the `(k, k+q)` first-order densities.

`keep_response` is off by default and it is worth knowing why: the solver **streams** the
perturbations — solve, contract into `C(q)`, drop — so the response never exists all at once
unless you ask for it. Setting the flag is what builds the `O(3N · n_k · nao²)` array, not what
returns one that was already there.

`DfptResult` also carries `bare_nonzeros` and `bare_dense_elements`, the entries the contraction
actually touches per `(j, j', k)` against the `nao²` a dense bare perturbation would force. They
are returned for the same reason `DcResult` returns its operation counters: assembling `C(q)` is
claimed to be `O(N³ n_k)` rather than `O(N⁴ n_k)`, and a claim about scaling should be checkable
from the result.

Read [pbc.md](pbc.md) for the conventions and the limitations.

## 10. Divide-and-conquer

```rust
use am1_rs::{run_divide_conquer, divide_conquer_gradient, DcOptions};
use am1_rs::fermi::Filling;

let dc = run_divide_conquer(&mol, &params, &opts, &DcOptions {
    core_size: 12,
    buffer_radius: 11.0,
    filling: Filling::Fermi { kt: 0.05 },
    ..DcOptions::default()
})?;
println!("{} eV in {} subsystems (largest {} AOs)",
         dc.total_ev, dc.subsystems, dc.largest_subsystem_aos);
if let Some(w) = &dc.small_gap_warning { eprintln!("{w}"); }

let gradient = divide_conquer_gradient(&mol, &params, &dc)?;
```

`partition_atoms`, `build_subsystems` and `partition_weight_sum` are public so the partition and
its sum rule can be inspected directly. See [divide-conquer.md](divide-conquer.md).

## 11. Errors

Every fallible call returns `am1_rs::Result<T>` (= `Result<T, Am1Error>`). Variants include
`Io`, `Parse`, `InvalidInput`, `MissingElement(u8)`, `MissingParameter(String)`,
`LinearAlgebra(String)`, `ScfNotConverged { iterations, error }`, `CphfNotConverged` and
`ElementNotParameterized`.

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
