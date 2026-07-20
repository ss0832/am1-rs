# Python API

The distribution is **`am1-rs-python`**; the import package is **`am1_rs`**. It has two layers:

- **`am1_rs` / `am1_rs.native`** — the raw model surface in **atomic units** (Hartree, Bohr).
- **`am1_rs.ase`** — an ASE `Calculator` in ASE's **eV / Å** convention.

## Install

```bash
pip install am1-rs-python           # (or the prebuilt wheel in dist/)
pip install am1-rs-python[ase]      # + ASE for the calculator
# from source:
maturin develop --release --features python
```

## Units

Input coordinates are always **Ångström**. The native API returns atomic units
(Hartree, Bohr) plus convenience `*_ev` fields and ΔHf in kcal/mol; the ASE layer converts to
eV / Å at its boundary.

Every native function takes `(numbers, positions, charge=0.0, multiplicity=1, reference="auto")` —
`numbers` is a sequence of atomic numbers, `positions` an `(N, 3)` array-like in Å. `multiplicity`
(2S+1) fixes the α/β electron counts. `reference` chooses the spin treatment **independently** of
the multiplicity: `"auto"` (RHF for a closed-shell singlet, UHF for an open shell), `"rhf"` (force
restricted — requires a closed shell), or `"uhf"` (force unrestricted, even for a singlet, e.g. as
a broken-symmetry starting point).

```python
import am1_rs
Z   = [8, 1, 1]
xyz = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]]
```

## `single_point(numbers, positions, charge=0.0, multiplicity=1, reference="auto") -> dict`

AM1 energy and properties.

```python
sp = am1_rs.single_point(Z, xyz)
```

Returns a dict with keys:

| key | type | meaning |
|---|---|---|
| `energy_hartree` | float | total AM1 energy, Hartree |
| `energy_ev` | float | total energy, eV (AM1's native unit) |
| `heat_of_formation_kcal` | float | AM1 heat of formation, kcal/mol |
| `electronic_ev`, `core_ev` | float | electronic and core–core parts, eV |
| `charges` | list[float] | Mulliken net atomic charges, e |
| `dipole_debye` | list[float] | dipole vector `[x, y, z]`, Debye |
| `homo_ev`, `lumo_ev` | float or None | frontier orbital energies, eV |
| `converged` | bool | SCF convergence flag |

## `gradient(numbers, positions, charge=0.0, multiplicity=1, reference="auto") -> dict`

Energy + analytic nuclear gradient.

```python
g = am1_rs.gradient(Z, xyz)
forces_ev_ang = [[-c for c in row] for row in g["gradient_ev_per_angstrom"]]
```

Keys: `energy_hartree`, `energy_ev`, `heat_of_formation_kcal`,
`gradient_hartree_per_bohr` (list of `[x, y, z]`, atomic units),
`gradient_ev_per_angstrom` (list of `[x, y, z]`, eV/Å; forces = −gradient).

## `optimize(numbers, positions, charge=0.0, multiplicity=1, reference="auto") -> dict`

L-BFGS geometry optimization on the analytic gradient.

```python
opt = am1_rs.optimize(Z, xyz)
relaxed = opt["positions_angstrom"]
```

Keys: `positions_angstrom` (list of `[x, y, z]`, Å), `energy_hartree`,
`heat_of_formation_kcal`, `converged` (bool), `iterations` (int).

## `frequencies(numbers, positions, charge=0.0, multiplicity=1, reference="auto") -> dict`

Harmonic vibrational frequencies from the analytic (CPHF) Hessian. **Evaluate at a stationary
point** (optimize first) for physically meaningful modes.

```python
freq = am1_rs.frequencies(Z, opt["positions_angstrom"])
print(freq["frequencies_cm"][-3:])      # three highest modes, cm^-1
```

Keys: `frequencies_cm` (list[float], cm⁻¹ ascending; negative = imaginary mode),
`eigenvalues` (list[float], mass-weighted Hessian eigenvalues, eV/(Å²·amu)).

## `hessian(numbers, positions, charge=0.0, multiplicity=1, reference="auto") -> dict`

Analytic (CPHF) Cartesian Hessian ∂²E/∂Rᵢ∂Rⱼ at the given geometry — the full `3N × 3N`
second-derivative matrix (closed-shell RHF and open-shell UHF). It is defined at any geometry;
**optimize first** for a physical force-constant matrix.

```python
h  = am1_rs.hessian(Z, xyz)
Hau = h["hessian_hartree_per_bohr2"]     # 3N×3N, atomic units
Hev = h["hessian_ev_per_angstrom2"]      # 3N×3N, eV/Å²
```

Keys: `hessian_hartree_per_bohr2` (list[list[float]], **atomic units**, the native convention),
`hessian_ev_per_angstrom2` (list[list[float]], eV/Å²), `ndof` (int, `3N`). Row/column index
`3*i + k` is atom `i`, axis `k` (0=x, 1=y, 2=z), in input atom order. Mass-weighting this matrix
reproduces the eigenvalues from `frequencies` exactly.

## `am1_bcc(numbers, positions, charge=0.0) -> dict`

AM1-BCC partial charges for AMBER (exact antechamber `BCCPARM.DAT`).

```python
bcc = am1_rs.am1_bcc(Z, xyz)
print(bcc["charges"], bcc["atom_types"])
```

Keys: `charges` (list[float], AM1-BCC net charges, e; Σ = `charge`),
`mulliken` (list[float], AM1 Mulliken charges before corrections),
`atom_types` (list[str], antechamber BCC codes, e.g. `"31"`, `"91"`).

## ASE calculator — `am1_rs.ase.AM1`

ASE convention throughout: energy in **eV**, forces in **eV/Å**, positions in **Å**.

```python
from am1_rs.ase import AM1
from ase.build import molecule

atoms = molecule("H2O")
atoms.calc = AM1(charge=0.0, multiplicity=1)   # multiplicity > 1 → UHF

atoms.get_potential_energy()   # eV
atoms.get_forces()             # eV/Å
atoms.get_charges()            # Mulliken, e
atoms.get_dipole_moment()      # e·Å
atoms.calc.results["heat_of_formation_kcal"]

H = atoms.calc.get_hessian(atoms)   # 3N×3N analytic Hessian, eV/Å²
```

`AM1(charge=0.0, multiplicity=1, reference="auto")`; `implemented_properties = ["energy",
"free_energy", "forces", "charges", "dipole"]`, plus `get_hessian(atoms=None) -> ndarray`
(eV/Å²). Works with any ASE workflow (e.g. `BFGS`, `Vibrations`, `ase.md`), though the crate's
own Rust L-BFGS (`am1_rs.optimize`) is usually faster.

**Charge, spin multiplicity & reference** may be given **either** at construction —
`AM1(charge=1.0, multiplicity=2, reference="uhf")` — **or** per structure at calculation time via
`atoms.info`:

```python
atoms.info["charge"] = 1.0           # overrides the constructor value for this structure
atoms.info["multiplicity"] = 2       # 2S+1
atoms.info["reference"] = "uhf"      # "auto" | "rhf" | "uhf"; changing any re-triggers the calc
```

An `atoms.info` entry takes precedence over the constructor value; a change in any of them
invalidates cached results. The native functions above take the same arguments directly, so both
layers receive charge, multiplicity and reference at every property evaluation.

## End-to-end example

```python
import am1_rs

Z   = [6, 8, 1, 1]                       # formaldehyde
xyz = [[0,0,0], [0,0,1.21], [0.94,0,-0.54], [-0.94,0,-0.54]]

opt  = am1_rs.optimize(Z, xyz)           # relax
freq = am1_rs.frequencies(Z, opt["positions_angstrom"])
bcc  = am1_rs.am1_bcc(Z, opt["positions_angstrom"])

print("ΔHf  =", round(opt["heat_of_formation_kcal"], 2), "kcal/mol")
print("freqs=", [round(f) for f in freq["frequencies_cm"][-6:]])
print("BCC  =", [round(q, 3) for q in bcc["charges"]])
```

See [`rust-api.md`](rust-api.md) for the native Rust API and [`scope.md`](scope.md) for the
feature matrix.
