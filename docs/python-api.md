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

Input coordinates are always **Ångström**.

| Function group | Returns |
|---|---|
| `single_point`, `gradient`, `optimize`, `hessian`, `frequencies`, `am1_bcc` | **atomic units** (Hartree, Bohr), plus `*_ev` conveniences and ΔHf in kcal/mol |
| `pbc_point`, `divide_conquer` | **eV / Å** directly |
| `am1_rs.ase.AM1` | **eV / Å / eV Å⁻³** (ASE's convention) |

The periodic and divide-and-conquer functions break the atomic-unit rule on purpose: their
consumer is ASE, and converting a stress tensor twice is an easy way to be quietly wrong.

Every molecular function takes
`(numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1")` —
`numbers` is a sequence of atomic numbers, `positions` an `(N, 3)` array-like in Å.
`multiplicity` (2S+1) fixes the α/β electron counts. `reference` chooses the spin treatment
**independently** of the multiplicity: `"auto"` (RHF for a closed-shell singlet, UHF for an open
shell), `"rhf"` (force restricted — requires a closed shell), or `"uhf"` (force unrestricted,
even for a singlet, e.g. as a broken-symmetry starting point). `method` selects the
parameterization, `"am1"` or `"rm1"` — see [methods.md](methods.md).

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

## `pbc_point(numbers, positions, cell, pbc, ...) -> dict`

Periodic single point: energy, analytic forces, analytic stress. Returns **eV / Å**.

```python
import numpy as np

r = am1_rs.pbc_point(
    Z, xyz,
    cell=np.eye(3) * 6.0,        # lattice vectors as ROWS, Ångström
    pbc=[True, True, True],      # a chain is [True, False, False], a slab [True, True, False]
    kpts=(2, 2, 2),
    exchange_cutoff=10.0,        # Bohr -- a documented approximation, see docs/pbc.md
)
print(r["energy_ev"], r["stress_voigt"], r["max_image_overlap"])
```

Further keywords: `charge`, `multiplicity`, `unrestricted`, `smearing_ev`, `realspace_cutoff`,
`method`, `e_tol`, `p_tol`, `max_scf`, `mixing`, `electric_field`.

`electric_field` is in **atomic units** (Hartree per e·Bohr), like the molecular functions, and
must be **orthogonal to every lattice vector** — normal to a slab, transverse to a chain. A
component along a periodic direction raises, naming itself: `F·R` is unbounded there. New in 0.2.2;
before that a field under any cell was refused.

Keys include `energy_ev`, `free_energy_ev`, `forces_ev_per_angstrom`, `stress_voigt`
(`[xx, yy, zz, yz, xz, xy]`), `stress_matrix`, `charges`, `fermi_energy_ev`, `entropy_ev`,
`k_points`, `n_periodic`, `max_image_overlap`, `charged_cell_warning`.

**Read [pbc.md](pbc.md) before relying on the numbers.** In particular the stress is per unit
area for a slab and per unit length for a chain, the exchange taper is an approximation rather
than a convergence parameter, and a charged cell's absolute energy does not converge.

## `divide_conquer(numbers, positions, ...) -> dict`

Linear-scaling-diagonalization SCF for large systems, molecular or periodic. Returns **eV / Å**.

Pass `cell` and `pbc` together to run under a cell — the subsystem buffers are then built from
the image-aware pair list and wrap through the cell boundary. `exchange_cutoff` matters there and
not for a molecule, because at Γ the two-centre exchange decays only as `1/R` while the density
matrix does not decay at all.

```python
r = am1_rs.divide_conquer(Z, xyz, core_size=12, buffer_radius=11.0)
periodic = am1_rs.divide_conquer(Z, xyz, cell=cell, pbc=[True] * 3, buffer_radius=14.0)
print(r["energy_ev"], r["subsystems"], r["largest_subsystem_aos"])
print("Σn³ per atom:", r["diagonalization_work"] / len(Z))
```

Keys include `energy_ev`, `forces_ev_per_angstrom`, `charges`, `fermi_energies_ev` (one per
spin channel), `homo_lumo_gap_ev`, `small_gap_warning`, and the scaling counters
`diagonalization_work`, `coulomb_work`, `exchange_work`, `retained_density_blocks`,
`diis_pattern_elements` and `dense_triangle_elements` — the last two being the DIIS history's
memory and what it would have cost dense, linear against quadratic.

Increase `buffer_radius` (Bohr) until the property you care about stops moving; a buffer
covering the molecule reproduces the full SCF exactly. See
[divide-conquer.md](divide-conquer.md) for a precise statement of what became linear.

---

## External electric field

Every molecular entry point takes `electric_field=[Fx, Fy, Fz]` in **atomic units**
(Hartree per e·Bohr). The energy becomes `E = E₀ − μ·F` with the model's own dipole.

```python
r = am1_rs.single_point(Z, xyz, electric_field=[0.0, 0.0, 0.005])
r["field_nuclear_ev"]        # the nuclear half of −μ·F; the electronic half is in electronic_ev
```

The gradient and the analytic Hessian both account for it. Because the dipole operator is
*linear* in the nuclear positions it adds nothing to the fixed-density second derivative — a
field reaches the Hessian only through the CPHF response.

**Under a cell, the field must be orthogonal to every lattice vector** — normal to a slab,
transverse to a chain. `F·R` shifts by `F·T` under translation by `T`, so the perturbation is
lattice-periodic exactly when `F·T = 0`; a component *along* a periodic direction is an error
naming itself. For the response along a periodic direction use `dielectric()`, which is the linear
regime; the finite/non-linear regime there needs the Berry-phase electric enthalpy and is not
implemented.

Refused for **any** cell through 0.2.1, which threw the well-defined cases out with the ill-defined
one.

## Wavefunction: `orbitals(...)`, `molden(...)`

```python
o = am1_rs.orbitals(Z, xyz)
o["energies_ev"], o["coefficients"], o["n_occupied"], o["ao_labels"]
o["beta_coefficients"]        # unrestricted runs only

open("water.molden", "w").write(am1_rs.molden(Z, xyz))
```

`coefficients` has atomic orbitals as **rows** and molecular orbitals as **columns**, and
`ao_labels` gives each row's `(atom_index, shell)` so a coefficient can be identified without
rebuilding the basis.

NDDO *assumes* an orthonormal AO basis, so these coefficients are in an implicitly orthogonalized
basis while the Molden `[STO]` section lists the raw Slater functions. Shapes, nodes and symmetry
are faithful; bonding-region amplitudes are approximate. See [theory.md](theory.md).

## Infrared: `ir_spectrum(...)`, `dipole_derivatives(...)`

**Expensive** — both run an analytic Hessian. They are separate calls rather than part of a
single point for exactly that reason.

```python
ir = am1_rs.ir_spectrum(Z, xyz)          # evaluate at a stationary point
ir["dipole_derivatives"]                 # the raw 3 × 3N atomic polar tensor, in e
ir["frequencies_cm"], ir["intensities_km_per_mol"]
ir["mode_dipole_derivatives"]            # dense per-mode tensor, keeps the transition-dipole direction
ir["vibrational_modes"]                  # indices that are not translations or rotations
```

`translation_rotation_overlap` says how much of each mode is rigid-body motion, so a linear
molecule's five rigid-body modes are reported rather than assumed from `3N − 6`.

## First-order orbital response: `orbital_response(...)`

```python
r = am1_rs.orbital_response(Z, xyz)
r["u_ov"]                       # one n_vir × n_occ CPHF block per Cartesian degree of freedom
r["hessian_ev_per_bohr2"]       # the Hessian it was solved alongside — both for one calculation
r = am1_rs.orbital_response(Z, xyz, response_density=True)   # opt-in: 3N AO matrices
```

## Periodic response: `pbc_hessian`, `born_charges`, `dielectric`, `dfpt`, `lo_to_frequencies`

```python
cell, pbc = np.eye(3) * 4.5, [True, True, True]

am1_rs.pbc_hessian(Z, xyz, cell, pbc, kpts=(2, 2, 2))   # q = 0 force constants with k-sampling
am1_rs.born_charges(Z, xyz, cell, pbc, kpts=(2, 2, 2))  # Z*, plus its acoustic sum rule error
am1_rs.dielectric(Z, xyz, cell, pbc, kpts=(2, 2, 2))    # α (Bohr³) and ε_∞ — 3D cells only
am1_rs.polarizability(Z, xyz, cell, pbc, kpts=(2, 2, 2))  # α alone — any dimensionality
am1_rs.dfpt(Z, xyz, cell, pbc, [[0.25, 0, 0]], kpts=(2, 2, 2))   # phonons at arbitrary q

# Supercell phonons *plus* the non-analytic term, from Z* and ε_∞. 3D cells only.
am1_rs.lo_to_frequencies(Z, xyz, cell, pbc, supercell=(2, 2, 2), direction=(1, 0, 0))
```

`dfpt` samples the ground state on the same k set as the response, deliberately: the
coupled-perturbed equations assume the zeroth order satisfies the SCF condition. If the response
solve hits its iteration cap, loosen `cpscf_tol` — it is a mixed fixed point, not a Newton solve.

**Pick one route for the long-range part, not both.** On a 3D cell `dfpt` returns the *full*
`D(q)` — the long-range monopole channel is inside it — so its `q → 0` limit is already direction
dependent. `lo_to_frequencies` exists to give that same physics to the *supercell* route, where a
truncated `Φ(T)` structurally cannot carry it. Applying both counts it twice.
`lo_to_frequencies` returns `frequencies_cm_no_lo_to` alongside `frequencies_cm` so the size of
the shift is visible rather than taken on trust.

`dielectric` refuses a chain or a slab: `ε_∞ = 1 + 4πα/Ω` needs `Ω` to be a volume. Three things
work there instead, in increasing order of how much you have to decide.

```python
am1_rs.polarizability(Z, xyz, cell, pbc)                       # α alone. No choice to make.
am1_rs.dielectric_function(Z, xyz, cell, pbc, q=[0.05, 0, 0])  # ε(q). No choice to make.
am1_rs.dielectric_with_extent(Z, xyz, cell, pbc, slab_thickness=6.0)   # ε_∞. You choose the 6.0.
```

`dielectric_with_extent` needs exactly one of `slab_thickness` (Bohr, 2D-periodic) or
`wire_cross_section` (Bohr², 1D-periodic), and there is **no default** — a supercell says where the
atoms are, not where the material stops. It is not a division: `α` is the response to the
*external* field, so the conversion carries the depolarization factor of the assumed body, which
makes a slab's out-of-plane law `1/(1 − 4πχ)` rather than `1 + 4πχ`. Returned alongside `ε` are the
two combinations that do **not** depend on the choice — `sheet_susceptibility`,
`inverse_sheet_response`, and for a slab `rytova_keldysh_length` — which are what a low-dimensional
calculation can report without adopting a convention. See [pbc.md](pbc.md).

`dielectric_function` is `ε(q)` along a Cartesian wavevector `q` (inverse Bohr, inside the periodic
subspace). In three dimensions it is independent of `|q|` and equals `dielectric`'s `ε_∞`; for a
slab and a chain it tends to **1** at long wavelength, because a sheet or a wire does not screen a
field whose wavelength exceeds its own extent. A chain additionally needs `chain_radius` (Bohr) —
its Coulomb kernel is a logarithm and has no value without a reference length, and there is no
natural choice, so it is required rather than guessed. See [pbc.md](pbc.md).
## Polarization and a finite field along a periodic direction

```python
am1_rs.dielectric_function(Z, xyz, cell, pbc, q=[0.05, 0, 0])   # eps(q), any dimensionality
am1_rs.polarization(Z, xyz, cell, pbc, kpts=(2, 2, 2), strings=8)
am1_rs.finite_field(Z, xyz, cell, pbc, field=[0, 0, 2e-3], kpts=(4, 4, 4))
```

`polarization` is the Berry phase (King-Smith–Vanderbilt), returned **modulo the quantum**
`e a_α/Ω` — only differences between two states on a common branch are physical.

`finite_field` solves in a field **along** a periodic direction, where `F·R` is unbounded and
cannot be used: it minimizes the electric enthalpy `E − Ω 𝓔·P` instead. The field is in atomic
units here, as everywhere else in this module. For a field **orthogonal** to every lattice
vector — normal to a slab, transverse to a chain — use `pbc_point(..., electric_field=...)`; that
is an ordinary calculation and needs none of this.

Both are three-dimensional and restricted, and `finite_field` needs at least three k points along
any direction the field has a component in. The Berry phase here tracks the charge **centres** and
carries no on-site `s`–`p` moment — a 12 % effect on `α` for a p-block cell, and the whole of the
out-of-plane response for a planar one. [pbc.md](pbc.md) measures it.

## ASE calculator — `am1_rs.ase.AM1`

ASE convention throughout: energy in **eV**, forces in **eV/Å**, stress in **eV/Å³**, positions
in **Å**. One class covers molecules, periodic systems and divide-and-conquer; which one you get
is decided by `atoms.pbc` and the `divide_conquer` flag.

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

`implemented_properties = ["energy", "free_energy", "forces", "charges", "dipole", "stress"]`.
Works with any ASE workflow (`BFGS`, `Vibrations`, `ase.md`), though the crate's own Rust L-BFGS
(`am1_rs.optimize`) is usually faster for a molecular relaxation.

### Everything the native API does, in ASE units

The two surfaces are meant to expose the same capabilities, so that choosing one is a choice of
unit convention and not of feature. `tests/test_new_api_0_2_1.py` asserts that.

| ASE method | native equivalent | units |
|---|---|---|
| `get_hessian(atoms)` | `hessian` / `pbc_hessian` | eV/Å² — molecular *or*, under a cell, the k-point `q = 0` force constants |
| `get_frequencies(atoms)` | `frequencies` | cm⁻¹ |
| `get_ir_spectrum(atoms)` | `ir_spectrum` | cm⁻¹, km/mol, `e` |
| `get_dipole_derivatives(atoms)` | `dipole_derivatives` | `e`, shape `(3, 3N)` |
| `get_orbitals(atoms)` | `orbitals` | eV and Hartree |
| `get_orbital_response(atoms)` | `orbital_response` | dimensionless CPHF blocks |
| `write_molden(path, atoms)` | `molden` | — |
| `get_am1_bcc_charges(atoms)` | `am1_bcc` | `e` |
| `optimize(atoms)` | `optimize` | Å, written back into `atoms` unless `apply=False` |
| `get_phonons(atoms, supercell=…)` | `phonons` | cm⁻¹ |
| `get_dfpt_frequencies(q, atoms)` | `dfpt` | cm⁻¹ |
| `get_lo_to_frequencies(direction=…)` | `lo_to_frequencies` | cm⁻¹, 3D cells only |
| `get_born_charges(atoms)` | `born_charges` | `e`, shape `(nat, 3, 3)` |
| `get_dielectric_tensor(atoms)` | `dielectric` | dimensionless 3×3; 3D cells only |
| `get_dielectric_tensor_with_extent(slab_thickness=…, wire_cross_section=…)` | `dielectric_with_extent` | dimensionless 3×3; **extent in Å / Å²** |
| `get_dielectric_function(q, atoms, chain_radius=…)` | `dielectric_function` | dimensionless; **q in 1/Å** |
| `get_polarizability(atoms)` | `polarizability` | Bohr³ 3×3; any dimensionality |
| `get_polarization(atoms, strings=…)` | `polarization` | `e/Bohr²`, modulo the quantum; 3D |
| `get_finite_field(field, atoms)` | `finite_field` | field in **V/Å**; returns the native dict; 3D |

**These are explicit methods and never run from `calculate()`.** A Hessian, a phonon band or an
infrared spectrum costs orders of magnitude more than an energy, and a property that appeared
merely because something touched `atoms.calc.results` would be a trap. They cache into `results`
and are invalidated by `check_state` like any other result.

### External field

`field=[Fx, Fy, Fz]` in **V/Å** — ASE's convention, converted at this boundary using the crate's
own `a0` rather than CODATA's, so that a field set here is the field the model applies. Settable
at construction or per structure via `atoms.info["field"]`, and either way a change invalidates
the cache.

```python
atoms.calc = AM1(field=[0.0, 0.0, 0.5])   # V/Å
atoms.get_forces()                        # includes +Q_a F
```

Works for molecules, and under a cell when the field is orthogonal to every lattice vector; a component along a periodic direction raises, naming itself.

### Parameters

All settable at construction and visible through `todict()`, so `restart` and `set()` work:

| Parameter | Default | Notes |
|---|---|---|
| `charge`, `multiplicity`, `reference` | `0.0`, `1`, `"auto"` | as for the native functions |
| `method` | `"am1"` | `"am1"` or `"rm1"` |
| `kpts` | `(1,1,1)` | periodic only; non-periodic axes collapse to 1 |
| `smearing` | `0.0` eV | Fermi–Dirac width; a metal needs a finite value |
| `realspace_cutoff`, `exchange_cutoff` | `40.0`, `20.0` Bohr | see [pbc.md](pbc.md) |
| `e_tol`, `p_tol`, `max_scf`, `mixing` | `1e-8`, `1e-7`, `300`, `0.3` | tighten before differentiating numerically |
| `divide_conquer` | `False` | molecular only; raises for a periodic cell |
| `core_size`, `buffer_radius`, `gap_warn_ev` | `12`, `11.0` Bohr, `0.5` eV | see [divide-conquer.md](divide-conquer.md) |

### Periodic systems

```python
import numpy as np
from ase import Atoms
from am1_rs.ase import AM1

slab = Atoms("OH2", positions=xyz,
             cell=[[3.4, 0, 0], [0, 3.4, 0], [0, 0, 24]], pbc=[True, True, False])
slab.calc = AM1(kpts=(4, 4, 1))
slab.get_potential_energy()
slab.get_stress()            # eV/Å² for a slab -- see pbc.md on the measure
```

A **molecular** structure raises `PropertyNotImplementedError` from `get_stress()` rather than
returning zeros: a molecule in free space has no stress, and zeros would let a variable-cell
optimizer run happily on nothing. `get_dipole_moment()` raises for a *periodic* structure, since
the dipole of a cell is not determined by its charge distribution.

`tests/test_ase_pbc_md.py` runs NVE, NPT and NVT ensembles and is the best worked example.

### Large systems

```python
atoms.calc = AM1(divide_conquer=True, core_size=12, buffer_radius=11.0)
```

Extra `results` keys: `subsystems`, `largest_subsystem_aos`, `homo_lumo_gap`. A system whose gap
falls below `gap_warn_ev` raises a `RuntimeWarning`, because divide-and-conquer assumes the
density matrix decays with distance and that is a gapped-system property.

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
