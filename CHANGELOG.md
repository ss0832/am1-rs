# Changelog

## 0.1.3

### Added
- **Explicit RHF/UHF reference selection**, independent of the spin multiplicity — a closed-shell
  singlet can now be run either restricted or unrestricted (e.g. as a broken-symmetry starting
  point):
  - Rust: new `ScfReference` enum (`Auto` / `Restricted` / `Unrestricted`) and an
    `Am1Options.reference` field (default `Auto`, preserving previous behavior). `run_am1` honors
    it; `Restricted` on an open shell is rejected (no ROHF).
  - Python native: every function (`single_point`, `gradient`, `optimize`, `frequencies`,
    `hessian`) takes a `reference="auto"|"rhf"|"uhf"` keyword.
  - ASE: `AM1(..., reference="auto")`, or per structure via `atoms.info["reference"]`; a change
    invalidates cached results.
  - CLI: `--reference auto|rhf|uhf`, with `--rhf` / `--uhf` shortcuts.

### Notes
- `Auto` reproduces the historical selection (RHF for a closed-shell singlet, UHF for an open
  shell), so existing callers are unaffected. Forcing UHF on a symmetric singlet converges to the
  RHF energy (zero spin density).

## 0.1.2

### Added
- **Hessian API for Python.** Both layers now expose the analytic (CPHF) Cartesian Hessian,
  which previously was only reachable indirectly through `frequencies`:
  - `am1_rs.hessian(numbers, positions, charge=0.0, multiplicity=1)` returns the full `3N × 3N`
    matrix in **atomic units** (`hessian_hartree_per_bohr2`) and, for convenience, in eV/Å²
    (`hessian_ev_per_angstrom2`), plus `ndof`. Row/column `3*i + k` is atom `i`, axis `k`.
  - `am1_rs.ase.AM1.get_hessian(atoms=None)` returns the Hessian as a NumPy array in **eV/Å²**
    (ASE convention). Closed-shell RHF and open-shell UHF are both supported.
- **Per-structure charge / multiplicity for the ASE calculator.** In addition to the constructor
  arguments `AM1(charge=…, multiplicity=…)`, charge and spin multiplicity may now be supplied at
  calculation time via `atoms.info["charge"]` / `atoms.info["multiplicity"]`. An `atoms.info`
  entry overrides the constructor value for that structure, and a change in either invalidates
  cached results (`check_state`).

### Verified
- Confirmed (with tests) that charge and spin multiplicity are received and actually used in
  both the Python-native functions (per call) and the ASE calculator (at construction *and* at
  calculation time): charge and multiplicity change the SCF energy, and an electron-count /
  multiplicity parity mismatch raises. See `tests/test_python_api.py`.

### Notes
- Units are unchanged and follow each layer's convention: the native surface reports atomic
  units (Hartree/Bohr²) with eV/Å² provided alongside; the ASE layer reports eV/Å².
- No changes to the Rust crate's public API (the analytic Hessian was already available there as
  `am1_rs::analytic_hessian`, eV/Bohr²).
