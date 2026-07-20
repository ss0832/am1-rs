# Scope / feature matrix

This mirrors the **Scope** section of [`README.md`](../README.md); it is the canonical
capability list for `am1-rs`.

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

## Units at the boundary

- Internal: eV energies, Bohr distances (MOPAC7 model constants `ev = 27.21`, `a0 = 0.529167`).
- Rust / Python-native API: Hartree, Bohr (atomic units) + ΔHf in kcal/mol.
- ASE `Calculator`: eV, Å (AM1 energies are natively eV).
