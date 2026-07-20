# AM1 theory as implemented in am1-rs

AM1 is an NDDO self-consistent-field method. Below is the working set of equations as
implemented, with the module that carries each piece.

## Basis and overlap (`basis.rs`, `overlap.rs`)

A minimal valence Slater basis: one `s` shell for H, an `s`+`p` set for heavier elements.
Overlaps `S_μν` are evaluated analytically over Slater orbitals in the diatomic local frame
(reduced A/B auxiliary integrals) and rotated to the molecular frame. Overlap enters the
method **only** through the resonance integral.

## Core Hamiltonian (`hamiltonian.rs`)

- Diagonal one-electron energies `U_ss`, `U_pp`.
- Electron–core attraction to every other atom `B`: `−Z_B (μ_A ν_A | s_B s_B)`.
- Inter-atomic resonance `H_μν = ½(β_μ + β_ν) S_μν`.

## Two-electron integrals (`integrals.rs`)

One-center integrals are the parameters `G_ss, G_sp, G_pp, G_p2, H_sp`. Two-center
integrals use the Dewar–Sabelli–Klopman multipole/point-charge expansion: each on-atom
charge distribution is a set of monopole/dipole/quadrupole point charges (separations
`dd`, `qq`) interacting through the Klopman–Ohno kernel
`1/√(R² + (ρ_a + ρ_b)²)`, evaluated as 22 local-frame integrals and rotated into the
molecular frame. The additive terms are

```
ρ0 = 0.5 · ev / G_ss
ρ1 : solved from  H_sp = ¼(1/ρ1 − 1/√(dd² + ρ1²))
ρ2 : solved from  H_pp = ⅛/ρ2 − ¼/√(qq² + ρ2²) + ⅛/√(2qq² + ρ2²),  H_pp = ½(G_pp − G_p2)
```

## Fock build and SCF (`fock.rs`, `scf.rs`)

`F = H_core + G(P)`, with the canonical MNDO one-center Fock formulas and two-center
Coulomb (J) and exchange (K) from the rotated integrals. Because the basis is treated as
orthonormal, the SCF is the plain eigenproblem `F C = C ε` (no `S`).

**Convergence acceleration** (`ScfAccelerator`, default `AdiisCdiis`): **A-DIIS** (Hu & Yang
2010) is used while the commutator error `‖[F,P]‖` is large — its coefficients are constrained
to the probability simplex (`c ≥ 0`, `Σc = 1`), which prevents the runaway extrapolation plain
DIIS can produce far from convergence — and the method **switches to Pulay CDIIS** (faster near
the solution) once the error drops below `adiis_switch`. All accelerators reach the same SCF
fixed point; the hybrid is the most robust and, in practice, also the fewest iterations.
Second-order SOSCF (orbital-Hessian Newton) would give quadratic final convergence but is heavy
for a minimal-basis semiempirical method — a documented optional follow-up.

## Core–core repulsion (`repulsion.rs`)

```
E_core(A,B) = Z_A Z_B γ_AB [1 + f_A + f_B]  +  (Z_A Z_B / R) Σ_k K_k e^{−L_k (R − M_k)²}
```

with `γ_AB` the screened monopole integral, `f = e^{−α R}`, the MNDO `R·e^{−α R}` special
cases for N–H and O–H, and the AM1 Gaussian corrections `(K, L, M)`. `R` is in Ångström.

## Energy and heat of formation

```
E_elec  = ½ Σ_μν P_μν (H_core + F)_μν
E_total = E_elec + Σ_{A<B} E_core(A,B)
ΔH_f    = E_total − Σ_A E_isol(A) + Σ_A ΔH_{f,atom}(A)
```

The isolated-atom energies `E_isol` use the MOPAC average-of-configuration coefficients.

## Nuclear gradient (`gradient.rs`)

Because the AO basis is orthonormal in NDDO, the SCF energy is stationary with respect to the
density, so the gradient is the plain **Hellmann–Feynman** derivative at the fixed converged
density — there is **no Pulay/overlap-constraint term**:

```
dE/dR = Σ P_μν ∂H_core,μν/∂R  +  ½ Σ P_μν P_λσ ∂(μν|λσ)/∂R  +  ∂E_core/∂R
```

`am1-rs` evaluates this in **fully closed form** ([`gradient::closed_form_gradient`]): the
integral kernels are generic over a `Scalar` trait, and instantiating them at a forward-mode
**dual number** (value + 3 spatial partials, `dual.rs`) yields the exact derivatives
`∂(μν|λσ)/∂R`, `∂S/∂R`, `∂e1b/∂R`, `∂E_core/∂R` of the same expressions used for the energy.
Per atom pair these are contracted with the fixed converged density (resonance + core-attraction
+ Coulomb + exchange), so no SCF re-runs or full-molecule finite differences are needed. The
result matches a full-SCF finite difference to ~1e-7 eV/Bohr. (Open-shell UHF uses the
fixed-density Hellmann–Feynman evaluation, which is also exact.)

## Open-shell UHF (`scf.rs`)

For `multiplicity > 1` (or an odd electron count) the SCF is spin-unrestricted: two densities
`P_α`, `P_β` are solved with `F^σ = H_core + J(P_tot) − K(P^σ)` (Coulomb from the total
density, exchange from the same spin), combined-DIIS accelerated, with
`E_elec = ½[P_tot·H_core + P_α·F_α + P_β·F_β]`. The spin density `P_α − P_β` is reported.

## Heavy elements (`overlap_numeric.rs`)

For valence shells with `n ≥ 4` (Zn, Ge, As, Se, Br, Sb, Te, I, Hg), for which no closed-form
Slater-overlap kernel is tabulated, the diatomic overlap is evaluated by Gauss–Legendre
quadrature in prolate-spheroidal coordinates. This general routine reproduces the analytic
`n ≤ 3` kernel to ~1e-8 and feeds the same rotation/assembly.

## AM1-BCC (`topology.rs`, `bcc.rs`)

Run the SCF for AM1 Mulliken charges, perceive the molecular graph
(bonds/hybridization/aromaticity), assign atom/bond types, and apply additive bond charge
corrections to reach AMBER-quality partial charges.
