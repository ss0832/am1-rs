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

## Analytic Hessian and CPHF (`hessian.rs`)

The second derivative is where the density's own response enters. Differentiating

```
E = ½ Σ P (H_core + F)  +  E_core
```

twice gives a term in `∂²` of the integrals contracted with the *fixed* density, plus a term in
`∂P/∂R` — and unlike the gradient, that second term does not vanish. The energy is stationary
with respect to `P`, which kills the first-order response in the gradient, but the second
derivative of a stationary quantity still needs the first derivative of the wavefunction.

`∂P/∂R` is obtained from **coupled-perturbed Hartree–Fock**. In an orthonormal basis the density
response is carried entirely by the occupied–virtual rotations `U_ai`, which satisfy

```
(ε_a − ε_i) U_ai  +  Σ_bj A_ai,bj U_bj  =  −F^{(1)}_ai
```

where `F^{(1)}` is the skeleton derivative Fock matrix (the derivative of `F` at fixed density)
and `A` is the orbital Hessian, `A_ai,bj = 4(ai|bj) − (ab|ij) − (aj|bi)` for the restricted case.
This is solved once per perturbation (3N of them). UCPHF is the same with the two spin channels
coupled through the Coulomb term.

### How it is solved, and why that choice

These equations are **linear**, and the operator on the left is the orbital Hessian: symmetric,
and positive definite at a stable SCF solution. So the solver is **preconditioned conjugate
gradient**, with `M = diag(ε_a − ε_i)`.

That is worth stating because the obvious alternative — rearranging to
`U ← (F^{(1)} + A U)/(ε_i − ε_a)` and iterating to a fixed point — is what this code did
previously, and it is *preconditioned Richardson with the same preconditioner*. CG builds a
Krylov space over the operator instead of taking fixed steps along it.

The cost that matters is not iterations but **applications of the operator**, because each one is
a full Fock build, and on a 150-atom cluster those builds are two thirds of an entire frequency
calculation. Measured: 6296 applications for the fixed point against 4931 for CG, and 23.2 s
against 15.4 s overall.

The convergence test is the same quantity in both, deliberately: the fixed point's step
`‖U_{n+1} − U_n‖` is identically `‖M⁻¹r‖`, the preconditioned residual CG already forms each
iteration. So the tolerance did not need retuning and the two are directly comparable.

If the operator turns out not to be positive definite along a search direction — an unstable SCF
solution, which this code does not otherwise detect — CG cannot proceed, and the solve falls back
to the fixed-point iteration rather than taking a step with no variational meaning.

Three implementation points worth stating, because each was a bug at some stage:

* **No Pulay term, again.** `S` does not appear, so there is no `S^{(1)}` and no
  orthonormality-constraint contribution — the same NDDO simplification that clears the
  gradient.
* **Non-convergence is an error.** An unconverged CPHF silently returns a plausible Hessian.
  `Am1Error::CphfNotConverged` is raised instead.
* **Degenerate orbital pairs.** `|ε_a − ε_i|` in the denominator is guarded; near-degenerate
  frontier pairs are what make small-gap systems hard here.

The Hessian is checked against a finite difference **of the analytic gradient** rather than of
the energy, which removes one order of finite-difference noise and makes a tight tolerance
meaningful.

### The rotation-frame bug, and why the frame is gone

The two-centre integrals are evaluated in a local frame with the `z` (or `x`) axis along the
internuclear vector, then rotated. The old code built that rotation from a quaternion and, when
the pair was antiparallel to the reference axis, substituted a **constant** fallback frame:

```rust
if qw.val().abs() < 1.0e-7 { qx = S::cst(0.0); /* ... */ }
```

Substituting a constant into a dual number **zeroes its derivatives**. For a pair aligned with
the axis, the transverse gradient and Hessian contributions vanished entirely. The gradient
survived by symmetry — the transverse first derivative of an axially aligned pair is genuinely
zero — but the **second** derivative is not protected by that symmetry, so transverse force
constants of any molecule with a bond on that axis were simply wrong. The existing tests all
used tilted geometries and never touched it.

The fix was not to stabilize the branch but to **remove the frame**. There is no globally smooth
choice of frame on the sphere (hairy ball), so every construction has a pole somewhere. But the
transverse basis vectors never survive into the result: they appear only as
`r₁ᵢr₁ⱼ + r₂ᵢr₂ⱼ`, which is the transverse projector

```
p_ij = δ_ij − n_i n_j
```

So the integrals are now written directly in terms of the unit internuclear vector `n` and `p`.
Both are polynomials in `n`, with no branch, no quaternion normalization, and exact derivatives
at every order — and it is faster, having dropped a square root and a reciprocal.

This is also a prerequisite for periodic boundary conditions, where lattice vectors are usually
axis-aligned and the old branch would have been hit constantly.

## The dipole operator and an external field (`dipole.rs`)

Three places need the same object, and until 0.2.1 two of them built it separately: the SCF
reports a dipole, the periodic field response differentiates one, and the Born charges and the
infrared atomic polar tensor differentiate it with respect to nuclear position. A sign or a
factor of two that disagreed between them would not fail loudly — it would produce a plausible
number with the wrong sign somewhere downstream. So it is defined once:

```text
Q_a = Z_a − p_a                          net charge; p_a = Σ_{μ ∈ a} P_μμ

M_α : +R_{a,α}  on every diagonal element of atom a's block
      +dd_a     on both (s, p_α) and (p_α, s), for atoms with a p shell

μ_α  = Σ_a Z_a R_{a,α} − Tr[P M_α]                                       (e·Bohr)

E(F) = E₀ − μ·F  ⇒  h^F = +Σ_α F_α M_α  into H_core,  −F·Σ_a Z_a R_a  into the core energy
```

`dd_a` is the NDDO `s`–`p` charge separation. The `−2 dd` that appears in the SCF's dipole
assembly is this `+dd` on two symmetric matrix elements, carried through the minus sign in
`μ = ΣZR − Tr[PM]` — exactly the factor this module exists to stop being rediscovered.

**The gradient is one line of physics.** `M_α` is *linear* in the nuclear positions: only its
diagonal `R_a` term moves, and the hybridization term does not depend on position at all. So

```text
∂E/∂R_{a,β} += −F_β Q_a        i.e. the force on atom a gains  +Q_a F
```

and that is the whole nuclear derivative. Two consequences worth stating. The net force on a
neutral molecule in a field is exactly zero and on an ion is exactly `qF` — constructive from
`Σ_a Q_a = q`, so a useful wiring check but not evidence the physics is right. And because the
operator is linear, it contributes **nothing** to the fixed-density second derivative: a field
reaches the Hessian *only* through the CPHF response. That is why the Hessian under a field is
checked against finite differences rather than assumed — omit the response term and the result is
still symmetric, still has sensible eigenvalues, and is still wrong.

Molecules only. `F·R` is unbounded along a periodic direction, so a cell plus a field is an error
rather than an approximation; the periodic analogue is the clamped-ion `ε_∞` response.

## Atomic polar tensor and infrared intensities (`ir.rs`)

Differentiating the dipole above at the self-consistent density gives

```text
∂μ_α/∂R_{a,β} = δ_αβ Q_a − Tr[ (∂P/∂R_{a,β}) M_α ]
```

because `∂M_α/∂R_{a,β}` is `δ_αβ` on atom `a`'s diagonal block and zero elsewhere, and its trace
against `P` supplies the `−δ_αβ p_a` that turns `Z_a` into the net charge. The second term is the
electrons rearranging, and it is the CPHF response the Hessian already solves for — so an
infrared spectrum costs a Hessian and nothing more. Expanding `Tr[∂P M_α]` reproduces the
charge-transfer and hybridization terms `pbc::born_charges` writes out separately; they are the
same expression, and keeping it compact here is what stops the molecular and periodic versions
from drifting.

The check that matters is the **translational sum rule** `Σ_a ∂μ_α/∂R_{a,β} = q δ_αβ`:
translating the whole molecule moves its net charge and nothing else. That is `3 × 3` exact
constraints on a `3 × 3N` tensor, it follows from charge conservation alone, and it is a far
sharper instrument than checking that a symmetric mode comes out dark. Measured: 3 × 10⁻¹⁵ e.

Intensities are a projection:

```text
∂μ/∂Q_k = Σ_j (∂μ/∂R_j) L_{jk} / √m_j        L mass-weighted, orthonormal
A_k     = 42.2561 × |∂μ/∂Q_k|²                km/mol, ∂μ/∂Q in D·Å⁻¹·amu^{−1/2}
```

`42.2561` is `N_A/(12 ε₀ c²)` in those units — a conversion between unit systems of an
already-computed observable, involving no model quantity, so it uses CODATA. The step *into*
those units, `1 e = 4.803 D/Å`, does involve one: it divides by the Bohr radius, and this crate's
Bohr is MOPAC7's `0.529167 Å`. Mixing the two would be a fraction of a percent on every
intensity — the kind of error that never announces itself.

Rigid-body modes are identified by each mode's **overlap with the translation/rotation subspace**
rather than by a frequency cutoff, so a linear molecule's five rigid-body modes are discovered
rather than assumed from `3N − 6`.

## Wavefunction output (`molden.rs`)

The AM1 valence basis is genuinely Slater-type, so Molden's `[STO]` section represents it exactly
and no Gaussian expansion has to be invented. Its line is `atom kx ky kz kr alfa norm` for the
primitive `norm · x^kx y^ky z^kz r^kr e^{−alfa·r}`, which maps on without residue:

```text
n s   →  kx=ky=kz=0, kr=n−1        (r^{n−1} e^{−ζr})
n p_i →  k_i=1, others 0, kr=n−2   (x r^{n−2} e^{−ζr} = r^{n−1} e^{−ζr} · x/r)
```

**The caveat is structural, not cosmetic.** NDDO *assumes* an orthonormal AO basis — its working
equations are `F C = C ε` with no overlap matrix — so the `[MO]` coefficients live in an
implicitly orthogonalized (Löwdin-like) basis while `[STO]` describes the *un*-orthogonalized
Slater functions a viewer will draw. The two differ by `S^{−1/2}`, and `S` is not the identity
for real Slater functions at bonding distances. Shapes, nodal structure and symmetry are right;
detailed amplitudes in the bonding region are not. This is the same compromise MOPAC's own Molden
output makes, and it is inherent to writing an NDDO wavefunction in a format that presumes a real
basis. It is written into the file as well as here, so it travels with the data.

## Periodic boundary conditions (`lattice.rs`, `pbc/`)

Since the Bloch phase `e^{ik·T}` is 1 at `k = 0`, the Γ-point Hamiltonian
`H(Γ)_μν = Σ_T H_μν(0,T)` is the molecular assembly run over image pairs — the same code, a
different pair list. With a mesh, `H(k) = Σ_T e^{ik·T} H(0,T)` and
`P(0,T) = Σ_k w_k e^{−ik·T} P(k)`.

`S(k) = I` because the AO basis is orthonormal, so this is a **standard** Hermitian
eigenproblem rather than a generalized one, and the gradient again has no Pulay term.

The stress is `σ_αβ = (1/measure) ∂E/∂ε_αβ` at fixed fractional coordinates, with the measure
being a volume, area or length according to the periodicity.

Two things NDDO makes harder than a tight-binding method. The two-centre **exchange** integral
decays as `1/R` and diverges over the image sum at Γ, which k-point sampling fixes and a taper
stands in for at Γ only. And the long-range Coulomb needs **Ewald summation**, which `ewald`,
`ewald1d` and `ewald2d` provide for 3D, 1D and 2D cells under tin-foil boundary conditions.

> **Corrected in 0.2.1.** This paragraph said Ewald summation "is not implemented". It shipped in
> 0.2.0.

### The phased kernel, for DFPT at `q ≠ 0`

The `q`-point response needs the lattice sum carrying a Bloch phase,

```text
Δ(q; d) = Σ_T e^{iq·T} erfc(α|d+T|)/|d+T|
        + (4π/V) Σ_{k ≠ 0} e^{−|k|²/4α²} e^{ik·d} / |k|²  − π/(α²V)·δ_{q≡0},   k = G − q
```

together with its first and second `d`-derivatives. Poisson summation with the phase carried
through puts the reciprocal sum on the lattice **shifted by `−q`**, so the truncating shell has to
be centred on `q` as well — centring it on the origin instead is a 12 eV error, not a small one.

The element dropped is `k = 0`, which occurs only when `q` folds to Γ and is exactly the divergent
term the neutralizing background cancels. Dropping the long-wavelength element `k = −q` instead —
which would keep the direction-dependent part out of `D(q)` so a post-hoc LO–TO term could supply
it — is **not periodic in `q`** and is undefined at a zone boundary where several `k` tie. It was
implemented and rejected on that measurement. The consequence is that `D(q)` is the full dynamical
matrix and must not be composed with the LO–TO path; see [pbc.md](pbc.md), which carries the
validation table.

See [pbc.md](pbc.md) for what all of this means in practice.

## Divide-and-conquer (`divide_conquer.rs`, `fermi.rs`)

Many small diagonalizations instead of one large one, with the pieces recombined by the Yang
partition weight `p^α_μν = ½(d^α_μ + d^α_ν)` and the subsystems sharing electrons through a
single bisected chemical potential. See [divide-conquer.md](divide-conquer.md) for the
formulation, the sum rule, and a precise statement of what became linear.

## Open-shell UHF (`scf.rs`)

For `multiplicity > 1` (or an odd electron count) the SCF is spin-unrestricted: two densities
`P_α`, `P_β` are solved with `F^σ = H_core + J(P_tot) − K(P^σ)` (Coulomb from the total
density, exchange from the same spin), combined-DIIS accelerated, with
`E_elec = ½[P_tot·H_core + P_α·F_α + P_β·F_β]`. The spin density `P_α − P_β` is reported.

## Heavy elements (`overlap_numeric.rs`)

For valence shells with `n ≥ 4` (Zn, Ge, As, Se, Br, Sb, Te, I, Hg), for which no closed-form
Slater-overlap kernel is tabulated, the diatomic overlap is evaluated by Gauss–Legendre
quadrature in prolate-spheroidal coordinates, feeding the same rotation and assembly as the
analytic path.

**Accuracy of the value.** Against the closed form, the quadrature reproduces `1s|1s` (ζ = 1) to
`1e-7` and `2s|2s` at a general exponent to `5e-4` — the figures the tests actually assert.
Earlier documentation claimed `1e-8`, which was never true for the general case. That error sets
where the energy surface sits for the `n ≥ 4` elements.

**Accuracy of the derivatives is a separate question, and a better answer.** The derivative is
taken *through* the quadrature — the quadrature nodes and weights are constants, and the
integrand is differentiated with the same `Dual`/`Dual2` machinery as everything else — so the
result is the exact derivative of the quantity actually being used, not an approximation to the
derivative of the closed form. The quadrature error therefore displaces the surface without
making the gradient inconsistent with it.

`tests/element_coverage.rs` measures this for all 21 elements: the analytic gradient agrees with
a finite difference of the full SCF energy to **1e-9 – 1e-7 eV/Bohr**, and the heavy elements are
no worse than carbon. It is also why molecular dynamics on those elements conserves energy
despite the value error — conservation depends on the force being the gradient of the reported
energy, which it is, not on either being exactly right.

## AM1-BCC (`topology.rs`, `bcc.rs`)

Run the SCF for AM1 Mulliken charges, perceive the molecular graph
(bonds/hybridization/aromaticity), assign atom/bond types, and apply additive bond charge
corrections to reach AMBER-quality partial charges.
