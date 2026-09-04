# Periodic boundary conditions

Energies, analytic forces and analytic stress for chains (1D), slabs (2D) and crystals (3D),
at Γ or on a Monkhorst–Pack k-point mesh, restricted or unrestricted.

The long-range **monopole** electrostatics is summed by Ewald summation, under the tin-foil
boundary condition; the higher multipole channels are still summed in real space to a cutoff.
**Read [Limitations](#limitations) before using this for production work** — that split is what
decides which numbers are converged and which are not.

---

## Setting up a cell

The dimensionality comes from which lattice directions are marked periodic — the same rule ASE
uses. A non-periodic direction still needs a lattice vector; make it long enough to separate the
images.

### ASE

```python
from ase import Atoms
from am1_rs.ase import AM1

# 3D crystal
crystal = Atoms("OH2", positions=..., cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
crystal.calc = AM1(kpts=(2, 2, 2))

# 2D slab: periodic in x and y, vacuum along z
slab = Atoms("OH2", positions=..., cell=[[3.4, 0, 0], [0, 3.4, 0], [0, 0, 24]],
             pbc=[True, True, False])
slab.calc = AM1(kpts=(4, 4, 1))

# 1D chain: periodic along x only
chain = Atoms("OH2", positions=..., cell=[[3.2, 0, 0], [0, 24, 0], [0, 0, 24]],
              pbc=[True, False, False])
chain.calc = AM1(kpts=(8, 1, 1))
```

Mesh entries along a non-periodic direction are ignored — `kpts=(8, 8, 8)` on a chain is an
8-point mesh, not 512. A slab has no dispersion normal to its surface, and sampling it would be
sampling nothing.

### Extended XYZ

`Molecule::from_xyz_str` reads the extended-XYZ comment line:

```
3
Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" pbc="T T T"
O  0.000  0.000  0.000
...
```

### Rust

```rust
use am1_rs::{Lattice, Molecule, Vec3, run_pbc_scf, pbc_energy_and_gradient, PbcOptions, KMesh};

let molecule = molecule.with_cell(
    Lattice::from_vectors(
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(0.0, b, 0.0),
        Vec3::new(0.0, 0.0, c),
        [true, true, false],   // a slab
    )?
);
let (scf, grad) = pbc_energy_and_gradient(&molecule, &params, &PbcOptions {
    kmesh: KMesh::MonkhorstPack([4, 4, 1]),
    ..PbcOptions::default()
})?;
```

Lattice vectors are in **Bohr** in the Rust API and **Ångström** in the Python and ASE APIs, in
each case matching the rest of that surface.

---

## How it works

### Γ is the molecular assembly over image pairs

The Bloch phase `e^{ik·T}` is 1 at `k = 0`, so the Γ-point Hamiltonian

```
H(Γ)_μν = Σ_T H_μν(0, T)
```

is the ordinary molecular assembly run over image pairs. There is no second implementation to
keep in step with the molecular one. Two details the molecular code could take for granted:
contributions **accumulate** (several translations connect the same pair of home-cell atoms),
and a pair may be an atom with **its own image** (`a == b`, `T ≠ 0`).

### k-points

```
H(k) = Σ_T e^{ik·T} H(0,T)
P(0,T) = Σ_k w_k e^{−ik·T} Σ_i f_i(k) C*(k) C(k)ᵀ
```

Because NDDO assumes an **orthonormal AO basis**, `S(k) = I`: this is a standard Hermitian
eigenproblem, not a generalized one, and there is no Pulay term in the gradient. Complex
Hermitian matrices are diagonalized through a real `2n × 2n` symmetric embedding, reusing the
real solver.

Monkhorst–Pack meshes fold `k` and `−k` together by default (exact for a real Hamiltonian,
roughly halving the count) and collapse non-periodic axes to a single point.

Occupations are decided **across all k-points at once** against a single Fermi level, not per
k-point. A band occupied at one `k` and empty at another is the normal case. Set `smearing` for
a metal; a gapped system does not need it.

### Truncation is by lattice translation, not by pair distance

This distinction is the correctness of the sum, not a tuning choice.

NDDO's electrostatics is three `1/R` pieces that cancel: electron–core attraction,
electron–electron Coulomb, and core–core repulsion. For a neutral cell their monopole parts sum
to `Σ_ab Q_a Q_b γ` with `Σ_a Q_a = 0`, so the leading term vanishes and what remains is a
rapidly decaying multipole series. That cancellation is a statement about a **whole image**: it
needs every atom pair of a given translation, or none of them.

A cutoff on the pair *distance* slices through image shells — it keeps an oxygen's contribution
and drops its hydrogens' — and the monopoles then no longer cancel. Early versions of this code
did exactly that, and a 12 Bohr water cell came out 36 Hartree too low. `realspace_cutoff`
therefore bounds `|T|`, and every pair of an admitted translation is included.

---

## Limitations

### The exchange taper

NDDO carries a genuine two-centre **exchange** term whose integral decays as `1/R`. It is finite
only because the density-matrix element it contracts against, `P(ν_a, σ_b)`, decays with
separation. At **Γ-only** sampling the real-space density matrix is `P(0,T) = P(Γ)` for *every*
translation — it does not decay at all — and the exchange sum over images diverges like
`Σ_T 1/|T|`. This is the standard Hartree–Fock exchange divergence at Γ, inherited by NDDO; it
is not an arithmetic slip.

Measured on a single neutral carbon atom, where the monopole terms must cancel exactly:
**−4.440 Ha** isolated, **−4.740 Ha** in a 40 Bohr cell, **−10.101 Ha** in a 15 Bohr cell.

`exchange_cutoff` (Bohr) truncates the exchange where the density matrix would have decayed. It
is a **quintic smoothstep** from `0.8·r_off` to `r_off`, not a step: a discontinuous weight would
make the energy discontinuous in geometry, forces would acquire a delta at the cutoff, and
molecular dynamics would stop conserving energy. The taper has continuous value, first and second
derivatives, and its own derivative is carried through the analytic gradient and stress.

It is applied by **distance alone**, deliberately, not by whether the partner is a periodic
image. The same physical pair is an intra-cell pair in a supercell and an image pair in the
primitive cell; keying the truncation on which one it is would make a supercell disagree with its
own primitive cell — the sharpest test this code has.

**This is a documented approximation, not a convergence parameter you can ignore.** With a
k-point mesh the density matrix decays on its own and the taper matters much less; at Γ in a
dense cell it matters a great deal. Check that your result is stable against it.

### Phonons at arbitrary `q`

Two routes, and they answer different questions.

`ForceConstants::from_supercell` reads `Φ(T)` off an `n`-fold supercell's Γ Hessian and Fourier
transforms it. Exact at every `q` the supercell can represent, an interpolation in between, and
the cost of a finer `q` is the cost of a larger supercell.

`dynamical_matrix_dfpt` solves the response at one `q` directly, at the cost of a primitive cell,
for any `q`. What makes it a different solver rather than the same one with a phase: displacing
atom `b` in cell `L` by `u e^{iq·L}` is not lattice periodic, so it connects `k` to `k + q` and
the whole response becomes a rectangle between the two. Three things follow, each of which is a
separate way to be wrong and each invisible at `q = 0` — the phase on a contribution depends on
which *block* it lands in and not only on which atom moves; the response kernel picks up
`e^{±iq·T}` on its Coulomb couplings, so it cannot be the real Fock builder run on the real and
imaginary parts separately; and the response runs over every band pair rather than occupied ×
virtual, because the empty→occupied half is the antiresonant term and dropping it halves the
answer.

Validated against identities rather than by inspection, because a wrong phase leaves the matrix
Hermitian and the frequencies real:

| check | result |
|---|---|
| `q = 0` against the `q = 0` k-point Hessian | 4 × 10⁻⁹ eV/Bohr², imaginary part exactly 0 |
| the same, on a **shifted** Monkhorst–Pack mesh | 1.3 × 10⁻⁹ relative |
| `q = 0` against a 2-fold supercell frozen phonon | 4.4 × 10⁻¹³ relative |
| `q = ½` against the same | 2.9 × 10⁻⁴ relative |
| `q = ⅓` against a 3-fold supercell | 1.4 × 10⁻⁴ relative |
| an explicit k-list against the mesh it enumerates | 0 (exactly) |
| `D(−q) = D(q)*` | 5 × 10⁻¹⁸ |
| continuity in `q` | 2 cm⁻¹ over `Δq = 0.01` |

The `q = ½` residual is not roundoff and is not any of the usual suspects: it is unchanged when
the real-space cutoff goes from 40 to 90 Bohr, when the long-range correction is switched off on
both sides, and when the degeneracy floor drops by four orders. It is recorded at the size it
has.

**The mesh must match the supercell** when comparing the two routes — a supercell at Γ *is* the
primitive cell at exactly `n` k-points.

#### Choosing `k` as well as `q`

`DfptOptions` takes an arbitrary `KMesh` or an explicit list of k points, and `DfptResult`
returns the band energies, occupations and first-order densities at every `(k, k+q)` pair.

The k set drives the **ground state too**, and that is deliberate rather than an implementation
shortcut: the coupled-perturbed equations assume the zeroth-order state satisfies the SCF
condition, so a response sampled more finely than the density it was built on would be the
response of a different functional, and the frozen-phonon identity would stop holding exactly.
Asking for a finer mesh therefore re-runs the SCF on it. `PbcOptions::kpoints` exists so that the
two share one *resolved* list rather than two independent resolutions of the same description.

Two things are refused rather than worked around. A **time-reversal-folded** mesh: folding pairs
`k` with `−k`, exact for the ground state and a different pairing from `k → k + q`. And a `q`
with a component along a **non-periodic axis**: no lattice translation carries that phase.

> **Fixed in 0.2.1.** Before this the solver built its own Γ-centred grid from `kmesh.sizes()`
> alone, so a `MonkhorstPackShifted` request gave the ground state `{−¼, +¼}` and the response
> `{0, ½}` — two different samplings, one of them not the one the density came from. Nothing
> announced it. The two meshes differ by 2.9 × 10⁻² eV/Bohr² on the test chain, against the
> 1.2 × 10⁻⁹ the identity now holds to.

#### The long-range term in the response

**New in 0.2.1**, and 3D only: `LongRangeMonopole` is three-dimensional, so on a chain or a slab
there is nothing to add and `LongRange::Require` is refused rather than silently ignored.
`LongRange::Auto` — the default — includes it on a 3D cell and accepts its absence elsewhere.

The phased kernel is `EwaldSum::phased_pair_potential`, which returns the value, gradient and
Hessian of

```text
Δ(q; d) = Σ_T e^{iq·T} erfc(α|d+T|)/|d+T|
        + (4π/V) Σ_{k ≠ 0} e^{−|k|²/4α²} e^{ik·d} / |k|²  − π/(α²V)·δ_{q≡0},   k = G − q
```

in one pass. Poisson summation with the phase carried through puts the reciprocal sum on the
lattice **shifted by `−q`**, not on `G` — so the shell that truncates it is centred on `q` too.

**Which element is dropped is `k = 0`, not `G = 0`.** This is a correction to what 0.2.0's docs
recorded as settled. `k = 0` occurs only when `q` is itself a reciprocal lattice vector, i.e.
when `q` folds to Γ, and there it is exactly the divergent term the neutralizing background
cancels — so the rule reduces to this module's tin-foil `Σ_{G≠0}` at `q = 0`. Dropping the
long-wavelength element `k = −q` instead is the tempting alternative, because it keeps the
direction-dependent part out of `D(q)` so that a post-hoc LO–TO term can supply it. It was
implemented, tested and **rejected on two counts**: that rule is not periodic in `q` (shifting
`q` by a reciprocal vector changes which element is dropped), and it has no well-defined answer
at a zone boundary, where several `k` tie for smallest and dropping one of them breaks the
crystal symmetry of `D(q)`.

The consequence follows directly, and it is a change of contract: `D(q)` here is the **full**
dynamical matrix, long-range monopole channel included, so its `q → 0` limit is direction
dependent — which is the physics. It must therefore **not** be combined with
`frequencies_with_lo_to`, whose job is to restore that same physics to the *supercell* route,
where a truncated `Φ(T)` structurally cannot carry it. Use one or the other, not both.

The acoustic sum rule survives because the fixed-charge second derivative carries the phase on
the pair term only, never on the self term:

```text
C(q)_{a,b} = δ_ab [ Q_a Σ_c Q_c Δ'(0; d_ac) ] − Q_a Q_b Δ'(q; d_ab)
```

At `q = 0` the two collapse into `Σ_b` of one bracket and cancel identically.

A phase error here would leave the dynamical matrix Hermitian and the frequencies real, so the
term is validated by identities rather than by inspection:

| check | result |
|---|---|
| `q = 0` reproduces the unphased kernel — value / gradient / Hessian | 2.8 × 10⁻¹⁴, 3.4 × 10⁻¹⁴, 5.8 × 10⁻¹⁴ eV; imaginary part 2.3 × 10⁻¹⁹ |
| gradient and Hessian against finite differences at finite `q` | 6.5 × 10⁻¹⁰ … 4.2 × 10⁻¹⁰ |
| `Δ(−q) = Δ(q)*` | 1.5 × 10⁻¹⁵ |
| `Δ(q + G) = Δ(q)` — the test that rejected the `k = −q` rule | 9.2 × 10⁻¹⁴ |
| **independence of the real-space cutoff at `q = ¼`** | **2.2 × 10⁻¹⁶**, against 1.4 × 10⁻¹ for the truncated sum alone |
| the same at `q = 0` | 4.4 × 10⁻¹⁶, against 1.8 × 10⁻³ |
| at Γ, `D` with the term on against `pbc_hessian` | 1.7 × 10⁻⁸ relative, the term itself being 3.6 × 10⁻² eV/Bohr² |
| acoustic sum rule with the term on | 1.2 × 10⁻⁹ of 1.4 × 10¹ |
| `D(−q) = D(q)*` on a 3D cell | 3.2 × 10⁻¹⁵ |

The cutoff-independence rows are the sharp ones: the correction is *defined* as the exact
lattice sum minus what the pair list already counted, so moving the cutoff moves work between
the two halves and the total must not budge. It does not, to machine precision, at finite `q`.

At the level of the assembled `D(q)` the same test leaves a residual — the correction takes the
cutoff dependence at `q = ¼` from 3.1 × 10⁻² to 2.0 × 10⁻², not to zero. That residual is **not**
the monopole channel, which the row above shows is exact; it is the `R⁻³` Klopman–Ohno tail,
which the monopole correction does not cover and which the real-space sum is left to handle. At
`q = 0` charge neutrality cancels it shell by shell, and at finite `q` the phases spoil that
cancellation, which is why the two `q` differ by an order of magnitude.

The response is **streamed**: each of the `3N` perturbations is solved, contracted into `C(q)`,
and dropped, with the perturbations running in parallel. Resident response memory is therefore
`O(threads · n_k · nao²)` rather than `O(3N · n_k · nao²)`, which was the largest array in the
calculation. `DfptOptions::keep_response` opts back into retaining all of it — that is what
*creates* the array, not merely what returns one — and a test asserts that asking for it leaves
`C(q)` bit-identical.

The contraction is **sparse**, which is the other half of the same problem. `h_j(k)` is held as
its nonzero entries grouped by translation rather than as a dense `nao²` matrix: displacing one
atom changes the Hamiltonian only where that atom appears — `O(1)` blocks — plus, on a 3D cell,
the on-site diagonal of *every* atom, because `∂Δ(q; R_b − R_a)/∂R_a` is nonzero for all `b`.
That last channel is why the structure needs a per-atom entry and not just a neighbour list.

Contracting every pair of perturbations against `O(N)` entries instead of `nao²` makes assembling
`C(q)` **`O(N³ n_k)` rather than `O(N⁴ n_k)`**. `DfptResult` returns `bare_nonzeros` and
`bare_dense_elements` so the claim is checkable from the result rather than believed. Measured on
a chain grown by repeating its cell, the contraction's extent scales as `N^-0.04` against
`N^2.00` for the dense form it replaces, and is already 4.2× smaller at twelve atoms — though
*larger* below about eight, which the test prints rather than hides.

The response solve is **DIIS-accelerated**. It is a linearly mixed fixed point, and without
acceleration a polar 3D cell did not converge inside 200 iterations at all — a water crystal
stalled at 5.6 × 10⁻⁹ against a 10⁻¹⁰ tolerance. Each iteration is a real-space two-electron
build plus a diagonalization per k point, so the count is the cost. `cpscf_tol`,
`cpscf_max_iter` and `cpscf_mixing` are options.

### Response properties: `Z*`, `ε_∞`, LO–TO

`born_charges` returns the Born effective charge tensor of each atom, `Z*_{a,αβ} =
∂(ΩP_α)/∂u_{a,β}` — the dipole a cell acquires per unit displacement of one atom. The cell's
*absolute* polarization is not well defined under periodic boundary conditions, but its
derivative is: charge is conserved, so `Σ_b ∂Q_b/∂u_a = 0` and every origin-dependent piece
cancels. The check that follows from the same argument, `Σ_a Z*_a = 0`, holds to 10⁻¹⁶.

`dielectric_tensor` returns the clamped-ion polarizability and `ε_∞`, from a uniform-field CPHF
coupled to this model's own dipole operator. It is **not** a Berry-phase polarization. The
usual worry about such a perturbation — that the position operator makes it depend on the cell
origin — turns out not to bite here, for the same charge-conservation reason, and
`dielectric_origin_sensitivity` measures it directly: shifting the cell by (1.7, −0.9, 2.3) Bohr
moves `ε_∞` by 9.5 × 10⁻¹⁵ relative. What remains approximate is the clamped-ion, dipole
character of the operator itself, which no amount of origin invariance repairs.

Its **magnitude** is checked separately, and that turned out to matter. Origin-independence,
symmetry and positive-definiteness all constrain `α`'s *shape*, and a value wrong by a constant
factor satisfies every one of them — which is exactly what 0.2.0 shipped: the CPHF is solved with
energies in eV and positions in Bohr, so the assembled `α` is in `e²·Bohr²/eV` and was returned
labelled Bohr³, leaving `ε_∞` **27.21× too close to 1**. The check that catches it compares
against the same molecule's *finite-field* polarizability with no cell — an independent route,
sharing only the SCF and the dipole operator — as the box grows:

| box | periodic mean `α` | distance from the isolated molecule |
|---|---|---|
| 7 Å | 3.4075 Bohr³ | 0.85 % |
| 9 Å | 3.3924 Bohr³ | 0.40 % |
| 12 Å | 3.3846 Bohr³ | **0.17 %** |

against 3.3789 Bohr³ isolated. On the 4.5 Å water crystal `ε_∞` is now `(1.111, 1.097, 1.005)`
rather than `(1.004, 1.004, 1.000)`.

Together those give **LO–TO splitting**, via `ForceConstants::frequencies_with_lo_to`.
`frequencies` leaves it out and is the right function for a non-polar system. The non-analytic
term needs the *direction* along which `q → 0` is being taken, because at exactly `q = 0` it is
undefined — that is the physics, and it is refused rather than guessed.

#### These three are three-dimensional, and 0.2.0 did not enforce it

```text
ε_∞ = 1 + 4πα/Ω          D_NA(q) ∝ (4π/Ω) (q·Z*_a)(q·Z*_b) / (q·ε_∞·q)
```

Both need `Ω` to be a **volume**. `Lattice::measure` returns a *length* for a chain and an *area*
for a slab, and 0.2.0 applied these formulas anyway — so the "127 cm⁻¹ shift and 1631 cm⁻¹ of
direction dependence on a polar water chain" recorded in the 0.2.0 notes was an artifact of a
dimensionally inconsistent denominator, not a physical splitting.

The physics: `4π/(Ω q·ε·q)` is the Fourier transform of the dipole–dipole interaction in three
dimensions. In two it is `2π/(A q)`; in one the non-analytic part vanishes as `q² ln q`, so a
genuinely 1D-periodic chain has **no** LO–TO splitting as `q → 0`. Since 0.2.1 a cell that is not
fully periodic is refused by both `dielectric_tensor` and `frequencies_with_lo_to`.

What is refused there is the **Γ-point** non-analytic correction, and below three dimensions that
is not a gap: multiply the kernels above by the `q²` that charge conservation puts in front of
them and the 2D term goes as `|q| → 0` and the 1D term as `q² ln(1/q) → 0`. There is nothing to
add at Γ. At **finite** `q` the low-dimensional long-range term is summed, and has been since
0.2.2: `LongRangeKernel::phased_pair_potential` dispatches the phased sum `Σ_T e^{iq·T}/|d+T|` to
the 2D Parry and 1D forms as well as to 3D, which is what removed the three-dimensional-only guard
from the DFPT response. The same fact reappears in `pbc::dielectric_function`, described under
"Not implemented, as of 0.2.2" below: `ε_∞` is a constant only in three dimensions, and a slab or
a chain has `ε(q) → 1`.

Measured on a polar molecular crystal (one water per 4.5 Å cubic cell): the added term equals its
closed form to 2 × 10⁻¹⁵, it raises eigenvalues by up to 3 × 10⁻³ eV/(Å²·amu) and lowers none
(a rank-one positive-semidefinite update cannot), and the `q → 0` limit differs by 1.04 cm⁻¹
between two approach directions. On a homonuclear crystal, where inversion symmetry and the
acoustic sum rule force `Z* = 0` — measured at 4 × 10⁻¹⁴ e — it is identically zero.

### Ewald summation covers the monopole channel only

The `1/R` **monopole** term is summed exactly by Ewald summation (`ewald: true`, the default),
in three dimensions. Everything else — the `R⁻³` correction that the Klopman–Ohno kernel adds to
the monopole channel, and the whole higher-multipole series from the Dewar–Sabelli–Klopman
expansion — is still summed in real space to `realspace_cutoff`.

That split decides what is converged:

* **A charged 3D cell now has a meaningful energy.** See [below](#charged-cells).
* **A neutral cell barely moves**, because its monopole terms already cancelled; switching Ewald
  on changes a neutral water cell by 3 × 10⁻³ eV.
* **A slab and a chain get their own correction**, not the three-dimensional one. `ewald` was
  silently inactive below three dimensions until 0.2.0 shipped the 2D **Parry** sum and the 1D
  real-space-plus-Hurwitz-zeta sum; `LongRangeKernel::for_lattice` dispatches on the cell and each
  form is Madelung-exact to 1e-10 and independent of the splitting parameter. A charged slab or
  chain is still refused, but for a different reason than a missing sum: its energy is not well
  defined without also
  choosing a convention for the neutralizing background, which this version does not offer.
* **The residual `R⁻³` term was logarithmically divergent in 3D.** It is small, but it did not
  converge: on the +1 water cell it contributed about 0.10 eV per unit `ln(r_c)`. **Corrected in
  0.2.2**; the same measurement now reads 0.000 eV per unit `ln(r_c)` to three decimals. See
  [The `R⁻³` tail](#the-r3-tail).

The remaining real-space series converges **slowly**, and the total energy keeps moving as the
cutoff grows. Measured on a 1D water chain (3.2 Å spacing, 3 k-points, where Ewald does not
apply at all):

| `realspace_cutoff` (Bohr) | energy per cell (eV) |
|---|---|
| 40 | −348.6200173 |
| 80 | −348.6202523 |
| 160 | −348.6203051 |
| 320 | −348.6203189 |
| 640 | −348.6203225 |

That is 3 × 10⁻⁴ eV of drift from 40 to 640 Bohr, still moving at the 4 × 10⁻⁶ eV level at the
end. For relative energies at fixed cell it largely cancels; for absolute energies, or for
comparing different cells, **converge the cutoff explicitly**.

The same effect means two different cells describing the same system agree only once both are
converged. A 3-k-point primitive cell and its Γ 3× supercell differ by 6.8 × 10⁻⁵ eV at the
40 Bohr default — purely because the same `|T|` bound admits different sets of physical pairs in
the two — and by 3.4 × 10⁻⁹ eV at 320 Bohr.

### Charged cells

A net charge per cell is supported in 3D. The electron count, the self-consistent density, the
Mulliken charges (which sum to the formal charge to `1e-10`), the forces and **the total energy**
are all correct and mutually consistent, under the tin-foil boundary condition that a
neutralizing background implies.

The energy part is what Ewald summation buys. Without it the monopole lattice sum `Σ_T Q²/|T|`
diverges, and the truncated result grows without bound with the cutoff. Measured on a +1 water
cell in an 8 Å cube:

| `realspace_cutoff` (Bohr) | `ewald: true` (eV) | `ewald: false` (eV) |
|---|---|---|
| 20 | −339.12305 | −331.198 |
| 40 | −339.20489 | −298.311 |
| 90 | −339.28437 | −137.227 |
| 130 | −339.31999 | **+72.191** |

Over that 6.5× range the corrected energy moves **0.20 eV**; the uncorrected one moves
**403 eV** and turns positive. The uncorrected number is exactly the missing background: the
continuum estimate `π Q² r_c² / V` predicts a 408 eV rise, and the measurement is 0.988 of it.

The remaining 0.20 eV is not noise, and `tests/pbc_charged.rs` identifies it rather than
tolerating it: its increments are constant per unit `ln(r_c)` (−0.118, −0.098, −0.097 eV), which
is the signature of the logarithmically divergent `R⁻³` part of the Klopman–Ohno kernel. Through
0.2.1 that was the one long-range piece Ewald did not cover, and the table above is what the run
looks like without it.

**Covered since 0.2.2** by `PbcOptions::klopman_ohno_tail`, on by default — see
[The `R⁻³` tail](#the-r3-tail). `tests/pbc_klopman_ohno_tail.rs` sweeps a 4× range of cutoff both
ways and measures the slope rather than asserting the fix: `dE/d(ln r_c)` falls by more than a
factor of four, and the residual steps go 5e-4 → 3e-4 → 1e-4 eV — shrinking, so what is left is a
power law and not a logarithm. It is a residual, not zero, and the switch exists so the difference
stays measurable.

**What this means in practice.** With the tail off, absolute charged-cell energies are meaningful
to about 0.1 eV, limited by that `R⁻³` residual rather than by a divergence; with it on, the limit
is the sub-meV power-law remainder. Comparisons at fixed cutoff are better than either, because
the residual is common to both sides.

#### Two consequences of the tin-foil convention

**A polar molecule in a box never reaches its gas-phase energy.** Its dipole interacts with the
infinite lattice of its own images, and that interaction is `−2π|p|²/3V` — it falls as `L⁻³`
rather than vanishing at any finite cell. Water in a 45/60/90 Bohr cube sits 1.17/0.49/0.15 ×
10⁻⁴ eV below the molecular result, and `d·L³` is constant to 0.6 %. The relevant dipole is the
**point-charge** one built from the net atomic charges, since the monopole correction never sees
the sp-hybridisation contribution to the reported AM1 dipole.

**Switching Ewald off changes the convention, not just the accuracy.** A spherically truncated
real-space sum silently imposes the vacuum (`ε = 1`) convention instead of tin-foil (`ε = ∞`).
The two differ by exactly that surface term, which is why the neutral-cell energy shifts by
3 × 10⁻³ eV when the flag is toggled.

Every charged periodic result carries a `charged_cell_warning` saying so; the ASE calculator
raises it as a `RuntimeWarning` once per calculator.

The fix is a neutralizing jellium background, which needs the Ewald split. Note that in 2D and
1D even that would not be enough: the background for a charged slab is a sheet whose pair energy
`−2πσ²|z|` depends on where the sheet is placed, and the potential of a charged line diverges
logarithmically. Those cases need an explicitly chosen convention, and this code does not pretend
to have chosen one.

### What implementing Ewald here actually requires (the design note)

Recorded because it is not the textbook problem, and the difference is the whole difficulty.

The kernel is not `1/R`. It is the Klopman–Ohno form `γ_η(R) = 1/√(R² + η²)` with
`η = ρ_a + ρ_b` depending on the pair. Splitting it the usual way,

```
γ_η(R) = [γ_η(R) − 1/R] + 1/R
```

sends the second term to a standard Ewald sum and leaves a bracket that behaves as `−η²/2R³` at
large `R`. **`Σ_T |T|⁻³` is logarithmically divergent in three dimensions**, so the bracket
cannot simply be summed in real space.

It is worth checking whether neutrality rescues it, because for the leading `1/R` term it does.
It does not. Expanding `η_ab² = ρ_a² + 2ρ_aρ_b + ρ_b²` and using `Σ_a Q_a = 0`:

```
Σ_ab Q_a Q_b η_ab²  =  (Σ_a Q_a ρ_a²)(Σ_b Q_b)  +  2(Σ_a Q_a ρ_a)²  +  (Σ_a Q_a)(Σ_b Q_b ρ_b²)
                    =  2 (Σ_a Q_a ρ_a)²
```

The first and third terms vanish for a neutral cell; the middle one does not. So a neutral NDDO
cell carries a genuine `R⁻³` log divergence with coefficient `2(Σ_a Q_a ρ_a)²`, and a correct
implementation needs the generalized Ewald machinery for that channel — a `G = 0` logarithmic
term and its matching self-energy — on top of the standard `1/R` sum.

For the **charged** case the missing piece is the neutralizing background, and its size is
already measured above: the divergence tracks the continuum estimate `π Q² r_c² / V` to 1.2 %,
so the jellium term is exactly what is absent, not something else.

The higher multipole channels (`R⁻²` through `R⁻⁵`) are easier than they look — `Σ_T R⁻p`
converges absolutely for `p > D`, so in 3D only ranks 0 and 1 need reciprocal-space treatment,
in 2D only rank 0 and 1, and in 1D only rank 0. Reduced dimensionality is *less* work than 3D
here, not more.

Any implementation should be validated three independent ways before being believed: the result
must be independent of the splitting parameter α; the Madelung constant of rocksalt must come out
as 1.747565; and the neutral-cell energy must agree with the converged real-space sum this
version already produces, which shares almost no code with an Ewald path.

The `R⁻³` channel this note is about **was implemented in 0.2.2**, and not the way the note
expected. See [The `R⁻³` tail](#the-r3-tail) below; the multipole ranks above it are still
real-space, and the argument here about which ranks need reciprocal treatment in which
dimensionality is unchanged.

<a id="the-r3-tail"></a>
### The `R⁻³` tail

The note above concluded that fixing this needs "the generalized Ewald machinery for that channel
— a `G = 0` logarithmic term and its matching self-energy". That is one way. What 0.2.2 does is
cheaper and, on the measurement below, sufficient.

**The one thing that genuinely needs a convention is the logarithm, and only in 3D.** Everything
else about `Σ_{|T|>r_c} [γ_η(|T|) − 1/|T|]` is an ordinary convergent sum that can simply be
evaluated. So:

1. **Sum the dropped translations explicitly**, out to three cutoffs, with the *exact*
   `γ_η(R) − 1/R` rather than its `−η²/2R³` expansion. This is the part where a continuum
   approximation is worst, because it is where the lattice is least like a uniform density, and it
   is cheap: the translations beyond `r_c` and within `3 r_c` number a few thousand, and the sum
   depends on the pair only through `η_ab = ρ_a + ρ_b`, so it is evaluated per *element pair*, not
   per atom pair.
2. **Hand over to a continuum remainder** past that, with a quintic taper so the energy, the force
   and the stress stay differentiable across the handover. A sharp handover makes the energy jump
   by `γ_η(R) − 1/R ≈ 1.6e-4` eV whenever a translation crosses it under strain, which is five
   orders above the stress tolerance.
3. **The remainder's integrand is per-dimensionality**: `4πr²/V` for a crystal, `2πr/A` for a slab,
   `2/L` for a chain. Only the crystal's is logarithmically divergent — `Σ_T |T|⁻³` *converges* in
   one and two dimensions — and only there does a reference length appear. It is
   `KLOPMAN_OHNO_REFERENCE = 1 Bohr`, the same treatment the charged chain's line charge gets, and
   `ln r₀ = 0` makes this particular choice add nothing of its own.

Applying the 3D form in every dimensionality was the first draft's bug, and it moved a charged
chain's energy by 3e-2 eV. It is the same dimensionality error `docs/scope.md` records for `ε_∞`
and LO–TO, made a second time.

**Measured**, on the +1 water cell over a 6.5× range of cutoff (20 → 130 Bohr):

| | residual per unit `ln r_c` (eV) | energy spread (eV) |
|---|---|---|
| no Ewald | — | 403.4 |
| Ewald, no tail | −0.118, −0.098, −0.097 — constant, i.e. the logarithm | 0.197 |
| Ewald + tail | 0.000, −0.000, −0.000 | 6e-5 |

Three numbers rather than one, because a *constant* increment per unit `ln r_c` is what identifies
the residual as the logarithm rather than as something else; asserting a small number would not
distinguish the two.

The tail is a per-pair constant, so it has no gradient at the order it is kept to — measured, it
moves the forces by 0.08 % of their scale, which is the converged density shifting under the Fock
diagonal it adds, not a stray gradient term. Its **strain** derivative is a separate term from the
pair virial for the same reason, and with it the stress matches a strain finite difference to
6.9e-9 eV/Bohr³.

It is carried through the **response** as well, with the same constant at every `q`. That is not an
approximation of convenience: the cutoff-dependent part of the tail is `−(4π/V) ln r_c`, which does
not depend on `q`, so one constant removes the truncation dependence at every `q`. Leaving it out
of the response while the ground state carried it made `D(q = 0)` disagree with the `q = 0`
Hessian by 4.6e-4 eV/Bohr² — the response of a Hamiltonian the SCF had not converged. What is *not*
captured is the residual `−(4π/V) ln(q r₀)`, a weak non-analyticity at `q → 0`, weaker than the
monopole's own `4π/(V q²)` which is the LO–TO discontinuity and is treated exactly.

**Where it is least good.** The continuum remainder assumes the lattice beyond `3 r_c` is a uniform
density. When `r_c` does not span several lattice repeats — a small molecule in a large box, where
the cutoff can be *smaller* than one repeat — that assumption covers a handful of discrete shells
and the remainder is crude. The explicit sum in step 1 is what makes this a bounded error rather
than a wrong one, but a dilute cell is where this correction is least trustworthy, and the honest
statement is that it is a continuum estimate there rather than a lattice sum.

### Not implemented, as of 0.2.2

This list was stale through 0.2.0 — it still named Ewald summation, the periodic Hessian, phonons,
DFPT and periodic divide-and-conquer as missing, all of which 0.2.0 shipped. Struck-through
entries are ones later releases closed, kept so the list reads as a record rather than a boast.
What is genuinely absent:

- ~~**The `R⁻³` Klopman–Ohno tail under Ewald.**~~ **Done in 0.2.2** — see
  [The `R⁻³` tail](#the-r3-tail) below. It is not the generalized Ewald sum this entry
  anticipated: the translations the pair list dropped are summed *explicitly* out to three cutoffs,
  and only the far remainder is continuum, which is where the one logarithm lives and where the
  reference length cuts it. What remains unfixed is the accuracy of that remainder when the cutoff
  does not span several lattice repeats — a dilute cell, where the near shells the explicit sum
  covers are few and the continuum is approximating a handful of discrete neighbours.
- ~~**`ε_∞` in 1D and 2D — the *thickness-assigning conversion* only.**~~ ✅ since 0.2.2,
  `pbc::dielectric_tensor_with_extent`. `ε_∞ = 1 + 4πα/Ω` needs `Ω` to be a volume, and turning a
  slab's `α/A` (a **length**) into a dimensionless constant needs a thickness: a choice about the
  material, not something a supercell fixes. So it is a **required argument** rather than a
  default, the way `chain_radius` and `AxisConvention` are — and see
  [The conversion is a depolarization problem](#the-conversion-is-a-depolarization-problem) below
  for why it is not a division.

  Everything either side of it was available first. The **polarizability `α`** in every
  dimensionality is `pbc::polarizability`, and the **dielectric function** is
  `pbc::dielectric_function`, which is `ε(q)` in every dimensionality:

  ```text
  ε(q) = 1 − v_d(q) χ⁰(q),      χ⁰(q) → −q² (q̂·α·q̂) / measure
  ```

  with `v_d` the bare Coulomb kernel — `4π/q²`, `2π/|q|`, `2K₀(|q|ρ)`. In three dimensions the `q²`
  cancels and what is left is the constant `ε_∞`; below three it does not, and `ε(q) → 1` at long
  wavelength. **A sheet or a wire has no long-wavelength dielectric constant** — it does not screen
  a field whose wavelength exceeds its own extent — and that is the same fact as the LO–TO entry
  below, not a second limitation. Measured: a slab's `ε(q) − 1` fits an exponent of 1.000, a
  chain's climbs 1.64 → 1.71 toward the 2 that `q²K₀` gives up to its logarithm, and in 3D `ε(q)`
  reproduces `dielectric_tensor`'s constant at every `q` to 1e-9.

  The two-dimensional form is **thickness-free**, which is what makes `2π χ₂D` — the
  Rytova–Keldysh screening length — an intrinsic property of the layer. A chain needs a transverse
  radius for its logarithm and is required to supply one.
- ~~**LO–TO in 1D and 2D.**~~ ✅ nothing to add, and measured. The long-range kernel diverges in
  every dimensionality, but the contribution to `D(q)` carries `q²` from charge conservation, so
  only 3D keeps a finite direction-dependent limit and is discontinuous at Γ — which *is* the
  splitting. `|D(q) − D(0)|` at `q = 0.02, 0.01, 0.005` gives 3.6e-3 → 1.8e-3 → 9.0e-4 for a
  chain, 4.6e-3 → 2.2e-3 → 1.1e-3 for a slab, and 1.071e-1 → 1.074e-1 → 1.075e-1 for a crystal.
  The low-dimensional cases reach Γ; the crystal does not. So there is no splitting to add below
  three dimensions, and the non-analytic *approach*, which is real, the DFPT path carries exactly.
- ~~**The long-range monopole term in the DFPT response on a chain or a slab.**~~ ✅ since 0.2.2.
  It landed in 0.2.1 for 3D cells; what was three-dimensional was the *phased* kernel, not
  `LongRangeMonopole`, and `LongRangeKernel::phased_pair_potential` now dispatches to the 2D Parry
  and 1D forms as well. `LongRange::Require` errors only on a cell with no periodic direction at
  all, where the statement is true by definition.
- ~~**Berry-phase polarization.**~~ ✅ since 0.2.2 — `pbc::polarization`, the King-Smith–Vanderbilt
  discretized phase. `ε_∞` remains a clamped-ion *dipole response*, which is a different quantity
  and still what `dielectric_tensor` reports; the Berry phase is what makes the finite field and
  the polarization quantum available. Three-dimensional and restricted, per the two entries above.
- ~~**A finite field *along* a periodic direction.**~~ ✅ since 0.2.2, `pbc::run_finite_field` —
  the Berry-phase electric enthalpy `E − Ω 𝓔·P`, not a modified `F·R`. A field **orthogonal** to
  every lattice vector needs none of that machinery and is `PbcOptions::electric_field`, also new
  in 0.2.2; through 0.2.1 both cases were refused together.
- ~~**The on-site `s`–`p` moment in the Berry phase.**~~ ✅ since 0.2.2. The link operator was a
  diagonal `e^{−ib·τ_μ}` and is now the exact same-atom block, `e^{−ib·τ_a} exp(−i b·D^a)`, whose
  generator is the same `dd` the CPHF dipole operator uses. Born charges against the CPHF went
  0.207 → **1.2e-3 e** at matched k-sampling, the finite-field `α` on a water crystal 12 % →
  **0.05 %**, and a planar cell's out-of-plane `α` — which is *entirely* that moment — from exactly
  zero to 0.25527 against the CPHF's 0.25564.
- ~~**Unrestricted (open-shell) k-point response.**~~ ✅ since 0.2.2 for the `q = 0` Hessian, the
  Born charges, **DFPT at finite `q`**, and `ε_∞`/the polarizability. Still restricted: the Berry
  phase and the finite field built on it.
- ~~**Open-shell analytic stress**~~ ✅ since 0.2.2 — the spin-resolved pair virial.
- **SAM1.** A different integral engine, not a reparameterization; AM1 and RM1 are available.

### The conversion is a depolarization problem

`pbc::dielectric_tensor_with_extent` takes a thickness (slab, Bohr) or a cross-section (wire,
Bohr²) and returns `ε_∞`. The obvious step would be `ε = 1 + 4πα/(measure · extent)`. That is
right along directions where the induced polarization creates no macroscopic field, and **wrong
along the others**, because the `α` this crate computes is the response to the *external* field:
the induced charges interact through the same Coulomb operator the SCF uses, so for a slab
polarized along its normal the depolarizing field is already inside `α`. Dividing and adding one
would count the screening once and the shape not at all.

With `χ = α/(measure · extent)` and `N` the depolarization factor of the assumed body,

```text
ε = 1 + 4πχ / (1 − 4πNχ)
```

| body | axis | `N` | `ε` |
|---|---|---|---|
| slab | in plane | 0 | `1 + 4πχ` |
| slab | along the normal | 1 | `1/(1 − 4πχ)` |
| wire | along the axis | 0 | `1 + 4πχ` |
| wire | transverse, circular section | ½ | `(1 + 2πχ)/(1 − 2πχ)` |
| crystal | any | 0 | `1 + 4πχ` — which is `dielectric_tensor` |

The crystal row is not bolted on. Three-dimensional tin-foil boundary conditions remove the
macroscopic depolarizing field, so `α` there is already the response to the internal macroscopic
field and `N = 0` is the correct entry; feeding a crystal's `α` through the `N = 0` branch
reproduces `dielectric_tensor` to 1e-13.

**Which direction the inverse law runs** is worth stating, because both readings sound right. At
fixed `α`, `ε_⊥ > ε_∥`: the same external response needs a stronger intrinsic one to overcome the
depolarization. At fixed `ε`, `α_⊥ < α_∥`, which is the observable statement — and it saturates,
`4πα_⊥/A < d` for any material at all, because a slab cannot expel more field than a perfect
conductor. That bound is why the out-of-plane law has a pole, and why an extent smaller than the
computed response implies is an error rather than a negative `ε`.

`tests/pbc_dielectric_extent.rs` measures the premise rather than asserting it. Tightening a
two-dimensional methane lattice from 14 Å to 6.5 Å moves the two channels in **opposite**
directions — `α_xx` 8.293 → 8.439 Bohr³, `α_zz` 8.230 → 7.920 — which is what a sheet of induced
dipoles does and what an internal-field response would not show at all.

#### What survives the choice

The thickness is a choice, so `ε` is a choice. Two combinations are not:

```text
(ε_∥ − 1) · d = 4π α_∥ / A            (1 − 1/ε_⊥) · d = 4π α_⊥ / A
```

Half the first is the Rytova–Keldysh screening length. Read the other way round these are
capacitor stacking — a layer of thickness `d₁` padded with vacuum out to `d₂` is `d₂ε(d₂) =
d₁ε(d₁) + (d₂ − d₁)` in parallel and `d₂/ε(d₂) = d₁/ε(d₁) + (d₂ − d₁)` in series — so the two
formulas are *forced* once the thickness is named. That is an independent derivation and is tested
as one. The first also has to agree with `dielectric_function`, which reaches the same sheet
susceptibility through a reciprocal-space Coulomb kernel instead of a real-space capacitor: the
measured ratio is 2.0000000000.

Both are returned alongside `ε` by the Python and ASE entry points, as `sheet_susceptibility` and
`inverse_sheet_response`.

---

## Stress

`σ_αβ = (1/measure) ∂E/∂ε_αβ` at fixed fractional coordinates.

The **measure** is a volume in 3D, an **area** in 2D and a **length** in 1D, because a
non-periodic direction has no extent to divide by. So the units are eV/Å³, eV/Å² and eV/Å
respectively. Only the 3D case is an ASE stress in the usual sense; do not hand a slab's or a
chain's Voigt vector to a variable-cell driver, which assumes 3D.

Stress components touching a non-periodic axis are **exactly zero** (`0.0`, not small).

Verified against a strain finite difference to 5 × 10⁻⁹ eV/Bohr³ in Rust and, through the ASE
boundary in ASE's own Voigt order and units, to 3 × 10⁻⁸ eV/Å³ on all six components.

```python
atoms.get_stress()              # Voigt: [xx, yy, zz, yz, xz, xy]
atoms.get_stress(voigt=False)   # full 3x3
```

A molecular structure raises `PropertyNotImplementedError` rather than returning zeros — a
molecule in free space has no stress, and zeros would let a variable-cell optimizer run happily
on nothing.

---

## The ZDO diagnostic: `max_image_overlap`

NDDO *assumes* an orthonormal AO basis. Under a periodic cell that assumption says an AO is
orthogonal to its own image one cell away, which is not defensible in a small cell.

Every periodic result reports `max_image_overlap` — the largest `|S_μν(T)|` over non-zero
translations. This is the size of the model's own error, and it is cheap. Above roughly **0.4**
the cell is too small for the method to mean anything. For reference: bulk Si in its 2-atom
primitive cell gives 0.221; a water molecule in an 8 Å cube gives 0.067.

---

## Molecular dynamics

`tests/test_ase_pbc_md.py` runs real ensembles as an acceptance test, and is the best worked
example of the API.

Measured there:

- **NVE** conserves energy in 1D, 2D and 3D, with drift 0.1–0.2 % of the potential energy
  exchanged during the run. This is the sharpest check on the forces: a dropped image pair, a
  missing taper derivative or a wrong Bloch phase all leave the single-point energy intact and
  make the total energy walk away.
- **NPT** (Parrinello–Rahman with a Nosé–Hoover thermostat, `ase.md.npt.NPT`) runs and stays
  bounded, and the cell volume responds monotonically to the applied pressure. ASE's `NPT`
  requires an upper-triangular cell, so reduce or triangularize first.
- **NVT** on a chain and a slab runs, with no stress in the non-periodic directions.

Use a timestep suited to the stiffest mode: unconstrained O–H stretches have a period of about
9 fs, so 1 fs is already too coarse and 0.25–0.5 fs is appropriate.

**Tighten `e_tol`/`p_tol` before differentiating the energy numerically.** The defaults suit
dynamics, where the geometry moves every step and the convergence error is common to consecutive
steps. Two geometries `1e-4` apart converge along different paths, and the error does not cancel
between them — an apparently large stress "error" is usually this.

---

## Options reference

| Option | Default | Meaning |
|---|---|---|
| `kpts` | `(1,1,1)` | Monkhorst–Pack mesh; non-periodic axes collapse to 1 |
| `smearing` | `0.0` eV | Fermi–Dirac width kT. Needed for a metal |
| `realspace_cutoff` | `40.0` Bohr | Largest lattice translation `\|T\|` included |
| `exchange_cutoff` | `20.0` Bohr | Distance over which two-centre exchange is tapered off |
| `charge` | `0.0` | Charge **per cell**; see [Charged cells](#charged-cells) |
| `multiplicity` | `1` | Fixes the α/β electron counts |
| `e_tol`, `p_tol` | `1e-8`, `1e-7` | SCF thresholds; tighten for finite differences |
| `max_scf` | `300` | Not converging raises, rather than returning a bad result |
| `mixing` | `0.3` | Linear mixing of the real-space density |
