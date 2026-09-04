# Divide-and-conquer

An SCF spends most of its time diagonalizing one `N × N` Fock matrix, at `O(N³)`.
Divide-and-conquer replaces that with many small diagonalizations whose size stops growing with
the molecule, so the diagonalization cost becomes linear in the number of atoms.

References: Yang, *Phys. Rev. Lett.* **66**, 1438 (1991); Yang & Lee, *J. Chem. Phys.* **103**,
5674 (1995); Akama, Kobayashi & Nakai, *J. Comput. Chem.* **28**, 2003 (2007).

---

## What is linear, and what is not

**Read this section before quoting a scaling figure from this project.**

| Part of the calculation | Scaling with divide-and-conquer | Counter |
|---|---|---|
| Subsystem diagonalizations | **`O(N)`** | `diagonalization_work` (`Σ_α n_α³`) |
| Two-centre **exchange** | **`O(N)`**, exactly | `exchange_work` |
| Density-matrix storage | **`O(N)`** | `retained_density_blocks` |
| **DIIS history** (the dominant memory term) | **`O(N)`** since 0.2.1 | `diis_pattern_elements` vs `dense_triangle_elements` |
| Two-centre **Coulomb** | `O(N²)` — unchanged | `coulomb_work` |
| Core–core repulsion | `O(N²)` — unchanged | — |

So the honest summary is: **divide-and-conquer makes the diagonalization linear; the whole
calculation is not linear, and asymptotically approaches `O(N²)` rather than `O(N)`.**

The reason the Coulomb sum stays quadratic is not an oversight. NDDO's two-centre two-electron
integrals decay as `1/R`. A distance cutoff on them does not screen away work that did not
matter — it changes the answer. Reducing that sum below `O(N²)` needs a multipole or Ewald
treatment of the long-range part.

`multipole_cutoff` is the partial version of that: beyond the cutoff a pair's full
Dewar–Sabelli–Klopman block collapses onto its monopole term, which keeps the interaction and
simplifies only its *shape*. It lowers the prefactor by roughly a hundred; it does not touch the
exponent.

**A tree evaluation would remove the exponent, and was not built.** Measured first, on a
1029-atom run with the far field on: `farfield:potential` is 0.5 % of the runtime, against 36 %
for the subsystem diagonalizations. A monopole pair costs about ten flops and the loop is
embarrassingly parallel, so an `O(N log N)` version would save half a percent — and would put an
acceptance-angle discontinuity into the energy surface to do it. The `O(N²)` term does win
eventually, but the crossover is around `10⁴–10⁵` atoms. This is precisely the trap the section
below warns about, and `src/farfield.rs` records the measurement so the decision can be revisited
against a number rather than re-argued.

The exchange, by contrast, becomes linear **exactly** rather than by approximation. The
exchange contribution for a pair contracts against the density-matrix block connecting the two
atoms, and divide-and-conquer sets that block to exactly zero beyond the buffer radius. Nothing
is being neglected at that step that was not already neglected by the method itself.

These counters are returned on every result so the claim can be checked rather than believed.
`tests/divide_conquer.rs` asserts each exponent separately — including the quadratic one, so
that a later change cannot quietly leave the impression that everything became linear.

Measured over 12 → 96 atoms (a water chain, 11 Bohr buffer):

```
scaling exponents: diagonalization 1.146, exchange 1.058,
                   retained blocks 1.053, Coulomb 2.015
```

And on 3D water clusters from 375 to 2187 atoms, where the buffer reaches in every direction
rather than along one:

| atoms | subsystems | largest (AO) | `Σn³` per atom | retained blocks per atom |
|---|---|---|---|---|
| 375 | 32 | 297 | 641 k | 30.7 |
| 648 | 54 | 302 | 654 k | 33.4 |
| 1029 | 86 | 344 | 991 k | 35.0 |
| 1536 | 128 | 233 | 478 k | 34.8 |
| 2187 | 183 | 391 | 1327 k | 35.5 |

Fitted exponent of `Σn³` against the atom count: **1.25**, against 3 for a single full
diagonalization. Retained density blocks per atom is flat, which is the `O(N)` memory claim.

The per-size spread is not noise and not growth — it is how compact the cores are, which depends
on how the core count factorizes. 1536 atoms at `core_size = 12` is exactly 128 cores, which
recursive bisection lays out as a regular 8×4×4 grid of near-cubic boxes; 2187 wants 183, which no
sequence of binary splits tiles evenly, so some come out elongated. An elongated core reaches far
more atoms within a buffer radius than a compact one holding the same atoms. Measured directly,
the ratio of largest subsystem to mean runs 1.62, 1.60, 1.62, **1.36**, 1.65 across those five
sizes — the dip falling exactly on the power-of-two count. Core *sizes* themselves are uniform
(11–12 atoms throughout), so this is shape, not balance.

---

## The formulation

### Partition

Atoms are split into disjoint **core** regions by recursive bisection along whichever axis the
group is currently widest in. Compactness is the point: a subsystem's cost is set by how many
atoms fall inside the buffer around its core, and a ball-shaped core has a much smaller buffer
than a slab-shaped one with the same atom count.

Each core is then padded into a **subsystem** with every atom within `buffer_radius` of any of
its core atoms.

### The Yang partition weight

Each subsystem contributes to the global density matrix with

```
p^α_μν = ½ (d^α_μ + d^α_ν),      d^α_μ = 1 if μ sits on a core atom of α, else 0
```

The cores are disjoint and cover every atom, so `Σ_α d^α_μ = 1` and the weights sum to one.

### Why the density is truncated at the buffer radius

That sum rule holds only if every subsystem owning half of a block also *contains* both atoms
of that block. With a buffer defined as "within `r_buf` of any core atom", this is **not**
automatic: atom `b` can be close to some core atom of α while being far from `a`, and then the
subsystem β that owns `b` has no reason to contain `a`. Half the weight would silently go
missing, and it would go missing in a geometry-dependent way.

This implementation fixes it at the source. A block `P_ab` is kept **only when
`|R_a − R_b| ≤ buffer_radius`**. Then `a` in the core of α forces `b` to lie within `r_buf` of a
core atom of α — namely `a` itself — so `b ∈ α`; and the mirror argument puts `a ∈ β`. The sum
rule becomes exact for every geometry, with no condition on how the partition happened to fall.

This is not an extra approximation added to make the bookkeeping work. It is the *same*
approximation divide-and-conquer already makes — that the density matrix of a gapped system is
short-ranged — written down explicitly instead of left to emerge from the buffer's shape.

`tests/divide_conquer.rs` checks the sum rule directly, at four buffer radii. It comes out
**exactly** `0.0` deviation, not "small".

### The common chemical potential

Solved separately, each subsystem would keep the electron count it started with, and charge
could never flow between subsystems. Instead every subsystem level from every subsystem is
filled against **one chemical potential**, bisected so the total electron count comes out right.

Each level enters the count weighted by the fraction of it that lives on the core:

```
n^α_i = Σ_{μ ∈ core α} |c^α_{μi}|²
```

with no overlap matrix, because NDDO's AO basis is orthonormal — the same simplification that
removes `S` from the SCF equations removes it from here.

Those fractions are not integers, so the occupied set never sums to a whole number and the
filling is genuinely fractional. That is why `smearing_ev` defaults to a small non-zero value:
with sharp aufbau filling the frontier would have to be resolved by sort order, which is a
discontinuous function of geometry, and dynamics cannot use it.

**Unrestricted runs get one chemical potential per spin channel**, not one shared level. The
multiplicity fixes the α and β electron counts separately, so each channel is filled to its own
count; a single shared level would let the two exchange electrons and the multiplicity would
drift away from what was asked for.

### Non-neutral systems

Nothing special is required. The charge sets the total electron count, the common chemical
potential distributes it across subsystems, and the Mulliken charges sum to the formal charge
to `1e-8` or better. `tests/divide_conquer.rs` checks −1, 0 and +1.

---

## Accuracy, and how to control it

`buffer_radius` is the method's one physical parameter. It is the distance beyond which the
density matrix is taken to vanish, and increasing it drives the answer monotonically to the full
SCF. Measured on a 6-water chain:

| buffer (Bohr) | `ΔE` vs full SCF (eV) | `ΔE` per atom (eV) | max gradient error (eV/Bohr) |
|---|---|---|---|
| 6 | 4.9 × 10⁻² | 2.7 × 10⁻³ | 1.8 × 10⁻² |
| 9 | 1.8 × 10⁻³ | 1.0 × 10⁻⁴ | 1.2 × 10⁻³ |
| 12 | 1.5 × 10⁻⁴ | 8.3 × 10⁻⁶ | 1.8 × 10⁻⁴ |
| 16 | 3.5 × 10⁻⁵ | 1.9 × 10⁻⁶ | — |
| 22 | 6.9 × 10⁻⁹ | 3.8 × 10⁻¹⁰ | — |
| 30 | 3.6 × 10⁻¹² | 2.0 × 10⁻¹³ | 6.5 × 10⁻⁹ |

With the buffer covering the whole molecule, divide-and-conquer reproduces the full SCF to
**9 × 10⁻¹³ eV** and the full gradient to **6 × 10⁻⁹ eV/Bohr**. That limiting case is the
sharpest correctness test available, because any error in the projection, the common Fermi
level or the assembly shows up there as a real disagreement rather than a small one.

**Practical advice:** increase `buffer_radius` until the property you care about stops moving.
Energies converge faster than gradients, and gradients faster than second derivatives.

### The gradient

`divide_conquer_gradient` is the Hellmann–Feynman (fixed-density) gradient evaluated at the
divide-and-conquer density.

For a full SCF that expression is the *exact* derivative, because the energy is stationary with
respect to the density and the `(∂E/∂P)(dP/dR)` term vanishes identically. The
divide-and-conquer density is **not** stationary — it is assembled from separately diagonalized
blocks — so that term does not vanish, and this gradient is exact only in the limit where the
buffer covers the system. The residual is controlled by the same parameter as the energy error,
and the table above measures it.

The forces sum to zero to `8 × 10⁻¹⁵ eV/Bohr`, independent of buffer radius, because
translational invariance does not depend on the density being variational.

---

## Where the method does not apply

Divide-and-conquer rests on the density matrix decaying with distance, which is a property of
**gapped** systems. In a metal it decays algebraically instead, so the buffer would have to grow
with the system and both the accuracy and the linear scaling quietly stop being true.

The result therefore carries `homo_lumo_gap_ev` and, below `gap_warn_ev`, a
`small_gap_warning` explaining what to do about it. The ASE calculator turns that into a
`RuntimeWarning`. Returning a plausible-looking number in silence is the failure mode this is
here to prevent.

---

## Periodic systems

Divide-and-conquer works under periodic boundary conditions, at Γ with an **image buffer**: a
subsystem's buffer reaches across the cell boundary and pulls in periodic images of the cell's
own atoms. Buffer membership and the density truncation both use the **minimum-image** distance,
so an atom near the boundary sees its neighbours on the other side of it rather than at the far
end of the cell.

There is no separate "k-point divide-and-conquer" and there should not be. Divide-and-conquer is
a statement about real-space locality; k-point sampling is the reciprocal-space expression of
enlarging the real-space cell. Treating them as independent options would be a category error.

**The buffer saturates at half the shortest periodic length.** Beyond that every atom is within
reach of every other, the subsystem becomes the whole cell, and the method stops approximating —
it returns the full Γ SCF. Measured on a 6-molecule water chain (36.3 Bohr period):

| `buffer_radius` (Bohr) | difference from the full periodic SCF (eV) |
|---|---|
| 8 | 2.6 × 10⁻⁴ |
| 12 | 5.2 × 10⁻⁴ |
| 18 (= L/2) | 3.4 × 10⁻¹⁰ |
| 26 | 6.4 × 10⁻¹² |

So there is no point setting `buffer_radius` above `L/2`, and the convergence is **not
monotone**: widening the buffer admits new density blocks whose errors have either sign, and a
smaller buffer can sit closer by cancellation. The table shows that happening between 8 and 12.

**Size consistency.** A supercell does not cost exactly `n ×` the primitive cell, and that is
not divide-and-conquer's doing: a supercell at Γ *is* the primitive cell at several k points, so
the Γ treatment carries a size inconsistency of its own — 2.1 × 10⁻² eV on the 4-vs-8 molecule
water chain. Divide-and-conquer adds **2.4 × 10⁻⁴ eV** on top of that.

**Analytic stress** is available for a periodic run: `divide_conquer_stress`, matching a strain
finite difference of the divide-and-conquer energy to 3.6 × 10⁻⁸ eV/Bohr³ at a saturating
buffer. Components touching a non-periodic axis are exactly zero. Restricted only — an
open-shell stress would need the spin-resolved pair virial, and is refused rather than
approximated.

Both the gradient and the stress are fixed-density (Hellmann–Feynman) expressions. The
divide-and-conquer density is not variational, so the `(∂E/∂P)(∂P/∂·)` term does not vanish and
they are exact only in the buffer-covers-the-cell limit; `tests/dc_periodic.rs` measures the
residual rather than asserting a tolerance.

---

## Using it

### Rust

```rust
use am1_rs::{run_divide_conquer, divide_conquer_gradient, DcOptions, Am1Options, Am1Parameters};
use am1_rs::fermi::Filling;

let params = Am1Parameters::standard()?;
let result = run_divide_conquer(
    &molecule,
    &params,
    &Am1Options::default(),
    &DcOptions {
        core_size: 12,
        buffer_radius: 11.0,
        filling: Filling::Fermi { kt: 0.05 },
        ..DcOptions::default()
    },
)?;
println!("{} eV in {} subsystems", result.total_ev, result.subsystems);
if let Some(warning) = &result.small_gap_warning {
    eprintln!("{warning}");
}
let gradient = divide_conquer_gradient(&molecule, &params, &result)?;
```

### Python

```python
import am1_rs

r = am1_rs.divide_conquer(numbers, positions, buffer_radius=11.0, core_size=12)
print(r["energy_ev"], r["subsystems"], r["largest_subsystem_aos"])
print("diagonalization work per atom:", r["diagonalization_work"] / len(numbers))
```

### ASE

```python
from am1_rs.ase import AM1

atoms.calc = AM1(divide_conquer=True, core_size=12, buffer_radius=11.0)
atoms.get_potential_energy()
atoms.get_forces()
```

### Under a cell

Since 0.2.1 both Python surfaces accept a periodic structure — pass `cell` and `pbc` to
`native.divide_conquer`, or simply set `atoms.pbc` before `AM1(divide_conquer=True)`. The
subsystem buffers are then built from the image-aware pair list, so a buffer wraps through the
cell boundary rather than stopping at it.

`exchange_cutoff` matters here and does not for a molecule: at Γ the two-centre exchange integral
decays only as `1/R` while the density matrix does not decay at all, so the image sum needs the
taper. `tests/dc_periodic.rs` checks convergence to the full periodic SCF as the buffer grows.

```python
r = am1_rs.divide_conquer(numbers, positions, cell=cell, pbc=[True] * 3, buffer_radius=14.0)
```

---

## Memory

The subsystem solutions are small and transient. What used to dominate was the **DIIS history**,
and since 0.2.1 it does not.

The history stores the density's own **sparsity pattern** rather than a dense packed triangle.
That is exact, not an approximation: [`assemble_density`](#why-the-density-is-truncated-at-the-buffer-radius)
never writes a block whose Yang weight is zero, so every element outside the pattern is
identically zero at every iteration, and storing it was storing zeros. The pattern is read off
the subsystem weights once, so it is the density's actual sparsity rather than a distance
criterion that happens to agree with it.

Measured over 12 → 96 atoms:

```
DIIS history per entry: exponent 1.053   (the dense triangle it replaces: 1.993)
```

Both are asserted in `tests/divide_conquer.rs`, because the point is the *difference* between
them — a linear number on its own could be an accident of size. At 96 atoms the pattern is 2718
elements against 18528 dense, and the gap widens linearly with the system.

The same change cut the memory traffic: on a 1029-atom cluster `dc:diis` fell from 1.070 s to
0.419 s and the whole run from 5.11 s to 4.45 s, with identical energies and the same iteration
count. A memory budget (512 MB by default) still bounds the depth, and above it the history is
shortened rather than the calculation failing — but the budget is now far harder to reach.

What remains dense is the **global density matrix** itself, which grows as `O(N²)` even though it
has only `O(N)` non-zero blocks. `retained_density_blocks` is the measurement that says how much
is left to gain there: about 35 blocks per atom on a 3D cluster, against the `N` blocks per atom
a dense matrix stores.

## When it is worth it

Divide-and-conquer has real overhead: many small diagonalizations, a bisection search for the
chemical potential every iteration, and an assembly step. Below a few hundred atoms the full SCF
wins outright.

Run the measurement yourself with:

```bash
cargo test --release --test scaling -- --ignored --nocapture
```

On a quiet machine (16 cores), 3D water clusters:

| waters | atoms | AOs | full SCF | divide-and-conquer | speedup | ΔE per atom |
|---|---|---|---|---|---|---|
| 32 | 96 | 192 | 0.27 s | 0.40 s | 0.7× | 5.4 × 10⁻⁸ eV |
| 64 | 192 | 384 | 1.12 s | 0.93 s | 1.2× | 7.4 × 10⁻⁸ eV |
| 128 | 384 | 768 | 13.70 s | 2.19 s | 6.3× | 1.3 × 10⁻⁷ eV |
| 256 | 768 | 1536 | 16.11 s | 10.13 s | 1.6× | 1.6 × 10⁻⁷ eV |
| 512 | 1536 | 3072 | — | 33.48 s | — | — |

Crossover is around 200 atoms. Fitted wall-clock exponents on that run were 2.13 for the full SCF
and 1.62 for divide-and-conquer — but treat those as indicative only, and note that the speedup
column is far from smooth. Repeated runs put the 768-atom figure anywhere between 1.4× and 6.3×.
That spread is contention on a shared machine, not the algorithm, which is exactly why the scaling
claims above are asserted on **counters** and not on a stopwatch: one run of this benchmark
produced an apparent full-SCF exponent of 0.90, which no cubic diagonalization can do. Give the
machine to it — the timings are only as good as the quiet.

**These figures predate the DIIS work described below and are therefore pessimistic** for the
divide-and-conquer column. The table has not been re-run because it costs an hour of a quiet
machine to be worth anything; the directly measured replacement is 1029 atoms, three runs each:
**14.0–14.6 s before, 8.2–8.6 s after.**

### Where the time actually went, and why it was invisible

Phase timers (`AM1_TIMING=1`) said the largest phases at 1029 atoms were the Fock build and the
diagonalization. They also summed to 8.8 s of a 16.1 s run, and the missing 7.3 s was the largest
single item in the calculation. It sat *between* the timers — in the DIIS, which had none.

The cost was `extrapolate` rebuilding the whole B matrix on every iteration: all `n²` ordered
pairs, both triangles of a symmetric matrix, of a quantity that cannot change. `⟨rᵢ, rⱼ⟩` depends
only on two residuals already in the history, and a residual is never modified after it is pushed.
At 1029 atoms a packed residual is 16.9 MB, so a depth-8 history meant 64 dot products over
2 × 16.9 MB — 2.2 GB of memory traffic per SCF iteration, for numbers already computed.

Now the new residual's row is computed once on `push` and cached, which turns 64 dot products per
iteration into at most 8. Three smaller things went with it: `residual_dot` became a flat
`2·(packed dot) − (diagonal dot)` that vectorizes, instead of a nested row walk carrying an index;
`pack` copies contiguous row runs (`Matrix` is row-major, so the packed triangle of a row *is* a
contiguous run) instead of walking `nao²` through a 2D index; and the extrapolated density is
accumulated in packed form and expanded once, rather than scattered into `nao²` once per history
entry.

The iteration count is unchanged, which is the check that matters: a cached value that changed the
convergence path would be a cached value that was wrong.

The lesson generalizes past this module. An untimed region is not a small one — it is an unmeasured
one, and the profile's own arithmetic (phases summing to less than the total) is what says so.
Note also that the timing report sums **thread**-seconds: a phase on sixteen threads reports about
sixteen times its wall clock, so the best-parallelized phase looks like the worst one.

### A trap worth knowing about, if you build your own benchmark

The water clusters in `tests/scaling.rs` are spaced at **4.0 Å**, not the 2.76 Å of real ice, and
the generator asserts that no intermolecular contact is shorter than 1.8 Å.

Ice gets away with 2.76 Å because its molecules are *oriented* — every hydrogen points at a
neighbouring oxygen. A benchmark that gives them pseudo-random orientations does not have that,
so hydrogens frequently end up pointing at each other and the worst contact is roughly
`spacing − 2 × 0.96 Å`. An earlier version of this file used 3.1 Å and produced minimum contacts
of 1.22–1.35 Å, with over a hundred pairs inside 1.6 Å at the larger sizes.

The symptom was not a warning. It was a **cliff**: the SCF converged in 14 iterations at 192,
375 and 648 atoms, then failed outright at 1029. That looks exactly like a large-system
divide-and-conquer defect — charge sloshing, or a convergence threshold that does not scale — and
it was neither. It was steric clashes producing pathological electronic structure. At 4.0 Å the
worst contact is 2.1–2.2 Å at every size and the iteration count stays flat.

The density is then lower than liquid water (64 Å³ per molecule against 30), which is the right
trade for a benchmark: cost per atom barely depends on the density, and conditioning depends on
it entirely.
