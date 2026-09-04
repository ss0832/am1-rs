# Scope / feature matrix

This mirrors the **Scope** section of [`README.md`](../README.md); it is the canonical
capability list for `am1-rs`.

Each entry says what was measured, not what was intended. Where a capability is partial, the
partial part is named.

## Methods

| Capability | Status |
|---|---|
| **AM1** (Dewar *et al.* 1985) | ✅ 21 elements; see [methods.md](methods.md) |
| **RM1** (Rocha *et al.* 2006) | ✅ 10 elements, same code path |
| **SAM1** | ⛔ not implemented — a different integral engine, not a reparameterization |

## Molecular

| Capability | Status |
|---|---|
| **Component-level correctness** of the formulas themselves | ✅ `tests/theory_components.rs` — the monopole limit *recovers* `rho0` from the integral's asymptotics (1.9946 against 1.994724); each multipole channel's decay exponent is measured (`R^-0.998/-1.996/-2.997`); permutation symmetries, rotation covariance, overlap parity, `P² = 2P`, `[F,P] = 0`, `E = ½Tr[P(H+F)]`, rigid-motion invariance |
| RHF/UHF NDDO SCF with **SAD** guess + **A-DIIS→CDIIS** acceleration | ✅ DIIS history stored as packed triangles since 0.2.1 (`F` and `P` symmetric, `[F,P]` antisymmetric with a zero diagonal), which halves it exactly: peak memory on an 801-atom cluster fell 877 MB → 632 MB |
| Mulliken charges, dipole, HOMO/LUMO | ✅ Mulliken charges match MOPAC's own to 1.4e-5 e |
| Nuclear gradient — closed-form (dual-number AD), RHF & UHF, all elements | ✅ matches full-SCF finite difference to ~1e-7 eV/Bohr |
| Analytic Hessian (CPHF/UCPHF) + harmonic frequencies, RHF & UHF, all elements | ✅ |
| L-BFGS geometry optimization | ✅ |
| Open-shell UHF (radicals, odd-electron ions, `multiplicity > 1`) | ✅ |
| AM1-BCC partial charges + mol2 export | ✅ exact `BCCPARM.DAT` values; ring perception, Hückel aromaticity and delocalized groups — see below |
| **External electric field** — energy, analytic gradient, analytic Hessian | ✅ molecular only; gradient matches a full-SCF FD to 1.8e-6 eV/Bohr, Hessian to 8.1e-7 relative; `−∂E/∂F` reproduces the dipole to 3e-8 e·Bohr |
| **Orbital energies, coefficients, occupations** (both spin channels) | ✅ β channel was solved for and discarded before 0.2.1 |
| **Wavefunction output, Molden format** (`[Atoms]`, `[STO]`, `[MO]`) | ✅ the AM1 basis is genuinely Slater-type, so `[STO]` is exact — but the coefficients are in NDDO's *assumed* orthonormal basis; see [theory.md](theory.md) |
| **Atomic polar tensor** `∂μ_α/∂R_{a,β}` (3 × 3N, raw) | ✅ sum rule to 3e-15 e; matches a dipole FD to 7e-7 e and the mixed field/nuclear derivative to 1.2e-6 e |
| **Infrared intensities** (km/mol, per normal mode) + dense per-mode tensor | ✅ CO₂'s symmetric stretch dark at 1.6e-15 against 8.75 for the antisymmetric |
| **First-order orbital response** `U^j_{ai}` (CPHF coefficients) | ✅ returned from the Hessian solve rather than recomputed; lazy — never runs from an energy call |
| Normal modes + rigid-body overlap per mode | ✅ a linear molecule's five rigid-body modes are discovered, not assumed from `3N − 6` |
| External field under periodic boundary conditions | ✅ since 0.2.2 — **orthogonal to every lattice vector**: normal to a slab, transverse to a chain. `F·R` shifts by `F·T` under translation, so the perturbation is lattice-periodic exactly when `F·T = 0`, and 0.2.1's blanket refusal threw those cases out with the ill-defined one. `PbcOptions::electric_field`; the force matches a finite difference of the energy to 8.8e-8 eV/Bohr, and a molecule in a 60 Bohr cell reproduces the isolated-molecule path to 5.0e-6 eV. A component **along** a periodic direction is still an error, and the message now names it |
| Finite field **along** a periodic direction | ✅ since 0.2.2 (`pbc::run_finite_field`) — the Berry-phase electric enthalpy `E − Ω 𝓔·P` of Nunes and Gonze, not `F·R`. The field term couples neighbouring k points through `S⁻¹`, so the SCF takes a k-resolved additive operator and an outer loop refreshes it. Validated where the two formalisms compute the same object: on a **hydrogen-only** cell `α = Ω ∂P/∂𝓔` matches the CPHF polarizability to **0.03–0.47 %**, converging as `O(1/J²)` (1.06 → 0.47 → 0.26 % for J = 4, 6, 8). Zero field reproduces the plain SCF to 5e-12 eV and `pbc::berry`'s phase to 3.6e-14 turns. 3D, restricted, no smearing |
| On-site `s`–`p` moment in the Berry phase | ✅ since 0.2.2 — the link operator `Λ_{μν} = ⟨χ_μ\|e^{−ib·r}\|χ_ν⟩` was a diagonal `e^{−ib·τ_μ}`, which tracked the charge **centres** only. It is now the exact same-atom block: `e^{−ib·τ_a}` times `exp(−i b·D^a)`, whose generator is the same `dd` the CPHF dipole operator uses, exponentiated so `Λ` stays unitary. Born charges against CPHF went **0.207 → 1.2e-3 e** (at matched k-sampling, falling as `O(1/J²)`); the finite-field `α` on a water crystal went **12 % → 0.05 %**; and a planar cell's out-of-plane `α`, which is *entirely* this moment, went from **exactly 0** to 0.25527 against the CPHF's 0.25564 |

## Periodic boundary conditions

See [pbc.md](pbc.md) for the conventions and the limitations, which matter here.

| Capability | Status |
|---|---|
| 1D chains, 2D slabs, 3D crystals | ✅ dimensionality from `atoms.pbc` |
| Γ-point energy | ✅ |
| Monkhorst–Pack k-points, time-reversal folding | ✅ band folding verified to 3e-9 eV |
| Fermi–Dirac smearing, electronic entropy, T→0 extrapolation | ✅ |
| RHF **and** UHF at Γ and at k-points | ✅ forced-UHF matches RHF to 1e-7 eV |
| **Convergence acceleration** in the periodic SCF | ✅ since 0.2.2 — Pulay (DIIS) mixing on the real-space density, `PbcOptions::diis_history` (8 by default, `0` restores plain linear mixing). Through 0.2.2 there was **none**, while the molecular path had A-DIIS→CDIIS: hydrogen fluoride 140 → 22 iterations, water 140 → 28, a methane slab 130 → 23. Memory is `2 × depth` copies of the density |
| Convergence of a **symmetry-degenerate** cell | ✅ since 0.2.2. A 2D methane lattice — closed shell, 9 eV gap, threefold degenerate HOMO — could not reach `p_tol = 1e-10` at any iteration count, mesh, cutoff, mixing or smearing, and stalled at `dP ≈ 3e-8 = √ε`. The cause was in `hermitian_eigen`: the real embedding doubles every eigenvalue, and picking one complex vector per pair used a **single** Gram–Schmidt pass with a `1e-8` acceptance cut, so on a degenerate level the accepted vector was cancellation noise. Projecting twice and cutting at 0.1 takes the degenerate projector from 3e-8 to **2.2e-16** |
| The reported energy is the **variational** energy of the reported density | ✅ since 0.2.2. Two defects: the energy was contracted from the *mixed* density against the *unmixed* Fock, and it was evaluated at the mixed input, which is not idempotent — `E[P]` is stationary only on the idempotent manifold, so that leaves a **first-order** error the energy itself hides. Fixed by evaluating before the mix and by spending one further pass at the converged output density. Measured on a water dimer: a finite-differenced energy against the analytic gradient went 1.20e-6 → **3.04e-7** eV/Bohr |
| Analytic forces | ✅ NVE conserves in 1D/2D/3D |
| Analytic stress (1D/2D/3D) | ✅ matches strain finite difference to 5e-9 eV/Bohr³ |
| AM1 core–core Gaussian corrections under PBC, at Γ and k-points | ✅ |
| Net charge per cell, **3D** | ✅ energy converged to ~0.1 eV; 0.20 eV of cutoff drift vs 403 eV without Ewald |
| Net charge per cell, 1D / 2D | ⚠️ needs an explicit convention (`SheetConvention` / `AxisConvention`); refused by default, because the energy is not defined without one |
| **Ewald summation**, 3D monopole channel | ✅ Madelung to 1e-10, `α`-independent, surface term exact |
| `R⁻³` Klopman–Ohno tail beyond the pair list | ✅ since 0.2.2 (`klopman_ohno_tail`, default on) — the dropped translations are summed **explicitly** out to 3 cutoffs using the exact `γ_η − 1/R`, with a quintic handover to a continuum remainder whose integrand is per-dimensionality (`4πr²`/`2πr`/`2`). Only the 3D remainder is log-divergent, and it is cut at a stated reference length (`KLOPMAN_OHNO_REFERENCE = 1 Bohr`), exactly as the charged chain is. Measured on a +1 water cell over a 6.5× cutoff range: the residual went from **−0.118, −0.098, −0.097** eV per unit `ln r_c` (constant, i.e. the logarithm) to **0.000, −0.000, −0.000**, and the energy spread from 0.197 eV to 6e-5 eV. Forces move only by the density shift (0.08 % of scale); the stress matches its strain finite difference to 6.9e-9 eV/Bohr³. Carried through the **response** too, so `D(q=0)` still equals the `q = 0` Hessian (1.7e-8 relative) and its cutoff drift falls 3.3× |
| **Ewald 2D (Parry)** | ✅ Madelung to 1e-10, `α`-independent; in-plane 2×2 stress direct |
| **Ewald 1D** | ✅ real-space sum + analytic Hurwitz-zeta tail; cutoff- and `N`-independent |
| Periodic Γ **analytic Hessian** | ✅ matches FD to 5e-7 eV/Bohr², ASR residual 1e-14 |
| **k-point analytic Hessian** at `q = 0` | ✅ matches FD to 1.7e-7 eV/Bohr² on a polar chain with a mesh; ASR exactly 0 |
| **Open-shell (UHF) k-point response** | ✅ since 0.2.2 — the Hessian, the Born charges and the CPHF behind them. Two coupled channels, `G^σ(ΔP) = J(ΔP_tot) − K(ΔP_σ)`, so α reads β's response density through the Coulomb half. Forcing UHF on a closed shell reproduces the restricted force constants to **8.9e-16** eV/Bohr² on a 3-point mesh and the Born charges **exactly**; a genuine doublet chain matches a gradient finite difference to 5.2e-7 of 15.9. Refused through 0.2.1, and by pm6-rs and pm7-rs in the same place, so there was nothing to port. Extended to **DFPT at finite `q`** and to the **dielectric/polarizability** field response in the same release: forced UHF reproduces `D(q = 0.3)` on a 4-point mesh to 6.1e-9, gives the restricted `α` and `ε_∞` **exactly**, and a genuine doublet's `D(q = 0)` matches the open-shell Hessian to 1.3e-7 (8.2e-9 relative). The Berry phase and the finite field built on it stay restricted |
| Phonons `Φ(T) → D(q)`, band structure, ASR | ✅ exact folding identity verified |
| **Born effective charges** `Z*` | ✅ `Σ_a Z*_a = 0` to 1e-15; matches a dipole FD to 5.7e-9 |
| **Electronic dielectric tensor** `ε_∞` | ✅ **3D only**; clamped-ion field CPHF, origin-independent to 9.5e-15. Its **magnitude** is validated against the isolated molecule's finite-field polarizability — water in a box converges to 0.17 % at 12 Å. That check found a 27× unit error in 0.2.0's `α`, which every shape-only test had passed |
| `ε_∞` for a chain or a slab | ✅ since 0.2.2 — `pbc::dielectric_tensor_with_extent`, `am1_rs.dielectric_with_extent`, `AM1.get_dielectric_tensor_with_extent`. The missing ingredient is a **thickness** (slab) or **cross-section** (wire), and it is a *required argument*: a supercell says where the atoms are, not where the material stops. It is not a division — `α` here is the response to the **external** field, so the conversion carries the depolarization factor of the assumed body, `ε = 1 + 4πχ/(1 − 4πNχ)` with `N` = 0 in a slab's plane and along a wire, 1 along a slab normal, ½ transverse to a wire. Three-dimensional tin-foil summation removes the macroscopic depolarizing field, so `N = 0` there and the table closes on `dielectric_tensor` (reproduced to 1e-13). What does **not** depend on the choice is reported alongside: `(ε_∥ − 1)d = 4πα_∥/A` and `(1 − 1/ε_⊥)d = 4πα_⊥/A`, half the first being the Rytova–Keldysh screening length. Both are capacitor stacking — parallel and series — which is an independent derivation and is tested as one. Measured: the ratio against `dielectric_function`'s independent route is 2.0000000000, and `ε` does not move when only the vacuum changes |
| Depolarization is *in* `α`, not applied on top | ✅ measured rather than argued — on a 2D methane lattice tightened from 14 Å to 6.5 Å, `α_xx` rises 8.293 → 8.439 Bohr³ while `α_zz` falls 8.230 → 7.920. The sign asymmetry is what separates the external-field response from the internal one, and hence `1/(1 − 4πχ)` from `1 + 4πχ`; both laws are positive and monotonic, so nothing weaker distinguishes them |
| **LO–TO splitting** | ✅ **3D only**; the added term matches its closed form to 2e-15, raises eigenvalues and lowers none, and gives 1.04 cm⁻¹ of direction dependence on a polar molecular crystal. Exactly 0 where `Z* = 0` (measured 4e-14 e). Reachable from all three APIs since 0.2.1 (`lo_to_frequencies` / `get_lo_to_frequencies`); in 0.2.0 it existed only in Rust. Applies to the **supercell** route — do not combine it with `dfpt`, which already carries the long-range channel |
| LO–TO for a chain or a slab | ✅ **nothing to add, and measured rather than argued.** The long-range kernel diverges in every dimensionality, but the contribution to `D(q)` carries `q²` from charge conservation — so only 3D keeps a finite direction-dependent limit and is discontinuous at Γ. Measured, `\|D(q) − D(0)\|` at `q = 0.02, 0.01, 0.005`: a chain gives **3.6e-3 → 1.8e-3 → 9.0e-4**, a slab **4.6e-3 → 2.2e-3 → 1.1e-3** (both → 0), a crystal **1.071e-1 → 1.074e-1 → 1.075e-1** (flat). So `frequencies_with_lo_to` has no splitting to add below three dimensions, and the DFPT path already carries the non-analytic *approach* exactly. 0.2.0's "127 cm⁻¹ on a polar chain" was an artifact |
| **DFPT at arbitrary `q`**, arbitrary `k` | ✅ reproduces the `q = 0` Hessian to 4e-13 relative and commensurate supercell frozen phonons to 3e-4 (2-fold) and 1.4e-4 (3-fold). Arbitrary mesh or explicit k-list; DIIS-accelerated |
| **Berry-phase polarization** | ✅ since 0.2.2 (`pbc::berry`, `am1_rs.polarization`, `AM1.get_polarization`) — King-Smith–Vanderbilt strings, returned modulo the polarization quantum with a `difference` that reduces two values to a common branch. Lattice-translation invariant **exactly**, zero on a centrosymmetric cell to 8.5e-18, converged to 4.0e-8 by 32 points per string. The link operator is the full same-atom block `e^{−ib·τ_a} exp(−i b·D^a)`, not a diagonal phase, so it carries the on-site `s`–`p` moment: `Ω ∂P/∂τ` reproduces the CPHF **Born charges** to **1.2e-3 e** at matched k-sampling, falling as `O(1/J²)`, and to **7.5e-13 e** on a hydrogen-only cell. 3D, restricted |
| Long-range monopole correction **in the DFPT response** | ✅ in **every dimensionality** since 0.2.2, at every `q` (`LongRange::Auto`, the default). 3D: phased Ewald, cutoff-independent to 2e-16 at `q = ¼` where the truncated sum alone moves by 1.4e-1; at Γ it reproduces `pbc_hessian` to 1.7e-8 relative, ASR 1.2e-9. 2D: phased Parry over the full shifted in-plane set, α-independent to **8.9e-16**. 1D: phased direct sum with an Abel-transformed tail, image-count-independent to **7.2e-16** and matching a Cesàro-averaged direct lattice sum to **1.6e-12**. The element dropped is `k = 0`, not `G = 0`, so `D(q)` is the **full** dynamical matrix and must not be combined with `frequencies_with_lo_to`. See [pbc.md](pbc.md) |
| `q → 0` behaviour of that term | The **kernel** diverges in all three (`4π/Vq²`, `2π/A|q|`, `−(2/L)ln|q|` — measured); the contribution to `D(q)` carries `q²` from charge conservation, so only **3D** keeps a finite direction-dependent limit. 2D goes as `O(|q|)`, 1D as `q² ln(1/q)`: both **continuous** at Γ with a non-analytic approach. An earlier draft recorded "2D is discontinuous", which conflated the two levels |

## Divide-and-conquer

See [divide-conquer.md](divide-conquer.md).

| Capability | Status |
|---|---|
| Molecular, RHF **and** UHF | ✅ open-shell matches full UHF to 2e-12 eV |
| Yang partition sum rule | ✅ exact (0.0 deviation) |
| Common chemical potential, fractional filling | ✅ one level per spin channel |
| Non-neutral systems | ✅ charge conserved to 1e-8 e |
| Energy and gradient | ✅ converge to the full SCF with buffer radius (not monotonically — see the docs) |
| **Under periodic boundary conditions** | ✅ Γ with a minimum-image buffer; exact at `buffer ≥ L/2` (3.4e-10 eV) |
| **Analytic stress** | ✅ matches a strain FD to 3.6e-8 eV/Bohr³; restricted only |
| **Linear-scaling diagonalization** | ✅ exponent 1.15; subsystem size saturates |
| **Linear-scaling exchange** | ✅ exponent 1.06, exactly (truncated density) |
| **Linear-scaling DIIS memory** | ✅ exponent 1.05, against 1.99 for the dense triangle it replaces. The dominant memory term of a large run; `dc:diis` fell from 1.07 s to 0.42 s on 1029 atoms |
| Linear-scaling **Coulomb** | ✅ since 0.2.2 — `FarField::tree(theta)` is a Barnes–Hut evaluation of the far field, **opt-in**. Fitted exponent **1.65 against 2.13** for the direct sum over 24→1029 atoms, and 8× fewer partner evaluations at 1029. Each accepted cluster becomes *two* pseudo-atoms, positive and negative parts at their own centroids, so the dipole survives — a monopole-only tree is useless here because the clusters are neutral molecules. Every consumer evaluates the same pair kernel against a shorter list, so the potential, gradient and virial cannot drift apart. At `theta = 0` it visits **exactly** the direct sum's pairs and agrees to 5e-15. Opt-in because an acceptance angle makes the energy discontinuous where an atom crosses it |
| Open-shell analytic stress | ✅ since 0.2.2 — the spin-resolved pair virial. Forced UHF reproduces the restricted stress to 2.5e-14, and a **methyl-radical** chain matches a strain finite difference to 2.4e-9 eV/Bohr³, measured across three step sizes so the SCF-noise and truncation limbs are both visible. The fixture was a triplet water chain until 0.2.2, where its 1.9e-8 was a coincidence: that system's energy jumps in quanta of 1.7e-5 eV as a level crosses the Fermi energy, and `E(+h)`/`E(-h)` happened to land on the same branch |

## Interfaces

| Capability | Status |
|---|---|
| Rust API | ✅ |
| Python-native API | ✅ `method=` everywhere; GIL released around solvers |
| ASE `Calculator` | ✅ energy, forces, stress, charges, dipole; reads `atoms.pbc`/`atoms.cell` |
| **Native/ASE feature parity** | ✅ every native capability has an ASE method in ASE units — frequencies, IR, orbitals, orbital response, Molden, AM1-BCC, optimize, phonons, DFPT, LO–TO, Born charges, `ε_∞`, periodic Hessian. `tests/test_new_api_0_2_1.py` **enumerates** `am1_rs.native`'s public functions and fails on any without a declared ASE counterpart; the hand-written list it replaced is what let `lo_to_frequencies` sit in Rust alone |
| Divide-and-conquer **under a cell** from Python/ASE | ✅ since 0.2.1 — `native.divide_conquer(cell=…, pbc=…)` and `AM1(divide_conquer=True)` on a periodic structure. Buffers are built from the image-aware pair list, so they wrap through the cell boundary |
| **Lazy evaluation of expensive properties** | ✅ Hessians, IR spectra, orbital responses, phonons, DFPT and LO–TO are explicit `get_*()` calls and never run from `calculate()`. Since 0.2.2 they memoize into their own store keyed on the geometry's **bytes**, the resolved state (including `atoms.info` overrides) and their own arguments. **Through 0.2.1 they cached into `results` and were not invalidated at all**: `results` is cleared by `Calculator.get_property`, which no lazy method goes through, so moving the atoms and asking again returned the previous geometry's answer. This row claimed the opposite. `tests/test_lazy_cache.py` asserts every method in the family against a displacement |
| **One CPHF solve for the whole vibrational group** | ✅ since 0.2.2 — `get_hessian`, `get_frequencies`, `get_ir_spectrum`, `get_dipole_derivatives` and `get_orbital_response` are contractions of one response and share it through `native.vibrations`. Previously five questions cost five full analytic-Hessian solves |
| External field through the APIs | ✅ atomic units (Hartree per e·Bohr) natively, **V/Å** in ASE, converted with the crate's own `a0`; `atoms.info["field"]` overrides per structure and invalidates the cache |
| CLI | ✅ `am1_rs_cli` from the crate; `am1-rs` on `PATH` after `pip install`, and `python -m am1_rs`. The two are diffed against each other per mode in `tests/test_cli.py` |

## Elements

Full published AM1 main-group set: H, Be, B, C, N, O, F, Al, Si, P, S, Cl (n ≤ 3, exact
analytic overlap) and Zn, Ge, As, Se, Br, Sb, Te, I, Hg (n ≥ 4, general numerical overlap).

Heavy-element accuracy is limited by the Gauss–Legendre quadrature in the numerical overlap;
the measured agreement against the closed form is ~5e-4, so gradients and Hessians involving
those elements inherit that floor.

## AM1-BCC parameter coverage

The correction values are the exact antechamber `BCCPARM.DAT`. Which of the 405 entries the
typing can reach was measured rather than assumed, and the measurement matters:

**All nine bond types are emitted since 0.2.2.**

| bond type | entries | what selects it |
|---|---|---|
| 1 single, 2 double, 3 triple | 287 | the perceived bond order |
| 6 conjugated | 6 | a delocalized bond whose centre is nitrogen — nitro, N-oxide |
| 7 aromatic single | 25 | an aromatic bond formally single in the Kekulé structure |
| 8 aromatic double | 15 | …and one formally double |
| 9 delocalized | 21 | carboxylate, phosphate, sulfonate |
| 10 aromatic | 25 | an aromatic bond with **no resolved** single/double character |
| 11 same type | 26 | both ends the same atom type, where no other code is tabulated |

7, 8 and 10 are separated by the Kekulé assignment: a six-membered aromatic has two equivalent
Kekulé structures, so no bond of it is "the" double bond and they take 10; a five-membered
heteroaromatic has one, so 7 and 8 apply.

Reaching 8 and 10 **changes no charge** — they are byte-identical to type 7 on every shared pair,
and every type-11 value is exactly 0.0. Both are asserted against the parameter file in
`tests/bcc_bond_types.rs`.

Type 11 is not cosmetic: ten of its atom types, **hydrogen and every halogen among them**, have no
single-bond entry, so through 0.2.1 an H–H or Cl–Cl bond found no parameter under any emitted code
and reported itself as left at raw Mulliken charges. The charges were right; the warning was not.

Perception now uses smallest-ring-through-each-bond (so naphthalene gives two 6-rings, not the
10-membered perimeter), ring-size-aware aromaticity with a Hückel 4n+2 π count and a planarity
test (so cycloheptatriene and cyclooctatetraene are correctly not aromatic, and pyrrole, furan
and thiophene are), a bond-order reference table extended to P, S and the mixed pairs (so C=S,
P=O and S=O are no longer typed as single bonds), and explicit warnings.

**The typing reads the definition file (0.2.2).** `src/bcc/atomtype.rs` parses and evaluates
`ATOMTYPE_BCC.DEF` — wildatoms, atom properties, nested environment patterns and the file's own
first-match-wins order — rather than transcribing it into `match` arms. The transcription had
drifted in ten places, all silent: a nitro nitrogen was off by **0.6745 e**, a pyrrole α carbon by
0.315 e, and every ketone, ester and amide carbonyl oxygen was mistyped. See the CHANGELOG.

The AR1..AR5 sub-classification is **not** a gap and was removed from this list on measurement:
every rule in the file asks for the union `[AR1.AR2]` and none asks for either class alone. The
indole rule (a five-membered ring sharing an edge with a six-membered aromatic ring is not
aromatic) *is* needed, and is implemented, as is a Kekulé assignment — `[2sb]` versus `[sb,db]`
separates nitrogen types 21 and 24, and the parameter file's `17–24` aromatic entry is what says
pyridine's nitrogen must be 24.

**Remaining gap.** The *perception* underneath is geometry-based — covalent radii, ring search, a
Hückel π count and a bond-length table — where antechamber uses penalty-based bond-order
assignment, so a molecule whose bond orders it would assign differently can still be typed
differently. And two `23` rules carry a trailing `a1:a2:any` chain constraint, read as its name
says (no restriction between the labelled atoms); that is a reading of the syntax rather than a
transcription of it, and it is recorded in the module. Anything the perception cannot do
confidently is reported in `BccResult::warnings` rather than guessed, so a molecule that returns no
warnings is one the rules covered.

## Units at the boundary

- Internal: eV energies, Bohr distances (MOPAC7 model constants `ev = 27.21`, `a0 = 0.529167`).
- Rust API: eV/Bohr internally; Hartree/Bohr at the molecular Python boundary.
- Python-native API: Hartree, Bohr, ΔHf in kcal/mol for molecular functions; **eV/Å** for
  `pbc_point` and `divide_conquer`, whose consumer is ASE.
- ASE `Calculator`: eV, Å, eV/Å³ throughout.
