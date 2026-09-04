# Changelog

## 0.2.2

### Fixed

- **A periodic SCF could not converge a symmetry-degenerate cell, and three separate defects were
  in the way.** A two-dimensional lattice of methane — closed shell, 9 eV gap, no hydrogen bonds,
  no magnetism, about as easy as a periodic calculation gets — could not reach `p_tol = 1e-10` at
  any iteration count, mesh, cutoff, mixing fraction or smearing width. The isolated molecule
  converged in 28 iterations, so nothing in the parameterization or the integrals was at fault.
  What it does have is a **threefold degenerate HOMO**.

  1. **`pbc::complex::hermitian_eigen` lost `√ε` on a degenerate level.** It solves the complex
     Hermitian problem through a `2n × 2n` real embedding, in which every eigenvalue appears twice
     — `(x, y)` and `(−y, x)` are the same complex vector, one times `i`. Picking one per pair used
     a **single** classical Gram–Schmidt pass and accepted any residual above `1e-8`. On a
     degenerate level the duplicate genuinely lies in the span already, so what survived the
     subtraction was cancellation noise, renormalized to unit length and accepted as a physical
     eigenvector. The occupied projector built from it carried about **3e-8**, which is exactly the
     floor the SCF stalled at, and `3e-8` is `√ε` — the signature of the loss. Fixed by projecting
     twice ("twice is enough") and cutting at `0.1`: a duplicate's residual is `O(ε)` and a
     genuinely new direction inside a `k`-fold block has residual² of at least `1 − 1/k ≥ ½` for
     some remaining column, so the two are seven orders apart and the old threshold sat between
     them. The degenerate projector goes from 3e-8 to **2.2e-16** against its closed form.

  2. **The periodic SCF had no convergence acceleration at all** — plain linear mixing at 0.3,
     while the molecular path had A-DIIS→CDIIS. It did not show because the covered systems were
     stiff: a hydrogen fluoride slab reached `1e-10` in about 140 passes, which reads as a slow
     system rather than a missing feature. The near-constant 139–141 across unrelated systems was
     the tell. Now Pulay (DIIS) mixing on the real-space density, `PbcOptions::diis_history`
     (8 by default; `0` restores the old behaviour). Hydrogen fluoride 140 → **22**, water
     140 → **28**, a methane slab 130 → **23**. Memory is `2 × depth` copies of the density, which
     for a large cell is the dominant allocation of the run.

  3. **The energy was not the energy of any density.** It was contracted from the *mixed* density
     against the *unmixed* Fock — `½Tr[P_mixed(H + F(P_in))]` — and inconsistent with the
     `total_origin` the same iteration built from `P_in`. At the fixed point all of them agree, so
     the converged number was right; what it corrupted was `de`, half the convergence test, which
     measured the mixer as much as the iteration. Moved above the mix.

     And even then, `E[P] = ½Tr[P(H + F(P))]` is stationary **only on the idempotent manifold**,
     which a mixed input is not on — a Pulay step is a signed combination of past densities and can
     sit further off idempotency than its distance to the fixed point suggests. Evaluating there
     leaves a *first-order* error that the energy hides and anything differenced does not. Once the
     tolerances are met the solver now spends one further pass at the converged **output** density,
     which is idempotent, so the returned energy is the variational energy of the returned density.
     Measured on a water dimer: a central-differenced total energy against the analytic gradient
     went **1.20e-6 → 3.04e-7** eV/Bohr, now slightly better than plain mixing rather than four
     times worse.

  The three share a lesson about ordering. A defect that only costs iterations hides a defect that
  costs correctness, because both present as "it needs more iterations" — and the unconverged runs
  reported energies differing by up to **0.6 eV** between k-meshes, every one of them plausible.
  `tests/pbc_scf_convergence.rs` pins all three, and `AM1_SCF_TRACE=1` prints `dE` and `dP` per
  iteration, which is the only way to tell a slow contraction from a stall from the outside.

- **A performance test asserted on wall clock and failed under load.**
  `factoring_project_ov_lowers_its_scaling_exponent` claimed a drop from `O(nao⁴)` to `O(nao³)` and
  checked it by requiring the measured *speedup ratio* to grow with `nao` — on the reasoning that a
  ratio divides the machine's load out. It does not: both halves are wall clock, and the largest
  case carries the most memory traffic and so suffers most. Idle it measured 54/108/357, 46/137/385
  and 48/94/376 across three runs; with a build running alongside it measured 61/258/**46** and
  failed. An exponent is a count of operations, not a duration, so the count is now what is
  asserted — 10.7× → 21.3× → 42.7×, exactly doubling per doubling of `nao`, which *is* the missing
  power — and the timings are printed with a load-proof 5× floor beneath them. Renamed to
  `factoring_project_ov_lowers_its_operation_count`, which is what it now measures. Same rule
  `DcResult::diagonalization_work` already followed.

- **AM1-BCC atom typing was wrong in ten places, and silently.** The typing was a hand
  transcription of antechamber's `ATOMTYPE_BCC.DEF` into Rust `match` arms, and it had drifted from
  the file. Every case is silent, because a parameter exists for the wrong type too — no warning
  fires, a plausible number comes back.

  The errors are counted **per atom**, not per bond, and that distinction is the point: a
  correction is looked up on the *pair* of types, so changing one atom's type changes **every bond
  at that atom**.

  | group | file says | 0.2.1 said | bonds affected | error on that atom |
  |---|---|---|---|---|
  | nitro N | 23 | 21 | N–O ×2, C–N | **0.6745 e** |
  | 3-coordinate P with a double bond | 42 | 41 | P=O, P–C ×2 | 0.44 e |
  | pyrrole-type α carbon | 16 | 17 | C–N, C–C, C–H | 0.315 e |
  | ester / carboxylic acid carbonyl O | 32 | 33 | C=O | 0.086 e |
  | ketone / aldehyde carbonyl O | 31 | 32 | C=O | 0.050 e |
  | amide carbonyl O | 31 | 33 | C=O | 0.036 e |

  plus four nitrogen rules with no counterpart at all (a three-coordinate N with a double bond, a
  two-coordinate amide N, a two-coordinate N with two single bonds, and a four-coordinate aromatic
  N, which the file types 21 because `21 * 7 4 &` precedes the aromatic rule).

  The cause was not a set of typos. It was that the *conditions* had been dropped: both `33` rules
  carry `[RG]` and apply to lactones and lactams only, but the code looked at no ring and instead
  counted the oxygens on the carbon; the `32` rule asks whether the carbon bears a **two-connected**
  oxygen; the `17` rule requires an aromatic **`N2`**, and pyrrole's nitrogen has three connections;
  and there is a nitro rule, `(O1,O1)`, that the transcription simply did not contain.

  **The fix reads the file.** `src/bcc/atomtype.rs` parses `ATOMTYPE_BCC.DEF` and evaluates it —
  `WILDATOM` expansion, atom properties, nested chemical-environment patterns with their `'`
  bond-to-predecessor suffix, and the file's own top-to-bottom first-match-wins order, which its
  closing note says is crucial and which is exactly what the four missing nitrogen rules turned on.
  The rules are now antechamber's rather than a reading of them.

  **This changes published charges** for every molecule with a carbonyl, a nitro group or a pyrrole
  ring. The 0.2.1 values for those were wrong.

  Two supporting changes were needed and are worth naming separately:

  - **A Kekulé assignment** (`Topology::kekule_double`). `[2sb]` and `[sb,db]` separate nitrogen
    types 21 and 24, so an aromatic ring bond has to be formally single *or* double, and which one
    is not a local property. The parameter file settles the intent: it carries a `17–24` entry at
    bond type 7, and since the `17` rule itself requires an aromatic two-connected nitrogen
    neighbour, that pair is pyridine and nothing else. Found by perfect matching over the atoms
    contributing one π electron.
  - **The indole rule** from the definition file's own closing note: a five-membered ring sharing
    an edge with a six-membered aromatic ring is not aromatic for AM1-BCC.

  What is *not* needed, on measurement: the AR1..AR5 sub-classification, which earlier notes listed
  as a gap. Every rule in this file asks for the union `[AR1.AR2]` and none asks for either class
  alone, so splitting them would be machinery with no consumer.

- **An H–H or halogen–halogen bond reported itself as uncorrected.** `BCCPARM.DAT` has a bond type
  for a pair with the *same atom type on both ends* — code 11, whose 26 entries are all `X–X` and
  all exactly 0.0 — and ten of those types, hydrogen and every halogen among them, have **no
  single-bond entry at all**. Emitting only codes 1, 2, 3, 6, 7 and 9 therefore left H₂, F₂, Cl₂,
  Br₂ and I₂ with no parameter for their one bond, and `BccResult::warnings` said the bond was left
  at its raw Mulliken charges. The charges were in fact right — the correction is zero — but the
  warning is the thing callers are told to check, and "a molecule that returns no warnings is one
  the rules covered" has to mean something.

  All nine bond types are emitted now. Types 7 (aromatic single), 8 (aromatic double) and 10
  (aromatic, no resolved order) are separated by the Kekulé structure the atom typing already
  needs: a six-membered aromatic has two equivalent Kekulé structures, so none of its bonds is
  *the* double bond and they take 10, while a five-membered heteroaromatic has a unique one and
  takes 7 and 8. Type 11 is a fallback, not an override — where the ordinary code is tabulated it
  wins, which `the_same_type_code_does_not_pre_empt_a_tabulated_one` pins.

  **No charge moves**: 8 and 10 are byte-identical to 7 on every shared pair and every type-11
  value is zero, both asserted against the parameter file rather than recited.

- **The antechamber licence was not in the repository, the crate, or the wheel.** `BCCPARM.DAT`
  (GPL-3) is `include_str!`-ed into every binary, and `ATOMTYPE_BCC.DEF` now is too, while MOPAC's
  and PySEQM's licences were retained and antechamber's was not. The CI job that checks
  "third-party licences are inside the wheel" did not catch it because it asserted only that the
  list was non-empty — two of three passed that. It now compares against the number of
  `third_party/` subdirectories, so a fourth bundled work cannot repeat this, and the sdist check
  additionally requires `ATOMTYPE_BCC.DEF`, which the build now needs. `third_party/antechamber/`
  gains the GPL-3 text and a `README.md` recording which upstream file each copy is and where it
  came from. `THIRD_PARTY_NOTICES.md` §4 and §6 are rewritten, including what "or later" does and
  does not say for this material.

- **The lazy `get_*()` cache was never invalidated by a geometry change.** `docs/scope.md` claimed
  these methods "cache into `results` and are invalidated by `check_state`". The first half was
  true and the second was not: `results` is cleared by `Calculator.get_property`, which calls
  `check_state` and then `reset()` — and none of the lazy methods go through `get_property`. So

  ```python
  f1 = atoms.calc.get_frequencies(atoms)
  atoms.positions += 0.5
  f2 = atoms.calc.get_frequencies(atoms)   # returned f1
  ```

  with nothing to announce it. The existing test could not catch it because it never moved the
  geometry between calls. They now memoize into their own store keyed on the geometry's **bytes**,
  the resolved state (including the `atoms.info` overrides ASE's own comparison cannot see) and
  their own arguments; `tests/test_lazy_cache.py` asserts every method in the family against a
  displacement, and asserts that it is still a cache.

- **`AM1.optimize(apply=True)` cleared `results` but left `self.atoms`** holding the
  pre-optimization geometry. It calls `reset()` now.

- **The charged-cell warning described a version of the code that no longer existed.** It said no
  compensating background is applied "because Ewald summation is not implemented", that "THE TOTAL
  ENERGY IS NOT CONVERGED", and quoted a −331 eV to +72 eV swing across real-space cutoffs. Ewald
  has been implemented since 0.2.0, in all three dimensionalities, and is on by default — those
  were the pre-Ewald numbers. It was telling users their converged 3D energies were meaningless,
  through every surface (a `RuntimeWarning` in ASE and a line in both CLIs).

  Measured rather than reasoned about (`tests/charged_cell_warning.rs`): a +1 water cell in an 8 Å
  cube across a 6.5× range of cutoff moves **0.197 eV with Ewald and 403.4 eV without**. The
  warning is now per-dimensionality — in 3D the tin-foil sum defines the energy and the residual is
  the `R⁻³` tail; in 1D/2D the monopole sum is applied but the neutralizing background's placement
  is a *convention* (`SheetConvention` / `AxisConvention`) that nothing in the SCF path consults,
  so the absolute energy is not defined there. Both texts are ASCII, for the cp932/C-locale reason
  0.2.1 recorded.

- **The phonon spectrum was not reproducible.** `ForceConstants::blocks` is a `HashMap`, and three
  float sums iterated it directly: the Bloch sum `D(q) = Σ_T Φ(T) e^{iq·T}`, the acoustic-sum-rule
  residual, and the acoustic-sum-rule *correction*. Rust seeds each `HashMap` instance from a
  thread-local counter, so two maps built from the same insertions in the same process iterate in
  different orders — and floating-point addition is not associative.

  Measured: five identical `lo_to_frequencies` calls in one process, on a water crystal in a 4.5 Å
  cube, agreed on four and differed by **1798 cm⁻¹** on the fifth — one O–H stretch collapsing into
  a near-zero mode. The periodic SCF underneath was bit-identical every time (same energy to the
  last digit, same 115 iterations), which is what located it in the phonon assembly rather than the
  electronic structure.

  The correction is the one that mattered: it is *subtracted* from the on-site block, so an
  order-dependent value there changes `Φ` itself and every `D(q)` built from it afterwards. The
  other two stayed within last bits.

  All three now iterate a translation-sorted view. `tests/phonon_determinism.rs` asserts
  **bit-identical** repeats rather than a tolerance — the claim is that the same input gives the
  same output, which is a property of the code, not of any crystal's conditioning. Found because
  `test_lo_to_frequencies_splits_and_matches_across_surfaces` failed intermittently and only in a
  full-file run; the ASE and native paths it compares call the same function, so the disagreement
  could not have been between them.

- **`LongRangeMonopole::for_molecule`'s documentation contradicted its code**, saying the
  correction applies "only to a fully three-dimensional cell" while the code accepts any
  `n_periodic() >= 1` and dispatches to the 1D and 2D kernels. Only the *phased* (DFPT) path is 3D
  only. Corrected in the docstring and in `docs/scope.md`.

### Added

- **`ε_∞` for a slab or a chain**, once the caller says how thick the material is:
  `pbc::dielectric_tensor_with_extent`, `am1_rs.dielectric_with_extent`,
  `AM1.get_dielectric_tensor_with_extent`. `ExtentConvention::SlabThickness` (Bohr) or
  `WireCrossSection` (Bohr²) is **required and never defaulted** — a supercell says where the atoms
  are, not where the material stops, and every choice changes `ε`. Same rule as `chain_radius` and
  `AxisConvention`.

  **It is not a division.** The `α` this crate computes is the response to the *external* field —
  the induced charges interact through the same Coulomb operator the SCF uses, so for a slab
  polarized along its normal the depolarizing field is already inside `α`. The conversion therefore
  carries the depolarization factor of the assumed body,
  `ε = 1 + 4πχ/(1 − 4πNχ)` with `χ = α/(measure · extent)`: `N` = 0 in a slab's plane and along a
  wire's axis, 1 along a slab normal, ½ transverse to a wire's circular section. Three-dimensional
  tin-foil summation removes the macroscopic depolarizing field, so `N = 0` there — which means the
  same arithmetic reproduces `dielectric_tensor` rather than sitting beside it, measured at 1e-13.

  Getting that factor backwards returns a plausible number: both laws are positive and monotonic in
  `α`. What separates them is a **sign asymmetry** in the response itself, and
  `tests/pbc_dielectric_extent.rs` measures it — tightening a 2D methane lattice from 14 Å to 6.5 Å
  moves `α_xx` up (8.293 → 8.439 Bohr³) and `α_zz` down (8.230 → 7.920), which is what a sheet of
  induced dipoles does and what an internal-field response would not show at all.

  The thickness is a choice, so `ε` is a choice; two combinations are not, and are returned
  alongside it: `(ε_∥ − 1)d = 4πα_∥/A` and `(1 − 1/ε_⊥)d = 4πα_⊥/A`, half the first being the
  Rytova–Keldysh screening length. Read the other way they are capacitor stacking — parallel and
  series — so the two formulas are forced rather than chosen once the thickness is named, which is
  a second derivation and is tested as one. The first must also equal what `dielectric_function`
  reaches through a reciprocal-space Coulomb kernel: measured ratio **2.0000000000**. And `ε` does
  not move when only the vacuum padding changes, which is precisely what 0.2.0 got wrong.

  Eleven component tests sit next to the arithmetic in `src/pbc/extent.rs`, where they can use a
  synthetic `α`, because the conversion is the model-dependent step and deserves to be checked
  without an SCF in the way.

- **`native.vibrations`** — the Hessian, frequencies, normal modes, atomic polar tensor,
  intensities and orbital response from **one** SCF and one CPHF solve. `hessian`, `frequencies`,
  `ir_spectrum`, `dipole_derivatives` and `orbital_response` each ran the whole analytic-Hessian
  solve and kept a different contraction of it, so a caller wanting a spectrum *and* the Hessian it
  came from — the ordinary case — paid for the CPHF once per question. The ASE calculator routes
  all five through it, and `tests/test_lazy_cache.py` asserts that the family leaves exactly one
  entry in the cache. The five original functions are unchanged.

- **A Barnes–Hut far field**, so the NDDO Coulomb is no longer `O(N²)`. `docs/scope.md` recorded
  "linear-scaling Coulomb ⛔ — stays `O(N²)` by construction", which was true: `FarField` keeps the
  interaction in full and simplifies only its *shape*, so the prefactor fell a hundredfold and the
  exponent did not move. `FarField::tree(theta)` moves it: fitted **1.65 against 2.13** over 24 to
  1029 atoms, with 131 515 partner evaluations against 1 043 490 at the top — an 8× reduction that
  grows with size.

  Each accepted cluster becomes **two** pseudo-atoms, the positive and negative charge at their own
  centroids. One would be a monopole expansion, and a monopole expansion is worthless here: the
  clusters are made of neutral molecules, so the net charge is near zero and the interaction is
  dipolar. The first draft did exactly that and the error against the direct sum was 64 % and did
  not shrink with the acceptance angle — there was no monopole for the angle to resolve. Splitting
  by sign carries the dipole while keeping the property that makes the design safe: every consumer
  evaluates the *ordinary pair kernel* against a shorter list, so the potential, the gradient and
  the virial cannot drift apart the way three separately truncated expansions would.

  At `theta = 0` the tree visits **exactly** the pairs the direct sum does — asserted as an
  equality on the count — and agrees to 5.3e-15, the residual being summation order. In between the
  error is monotone in `theta`: 2.7 % at 0.8, 0.3 % at 0.05.

  **Opt-in**, because an acceptance angle makes the energy a discontinuous function of the geometry
  where an atom crosses the boundary. The jump is of the order of the truncation error, but it is a
  jump, and molecular dynamics should either leave it off or accept it knowingly.

- **Berry-phase polarization** (`pbc::berry`), the modern theory of polarization. Listed as ⛔
  through 0.2.1 — "`ε_∞` is the clamped-ion dipole response, not a Berry phase" — which was accurate
  and was a gap: the dipole of a periodic cell is not a property of the crystal, so the crate had
  polarization's *second* derivative and not polarization.

  `P_el = (e/Ω) Σ_α a_α · Im ln Π_j det S(k_j, k_{j+1})/2π` over strings of k points, with the
  occupied-manifold overlap in this basis being `S_mn = Σ_μ c*_{μm}(k) e^{−ib·τ_μ} c_{μn}(k+b)` —
  the `e^{−ib·τ_μ}` being the same "an orbital sits at its atom" approximation the dipole operator
  already makes. Returned modulo the polarization quantum, with `BerryPolarization::difference`
  reducing two values to a common branch, because subtracting absolute polarizations is the
  standard way to be wrong by exactly one quantum.

  **The sign was derived rather than looked up**, sources differing on the convention: for a single
  electron whose only orbital sits at `τ`, every link contributes `e^{−ib·τ}` and the string product
  is `e^{−iB·τ}`, so `φ = −τ_α/a_α` in turns; that electron is a charge `−1` at `τ`, which fixes the
  prefactor to `+e/Ω`. The first draft had it negative and the acoustic sum rule found it at once —
  the Born charges summed to `+2 n_elec` instead of zero.

  Validated four ways, none of which compares `P` to a number (an absolute polarization is not a
  physical prediction): translating the cell by a lattice vector leaves it unchanged **exactly**, a
  centrosymmetric cell gives zero to 8.5e-18, the phase converges to 4.0e-8 by 32 points per string,
  and — the sharp one — `Ω ∂P/∂τ_A` reproduces the **Born effective charges** the CPHF dipole
  response produces, two formalisms sharing only the SCF.

  That last comparison differs by 0.207 e on hydrogen fluoride, and the reason is *measured* rather
  than asserted: the dipole operator additionally carries the on-site `s`–`p` hybridization moment
  `dd`, which this basis's Berry phase does not. On a **hydrogen-only** cell, where hydrogen has no
  `p` shell and the `dd` term is structurally unreachable, the two routes agree to **7.5e-13 e**.

- **The long-range monopole term in the DFPT response, in 1D and 2D.** 0.2.1 shipped it for 3D
  cells and named its absence on a chain or a slab as the release's one unfinished item:
  `LongRange::Require` was an error there, and `Auto` silently dropped the channel. Both
  dimensionalities now have a phased kernel, each the `q`-shifted form of the machinery its
  unphased sum already used:

  - **2D** — Parry's slab sum over the **full shifted in-plane set** with prefactor `π/(A|k|)`,
    `k = G − q`. There is no ±G folding to exploit once `q ≠ 0`, and at `q = 0` the full set with
    `π` reproduces the folded half set with `2π` — which is what makes a wrong factor of two here
    visible to the splitting-parameter test and almost nothing else.
  - **1D** — the chain's direct summation, phased image by image, with the truncated tail summed by
    **repeated Abel transformation**. Truncating the oscillating sum directly is only `O(1/N)`:
    Dirichlet converges it because the partial sums of `e^{iθn}` are bounded by `1/(2|sin(θ/2)|)`,
    but that bound multiplies the first neglected term and blows up as `q → 0`. Summation by parts
    trades it for a series over exact forward differences of the kernel, truncated at its smallest
    term.

  Both delegate to their unphased counterpart where `q` is a reciprocal lattice vector — not where
  `q = 0`, which is the silent version of the same test — so the neutralizing background, the sheet
  term and the chain's line charge each keep exactly one derivation.

  Validated against a **direct lattice sum**, Cesàro-averaged to damp the conditionally convergent
  boundary term (without which the oracle is less accurate than the thing it checks): 1D agrees to
  **1.6e-12**, 2D to 1.2e-5…9.7e-5 where the oracle's own drift is 8e-5…2.3e-4. The sharp checks are
  the internal ones a wrong prefactor cannot survive — the slab sum is independent of the splitting
  parameter to **8.9e-16** across a 2.8× range, the chain sum independent of its explicit image
  count to **7.2e-16** across a 6× range — plus `S(−q) = S(q)*`, periodicity in `q`, and derivatives
  against finite differences to 8e-12. On a polar HF chain the term moves `D(q)` by 3.9e-5 eV/Bohr²,
  so it is doing something rather than merely running.

  **What `q → 0` does, corrected.** An earlier draft of this work recorded "2D is discontinuous at
  Γ". That conflated two levels. The *kernel* diverges in every dimensionality — `4π/(Vq²)`,
  `2π/(A|q|)`, `−(2/L)ln|q|`, all three measured here — but the contribution to `D(q)` carries two
  factors of `q` from charge conservation, so only **3D** is left with a finite direction-dependent
  limit. 2D goes as `O(|q|)` and 1D as `q² ln(1/q)`: both continuous at Γ, with a non-analytic
  approach. There is no LO–TO splitting at Γ in 2D, only a linear kink.

- **Divide-and-conquer open-shell analytic stress.** Refused through 0.2.1 for want of a
  spin-resolved pair virial. `electronic_gradient_and_virial_fixed_density_spin` is the restricted
  loop with the exchange coefficient reading `Pα`/`Pβ` instead of half the total, and returns the
  virial alongside the gradient from one pass for the same reason the restricted one does.

  Validated two ways, because either alone would pass for the wrong reason: forced UHF on a closed
  shell reproduces the restricted stress to **2.5e-14** (different code, algebraically identical
  answer), and a neutral triplet chain matches a strain finite difference to **1.9e-8 eV/Bohr³**.
  The finite difference is reported across three step sizes rather than one, and shows the V a
  correct derivative makes: 1.1e-5 at `h = 1e-6` where the SCF's own convergence dominates,
  1.9e-8 at `1e-5`, 2.2e-2 at `1e-4` where harmonic truncation does.

- **The `R⁻³` Klopman–Ohno tail beyond the pair list is summed.** `Am1Options::klopman_ohno_tail`
  and `PbcOptions::klopman_ohno_tail`, default `true`; `false` restores 0.2.1.

  `ewald` made the `1/R` channel exact, but NDDO's kernel is `γ_η(R) = 1/√(R² + η²)`, and
  `γ_η − 1/R = −η²/2R³ + …` was left truncated at the cutoff. `Σ_T |T|⁻³` diverges logarithmically
  in three dimensions, so the total energy drifted with `realspace_cutoff` and converged to
  nothing. `docs/scope.md` recorded it as "⛔ real-space; logarithmically divergent, 0.10 eV per
  unit `ln r_c`".

  The translations the pair list dropped are now summed **explicitly**, out to three cutoffs, using
  the exact `γ_η − 1/R` and not its expansion — the sum depends on the pair only through
  `η_ab = ρ_a + ρ_b`, so it costs one lattice sum per *element* pair. Past that a continuum
  remainder takes over through a quintic taper, and its integrand is per-dimensionality. Only the
  three-dimensional remainder carries a logarithm, and only there is a reference length needed;
  `Σ_T |T|⁻³` converges outright in 1D and 2D.

  Measured on a +1 water cell over a 6.5× range of cutoff, the residual per unit `ln r_c` went from
  **−0.118, −0.098, −0.097 eV** — constant, which is what identifies it as the logarithm — to
  **0.000, −0.000, −0.000**, and the energy spread from 0.197 eV to 6e-5 eV. Forces move by 0.08 %
  of their scale, which is the density shifting under the Fock diagonal the tail adds; the stress
  matches its strain finite difference to 6.9e-9 eV/Bohr³.

  Two things went wrong on the way and are worth recording, because both are invisible in a
  passing test:

  - **The first draft applied the three-dimensional formula in every dimensionality**, and moved a
    charged chain's energy by 3e-2 eV. This is the same error `docs/scope.md` already records for
    `ε_∞` and LO–TO, committed a second time in the same file.
  - **The response was left without it** while the ground state had it, which is the response of a
    Hamiltonian the SCF never converged. It showed up as `D(q = 0)` missing the `q = 0` Hessian by
    4.6e-4 eV/Bohr² — two numbers that are the same number. The tail is now carried through
    `solve_bands` and the DFPT response kernel; the *cutoff-dependent* part of the tail is
    `−(4π/V) ln r_c`, which does not depend on `q`, so the same constant is correct at every `q`.

- **`tests/dc_open_shell_stress.rs` was passing on a coincidence, and now measures something.** Its
  finite difference used a **triplet water chain**, whose energy is not a smooth function of strain
  at all: it jumps in quanta of about 1.7e-5 eV as an occupation switches at the Fermi level, which
  a triplet built from closed-shell waters invites. The quoted 1.9e-8 eV/Bohr³ agreement was
  `E(+h)` and `E(-h)` happening to land on the same branch. Perturbing the Hamiltonian at the 1e-8
  level — all the Klopman–Ohno tail does there — moved that "agreement" to 1.0e-1.

  The fixture is now a **methyl-radical chain**: a doublet with one well-separated singly-occupied
  orbital, whose energy over the same strain sweep is linear to eight figures. The finite
  difference now shows a real V — 1.07e-5, **2.41e-9**, 4.53e-8 across `h = 1e-6, 1e-5, 1e-4` —
  so the minimum is a converged derivative and not a slope. A finite difference quoted at one step
  size cannot tell those apart.
- **The k-point periodic response handles open shells.** `pbc_hessian`, `born_charges` and the
  CPHF behind them accept an unrestricted ground state; 0.2.1 refused with "the k-point periodic
  response is restricted-only". This is the one item of the 0.2.2 list with no sibling crate to
  port from — pm6-rs and pm7-rs refuse in the same place.

  The restricted path solves one CPHF; this solves two, **coupled**, because the kernel is
  `G^σ(ΔP) = J(ΔP_tot) − K(ΔP_σ)` and α reads β's response density through the Coulomb half.
  Solving the channels independently would drop `J(ΔP_β)` from `G^α` and return a plausible
  number. Three factor conventions move with it — what one orbital holds (2 restricted, 1 per
  channel), the exchange weight in both the skeleton and the perturbed Fock, and the relaxation
  term's 4 becoming 2 per channel.

  Forcing UHF on a **closed** shell reproduces the restricted answer to **8.9e-16** eV/Bohr² on a
  3-point mesh, and the Born charges exactly. A genuine doublet chain matches a finite difference
  of the analytic gradient to 5.2e-7 of 15.9. The first of those is the sharp check: on a closed
  shell `P^α = P^β = P/2` makes the two algebraically identical, so any one of the three factors
  being wrong breaks it loudly.

  It found one such break immediately, and it is worth naming because it fails **silently**: the
  occupied/virtual classification tested each level's occupation against a hard-coded `2.0`. On
  the unrestricted path a full level holds 1, so no level was ever classified occupied, `n_ov` was
  zero at every k, and the entire orbital-relaxation term vanished — 74 % of the force constants,
  with no error raised.

  Two things deliberately stay restricted and now say so rather than being answered with the
  restricted equations: **DFPT at finite `q`** (a larger machine — band pairs across `k` and
  `k + q` weighted by occupation differences) and the **field response** behind `ε_∞` and the
  polarizability, which is already three-dimensional-only.

- **The CLI printed `-0.0` for a rigid-body frequency**, and the Rust and Python front ends
  disagreed about which side of zero it fell on. Both are numerically zero; the sign of a value
  below the print precision is not information, but printing it made `tests/test_cli.py` compare
  the two front ends' last bits. Both now print `0.0`.
- **An external electric field works under periodic boundary conditions**, when it is orthogonal to
  every lattice vector. `PbcOptions::electric_field`, and `Am1Options::electric_field` no longer
  refuses a cell outright.

  0.2.1 rejected any field under any cell, with the reason "`F·R` is unbounded along a periodic
  direction". The reason is right and the rule drawn from it was too broad: `F·R` shifts by `F·T`
  under translation by `T`, so the perturbation repeats with the lattice **exactly when
  `F·T = 0` for every lattice vector**. A slab in a field along its normal and a chain in a
  transverse field satisfy that and are ordinary calculations; they were being refused along with
  the ill-defined case.

  The check is now on the direction and names the offending component when it fires. Measured: the
  periodic gradient in a transverse field matches a finite difference of the periodic energy to
  **8.8e-8** eV/Bohr, and a water molecule in a 60 Bohr cell with a field along a non-periodic axis
  reproduces the isolated-molecule path to **5.0e-6** eV — two code paths sharing only
  `crate::dipole`, one number.

  **Not** done, and named so it is not mistaken for done: a finite field *along* a periodic
  direction. That needs the Berry-phase electric enthalpy `E − Ω F·P`, whose field term couples
  neighbouring k-points through `S⁻¹` and therefore requires the SCF to solve its k-points
  together rather than one at a time. The **linear** response along a periodic direction is
  available and validated — `dielectric_tensor` / `ε_∞` through the CPHF — so what is missing is
  the non-linear regime and finite-field geometry optimization. The polarization half of the
  machinery already exists (`pbc::berry`, new in this release).
- **A finite electric field along a periodic direction**, by the Berry-phase electric enthalpy.
  `pbc::run_finite_field`.

  `F·R` is unbounded there, so there is nothing to fix about it: what replaces it is Nunes and
  Gonze's `E − Ω 𝓔·P`, minimized instead of the energy, with `P` the Berry phase rather than `⟨r⟩`.
  Its derivative with respect to the orbitals is built from overlaps between **neighbouring k
  points**, so the k points can no longer be solved one at a time — the SCF gained a `pub(crate)`
  entry point taking a k-resolved additive operator, and an outer loop refreshes it until it stops
  moving.

  The coupling constant is derived from this crate's own polarization convention rather than
  quoted, because the conventions differ between sources and a wrong factor here does not fail —
  it returns a plausible polarizability. **What says it is right** is that `α = Ω ∂P/∂𝓔` by finite
  differences matches the **CPHF** polarizability, two formalisms sharing only the SCF: on a
  hydrogen-only cell they agree to **0.03–0.47 %**, and the residual falls as `O(1/J²)` with the
  string length (1.06 → 0.47 → 0.26 % for J = 4, 6, 8). It caught the one real error on the way:
  the first draft symmetrized the field operator as `(M + M†)/2`, which halves the
  occupied–virtual coupling — the whole of the response — and gave 0.56 of the CPHF value. The
  construction that is both Hermitian and faithful is `A = H − ½PHP` with `H = M + M†`.

  **The comparison is exact only where the two compute the same object.** On a p-block cell they
  differ by 12 %, and that is the Berry phase's own limitation, not the field's: in an atom-centred
  minimal basis the phase tracks the charge *centres* and carries no `dd`, the on-site moment
  between an `s` and a `p` on the same atom. `pbc::berry` already records the same gap for the Born
  charges (0.207 e on HF, 7.5e-13 e with no p orbitals). A planar cell's out-of-plane response from
  this path is **exactly zero**, because there that moment is the whole of it — recorded as a test
  rather than left as a surprise.

  3D, restricted, no smearing, and at least three k points along any direction the field has a
  component in.

  Reachable from all three surfaces: `pbc::run_finite_field`, `am1_rs.finite_field`, and
  `AM1.get_finite_field` (which takes **V/Å** like the rest of the ASE layer and converts with the
  crate's own constants). **Berry-phase polarization**, added earlier in this release, was
  Rust-only until now and gained the same three — `am1_rs.polarization` and `AM1.get_polarization`
  — which is what the project's own native↔ASE parity rule asks for and what
  `tests/test_new_api_0_2_1.py` enforces.

- **The open-shell k-point response now covers DFPT at finite `q` and the dielectric response.**
  Both refused earlier in this release's own notes; both go through the same two coupled spin
  channels as the `q = 0` Hessian, from one shared split of the density
  (`pbc::scf::spin_channel_densities`) so the three cannot disagree about it.

  Forcing UHF on a closed shell reproduces `D(q = 0.3)` on a 4-point mesh to **6.1e-9** eV/Bohr²,
  and gives the restricted `α` and `ε_∞` back **exactly**. A genuine doublet chain's DFPT
  `D(q = 0)` matches the open-shell `q = 0` Hessian to **1.3e-7** of 15.9 (8.2e-9 relative) — the
  same number by two different machines, each running two coupled channels. An open-shell radical
  in a 12 Å box has the isolated radical's finite-field polarizability to **0.41 %**, against 0.17 %
  for the restricted analogue at the same box size.
- **The Berry phase carries the on-site `s`–`p` moment.** It tracked only the charge *centres*
  until now, and that was the single largest reason the Berry route and the CPHF route disagreed.

  The link operator `Λ_{μν} = ⟨χ_μ| e^{−i b·r} |χ_ν⟩` was the diagonal `e^{−i b·τ_μ}`: each orbital
  treated as a point at its own atom. The exact same-atom block, which is all NDDO keeps, is
  `e^{−i b·τ_a}` times `exp(−i b·D^a)` with `D^a_{μν} = ⟨χ_μ|(r − τ_a)|χ_ν⟩` — and in a minimal
  `sp` basis that is exactly the `dd` [`crate::dipole::dipole_operator`] already puts on the
  `(s, p_α)` elements. Both now read it from the same parameter. `b·D^a` is a rank-two operator, so
  its exponential is a rotation in the `(s, u)` subspace and is available in closed form;
  exponentiating rather than truncating at `I − i b·D` keeps `|det Λ| = 1`, so the string's product
  drifts only in phase.

  Measured three ways:

  | | before | after |
  |---|---|---|
  | Born charges vs CPHF, HF cell | 0.207 e | **1.2e-3 e** at 8 points per string, falling as `O(1/J²)` |
  | finite-field `α` vs CPHF, water crystal | 12 % | **0.05 %** |
  | `α_zz` of a planar cell, which is *entirely* this moment | **exactly 0** | 0.25527 against the CPHF's 0.25564 |

  The planar case is the sharpest: with a diagonal `Λ` the `z → −z` mirror made the occupied bands
  parity eigenstates and the link overlaps block-diagonal in that parity, so the field operator
  could not mix them and the out-of-plane response was identically zero. It is the on-site moment
  that couples `s` to `p_z`, so a wrong sign there would have moved it to the wrong number rather
  than merely scaling it.

  A second finding came out of the same comparison: the old 0.207 e was **not** all on-site moment.
  `tests/pbc_berry.rs` compared a Γ-only CPHF against a 12-point string — two different samplings
  of the Brillouin zone — and read the difference as physics. With the sampling matched the
  residual is 1.2e-3 and converging. The test now matches them and asserts the convergence.

- **The polarizability is available for a chain and a slab.** `pbc::polarizability`,
  `am1_rs.polarizability`, `AM1.get_polarizability`.

  `dielectric_tensor` was the only entry point and refused a reduced-dimensional cell — correctly,
  for the `ε_∞ = 1 + 4πα/Ω` step, which needs `Ω` to be a volume — but it took `α` down with it.
  `α` is a *response*, and a response is well defined whatever the cell is periodic in: the origin
  dependence that would spoil an absolute dipole cancels in the derivative because charge is
  conserved. The two are now separate functions, and the 3D refusal names the one that works.

  What stays refused is only the conversion. A slab's `α/A` has units of **length** and is the
  quantity the monolayer literature reports; turning it into a dielectric constant needs a
  thickness, which is a choice about the material rather than something a supercell fixes. The
  units per dimensionality are tabulated on `polarizability` so the 0.2.0 mistake — dividing by a
  length and calling the result `ε_∞` — cannot be repeated by accident.

- **`E_inf(q)` in every dimensionality**, `pbc::dielectric_function` / `am1_rs.dielectric_function`.

  `eps_inf = 1 + 4*pi*alpha/Omega` is a **constant**, and that is a three-dimensional accident
  rather than the general case. The general relation is `eps(q) = 1 - v_d(q) chi0(q)` with `v_d`
  the bare Coulomb kernel of that dimensionality — the same object `pbc::ewald::LongRangeKernel`
  is built around — and `chi0 -> -q^2 (qhat.alpha.qhat)/measure`. Putting the three kernels in:

  | | `v_d(q)` | `eps(q)` | at `q -> 0` |
  |---|---|---|---|
  | crystal | `4pi/q^2` | `1 + 4pi (qhat.alpha.qhat)/Omega` | a constant — this is `eps_inf` |
  | slab, `q` in plane | `2pi/|q|` | `1 + 2pi (qhat.alpha.qhat)|q|/A` | **-> 1** |
  | chain, `q` along it | `2 K0(|q|rho)` | `1 + 2 K0 q^2 (qhat.alpha.qhat)/L` | **-> 1** |

  So a sheet or a wire has no long-wavelength dielectric constant: it does not screen a field whose
  wavelength exceeds its own extent. That is not a limitation of the implementation — it is *why*
  `1 + 4pi*alpha/Omega` cannot be evaluated there, and it is the same fact as a slab having no
  LO-TO splitting at Gamma. Measured: in three dimensions `eps(q)` reproduces `dielectric_tensor`'s
  constant at every `q` to 1e-9; a slab's `eps(q) - 1` fits an exponent of **1.000**; a chain's
  climbs 1.64 -> 1.71 toward the 2 that `q^2 K0` gives up to its logarithm.

  The two-dimensional form is thickness-free, which is what makes `2pi chi_2D` — the
  Rytova-Keldysh screening length — an intrinsic property of the layer. Assigning a slab a
  thickness and quoting `1 + 4pi chi_2D/d` is a different, model-dependent number and is
  deliberately not offered. A chain needs a transverse radius for its logarithm, and it is
  **required** rather than guessed.

  `K0` is the one special function the crate carries beyond `erf`, and it is checked against
  Abramowitz & Stegun's own table before anything is built on it — which caught a real bug on the
  way in: the `I0` series inside it runs in `(x/3.75)^2` and the `K0` series beside it in
  `(x/2)^2`, and writing one variable for both put `K0(0.1)` 0.8 % off with nothing else in the
  crate able to notice.
- **LO–TO below three dimensions: there is nothing to add, and it is now measured rather than
  argued.** The long-range kernel diverges in every dimensionality — `4π/(Vq²)`, `2π/(A|q|)`,
  `−(2/L)ln|q|` — but the *contribution to `D(q)`* carries `q²` from charge conservation, so only
  three dimensions keeps a finite direction-dependent limit and is discontinuous at Γ. That
  discontinuity **is** the LO–TO splitting.

  `|D(q) − D(0)|` at `q = 0.02, 0.01, 0.005` along the periodic axis:

  | | | | |
  |---|---|---|---|
  | 1D chain | 3.6e-3 | 1.8e-3 | **9.0e-4** |
  | 2D slab | 4.6e-3 | 2.2e-3 | **1.1e-3** |
  | 3D crystal | 1.071e-1 | 1.074e-1 | **1.075e-1** |

  The low-dimensional cases converge to Γ; the crystal does not. So `frequencies_with_lo_to`
  refusing a chain or a slab is the physics and not a gap — there is no splitting at Γ to add —
  and the non-analytic *approach*, which is real, the DFPT path already carries exactly. 0.2.0's
  "127 cm⁻¹ of splitting on a polar chain" was an artifact of applying the 3D kernel.

### Performance

- **The parameter set is cached per method.** `Am1Parameters::for_method` re-parsed the embedded
  CSV and re-ran the `rho1`/`rho2` secant solves for every element on **every call** — and every
  function on the Python surface calls it at its top. Measured at **270 µs**, against a 1361 µs
  water single point: about 17 % of every small-molecule call, paid again on every step of a
  molecular-dynamics loop. It is a fixed per-call cost, which is exactly the shape a large-system
  profile cannot see. Now 2 µs for a clone, and `Am1Parameters::shared` borrows for the callers
  that only read.

- **The infrared atomic polar tensor is `O(N³)`, not `O(N⁴)`.** It built `∂P/∂R_j` — an `nao²`
  matrix, `O(nao² n_occ)` each — for all `3N` perturbations and traced each against `M_α`, to keep
  three numbers per perturbation. Writing `∂P = B + Bᵀ` and using that `M_α` is symmetric gives
  `Tr[∂P M_α] = 2w Tr[Uᵀ (C_vᵀ M_α C_o)]`, so the `nao²` object never has to exist: project `M_α`
  into the occupied–virtual block once, and each perturbation is one Frobenius product of
  `n_vir × n_occ`. The factor `2w` is 4 for RHF and 2 per spin for UHF — the same convention the
  periodic relaxation term uses. Exact, and checked by the three independent identities already in
  `tests/ir.rs` (the sum rule at 3e-15, a dipole finite difference, and the interchange theorem).

- **A pack-index table** in the two-centre Fock contraction, replacing a branch and a multiply in
  the innermost loop. Bit-identical to the closed form and strictly less work, with
  `the_pack_table_is_the_closed_form` asserting the equivalence over the whole domain.

- **The `q = 0` periodic response no longer holds a density per perturbation.** The
  coupled-perturbed solve consumes one perturbation's response density at a time, but built all
  `3N` of them before the loop and kept a second array of the same size for the spin-summed total.
  The arithmetic is identical either way — the loop nest is the same, only its order changed — but
  the resident set was `(1 + n_channels) · ndof · n_T · nao²` doubles where
  `(1 + n_channels) · n_T · nao²` will do. That is a factor of `3N` on two of the three arrays of
  that shape, and the Born charges and the polarizability, which read only each perturbation's
  **origin** block, now stream as well.

  Measured with a peak-tracking global allocator rather than reasoned about
  (`tests/response_memory.rs`): on 27 atoms with 7 translations the response adds **14.7 MB** over
  the ground-state SCF's own, against the **39.7 MB** the old shape needed. The remaining third is
  the bare `∂F/∂R`, which is assembled pair-major and so cannot be streamed without `O(N)` passes
  over the pair list; holding it sparsely is an `O(N)` win asymptotically but costs more below
  about a dozen atoms, so it is left as it is and named here rather than half-done.
- **`Matrix::frobenius_dot` accumulates in eight lanes.** A single running total is a dependency
  chain the compiler may not reorder, so the loop ran at one add per latency however wide the
  machine. This changes the summation order, as the 0.2.1 DIIS packing did.

- AM1-BCC no longer perceives the topology twice (`write_mol2` re-derived it, which also meant the
  file could in principle disagree with the charges beside it — `BccResult` carries the bonds now),
  and the 405-entry parameter table is parsed once per process rather than per call.

### Not done, and named so it is not mistaken for done

- **The Berry phase and the finite field below three dimensions, or open-shell.** Both are
  *implemented* in 0.2.2 — the heading is about where they stop. `pbc::berry` and
  `pbc::run_finite_field` require a three-dimensional restricted cell: the polarization quantum is
  `e a/Ω` and Ω has to be a volume, a slab or a chain has a polarization along its periodic
  directions only which the module does not separate out, and an open-shell cell would need each
  spin manifold's phase separately. Both refuse rather than answering with the three-dimensional
  closed-shell expression. A field *orthogonal* to every lattice vector needs none of this and is
  supported in every dimensionality and for open shells (`PbcOptions::electric_field`); so is the
  polarizability itself, which is a response rather than a phase.
- **The CPHF perturbation batching named in 0.2.1's `fock.rs` as "the next thing to try" is not
  done, because it was measured and it is slower.** The experiment was run in a sibling NDDO crate
  with the same loop: batching the response Fock across degrees of freedom went 5.2 → 8.8 s on a
  102-atom Hessian, gathering the density sub-blocks 5.2 → 9.8 s, and packing the Coulomb
  contraction 4.17 → 4.39 s. The batching did what it was meant to structurally — 70 Fock passes
  instead of 3961 — and was still slower: at that size the whole integral set is about 4 MB, so it
  sits in L3 across calls and there is no traffic to save, while batching costs the per-DOF
  parallelism the `par_iter` gets for free. At NDDO block sizes this loop is bound by **per-pair
  overhead**, not by memory or arithmetic. The measurement is recorded in `src/fock.rs` in place of
  the suggestion, so the afternoon is not spent again.

## 0.2.1

### Added

- **External electric field** for molecules: energy, analytic gradient and analytic Hessian.
  `Am1Options::electric_field` (eV per e·Bohr). `E(F) = E₀ − μ·F` with this model's own dipole;
  the operator is the new `dipole` module, which the molecular field and the periodic field
  response both call rather than transcribing. `born_charges_from_response` still writes its own
  three-term derivative form, so the sign convention is shared by two of its three consumers.
  Because that operator is *linear* in the nuclear positions the field adds nothing to the
  fixed-density second derivative and reaches the Hessian only through the CPHF response — which
  is why the Hessian is checked against finite differences under a field rather than assumed.
  Measured: gradient 1.8e-6 eV/Bohr and Hessian 8.1e-7 relative against a full-SCF finite
  difference; `−∂E/∂F` reproduces the reported dipole to 3e-8 e·Bohr. Refused under a cell, since
  `F·R` is unbounded along a periodic direction.
- **Infrared spectra** (`ir`): the atomic polar tensor `∂μ_α/∂R_{a,β}` as a raw `3 × 3N` matrix,
  and km/mol intensities projected onto normal modes. Validated three ways — the translational
  sum rule `Σ_a ∂μ/∂R_a = q δ` (3e-15), a full-SCF dipole finite difference (7e-7 e), and the
  interchange theorem `∂μ_α/∂R_j = −∂²E/∂F_α∂R_j`, the right-hand side taken as a finite
  difference of the analytic gradient in the field (1.2e-6 e). The two routes share no code past
  the SCF, which is the point; a *field* CPHF would make the second route analytic too, and there
  is no molecular field-CPHF path in the crate. CO₂'s symmetric stretch comes out dark at 1.6e-15
  against 8.75 for the antisymmetric one.
- **Wavefunction output in Molden format** (`molden`): `[Atoms]`, `[STO]` and `[MO]`. The AM1
  basis is Slater-type, so `[STO]` represents it exactly with no Gaussian expansion invented. The
  file and the docs both state the caveat that matters: NDDO *assumes* an orthonormal AO basis, so
  the coefficients are in an implicitly orthogonalized basis while the listed Slater functions are
  the raw ones.
- **First-order orbital response** is returned rather than discarded:
  `analytic_hessian_with_response` hands back `U`, `G` and the response density the CPHF already
  solved for. An infrared spectrum therefore costs a Hessian and nothing more.
- **Normal modes** on `VibrationalModes` — mass-weighted eigenvectors, Cartesian displacements,
  and each mode's overlap with the rigid-body subspace, so a linear molecule's five rigid-body
  modes are *discovered* rather than assumed from `3N − 6`.
- **β orbitals for UHF.** `Am1Result` carries the β energies and coefficients; the SCF solved for
  them and then threw them away, which made a spin-polarized wavefunction unreportable.
- **DFPT is generalized in `k` as well as `q`.** `DfptOptions` takes an arbitrary mesh or an
  explicit k-point list, and `DfptResult` returns the `(k, k+q)` band energies, occupations and
  first-order densities. `PbcOptions::kpoints` lets the response and the ground state share one
  *resolved* k-set rather than two independent resolutions of the same description.
- **The long-range monopole term is in the DFPT response**, on a 3D cell, at every `q`
  (`LongRange::Auto`, the default). `EwaldSum::phased_pair_potential` returns the value, gradient
  and Hessian of the phased sum `Σ_T e^{iq·T} erfc(α|d+T|)/|d+T| + …` in one pass, with the
  reciprocal half summed over `k = G − q` — the phase moves the *shell*, not just the summand.

  **The element dropped is `k = 0`, not `G = 0`**, which is a correction to the convention 0.2.0's
  docs recorded as settled. `k = 0` arises only when `q` folds to Γ, where it is exactly the
  divergent term the neutralizing background cancels, so the rule reduces to this crate's tin-foil
  `Σ_{G≠0}` at `q = 0`. Dropping the long-wavelength element `k = −q` instead — the alternative,
  which keeps the direction-dependent part out of `D(q)` so LO–TO can supply it — was implemented
  and **rejected on measurement**: that rule is not periodic in `q`, failing `Δ(q+G) = Δ(q)` by
  1.2e1 where the accepted rule gives 9.2e-14, and it has no well-defined answer at a zone
  boundary where several `k` tie for smallest.

  So `D(q)` is now the **full** dynamical matrix and its `q → 0` limit is direction dependent,
  which is the physics. It must **not** be combined with `frequencies_with_lo_to`, which exists to
  restore that same physics to the supercell route; use one or the other.

  A phase error here leaves the matrix Hermitian and the frequencies real, so this is validated by
  identities: the kernel is independent of the real-space cutoff to **2.2e-16 at `q = ¼`**, where
  the truncated sum alone moves by 1.4e-1; `Δ(−q) = Δ(q)*` to 1.5e-15; derivatives against finite
  differences to 4e-10; at Γ it reproduces `pbc_hessian` to 1.7e-8 relative while contributing
  3.6e-2 eV/Bohr²; and the acoustic sum rule holds to 1.2e-9, which it does because the
  fixed-charge second derivative phases the pair term and never the self term.

  What the correction does *not* cover is the `R⁻³` Klopman–Ohno tail, which stays with the
  real-space sum — so the assembled `D(q)`'s cutoff dependence at `q = ¼` falls from 3.1e-2 to
  2.0e-2 rather than to zero. That residual is the tail, not the monopole channel; the 2.2e-16
  above is what separates the two claims.

### Fixed

- **DFPT sampled a different Brillouin zone from its own ground state.** The response mesh was a
  hand-rolled Γ-centred grid built from `kmesh.sizes()` alone, so a `MonkhorstPackShifted` request
  gave the ground state `{−1/4, +1/4}` and the response `{0, 1/2}`. Nothing announced it: the
  force constants stayed real and the frequencies plausible. The regression test asserts the
  `q = 0` identity on a shifted mesh — agreement is now 1.2e-9 relative, and the test also
  measures that the two meshes differ by 2.9e-2 eV/Bohr², so it could not have passed by accident.
  Non-periodic axes are collapsed too, which removes `n²` redundant diagonalizations on a slab.
- **`ε_∞` was 27× too close to 1: the polarizability was never converted to atomic units.**
  The field CPHF is solved in this crate's interior units — orbital energies in eV, positions in
  Bohr — so `U ~ M/Δε` carries Bohr/eV and the assembled `α = Σ_a R_a ΔQ_a` is in `e²·Bohr²/eV`.
  It was returned labelled Bohr³ and fed straight into `ε_∞ = 1 + 4πα/Ω`, which needs atomic
  units. The missing factor is one Hartree in eV.

  Nothing caught it because every test checked `α`'s **shape** — symmetric, positive-definite,
  independent of the cell origin — and a value wrong by a constant factor satisfies all three.
  The new `a_molecule_in_a_large_box_has_the_isolated_molecule_polarizability` checks its
  *magnitude* instead, against the finite-field polarizability of the same molecule with no cell:
  two routes sharing only the SCF and the dipole operator, one an analytic CPHF and the other two
  extra SCF solves per axis. Water's mean `α` is 3.379 Bohr³ isolated, and the periodic value
  converges to it as the box grows — 0.85 % at 7 Å, 0.40 % at 9 Å, **0.17 % at 12 Å** — where
  before the fix it sat at 0.125 Bohr³ and did not converge to anything.

  This changes every `ε_∞` and therefore every LO–TO splitting reported by 0.2.0 and by earlier
  drafts of 0.2.1.
- **LO–TO splitting and `ε∞` were three-dimensional formulas applied to chains.**
  `ε∞ = 1 + 4πα/Ω` and `D_NA ∝ 4π/(Ω q·ε∞·q)` need `Ω` to be a volume, but `Lattice::measure`
  returns a *length* for a chain and an *area* for a slab, and `tests/pbc_lo_to.rs` ran both on 1D
  chains. A genuinely 1D-periodic chain has **no** LO–TO splitting as `q → 0` (the term vanishes
  as `q² ln q`), so the 127 cm⁻¹ and 1631 cm⁻¹ figures recorded in the 0.2.0 notes below were
  artifacts. Both functions now require a fully periodic cell, and the tests were moved to a 3D
  polar crystal, where the added term matches its closed form to 2e-15.
- **DFPT reported `residual: NaN`** when the coupled-perturbed solve hit its iteration cap, so a
  caller could not tell a stiff system from a broken one. It now reports the residual it reached,
  and the tolerances and iteration cap are options rather than private constants.
- **The analytic UHF Hessian built a different Hamiltonian from the SCF**, always molecular and
  always without the long-range or far-field corrections, whatever the options said. Its skeleton
  loop is structurally molecular, so a periodic or far-field-screened request is now refused
  rather than silently answered with a molecular result.
- A time-reversal-folded mesh, a `q` component along a non-periodic axis, and an explicit k-list
  whose weights do not sum to 1 are all refused by DFPT instead of quietly producing an answer.
- **`am1-rs energy` crashed part-way through its output on any machine whose locale is not
  UTF-8.** Python encodes `print` with the locale's codec, and the dipole line was written
  `e·a0`: on a Japanese Windows (cp932) or under the `C` locale that minimal Docker images ship
  with, that raised `UnicodeEncodeError` after six lines had already gone to stdout, and the
  command exited 1 with a truncated report. `gradient` and `optimize` went the same way. The Rust
  binary never raised — it writes UTF-8 whatever the locale — but rendered mojibake on the same
  console, so the two front ends did not agree there either.

  Both CLIs now print **ASCII only** (`e*a0`, `cm^-1`, `eV/A`), which is the fix that works on
  every console rather than merely avoiding the exception, and the Python front end additionally
  forces its streams to UTF-8 so that a non-ASCII message from the native layer cannot kill it.
  `pip install` and the `am1-rs` console script were verified end to end in a clean virtualenv
  under both cp932 and a forced ASCII stdout.

  The test suite had not caught this because the development machine had `PYTHONIOENCODING`
  set in its shell and pytest's subprocesses inherited it, giving every child a UTF-8 stdout that
  no user would have. `tests/test_cli.py` now strips that variable from the child environment,
  asserts that both CLIs' bytes are ASCII in every mode, and runs `energy` under a deliberately
  ASCII stdout.

### Performance

- **The molecular SCF's DIIS history is half the size, and peak memory fell 28 %.** `rhf_loop`
  kept three depth-8 histories — Fock, `[F,P]` error, density — as dense `nao²` matrices: at 1602
  AOs (an 801-atom water cluster) twenty-four of them are **492 MB**, against a measured 877 MB
  peak for the whole run. It was the single largest term.

  Every matrix in all three is either symmetric (`F`, `P`) or **anti**symmetric (`[F,P] = FP−PF`,
  whose diagonal is identically zero), so one triangle determines the other and packing loses
  nothing — `packing_a_diis_history_preserves_it_exactly` checks the round trip and the Frobenius
  products the extrapolation actually consumes, including that the commutator's diagonal really
  is zeroed. The density history is additionally only built for the accelerator that reads it
  (`AdiisCdiis`), which is a further third off for a CDIIS-only run. `uhf_loop` gets the same
  treatment, its stacked two-spin error packed as two triangles end to end.

  Measured on the 801-atom cluster: **877 MB → 632 MB** peak working set, same energies, and the
  SCF converging in 13 iterations against 14 — the histories are bit-equivalent, but the
  Frobenius products are summed in a different order, which moves the last iteration across the
  convergence threshold.
- **The Hessian's orbital-relaxation contraction is a matrix product.** `H_relax[a][b] =
  4 G^a : U^b` was `ndof²` independent Frobenius dots, which re-reads every `G` row `ndof` times
  and is memory bound; stacking the ov-blocks makes it `G Uᵀ`. The periodic version was the same
  nest four levels deep and *not parallelized at all*; it is now two products per k point.

  The molecular one is **tiled**, and that is not incidental: stacking `G` and `U` whole would
  add two `ndof × n_ov` buffers, which is `O(N³)` and would double the largest array the Hessian
  holds. Copying 64-row tiles bounds the extra at `O(N²)` and trades for redundant copying worth
  `1/64` of the arithmetic.
- **The DFPT response is streamed and solved in parallel.** Each perturbation is solved,
  contracted into `C(q)`, and dropped, so the resident response is `O(threads · n_k · nao²)`
  rather than `O(ndof · n_k · nao²)` — and the `j'` loop, which was serial, now runs under rayon.
  The `pbc_dfpt` suite fell from 4.25 s to 2.50 s. `keep_response` restores the full array for a
  caller that wants it, and a test asserts that asking for it leaves `C(q)` bit-identical.
- **Assembling `C(q)` is `O(N³·n_k)`, not `O(N⁴·n_k)`.** The bare perturbation is held as its
  nonzero entries grouped by translation, not as a dense `nao²` matrix per k point. Displacing
  one atom changes the Hamiltonian only where that atom appears — `O(1)` blocks — plus, on a 3D
  cell, the on-site diagonal of every atom from the long-range monopole channel, which is `O(N)`.
  Contracting every pair of perturbations against that is an order cheaper than against `nao²`.

  Measured on a chain grown by repeating its cell, with the counts returned on `DfptResult` so
  the claim is checkable rather than asserted: the contraction's extent scales as **N^-0.04**
  against **N^2.00** for the dense one it replaces — 2.0 orders removed — and is 4.2× smaller
  already at twelve atoms. Below about eight atoms the sparse form is *larger*, which the test
  prints rather than hides.
- **The divide-and-conquer DIIS history is now linear in the atom count, not quadratic.** It
  stored a dense packed triangle while the divide-and-conquer density is *identically* zero
  beyond the buffer radius — so most of what it held was zeros. Storing the density's actual
  sparsity pattern instead gives a measured scaling exponent of **1.05**, against **1.99** for
  the dense triangle it replaces; both are asserted in `tests/divide_conquer.rs`, because a
  linear number on its own could be an accident of size. This is the dominant memory term of a
  large run. `DcResult` gains `diis_pattern_elements` and `dense_triangle_elements` so the claim
  is inspectable rather than believed.

  The same change cuts the memory traffic: on a 1029-atom cluster `dc:diis` fell from 1.070 s to
  0.419 s and the run from 5.11 s to 4.45 s, with identical energies and the same iteration count.
- **The DFPT response is DIIS-accelerated**, and it needed to be: the coupled-perturbed solve is
  a linearly mixed fixed point, and on a polar 3D cell it did not converge at all within its
  200-iteration cap — a water crystal stalled at `5.6 × 10⁻⁹` against a `10⁻¹⁰` tolerance and
  raised. Every one of those iterations is a real-space two-electron build plus a diagonalization
  per k point. With Pulay extrapolation on the fixed-point residual the same system converges at
  the default tolerance, and every DFPT identity reproduces bit for bit — it is the same fixed
  point, reached sooner.
- **The periodic CPHF's two basis transforms were `O(nao⁴)`; they are now `O(nao³)`.** Both
  `project_ov` (`Cᵥ† M C_o`) and the response density (`C_v U C_o† + h.c.`) were written as a
  single loop nest over `(v, o, μ, ν)`, which rebuilds the inner `M C_o` — a quantity
  independent of the virtual index — once for **every** virtual. Since `n_v` and `n_o` both grow
  with `nao`, `n_v n_o nao²` is a fourth power. Factoring each into two products makes it
  `nao² n_o + nao n_v n_o`.

  Measured against the loop nest it replaces, which is kept in the test suite as the reference:

  | `nao` | loop nest | factored | speedup |
  |---|---|---|---|
  | 32 | 1.197 ms | 0.041 ms | **29×** |
  | 64 | 17.043 ms | 0.185 ms | **92×** |
  | 128 | 360.044 ms | 0.996 ms | **362×** |

  The two agree to 3.1 × 10⁻¹⁵ relative, and `factoring_project_ov_lowers_its_scaling_exponent`
  asserts that the advantage *grows* with size, since a constant-factor win would not.

  The compact occupied/virtual coefficient blocks the products need are gathered once per k
  point, into `KOrbitals`. Gathering them per call instead — the obvious first cut — made the
  DFPT suite **five times slower** (3.7 s to 18.5 s), because at the `nao ≈ 8–40` of a small
  cell the allocations cost more than the fourth power saved. The complex arithmetic likewise
  accumulates in place (`matmul_acc_seq`) rather than allocating a matrix per real product and
  combining afterwards.
- **The molecular CPHF called faer's *parallel* matmul from inside its own rayon loop.**
  `Matrix::matmul_seq` exists precisely for this and its documentation names the CPHF
  perturbation loop as the case, but `project_ov` and `ao_response_density` used the parallel
  form — so faer's workers contended with the outer pool over the same threads. Both now use the
  sequential, transpose-free products. `cphf:to_ao` fell from 1.447 s to 1.034 s and
  `cphf:to_mo` from 1.346 s to 1.000 s of thread time on a 48-atom Hessian, about 27 % off each.
  The divide-and-conquer density build had the same nesting and is fixed with it.
- **Four more hand-written `n³` loop nests went to the blocked kernel**, all of them on inner
  paths. These are counted rather than timed — the machine they were developed on was running
  other work, and a stopwatch there measures the load, not the code:

  | site | what it is | per what |
  |---|---|---|
  | `pbc::dfpt`'s `mul` / `adjoint_mul` / `mul_adjoint` | the CPSCF's complex transforms | 4 × per k point per iteration |
  | `pbc::scf`'s `P(k) = Σ_i f_i c_{μi} c*_{νi}` | the periodic density build | per k point per SCF iteration |

  The periodic density build also stopped walking the empty levels. It looped every orbital and
  `continue`d on `f_i = 0`, which skipped the arithmetic but not the traversal; gathering the
  filled columns first makes the products `nao² · n_occ` rather than `nao³`.
- **The Fock build no longer copies the density to halve it.** `build_fock` and `build_g_matrix`
  each cloned the density and scaled it by ½ to make the same-spin matrix for the exchange. The
  exchange is linear in that argument, so `build_fock_spin_with` now takes a `spin_scale` and the
  callers pass the total density with `0.5` — exactly as `pbc::scf::build_realspace_fock` already
  did. That removes an `nao²` allocation, copy and scale from every call, and `build_g_matrix` is
  called `3N` times per CPHF iteration: at 1602 AOs each of those copies was 20 MB.
- **The perturbed Fock's long-range term evaluates a quarter of the lattice sums.** The nest
  built `Δ'(R_b − R_a)` about `2·nat²` times — the `a == c` branch walks every `b` for each `c`,
  and the other branch asks for every ordered pair — where only `nat(nat−1)/2` are distinct.
  `Δ'` is **odd** in the separation (`Δ` is even), so one triangle tabulated once serves both
  halves. `the_pair_gradient_is_odd_in_the_separation` pins that at 1.6e-15.
- **The Ewald pair Hessian is evaluated on half the pairs.** `LongRangeMonopole::energy_hessian`
  called `delta_hessian` for both `(a,b)` and `(b,a)` — the same lattice sum twice, and the
  lattice sum over every translation and reciprocal vector is what costs. That Hessian is *even*
  in the separation (it is built from `d̂_i d̂_j` and even powers of `|d|`, and the translation
  set is symmetric), so one triangle suffices: exactly half the work, with
  `the_pair_hessian_is_even_in_the_separation` pinning the symmetry it rests on at 1.2e-15. The
  region also gained the `ewald:hessian` timer it had never had, which is why the plan's
  "profile before optimizing" step had never been possible there.
- **No `unwrap` or `expect` outside tests, anywhere in `src/`** — down from eighteen. Most were
  `blocks.get(ImageOffset::origin()).unwrap()`: "the origin is always in the translation set" is
  a property of how that set is built, not one the type enforces, and a panic from inside an SCF
  iteration is the worst way to learn otherwise. `RealSpaceBlocks::origin`/`origin_mut` return a
  `Result` and every site propagates it. The two that remained were genuinely infallible and are
  now infallible *structurally*: the longest-axis search is a fold over three fixed axes rather
  than `max_by(..).unwrap()`, and an empty level set is handled rather than indexed into.
- `Matrix` gained transpose-free products (`transpose_matmul`, `matmul_transpose`, and their
  `_seq`/accumulating variants). Materializing a transpose to multiply by it is an extra
  allocation and copy per call; on `P = C_occ C_occᵀ`, once per SCF iteration, the transposed
  view is **3.2×** faster than the copy at 600 AOs and 400 occupied orbitals — measured, because
  handing a kernel a non-native layout can as easily cost as save.

  That figure is a **minimum over repetitions**, not a mean, and the difference matters: a
  three-run mean of the same code reported 1.24× on an idle machine and 1.65× *slower* on a busy
  one. Interference only ever makes a sample slower, so the minimum is the least-contended
  estimate; the test asserts only against a 3× catastrophe, because anything tighter would be
  asserting that the machine is idle.
- DFPT no longer rebuilds `h_j(k)` inside the `j'` loop or inside the CPSCF iteration; it is
  invariant in both, and the Bloch sum costs `O(n_T · nao²)` each time.
- `farfield` is instrumented, and `tests/dc_where_the_time_goes.rs` reports where a large run's
  time actually goes.

### Not done, and named so it is not mistaken for done

An audit of this release against its own plan turned these up. They are recorded here rather than
left for a reader to discover.

- **Parameter structs landed for two of the plan's four targets.** `build_core_with_neighbors`
  takes `CoreBuildOptions`, and the CPHF trio — `apply_orbital_hessian`, `cphf_ov` and
  `cphf_ov_fixed_point` — now share a `CphfContext` holding what does not change from one
  perturbation to the next, which also makes it impossible to hand the fixed-point fallback a
  different Hamiltonian from the solver that gave up. `skeleton_fock_ov` and the DFPT helpers
  still take long positional lists; 20 `#[allow(clippy::too_many_arguments)]` remain.
- **The MOPAC oracle covers one molecule.** Verified, not assumed: of the 61 cases in MOPAC's
  `tests/keywords`, `AM1.mop` and `RM1.mop` are the only ones selecting these methods and both
  are CO₂. Widening it means *running* MOPAC rather than reading it. The comparison was deepened
  instead — the whole orbital spectrum rather than one eigenvalue.
- **The long-range monopole term in the DFPT response on a chain or a slab.** It landed for 3D
  cells (above). `LongRangeMonopole` is itself three-dimensional, so in 1D and 2D there is no such
  correction anywhere in the crate and nothing to generalize; `LongRange::Require` errors there
  rather than approximating quietly. Implementing the low-dimensional kernels — 2D `2π/(A q)` with
  a slab convention, 1D vanishing as `q² ln q` — is a separate piece of work.

- **No Barnes–Hut tree for the far field**, despite the module documentation suggesting one
  belongs there. Measured first: on a 1029-atom divide-and-conquer run with the far field on,
  `farfield:potential` is **0.5 %** of the runtime, against 36 % for the subsystem
  diagonalizations. A monopole pair costs about ten flops and the loop is embarrassingly
  parallel, so making it `O(N log N)` would save half a percent — and would put an
  acceptance-angle discontinuity into the energy surface to do it. The `O(N²)` term does win
  eventually, but the crossover is around `10⁴–10⁵` atoms. `src/farfield.rs` records the
  measurement, so the decision can be revisited against a number rather than re-argued.

### Changed

- `build_core_with_neighbors` takes a `CoreBuildOptions` struct instead of four positional
  arguments; `Am1Options::core_build()` derives it, so a path that builds `H_core` for itself
  cannot disagree with the SCF about which corrections are on.
- `#![forbid(unsafe_code)]` — there was none, and now there cannot be.
- **The Python API roughly doubled.** New in `am1_rs.native`: `orbitals`, `molden`,
  `ir_spectrum`, `dipole_derivatives`, `orbital_response`, `pbc_hessian`, `born_charges`,
  `dielectric`, `dfpt`, `lo_to_frequencies`; `electric_field=` on `single_point`, `gradient`,
  `optimize`, `hessian`, `frequencies` and six more; `multipole_cutoff=` on `divide_conquer`.
  The ASE calculator gains the matching `get_ir_spectrum`, `get_dipole_derivatives`,
  `get_orbitals`, `get_orbital_response`, `write_molden`, `get_frequencies`,
  `get_am1_bcc_charges`, `get_phonons`, `get_born_charges`, `get_dielectric_tensor`,
  `get_dfpt_frequencies`, `get_lo_to_frequencies` and `optimize`, plus `field=` and an
  `atoms.info["field"]` override routed through `check_state`. Both CLIs gain the modes
  `orbitals`, `ir`, `molden` and the flags `--field FX FY FZ`, `--molden-output FILE`.

  `tests/test_new_api_0_2_1.py` now *enumerates* `am1_rs.native`'s public functions and requires
  each to name its ASE counterpart, instead of checking a hand-written list. The hand-written
  list is what let `lo_to_frequencies` exist in Rust and in neither Python surface.
- **Divide-and-conquer under a periodic cell, from Python and ASE.** `native.divide_conquer`
  takes `cell`/`pbc` (with `realspace_cutoff`/`exchange_cutoff`, which matter under a cell and
  not for a molecule), and `AM1(divide_conquer=True)` now routes a periodic structure through it
  instead of raising. The Rust API had accepted a lattice since 0.2.0; the ASE error message
  saying the buffers were "not wired up yet" described 0.2.0 and had outlived it.
- **`tests/theory_components.rs`** — the *pieces* of the formulas, against what theory says each
  one must be. Everything else in the suite is an end-to-end identity, which is strong and also
  blunt: it says a chain is wrong without saying which link, and a compensating pair of errors
  passes it. These twelve tests each check a property that follows from the mathematics alone.

  The sharpest is the monopole limit. `(ss|ss) = e²/√(R² + ρ²)` implies that the relative
  deviation from `e²/R` is `−ρ²/(2R²)`, so `deviation × R²` is constant *and identifies `ρ`* —
  and the value recovered from the integral's long-range behaviour, 1.9873…1.9946 Bohr across
  four radii, is the parameter table's own `rho0(O) + rho0(C) = 1.994724`. That pins the
  functional form, both elements' parameters and the `AM1_EV` conversion in one measurement.

  The rest: each multipole channel decays at the order its expansion demands (measured
  `R^-0.998`, `R^-1.996`, `R^-2.997` against `−1, −2, −3`); the two-electron integrals have their
  three permutation symmetries, including the electron exchange that swaps the two *atoms* and
  reaches them through different branches; the whole `10 × 10` block is rotation covariant, tested
  on a contracted quantity so every index has to transform correctly; the overlap has the
  inversion parity its orbitals imply and is unchanged by relabelling; the converged density is
  an idempotent projector with `Tr P` the electron count; `[F, P] = 0` at the SCF solution; the
  reported electronic energy really is `½Tr[P(H + F)]`; Koopmans uses the level it should; and
  the energy is invariant under rigid motion.
- **`tests/orbital_response.rs`**, checking `U^j_{ai}` against a finite difference of the MO
  coefficients rather than only through what it is contracted into. Two things make that
  comparison delicate and both are handled explicitly: eigenvector *phase* (aligned against the
  response channel's own coefficients — aligning against a separately re-run SCF flips `U`
  wholesale, which the first draft did and measured as `|Δ| = 2|U|`), and *degeneracy* (methyl's
  `e′` pair mixes arbitrarily under displacement, so the coefficient comparison is done on the
  non-degenerate `H₂O⁺` while the phase-invariant response *density* is checked on methyl).
  Measured: `U` to 9.6e-7 (RHF), 7.0e-7 and 5.6e-7 (UHF α and β); `∂P/∂R` to 1.7e-7.
- **The MOPAC oracle now compares the whole orbital spectrum**, twelve eigenvalues per method
  rather than the single Koopmans IP, including CO₂'s two degenerate pairs — which a broken
  two-centre rotation would split while leaving `ΔHf` and the HOMO almost unmoved. Worst case
  across all twelve: **0.0022 eV for AM1, 0.0034 eV for RM1**, both at the deepest level.

  It still covers one molecule, and that was verified rather than assumed: of the 61 cases in
  MOPAC's `tests/keywords`, `AM1.mop` and `RM1.mop` are the only ones selecting these methods and
  both are CO₂. Widening it means *running* MOPAC, not reading it.


## 0.2.0

### Added

- **RM1** (Rocha *et al.* 2006), sharing AM1's functional form and therefore its entire code
  path — gradients, Hessians, periodic boundary conditions and divide-and-conquer all work
  unchanged. Select with `method="rm1"` on any Python entry point, `--method rm1` on the CLI, or
  `Am1Parameters::for_method(NddoMethod::Rm1)` in Rust. Covers H, C, N, O, P, S, F, Cl, Br, I.
  Parameter provenance in `THIRD_PARTY_NOTICES.md` §3. See `docs/methods.md`.

- **Periodic boundary conditions** — 1D chains, 2D slabs and 3D crystals, dimensionality taken
  from `atoms.pbc`:
  - Γ-point and Monkhorst–Pack k-point sampling, with time-reversal folding and automatic
    collapse of non-periodic axes.
  - Fermi–Dirac smearing with a bisected chemical potential, electronic entropy and `T→0`
    extrapolation (new `fermi` module).
  - **RHF and UHF at both Γ and k-points.** Forced-UHF reproduces RHF on a closed shell to
    1e-7 eV with an exactly vanishing spin density.
  - Analytic forces and **analytic stress** for all three dimensionalities. Stress components
    touching a non-periodic axis are exactly zero; the periodic measure is a volume, area or
    length as appropriate.
  - The AM1 core–core Gaussian corrections are included in the lattice sum, its gradient and its
    virial.
  - Net charge per cell, including the absolute energy in 3D — see Ewald summation below.
  - New `lattice`, `neighbors` and `pbc` modules; extended-XYZ `Lattice=`/`pbc=` parsing.
  - `docs/pbc.md`.

- **Ewald summation** for the long-range monopole electrostatics of a 3D cell (`ewald`, default
  on), under the tin-foil boundary condition:
  - Makes a **charged cell's total energy meaningful**. Across a 6.5× range of real-space cutoff
    a +1 water cell moves 0.20 eV with it and 403 eV without.
  - Applied through the **net** charges as `−V_a`, `V_a = Σ_b Δ_ab Q_b`, rather than split
    across the electron–core, Coulomb and core–core terms the way `γ_ab` is. The split form is
    algebraically identical and numerically ruinous — it shifts `H_core` and the Coulomb term by
    ±660 eV for a carbon in a 12 Bohr cell — and it stopped a lone neutral carbon from
    converging at all.
  - Energy, analytic gradient, analytic stress and analytic Hessian, the last including both the
    fixed-charge second derivative and the charge-response term in the CPHF.
  - Validated four independent ways: the rock-salt Madelung constant to 10 digits, exact
    independence of the splitting parameter `α` for both the potential and the **stress**, the
    dipole surface term `2π|p|²/3V` against a direct lattice sum, and finite differences for
    every derivative.
- **Ewald in 2D and 1D**, so a slab or a chain gets the same treatment a crystal does rather than
  no treatment at all:
  - **2D by Parry**, not by the Yeh–Berkowitz vacuum-slab trick. Parry gives the in-plane 2×2
    stress directly; a vacuum slab has no meaningful `∂E/∂ε_zz` because the `c` axis is fictitious,
    and is only asymptotically exact. The implementation needs `erfcx(x) = e^{x²}erfc(x)`, added
    to the `Scalar` trait with its derivative rules, because the naive `e^{hz}·erfc(h/2α + αz)`
    overflows at moderate `hz`; the exponentials are composed analytically instead.
  - **1D without a Bessel-function reciprocal sum.** Only the monopole channel needs summing in
    1D, so this is a real-space sum plus an analytic tail: the `ρ²/(nL)²` and `z/(nL)` expansion
    of `1/√(ρ²+(z+nL)²)`, whose coefficients are Hurwitz zeta values. No special functions, and
    exactly differentiable — which is what makes forces and stress available.
  - Validated the same way 3D was: Madelung constants to 10–12 digits, independence of `α` and of
    the image count, and finite differences for every derivative.

- **k-point analytic Hessian at `q = 0`**, so second derivatives no longer depend on the Γ-point
  exchange taper. Matches finite differences to 1.7e-7 eV/Bohr² on a polar chain with a mesh, and
  the acoustic sum rule holds exactly. Three defects found on the way there, each invisible at Γ:
  the Coulomb and exchange factors were 2× too small on unordered pairs, the exchange derivative
  Fock was missing its `−T` mirror, and the resonance derivative Fock was missing a factor of ½.

- **Born effective charges `Z*` and the electronic dielectric tensor `ε_∞`**, sharing the phonon
  response solve so the two cannot drift apart. `Σ_a Z*_a = 0` to 1e-16; `ε_∞` is origin
  independent to 1.6e-15, which was measured rather than predicted — the module doc had predicted
  a dependence and was corrected to record the measurement.

- **LO–TO splitting.** `D(q) = D_analytic(q) + D_NA(q)` with the non-analytic term built from
  `Z*` and `ε_∞`. Exactly zero for a non-polar system.

  > **Corrected in 0.2.1.** The "127 cm⁻¹ shift and 1631 cm⁻¹ of direction dependence for a
  > polar one" measured here was taken on a **1D chain**, where the three-dimensional
  > `4π/(Ω q·ε∞·q)` kernel does not apply and `Ω` was silently a length. Those numbers are
  > artifacts. See the 0.2.1 entry.

- **DFPT at arbitrary `q`** — a CPSCF connecting `k` to `k+q`, so a phonon at any `q` no longer
  needs a commensurate supercell. Reproduces the `q = 0` Hessian to 4e-13 relative and a 2-fold
  supercell's frozen phonon to 3e-4. The identities that pin the phases down (`D(−q) = D(q)*`,
  continuity in `q`) are asserted too, because a wrong phase leaves the matrix Hermitian and the
  frequencies real — it does not announce itself.

- **Divide-and-conquer under periodic boundary conditions and with an analytic stress.** Γ with a
  minimum-image buffer, exact once the buffer reaches `L/2` (3.4e-10 eV); the stress matches a
  strain finite difference to 3.6e-8 eV/Bohr³.

- **Far-field monopole screening** (`multipole_cutoff`, opt-in, default off): pairs beyond the
  cutoff contribute through atomic monopoles instead of the full multipole block.

- **Divide-and-conquer SCF** for large molecules, restricted and unrestricted:
  - Disjoint cores by recursive spatial bisection, Yang partition weights, one common chemical
    potential shared across all subsystems (two, one per spin channel, when unrestricted).
  - The density is truncated explicitly at the buffer radius, which makes the Yang sum rule
    exact for every geometry — verified at `0.0` deviation — and makes the two-centre exchange
    exactly linear-scaling rather than approximately so.
  - Non-neutral systems; Mulliken charges conserve the formal charge to 1e-8 e.
  - Hellmann–Feynman gradient at the assembled density.
  - Scaling counters (`diagonalization_work`, `coulomb_work`, `exchange_work`,
    `retained_density_blocks`) returned on every result, so the cost claim is inspectable.
  - `docs/divide-conquer.md`.

- **ASE calculator** now reads `atoms.pbc` and `atoms.cell`, implements `stress`, and exposes
  `method`, `kpts`, `smearing`, the cutoffs, the SCF tolerances and `divide_conquer`. Parameters
  moved into `self.parameters`, so `todict()`, `set()` and restart work.

- `method=` on every Python entry point; new `pbc_point` and `divide_conquer` native functions;
  the GIL is released around every solver.

- `tests/test_ase_pbc_md.py` — real molecular dynamics as an acceptance test: NVE in 1D/2D/3D,
  Parrinello–Rahman NPT, NPT-Berendsen pressure response, NVT on partially periodic cells.

### Fixed

- **The `pip`-installed CLI was broken in three of its five modes.** `am1-rs energy` exited
  non-zero on a water molecule while every other test in the suite passed: `__main__.py` read keys
  the bindings never emitted — `total_ev`, `dipole_magnitude`, `iterations`, `max_gradient`,
  `forces`, `positions`, `steps` — and a missing dictionary key raises only at the moment it is
  printed, which no test reached. Found by installing the sdist into a clean environment and
  running the console script, which is now `tests/test_cli.py` and a CI job rather than something
  that happened to get tried. Every mode's output is diffed against the Rust CLI's, which is what
  the packaging has been claiming; four of the five needed fixing to make that true, including
  Rust's `{:.6e}` exponent format, which differs from Python's.
- **`iterations`, `unrestricted` and `dipole_magnitude` were missing from the molecular Python
  results** while the periodic ones carried them, so a caller could report how an SCF went for a
  crystal and not for a molecule. `gradient` and `optimize` now also return the SCF breakdown
  they already computed, instead of forcing a second SCF to get it. New `native.constants()`
  exports the model's unit conversions — deliberately MOPAC7's `ev = 27.21` rather than CODATA —
  so nothing on the Python side has to write them down and drift.
- **The Hessian bug.** `rotation_to_x_g` replaced live dual numbers with constants when an atom
  pair was antiparallel to the reference axis, zeroing the derivatives. The gradient was
  protected by symmetry; the **second** derivative was not, so transverse force constants of any
  molecule with a bond on that axis were wrong. Fixed by removing the local frame entirely: the
  integrals are now written in terms of the internuclear unit vector and the transverse
  projector `δ_ij − n_i n_j`, which is branch-free, exactly differentiable at every order, and
  faster. This was also a prerequisite for periodic boundary conditions, where axis-aligned
  lattice vectors would have hit the branch constantly.

- **Periodic stress unit conversion** at the Python boundary used `ANGSTROM_TO_BOHR^(d−1)`
  instead of `^d`, so every periodic stress reaching Python was 1.89× too small. Found by the
  new NPT acceptance test; the Rust tests could not see it because they never cross that
  boundary.

- The divide-and-conquer UHF initial guess did not preserve the electron count, which put the
  open-shell case in a different SCF basin (0.66 eV high). Now split in proportion to the α/β
  counts, matching the full SCF.

- CPHF non-convergence is now an error (`Am1Error::CphfNotConverged`) rather than a silently
  returned plausible Hessian.

- Documented accuracy claims corrected to the values the tests actually assert: the numerical
  Slater overlap agrees with the analytic kernel to ~1e-7 (`1s|1s`) and ~5e-4 (`2s|2s`), not
  1e-8. The CLI `gradient` help said eV/Å; it prints Hartree/Bohr.

- Be and B are parameterized for AM1 and were undocumented.

- **AM1-BCC perception**, several distinct defects:
  - Ring perception was a union-find spanning tree with fundamental cycles, so a fused system
    could return the 10-membered perimeter of naphthalene instead of its two 6-rings. Replaced
    by the smallest ring through each bond.
  - Aromaticity never looked at ring size at all — the detector returned only a boolean — so
    cycloheptatriene and macrocyclic lactones came out aromatic. Now ring size, a planarity test
    and a Hückel 4n+2 π count, which also correctly rejects cyclooctatetraene.
  - Sulfur could **never** be aromatic: `perceive_hybridization` returned `Sp3` for every
    sulfur, and the aromaticity test required `Sp2`, so the `| 16` in its element match was
    unreachable code. Thiophene is now aromatic, as are pyrrole and furan, each by its own π
    contribution.
  - The bond-order reference table held only six C/N/O pairs, so every C=S, P=O and S=O was
    perceived as a single bond and every thiocarbonyl, phosphate and sulfonyl group was
    mistyped. Extended to P and S pairs.
  - Bond types 6 and 9 — the symmetric delocalized groups (nitro, N-oxide; carboxylate,
    phosphate, sulfonate) — were never emitted, leaving 27 consequential parameters unreachable.
    Now selected by a chemical rule rather than a bond length. Measured: of the remaining 66
    unreachable entries, 26 are identically zero and 40 are byte-identical to the aromatic type,
    so nothing that can affect a charge is now unreachable.
  - An unparameterized element, or a typed bond with no tabulated parameter, returned raw
    Mulliken charges **in silence**. Both now appear in `BccResult::warnings`.

### Performance

- **8.2× faster** on the molecular path, from profiling rather than guessing: faer global
  parallelism enabled (it defaults to sequential with `default-features = false`), blocked
  parallel `matmul` through faer views, a parallel chunked Fock build, `C_occ·C_occᵀ` density
  formation, a single matrix product for the DIIS commutator, and flattened pair-integral
  storage.
- **CPHF solved by preconditioned conjugate gradient** instead of DIIS-accelerated Richardson.
  The CPHF equations are linear and their operator — the orbital Hessian — is symmetric and
  positive definite at a stable SCF solution, which is exactly what CG is for. Each application
  of that operator is a full Fock build, and those builds are **two thirds of an entire frequency
  calculation**, so the figure of merit is simply how many are needed: 6296 → 4931 for a 150-atom
  cluster. The convergence test is deliberately the same quantity the fixed-point solver used
  (the fixed point's step `‖U_{n+1} − U_n‖` *is* the preconditioned residual), so the tolerance
  did not have to be retuned and the two are directly comparable. If the operator turns out not
  to be positive definite along a search direction, the solve falls back to the fixed-point
  iteration rather than returning something meaningless.
- **`fock::build_g_matrix`** builds the two-electron matrix directly instead of assembling the
  full Fock matrix and subtracting `H_core` again — two wasted `nao²` passes per call, about
  9 GB of memory traffic over one Hessian. The CPHF also now uses a sequential pair loop
  (`fock::PairLoop`), because it already runs under rayon across the `3N` perturbations and an
  inner rayon pool was contending with the outer one for the same threads.
- Together: a 150-atom frequency calculation went **23.2 s → 15.4 s (1.51×)**, with the Rust
  suite unchanged.
- Opt-in phase timing with `AM1_TIMING=1`.
- **Fixed a blind spot in that timing.** `report` was called from inside `run_am1`, and reporting
  clears the accumulator — so profiling a *gradient* or a *Hessian* printed only the SCF phases,
  and the single most expensive phase of those commands was invisible in the profile meant to
  find it. Reporting now belongs to the top-level caller. The CPHF work above is what that
  immediately revealed.
- The ASE molecular path now runs **one** SCF per force call instead of two.
- The divide-and-conquer DIIS history is stored as **packed upper triangles** with a memory
  budget rather than as full matrices at a fixed depth. A depth-8 history of densities and
  residuals is 16 dense matrices — 1.2 GB at 1536 atoms, most of the peak footprint, and growing
  quadratically, which is the wrong shape for the one part of the code meant for large systems.
  Packing is exact (both matrices are symmetric) and halves it; the budget shortens the history
  instead of letting it grow without bound.
- **1.7× on the divide-and-conquer path** at 1029 atoms — 14.0–14.6 s down to 8.2–8.6 s, measured
  three times each on the same machine because a single pair of runs on this one differs by 70 %.
  The cost was in the DIIS, and it was invisible: the labelled phases summed to 8.8 s of a 16.1 s
  run, and the missing 7.3 s sat *between* the timers. `extrapolate` rebuilt the entire B matrix
  every iteration — all `n²` ordered pairs, both triangles — when every entry but the newest row
  is already known and cannot change. At 1029 atoms each packed residual is 16.9 MB, so that was
  2.2 GB of memory traffic per SCF iteration for numbers already computed. Now the new row is
  computed on `push` and cached; `residual_dot` is a flat vectorizable `2·(packed dot) −
  (diagonal dot)` instead of a nested row walk; `pack` copies contiguous row runs instead of
  walking `nao²` through a 2D index; and the extrapolated density is accumulated packed and
  expanded once. Iteration counts are unchanged, as they must be — the cached values are the same
  values.
- **The timing report says what it measures.** It sums *thread*-seconds, so a phase running on
  sixteen threads reports about sixteen times its wall clock. Read as wall clock it makes the
  best-parallelized phase look like the bottleneck: the CPHF Fock builds in a 102-atom frequency
  run report 39 s against a 4.8 s calculation. The header now states this, and the total is
  labelled `TOTAL (thread-seconds)`.

### Test infrastructure

- The scaling benchmark's water clusters are spaced at 4.0 Å and the generator now **asserts**
  that no intermolecular contact is shorter than 1.8 Å. An earlier version used 3.1 Å with
  pseudo-random molecular orientations, which put hydrogens 1.22–1.35 Å apart — over a hundred
  pairs inside 1.6 Å at the larger sizes. The symptom was a cliff rather than a warning: the SCF
  converged in 14 iterations at 192, 375 and 648 atoms and then failed outright at 1029, which
  reads as a large-system divide-and-conquer defect and was nothing of the kind. Ice gets away
  with 2.76 Å because its molecules are oriented; a random-orientation benchmark cannot.

### Known limitations

These are measured, not suspected. See `docs/scope.md`.

- **Ewald summation covers the monopole channel only.** In every dimensionality the `1/R` term is
  now summed exactly, but the `R⁻³` Klopman–Ohno correction and the higher multipoles are still a
  real-space cutoff on the lattice translation `|T|`. Consequences, all measured: a charged 3D
  cell converges to about 0.1 eV, limited by the logarithmically divergent `R⁻³` residual
  (0.10 eV per unit `ln r_c`); neutral cells still converge slowly in the residual channels
  (3e-4 eV between a 40 and a 640 Bohr cutoff).
- **A charged slab or chain needs a stated convention.** The energy of a charged 2D or 1D cell is
  not defined without one — the neutralizing sheet's position enters a slab's energy, and a
  charged line's potential diverges logarithmically. Both are refused by default with an error
  naming the convention enum, rather than answered under a convention the caller never chose.
- **NDDO exchange at Γ diverges** and is tapered by a quintic smoothstep at `exchange_cutoff`.
  This is a documented approximation, not a convergence parameter. k-point sampling makes the
  density matrix decay on its own and largely removes the dependence.
- **Divide-and-conquer makes the diagonalization linear, not the whole calculation.** The NDDO
  Coulomb sum stays `O(N²)`. Measured exponents from operation counters: diagonalization 1.15,
  exchange 1.06, retained density blocks 1.05, Coulomb 2.02; on 3D clusters to 2187 atoms the
  fitted `Σn³` exponent is 1.25 against 3 for a full diagonalization. In wall clock it crosses
  over around 200 atoms; the 768-atom speedup ranged 1.4–6.3× across runs, which is machine load
  rather than the algorithm — hence counters, not a stopwatch. The open-shell analytic stress is
  refused rather than approximated: it needs the spin-resolved pair virial.
- **The long-range monopole correction is not in the DFPT response.** Generalizing it to a
  `q`-point response needs a phased Ewald sum `Σ_T e^{iq·T}/|d+T|`, which is not implemented.

  > **Corrected in 0.2.1.** This entry read "applied only at `q = 0`". In 0.2.0 it was not
  > applied at `q = 0` either — `force_constants_at_q` omitted it at every `q`, while
  > `pbc_hessian` included it. The phased sum **is** implemented in 0.2.1 for 3D cells; see the
  > 0.2.1 notes, including why the element dropped turned out to be `k = 0` and not the `G = 0`
  > this entry's successor originally assumed.
- **`ε_∞` is a clamped-ion dipole response, not a Berry phase.** For a system where charge
  circulates around the periodic loop rather than responding locally, it is not the right
  quantity. Origin independence was measured (1.6e-15) rather than assumed.
- **SAM1 is not implemented.** It replaces the multipole expansion with scaled STO-3G integrals,
  so it is a different integral engine rather than a reparameterization and does not fit the
  shared code path.
- **AM1-BCC typing gaps** — 23 % of the bond parameters are unreachable, ring perception is not
  SSSR, aromaticity ignores ring size, and the bond-order table covers only C/N/O pairs. The
  correction values themselves are exact.

### Packaging

- PEP 639 licence metadata with `license-files`, so `THIRD_PARTY_NOTICES.md` and the retained
  third-party licences ship **inside the wheel** — required by clause 2 of the BSD-3-Clause
  covering the bundled PySEQM-derived parameters.
- `extension-module` split into its own Cargo feature, so `cargo test --features python` and the
  CLI link on Linux and macOS.
- `abi3-py311`, `requires-python = ">=3.11"` (3.9 is end-of-life), trove classifiers,
  `[project.urls]`, keywords, and `dynamic = ["version"]` — the version now has one source,
  `Cargo.toml`, read back through `importlib.metadata`.
- `rust-version = "1.75"`, crate metadata, and `docs.rs` configuration.
- **Wheels are built and published for every common platform** (`.github/workflows/release.yml`):
  manylinux and musllinux on x86_64 and aarch64, macOS on both architectures, Windows x64, plus
  the sdist, published to PyPI by trusted publishing on a version tag. `abi3-py311` means one
  wheel per platform covers 3.11 and up. This is the real defence against a failed install: a
  source install builds under the shipping profile — fat LTO, one codegen unit — which measures
  1.9 GB peak resident and over ten minutes on a warm dependency cache, and on a small VM that is
  an out-of-memory failure rather than a slow install.
- **The sdist is tested by installing it**, in CI, into a clean virtual environment, followed by
  running the console script. Three things can break a source install silently, and all three are
  now asserted: a file the *build* needs missing from the tarball (`[[bin]]` points at
  `src/bin/am1_rs.rs`, so cargo aborts without it even though no wheel ever contains that binary;
  PEP 639 `license-files` names `third_party/*/LICENSE`; the parameter CSVs are `include_str!`-ed),
  `Cargo.lock` absent so dependencies re-resolve forward on the user's machine, and the
  `extension-module` feature reaching a target that has to link Python's symbols. On that last
  one: maturin builds only the lib target, so the CLI binary is never linked during a `pip
  install` — verified against the build log rather than assumed, and now held by the test.

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
