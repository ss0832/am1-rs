// SPDX-License-Identifier: GPL-3.0-or-later

//! Self-consistent linear response: the dynamical matrix at **arbitrary** `q`.
//!
//! # The name, since AM1 has no density functional in it
//!
//! What is implemented is **periodic CPHF**: coupled-perturbed Hartree–Fock at a wavevector. The
//! module is called `dfpt` because that is the name the *method* goes by, and the name is a
//! pointer to the right technique rather than a claim about the reference:
//!
//! | | kernel `G(ΔP) = δF/δP · ΔP` | name |
//! |---|---|---|
//! | general | any self-consistent mean field | Sternheimer / self-consistent linear response |
//! | Kohn–Sham | Hartree + xc | DFPT, a.k.a. CPKS |
//! | Hartree–Fock | Coulomb + exact exchange | CPHF — this |
//!
//! **The equations are the same and only `G` differs.** The property this module is built on —
//! that a perturbation of wavevector `q` connects `k` to `k + q` and to nothing else — follows
//! from the Hamiltonian being a lattice-periodic mean field, and from nothing else. No functional
//! enters it. Nor is there a boundary to cross: a hybrid functional's DFPT already carries exact
//! exchange, and Hartree–Fock is that construction at `a_x = 1`.
//!
//! # What exact exchange costs, and what is NDDO's own
//!
//! Two of the three complications below are consequences of **exact exchange**, not of HF versus
//! DFT — a hybrid DFPT meets them too. Only the third belongs to this model:
//!
//! * **The kernel is non-local.** `G^σ(ΔP) = J(ΔP_tot) − K(ΔP_σ)` in `k` space is not the local
//!   multiply an LDA kernel is.
//! * **Exact exchange diverges at Γ.** `Σ_T |T|⁻¹` does not converge; see
//!   [`crate::hamiltonian::PairIntegral::exchange_scale`] for the taper standing in for the
//!   density-matrix decay that k-point sampling would supply.
//! * **The periodic Fock is affine, not linear**, in the density — and this one *is* NDDO's own.
//!   The long-range potential is built from the *net* charges `Q_a = Z_a − p_a`, so the nuclear
//!   part rides along and `G(ΔP) = F(ΔP) − H_core` is not a valid shortcut.
//!
//! # What no test here establishes
//!
//! Whether AM1 *should* be used for phonons. The parameters were fitted to molecular heats of
//! formation, geometries, dipoles and ionization potentials — not to solid-state vibrational
//! spectra. The tests establish that `D(q)` is the correct response **of this model** (against
//! finite differences, against a supercell, against the acoustic sum rule); they establish
//! nothing about agreement with a measured dispersion, and no test in this crate does.
//!
//! # What this is for
//!
//! [`crate::pbc::phonon`] gets `D(q)` by Fourier-transforming force constants read off a
//! supercell's Γ Hessian. That is exact — but only at the `q` the supercell can represent, and
//! reaching a finer `q` costs a larger supercell, which grows as the cube of the refinement.
//! DFPT computes the response at one `q` directly, at the cost of a primitive cell, for any `q`.
//!
//! # What makes it different from the `q = 0` solver
//!
//! Displace atom `b` in every cell `L` by `u e^{iq·L}`. That perturbation is not lattice
//! periodic, so it does not leave `k` alone: writing the cell-periodic part of the first-order
//! Hamiltonian as `h⁽¹⁾(T)`, translational covariance gives
//!
//! ```text
//! ⟨μ k'| H⁽¹⁾ |ν k⟩ = δ_{k', k+q} Σ_T e^{ik·T} h⁽¹⁾(T)
//! ```
//!
//! — the perturbation connects `k` to `k + q` and to nothing else. Three consequences, each of
//! which is a separate opportunity to be wrong, and each invisible at `q = 0`:
//!
//! **The phase depends on which block, not only on which atom moves.** A contribution lands in a
//! block whose *row* atom sits in cell 0, and carries `e^{iq·S}` with `S` the cell of the atom
//! being displaced **measured from that row atom's cell**. For a pair `(a in cell 0, b in cell
//! T)` that is four cases, not two: displacing `a` is unphased for the `a`-row blocks and carries
//! `e^{−iq·T}` for the `b`-row ones, and displacing `b` is the mirror.
//!
//! **The response kernel carries phases too.** `G(ΔP)` is not the real Fock builder run on the
//! real and imaginary parts separately: the Coulomb couplings pick up `e^{±iq·T}`, which mixes
//! them. (The exchange does not — it connects the `(0,T)` block, whose prefactor is `e^{iq·0}`.)
//!
//! **The response runs over every band pair, not occupied × virtual.** The occupation difference
//! `f_n(k) − f_m(k+q)` is what selects them, and it is nonzero in *both* directions: the
//! empty→occupied half is the antiresonant response of the bra at `k + q`. Dropping it halves the
//! orbital-relaxation term.
//!
//! # How the phases are pinned down
//!
//! Not by argument. At a `q` commensurate with an `n`-fold supercell, DFPT on the primitive cell
//! must reproduce that supercell's frozen phonon, having shared no code beyond the SCF; and at
//! `q = 0` it must collapse onto the already-validated `q = 0` Hessian. `tests/pbc_dfpt.rs` does
//! both. A wrong phase leaves the matrix Hermitian and the frequencies real, so nothing weaker
//! than an identity would catch it.

use crate::basis::Basis;
use crate::error::{Am1Error, Result};
use crate::lattice::ImageOffset;
use crate::linalg::Matrix;
use crate::neighbors::NeighborList;
use crate::params::Am1Parameters;
use crate::pbc::complex::{hermitian_eigen, CMatrix};
use crate::pbc::kpoints::KPoint;
use crate::pbc::scf::{run_pbc_scf, PbcOptions, RealSpaceBlocks};
use crate::system::Molecule;

/// Band pairs closer than this in energy have their response dropped — the rotation between two
/// degenerate states is not determined by the equations and does not change the density either.
const DEGENERACY_FLOOR: f64 = 1.0e-8;

/// Whether the long-range monopole term is included in the response.
///
/// The correction is the phased sum [`crate::pbc::ewald::phased_delta`] evaluates,
///
/// ```text
/// Δ(q; d) = Σ_T e^{iq·T} erfc(α|d+T|)/|d+T|
///         + (4π/V) Σ_{k ≠ 0} e^{−k²/4α²} e^{ik·d} / k²,        k = G − q
/// ```
///
/// wired into all three places it has to appear consistently: the fixed-charge second derivative,
/// the bare perturbation's per-atom diagonal channel, and the `−V_a(q)` shift in the
/// coupled-perturbed kernel. Leaving it out of any one of them would let the skeleton carry a
/// long-range term the response could not screen.
///
/// **Every dimensionality since 0.2.2.** The `(4π/V)` reciprocal half above is the 3D form; a slab
/// uses Parry's `(π/A)Σ_k e^{ik·ρ}K(|k|,z)/|k|` over the shifted in-plane set and a chain uses a
/// phased direct sum with an Abel-transformed tail. Through 0.2.1 only the 3D kernel existed, and
/// [`LongRange::Require`] on a chain or a slab was an error for that reason alone.
///
/// Because the sum keeps every `k ≠ 0`, `D(q)` is the **full** dynamical matrix at finite `q` and
/// its `q → 0` limit carries the long-range channel — which is the physics. It must therefore
/// **not** be combined with
/// [`crate::pbc::phonon::ForceConstants::frequencies_with_lo_to`], whose job is to restore that
/// same physics to the *supercell* route, where a truncated `Φ(T)` structurally cannot carry it.
///
/// # What `q → 0` does, by dimensionality
///
/// The kernel diverges in all three — `4π/(Vq²)`, `2π/(A|q|)`, `−(2/L)ln|q|` — but the *contribution
/// to `D(q)`* carries two factors of `q` from charge conservation, so:
///
/// | | `q²` × kernel | at Γ |
/// |---|---|---|
/// | 3D | finite, direction dependent | **discontinuous** — this is LO–TO |
/// | 2D | `O(|q|) → 0` | continuous, slope `∝|q|`: no splitting at Γ, a linear kink |
/// | 1D | `q² ln(1/q) → 0` | continuous, non-analytic at higher order |
///
/// Only 3D is discontinuous. See `docs/pbc.md`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LongRange {
    /// Include it wherever there is a lattice to sum over. The default.
    #[default]
    Auto,
    /// Require it. A molecule — which has no lattice sum at all — is an error rather than a quiet
    /// difference.
    Require,
    /// Leave it out even for a 3D cell — the state of this module before 0.2.1, kept so that the
    /// term's effect can be measured rather than argued.
    Off,
}

/// Controls for the `q`-point response, over and above the ground-state [`PbcOptions`].
///
/// # The k-set is the SCF's k-set
///
/// [`DfptOptions::kmesh`] and [`DfptOptions::kpoints`] change **which** Brillouin-zone sampling
/// the whole calculation uses, ground state included — they do not let the response be sampled
/// more finely than the density it is built on.
///
/// That restriction is not an oversight. The coupled-perturbed equations assume the zeroth-order
/// state satisfies the SCF condition. Diagonalizing a Fock matrix converged on a coarse mesh at
/// some finer `k` gives exact eigenpairs *of that matrix*, but not a self-consistent ground
/// state, so the response would be the response of a different functional and the frozen-phonon
/// identity would no longer hold exactly. Asking for a finer mesh therefore re-runs the SCF on it.
// `Default` is written out rather than derived: a derived one would give `cpscf_tol = 0.0` and
// `cpscf_max_iter = 0`, which is a solver that never converges and never iterates.
#[derive(Clone, Debug)]
pub struct DfptOptions {
    /// Brillouin-zone mesh for the whole calculation. `None` uses [`PbcOptions::kmesh`].
    pub kmesh: Option<crate::pbc::kpoints::KMesh>,
    /// An explicit k-point list instead of a mesh; weights must sum to 1. Mutually exclusive
    /// with [`DfptOptions::kmesh`].
    pub kpoints: Option<Vec<KPoint>>,
    /// Whether the long-range monopole (Ewald) term is included in the response.
    pub long_range: LongRange,
    /// Convergence tolerance on the response density's RMS change between iterations.
    pub cpscf_tol: f64,
    /// Iteration cap for the coupled-perturbed self-consistent solve.
    pub cpscf_max_iter: usize,
    /// Mixing fraction on the response density: `mix·new + (1−mix)·old`.
    pub cpscf_mixing: f64,
    /// Return the `(k, k+q)` first-order densities in [`DfptResult::response`].
    ///
    /// Off by default because it is `O(ndof · n_k · nao²)` — the largest array in the
    /// calculation, and one the force constants themselves do not need retained.
    pub keep_response: bool,
}

impl DfptOptions {
    /// Validate the combination, naming what is wrong rather than picking a winner.
    fn check(&self, molecule: &Molecule, q: KPoint) -> Result<()> {
        if self.kmesh.is_some() && self.kpoints.is_some() {
            return Err(Am1Error::InvalidInput(
                "give DFPT either `kmesh` or `kpoints`, not both".into(),
            ));
        }
        let Some(cell) = molecule.cell else {
            return Err(Am1Error::InvalidInput("DFPT needs a periodic cell".into()));
        };
        // A `q` component along a non-periodic axis has no meaning: every lattice translation has
        // a zero component there, so the phase `e^{iq·T}` never sees it and the request is a
        // misunderstanding rather than a harmless no-op.
        for axis in 0..3 {
            if !cell.periodic[axis] && q.fractional[axis] != 0.0 {
                return Err(Am1Error::InvalidInput(format!(
                    "q has a component {} along non-periodic axis {axis}; there is no dispersion \
                     in that direction and no lattice translation to carry the phase",
                    q.fractional[axis]
                )));
            }
        }
        // The long-range term exists in every dimensionality since 0.2.2, so the only cell it
        // cannot apply to is one with no lattice at all. This used to refuse a chain or a slab,
        // because only the 3D phased kernel existed.
        if self.long_range == LongRange::Require && cell.n_periodic() == 0 {
            return Err(Am1Error::InvalidInput(
                "`LongRange::Require` was asked for on a cell with no periodic direction. There is \
                 no lattice sum to correct, so there is no long-range monopole term to require. \
                 See docs/pbc.md."
                    .into(),
            ));
        }
        Ok(())
    }
}

impl Default for DfptOptions {
    /// The constants the solver used before they were exposed, with the long-range term on
    /// wherever it applies.
    fn default() -> Self {
        Self {
            kmesh: None,
            kpoints: None,
            long_range: LongRange::Auto,
            cpscf_tol: 1.0e-10,
            cpscf_max_iter: 200,
            cpscf_mixing: 0.7,
            keep_response: false,
        }
    }
}

#[inline]
fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

/// Complex real-space blocks: the cell-periodic part of a `q`-dependent quantity.
#[derive(Clone)]
struct ComplexBlocks {
    re: RealSpaceBlocks,
    im: RealSpaceBlocks,
}

/// One perturbation's bare Fock derivative, as a list of its nonzero entries.
///
/// # Why this is not `nao²` per translation
///
/// Displacing atom `a` changes the Hamiltonian in two places, and neither is dense. The
/// **short-range** part touches only blocks in which `a` appears — `(a,b)`, `(b,a)`, `(a,a)` and
/// `(b,b)` for the neighbours `b` of `a` — which is `O(1)` blocks of at most `4 × 4`. The
/// **long-range** monopole part adds a shift to the on-site diagonal of *every* atom, because
/// `∂Δ(q; R_b − R_a)/∂R_a` is nonzero for all `b`; that is `O(N)` entries, and it is the reason
/// the structure needs a per-atom channel rather than a neighbour list alone.
///
/// So `nnz` is `O(N)` per perturbation against `nao²` dense. Contracting `C(q)_{j,j'}` over every
/// pair of perturbations is then `ndof² · n_k · nnz = O(N³ n_k)` rather than
/// `ndof² · n_k · nao² = O(N⁴ n_k)`, which is the one genuine order reduction in the DFPT path.
///
/// Grouping by translation matters as much as the sparsity: the Bloch phase `e^{ik·T}` depends
/// only on `(k, T)`, so grouping lets it be computed once per group instead of once per entry.
/// One nonzero of a real-space block: row, column, and the complex value.
type BareEntry = (u32, u32, [f64; 2]);

struct SparseBare {
    /// One group per translation that has any nonzero entry; empty translations are omitted.
    groups: Vec<(ImageOffset, Vec<BareEntry>)>,
    nao: usize,
}

impl SparseBare {
    /// `Σ_T e^{ik·T} M(T)`, materialized densely.
    ///
    /// The CPSCF right-hand side needs the full matrix, so it is rebuilt here rather than stored.
    /// That is `O(n_k · nao²)` transiently per perturbation — and since the solver streams the
    /// perturbations, per *thread* rather than per degree of freedom.
    fn at_k(&self, k: &KPoint) -> CMatrix {
        let mut out = CMatrix::zeros(self.nao);
        for (t, entries) in &self.groups {
            let (c, s) = k.phase(*t);
            for &(i, j, v) in entries {
                let (i, j) = (i as usize, j as usize);
                out.re[(i, j)] += c * v[0] - s * v[1];
                out.im[(i, j)] += c * v[1] + s * v[0];
            }
        }
        out
    }

    fn nnz(&self) -> usize {
        self.groups.iter().map(|(_, e)| e.len()).sum()
    }
}

impl ComplexBlocks {
    fn zeros(translations: &[ImageOffset], nao: usize) -> Self {
        Self {
            re: RealSpaceBlocks::zeros(translations, nao),
            im: RealSpaceBlocks::zeros(translations, nao),
        }
    }

    /// Add `other` block by block, matching on translation. See
    /// [`crate::pbc::scf::RealSpaceBlocks::add_assign`].
    fn add_assign(&mut self, other: &Self) {
        self.re.add_assign(&other.re);
        self.im.add_assign(&other.im);
    }

    /// Accumulate into one block, silently ignoring a translation outside the stored set.
    #[inline]
    fn add(&mut self, t: ImageOffset, i: usize, j: usize, v: [f64; 2]) {
        if let Some(idx) = self.re.position(t) {
            self.re.blocks[idx][(i, j)] += v[0];
            self.im.blocks[idx][(i, j)] += v[1];
        }
    }

    #[inline]
    fn get(&self, t: ImageOffset) -> Option<(&Matrix, &Matrix)> {
        self.re
            .position(t)
            .map(|idx| (&self.re.blocks[idx], &self.im.blocks[idx]))
    }

    /// The `T = 0` block.
    ///
    /// A `Result` rather than an `expect`: "the origin is always present" is a property of how
    /// the translation set is built, not something this type enforces, and if it ever stopped
    /// holding a panic deep inside the response solve is the worst way to find out.
    fn onsite(&self) -> Result<(&Matrix, &Matrix)> {
        self.get(ImageOffset::origin()).ok_or_else(|| {
            Am1Error::InvalidInput(
                "the density's translation set is missing the origin block, which every \
                 real-space matrix must carry"
                    .into(),
            )
        })
    }

    /// The nonzero entries, grouped by translation.
    ///
    /// See [`SparseBare`] for why the bare perturbation is worth storing this way.
    fn sparse(&self) -> SparseBare {
        let nao = self.re.blocks[0].rows;
        let mut groups = Vec::new();
        for ((t, br), bi) in self
            .re
            .translations
            .iter()
            .zip(&self.re.blocks)
            .zip(&self.im.blocks)
        {
            let mut entries = Vec::new();
            for i in 0..nao {
                for j in 0..nao {
                    let (r, m) = (br[(i, j)], bi[(i, j)]);
                    if r != 0.0 || m != 0.0 {
                        entries.push((i as u32, j as u32, [r, m]));
                    }
                }
            }
            if !entries.is_empty() {
                entries.shrink_to_fit();
                groups.push((*t, entries));
            }
        }
        SparseBare { groups, nao }
    }

    /// `Σ_T e^{ik·T} M(T)`.
    fn at_k(&self, k: &KPoint) -> CMatrix {
        let nao = self.re.blocks[0].rows;
        let mut out = CMatrix::zeros(nao);
        for ((t, br), bi) in self
            .re
            .translations
            .iter()
            .zip(&self.re.blocks)
            .zip(&self.im.blocks)
        {
            let (c, s) = k.phase(*t);
            for i in 0..nao {
                for j in 0..nao {
                    let (r, m) = (br[(i, j)], bi[(i, j)]);
                    if r != 0.0 || m != 0.0 {
                        out.re[(i, j)] += c * r - s * m;
                        out.im[(i, j)] += c * m + s * r;
                    }
                }
            }
        }
        out
    }

    fn rms_diff(&self, other: &Self) -> f64 {
        let mut acc = 0.0;
        let mut n = 0usize;
        for (i, br) in self.re.blocks.iter().enumerate() {
            let bi = &self.im.blocks[i];
            let (or_, oi) = (&other.re.blocks[i], &other.im.blocks[i]);
            for (k, v) in br.as_slice().iter().enumerate() {
                let dr = v - or_.as_slice()[k];
                let di = bi.as_slice()[k] - oi.as_slice()[k];
                acc += dr * dr + di * di;
                n += 1;
            }
        }
        (acc / n.max(1) as f64).sqrt()
    }

    /// Every stored number as one flat vector — real blocks then imaginary — for the DIIS.
    fn as_flat(&self) -> Vec<f64> {
        let mut out =
            Vec::with_capacity(2 * self.re.blocks.len() * self.re.blocks[0].as_slice().len());
        for b in &self.re.blocks {
            out.extend_from_slice(b.as_slice());
        }
        for b in &self.im.blocks {
            out.extend_from_slice(b.as_slice());
        }
        out
    }

    /// The inverse of [`ComplexBlocks::as_flat`], writing into an existing set of blocks.
    fn set_from_flat(&mut self, v: &[f64]) {
        let mut k = 0;
        for b in self.re.blocks.iter_mut() {
            let n = b.as_slice().len();
            b.as_mut_slice().copy_from_slice(&v[k..k + n]);
            k += n;
        }
        for b in self.im.blocks.iter_mut() {
            let n = b.as_slice().len();
            b.as_mut_slice().copy_from_slice(&v[k..k + n]);
            k += n;
        }
    }

    fn mixed(&self, previous: &Self, mix: f64) -> Self {
        let mut out = self.clone();
        for (i, b) in out.re.blocks.iter_mut().enumerate() {
            for (k, v) in b.as_mut_slice().iter_mut().enumerate() {
                *v = mix * *v + (1.0 - mix) * previous.re.blocks[i].as_slice()[k];
            }
        }
        for (i, b) in out.im.blocks.iter_mut().enumerate() {
            for (k, v) in b.as_mut_slice().iter_mut().enumerate() {
                *v = mix * *v + (1.0 - mix) * previous.im.blocks[i].as_slice()[k];
            }
        }
        out
    }
}

/// One `k` of the response mesh, with its partner at `k + q`.
struct Band {
    k: KPoint,
    eps_k: Vec<f64>,
    eps_kq: Vec<f64>,
    /// Electrons per orbital, `0 … 2`.
    occ_k: Vec<f64>,
    occ_kq: Vec<f64>,
    c_k: CMatrix,
    c_kq: CMatrix,
}

/// Everything one `q`-point response produces.
///
/// The `(k, k+q)` first-order quantities used to be computed and discarded; they are the whole
/// content of the perturbation theory, and a caller wanting an electron–phonon matrix element or
/// a first-order wavefunction has no other way to get them.
#[derive(Clone, Debug)]
pub struct DfptResult {
    /// `C(q)_{aα,bβ} = Σ_T Φ(T) e^{iq·T}`, eV/Bohr², Hermitian.
    pub force_constants: CMatrix,
    /// The `q` this was evaluated at.
    pub q: KPoint,
    /// The k-set actually used, with weights.
    pub k_points: Vec<KPoint>,
    /// Band energies at `k` and at `k + q` (eV), one entry per k point.
    pub eigenvalues: Vec<(Vec<f64>, Vec<f64>)>,
    /// Occupations at `k` and at `k + q`, electrons per orbital (`0 … 2`).
    pub occupations: Vec<(Vec<f64>, Vec<f64>)>,
    /// First-order density `ΔP^j(k)` per degree of freedom per k point, rows at `k + q` and
    /// columns at `k`, in the AO basis.
    ///
    /// `None` unless it was asked for: this is `O(ndof · n_k · nao²)` and is the largest array in
    /// the calculation. The solver streams each perturbation — solve, contract, drop — so asking
    /// for this is what *creates* the array rather than merely retaining one that already
    /// existed. Resident response memory is otherwise `O(threads · n_k · nao²)`.
    pub response: Option<Vec<Vec<CMatrix>>>,
    /// Entries touched per `(j, j', k)` when contracting `C(q)`: the sparse count actually used,
    /// and the `nao²` a dense `h_j(k)` would force.
    ///
    /// Returned for the same reason `DcResult` returns its operation counters: the contraction's
    /// cost is a *claim about scaling*, and such a claim should be checkable from the result
    /// rather than believed. `bare_nonzeros` is set by the displaced atom's neighbourhood plus,
    /// on a 3D cell, the long-range monopole channel's per-atom diagonal; `nao²` grows as `N²`
    /// regardless. That difference is what makes assembling `C(q)` `O(N³ n_k)` instead of
    /// `O(N⁴ n_k)`.
    pub bare_nonzeros: usize,
    pub bare_dense_elements: usize,
}

/// The dynamical matrix at `q`, in eV/(Å²·amu) — the same units [`crate::pbc::phonon`] uses.
///
/// `q` is in **fractional** reciprocal coordinates, the same convention as the k-mesh.
pub fn dynamical_matrix_dfpt(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    q: KPoint,
) -> Result<CMatrix> {
    dynamical_matrix_dfpt_with(molecule, params, options, &DfptOptions::default(), q)
}

/// [`dynamical_matrix_dfpt`] with explicit response controls.
pub fn dynamical_matrix_dfpt_with(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    dfpt: &DfptOptions,
    q: KPoint,
) -> Result<CMatrix> {
    let c = force_constants_at_q_with(molecule, params, options, dfpt, q)?.force_constants;
    mass_weight(molecule, &c)
}

/// Divide a force-constant matrix by `√(m_a m_b)` and convert eV/Bohr² to eV/(Å²·amu).
fn mass_weight(molecule: &Molecule, c: &CMatrix) -> Result<CMatrix> {
    let nat = molecule.atoms.len();
    let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
    let mut d = CMatrix::zeros(3 * nat);
    for a in 0..nat {
        let ma = crate::data_tables::MASS[molecule.atoms[a].z as usize];
        for b in 0..nat {
            let mb = crate::data_tables::MASS[molecule.atoms[b].z as usize];
            let scale = a0_sq / (ma * mb).sqrt();
            for i in 0..3 {
                for j in 0..3 {
                    let (re, im) = c.get(3 * a + i, 3 * b + j);
                    d.add(3 * a + i, 3 * b + j, re * scale, im * scale);
                }
            }
        }
    }
    d.hermitianize();
    Ok(d)
}

/// Harmonic frequencies at `q`, cm⁻¹, ascending. Negative denotes an imaginary mode.
pub fn frequencies_dfpt(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    q: KPoint,
) -> Result<Vec<f64>> {
    frequencies_dfpt_with(molecule, params, options, &DfptOptions::default(), q)
}

/// [`frequencies_dfpt`] with explicit response controls.
pub fn frequencies_dfpt_with(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    dfpt: &DfptOptions,
    q: KPoint,
) -> Result<Vec<f64>> {
    let d = dynamical_matrix_dfpt_with(molecule, params, options, dfpt, q)?;
    frequencies_from_dynamical_matrix(&d)
}

/// `ν = ±√|λ|` in cm⁻¹, ascending; the sign records an imaginary mode.
pub fn frequencies_from_dynamical_matrix(d: &CMatrix) -> Result<Vec<f64>> {
    let eigen = hermitian_eigen(d)?;
    Ok(eigen
        .values
        .iter()
        .map(|&lambda| {
            if lambda >= 0.0 {
                crate::pbc::phonon::SQRT_EV_PER_ANG2_AMU_TO_CM * lambda.sqrt()
            } else {
                -crate::pbc::phonon::SQRT_EV_PER_ANG2_AMU_TO_CM * (-lambda).sqrt()
            }
        })
        .collect())
}

/// `C(q)_{aα,bβ} = Σ_T Φ_{aα,bβ}(T) e^{iq·T}`, in eV/Bohr².
pub fn force_constants_at_q(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    q: KPoint,
) -> Result<CMatrix> {
    Ok(
        force_constants_at_q_with(molecule, params, options, &DfptOptions::default(), q)?
            .force_constants,
    )
}

/// The full `q`-point response: force constants and the `(k, k+q)` first-order quantities.
///
/// # The contraction, written down
///
/// ```text
/// C(q)_{j,j'} = Skel_{j,j'} + Σ_k w_k Σ_{μν} conj( h_j(k)_{μν} ) · ΔP_{j'}(k)_{μν}
/// D(q)        = ½ ( C(q) + C(q)† )
/// h_j(k)      = Σ_T e^{ik·T} h_j(T)
/// ```
///
/// `h_j` is the **bare** perturbation, not the self-consistent one: that is the non-variational
/// form the 2n+1 theorem leaves once the two-electron double counting cancels. Contracting the
/// screened potential instead double-counts the kernel and gives a moderate, structure-preserving
/// error — the kind that survives every symmetry check. The prefactor is exactly 1; all the
/// occupancy lives in `ΔP` and the `±q` doubling lives in the band-pair loop.
pub fn force_constants_at_q_with(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    dfpt: &DfptOptions,
    q: KPoint,
) -> Result<DfptResult> {
    dfpt.check(molecule, q)?;
    if options.fold_time_reversal && dfpt.kpoints.is_none() {
        return Err(Am1Error::InvalidInput(
            "DFPT cannot use a time-reversal-folded mesh: folding merges k with −k, which is \
             exact for the ground state, but a q-point response relates k to k + q — a different \
             pairing — so the two halves of a folded pair are inequivalent. Set \
             `fold_time_reversal: false`."
                .into(),
        ));
    }
    let cell = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("DFPT needs a periodic cell".into()))?;
    // The k-set drives the ground state too, so that the zeroth order the response is built on
    // actually satisfies the SCF condition on the very points being sampled. Handing the resolved
    // list to the SCF — rather than a mesh description it would resolve again — is what makes
    // that identity rather than a convention two call sites have to keep agreeing about.
    let kpoints = response_kpoints(&cell, options, dfpt)?;
    let scf_options = PbcOptions {
        kpoints: Some(kpoints.clone()),
        fold_time_reversal: false,
        ..options.clone()
    };
    let options = &scf_options;
    let scf = run_pbc_scf(molecule, params, options)?;
    if !scf.converged {
        return Err(Am1Error::InvalidInput(
            "the periodic SCF did not converge; a DFPT response built on it would be meaningless"
                .into(),
        ));
    }

    let basis = Basis::build(molecule, params)?;
    let nao = basis.nao;
    let ndof = 3 * molecule.atoms.len();
    let neighbors = NeighborList::build(molecule, options.realspace_cutoff);
    let translations = cell.image_offsets(options.realspace_cutoff);

    let (core, pair_ints) = crate::pbc::scf::build_realspace_core(
        molecule,
        &basis,
        params,
        &neighbors,
        &translations,
        options.exchange_cutoff,
        options.electric_field,
    )?;

    // The **bare** perturbation and the fixed-density second derivative, in one pass over pairs.
    // The long-range monopole correction at this `q`, built once and shared by the skeleton, the
    // bare perturbation and the response kernel — the three places it has to appear consistently.
    // The `Off` check is here rather than inside the builder, so that the builder takes only what
    // it needs to build something and `DfptOptions` is not threaded through it to be compared
    // against a single variant.
    let long_range = if dfpt.long_range == LongRange::Off {
        None
    } else {
        LongRangeQ::build(
            molecule,
            params,
            &basis,
            &scf.density,
            &neighbors,
            options,
            q,
        )?
    };

    // The spin channels. One restricted, two unrestricted — the same split the `q = 0` response
    // uses, from the same function, so the two cannot disagree about it.
    let (channel_densities, fill) = crate::pbc::scf::spin_channel_densities(&scf);
    let channel_refs: Vec<&RealSpaceBlocks> = channel_densities.iter().collect();

    // One context, built once, shared by all four helpers — which is what guarantees they see the
    // same `long_range` object rather than three independently constructed ones.
    let ctx = DfptContext {
        molecule,
        params,
        basis: &basis,
        neighbors: &neighbors,
        pairs: &pair_ints,
        translations: &translations,
        options,
        long_range: long_range.as_ref(),
        nao,
        q,
        fill,
    };

    let (bare, skeleton) = bare_and_skeleton(&ctx, &scf.density, &channel_refs)?;
    let ex_scale = crate::pbc::scf::exchange_scale_for(fill);
    let counts: Vec<f64> = if channel_densities.len() == 2 {
        let (na, nb) = crate::pbc::hessian::spin_populations(molecule, params, options)?;
        vec![na, nb]
    } else {
        vec![crate::pbc::hessian::cell_electrons(
            molecule, params, options,
        )?]
    };
    let mut bands: Vec<Vec<Band>> = Vec::with_capacity(channel_densities.len());
    for (density, count) in channel_densities.iter().zip(&counts) {
        bands.push(solve_bands(
            &ctx,
            &core,
            &scf,
            SpinChannel {
                density,
                scale: ex_scale,
                fill,
                count: *count,
            },
            &kpoints,
        )?);
    }

    // The bare perturbations, compressed to their nonzero entries and the dense form dropped.
    //
    // `bare_and_skeleton` assembles densely because it scatters pair-major, but what is *kept*
    // from here on is `O(N)` per perturbation instead of `n_T · nao²` — see `SparseBare`.
    let bare: Vec<Vec<SparseBare>> = bare
        .iter()
        .map(|per_dof| per_dof.iter().take(ndof).map(|c| c.sparse()).collect())
        .collect();
    // The largest perturbation's count, not the first: the sparsity varies with how many
    // neighbours the displaced atom has, and a claim about cost should quote the worst case.
    let bare_nonzeros = bare
        .iter()
        .flat_map(|per_dof| per_dof.iter().map(|c| c.nnz()))
        .max()
        .unwrap_or(0);
    let bare_dense_elements = nao * nao;
    if crate::timing::enabled() {
        eprintln!(
            "  dfpt: bare perturbation {bare_nonzeros} nonzeros per DOF against \
             {bare_dense_elements} dense ({:.1}x)",
            bare_dense_elements as f64 / bare_nonzeros.max(1) as f64
        );
    }

    // `C(q)_{j,j'} = Skeleton + (1/N_k) Σ_k Tr[ h⁽¹⁾ʲ(k)† ΔPʲ'(k) ]`.
    //
    // The **bare** perturbation on the left, not the self-consistent one: this is the
    // non-variational form the 2n+1 theorem leaves once the two-electron double counting
    // cancels. Contracting the self-consistent potential instead double-counts the kernel and
    // gives a moderate, structure-preserving error — the kind that survives every symmetry check.
    //
    // The prefactor is exactly 1. All the occupancy lives in `ΔP` (occupations are electrons per
    // orbital, 0…2) and the ±q doubling lives in the band-pair loop.
    //
    // # Streamed, not accumulated
    //
    // Each `j'` is solved, contracted against every `j` immediately, and then **dropped**. The
    // alternative — solving all `3N` perturbations, keeping them, and contracting afterwards —
    // holds `ndof · n_k · nao²` complex numbers at once, which is `O(N³)` and was the largest
    // array in the calculation. Streaming makes the resident response `O(threads · n_k · nao²)`.
    // `keep_response` is the opt-in that restores the full array for a caller that wants it, and
    // it is off by default precisely because of that cost.
    //
    // # And contracted sparsely
    //
    // `h_j(k)` is held as its nonzero entries (see `SparseBare`), so this double loop costs
    // `ndof² · n_k · nnz` with `nnz = O(N)`, i.e. `O(N³ n_k)`, where contracting the dense form
    // was `ndof² · n_k · nao² = O(N⁴ n_k)`. The translation is the outer loop so the Bloch phase
    // is computed once per `(k, T)` rather than once per entry.
    use rayon::prelude::*;
    /// One perturbation's contribution: its column of `C(q)`, and the response itself when the
    /// caller asked to keep it.
    type Column = (Vec<[f64; 2]>, Option<Vec<CMatrix>>);
    let keep = dfpt.keep_response;
    let columns: Vec<Result<Column>> = (0..ndof)
        .into_par_iter()
        .map(|jp| {
            // One CPSCF per perturbation, solving **all** spin channels together — they are
            // coupled through the Coulomb half of the kernel.
            let column: Vec<&SparseBare> = bare.iter().map(|per_dof| &per_dof[jp]).collect();
            let delta = solve_response_channels(&ctx, &bands, &column, ndof, dfpt)?;
            let mut col = vec![[0.0_f64; 2]; ndof];
            for (j, acc) in col.iter_mut().enumerate() {
                // `Σ_σ Σ_k w_k Tr[h⁽¹⁾ʲσ(k)† ΔPʲ'σ(k)]`. Each channel contracts its **own** bare
                // perturbation against its own response: the two differ by the exchange, and
                // crossing them would contract `∂F^α/∂R` with `ΔP^β`.
                for (ci, per_dof) in bare.iter().enumerate() {
                    for (t, entries) in &per_dof[j].groups {
                        for (slot, band) in bands[ci].iter().enumerate() {
                            let (c, s) = band.k.phase(*t);
                            let w = band.k.weight;
                            let p = &delta[ci][slot];
                            for &(mu, nu, v) in entries {
                                // The Bloch-summed entry, then `conj(h) · ΔP`.
                                let hr = c * v[0] - s * v[1];
                                let hi = c * v[1] + s * v[0];
                                let (pr, pi) = p.get(mu as usize, nu as usize);
                                acc[0] += w * (hr * pr + hi * pi);
                                acc[1] += w * (hr * pi - hi * pr);
                            }
                        }
                    }
                }
            }
            // `keep_response` returns the **first** channel's response, which is the whole of it
            // restricted and the alpha half unrestricted. The consumers of this hook are
            // diagnostics for the restricted path.
            Ok((
                col,
                keep.then(|| delta.into_iter().next().unwrap_or_default()),
            ))
        })
        .collect();

    let mut total = skeleton;
    let mut response: Vec<Vec<CMatrix>> = Vec::with_capacity(if keep { ndof } else { 0 });
    for (jp, column) in columns.into_iter().enumerate() {
        let (col, kept) = column?;
        for (j, acc) in col.iter().enumerate() {
            let (re, im) = total.get(j, jp);
            total.re[(j, jp)] = re + acc[0];
            total.im[(j, jp)] = im + acc[1];
        }
        if let Some(d) = kept {
            response.push(d);
        }
    }

    // `C(q)` is Hermitian by construction; any deviation is numerical.
    let n = total.re.rows;
    let mut hermitian = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = total.get(i, j);
            let (br, bi) = total.get(j, i);
            hermitian.re[(i, j)] = 0.5 * (ar + br);
            hermitian.im[(i, j)] = 0.5 * (ai - bi);
        }
    }
    Ok(DfptResult {
        force_constants: hermitian,
        q,
        // The **first** channel's bands: the whole band structure restricted, the alpha half
        // unrestricted. These fields are reported for inspection, and a second set of them would
        // change the shape of a public struct for a diagnostic.
        eigenvalues: bands[0]
            .iter()
            .map(|b| (b.eps_k.clone(), b.eps_kq.clone()))
            .collect(),
        occupations: bands[0]
            .iter()
            .map(|b| (b.occ_k.clone(), b.occ_kq.clone()))
            .collect(),
        k_points: kpoints,
        response: dfpt.keep_response.then_some(response),
        bare_nonzeros,
        bare_dense_elements,
    })
}

/// The response k-set: whatever the caller asked for, resolved **unfolded**.
///
/// Unfolded deliberately. Time-reversal folding pairs `k` with `−k`, which is exact for the
/// ground state; a `q ≠ 0` response relates `k` to `k + q`, a different pairing entirely, and
/// folding would quietly average two inequivalent responses together. [`KMesh::resolve`] performs
/// no other symmetry reduction — [`crate::pbc::kpoints`]'s `fold` merges only `k` with `−k` — so
/// refusing time reversal is enough to make the set safe here.
///
/// This used to be a hand-rolled Γ-centred grid built from `kmesh.sizes()` alone. That dropped
/// the offset of a [`KMesh::MonkhorstPackShifted`] mesh, so the response was sampled on a
/// *different* set of k points from the ground state whose density it was built on, and it also
/// failed to collapse non-periodic axes — sampling a slab's surface normal `n` times over,
/// which is `n` times the diagonalizations for identical results.
fn response_kpoints(
    lattice: &crate::lattice::Lattice,
    options: &PbcOptions,
    dfpt: &DfptOptions,
) -> Result<Vec<KPoint>> {
    if let Some(points) = &dfpt.kpoints {
        if points.is_empty() {
            return Err(Am1Error::InvalidInput(
                "the explicit DFPT k-point list is empty".into(),
            ));
        }
        let total: f64 = points.iter().map(|k| k.weight).sum();
        if (total - 1.0).abs() > 1.0e-12 {
            return Err(Am1Error::InvalidInput(format!(
                "explicit DFPT k-point weights must sum to 1, they sum to {total}"
            )));
        }
        return Ok(points.clone());
    }
    let mesh = dfpt.kmesh.unwrap_or(options.kmesh);
    mesh.resolve(lattice, false)
}

/// Everything the `q`-point response needs that is fixed for the whole calculation.
///
/// The four helpers below each took eight to twelve of these positionally, and each carried an
/// `#[allow(clippy::too_many_arguments)]` to say so — the last of the four targets 0.2.1's plan
/// named and did not reach. Bundling them separates the setting from the problem, and it does one
/// thing a shorter argument list cannot: `long_range` has to be the *same* object in the skeleton,
/// the bare perturbation and the response kernel — the three places the correction must appear
/// consistently — and passing it once makes that structural instead of a convention.
struct DfptContext<'a> {
    molecule: &'a Molecule,
    params: &'a Am1Parameters,
    basis: &'a Basis,
    neighbors: &'a NeighborList,
    pairs: &'a crate::pbc::scf::PeriodicPairs,
    translations: &'a [ImageOffset],
    options: &'a PbcOptions,
    long_range: Option<&'a LongRangeQ>,
    nao: usize,
    q: KPoint,
    /// Electrons one orbital holds: `2` restricted, `1` per unrestricted channel. See
    /// [`crate::pbc::scf::spin_channel_densities`].
    fill: f64,
}

/// Ground-state bands at every `k` and its partner `k + q`, filled against one chemical potential.
/// One spin channel's ground state: the density its exchange contracts against, the exchange
/// weight, and how many of its electrons there are to place.
///
/// One of these restricted, two unrestricted. The split comes from
/// [`crate::pbc::scf::spin_channel_densities`], shared with the `q = 0` response so the two cannot
/// disagree about it.
#[derive(Clone, Copy)]
struct SpinChannel<'a> {
    density: &'a RealSpaceBlocks,
    /// `0.5` when this channel stands for both spins, `1.0` otherwise.
    scale: f64,
    /// Electrons one orbital holds: `2` restricted, `1` unrestricted.
    fill: f64,
    /// Electrons in this channel.
    count: f64,
}

fn solve_bands(
    ctx: &DfptContext<'_>,
    core: &RealSpaceBlocks,
    scf: &crate::pbc::scf::PbcResult,
    channel: SpinChannel<'_>,
    kpoints: &[KPoint],
) -> Result<Vec<Band>> {
    let (molecule, params, basis, pair_ints, options, q) = (
        ctx.molecule,
        ctx.params,
        ctx.basis,
        ctx.pairs,
        ctx.options,
        ctx.q,
    );
    // Through `long_range_delta`, so this carries the Klopman–Ohno tail. Building it without the
    // tail here diagonalised a Hamiltonian the SCF had not converged: the bands were eigenvectors
    // of a Fock differing from the ground-state one by the tail's diagonal shift, and `D(q = 0)`
    // then missed the q = 0 Hessian by 4e-5 eV/Bohr² where the two are meant to be the same number.
    //
    // `ctx.neighbors` rather than a fresh list: it is built from the same cutoff, so rebuilding it
    // per `solve_bands` call was pure repetition.
    let delta = crate::pbc::hessian::long_range_delta(molecule, params, ctx.neighbors, options)?;
    // Coulomb from the **total** density, exchange from this channel's own at its own weight —
    // the same `(total, spin, scale)` triple `build_realspace_fock` takes everywhere else.
    let fock = crate::pbc::scf::build_realspace_fock(
        core,
        pair_ints,
        scf.density.origin()?,
        channel.density,
        channel.scale,
        basis,
        molecule,
        params,
        delta.as_ref(),
    )?;

    let mut raw = Vec::with_capacity(kpoints.len());
    for k in kpoints {
        let kq = KPoint {
            fractional: [
                k.fractional[0] + q.fractional[0],
                k.fractional[1] + q.fractional[1],
                k.fractional[2] + q.fractional[2],
            ],
            weight: k.weight,
        };
        let a = hermitian_eigen(&fock.bloch_sum(k))?;
        let b = hermitian_eigen(&fock.bloch_sum(&kq))?;
        raw.push((*k, a, b));
    }

    // One chemical potential over the union of the `k` set and the `k+q` set, each at half
    // weight. Filling the two meshes independently is a different answer whenever a band crosses
    // the Fermi level, and the response is precisely where that matters.
    //
    // The two spin channels are **not** filled against a shared potential: the multiplicity fixes
    // `n_α` and `n_β` separately, which is the convention the periodic SCF uses.
    //
    // Each level carries this point's own weight — taking it from the k point rather than from
    // `1/n_k` is what makes a non-uniform set (a shifted mesh with collapsed axes, or an explicit
    // list) come out with the right electron count. The `k` set and the `k+q` set each contribute
    // one copy of the same physical band structure, so the target passed to `fill` is **twice**
    // the channel's electron count divided by what one orbital holds: that leaves
    // `count / fill` orbitals filled per copy, each holding `fill`, which is `count` electrons.
    let mut levels = Vec::with_capacity(2 * raw.len() * basis.nao);
    for (k, a, _) in &raw {
        for e in &a.values {
            levels.push(crate::fermi::Level {
                energy: *e,
                weight: k.weight,
            });
        }
    }
    for (k, _, b) in &raw {
        for e in &b.values {
            levels.push(crate::fermi::Level {
                energy: *e,
                weight: k.weight,
            });
        }
    }
    let filling = if options.smearing_ev > 0.0 {
        crate::fermi::Filling::Fermi {
            kt: options.smearing_ev,
        }
    } else {
        crate::fermi::Filling::Aufbau
    };
    let filled = crate::fermi::fill(&levels, 2.0 * channel.count / channel.fill, filling)?;

    let nao = basis.nao;
    let n_k = raw.len();
    let mut bands = Vec::with_capacity(n_k);
    for (slot, (k, a, b)) in raw.into_iter().enumerate() {
        // Electrons per filled orbital, matching the normalization of this channel's `P`.
        let occ_k = (0..nao)
            .map(|i| channel.fill * filled.fractions[slot * nao + i])
            .collect();
        let occ_kq = (0..nao)
            .map(|i| channel.fill * filled.fractions[(n_k + slot) * nao + i])
            .collect();
        bands.push(Band {
            k,
            eps_k: a.values,
            eps_kq: b.values,
            occ_k,
            occ_kq,
            c_k: CMatrix {
                n: nao,
                re: a.vectors_re,
                im: a.vectors_im,
            },
            c_kq: CMatrix {
                n: nao,
                re: b.vectors_re,
                im: b.vectors_im,
            },
        });
    }
    Ok(bands)
}
/// The long-range monopole correction, evaluated once per `q` for every atom pair.
///
/// # The three places it enters, and the one formula behind them
///
/// The long-range energy is `E = ½ Σ_{a,b,T} Q_a Q_b φ(R_b + T − R_a)` with `φ` the part of the
/// `1/R` lattice sum the truncated pair list did not already count. Differentiating it twice and
/// Fourier transforming gives, for the force constants,
///
/// ```text
/// C(q)_{a,b} = δ_ab [ Q_a Σ_c Q_c Δ''(0; d_ac) ]  −  Q_a Q_b Δ''(q; d_ab)
/// ```
///
/// The `δ_ab` term carries **no** phase — both its indices sit in the home cell — and it is what
/// makes the acoustic sum rule come out: summing over `b` at `q = 0` cancels the two terms
/// exactly. Getting that term phased, or omitting it, leaves a matrix that is still Hermitian
/// with real frequencies and a broken sum rule.
///
/// The first derivative has the same shape and supplies the bare perturbation's per-atom diagonal
/// channel, and the value `Δ(q; d_ab)` supplies the response kernel's charge-rearrangement term.
struct LongRangeQ {
    charges: Vec<f64>,
    nat: usize,
    /// `Δ(q; d_ab)`, complex — for the response kernel.
    value_q: Vec<[f64; 2]>,
    /// `∂Δ(q; d_ab)/∂d`, complex.
    grad_q: Vec<[[f64; 2]; 3]>,
    /// `∂Δ(0; d_ab)/∂d`, real — the unphased partner the `δ_ab` term needs.
    grad_0: Vec<[f64; 3]>,
    /// `∂²Δ(q; d_ab)/∂d²`, complex.
    hess_q: Vec<[[[f64; 2]; 3]; 3]>,
    /// `∂²Δ(0; d_ab)/∂d²`, real.
    hess_0: Vec<[[f64; 3]; 3]>,
}

impl LongRangeQ {
    #[inline]
    fn at(&self, a: usize, b: usize) -> usize {
        a * self.nat + b
    }

    /// Build it, or `None` when the correction does not apply.
    fn build(
        molecule: &Molecule,
        params: &Am1Parameters,
        basis: &Basis,
        density: &RealSpaceBlocks,
        neighbors: &NeighborList,
        options: &PbcOptions,
        q: KPoint,
    ) -> Result<Option<Self>> {
        let cell = molecule
            .cell
            .ok_or_else(|| Am1Error::InvalidInput("DFPT needs a periodic cell".into()))?;
        // Every dimensionality since 0.2.2. `LongRangeKernel::for_lattice` dispatches to the 3D
        // Ewald sum, the 2D Parry sum or the 1D chain sum, and each now has a phased counterpart —
        // which is what this used to be missing, and the only reason a slab or a chain was refused.
        let Some(ewald) = crate::pbc::ewald::LongRangeKernel::for_lattice(&cell)? else {
            return Ok(None); // a molecule: no lattice sum to correct
        };
        let origin = density
            .get(ImageOffset::origin())
            .ok_or_else(|| Am1Error::InvalidInput("density is missing the origin block".into()))?;
        let charges = crate::pbc::ewald::net_charges(molecule, basis, params, origin)?;

        let nat = molecule.atoms.len();
        let q_cart = crate::pbc::ewald::q_cartesian(&cell, q.fractional);
        let translations = &neighbors.translations;

        let mut me = Self {
            charges,
            nat,
            value_q: vec![[0.0; 2]; nat * nat],
            grad_q: vec![[[0.0; 2]; 3]; nat * nat],
            grad_0: vec![[0.0; 3]; nat * nat],
            hess_q: vec![[[[0.0; 2]; 3]; 3]; nat * nat],
            hess_0: vec![[[0.0; 3]; 3]; nat * nat],
        };

        use rayon::prelude::*;
        type Row = (
            Vec<[f64; 2]>,
            Vec<[[f64; 2]; 3]>,
            Vec<[f64; 3]>,
            Vec<[[[f64; 2]; 3]; 3]>,
            Vec<[[f64; 3]; 3]>,
        );
        let rows: Vec<Row> = (0..nat)
            .into_par_iter()
            .map(|a| {
                let mut vq = vec![[0.0; 2]; nat];
                let mut gq = vec![[[0.0; 2]; 3]; nat];
                let mut g0 = vec![[0.0; 3]; nat];
                let mut hq = vec![[[[0.0; 2]; 3]; 3]; nat];
                let mut h0 = vec![[[0.0; 3]; 3]; nat];
                for b in 0..nat {
                    let d = molecule.atoms[b].position - molecule.atoms[a].position;
                    let is_self = a == b;
                    let pq = crate::pbc::ewald::phased_delta(
                        q_cart,
                        d,
                        &cell,
                        translations,
                        &ewald,
                        is_self,
                    );
                    let p0 = crate::pbc::ewald::phased_delta(
                        crate::math::Vec3::zero(),
                        d,
                        &cell,
                        translations,
                        &ewald,
                        is_self,
                    );
                    vq[b] = pq.value;
                    gq[b] = pq.gradient;
                    hq[b] = pq.hessian;
                    for i in 0..3 {
                        g0[b][i] = p0.gradient[i][0];
                        for j in 0..3 {
                            h0[b][i][j] = p0.hessian[i][j][0];
                        }
                    }
                }
                (vq, gq, g0, hq, h0)
            })
            .collect();

        for (a, (vq, gq, g0, hq, h0)) in rows.into_iter().enumerate() {
            for b in 0..nat {
                let idx = a * nat + b;
                me.value_q[idx] = vq[b];
                me.grad_q[idx] = gq[b];
                me.grad_0[idx] = g0[b];
                me.hess_q[idx] = hq[b];
                me.hess_0[idx] = h0[b];
            }
        }

        // The Klopman–Ohno tail, on the **value** only: it is a per-pair constant, so it enters the
        // response kernel and neither derivative (see `klopman_ohno_tail_matrix`, which also derives
        // why the same constant is right at every `q`).
        //
        // Not optional. The ground state is converged with the tail in its Fock, so a response
        // kernel built without it is the response of a different Hamiltonian — and at `q = 0` that
        // showed up directly, as `D(0)` missing the `q = 0` Hessian by 4.6e-4 eV/Bohr².
        if options.klopman_ohno_tail {
            if let Some(ko) = crate::pbc::ewald::klopman_ohno_tail_matrix(
                molecule,
                params,
                options.realspace_cutoff,
            )? {
                for a in 0..nat {
                    for b in 0..nat {
                        me.value_q[a * nat + b][0] += ko.value(a, b);
                    }
                }
            }
        }
        Ok(Some(me))
    }
}

/// DIIS depth for the coupled-perturbed response.
const CPSCF_DIIS_DEPTH: usize = 10;

/// Pulay coefficients for a set of flattened fixed-point residuals.
///
/// The response iteration is a linearly mixed fixed point, and on a dense polar cell it converges
/// slowly enough to hit the iteration cap: a water crystal needed more than 200 passes to reach
/// `1e-10`, stalling at `5.6e-9`. Each of those passes is a real-space two-electron build plus a
/// diagonalization per k point, so the count is the whole cost.
///
/// Normalised and ridged for the same reason [`crate::hessian`]'s CPHF DIIS is: the residuals
/// span many orders of magnitude over the solve, and an unscaled `B` matrix goes numerically
/// singular long before the history is actually redundant — at which point the pivot guard
/// (correctly) refuses it and the acceleration silently stops happening.
fn cpscf_diis_coeffs(residuals: &[Vec<f64>]) -> Option<Vec<f64>> {
    let refs: Vec<&[f64]> = residuals.iter().map(|r| r.as_slice()).collect();
    crate::pbc::scf::pulay_coefficients(&refs)
}

/// Solve the self-consistent response, returning `ΔP(k)` per perturbation per k point.
/// Solve the CPSCF at this `q` for one perturbation, over **all** spin channels together.
///
/// The channels are coupled: `G^σ(ΔP) = J(ΔP_tot) − K(ΔP_σ)` reads the total response density in
/// its Coulomb half, so α sees β's response. Solving them one at a time would drop `J(ΔP_β)` from
/// `G^α` and converge to a plausible wrong answer.
///
/// Returns one `Vec<CMatrix>` per channel, each indexed by k.
fn solve_response_channels(
    ctx: &DfptContext<'_>,
    channels: &[Vec<Band>],
    column: &[&SparseBare],
    ndof: usize,
    dfpt: &DfptOptions,
) -> Result<Vec<Vec<CMatrix>>> {
    let (translations, nao) = (ctx.translations, ctx.nao);
    let scale = crate::pbc::scf::exchange_scale_for(ctx.fill);

    // `h_j(k)` does not change between iterations, so it is built once outside the loop rather
    // than re-Bloch-summed every pass. The bare perturbation is the **same** for both channels —
    // it is `∂F^σ/∂R`, and its spin dependence is already inside `column`, which is built per
    // channel by the caller.
    let bare_at_k: Vec<Vec<CMatrix>> = channels
        .iter()
        .enumerate()
        .map(|(ci, bands)| bands.iter().map(|b| column[ci].at_k(&b.k)).collect())
        .collect();

    let mut delta_p: Vec<ComplexBlocks> = channels
        .iter()
        .map(|_| ComplexBlocks::zeros(translations, nao))
        .collect();
    let mut per_k: Vec<Vec<CMatrix>> = channels
        .iter()
        .map(|bands| bands.iter().map(|_| CMatrix::zeros(nao)).collect())
        .collect();
    let mut converged = false;
    let mut last_residual = f64::INFINITY;
    let mut trials: Vec<Vec<f64>> = Vec::with_capacity(CPSCF_DIIS_DEPTH);
    let mut resid_history: Vec<Vec<f64>> = Vec::with_capacity(CPSCF_DIIS_DEPTH);

    for iteration in 0..dfpt.cpscf_max_iter {
        // The screened potential: bare plus the kernel's response to the current `ΔP`.
        // Iteration 0 is the uncoupled (independent-particle) response.
        let kernels: Option<Vec<ComplexBlocks>> = if iteration == 0 {
            None
        } else {
            let mut total = delta_p[0].clone();
            for extra in &delta_p[1..] {
                total.add_assign(extra);
            }
            let mut out = Vec::with_capacity(channels.len());
            for spin in &delta_p {
                out.push(fock_response_q(ctx, &total, spin, scale)?);
            }
            Some(out)
        };

        let mut next: Vec<ComplexBlocks> = channels
            .iter()
            .map(|_| ComplexBlocks::zeros(translations, nao))
            .collect();
        for (ci, bands) in channels.iter().enumerate() {
            for (slot, band) in bands.iter().enumerate() {
                let mut potential = bare_at_k[ci][slot].clone();
                if let Some(k) = &kernels {
                    let extra = k[ci].at_k(&band.k);
                    for i in 0..nao {
                        for j in 0..nao {
                            let (ar, ai) = potential.get(i, j);
                            let (br, bi) = extra.get(i, j);
                            potential.re[(i, j)] = ar + br;
                            potential.im[(i, j)] = ai + bi;
                        }
                    }
                }
                // Band basis: rows at `k + q`, columns at `k`.
                let projected = adjoint_mul(&band.c_kq, &mul(&potential, &band.c_k));

                // Every band pair, not occupied × virtual. `f_n(k) − f_m(k+q)` selects them, and
                // it is nonzero in both directions — the empty→occupied half is the antiresonant
                // response of the bra at `k + q`, and dropping it halves the answer.
                let mut response = CMatrix::zeros(nao);
                for m in 0..nao {
                    let f_m = band.occ_kq[m];
                    for n in 0..nao {
                        let df = band.occ_k[n] - f_m;
                        if df == 0.0 {
                            continue;
                        }
                        let de = band.eps_k[n] - band.eps_kq[m];
                        if de.abs() < DEGENERACY_FLOOR {
                            continue;
                        }
                        let (re, im) = projected.get(m, n);
                        response.re[(m, n)] = df * re / de;
                        response.im[(m, n)] = df * im / de;
                    }
                }
                let ao = mul(&band.c_kq, &mul_adjoint(&response, &band.c_k));
                per_k[ci][slot] = ao.clone();

                // Back to real space for the kernel: `Δp(T) = Σ_k w_k e^{−ik·T} ΔP(k)`.
                let w = band.k.weight;
                for t in translations {
                    let (c, s) = band.k.phase(*t);
                    for i in 0..nao {
                        for j in 0..nao {
                            let (xr, xi) = ao.get(i, j);
                            // e^{−ik·T}(xr + i xi)
                            next[ci].add(*t, i, j, [w * (xr * c + xi * s), w * (xi * c - xr * s)]);
                        }
                    }
                }
            }
        }

        let residual = next
            .iter()
            .zip(&delta_p)
            .map(|(a, b)| a.rms_diff(b))
            .fold(0.0_f64, f64::max);
        last_residual = residual;
        if iteration > 0 && residual < dfpt.cpscf_tol {
            converged = true;
            // `delta_p` is already the converged answer to within the tolerance; the `per_k`
            // blocks were built from it, and mixing again would move them apart.
            break;
        }

        // Pulay extrapolation on the fixed-point residual, falling back to linear mixing
        // whenever the history is redundant — which it is on the first pass, and again whenever
        // the residuals go linearly dependent near convergence.
        //
        // The channels are extrapolated **together**, on one concatenated vector: they are
        // coupled, so a separate history per channel would let them take inconsistent steps.
        let trial: Vec<f64> = next.iter().flat_map(|b| b.as_flat()).collect();
        let previous: Vec<f64> = delta_p.iter().flat_map(|b| b.as_flat()).collect();
        let residual_vec: Vec<f64> = trial.iter().zip(&previous).map(|(a, b)| a - b).collect();
        if trials.len() == CPSCF_DIIS_DEPTH {
            trials.remove(0);
            resid_history.remove(0);
        }
        trials.push(trial);
        resid_history.push(residual_vec);

        match cpscf_diis_coeffs(&resid_history) {
            Some(c) => {
                let mut combined = vec![0.0; trials[0].len()];
                for (i, ci) in c.iter().take(trials.len()).enumerate() {
                    for (out, v) in combined.iter_mut().zip(&trials[i]) {
                        *out += ci * v;
                    }
                }
                let stride = combined.len() / delta_p.len();
                for (ci, block) in delta_p.iter_mut().enumerate() {
                    block.set_from_flat(&combined[ci * stride..(ci + 1) * stride]);
                }
            }
            None => {
                for (ci, block) in delta_p.iter_mut().enumerate() {
                    *block = next[ci].mixed(block, dfpt.cpscf_mixing);
                }
            }
        }
    }
    if !converged {
        // The residual the solve actually reached, not `NaN`: a caller that cannot see how close
        // it got cannot tell a stiff system from a broken one.
        return Err(Am1Error::CphfNotConverged {
            perturbations: ndof,
            iterations: dfpt.cpscf_max_iter,
            residual: last_residual,
        });
    }
    Ok(per_k)
}
// The three complex products the CPSCF transforms the potential and the response with.
//
// Each is four real matrix products, handed to the blocked/SIMD kernel rather than written as a
// scalar `n³` triple loop through `CMatrix::get`. They run four times per k point per CPSCF
// iteration, on the innermost path in this module, and the loop form was leaving most of the
// machine unused: the same rewrite applied to `pbc::hessian`'s `project_ov` measured 29x at
// `nao = 32` and 362x at `nao = 128`.
//
// `matmul_seq`, not `matmul`: `solve_response` runs under a rayon `par_iter` over the `3N`
// perturbations, and faer's own threads would contend with that pool for the same workers.
//
// Complex arithmetic out of real blocks, with the conjugations written out so the signs are
// checkable against the definitions:
//
// ```text
// A  B  = (Ar Br - Ai Bi) + i(Ar Bi + Ai Br)
// A† B  = (ArᵀBr + AiᵀBi) + i(ArᵀBi - AiᵀBr)      A† = (Ar - iAi)ᵀ
// A  B† = (Ar Brᵀ + Ai Biᵀ) + i(Ai Brᵀ - Ar Biᵀ)  B† = (Br - iBi)ᵀ
// ```

/// `A B`.
fn mul(a: &CMatrix, b: &CMatrix) -> CMatrix {
    let mut re = a.re.matmul_seq(&b.re);
    a.im.matmul_acc_seq(&b.im, &mut re, -1.0);
    let mut im = a.re.matmul_seq(&b.im);
    a.im.matmul_acc_seq(&b.re, &mut im, 1.0);
    CMatrix { n: re.rows, re, im }
}

/// `A† B`.
fn adjoint_mul(a: &CMatrix, b: &CMatrix) -> CMatrix {
    let mut re = a.re.transpose_matmul_seq(&b.re);
    a.im.transpose_matmul_acc_seq(&b.im, &mut re, 1.0);
    let mut im = a.re.transpose_matmul_seq(&b.im);
    a.im.transpose_matmul_acc_seq(&b.re, &mut im, -1.0);
    CMatrix { n: re.rows, re, im }
}

/// `A B†`.
fn mul_adjoint(a: &CMatrix, b: &CMatrix) -> CMatrix {
    let mut re = a.re.matmul_transpose_seq(&b.re);
    a.im.matmul_transpose_acc_seq(&b.im, &mut re, 1.0);
    let mut im = a.im.matmul_transpose_seq(&b.re);
    a.re.matmul_transpose_acc_seq(&b.im, &mut im, -1.0);
    CMatrix { n: re.rows, re, im }
}

/// The two-electron response of the Fock to a density change at wavevector `q`.
///
/// The linear kernel `K[Δp]` — no core Hamiltonian, nothing that does not move with the density.
/// It mirrors [`crate::pbc::scf::build_realspace_fock`] term for term, and the only additions are
/// the `e^{±iq·T}` factors on the **Coulomb** couplings.
///
/// Those factors are why this cannot be the real Fock builder run twice, once on `Re Δp` and once
/// on `Im Δp`: multiplying by `e^{iq·T}` mixes the two parts, so a real builder can only be
/// correct at `q = 0`. The exchange takes no phase — it connects the `(0,T)` block, whose
/// perturbation prefactor is `e^{iq·0}`.
fn fock_response_q(
    ctx: &DfptContext<'_>,
    delta_p: &ComplexBlocks,
    spin: &ComplexBlocks,
    scale: f64,
) -> Result<ComplexBlocks> {
    let (molecule, basis, params, pairs, translations, long_range, nao, q) = (
        ctx.molecule,
        ctx.basis,
        ctx.params,
        ctx.pairs,
        ctx.translations,
        ctx.long_range,
        ctx.nao,
        ctx.q,
    );
    let mut out = ComplexBlocks::zeros(translations, nao);
    let (p0r, p0i) = delta_p.onsite()?;
    let (s0r, s0i) = spin.onsite()?;

    // The long-range correction's response to the charges rearranging: `V_a` shifts by
    // `Σ_b Δ_ab(q) ΔQ_b`, and the Fock carries `−V_a`. `ΔQ_b = −Δp_b` is the population response
    // read off the origin block, exactly as the `q = 0` path reads it.
    //
    // Without this the skeleton would carry a long-range term the response could not screen, and
    // the two halves of the Hessian would describe different Hamiltonians.
    if let Some(lr) = long_range {
        let nat = molecule.atoms.len();
        let dq: Vec<[f64; 2]> = (0..nat)
            .map(|b| {
                let off = basis.atom_offset[b];
                let mut acc = [0.0_f64; 2];
                for k in 0..basis.atom_norb[b] {
                    acc[0] -= p0r[(off + k, off + k)];
                    acc[1] -= p0i[(off + k, off + k)];
                }
                acc
            })
            .collect();
        for a in 0..nat {
            let mut v = [0.0_f64; 2];
            for (b, dqb) in dq.iter().enumerate() {
                let d = lr.value_q[lr.at(a, b)];
                v[0] += d[0] * dqb[0] - d[1] * dqb[1];
                v[1] += d[0] * dqb[1] + d[1] * dqb[0];
            }
            let off = basis.atom_offset[a];
            for k in 0..basis.atom_norb[a] {
                out.add(ImageOffset::origin(), off + k, off + k, [-v[0], -v[1]]);
            }
        }
    }

    // One-centre: local, so no phase, and the kernel is linear — the real and imaginary parts go
    // through the same expression independently.
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let (gss, gsp, gpp, gp2, hsp) = (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = [0.0_f64; 2];
                for la in 0..n {
                    for si in 0..n {
                        let coul =
                            crate::fock::oc_two_electron(mu, nu, la, si, gss, gsp, gpp, gp2, hsp);
                        let exch =
                            crate::fock::oc_two_electron(mu, la, nu, si, gss, gsp, gpp, gp2, hsp);
                        // Coulomb reads the **total** response density, exchange this channel's
                        // own at its own weight. Restricted, one channel carries both spins with
                        // `scale = ½` and the two densities are the same object.
                        acc[0] += p0r[(off + la, off + si)] * coul
                            - s0r[(off + la, off + si)] * scale * exch;
                        acc[1] += p0i[(off + la, off + si)] * coul
                            - s0i[(off + la, off + si)] * scale * exch;
                    }
                }
                out.add(ImageOffset::origin(), off + mu, off + nu, acc);
            }
        }
    }

    for pair in &pairs.pairs {
        let te = &pair.te;
        let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
        let (na, nb) = (te.norb_i, te.norb_j);
        let t = pair.offset;
        let neg_t = t.negated();
        let angle = q.phase(t);
        let phase = [angle.0, angle.1];
        let conj = [phase[0], -phase[1]];

        // Coulomb: `A`'s on-site block sees `B`'s density in cell `T` (phase `e^{+iq·T}`), and
        // `B`'s sees `A`'s in cell `−T` (phase `e^{−iq·T}`).
        for mu in 0..na {
            for nu in 0..na {
                let mut acc = [0.0_f64; 2];
                for la in 0..nb {
                    for si in 0..nb {
                        let w = te.two_e(mu, nu, la, si);
                        acc[0] += p0r[(ob + la, ob + si)] * w;
                        acc[1] += p0i[(ob + la, ob + si)] * w;
                    }
                }
                out.add(ImageOffset::origin(), oa + mu, oa + nu, cmul(acc, phase));
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let mut acc = [0.0_f64; 2];
                for mu in 0..na {
                    for nu in 0..na {
                        let w = te.two_e(mu, nu, la, si);
                        acc[0] += p0r[(oa + mu, oa + nu)] * w;
                        acc[1] += p0i[(oa + mu, oa + nu)] * w;
                    }
                }
                out.add(ImageOffset::origin(), ob + la, ob + si, cmul(acc, conj));
            }
        }

        // Exchange: the response block at this translation, unphased, into `F(T)` and `F(−T)`.
        if pair.exchange_scale == 0.0 {
            continue;
        }
        // This channel's response density, not the total: exchange contracts `ΔP^σ(0,T)`.
        let Some((ptr, pti)) = spin.get(t) else {
            continue;
        };
        for mu in 0..na {
            for la in 0..nb {
                let mut acc = [0.0_f64; 2];
                for nu in 0..na {
                    for si in 0..nb {
                        let w = te.two_e(mu, nu, la, si) * pair.exchange_scale;
                        acc[0] += scale * ptr[(oa + nu, ob + si)] * w;
                        acc[1] += scale * pti[(oa + nu, ob + si)] * w;
                    }
                }
                out.add(t, oa + mu, ob + la, [-acc[0], -acc[1]]);
                out.add(neg_t, ob + la, oa + mu, [-acc[0], -acc[1]]);
            }
        }
    }
    Ok(out)
}

/// The bare perturbation `h⁽¹⁾(T)` per degree of freedom, and the fixed-density `Σ_T Φ(T)e^{iq·T}`.
fn bare_and_skeleton(
    ctx: &DfptContext<'_>,
    density: &RealSpaceBlocks,
    channels: &[&RealSpaceBlocks],
) -> Result<(Vec<Vec<ComplexBlocks>>, CMatrix)> {
    let (molecule, params, basis, neighbors, translations, options, long_range, q) = (
        ctx.molecule,
        ctx.params,
        ctx.basis,
        ctx.neighbors,
        ctx.translations,
        ctx.options,
        ctx.long_range,
        ctx.q,
    );
    use crate::dual::{Dual, Scalar};
    use crate::dual2::Dual2;
    let nat = molecule.atoms.len();
    let nao = basis.nao;
    let beta = |e: &crate::params::Am1Element, orb: u8| if orb == 0 { e.beta_s } else { e.beta_p };
    let onsite = density
        .get(ImageOffset::origin())
        .ok_or_else(|| Am1Error::InvalidInput("density is missing the origin block".into()))?;

    // `bare[channel][dof]`. The spin-independent half of `∂F^σ/∂R` — resonance, core attraction,
    // Coulomb — is the **same** in every channel and is written to each in full; only the
    // exchange reads the channel's own density, at its own weight.
    let ex_scale = crate::pbc::scf::exchange_scale_for(ctx.fill);
    let mut bare: Vec<Vec<ComplexBlocks>> = channels
        .iter()
        .map(|_| {
            (0..3 * nat)
                .map(|_| ComplexBlocks::zeros(translations, nao))
                .collect()
        })
        .collect();
    let mut skeleton = CMatrix::zeros(3 * nat);

    for pair in &neighbors.pairs {
        let eu = params.element(molecule.atoms[pair.i].z)?;
        let ev = params.element(molecule.atoms[pair.j].z)?;
        // Reordered so `a` is the atom with the larger basis. The phase below is defined against
        // the **reordered** translation; keeping the raw ordering puts the conjugate phase on
        // half the pairs, which is invisible at `q = 0` and fatal beyond it.
        let (a, b, delta, t) = if eu.has_p() || !ev.has_p() {
            (pair.i, pair.j, pair.delta, pair.offset)
        } else {
            (pair.j, pair.i, pair.delta * -1.0, pair.offset.negated())
        };
        let neg_t = t.negated();
        let ea = params.element(molecule.atoms[a].z)?;
        let eb = params.element(molecule.atoms[b].z)?;
        let pa = molecule.atoms[a].position;
        let pb = pa + delta;
        let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
        let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);
        let (pc, ps) = q.phase(t);
        let phase = [pc, ps];
        let conj = [pc, -ps];
        let one = [1.0, 0.0];
        // An explicit error, not `unwrap_or(onsite)`. Falling back to the origin block when a
        // translation is missing substitutes a *different physical quantity* — `P(0,0)` where
        // `P(0,T)` was meant — and the calculation then completes and returns a plausible wrong
        // number. If the pair list and the density's translation set disagree, that is a bug in
        // whoever built them, and it should say so.
        let block_at = |want: ImageOffset| -> Result<Matrix> {
            density.get(want).cloned().ok_or_else(|| {
                Am1Error::InvalidInput(format!(
                    "the density has no block for translation {want:?}, which the pair list \
                     requires; the two were built from different translation sets"
                ))
            })
        };
        let pt = block_at(t)?;
        // The same two blocks of each spin channel's density, for the exchange.
        let channel_at = |want: ImageOffset| -> Result<Vec<Matrix>> {
            channels
                .iter()
                .map(|d| {
                    d.get(want).cloned().ok_or_else(|| {
                        Am1Error::InvalidInput(format!(
                            "a spin channel's density has no block for translation {want:?}"
                        ))
                    })
                })
                .collect()
        };
        let ct = channel_at(t)?;
        let ct_neg = channel_at(neg_t)?;

        // ---- first derivatives, for the bare perturbation ----
        let te1 = crate::integrals::pair_two_electron_dual(ea, eb, delta);
        let s1 = crate::overlap::diatom_overlap_dual(ea, pa, eb, pb)?;
        let r_dual = Dual {
            v: pair.r,
            d: [delta.x / pair.r, delta.y / pair.r, delta.z / pair.r],
        };
        let taper = match options.exchange_cutoff {
            Some(rc) => crate::hamiltonian::exchange_taper_scalar::<Dual>(r_dual, rc),
            None => Dual::constant(1.0),
        };

        for axis in 0..3 {
            // A displacement of `b` moves `delta` forwards and one of `a` moves it backwards.
            // The phase is `e^{iq·S}` with `S` the cell of the displaced atom **measured from the
            // row atom's cell** — so it is four cases, not two.
            let cases: [(usize, f64, [f64; 2], bool); 4] = [
                (a, -1.0, one, true),   // a in cell 0, blocks whose row atom is a
                (b, 1.0, phase, true),  // b in cell T, same blocks
                (b, 1.0, one, false),   // b in cell 0, blocks whose row atom is b
                (a, -1.0, conj, false), // a in cell −T, same blocks
            ];
            for (atom, sign, ph, row_is_a) in cases {
                let scale = |x: f64| -> [f64; 2] { [ph[0] * sign * x, ph[1] * sign * x] };
                for (ci, target_set) in bare.iter_mut().enumerate() {
                    let target = &mut target_set[3 * atom + axis];
                    let (pt, pt_neg) = (&ct[ci], &ct_neg[ci]);

                    if row_is_a {
                        for i in 0..na {
                            let bi = beta(ea, basis.aos[oa + i].orb);
                            for j in 0..nb {
                                let bj = beta(eb, basis.aos[ob + j].orb);
                                target.add(
                                    t,
                                    oa + i,
                                    ob + j,
                                    scale(0.5 * (bi + bj) * s1[i][j].d[axis]),
                                );
                            }
                        }
                        for i in 0..na {
                            for j in 0..na {
                                target.add(
                                    ImageOffset::origin(),
                                    oa + i,
                                    oa + j,
                                    scale(te1.e1b[i][j].d[axis]),
                                );
                            }
                        }
                        for mu in 0..na {
                            for nu in 0..na {
                                let mut acc = 0.0;
                                for la in 0..nb {
                                    for si in 0..nb {
                                        acc += onsite[(ob + la, ob + si)]
                                            * te1.two_e(mu, nu, la, si).d[axis];
                                    }
                                }
                                target.add(ImageOffset::origin(), oa + mu, oa + nu, scale(acc));
                            }
                        }
                        for mu in 0..na {
                            for la in 0..nb {
                                let mut acc = 0.0;
                                for nu in 0..na {
                                    for si in 0..nb {
                                        let w = te1.two_e(mu, nu, la, si);
                                        acc += pt[(oa + nu, ob + si)]
                                            * (taper.v * w.d[axis] + taper.d[axis] * w.v);
                                    }
                                }
                                target.add(t, oa + mu, ob + la, scale(-ex_scale * acc));
                            }
                        }
                    } else {
                        for i in 0..na {
                            let bi = beta(ea, basis.aos[oa + i].orb);
                            for j in 0..nb {
                                let bj = beta(eb, basis.aos[ob + j].orb);
                                target.add(
                                    neg_t,
                                    ob + j,
                                    oa + i,
                                    scale(0.5 * (bi + bj) * s1[i][j].d[axis]),
                                );
                            }
                        }
                        for k in 0..nb {
                            for l in 0..nb {
                                target.add(
                                    ImageOffset::origin(),
                                    ob + k,
                                    ob + l,
                                    scale(te1.e2a[k][l].d[axis]),
                                );
                            }
                        }
                        for la in 0..nb {
                            for si in 0..nb {
                                let mut acc = 0.0;
                                for mu in 0..na {
                                    for nu in 0..na {
                                        acc += onsite[(oa + mu, oa + nu)]
                                            * te1.two_e(mu, nu, la, si).d[axis];
                                    }
                                }
                                target.add(ImageOffset::origin(), ob + la, ob + si, scale(acc));
                            }
                        }
                        for mu in 0..na {
                            for la in 0..nb {
                                let mut acc = 0.0;
                                for nu in 0..na {
                                    for si in 0..nb {
                                        let w = te1.two_e(mu, nu, la, si);
                                        acc += pt_neg[(ob + si, oa + nu)]
                                            * (taper.v * w.d[axis] + taper.d[axis] * w.v);
                                    }
                                }
                                target.add(neg_t, ob + la, oa + mu, scale(-ex_scale * acc));
                            }
                        }
                    }
                }
            }
        }

        // ---- second derivative, for the skeleton ----
        let dvec = [
            Dual2::var(delta.x, 0),
            Dual2::var(delta.y, 1),
            Dual2::var(delta.z, 2),
        ];
        let te2 = crate::integrals::pair_two_electron_g::<Dual2>(ea, eb, dvec);
        let s2 = crate::overlap::diatom_overlap_dual2(ea, pa, eb, pb)?;
        let mut epair = Dual2::constant(0.0);
        for i in 0..na {
            let bi = beta(ea, basis.aos[oa + i].orb);
            for j in 0..nb {
                let bj = beta(eb, basis.aos[ob + j].orb);
                epair = epair + s2[i][j] * (pt[(oa + i, ob + j)] * (bi + bj));
            }
        }
        for i in 0..na {
            for j in 0..na {
                epair = epair + te2.e1b[i][j] * onsite[(oa + i, oa + j)];
            }
        }
        for k in 0..nb {
            for l in 0..nb {
                epair = epair + te2.e2a[k][l] * onsite[(ob + k, ob + l)];
            }
        }
        let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
        let taper2 = match options.exchange_cutoff {
            Some(rc) => crate::hamiltonian::exchange_taper_scalar::<Dual2>(r, rc),
            None => Dual2::constant(1.0),
        };
        for mu in 0..na {
            for nu in 0..na {
                for la in 0..nb {
                    for si in 0..nb {
                        let w = te2.two_e(mu, nu, la, si);
                        let coul = onsite[(oa + mu, oa + nu)] * onsite[(ob + la, ob + si)];
                        // Summed over spin channels: `−Σ_σ s P^σ P^σ`, with `s = ½` and one
                        // channel carrying the total restricted, `s = 1` and two channels
                        // unrestricted. Those agree when `P^α = P^β = P/2`, which is what makes
                        // forcing UHF on a closed shell reproduce the restricted `D(q)`.
                        let mut exch = Dual2::constant(0.0);
                        for cp in &ct {
                            exch =
                                exch - cp[(oa + mu, ob + la)] * cp[(oa + nu, ob + si)] * ex_scale;
                        }
                        epair = epair + w * coul + w * taper2 * exch;
                    }
                }
            }
        }
        epair = epair
            + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                ea,
                eb,
                molecule.atoms[a].z,
                molecule.atoms[b].z,
                r,
            );

        // The diagonal blocks carry no phase; the cross terms carry `e^{±iq·T}`. A self-image
        // pair, which cancels to nothing at `q = 0`, contributes `2h(1 − cos q·T)` away from it —
        // an atom really does feel its own images when they move out of step with it.
        for (i, row) in epair.h.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let w = *value;
                skeleton.add(3 * a + i, 3 * a + j, w, 0.0);
                skeleton.add(3 * b + i, 3 * b + j, w, 0.0);
                skeleton.add(3 * a + i, 3 * b + j, -w * phase[0], -w * phase[1]);
                skeleton.add(3 * b + i, 3 * a + j, -w * conj[0], -w * conj[1]);
            }
        }
    }

    // ---- the long-range monopole correction, at this `q` ----
    //
    // See [`LongRangeQ`] for the derivation. Two contributions, and the first of them is the one
    // that is easy to get wrong: the `δ_ab` term carries **no** phase, because both of its
    // indices sit in the home cell. It is what makes the acoustic sum rule hold.
    if let Some(lr) = long_range {
        let q_charges = &lr.charges;
        for a in 0..nat {
            // The diagonal, unphased: `Q_a Σ_c Q_c Δ''(0; d_ac)`.
            for c in 0..nat {
                let w = q_charges[a] * q_charges[c];
                let h0 = &lr.hess_0[lr.at(a, c)];
                for i in 0..3 {
                    for j in 0..3 {
                        skeleton.add(3 * a + i, 3 * a + j, w * h0[i][j], 0.0);
                    }
                }
            }
            // The off-diagonal, phased: `− Q_a Q_b Δ''(q; d_ab)`.
            for b in 0..nat {
                let w = q_charges[a] * q_charges[b];
                let hq = &lr.hess_q[lr.at(a, b)];
                for i in 0..3 {
                    for j in 0..3 {
                        skeleton.add(3 * a + i, 3 * b + j, -w * hq[i][j][0], -w * hq[i][j][1]);
                    }
                }
            }
        }

        // The bare perturbation. The Fock carries `−V_a` on atom `a`'s diagonal with
        // `V_a = Σ_b Δ_ab Q_b`, so displacing atom `c` along `β` shifts it by `−G_β(a, c, q)`
        // with
        //
        //     G_β(a, c, q) = −δ_ac Σ_b Q_b Δ'_β(0; d_ab)  +  Q_c Δ'_β(q; d_ac)
        //
        // — the same `δ`-term-without-a-phase structure as the skeleton above.
        //
        // Written to **every** spin channel in full: this is a Coulomb term, built from the total
        // net charges, and both channels feel all of it.
        for c in 0..nat {
            for axis in 0..3 {
                for target_set in bare.iter_mut() {
                    let target = &mut target_set[3 * c + axis];
                    for a in 0..nat {
                        let mut g = [0.0_f64; 2];
                        if a == c {
                            for b in 0..nat {
                                g[0] -= q_charges[b] * lr.grad_0[lr.at(a, b)][axis];
                            }
                        }
                        let gq = &lr.grad_q[lr.at(a, c)];
                        g[0] += q_charges[c] * gq[axis][0];
                        g[1] += q_charges[c] * gq[axis][1];

                        let off = basis.atom_offset[a];
                        for k in 0..basis.atom_norb[a] {
                            target.add(ImageOffset::origin(), off + k, off + k, [-g[0], -g[1]]);
                        }
                    }
                }
            }
        }
    }

    Ok((bare, skeleton))
}
