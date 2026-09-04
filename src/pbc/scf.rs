// SPDX-License-Identifier: GPL-3.0-or-later

//! k-point self-consistent field for a periodic NDDO cell.
//!
//! # Why this exists, and what it fixes
//!
//! The Γ-only path works with one Bloch-summed Hamiltonian and one density matrix. That is
//! exact for the *energy expression*, but it carries a defect that no amount of care in the
//! assembly removes: at Γ the real-space density matrix does not decay. `P(0,T) = P(Γ)` for
//! every translation, so NDDO's two-centre exchange — whose integral falls off only as `1/R`
//! — sums to infinity, and the Γ path has to taper it away by hand.
//!
//! Sampling k restores the physics. The real-space density matrix becomes
//!
//! ```text
//! P(0,T) = Σ_k w_k e^{−ik·T} P(k)
//! ```
//!
//! and for a gapped system that sum decays with `|T|` because the phases interfere. The
//! exchange then converges on its own, and the truncation stops being an approximation and
//! becomes a screening threshold.
//!
//! # Structure of the working equations
//!
//! NDDO's differential-overlap neglect decides which real-space block each term lands in, and
//! it is worth being explicit because it is what makes this tractable:
//!
//! * **Electron–core attraction** couples `μ` and `ν` on the *same* atom, so it contributes to
//!   `H(0,0)` only — for every translation of the partner. It never enters `H(0,T≠0)`.
//! * **Resonance** `½(β_μ+β_ν) S_μν` couples atoms across a translation, so it fills
//!   `H(0,T)`. The overlap decays exponentially, so these blocks die quickly with `|T|`.
//! * **Coulomb** couples on-site charge distributions, so it lands in `F(0,0)`, again summed
//!   over all partner translations.
//! * **Exchange** couples `μ` on one atom with `λ` on a partner across a translation, so it
//!   fills `F(0,T)` and contracts against `P(0,T)` — the element that actually decays.
//!
//! Because `S(k) = I` in NDDO, each k-point is a plain Hermitian eigenproblem, not a
//! generalized one.

use std::collections::HashMap;

use crate::basis::Basis;
use crate::error::{Am1Error, Result};
use crate::fermi::{fill, Filling, Level};
use crate::hamiltonian::{exchange_taper, PairIntegral};
use crate::integrals::pair_two_electron;
use crate::lattice::{ImageOffset, Lattice};
use crate::linalg::Matrix;
use crate::neighbors::NeighborList;
use crate::overlap::diatom_overlap;
use crate::params::Am1Parameters;
use crate::pbc::complex::{hermitian_eigen, CMatrix};
use crate::pbc::kpoints::{KMesh, KPoint};
use crate::system::Molecule;

/// Controls for a periodic calculation.
#[derive(Clone, Debug)]
pub struct PbcOptions {
    pub kmesh: KMesh,
    /// An explicit k-point list, overriding [`PbcOptions::kmesh`] when present.
    ///
    /// Weights must sum to 1. This exists so that a response calculation and the ground state it
    /// is built on can be made to sample *exactly* the same points: `kmesh` describes a family of
    /// grids, and "the same grid" is not something two call sites can be relied on to agree about
    /// once shifts and non-periodic-axis collapse are in play. See [`crate::pbc::dfpt`].
    pub kpoints: Option<Vec<KPoint>>,
    /// Merge `k` with `−k`. Exact for a real Hamiltonian and halves the diagonalizations.
    /// Ignored when [`PbcOptions::kpoints`] is given — an explicit list is taken as written.
    pub fold_time_reversal: bool,
    /// Cutoff (Bohr) on the lattice **translation** for the image sums.
    ///
    /// On the translation, not on the pair distance: the `1/R` monopole terms cancel only if
    /// every atom pair of an image is present, so a distance cutoff breaks charge neutrality.
    pub realspace_cutoff: f64,
    /// Distance (Bohr) at which the two-centre exchange is tapered off, or `None` to keep it
    /// all. With a converged k-mesh `P(0,T)` decays on its own and this is a screening
    /// threshold; at Γ-only it is a genuine approximation standing in for that decay.
    pub exchange_cutoff: Option<f64>,
    /// Sum the long-range monopole electrostatics by Ewald summation. See [crate::scf::Am1Options::ewald].
    pub ewald: bool,
    /// Add the analytic Klopman–Ohno `R⁻³` tail beyond the real-space cutoff.
    ///
    /// Without it the total energy drifts logarithmically with `realspace_cutoff` — measured at
    /// about 0.1 eV per unit `ln r_c` — because `γ_η − 1/R` decays as `R⁻³` and `Σ_T |T|⁻³`
    /// diverges in three dimensions. With it the `ln r_c` dependence cancels analytically against
    /// a stated reference length.
    ///
    /// On by default, and kept as a switch so the drift it removes can be *measured* rather than
    /// asserted — `tests/pbc_klopman_ohno_tail.rs` sweeps the cutoff both ways. See
    /// [`crate::pbc::ewald::LongRangeMonopole::with_klopman_ohno_tail`].
    pub klopman_ohno_tail: bool,
    /// Electronic temperature (eV) for Fermi–Dirac filling. Zero selects sharp aufbau, which
    /// is only appropriate for a gapped system with a converged mesh.
    pub smearing_ev: f64,
    pub charge: f64,
    pub max_scf: usize,
    pub e_tol: f64,
    pub p_tol: f64,
    /// Linear mixing fraction for the real-space density blocks, and the relaxation the DIIS
    /// extrapolation carries.
    pub mixing: f64,
    /// How many previous densities the Pulay (DIIS) mixer keeps. `0` or `1` disables it and
    /// leaves plain linear mixing, which is what every release through 0.2.2 did.
    ///
    /// Through 0.2.2 the periodic SCF had **no convergence acceleration at all** while the
    /// molecular one had A-DIIS→CDIIS, and the gap did not show on the systems the tests covered:
    /// a hydrogen fluoride slab is stiff and reaches `1e-10` in about 140 linear passes. A
    /// molecular-crystal slab is not. Measured on a 2D methane lattice at 9 Å with a 4×4 mesh —
    /// a closed-shell insulator with a 9 eV gap, about as easy as a periodic system gets — plain
    /// mixing was still 1e-8 from converged after **800** iterations, and the energy it stopped
    /// at was wrong by 1e-4 eV. The failing mode is long-wavelength charge sloshing between
    /// cells, whose linear convergence rate approaches 1 as the lattice loosens, which is why a
    /// tighter `p_tol` did not help and a **looser** one only stopped the loop earlier at a worse
    /// answer.
    ///
    /// Memory is `2 × depth` copies of the real-space density: for a large cell this is the
    /// dominant allocation of the run, the same way `dc:diis` is in divide-and-conquer. Lower it
    /// before lowering anything else if a run is tight on memory.
    pub diis_history: usize,
    /// Spin multiplicity `2S+1`. Anything above 1 selects the unrestricted path.
    pub multiplicity: usize,
    /// Force the unrestricted path even for a closed shell.
    pub unrestricted: bool,
    /// Uniform external electric field, in eV per (e·Bohr), or `None`.
    ///
    /// **Must be orthogonal to every periodic lattice vector** — normal to a slab, transverse to a
    /// chain — and a component along a periodic direction is an error naming itself. `F·R` shifts
    /// by `F·T` under translation by `T`, so the perturbation repeats with the lattice exactly
    /// when `F·T = 0`; along a periodic direction it is unbounded, and the object that replaces it
    /// there is the Berry-phase polarization ([`crate::pbc::berry`]), not a modified `F·R`.
    ///
    /// Refused outright under any cell through 0.2.1, which threw the well-defined cases out with
    /// the ill-defined one. The sign convention is [`crate::dipole`]'s, shared with the molecular
    /// field: `E(F) = E₀ − μ·F`.
    pub electric_field: Option<crate::math::Vec3>,
}

impl PbcOptions {
    /// The k points this configuration actually samples: the explicit list if there is one,
    /// otherwise the resolved mesh.
    ///
    /// One accessor rather than a `kmesh.resolve(...)` at each call site, so that "which k points
    /// is this calculation using" has exactly one answer.
    pub fn resolve_kpoints(&self, cell: &crate::lattice::Lattice) -> Result<Vec<KPoint>> {
        if let Some(points) = &self.kpoints {
            if points.is_empty() {
                return Err(Am1Error::InvalidInput(
                    "the explicit k-point list is empty".into(),
                ));
            }
            let total: f64 = points.iter().map(|k| k.weight).sum();
            if (total - 1.0).abs() > 1.0e-12 {
                return Err(Am1Error::InvalidInput(format!(
                    "explicit k-point weights must sum to 1, they sum to {total}"
                )));
            }
            return Ok(points.clone());
        }
        self.kmesh.resolve(cell, self.fold_time_reversal)
    }
}

/// The largest `Σ|c_i|` a DIIS step is allowed before it is thrown away.
///
/// Pulay coefficients sum to one but are otherwise unbounded, and a near-dependent history
/// produces large cancelling entries — an extrapolation far outside the span the residuals
/// actually resolve. The usual value; the point of having it is that the failure is a dropped
/// step rather than a diverged run.
const MAX_DIIS_WEIGHT: f64 = 40.0;

/// Pulay (DIIS) coefficients for a set of flattened fixed-point residuals.
///
/// Solves `min ‖Σ c_i r_i‖` subject to `Σ c_i = 1` through the bordered normal equations, and
/// returns `None` when the history is too short or the system is singular — in which case the
/// caller keeps whatever it was doing.
///
/// **Normalised and ridged.** Residual norms fall by many orders of magnitude over a solve, so an
/// unscaled `B` goes numerically singular long before the history is genuinely redundant; the
/// pivot guard then (correctly) refuses it and the acceleration silently stops. Dividing by the
/// largest entry and adding `1e-10` to the diagonal keeps the same solution and a usable
/// condition number.
///
/// Shared by the periodic SCF and the periodic CPSCF, which iterate different quantities to the
/// same fixed-point form.
pub(crate) fn pulay_coefficients(residuals: &[&[f64]]) -> Option<Vec<f64>> {
    let n = residuals.len();
    if n < 2 {
        return None;
    }
    let dim = n + 1;
    let mut b = Matrix::zeros(dim, dim);
    let mut scale = 0.0_f64;
    for i in 0..n {
        for j in 0..=i {
            let v: f64 = residuals[i]
                .iter()
                .zip(residuals[j])
                .map(|(x, y)| x * y)
                .sum();
            b[(i, j)] = v;
            b[(j, i)] = v;
            scale = scale.max(v.abs());
        }
        b[(i, n)] = -1.0;
        b[(n, i)] = -1.0;
    }
    if scale <= 0.0 || !scale.is_finite() {
        return None;
    }
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] /= scale;
        }
        b[(i, i)] += 1.0e-10;
    }
    let mut rhs = vec![0.0; dim];
    rhs[n] = -1.0;
    let c = crate::scf::solve_bordered_small(&b, &rhs)?;
    if c.iter().take(n).any(|v| !v.is_finite()) {
        return None;
    }
    Some(c)
}

impl Default for PbcOptions {
    fn default() -> Self {
        Self {
            kmesh: KMesh::Gamma,
            kpoints: None,
            fold_time_reversal: true,
            realspace_cutoff: 40.0,
            exchange_cutoff: Some(20.0),
            ewald: true,
            klopman_ohno_tail: true,
            smearing_ev: 0.05,
            charge: 0.0,
            max_scf: 300,
            e_tol: 1.0e-8,
            p_tol: 1.0e-7,
            mixing: 0.3,
            diis_history: 8,
            multiplicity: 1,
            unrestricted: false,
            electric_field: None,
        }
    }
}

/// Real-space matrices indexed by lattice translation.
///
/// One `nao x nao` block per translation. Indexed through a map so a Bloch sum can walk the
/// translations in any order and a caller can ask for a particular one.
#[derive(Clone, Debug)]
pub struct RealSpaceBlocks {
    pub translations: Vec<ImageOffset>,
    pub blocks: Vec<Matrix>,
    index: HashMap<[i32; 3], usize>,
}

impl RealSpaceBlocks {
    pub fn zeros(translations: &[ImageOffset], nao: usize) -> Self {
        let index = translations
            .iter()
            .enumerate()
            .map(|(i, t)| (t.n, i))
            .collect();
        Self {
            translations: translations.to_vec(),
            blocks: translations
                .iter()
                .map(|_| Matrix::zeros(nao, nao))
                .collect(),
            index,
        }
    }

    #[inline]
    pub fn position(&self, offset: ImageOffset) -> Option<usize> {
        self.index.get(&offset.n).copied()
    }

    #[inline]
    pub fn get_mut(&mut self, offset: ImageOffset) -> Option<&mut Matrix> {
        self.position(offset).map(|i| &mut self.blocks[i])
    }

    #[inline]
    pub fn get(&self, offset: ImageOffset) -> Option<&Matrix> {
        self.position(offset).map(|i| &self.blocks[i])
    }

    /// The `T = 0` block, or an error naming what is missing.
    ///
    /// Almost every consumer wants this one block, and almost every one of them used to reach it
    /// with `.get(ImageOffset::origin()).unwrap()`. "The origin is always in the translation set"
    /// is a property of how that set is built, not something this type enforces — and if it ever
    /// stopped holding, a panic from inside an SCF iteration is the worst way to find out. These
    /// two accessors make the assumption explicit and cheap to propagate.
    pub fn origin(&self) -> Result<&Matrix> {
        self.get(ImageOffset::origin()).ok_or_else(Self::missing)
    }

    /// The `T = 0` block, mutably. See [`Self::origin`].
    pub fn origin_mut(&mut self) -> Result<&mut Matrix> {
        self.get_mut(ImageOffset::origin())
            .ok_or_else(Self::missing)
    }

    fn missing() -> Am1Error {
        Am1Error::InvalidInput(
            "a real-space matrix is missing its origin block, which every translation set must \
             contain"
                .into(),
        )
    }

    /// Add `other` block by block, in place.
    ///
    /// Blocks are matched by *translation*, not by position: two sets built from the same cutoff
    /// agree on both, but nothing here guarantees that, and a mismatched pair would otherwise add
    /// the wrong images together silently. A translation `other` has and `self` does not is
    /// skipped — `self` decides which images exist.
    pub fn add_assign(&mut self, other: &Self) {
        for (t, block) in other.translations.iter().zip(&other.blocks) {
            let Some(i) = self.position(*t) else { continue };
            for (a, b) in self.blocks[i]
                .as_mut_slice()
                .iter_mut()
                .zip(block.as_slice())
            {
                *a += *b;
            }
        }
    }

    /// `M(k) = Σ_T e^{ik·T} M(0,T)`.
    pub fn bloch_sum(&self, k: &KPoint) -> CMatrix {
        let nao = self.blocks.first().map(|m| m.rows).unwrap_or(0);
        let mut out = CMatrix::zeros(nao);
        for (t, block) in self.translations.iter().zip(&self.blocks) {
            let (c, s) = k.phase(*t);
            for i in 0..nao {
                for j in 0..nao {
                    let v = block[(i, j)];
                    if v != 0.0 {
                        out.re[(i, j)] += c * v;
                        out.im[(i, j)] += s * v;
                    }
                }
            }
        }
        out.hermitianize();
        out
    }

    /// `Σ_T Σ_μν A(0,T)_μν B(0,T)_μν`, the contraction the energy expression needs.
    pub fn contract(&self, other: &RealSpaceBlocks) -> f64 {
        let mut acc = 0.0;
        for (t, block) in self.translations.iter().zip(&self.blocks) {
            if let Some(o) = other.get(*t) {
                acc += block.frobenius_dot(o);
            }
        }
        acc
    }

    /// Largest magnitude in the block at `offset`, for decay diagnostics.
    pub fn block_norm(&self, offset: ImageOffset) -> f64 {
        self.get(offset)
            .map(|m| m.as_slice().iter().fold(0.0_f64, |a, v| a.max(v.abs())))
            .unwrap_or(0.0)
    }
}

/// Everything a converged periodic calculation produced.
#[derive(Clone, Debug)]
pub struct PbcResult {
    /// Total real-space density blocks (both spins summed).
    pub density: RealSpaceBlocks,
    /// Spin density `Pα − Pβ`, present only on the unrestricted path.
    pub spin_density: Option<RealSpaceBlocks>,
    /// True when the unrestricted path was used.
    pub unrestricted: bool,
    pub electronic_ev: f64,
    pub core_ev: f64,
    pub total_ev: f64,
    pub band_energy_ev: f64,
    /// Chemical potential (eV). With a fixed multiplicity the two spin channels have separate
    /// Fermi levels; this is the higher of those that actually hold electrons.
    pub fermi_energy_ev: f64,
    /// `T·S`, eV.
    pub entropy_ev: f64,
    pub charges: Vec<f64>,
    pub k_points: usize,
    pub iterations: usize,
    pub converged: bool,
    /// Largest `|S_μν(0,T)|` over `T ≠ 0`, a check on the orthonormal-basis assumption.
    pub max_image_overlap: f64,
    /// Set when the cell carries a net charge. See [`PbcResult::charged_cell_warning`].
    pub charged_cell_warning: Option<String>,
}

/// The warning attached to any periodic result with a net charge.
///
/// A charged cell has no converged total energy here, and the reason is structural rather than a
/// tuning problem. Without a compensating background the monopole lattice sum `Σ_T Q²/|T|`
/// diverges, and truncating it at `realspace_cutoff` leaves the answer growing without bound as
/// that cutoff grows. Measured on a +1 water cell in an 8 Å cube: −331.2 eV at a 20 Bohr cutoff,
/// −298.3 at 40, −137.2 at 90, **+72.2 at 130** — a divergence, tracking the `π Q² r_c² / V`
/// the continuum estimate predicts, not a slow convergence.
///
/// What *is* well defined, and is what the charge support is for: the electron count, the
/// self-consistent density, the Mulliken charges (which sum to the formal charge), and the
/// forces — which are consistent with the energy being differentiated, so dynamics conserves.
/// Energy differences at fixed cell **and fixed cutoff** are meaningful. Absolute charged-cell
/// energies, and any comparison across different cells or cutoffs, are not.
///
/// # Corrected in 0.2.2
///
/// This warning used to say that no compensating background is applied "because Ewald summation
/// is not implemented", that the total energy is not converged, and it quoted a −331 eV to +72 eV
/// swing across a range of real-space cutoffs. Ewald summation has been implemented since 0.2.0,
/// in all three dimensionalities, and is on by default — those were the pre-Ewald numbers, and the
/// text had outlived the version it described. It told users their converged 3D energies were
/// meaningless.
///
/// What is true now differs by dimensionality, so the message does too:
///
/// * **3D**: the tin-foil Ewald sum defines the energy. `tests/charged_cell_warning.rs` measures
///   a +1 water cell across a 6.5× range of cutoff and finds **0.197 eV** of drift with Ewald
///   against **403 eV** without. The residual is the `R⁻³` Klopman–Ohno tail, which is still a
///   real-space cutoff.
/// * **1D / 2D**: the monopole channel is summed too, but the *neutralizing background* is where
///   the difficulty is, and it is a convention rather than a calculation — a charged slab's energy
///   depends on where the compensating sheet is placed, and a charged line's potential diverges
///   logarithmically. `SheetConvention` and `AxisConvention` name the choices; nothing in the SCF
///   path consults them for the net charge, so this code has not chosen one and the absolute
///   energy is not defined. Densities, charges and forces are self-consistent and usable.
///
/// ASCII only: this reaches users through both CLIs, whose output has to encode under a cp932 or
/// C locale. See the note in `src/bin/am1_rs.rs`.
fn charged_cell_warning(charge: f64, n_periodic: usize, ewald: bool) -> Option<String> {
    if charge.abs() < 1.0e-9 {
        return None;
    }
    let head = format!("this cell carries a net charge of {charge:+}");
    if !ewald {
        return Some(format!(
            "{head}, and `ewald` is off, so no compensating background is applied. The monopole \
             lattice sum diverges: the TOTAL ENERGY IS NOT CONVERGED and grows without bound with \
             realspace_cutoff (measured: a +1 water cell in an 8 A cube moves 403 eV between a 20 \
             and a 130 Bohr cutoff). The density, the Mulliken charges and the forces are still \
             self-consistent, and energy differences at fixed cell and fixed cutoff are \
             meaningful. Turn `ewald` on. See docs/pbc.md."
        ));
    }
    Some(match n_periodic {
        3 => format!(
            "{head}. The 3D Ewald sum applies the tin-foil neutralizing background, so the total \
             energy IS defined and converged: the same +1 water cell moves 0.20 eV across a 6.5x \
             range of realspace_cutoff, against 403 eV with `ewald` off. The residual is the R^-3 \
             Klopman-Ohno tail, which is still a real-space cutoff, so quote energies at a stated \
             cutoff and compare at the same one. See docs/pbc.md."
        ),
        d => format!(
            "{head}, in {d}D. The monopole lattice sum is applied, but the neutralizing \
             background's placement is a CONVENTION this code has not chosen: a charged slab's \
             energy depends on where the compensating sheet sits, and a charged line's potential \
             diverges logarithmically. So the ABSOLUTE ENERGY IS NOT DEFINED here, and neither are \
             comparisons across cells. The density, the Mulliken charges and the forces are \
             self-consistent and usable, and so are energy differences at a fixed cell. See \
             `SheetConvention` / `AxisConvention` and docs/pbc.md."
        ),
    })
}

impl PbcResult {
    /// `E − TS`, the quantity that is variational at finite electronic temperature.
    pub fn free_energy_ev(&self) -> f64 {
        self.total_ev - self.entropy_ev
    }
    /// Energy extrapolated to `T → 0`.
    pub fn extrapolated_energy_ev(&self) -> f64 {
        self.total_ev - 0.5 * self.entropy_ev
    }
}

/// Per-pair integrals plus the translation they belong to.
pub(crate) struct PeriodicPairs {
    pub(crate) pairs: Vec<PairIntegral>,
    /// Largest image overlap seen while building, for the ZDO diagnostic.
    pub(crate) max_image_overlap: f64,
}

/// Build the real-space core Hamiltonian blocks and the pair integrals.
/// A converged density, split into the channels a Fock build's **exchange** contracts against.
///
/// Returns `(densities, fill)`: one density restricted, `P^α = (P + S)/2` and `P^β = (P − S)/2`
/// unrestricted; `fill` is what one orbital holds, `2` or `1`. The exchange weight is the
/// complement — `0.5` when a single channel stands for both spins, `1.0` otherwise — so that
/// `scale · P^σ` is the same object either way and a closed shell gives the restricted answer
/// back exactly.
///
/// Shared by the `q = 0` response ([`crate::pbc::hessian`]) and the finite-`q` one
/// ([`crate::pbc::dfpt`]) rather than written twice: the split, the weight and their consistency
/// are the whole content of the open-shell generalisation, and two copies is two places for them
/// to drift.
pub(crate) fn spin_channel_densities(scf: &PbcResult) -> (Vec<RealSpaceBlocks>, f64) {
    let total = &scf.density;
    match &scf.spin_density {
        Some(spin) => {
            let mut alpha = total.clone();
            let mut beta = total.clone();
            for (t, s) in spin.translations.iter().zip(&spin.blocks) {
                if let Some(a) = alpha.get_mut(*t) {
                    for (v, d) in a.as_mut_slice().iter_mut().zip(s.as_slice()) {
                        *v = 0.5 * (*v + *d);
                    }
                }
                if let Some(b) = beta.get_mut(*t) {
                    for (v, d) in b.as_mut_slice().iter_mut().zip(s.as_slice()) {
                        *v = 0.5 * (*v - *d);
                    }
                }
            }
            (vec![alpha, beta], 1.0)
        }
        None => (vec![total.clone()], 2.0),
    }
}

/// The exchange weight that goes with [`spin_channel_densities`]'s `fill`.
#[inline]
pub(crate) fn exchange_scale_for(fill: f64) -> f64 {
    // A midpoint test rather than `fill == 2.0`. `fill` is only ever assigned the literals `2.0`
    // and `1.0` by [`spin_channel_densities`], so the two are equivalent today — but this function
    // is the single place the open-shell factor conventions meet, and a `fill` that ever came out
    // of an arithmetic expression rather than a literal would take the wrong branch in silence.
    // What would catch it — forcing UHF on a closed shell — is sharp, but it lives three modules
    // away, and the cost of not needing it here is one comparison.
    if fill > 1.5 {
        0.5
    } else {
        1.0
    }
}

/// A uniform field is a lattice-periodic perturbation only when it is orthogonal to every
/// periodic lattice vector; reject it otherwise, naming the offending component.
///
/// `F·R` shifts by `F·T` under translation by `T`, so the perturbation repeats with the lattice
/// exactly when `F·T = 0` for every `T`. A slab in a normal field and a chain in a transverse
/// field satisfy that and are ordinary calculations. A field **along** a periodic direction does
/// not, and no amount of care in the assembly fixes it: the potential is unbounded, the spectrum
/// has no lower bound, and the object that replaces `F·R` there is the Berry-phase polarization —
/// see [`crate::pbc::berry`] and `docs/pbc.md`.
pub(crate) fn check_periodic_field(molecule: &Molecule, field: crate::math::Vec3) -> Result<()> {
    let Some(cell) = molecule.cell else {
        return Ok(());
    };
    let along = cell.periodic_component(field);
    if along.norm() > 1.0e-10 * field.norm().max(1.0) {
        return Err(Am1Error::InvalidInput(format!(
            "the electric field has a component ({:.4}, {:.4}, {:.4}) along a periodic direction, \
             where `F·R` is unbounded and the perturbation is not lattice-periodic. A field \
             orthogonal to every lattice vector — normal to a slab, transverse to a chain — is \
             supported. For the linear response along a periodic direction see \
             `pbc::dielectric_tensor`.",
            along.x, along.y, along.z
        )));
    }
    Ok(())
}

pub(crate) fn build_realspace_core(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    neighbors: &NeighborList,
    translations: &[ImageOffset],
    exchange_cutoff: Option<f64>,
    electric_field: Option<crate::math::Vec3>,
) -> Result<(RealSpaceBlocks, PeriodicPairs)> {
    use rayon::prelude::*;

    let nao = basis.nao;
    let mut h = RealSpaceBlocks::zeros(translations, nao);

    // Diagonal U_ss / U_pp go into the T = 0 block.
    {
        let origin = h
            .get_mut(ImageOffset::origin())
            .ok_or_else(|| Am1Error::InvalidInput("the origin translation is missing".into()))?;
        for (mu, ao) in basis.aos.iter().enumerate() {
            let elem = params.element(ao.z)?;
            origin[(mu, mu)] = if ao.orb == 0 { elem.u_ss } else { elem.u_pp };
        }
    }

    // The external field, on the `T = 0` block only.
    //
    // On that block alone because `F·R` is an *on-site* operator in NDDO: it couples `μ` and `ν`
    // on the same atom, exactly like the diagonal above. It carries no Bloch phase, which is what
    // makes it representable at all — and is legitimate only when `F` is orthogonal to every
    // periodic lattice vector, checked in [`check_periodic_field`].
    if let Some(field) = electric_field {
        check_periodic_field(molecule, field)?;
        let hf = crate::dipole::field_hamiltonian(molecule, basis, params, field)?;
        let origin = h.origin_mut()?;
        for (hv, fv) in origin.as_mut_slice().iter_mut().zip(hf.as_slice()) {
            *hv += fv;
        }
    }

    type Computed = (
        usize,
        usize,
        ImageOffset,
        f64,
        crate::integrals::PairTwoElec,
        [[f64; 4]; 4],
    );
    let computed: Vec<Computed> = neighbors
        .pairs
        .par_iter()
        .map(|p| -> Result<Computed> {
            let eu = params.element(molecule.atoms[p.i].z)?;
            let ev = params.element(molecule.atoms[p.j].z)?;
            let heavy_first = eu.has_p() || !ev.has_p();
            let (a, b, delta, offset) = if heavy_first {
                (p.i, p.j, p.delta, p.offset)
            } else {
                (p.j, p.i, p.delta * -1.0, p.offset.negated())
            };
            let (ea, eb) = (
                params.element(molecule.atoms[a].z)?,
                params.element(molecule.atoms[b].z)?,
            );
            let r = delta.norm();
            let te = pair_two_electron(ea, eb, delta / r, r);
            let pos_a = molecule.atoms[a].position;
            let s_block = diatom_overlap(ea, pos_a, eb, pos_a + delta)?;
            Ok((a, b, offset, r, te, s_block))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut pairs = Vec::with_capacity(computed.len());
    let mut max_image_overlap = 0.0_f64;

    for (a, b, offset, r, te, s_block) in computed {
        let (ea, eb) = (
            params.element(molecule.atoms[a].z)?,
            params.element(molecule.atoms[b].z)?,
        );
        let (off_a, off_b) = (basis.atom_offset[a], basis.atom_offset[b]);
        let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

        // Electron-core attraction lands on the on-site (T = 0) blocks, whatever the partner's
        // translation: NDDO couples mu and nu on the same atom here.
        {
            let origin = h.origin_mut()?;
            for i in 0..na {
                for j in 0..na {
                    origin[(off_a + i, off_a + j)] += te.e1b[i][j];
                }
            }
            for i in 0..nb {
                for j in 0..nb {
                    origin[(off_b + i, off_b + j)] += te.e2a[i][j];
                }
            }
        }

        // Resonance fills the T block and, for the mirror pair, the -T block.
        if !offset.is_origin() {
            for i in 0..na {
                for j in 0..nb {
                    max_image_overlap = max_image_overlap.max(s_block[i][j].abs());
                }
            }
        }
        let beta = |elem: &crate::params::Am1Element, orb: u8| {
            if orb == 0 {
                elem.beta_s
            } else {
                elem.beta_p
            }
        };
        let forward: Vec<(usize, usize, f64)> = (0..na)
            .flat_map(|i| {
                let bi = beta(ea, basis.aos[off_a + i].orb);
                (0..nb).map(move |j| (i, j, bi))
            })
            .map(|(i, j, bi)| {
                let bj = beta(eb, basis.aos[off_b + j].orb);
                (i, j, 0.5 * (bi + bj) * s_block[i][j])
            })
            .collect();

        if let Some(block) = h.get_mut(offset) {
            for &(i, j, v) in &forward {
                block[(off_a + i, off_b + j)] += v;
            }
        }
        if let Some(block) = h.get_mut(offset.negated()) {
            for &(i, j, v) in &forward {
                block[(off_b + j, off_a + i)] += v;
            }
        }

        let exchange_scale = match exchange_cutoff {
            Some(rc) => exchange_taper(r, rc),
            None => 1.0,
        };
        pairs.push(PairIntegral {
            a,
            b,
            offset,
            r,
            exchange_scale,
            te,
        });
    }

    Ok((
        h,
        PeriodicPairs {
            pairs,
            max_image_overlap,
        },
    ))
}

/// Build the real-space Fock blocks for one spin channel.
///
/// `total_origin` is the **total** on-site density (both spins), which is what the Coulomb
/// terms see. `spin` carries the same-spin density blocks that the exchange contracts
/// against, scaled by `spin_scale`. A restricted calculation passes the total density with
/// `spin_scale = 0.5`; an unrestricted one passes that channel's own blocks with
/// `spin_scale = 1`, so one routine serves both.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_realspace_fock(
    core: &RealSpaceBlocks,
    pairs: &PeriodicPairs,
    total_origin: &Matrix,
    spin: &RealSpaceBlocks,
    spin_scale: f64,
    basis: &Basis,
    molecule: &Molecule,
    params: &Am1Parameters,
    long_range: Option<&Matrix>,
) -> Result<RealSpaceBlocks> {
    let nao = basis.nao;
    let mut f = core.clone();
    let p0 = total_origin;
    let ps0 = spin
        .get(ImageOffset::origin())
        .ok_or_else(|| Am1Error::InvalidInput("density is missing the origin block".into()))?;

    // Long-range monopole correction. It is a shift on each atom's own diagonal and carries no
    // Bloch phase, so it belongs entirely to the `T = 0` block — the same `−V_a` the molecular
    // path applies in `crate::fock::build_fock_spin_with`.
    if let Some(delta) = long_range {
        let v = crate::fock::long_range_potential_from_delta(
            molecule,
            basis,
            params,
            delta,
            total_origin,
        )?;
        let origin = f.origin_mut()?;
        for (a, va) in v.iter().enumerate() {
            let off = basis.atom_offset[a];
            for k in 0..basis.atom_norb[a] {
                origin[(off + k, off + k)] -= va;
            }
        }
    }

    // One-centre (intra-atomic) two-electron terms, on the T = 0 block.
    {
        let origin = f.origin_mut()?;
        for (ia, atom) in molecule.atoms.iter().enumerate() {
            let elem = params.element(atom.z)?;
            let off = basis.atom_offset[ia];
            let n = basis.atom_norb[ia];
            let (gss, gsp, gpp, gp2, hsp) = (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
            for mu in 0..n {
                for nu in 0..n {
                    let mut acc = 0.0;
                    for la in 0..n {
                        for si in 0..n {
                            acc += p0[(off + la, off + si)]
                                * crate::fock::oc_two_electron(
                                    mu, nu, la, si, gss, gsp, gpp, gp2, hsp,
                                );
                            acc -= spin_scale
                                * ps0[(off + la, off + si)]
                                * crate::fock::oc_two_electron(
                                    mu, la, nu, si, gss, gsp, gpp, gp2, hsp,
                                );
                        }
                    }
                    origin[(off + mu, off + nu)] += acc;
                }
            }
        }
    }

    for pair in &pairs.pairs {
        let te = &pair.te;
        let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
        let (na, nb) = (te.norb_i, te.norb_j);

        // Coulomb: on-site blocks, from the partner's on-site density. Lands on T = 0.
        {
            let origin = f.origin_mut()?;
            for mu in 0..na {
                for nu in 0..na {
                    let mut acc = 0.0;
                    for la in 0..nb {
                        for si in 0..nb {
                            acc += p0[(ob + la, ob + si)] * te.two_e(mu, nu, la, si);
                        }
                    }
                    origin[(oa + mu, oa + nu)] += acc;
                }
            }
            for la in 0..nb {
                for si in 0..nb {
                    let mut acc = 0.0;
                    for mu in 0..na {
                        for nu in 0..na {
                            acc += p0[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si);
                        }
                    }
                    origin[(ob + la, ob + si)] += acc;
                }
            }
        }

        // Exchange: couples atom a in cell 0 with atom b in cell T, so it lands in F(0,T) and
        // contracts against P(0,T) -- the element that decays once k is sampled.
        if pair.exchange_scale == 0.0 {
            continue;
        }
        let pt = match spin.get(pair.offset) {
            Some(m) => m,
            None => continue,
        };
        let mut k_block = vec![0.0; na * nb];
        for mu in 0..na {
            for la in 0..nb {
                let mut acc = 0.0;
                for nu in 0..na {
                    for si in 0..nb {
                        acc += spin_scale * pt[(oa + nu, ob + si)] * te.two_e(mu, nu, la, si);
                    }
                }
                k_block[mu * nb + la] = -acc * pair.exchange_scale;
            }
        }
        if let Some(block) = f.get_mut(pair.offset) {
            for mu in 0..na {
                for la in 0..nb {
                    block[(oa + mu, ob + la)] += k_block[mu * nb + la];
                }
            }
        }
        if let Some(block) = f.get_mut(pair.offset.negated()) {
            for mu in 0..na {
                for la in 0..nb {
                    block[(ob + la, oa + mu)] += k_block[mu * nb + la];
                }
            }
        }
    }

    debug_assert_eq!(f.blocks[0].rows, nao);
    Ok(f)
}

/// Run a periodic NDDO SCF.
pub fn run_pbc_scf(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<PbcResult> {
    run_pbc_scf_with_k_terms(molecule, params, options, None)
}

/// [`run_pbc_scf`], with an extra **k-resolved** term added to each `H(k)` before it is
/// diagonalized.
///
/// `k_terms`, when given, must be one Hermitian `nao × nao` matrix per k point, in the order
/// [`PbcOptions::resolve_kpoints`] returns them — so a caller that wants a specific alignment
/// should pass its k list through [`PbcOptions::kpoints`] rather than trusting two meshes to agree.
///
/// The one caller is the finite-field driver ([`crate::pbc::finite_field`]). The Berry-phase field
/// operator is built from the coefficients at *neighbouring* k points, so it is not the Bloch sum
/// of anything in real space and cannot enter through `H_core` the way a molecular `F·R` does. It
/// is held fixed through one SCF here and refreshed by that driver's outer loop.
pub(crate) fn run_pbc_scf_with_k_terms(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    k_terms: Option<&[CMatrix]>,
) -> Result<PbcResult> {
    crate::linalg::enable_parallelism();
    let cell: Lattice = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a periodic calculation needs a cell".into()))?;
    if cell.n_periodic() == 0 {
        return Err(Am1Error::InvalidInput(
            "the cell has no periodic direction; use the molecular path".into(),
        ));
    }

    let basis = Basis::build(molecule, params)?;
    let nao = basis.nao;
    let neighbors = NeighborList::build(molecule, options.realspace_cutoff);
    // Long-range monopole correction, shared by every k point: it is a real-space, atom-pair
    // quantity with no Bloch phase, so one matrix serves the whole mesh.
    let long_range = crate::pbc::ewald::LongRangeMonopole::for_molecule_with(
        molecule,
        options
            .klopman_ohno_tail
            .then_some((params, options.realspace_cutoff)),
        &neighbors,
        options.ewald,
    )?;
    let delta = long_range.as_ref().map(|(m, _)| &m.delta);
    let translations = cell.image_offsets(options.realspace_cutoff);
    let k_points = options.resolve_kpoints(&cell)?;

    let (core, pairs) = build_realspace_core(
        molecule,
        &basis,
        params,
        &neighbors,
        &translations,
        options.exchange_cutoff,
        options.electric_field,
    )?;

    let mut n_elec = 0.0;
    for atom in &molecule.atoms {
        n_elec += params.element(atom.z)?.core_charge;
    }
    n_elec -= options.charge;
    if n_elec < 0.0 {
        return Err(Am1Error::InvalidInput(format!(
            "charge {} leaves a negative electron count",
            options.charge
        )));
    }

    // Spin channels. Restricted keeps one channel of capacity 2 per orbital; unrestricted
    // keeps two of capacity 1, each with its own electron count and its own Fermi level, which
    // is what fixing the multiplicity means.
    let n_unpaired = options.multiplicity.saturating_sub(1) as f64;
    let use_uhf = options.unrestricted || n_unpaired > 0.0;
    if use_uhf {
        let rounded = n_elec.round();
        if ((n_elec - n_unpaired) / 2.0).fract().abs() > 1.0e-6 && (rounded - n_elec).abs() < 1.0e-6
        {
            return Err(Am1Error::InvalidInput(format!(
                "electron count {n_elec} is incompatible with multiplicity {} (parity)",
                options.multiplicity
            )));
        }
    }
    let channel_electrons: Vec<f64> = if use_uhf {
        vec![(n_elec + n_unpaired) / 2.0, (n_elec - n_unpaired) / 2.0]
    } else {
        vec![n_elec]
    };
    let channel_capacity = if use_uhf { 1.0 } else { 2.0 };

    // Start from a superposition of atomic densities in the T = 0 block, split between the
    // channels in proportion to their electron counts.
    let mut channels: Vec<RealSpaceBlocks> = Vec::with_capacity(channel_electrons.len());
    for &ne in &channel_electrons {
        let mut d = RealSpaceBlocks::zeros(&translations, nao);
        let share = if n_elec > 0.0 { ne / n_elec } else { 0.0 };
        let origin = d.origin_mut()?;
        for (ia, atom) in molecule.atoms.iter().enumerate() {
            let elem = params.element(atom.z)?;
            let off = basis.atom_offset[ia];
            let n = basis.atom_norb[ia];
            let z = elem.core_charge;
            origin[(off, off)] = z.min(2.0) * share;
            if n == 4 {
                let per_p = (z - z.min(2.0)) / 3.0 * share;
                for k in 1..4 {
                    origin[(off + k, off + k)] = per_p;
                }
            }
        }
        channels.push(d);
    }

    let filling = if options.smearing_ev > 0.0 {
        Filling::Fermi {
            kt: options.smearing_ev,
        }
    } else {
        Filling::Aufbau
    };

    let mut e_old = 0.0;
    let mut converged = false;
    let mut iterations = 0usize;
    let mut band_energy = 0.0;
    let mut fermi_energy = 0.0_f64;
    let mut entropy = 0.0;
    let mut electronic = 0.0;

    // Pulay mixing state. The two scratch vectors are reused across iterations so that the
    // flattening does not allocate on the critical path; `history` holds the `(input, residual)`
    // pairs, and is the only thing here whose size grows with the depth.
    let diis_depth = options.diis_history;
    let mut history: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut diis_input: Vec<f64> = Vec::new();
    let mut diis_resid: Vec<f64> = Vec::new();

    // `AM1_SCF_TRACE=1` prints `dE` and `dP` per iteration, which is the only way to tell a slow
    // contraction from a stall from the outside — the difference that separates "raise `max_scf`"
    // from "this will never converge". Same switch style as `AM1_TIMING`.
    let trace = std::env::var("AM1_SCF_TRACE").is_ok_and(|v| v != "0");

    // Set once the tolerances are met, to spend one further pass evaluating the energy at the
    // **output** density instead of the mixed input. See where it is set for why that matters.
    let mut final_pass = false;

    for iter in 0..options.max_scf {
        iterations = iter + 1;

        // Total on-site density drives the Coulomb terms for every channel.
        let mut total_origin = Matrix::zeros(nao, nao);
        for ch in &channels {
            let o = ch.origin()?;
            for (t, v) in total_origin.as_mut_slice().iter_mut().zip(o.as_slice()) {
                *t += *v;
            }
        }

        let mut focks = Vec::with_capacity(channels.len());
        let mut new_channels = Vec::with_capacity(channels.len());
        band_energy = 0.0;
        entropy = 0.0;
        fermi_energy = 0.0;

        for (ci, spin_density) in channels.iter().enumerate() {
            let spin_scale = if use_uhf { 1.0 } else { 0.5 };
            let fock = build_realspace_fock(
                &core,
                &pairs,
                &total_origin,
                spin_density,
                spin_scale,
                &basis,
                molecule,
                params,
                delta,
            )?;

            // Diagonalize at every k; one Fermi level across the whole zone for this channel.
            let mut eigen = Vec::with_capacity(k_points.len());
            let mut levels = Vec::with_capacity(k_points.len() * nao);
            for (kidx, kp) in k_points.iter().enumerate() {
                let mut hk = fock.bloch_sum(kp);
                if let Some(extra) = k_terms {
                    let add = extra.get(kidx).ok_or_else(|| {
                        Am1Error::InvalidInput(
                            "the k-resolved extra term has fewer entries than there are k points"
                                .into(),
                        )
                    })?;
                    for i in 0..nao {
                        for j in 0..nao {
                            let (r, m) = add.get(i, j);
                            hk.re[(i, j)] += r;
                            hk.im[(i, j)] += m;
                        }
                    }
                }
                let e = hermitian_eigen(&hk)?;
                for &value in &e.values {
                    levels.push(Level {
                        energy: value,
                        weight: channel_capacity * kp.weight,
                    });
                }
                eigen.push(e);
            }

            let occ = fill(&levels, channel_electrons[ci], filling)?;
            band_energy += occ.band_energy;
            entropy += occ.ts;
            // An empty channel has no meaningful chemical potential, so it must not be allowed
            // to set the reported one. With a fixed multiplicity the two channels genuinely
            // have separate Fermi levels; what is reported is the highest among the channels
            // that actually hold electrons.
            if channel_electrons[ci] > 1.0e-12 {
                fermi_energy = fermi_energy.max(occ.fermi_energy);
            }

            let mut new_density = RealSpaceBlocks::zeros(&translations, nao);
            for (ki, kp) in k_points.iter().enumerate() {
                let e = &eigen[ki];
                // `P(k)_{μν} = Σ_i f_i c_{μi} conj(c_{νi})`, i.e. `(C diag(f)) C†`.
                //
                // Two things matter here, and the scalar triple loop this replaces had neither.
                //
                // **Only the filled levels are gathered.** The sum runs over orbitals with
                // `f_i ≠ 0` — `n_occ` of them for a gapped system, plus whatever smearing adds —
                // so the products are `nao² · n_occ` rather than `nao³`. The old loop visited
                // every `i` and `continue`d, which skipped the arithmetic but not the traversal.
                //
                // **The products go to the blocked kernel.** This runs once per k point per SCF
                // iteration, on the critical path, and a hand-written `nao³` nest through 2D
                // indexing leaves most of the machine idle — the same rewrite on
                // `pbc::hessian::project_ov` measured 29x at `nao = 32` and 362x at `nao = 128`.
                //
                // Complex, from real blocks: `(W_r + iW_i)(C_r − iC_i)ᵀ`
                // `= (W_r C_rᵀ + W_i C_iᵀ) + i(W_i C_rᵀ − W_r C_iᵀ)`.
                let filled: Vec<usize> = (0..nao)
                    .filter(|&i| {
                        (occ.fractions[ki * nao + i] * channel_capacity * kp.weight).abs() > 1.0e-16
                    })
                    .collect();
                let (mut wr, mut wi) = (
                    Matrix::zeros(nao, filled.len()),
                    Matrix::zeros(nao, filled.len()),
                );
                let (mut cr, mut ci) = (
                    Matrix::zeros(nao, filled.len()),
                    Matrix::zeros(nao, filled.len()),
                );
                for (col, &i) in filled.iter().enumerate() {
                    let f = occ.fractions[ki * nao + i] * channel_capacity * kp.weight;
                    for mu in 0..nao {
                        let (ar, ai) = (e.vectors_re[(mu, i)], e.vectors_im[(mu, i)]);
                        cr[(mu, col)] = ar;
                        ci[(mu, col)] = ai;
                        wr[(mu, col)] = f * ar;
                        wi[(mu, col)] = f * ai;
                    }
                }
                let mut pk_re = wr.matmul_transpose(&cr);
                wi.matmul_transpose_acc(&ci, &mut pk_re, 1.0);
                let mut pk_im = wi.matmul_transpose(&cr);
                wr.matmul_transpose_acc(&ci, &mut pk_im, -1.0);
                let pk = CMatrix {
                    n: nao,
                    re: pk_re,
                    im: pk_im,
                };
                // P(0,T) += e^{-ik·T} P(k); the weight is already folded into f.
                for (ti, t) in translations.iter().enumerate() {
                    let (c, s) = kp.phase(*t);
                    let block = &mut new_density.blocks[ti];
                    for mu in 0..nao {
                        for nu in 0..nao {
                            block[(mu, nu)] += c * pk.re[(mu, nu)] + s * pk.im[(mu, nu)];
                        }
                    }
                }
            }

            focks.push(fock);
            new_channels.push(new_density);
        }

        // The fixed-point residual `P_out − P_in`, flattened once for both the convergence test
        // and the mixer below.
        let mut dp = 0.0_f64;
        diis_input.clear();
        diis_resid.clear();
        for (old_ch, new_ch) in channels.iter().zip(&new_channels) {
            for (old, new) in old_ch.blocks.iter().zip(&new_ch.blocks) {
                for (o, n) in old.as_slice().iter().zip(new.as_slice()) {
                    let r = *n - *o;
                    dp = dp.max(r.abs());
                    diis_input.push(*o);
                    diis_resid.push(r);
                }
            }
        }

        // Energy per cell: 1/2 sum_sigma sum_T P_sigma(0,T) . [H(0,T) + F_sigma(0,T)].
        //
        // **Before the mixing**, so that the `P` contracted here is the one `F` was built from.
        // Through 0.2.2 this block sat after the mix, which made the reported energy
        // `½Tr[P_mixed(H + F(P_in))]` — the energy of no density at all, and inconsistent with
        // the `total_origin` above, which is built from `P_in`. At the fixed point the three
        // agree, so the converged number was right; what it corrupted was `de`, which is half of
        // the convergence test. It measured the mixer as much as the iteration.
        //
        // The convergence test moved with it, so a run that stops here returns `channels` still
        // holding the density this energy belongs to. Mixing first and *then* declaring success
        // returned a density one step past the reported energy — within `p_tol`, so harmless, but
        // it meant the two outputs did not describe the same state.
        electronic = 0.0;
        for (ch, fock) in channels.iter().zip(&focks) {
            let mut sum = core.clone();
            for (block, fb) in sum.blocks.iter_mut().zip(&fock.blocks) {
                for (b, f) in block.as_mut_slice().iter_mut().zip(fb.as_slice()) {
                    *b += *f;
                }
            }
            electronic += 0.5 * ch.contract(&sum);
        }
        // The `+½ Σ_a Z_a V_a` that pairs with the `−V_a` Fock shift — see
        // `crate::fock::long_range_potential` for why the correction takes this form.
        if let Some(d) = delta {
            let v = crate::fock::long_range_potential_from_delta(
                molecule,
                &basis,
                params,
                d,
                &total_origin,
            )?;
            electronic += crate::fock::long_range_energy_from_potential(molecule, params, &v)?;
        }
        let de = (electronic - e_old).abs();
        e_old = electronic;

        if final_pass {
            break;
        }
        if iter > 0 && de < options.e_tol && dp < options.p_tol {
            converged = true;
            // Take the **output** density and spend one more pass on it.
            //
            // `E[P] = ½Tr[P(H + F(P))]` is stationary only on the idempotent manifold, and the
            // mixed input is not on it — a Pulay step is a signed combination of past densities,
            // so it can sit further off idempotency than its distance to the fixed point suggests.
            // Evaluating there leaves a **first-order** error, which is invisible in the energy
            // itself and shows up in anything differenced: a finite-difference gradient on a water
            // dimer was 1.2e-6 eV/Bohr from the analytic one, against 3.5e-7 for plain mixing at
            // the same stated tolerance. The extra pass is one Fock build in about twenty-five,
            // and it makes the returned energy the variational energy of the returned density.
            for (ch, new_ch) in channels.iter_mut().zip(&new_channels) {
                for (block, nb) in ch.blocks.iter_mut().zip(&new_ch.blocks) {
                    block.as_mut_slice().copy_from_slice(nb.as_slice());
                }
            }
            final_pass = true;
            continue;
        }
        if trace {
            eprintln!(
                "[am1 pbc scf] iter {iterations:4}  E {electronic:.12}  dE {de:.3e}  dP {dp:.3e}"
            );
        }

        // Pulay (DIIS) mixing on the real-space blocks, falling back to the linear mix. `dp` above
        // is the residual's sup norm and means the same thing however the next input is built:
        // the criterion measures the iteration, not the mixer.
        let mut extrapolated: Option<Vec<f64>> = None;
        if diis_depth >= 2 {
            if history.len() == diis_depth {
                history.remove(0);
            }
            history.push((diis_input.clone(), diis_resid.clone()));
            let residuals: Vec<&[f64]> = history.iter().map(|(_, r)| r.as_slice()).collect();
            if let Some(c) = pulay_coefficients(&residuals) {
                // A DIIS whose coefficients have blown up is extrapolating along a direction the
                // history does not actually resolve. Dropping the step is cheaper than the
                // divergence, and the history is cleared so the next one starts from a mix that
                // is known to be a contraction.
                let weight: f64 = c.iter().take(history.len()).map(|v| v.abs()).sum();
                if weight <= MAX_DIIS_WEIGHT {
                    let mut next = vec![0.0; diis_input.len()];
                    for (ci, (p_in, r)) in c.iter().zip(&history) {
                        for ((n, p), rv) in next.iter_mut().zip(p_in).zip(r) {
                            *n += ci * (p + options.mixing * rv);
                        }
                    }
                    extrapolated = Some(next);
                } else {
                    history.clear();
                }
            }
        }

        match extrapolated {
            Some(next) => {
                let mut k = 0;
                for ch in channels.iter_mut() {
                    for block in ch.blocks.iter_mut() {
                        for o in block.as_mut_slice().iter_mut() {
                            *o = next[k];
                            k += 1;
                        }
                    }
                }
            }
            None => {
                for (old_ch, new_ch) in channels.iter_mut().zip(&new_channels) {
                    for (old, new) in old_ch.blocks.iter_mut().zip(&new_ch.blocks) {
                        for (o, n) in old.as_mut_slice().iter_mut().zip(new.as_slice()) {
                            *o += options.mixing * (*n - *o);
                        }
                    }
                }
            }
        }
    }

    // Total and spin densities.
    let mut density = RealSpaceBlocks::zeros(&translations, nao);
    for ch in &channels {
        for (tot, c) in density.blocks.iter_mut().zip(&ch.blocks) {
            for (t, v) in tot.as_mut_slice().iter_mut().zip(c.as_slice()) {
                *t += *v;
            }
        }
    }
    let spin_density = if use_uhf {
        let mut s = RealSpaceBlocks::zeros(&translations, nao);
        for (sb, (a, b)) in s
            .blocks
            .iter_mut()
            .zip(channels[0].blocks.iter().zip(&channels[1].blocks))
        {
            for ((v, av), bv) in sb
                .as_mut_slice()
                .iter_mut()
                .zip(a.as_slice())
                .zip(b.as_slice())
            {
                *v = *av - *bv;
            }
        }
        Some(s)
    } else {
        None
    };

    let core_ev = crate::repulsion::core_core_energy_with_neighbors(molecule, params, &neighbors)?
        // The field's **nuclear** half. The electronic half is already in `electronic_ev`, having
        // entered through `H_core`; without this the reported energy is not the one whose
        // stationary point the SCF found, and the force would differentiate a different function.
        + match options.electric_field {
            Some(f) => crate::dipole::field_core_energy(molecule, params, f)?,
            None => 0.0,
        };

    // Mulliken charges from the on-site density block.
    let p0 = density.origin()?;
    let mut charges = Vec::with_capacity(molecule.atoms.len());
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let pop: f64 = (0..n).map(|k| p0[(off + k, off + k)]).sum();
        charges.push(elem.core_charge - pop);
    }

    Ok(PbcResult {
        density,
        spin_density,
        unrestricted: use_uhf,
        electronic_ev: electronic,
        core_ev,
        total_ev: electronic + core_ev,
        band_energy_ev: band_energy,
        fermi_energy_ev: fermi_energy,
        entropy_ev: entropy,
        charges,
        k_points: k_points.len(),
        iterations,
        converged,
        max_image_overlap: pairs.max_image_overlap,
        charged_cell_warning: charged_cell_warning(
            options.charge,
            cell.n_periodic(),
            options.ewald,
        ),
    })
}

#[cfg(test)]
mod spin_channel_tests {
    use super::*;

    fn blocks(translations: &[ImageOffset], nao: usize, seed: f64) -> RealSpaceBlocks {
        let mut b = RealSpaceBlocks::zeros(translations, nao);
        for (t, block) in b.translations.clone().iter().zip(b.blocks.iter_mut()) {
            let tag = (t.n[0] + 2 * t.n[1] + 3 * t.n[2]) as f64;
            for i in 0..nao {
                for j in 0..nao {
                    block[(i, j)] = ((i * 3 + j * 7) as f64 * seed + tag).sin();
                }
            }
        }
        b
    }

    fn result(total: RealSpaceBlocks, spin: Option<RealSpaceBlocks>) -> PbcResult {
        PbcResult {
            density: total,
            spin_density: spin.clone(),
            unrestricted: spin.is_some(),
            electronic_ev: 0.0,
            core_ev: 0.0,
            total_ev: 0.0,
            band_energy_ev: 0.0,
            fermi_energy_ev: 0.0,
            entropy_ev: 0.0,
            charges: Vec::new(),
            k_points: 1,
            iterations: 1,
            converged: true,
            max_image_overlap: 0.0,
            charged_cell_warning: None,
        }
    }

    /// The split has to invert: `P^α + P^β = P` and `P^α − P^β = S`, block by block.
    ///
    /// This is the identity the whole open-shell response rests on, and it is the kind of thing
    /// that is written once and then trusted. It is also asymmetric in a way that invites a slip —
    /// the SCF returns the *total* and the *difference*, not the two channels — so both directions
    /// are checked rather than one.
    #[test]
    fn the_spin_split_inverts_the_total_and_the_difference() {
        let translations = [
            ImageOffset::origin(),
            ImageOffset { n: [1, 0, 0] },
            ImageOffset { n: [-1, 0, 0] },
        ];
        let nao = 5;
        let total = blocks(&translations, nao, 0.31);
        let spin = blocks(&translations, nao, 0.17);
        let scf = result(total.clone(), Some(spin.clone()));

        let (channels, fill) = spin_channel_densities(&scf);
        assert_eq!(channels.len(), 2, "an open shell must give two channels");
        assert_eq!(fill, 1.0, "one unrestricted orbital holds one electron");

        let (alpha, beta) = (&channels[0], &channels[1]);
        let mut worst_sum = 0.0_f64;
        let mut worst_diff = 0.0_f64;
        for t in translations {
            let (a, b) = (alpha.get(t).unwrap(), beta.get(t).unwrap());
            let (p, s) = (total.get(t).unwrap(), spin.get(t).unwrap());
            for i in 0..nao {
                for j in 0..nao {
                    worst_sum = worst_sum.max((a[(i, j)] + b[(i, j)] - p[(i, j)]).abs());
                    worst_diff = worst_diff.max((a[(i, j)] - b[(i, j)] - s[(i, j)]).abs());
                }
            }
        }
        assert!(worst_sum < 1.0e-15, "Pα + Pβ ≠ P by {worst_sum:.3e}");
        assert!(worst_diff < 1.0e-15, "Pα − Pβ ≠ S by {worst_diff:.3e}");
    }

    /// Restricted, one channel carries both spins — and `scale · P^σ` summed over channels has to
    /// be the *same object* either way.
    ///
    /// That equality is what makes forcing UHF on a closed shell reproduce the restricted answer,
    /// and it is a property of the pair `(fill, exchange_scale_for(fill))` rather than of any
    /// particular Fock build. Checked here on the pair itself so the arithmetic is pinned before
    /// anything contracts against it.
    #[test]
    fn the_exchange_weight_and_the_fill_are_consistent() {
        let translations = [ImageOffset::origin()];
        let nao = 4;
        let total = blocks(&translations, nao, 0.29);
        let zero = RealSpaceBlocks::zeros(&translations, nao);

        // Restricted: one channel, the total at half weight, two electrons per orbital.
        let restricted = spin_channel_densities(&result(total.clone(), None));
        assert_eq!(restricted.0.len(), 1);
        assert_eq!(restricted.1, 2.0);
        assert_eq!(exchange_scale_for(restricted.1), 0.5);

        // A closed shell forced through the unrestricted path: zero spin density, so the two
        // channels are each `P/2` at full weight.
        let forced = spin_channel_densities(&result(total.clone(), Some(zero)));
        assert_eq!(forced.1, 1.0);
        assert_eq!(exchange_scale_for(forced.1), 1.0);

        let t = ImageOffset::origin();
        for i in 0..nao {
            for j in 0..nao {
                let restricted_term =
                    exchange_scale_for(restricted.1) * restricted.0[0].get(t).unwrap()[(i, j)];
                let forced_term: f64 = forced
                    .0
                    .iter()
                    .map(|c| exchange_scale_for(forced.1) * c.get(t).unwrap()[(i, j)])
                    .sum::<f64>()
                    / 2.0;
                // `Σ_σ s P^σ` is `2 · 1 · (P/2) = P` unrestricted and `0.5 · P` restricted, so the
                // restricted one is half the unrestricted *sum* — which is the same statement as
                // "the exchange sees `P/2` either way", written where it can be checked.
                assert!(
                    (restricted_term - forced_term).abs() < 1.0e-15,
                    "the exchange sees {restricted_term} restricted and {forced_term} forced-UHF \
                     at ({i},{j})"
                );
            }
        }
    }
}
