// SPDX-License-Identifier: GPL-3.0-or-later

//! Core (one-electron) Hamiltonian assembly.
//!
//! `H_core` holds the diagonal atomic energies `U_ss/U_pp`, the electron–core attraction
//! to every other atom (from the NDDO integrals), and the inter-atomic resonance
//! `H_μν = ½(β_μ + β_ν) S_μν`. The per-pair two-electron integrals are returned alongside
//! for reuse in the Fock build.

use crate::basis::Basis;
use crate::error::Result;
use crate::integrals::{pair_two_electron, PairTwoElec};
use crate::lattice::ImageOffset;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::neighbors::NeighborList;
use crate::overlap::diatom_overlap;
use crate::params::Am1Parameters;
use crate::system::Molecule;

/// Everything [`build_core_with_neighbors`] needs beyond the geometry, the basis and the pair
/// list.
///
/// Gathered into a struct rather than left as four positional arguments because they had already
/// grown past the point where a call site said what it meant — `(…, None, true, None)` is three
/// decisions that read as noise — and because adding the external field would have made it five.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoreBuildOptions {
    /// Distance (Bohr) beyond which an image pair's exchange is tapered off. See
    /// [`PairIntegral::exchange_scale`].
    pub exchange_cutoff: Option<f64>,
    /// Whether the long-range monopole electrostatics is summed by Ewald.
    pub use_ewald: bool,
    /// The Klopman–Ohno `R⁻³` tail beyond the pair list, carrying the real-space cutoff it begins
    /// at; `None` leaves that channel truncated. See
    /// `crate::pbc::ewald::klopman_ohno_tail_matrix`.
    ///
    /// The cutoff rather than a `bool` because the tail *is* a function of where the sum stopped,
    /// and `CoreBuildOptions` does not otherwise carry a cutoff — a `bool` here would have meant
    /// reaching for the neighbour list's largest translation, which jitters under strain.
    pub klopman_ohno_tail: Option<f64>,
    /// Separation (Bohr) beyond which a pair is treated as a monopole; see [`crate::farfield`].
    pub multipole_cutoff: Option<f64>,
    /// Uniform external electric field, in eV per (e·Bohr). **Molecules only** — see
    /// [`crate::dipole`] for why `F·R` is not a lattice-periodic perturbation.
    pub electric_field: Option<Vec3>,
}

impl CoreBuildOptions {
    /// The plain molecular case: every pair exact, no lattice sum, no field.
    pub fn molecular() -> Self {
        Self::default()
    }
}

/// Rotated two-electron integrals for one atom pair, tagged with the ordered atom indices
/// (`a` is the heavy atom when the other is H).
pub struct PairIntegral {
    pub a: usize,
    pub b: usize,
    /// Lattice translation applied to atom `b`. Always the origin for a molecule.
    pub offset: ImageOffset,
    /// Separation of the pair, Bohr.
    pub r: f64,
    /// Weight applied to this pair's **exchange** contribution; 1 for a molecule.
    ///
    /// NDDO carries a genuine two-centre exchange term whose integral decays as `1/R`, so it
    /// is finite only because the density matrix element it contracts against,
    /// `P_{ν_a σ_b}`, decays with the separation. Under a periodic cell the Fock build reads
    /// that element from the *home-cell* density matrix — which at Γ-only sampling is
    /// `P(0,T) = P(Γ)` for every translation, i.e. a density matrix that does not decay at
    /// all. The exchange sum over images then diverges like `Σ_T 1/|T|`: this is the standard
    /// Hartree–Fock exchange divergence at Γ, not an arithmetic slip.
    ///
    /// Measured on a single neutral carbon atom, where the monopole terms must cancel
    /// exactly: −4.440 Ha isolated, −4.740 in a 40 Bohr cell, −10.101 in a 15 Bohr cell.
    ///
    /// The physical fix is a density matrix that decays, which needs k-point sampling. Until
    /// then this weight truncates the exchange where `P` would have decayed, and the
    /// truncation distance is an explicit, documented approximation rather than a silent one.
    pub exchange_scale: f64,
    pub te: PairTwoElec,
}

pub struct CoreHamiltonian {
    pub h_core: Matrix,
    pub pairs: Vec<PairIntegral>,
    /// Long-range monopole correction `Δ_ab`, `nat × nat` in eV, when Ewald summation is in use.
    ///
    /// `None` for a molecule, for a slab or a chain, and whenever it is switched off. See
    /// [`crate::pbc::ewald`] for what it contains and why it is a *correction* rather than the
    /// whole electrostatics: the pair list has already summed the truncated real-space series,
    /// and this replaces its `1/R` part with the exact lattice sum.
    ///
    /// # Why it is applied through the *net* charges
    ///
    /// The obvious way to apply it is to mirror how `γ_ab` already enters: a term in the
    /// electron–core attraction here, one in the Coulomb build, and one in the core–core energy.
    /// That is algebraically exact — the three combine to `½ Σ_ab Q_a Q_b Δ_ab` through the
    /// ordinary `E = ½ Tr[P(H + F)]` — and it is numerically bad.
    ///
    /// `Δ_ab` is *large* whenever the real-space cutoff exceeds the cell, because it has to
    /// cancel a divergent truncated sum: for a single carbon in a 12 Bohr cell with a 40 Bohr
    /// cutoff it is around −165 eV. Split three ways it shifts `H_core` by `+660 eV` and the
    /// Coulomb term by `−660 eV`, which cancel exactly for a neutral system and destroy the
    /// conditioning of the SCF on the way. Measured: a lone neutral carbon stopped converging in
    /// cells below 20 Bohr.
    ///
    /// So it is applied once, through `V_a = Σ_b Δ_ab Q_b` with the **net** charges
    /// `Q_b = Z_b − p_b` — which are small for a neutral system, so nothing large is introduced.
    /// The Fock shift is `−V_a` on atom `a`'s diagonal, and the energy expression carries a
    /// matching `+½ Σ_a Z_a V_a`; see [`crate::fock::long_range_potential`] for the derivation.
    pub long_range: Option<Matrix>,
    /// Far-field monopole treatment of pairs the neighbour list dropped, when
    /// [`crate::scf::Am1Options::multipole_cutoff`] is set. Applied through the same net-charge
    /// potential as `long_range`; see [`crate::farfield`].
    pub far_field: Option<crate::farfield::FarField>,
}

impl CoreHamiltonian {
    /// Long-range correction between two atoms, or zero when Ewald is not in use.
    #[inline]
    pub fn long_range_at(&self, a: usize, b: usize) -> f64 {
        match &self.long_range {
            Some(d) => d[(a, b)],
            None => 0.0,
        }
    }
}

/// Fraction of the exchange kept at separation `r` for a cutoff `r_off`.
///
/// A quintic smoothstep from `0.8 r_off` to `r_off`, not a step.
///
/// Two reasons it has to be smooth. A discontinuous weight makes the energy a discontinuous
/// function of geometry, so forces acquire a delta at the cutoff and molecular dynamics stops
/// conserving energy — and the acceptance test for the periodic work is an actual NPT run.
/// The quintic form has continuous value, first and second derivatives, so the analytic
/// gradient and Hessian stay valid across the taper as well.
///
/// Applied by **distance alone**, deliberately, not by whether the partner happens to be a
/// periodic image. The same physical pair is an intra-cell pair in a supercell and an image
/// pair in the primitive cell; keying the truncation on which one it is would make a
/// supercell disagree with its own primitive cell, which is the sharpest test the periodic
/// code has.
#[inline]
pub fn exchange_taper(r: f64, r_off: f64) -> f64 {
    exchange_taper_scalar::<f64>(r, r_off)
}

/// [`exchange_taper`] generic over the scalar type, so a `Dual` argument yields the taper's
/// own derivative.
///
/// That derivative is not optional. The exchange energy is `taper(r) · C(δ)`, so the force
/// carries a `taper'(r)` term as well as `taper(r) ∂C/∂δ`; dropping it leaves a force that is
/// not the gradient of the energy being reported, and molecular dynamics stops conserving.
#[inline]
pub fn exchange_taper_scalar<S: crate::dual::Scalar>(r: S, r_off: f64) -> S {
    let r_on = 0.8 * r_off;
    if r.val() <= r_on {
        return S::cst(1.0);
    }
    if r.val() >= r_off {
        return S::cst(0.0);
    }
    let t = (r - r_on) * (1.0 / (r_off - r_on));
    // 1 - (6t^5 - 15t^4 + 10t^3)
    let poly = t * t * t * ((t * (t * 6.0 - 15.0)) + 10.0);
    -poly + 1.0
}

/// Return the resonance β for orbital index `orb` (0 = s, else p).
#[inline]
fn beta_of(elem: &crate::params::Am1Element, orb: u8) -> f64 {
    if orb == 0 {
        elem.beta_s
    } else {
        elem.beta_p
    }
}

pub fn build_core(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
) -> Result<CoreHamiltonian> {
    build_core_with_neighbors(
        molecule,
        basis,
        params,
        &NeighborList::molecular(molecule),
        CoreBuildOptions::molecular(),
    )
}

/// Core Hamiltonian over an explicit pair list.
///
/// With [`NeighborList::molecular`] this is the ordinary molecular `H_core`. With a periodic
/// list it is the **Γ-point** Bloch sum `H(Γ)_μν = Σ_T H_μν(0, T)`, because the Bloch phase
/// `e^{ik·T}` is 1 at `k = 0` — so the periodic Γ Hamiltonian is the molecular assembly run
/// over image pairs, and there is no second implementation to keep in step.
///
/// Two details the molecular version could take for granted and this one cannot:
///
/// * Contributions **accumulate**. Several lattice translations connect the same pair of
///   home-cell atoms, so writing a block instead of adding to it would keep only the last
///   image. (For a molecule each block is touched once, so the result is unchanged.)
/// * A pair may be an atom with **its own image**, `a == b` with `T ≠ 0`. Its resonance and
///   core-attraction land on that atom's own diagonal block, and the mirror translation `−T`
///   contributes there too — which the symmetric scatter below produces.
pub fn build_core_with_neighbors(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    neighbors: &NeighborList,
    options: CoreBuildOptions,
) -> Result<CoreHamiltonian> {
    let CoreBuildOptions {
        exchange_cutoff,
        use_ewald,
        klopman_ohno_tail,
        multipole_cutoff,
        electric_field,
    } = options;
    let nao = basis.nao;
    let mut h = Matrix::zeros(nao, nao);

    // Diagonal U_ss / U_pp.
    for (mu, ao) in basis.aos.iter().enumerate() {
        let elem = params.element(ao.z)?;
        h[(mu, mu)] = if ao.orb == 0 { elem.u_ss } else { elem.u_pp };
    }

    // The uniform external field, `h^F = +Σ_α F_α M_α`.
    //
    // Under a cell, allowed exactly when `F` is orthogonal to every **periodic** lattice vector.
    // That is not a convenience: `F·R` shifts by `F·T` under translation by `T`, so the
    // perturbation is lattice-periodic precisely when `F·T = 0` for all `T`, and unbounded
    // otherwise. A slab with a field along its normal and a chain with a transverse field are
    // therefore ordinary calculations; a field along a periodic direction is not, and needs the
    // Berry-phase finite-field construction rather than `F·R`.
    //
    // Through 0.2.1 this refused *any* field under *any* cell, which threw the well-defined cases
    // out with the ill-defined one. Every path reaches the core build, so this is the one place
    // the check has to be.
    if let Some(field) = electric_field {
        crate::pbc::scf::check_periodic_field(molecule, field)?;
        let hf = crate::dipole::field_hamiltonian(molecule, basis, params, field)?;
        for (hv, fv) in h.as_mut_slice().iter_mut().zip(hf.as_slice()) {
            *hv += fv;
        }
    } // Long-range monopole correction, in whichever dimensionality the cell has — 3D Ewald, 2D
      // Parry, or the regularized chain sum. A molecule has no lattice sum to correct and gets
      // `None`; see [`crate::pbc::ewald::LongRangeKernel`].
    let long_range = match (use_ewald, molecule.cell) {
        (true, Some(cell)) => match crate::pbc::ewald::LongRangeKernel::for_lattice(&cell)? {
            Some(kernel) => {
                let mut m =
                    crate::pbc::ewald::LongRangeMonopole::new(molecule, neighbors, &kernel)?;
                if let Some(cutoff) = klopman_ohno_tail {
                    m = m.with_klopman_ohno_tail(molecule, params, cutoff)?;
                }
                Some(m.delta)
            }
            None => None,
        },
        _ => None,
    };

    // No shift is applied to `H_core` here. The correction enters once, through the net-charge
    // potential built in [`crate::fock::long_range_potential`], for the conditioning reason set
    // out on [`CoreHamiltonian::long_range`].

    let far_field =
        crate::farfield::FarField::new(molecule, params, multipole_cutoff.unwrap_or(0.0))?;

    use rayon::prelude::*;

    // Compute the (independent) per-pair integrals in parallel.
    type Computed = (usize, usize, ImageOffset, f64, PairTwoElec, [[f64; 4]; 4]);
    let computed: Vec<Computed> = neighbors
        .pairs
        .par_iter()
        .map(|p| -> Result<Computed> {
            let eu = params.element(molecule.atoms[p.i].z)?;
            let ev = params.element(molecule.atoms[p.j].z)?;
            // Ordered pair: heavy atom first when the other is H. Swapping also flips the
            // displacement, since it points from the first atom to the second.
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

    // Assemble H_core serially from the precomputed per-pair integrals.
    let mut pairs = Vec::with_capacity(computed.len());
    for (a, b, offset, r, te, s_block) in computed {
        let (ea, eb) = (
            params.element(molecule.atoms[a].z)?,
            params.element(molecule.atoms[b].z)?,
        );
        let off_a = basis.atom_offset[a];
        let off_b = basis.atom_offset[b];
        let na = basis.atom_norb[a];
        let nb = basis.atom_norb[b];

        // Electron–core attraction: e1b onto atom a's block, e2a onto atom b's block. When
        // a == b (a self-image pair) both land on the same block, which is right: they are
        // the attraction to the image core at +T and at −T.
        for i in 0..na {
            for j in 0..na {
                h[(off_a + i, off_a + j)] += te.e1b[i][j];
            }
        }
        for i in 0..nb {
            for j in 0..nb {
                h[(off_b + i, off_b + j)] += te.e2a[i][j];
            }
        }

        // Resonance β·S. The pair list holds one representative per physical pair, so each
        // entry stands for both (a, b+T) and (b, a−T); adding the same value to both
        // triangles accounts for the mirror, whose overlap block is the transpose.
        for i in 0..na {
            let bi = beta_of(ea, basis.aos[off_a + i].orb);
            for j in 0..nb {
                let bj = beta_of(eb, basis.aos[off_b + j].orb);
                let value = 0.5 * (bi + bj) * s_block[i][j];
                h[(off_a + i, off_b + j)] += value;
                h[(off_b + j, off_a + i)] += value;
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

    Ok(CoreHamiltonian {
        h_core: h,
        pairs,
        long_range,
        far_field,
    })
}
