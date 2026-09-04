// SPDX-License-Identifier: GPL-3.0-or-later

//! Nuclear Hessian and harmonic vibrational analysis.
//!
//! [`analytic_hessian`] is the primary path: a **fully analytic** RHF Hessian combining a
//! closed-form skeleton second derivative (second-order forward-AD, [`crate::dual2::Dual2`],
//! of the two-center integral kernels) with the CPHF orbital-relaxation response — no finite
//! differences. [`numerical_hessian`] (central differences of the analytic gradient, `3N`
//! columns in parallel on rayon) is retained as the independent validation reference and as
//! the fallback for open-shell / heavy-element systems. Mass-weighting and diagonalization
//! (faer) give harmonic frequencies; overall translations and rotations appear as the ~`6`
//! (5 for linear molecules) near-zero modes.

use crate::data_tables::MASS;
use crate::dual::Scalar;
use crate::error::Result;
use crate::gradient::closed_form_gradient;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::params::Am1Parameters;
use crate::scf::Am1Options;
use crate::system::Molecule;

/// `sqrt(eV / (Å²·amu))` → cm⁻¹ (standard vibrational conversion; 1 unit = 521.47 cm⁻¹).
pub const SQRT_EV_PER_ANG2_AMU_TO_CM: f64 = 521.470_9;

/// One atom pair's `3 × 3` skeleton Hessian block, tagged with the pair it belongs to.
///
/// The pair energy depends only on the displacement `R_b − R_a`, so this single block scatters
/// as `+H` onto the `(a,a)` and `(b,b)` diagonal blocks and `−H` onto `(a,b)` and `(b,a)`.
type HessianBlock = (usize, usize, [[f64; 3]; 3]);

#[derive(Clone, Debug)]
pub struct VibrationalModes {
    /// Cartesian Hessian (eV/Bohr²), symmetric, size `3N × 3N`.
    pub hessian: Matrix,
    /// Harmonic frequencies (cm⁻¹), ascending; negative = imaginary (saddle/unconverged).
    pub frequencies_cm: Vec<f64>,
    /// Mass-weighted eigenvalues (eV/(Å²·amu)).
    pub eigenvalues: Vec<f64>,
    /// Mass-weighted normal-mode eigenvectors, `3N × 3N`, **columns** are modes and are
    /// orthonormal (`LᵀL = I`). Column `k` matches `frequencies_cm[k]`.
    ///
    /// These, not the Cartesian displacements, are what `∂μ/∂Q_k = Σ_j (∂μ/∂R_j) L_{jk}/√m_j`
    /// needs; see [`crate::ir`]. They used to be discarded, which is why there was no way to get
    /// an infrared intensity out of a frequency calculation.
    pub modes: Matrix,
    /// Cartesian displacements `M^{−1/2} L`, `3N × 3N`, columns are modes.
    ///
    /// Deliberately **not** renormalized: the normalization lives on `modes`, and rescaling
    /// these would silently change what `∂μ/∂Q` means.
    pub cartesian_displacements: Matrix,
    /// For each mode, the fraction of its norm lying in the translation/rotation subspace, `0…1`.
    ///
    /// A rigid-body mode scores ≈ 1 and a vibration ≈ 0. Reported rather than filtered by a
    /// frequency threshold because "is this a vibration" is a question about the eigenvector, and
    /// a linear molecule has five rigid-body modes where a bent one has six — data, not an
    /// assumption about `3N − 6`.
    pub translation_rotation_overlap: Vec<f64>,
}

/// Cartesian Hessian (eV/Bohr²) by central differences of the analytic gradient.
pub fn numerical_hessian(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    step: f64,
) -> Result<Matrix> {
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;

    // Column j = (grad(+step·e_j) − grad(−step·e_j)) / (2·step); columns are independent.
    let columns: Vec<Result<Vec<f64>>> = (0..ndof)
        .into_par_iter()
        .map(|j| {
            let (atom, k) = (j / 3, j % 3);
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            displace(&mut plus.atoms[atom].position, k, step);
            displace(&mut minus.atoms[atom].position, k, -step);
            let gp = closed_form_gradient(&plus, params, options)?;
            let gm = closed_form_gradient(&minus, params, options)?;
            let mut col = vec![0.0; ndof];
            for a in 0..nat {
                for c in 0..3 {
                    let idx = 3 * a + c;
                    col[idx] = (component(&gp.gradient[a], c) - component(&gm.gradient[a], c))
                        / (2.0 * step);
                }
            }
            Ok(col)
        })
        .collect();

    let mut h = Matrix::zeros(ndof, ndof);
    for (j, col) in columns.into_iter().enumerate() {
        let col = col?;
        for (i, &v) in col.iter().enumerate() {
            h[(i, j)] = v;
        }
    }
    // Symmetrize (FD asymmetry).
    let mut hs = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            hs[(i, j)] = 0.5 * (h[(i, j)] + h[(j, i)]);
        }
    }
    Ok(hs)
}

/// Harmonic vibrational analysis at the given geometry (should be a stationary point).
pub fn vibrational_analysis(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    step: f64,
) -> Result<VibrationalModes> {
    // CPHF analytic Hessian (no SCF re-runs); UHF takes the coupled α/β path internally.
    let hessian = analytic_hessian(molecule, params, options, step)?;
    vibrational_analysis_from_hessian(molecule, hessian)
}

/// [`vibrational_analysis`] on a Hessian that has already been computed.
///
/// Split out so that infrared intensities — which need the Hessian *and* the CPHF response that
/// produced it — do not pay for a second Hessian to get the normal modes. See [`crate::ir`].
pub fn vibrational_analysis_from_hessian(
    molecule: &Molecule,
    hessian: Matrix,
) -> Result<VibrationalModes> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;

    // Mass-weight: H'_ij = H_ij / sqrt(m_i m_j), converting eV/Bohr² → eV/Å².
    let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
    let mass_of = |dof: usize| MASS[molecule.atoms[dof / 3].z as usize];
    let mut mw = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let mij = (mass_of(i) * mass_of(j)).sqrt();
            mw[(i, j)] = hessian[(i, j)] * a0_sq / mij; // eV/(Å²·amu)
        }
    }
    let (eigs, vecs) = symmetric_eigen(&mw)?;
    let frequencies_cm: Vec<f64> = eigs
        .iter()
        .map(|&lam| {
            if lam >= 0.0 {
                SQRT_EV_PER_ANG2_AMU_TO_CM * lam.sqrt()
            } else {
                -SQRT_EV_PER_ANG2_AMU_TO_CM * (-lam).sqrt()
            }
        })
        .collect();

    // Cartesian displacements `M^{−1/2} L`, column by column.
    let mut cartesian = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        let inv_sqrt_m = 1.0 / mass_of(i).sqrt();
        for k in 0..ndof {
            cartesian[(i, k)] = vecs[(i, k)] * inv_sqrt_m;
        }
    }

    let tr_basis = translation_rotation_basis(molecule);
    let translation_rotation_overlap = (0..ndof)
        .map(|k| {
            tr_basis
                .iter()
                .map(|b| {
                    let dot: f64 = (0..ndof).map(|i| b[i] * vecs[(i, k)]).sum();
                    dot * dot
                })
                .sum::<f64>()
                .min(1.0)
        })
        .collect();

    Ok(VibrationalModes {
        hessian,
        frequencies_cm,
        eigenvalues: eigs,
        modes: vecs,
        cartesian_displacements: cartesian,
        translation_rotation_overlap,
    })
}

/// An orthonormal basis for the rigid-body subspace, in **mass-weighted** coordinates.
///
/// Three translations `√m_a e_α` and three rotations `√m_a (e_α × (R_a − R_cm))`, orthonormalized
/// by modified Gram–Schmidt with near-null vectors dropped. A linear molecule yields five vectors
/// and an atom three, because the rotation about the molecular axis (or any rotation of a single
/// atom) has zero norm and falls out — which is why the count is discovered rather than assumed.
fn translation_rotation_basis(molecule: &Molecule) -> Vec<Vec<f64>> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mass = |a: usize| MASS[molecule.atoms[a].z as usize];

    let total: f64 = (0..nat).map(mass).sum();
    let mut com = Vec3::zero();
    if total > 0.0 {
        for a in 0..nat {
            com += molecule.atoms[a].position * mass(a);
        }
        com = com / total;
    }

    let mut raw: Vec<Vec<f64>> = Vec::with_capacity(6);
    for axis in 0..3 {
        let mut v = vec![0.0; ndof];
        for a in 0..nat {
            v[3 * a + axis] = mass(a).sqrt();
        }
        raw.push(v);
    }
    for axis in 0..3 {
        let e = match axis {
            0 => Vec3::new(1.0, 0.0, 0.0),
            1 => Vec3::new(0.0, 1.0, 0.0),
            _ => Vec3::new(0.0, 0.0, 1.0),
        };
        let mut v = vec![0.0; ndof];
        for a in 0..nat {
            let d = e.cross(molecule.atoms[a].position - com) * mass(a).sqrt();
            v[3 * a] = d.x;
            v[3 * a + 1] = d.y;
            v[3 * a + 2] = d.z;
        }
        raw.push(v);
    }

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(6);
    for mut v in raw {
        for b in &basis {
            let dot: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
            for (x, y) in v.iter_mut().zip(b) {
                *x -= dot * y;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        // A rotation about a linear molecule's own axis has zero norm; so does any rotation of a
        // single atom. Dropping it here is what makes the count 5 or 3 rather than always 6.
        if norm > 1.0e-8 {
            for x in v.iter_mut() {
                *x /= norm;
            }
            basis.push(v);
        }
    }
    basis
}

/// **Analytic (CPHF) Cartesian Hessian** (eV/Bohr²).
///
/// `H_ab = E^(2,skel)_ab + Σ_μν F^a_μν (∂P/∂R_b)_μν`, where:
///
/// * the **skeleton** (fixed-density) second derivative `E^(2,skel)` is computed in **closed
///   form** by second-order forward-mode automatic differentiation ([`crate::dual2::Dual2`])
///   of the two-center integral kernels — resonance `β·S`, electron–core attraction, the
///   Dewar–Sabelli–Klopman two-electron integrals, and the AM1 core–core repulsion — with **no
///   finite differences**; and
/// * the density response `∂P/∂R_b` solves the coupled-perturbed (CPHF) equations, whose kernel
///   is the **orbital Hessian** (the same object a second-order SOSCF would use). This is done
///   entirely in the compact MO occupied–virtual subspace (`H_relax[a][b] = 4 G^a·U^b`), so the
///   working set is `O(ndof · n_occ · n_vir)` — no dense `ndof × nao²` derivative-Fock or
///   response intermediates are ever materialized (memory-lean and rayon-parallel over DOFs).
///
/// Fully analytic for both closed-shell RHF and open-shell UHF (the latter via [`analytic_hessian_uhf`]
/// with coupled α/β CPHF), across all valence shells (`n ≤ 3` analytic kernel, `n ≥ 4` via AD
/// through the numerical overlap quadrature). No finite differences.
pub fn analytic_hessian(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    step: f64,
) -> Result<Matrix> {
    Ok(analytic_hessian_with_response(molecule, params, options, step)?.hessian)
}

/// One spin channel's converged CPHF solution, in the compact MO occupied–virtual block.
#[derive(Clone, Debug)]
pub struct ResponseChannel {
    /// `U^j`, one `n_vir × n_occ` block per Cartesian degree of freedom `j = 3a + axis`.
    pub u_ov: Vec<Matrix>,
    /// `G^j`, the skeleton derivative Fock projected to the same block.
    pub g_ov: Vec<Matrix>,
    /// Occupied MO coefficients, `nao × n_occ`.
    pub occupied: Matrix,
    /// Virtual MO coefficients, `nao × n_vir`.
    pub virtuals: Matrix,
    /// Electrons per orbital in this channel: 2 for RHF, 1 for one UHF spin.
    pub occupation: f64,
}

impl ResponseChannel {
    /// This channel's AO-basis first-order density `∂P/∂R_j`, built on demand.
    ///
    /// On demand, and not stored, because the `3N` response densities are `O(ndof · nao²)` — the
    /// one genuinely large object in the whole calculation, and almost never all wanted at once.
    /// `U` itself is `O(ndof · n_occ · n_vir)` and is kept.
    pub fn response_density(&self, dof: usize) -> Matrix {
        ao_response_density_w(
            &self.u_ov[dof],
            &self.virtuals,
            &self.occupied,
            self.occupation,
        )
    }
}

/// An analytic Hessian together with the first-order orbital response it was built from.
///
/// The CPHF solve *is* the expensive part of a Hessian — on a 150-atom cluster its Fock builds
/// are 65 % of the whole calculation — and `U` is exactly what the infrared atomic polar tensor
/// and any other first-order property needs. Returning it costs nothing and saves recomputing it.
#[derive(Clone, Debug)]
pub struct HessianResponse {
    /// Cartesian Hessian (eV/Bohr²), symmetric, `3N × 3N`.
    pub hessian: Matrix,
    /// The converged SCF the response was built on.
    pub scf: crate::scf::Am1Result,
    /// α (or, for RHF, the single restricted) channel.
    pub alpha: ResponseChannel,
    /// β channel, present only for an unrestricted run.
    pub beta: Option<ResponseChannel>,
    /// What each perturbation's CPHF solve actually did.
    pub cphf: Vec<CphfOutcome>,
}

impl HessianResponse {
    /// The **total** AO first-order density `∂P/∂R_j`, summed over spin channels.
    pub fn response_density(&self, dof: usize) -> Matrix {
        let mut r = self.alpha.response_density(dof);
        if let Some(b) = &self.beta {
            let rb = b.response_density(dof);
            for (x, y) in r.as_mut_slice().iter_mut().zip(rb.as_slice()) {
                *x += *y;
            }
        }
        r
    }

    /// Number of Cartesian degrees of freedom, `3N`.
    pub fn ndof(&self) -> usize {
        self.alpha.u_ov.len()
    }
}

/// [`analytic_hessian`], keeping the CPHF response instead of discarding it.
pub fn analytic_hessian_with_response(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    step: f64,
) -> Result<HessianResponse> {
    use crate::dual2::Dual2;

    let scf = crate::scf::run_am1(molecule, params, options)?;
    if scf.unrestricted {
        let _ = step; // UHF path is fully analytic (no finite-difference step)
        return analytic_hessian_uhf(molecule, params, options, scf);
    }

    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = crate::basis::Basis::build(molecule, params)?;
    // The same pair list the SCF used. For a molecule that is every pair; for a periodic cell it
    // is every image pair within the real-space cutoff — which is what makes this function the
    // **Γ-point periodic Hessian** as well as the molecular one, with no second implementation.
    //
    // The reason one implementation covers both is specific to Γ: the Bloch phase `e^{ik·T}` is 1
    // at `k = 0`, so `P(0,T) = P(Γ)` for *every* translation and the density that multiplies each
    // image pair's integrals is the same matrix the molecular code already has. Away from Γ that
    // is false and this would not work.
    let neighbors = crate::neighbors::NeighborList::build_screened(
        molecule,
        options.realspace_cutoff,
        options.multipole_cutoff,
    );
    let core = crate::hamiltonian::build_core_with_neighbors(
        molecule,
        &basis,
        params,
        &neighbors,
        options.core_build(),
    )?;
    let p = scf.density.clone();
    let c = scf.mo_coeff.clone();
    let eps = scf.mo_energies.clone();
    let n_occ = scf.n_occ;

    // 1) Skeleton (fixed-density) second derivative — fully analytic via second-order AD
    //    (Dual2) of each two-center pair's energy contribution E_pair(R_ab). Since E_pair
    //    depends only on the displacement R_ab = R_b − R_a, its 3×3 Hessian block scatters as
    //    +H onto the (a,a) and (b,b) diagonal blocks and −H onto the (a,b)/(b,a) blocks.
    let mut hess = Matrix::zeros(ndof, ndof);
    let beta = |elem: &crate::params::Am1Element, orb: u8| {
        if orb == 0 {
            elem.beta_s
        } else {
            elem.beta_p
        }
    };
    let blocks: Vec<Result<HessianBlock>> = {
        let _t = crate::timing::Timer::start("hess:skeleton");
        use rayon::prelude::*;
        neighbors
            .pairs
            .par_iter()
            .map(|pair| -> Result<HessianBlock> {
                let eu = params.element(molecule.atoms[pair.i].z)?;
                let ev = params.element(molecule.atoms[pair.j].z)?;
                // Heavy atom first when the other is H; swapping flips the displacement, which
                // points from the first atom to the second.
                let (a, b, delta) = if eu.has_p() || !ev.has_p() {
                    (pair.i, pair.j, pair.delta)
                } else {
                    (pair.j, pair.i, pair.delta * -1.0)
                };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let pa = molecule.atoms[a].position;
                // The displacement, not the difference of positions: under a cell it carries the
                // lattice translation, and for a molecule the two are the same thing.
                let pb = pa + delta;
                let dvec = [
                    Dual2::var(delta.x, 0),
                    Dual2::var(delta.y, 1),
                    Dual2::var(delta.z, 2),
                ];
                let te = crate::integrals::pair_two_electron_g::<Dual2>(ea, eb, dvec);
                let s = crate::overlap::diatom_overlap_dual2(ea, pa, eb, pb)?;
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                let mut epair = Dual2::constant(0.0);
                // Resonance β·S energy (both μν and νμ orderings → factor (β_i+β_j)).
                for i in 0..na {
                    let bi = beta(ea, basis.aos[oa + i].orb);
                    for j in 0..nb {
                        let bj = beta(eb, basis.aos[ob + j].orb);
                        let coef = p[(oa + i, ob + j)] * (bi + bj);
                        epair = epair + s[i][j] * coef;
                    }
                }
                // Electron–core attraction.
                for i in 0..na {
                    for j in 0..na {
                        epair = epair + te.e1b[i][j] * p[(oa + i, oa + j)];
                    }
                }
                for k in 0..nb {
                    for l in 0..nb {
                        epair = epair + te.e2a[k][l] * p[(ob + k, ob + l)];
                    }
                }
                // Two-electron Coulomb (J), fixed density.
                for mu in 0..na {
                    for nu in 0..na {
                        for la in 0..nb {
                            for si in 0..nb {
                                let coul = p[(oa + mu, oa + nu)] * p[(ob + la, ob + si)];
                                epair = epair + te.two_e(mu, nu, la, si) * coul;
                            }
                        }
                    }
                }
                let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();

                // Two-electron exchange (K), accumulated separately so the periodic taper can
                // multiply the whole thing. The taper is a function of the separation, so its
                // own first *and second* derivatives enter through the product rule — dropping
                // either would leave a Hessian that is not the second derivative of the energy
                // being reported. Instantiating it at `Dual2` is what supplies them.
                let mut exchange = Dual2::constant(0.0);
                for mu in 0..na {
                    for nu in 0..na {
                        for la in 0..nb {
                            for si in 0..nb {
                                let coef = -0.5 * p[(oa + mu, ob + la)] * p[(oa + nu, ob + si)];
                                exchange = exchange + te.two_e(mu, nu, la, si) * coef;
                            }
                        }
                    }
                }
                epair = epair
                    + match options.exchange_cutoff {
                        Some(rc) => {
                            exchange * crate::hamiltonian::exchange_taper_scalar::<Dual2>(r, rc)
                        }
                        None => exchange,
                    };

                // Core–core repulsion (function of |R_ab|).
                epair = epair
                    + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                        ea,
                        eb,
                        molecule.atoms[a].z,
                        molecule.atoms[b].z,
                        r,
                    );
                Ok((a, b, epair.h))
            })
            .collect()
    };
    for blk in blocks {
        let (a, b, hb) = blk?;
        for i in 0..3 {
            for j in 0..3 {
                let val = hb[i][j];
                hess[(3 * a + i, 3 * a + j)] += val;
                hess[(3 * b + i, 3 * b + j)] += val;
                hess[(3 * a + i, 3 * b + j)] -= val;
                hess[(3 * b + i, 3 * a + j)] -= val;
            }
        }
    }

    // 1b) Long-range monopole correction, fixed-charge part. Same pairwise structure as the
    //     skeleton above but assembled from Ewald derivatives rather than `Dual2`, because
    //     `Scalar` has no `erfc`; `energy_hessian` returns the already-scattered blocks.
    //
    //     Its density-response counterpart rides along in the CPHF below, through the `−V_a`
    //     term that `skeleton_fock_ov` adds to the perturbed Fock matrix.
    if let Some((_, ewald)) =
        crate::pbc::ewald::LongRangeMonopole::for_molecule(molecule, &neighbors, options.ewald)?
    {
        let charges = crate::pbc::ewald::net_charges(molecule, &basis, params, &p)?;
        for (c, d, block) in crate::pbc::ewald::LongRangeMonopole::energy_hessian(
            molecule, &neighbors, &ewald, &charges,
        )? {
            for i in 0..3 {
                for j in 0..3 {
                    hess[(3 * c + i, 3 * d + j)] += block[i][j];
                    if c != d {
                        hess[(3 * d + j, 3 * c + i)] += block[i][j];
                    }
                }
            }
        }
    }

    // 2+3) Orbital-relaxation (CPHF) term in the compact MO occupied–virtual subspace:
    //   H_relax[a][b] = 4 Σ_{ov} G^a_{ov} U^b_{ov},
    // where G^t is the skeleton derivative Fock projected to the occ–virt block (n_vir × n_occ)
    // and U^b solves the coupled-perturbed equations against the orbital Hessian. Keeping
    // everything in the n_vir × n_occ block (never ndof × nao²) makes this both fast and
    // memory-lean: the response density is formed by matrix products (O(nao²·n_occ)), not the
    // O(n_occ·n_vir·nao²) outer-product loop, and no 3N full Fock/response matrices are stored.
    let nvir = basis.nao - n_occ;
    let cv = submatrix_cols(&c, n_occ, nvir); // virtual MOs, nao × n_vir
    let co = submatrix_cols(&c, 0, n_occ); // occupied MOs, nao × n_occ
    let mut channel = ResponseChannel {
        u_ov: vec![Matrix::zeros(nvir, n_occ); ndof],
        g_ov: vec![Matrix::zeros(nvir, n_occ); ndof],
        occupied: co.clone(),
        virtuals: cv.clone(),
        occupation: 2.0,
    };
    let mut outcomes_out: Vec<CphfOutcome> = Vec::new();
    if nvir > 0 && n_occ > 0 {
        let denom = ov_denominators(&eps, n_occ, nvir); // ε_i − ε_a, n_vir × n_occ

        // Skeleton derivative Fock ov-blocks, built one atom at a time (peak memory O(nao²)).
        let gov = {
            let _t = crate::timing::Timer::start("hess:skeleton_fock_ov");
            skeleton_fock_ov(
                &SkeletonContext {
                    molecule,
                    params,
                    basis: &basis,
                    neighbors: &neighbors,
                    exchange_cutoff: options.exchange_cutoff,
                    use_ewald: options.ewald,
                    electric_field: options.electric_field,
                },
                &p,
                OvBlocks { cv: &cv, co: &co },
            )?
        };

        // CPHF response ov-blocks — independent per DOF, solved in parallel. Everything the
        // solve needs but the perturbation itself is invariant across them, and is bundled so
        // the solver and its fallback cannot be handed different Hamiltonians.
        let ctx = CphfContext {
            cv: &cv,
            co: &co,
            molecule,
            params,
            basis: &basis,
            core: &core,
        };
        let solved: Vec<(Matrix, CphfOutcome)> = {
            let _t = crate::timing::Timer::start("hess:cphf");
            use rayon::prelude::*;
            gov.par_iter()
                .map(|g| cphf_ov(&ctx, g, &denom))
                .collect::<Result<Vec<_>>>()?
        };
        let (uov, outcomes): (Vec<Matrix>, Vec<CphfOutcome>) = solved.into_iter().unzip();
        check_cphf(&outcomes)?;

        // Assemble H_relax[a][b] = 4 G^a : U^b.
        //
        // Matrix products, not `ndof²` independent Frobenius dots: each `G^a` block is a row of
        // an `ndof × n_ov` matrix and each `U^b` a row of another, so the whole thing is `G Uᵀ`.
        // Written as dots it re-reads every `G` row `ndof` times and is memory bound; a blocked
        // kernel reads each row once and reuses it out of cache.
        //
        // **Tiled**, so the stacking does not cost an order of memory. Stacking `G` and `U` whole
        // would be two more `ndof × n_ov` buffers — `O(N³)`, doubling the largest array the
        // Hessian holds, since `gov` and `uov` are already that size. Copying `TILE`-row blocks
        // instead bounds the extra at `O(N²)`, and the redundant copying it trades for is
        // `1/TILE` of the arithmetic, which is nothing.
        let n_ov = gov[0].as_slice().len();
        if n_ov > 0 {
            const TILE: usize = 64;
            let stack = |rows: &[Matrix], from: usize, to: usize| -> Matrix {
                let mut m = Matrix::zeros(to - from, n_ov);
                for (r, src) in rows[from..to].iter().enumerate() {
                    m.as_mut_slice()[r * n_ov..(r + 1) * n_ov].copy_from_slice(src.as_slice());
                }
                m
            };
            let mut a0 = 0;
            while a0 < ndof {
                let a1 = (a0 + TILE).min(ndof);
                let g_tile = stack(&gov, a0, a1);
                let mut b0 = 0;
                while b0 < ndof {
                    let b1 = (b0 + TILE).min(ndof);
                    let u_tile = stack(&uov, b0, b1);
                    let block = g_tile.matmul_transpose(&u_tile);
                    for a in a0..a1 {
                        for b in b0..b1 {
                            hess[(a, b)] += 4.0 * block[(a - a0, b - b0)];
                        }
                    }
                    b0 = b1;
                }
                a0 = a1;
            }
        }
        channel.u_ov = uov;
        channel.g_ov = gov;
        outcomes_out = outcomes;
    }

    // Symmetrize.
    let mut sym = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            sym[(i, j)] = 0.5 * (hess[(i, j)] + hess[(j, i)]);
        }
    }
    Ok(HessianResponse {
        hessian: sym,
        scf,
        alpha: channel,
        beta: None,
        cphf: outcomes_out,
    })
}

/// Copy `count` columns of `c` starting at `start` into a fresh `nao × count` matrix.
fn submatrix_cols(c: &Matrix, start: usize, count: usize) -> Matrix {
    let nao = c.rows;
    let mut m = Matrix::zeros(nao, count);
    for mu in 0..nao {
        for k in 0..count {
            m[(mu, k)] = c[(mu, start + k)];
        }
    }
    m
}

/// Orbital-energy denominators `ε_i − ε_a` (occupied `i`, virtual `a`), as an `n_vir × n_occ`
/// matrix — the diagonal of the uncoupled orbital Hessian.
fn ov_denominators(eps: &[f64], n_occ: usize, nvir: usize) -> Matrix {
    let mut d = Matrix::zeros(nvir, n_occ);
    for a in 0..nvir {
        for i in 0..n_occ {
            d[(a, i)] = eps[i] - eps[n_occ + a];
        }
    }
    d
}

/// Project an AO-basis matrix `f` onto the MO occupied–virtual block `Cvᵀ F Co` (n_vir × n_occ).
fn project_ov(f: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    // Sequential, and transpose-free: this runs inside the rayon loop over the `3N`
    // perturbations, where faer's own threads fight the outer pool for the same workers. See
    // `Matrix::matmul_seq`.
    let m = f.matmul_seq(co); // nao × n_occ
    cv.transpose_matmul_seq(&m) // n_vir × n_occ
}

/// Skeleton derivative Fock, projected to the MO occ–virt block, one entry per Cartesian DOF.
///
/// Built **one atom at a time**: for atom `c` its three axis-derivative Fock matrices are
/// accumulated from the pairs `{c, x}` and immediately projected to the compact `n_vir × n_occ`
/// block, so peak memory is `O(nao²)` (a few transient matrices per thread) rather than
/// `O(ndof · nao²)`. Each pair's dual integrals are evaluated twice overall (once per endpoint),
/// a negligible cost next to the CPHF solve.
/// Everything the skeleton derivative Fock needs that is a property of the **calculation** rather
/// than of the perturbation or the spin channel.
///
/// The counterpart of [`CphfContext`], and introduced for the same reason: these arguments were
/// passed positionally to two long signatures, both of which carried an
/// `#[allow(clippy::too_many_arguments)]` to say so. Bundling them separates the setting from the
/// problem — `p` and the coefficient blocks are the problem — and makes it impossible for the
/// restricted and unrestricted skeletons to be handed different Hamiltonians, which is a mistake
/// this file has made before: through 0.2.0 the unrestricted path built a molecular Hamiltonian
/// whatever the options said.
struct SkeletonContext<'a> {
    molecule: &'a Molecule,
    params: &'a Am1Parameters,
    basis: &'a crate::basis::Basis,
    neighbors: &'a crate::neighbors::NeighborList,
    exchange_cutoff: Option<f64>,
    use_ewald: bool,
    electric_field: Option<Vec3>,
}

/// One spin channel's occupied and virtual coefficient blocks.
///
/// They are always used together and always in that pairing; passing them as four loose matrices
/// is how `cva`/`coa`/`cvb`/`cob` end up transposed at a call site with nothing to catch it.
#[derive(Clone, Copy)]
struct OvBlocks<'a> {
    cv: &'a Matrix,
    co: &'a Matrix,
}

fn skeleton_fock_ov(
    ctx: &SkeletonContext<'_>,
    p: &Matrix,
    blocks: OvBlocks<'_>,
) -> Result<Vec<Matrix>> {
    let SkeletonContext {
        molecule,
        params,
        basis,
        neighbors,
        exchange_cutoff,
        use_ewald,
        electric_field,
    } = *ctx;
    let (cv, co) = (blocks.cv, blocks.co);
    use crate::integrals::pair_two_electron_dual;
    use crate::overlap::diatom_overlap_dual;
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let nao = basis.nao;
    let beta = |e: &crate::params::Am1Element, orb: u8| if orb == 0 { e.beta_s } else { e.beta_p };

    // Ewald data for the long-range correction's contribution to the perturbed Fock, or `None`
    // when the correction does not apply. Built once, outside the per-atom loop.
    let long_range = match (
        use_ewald.then_some(()),
        molecule.cell.filter(|c| c.n_periodic() >= 1),
    ) {
        (Some(()), Some(cell)) => match crate::pbc::ewald::LongRangeKernel::for_lattice(&cell)? {
            Some(kernel) => {
                let charges = crate::pbc::ewald::net_charges(molecule, basis, params, p)?;
                Some((kernel, cell, charges))
            }
            None => None,
        },
        _ => None,
    };

    // Per atom: the three projected ov-blocks (x, y, z).
    let per_atom: Vec<Result<[Matrix; 3]>> = (0..nat)
        .into_par_iter()
        .map(|c| -> Result<[Matrix; 3]> {
            let mut fmat = [
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
            ];
            // Every pair this atom takes part in. Under a cell an atom appears in several image
            // pairs with the same partner, and a self-image pair `(c, c+T)` contributes nothing:
            // moving `c` moves its image with it, so the separation — and the energy — is
            // unchanged. The `sign` below produces exactly that cancellation.
            for pair in &neighbors.pairs {
                if pair.i != c && pair.j != c {
                    continue;
                }
                let eu = params.element(molecule.atoms[pair.i].z)?;
                let ev = params.element(molecule.atoms[pair.j].z)?;
                let (a, b, delta) = if eu.has_p() || !ev.has_p() {
                    (pair.i, pair.j, pair.delta)
                } else {
                    (pair.j, pair.i, pair.delta * -1.0)
                };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let pa = molecule.atoms[a].position;
                let pb = pa + delta;
                // E_pair depends on the displacement R_b + T − R_a; ∂/∂R_c is +∂/∂delta when c
                // is the second atom and −∂/∂delta when it is the first. A self-image pair has
                // c as *both*, and the two contributions cancel, which is correct.
                let mut sign = 0.0;
                if b == c {
                    sign += 1.0;
                }
                if a == c {
                    sign -= 1.0;
                }
                if sign == 0.0 {
                    continue;
                }
                let te = pair_two_electron_dual(ea, eb, delta);
                let s = diatom_overlap_dual(ea, pa, eb, pb)?;
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);
                // The exchange taper and its derivative, for the periodic case.
                let (taper_v, taper_d) = match exchange_cutoff {
                    Some(rc) => {
                        let r = crate::dual::Dual {
                            v: pair.r,
                            d: [delta.x / pair.r, delta.y / pair.r, delta.z / pair.r],
                        };
                        let t =
                            crate::hamiltonian::exchange_taper_scalar::<crate::dual::Dual>(r, rc);
                        (t.v, t.d)
                    }
                    None => (1.0, [0.0; 3]),
                };

                for axis in 0..3 {
                    let fm = &mut fmat[axis];
                    // Resonance β·S.
                    for i in 0..na {
                        let bi = beta(ea, basis.aos[oa + i].orb);
                        for j in 0..nb {
                            let bj = beta(eb, basis.aos[ob + j].orb);
                            let val = sign * 0.5 * (bi + bj) * s[i][j].d[axis];
                            fm[(oa + i, ob + j)] += val;
                            fm[(ob + j, oa + i)] += val;
                        }
                    }
                    // Electron–core attraction.
                    for i in 0..na {
                        for j in 0..na {
                            fm[(oa + i, oa + j)] += sign * te.e1b[i][j].d[axis];
                        }
                    }
                    for k in 0..nb {
                        for l in 0..nb {
                            fm[(ob + k, ob + l)] += sign * te.e2a[k][l].d[axis];
                        }
                    }
                    // Two-electron Coulomb (J).
                    for mu in 0..na {
                        for nu in 0..na {
                            let mut acc = 0.0;
                            for la in 0..nb {
                                for si in 0..nb {
                                    acc += p[(ob + la, ob + si)] * te.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            fm[(oa + mu, oa + nu)] += sign * acc;
                        }
                    }
                    for la in 0..nb {
                        for si in 0..nb {
                            let mut acc = 0.0;
                            for mu in 0..na {
                                for nu in 0..na {
                                    acc += p[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            fm[(ob + la, ob + si)] += sign * acc;
                        }
                    }
                    // Two-electron exchange (K), with the taper's own derivative. The energy
                    // contribution is `taper(r) · K`, so its derivative carries both
                    // `taper' · K` and `taper · K'`; the first is what makes the periodic
                    // response consistent with the periodic energy.
                    for mu in 0..na {
                        for la in 0..nb {
                            let mut d_acc = 0.0;
                            let mut v_acc = 0.0;
                            for nu in 0..na {
                                for si in 0..nb {
                                    let w = te.two_e(mu, nu, la, si);
                                    let pv = p[(oa + nu, ob + si)];
                                    d_acc += pv * w.d[axis];
                                    v_acc += pv * w.v;
                                }
                            }
                            let val = sign * (-0.5) * (taper_v * d_acc + taper_d[axis] * v_acc);
                            fm[(oa + mu, ob + la)] += val;
                            fm[(ob + la, oa + mu)] += val;
                        }
                    }
                }
            }
            // Long-range monopole correction. The Fock carries `−V_a` on atom `a`'s diagonal
            // with `V_a = Σ_b Δ_ab Q_b`, so at fixed charges the perturbed Fock carries
            // `−∂V_a/∂R_c = −Σ_b g_ab·(δ_bc − δ_ac) Q_b` — the piece that lets the CPHF see the
            // charges rearranging in response to the displacement.
            if let Some((ewald, lattice, charges)) = &long_range {
                for a in 0..nat {
                    // `g_ab ≡ ∂Δ_ab/∂r` at `r = R_b − R_a`, so `∂Δ_ab/∂R_c = g_ab (δ_bc − δ_ac)`.
                    // Only terms involving `c` survive.
                    let mut dv = Vec3::zero();
                    if a == c {
                        for (b, qb) in charges.iter().enumerate() {
                            if b == c {
                                continue;
                            }
                            let r = molecule.atoms[b].position - molecule.atoms[a].position;
                            dv -= crate::pbc::ewald::delta_gradient(
                                r,
                                lattice,
                                &neighbors.translations,
                                ewald,
                            ) * *qb;
                        }
                    } else {
                        let r = molecule.atoms[c].position - molecule.atoms[a].position;
                        dv += crate::pbc::ewald::delta_gradient(
                            r,
                            lattice,
                            &neighbors.translations,
                            ewald,
                        ) * charges[c];
                    }
                    let off = basis.atom_offset[a];
                    for (axis, d) in [dv.x, dv.y, dv.z].iter().enumerate() {
                        for k in 0..basis.atom_norb[a] {
                            fmat[axis][(off + k, off + k)] -= d;
                        }
                    }
                }
            }

            // The external field's derivative Fock. The field operator carries `+F·R_a` on atom
            // `a`'s diagonal block, so `∂h^F/∂R_{c,axis} = F_axis` there and nowhere else — one
            // constant per diagonal element of atom `c`'s own block. This is the *only* way a
            // field reaches the Hessian: being linear in the positions, it adds nothing to the
            // fixed-density second derivative, so if this term were missing the Hessian under a
            // field would still look entirely reasonable.
            if let Some(field) = electric_field {
                let off = basis.atom_offset[c];
                for axis in 0..3 {
                    let f = field.get(axis);
                    if f == 0.0 {
                        continue;
                    }
                    for k in 0..basis.atom_norb[c] {
                        fmat[axis][(off + k, off + k)] += f;
                    }
                }
            }

            Ok([
                project_ov(&fmat[0], cv, co),
                project_ov(&fmat[1], cv, co),
                project_ov(&fmat[2], cv, co),
            ])
        })
        .collect();

    let mut gov: Vec<Matrix> = Vec::with_capacity(3 * nat);
    for res in per_atom {
        let [gx, gy, gz] = res?;
        gov.push(gx);
        gov.push(gy);
        gov.push(gz);
    }
    Ok(gov)
}

/// AO-basis first-order density response from the MO occ–virt response coefficients `u`
/// (n_vir × n_occ): `R = Cv (w·U) Coᵀ + Co (w·U)ᵀ Cvᵀ`, built by matrix products
/// (O(nao²·n_occ)). The occupation weight `w` is 2 for RHF (spin-summed) and 1 for a single UHF
/// spin channel.
fn ao_response_density_w(u: &Matrix, cv: &Matrix, co: &Matrix, weight: f64) -> Matrix {
    let mut uw = u.clone();
    for x in uw.as_mut_slice() {
        *x *= weight;
    }
    // Sequential and transpose-free, for the reason given in `project_ov`.
    let a = cv.matmul_seq(&uw); // nao × n_occ
    let mut r = a.matmul_transpose_seq(co); // nao × nao
                                            // Symmetrize in place. The previous form built a full transposed copy and added it, which
                                            // is another `nao²` allocation on a path taken once per perturbation per iteration.
    let n = r.rows;
    for i in 0..n {
        for j in 0..i {
            let s = r[(i, j)] + r[(j, i)];
            r[(i, j)] = s;
            r[(j, i)] = s;
        }
        r[(i, i)] *= 2.0;
    }
    r
}

/// RHF response density (occupation weight 2).
fn ao_response_density(u: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    ao_response_density_w(u, cv, co, 2.0)
}

/// Trace the CPHF residual per iteration when `AM1_CPHF_DEBUG` is set.
fn cphf_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AM1_CPHF_DEBUG").is_some())
}

/// Maximum CPHF fixed-point iterations per nuclear perturbation.
const CPHF_MAX_ITER: usize = 100;
/// Convergence threshold on the CPHF fixed-point residual `‖U_{n+1} − U_n‖₂`.
const CPHF_TOL: f64 = 1.0e-9;
/// DIIS subspace size for the CPHF solve.
const CPHF_DIIS_DEPTH: usize = 8;

/// What a CPHF solve actually did. Carried out of the solver so a non-convergence cannot be
/// folded silently into the Hessian, which is what used to happen: the loop simply fell out
/// after its iteration limit and returned the last iterate.
#[derive(Clone, Copy, Debug)]
pub struct CphfOutcome {
    pub iterations: usize,
    pub residual: f64,
    pub converged: bool,
}

/// Pulay DIIS coefficients for the CPHF fixed-point residuals.
///
/// Plain DIIS does not work here. A single-mode stiff fixed point drives the residuals
/// near-linearly-dependent within a few iterations, the `B` matrix goes singular, and
/// [`crate::scf::solve_bordered_small`] (correctly) refuses it — so the solver silently falls
/// back to unaccelerated iteration and nothing improves. Normalising `B` by the residual
/// magnitudes puts it on a scale where the pivot guard is meaningful, a small Tikhonov ridge
/// keeps it invertible, and renormalising `Σc = 1` undoes the scaling.
/// Smallest `LDLᵀ` pivot of a symmetric positive-(semi)definite matrix; 0 if it is not
/// positive definite. Used as a cheap rank test on the normalised DIIS Gram matrix.
fn min_ldl_pivot(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    let mut d = vec![0.0_f64; n];
    let mut minp = f64::INFINITY;
    for i in 0..n {
        let mut s = a[i][i];
        for k in 0..i {
            s -= l[i][k] * l[i][k] * d[k];
        }
        // Negated `>` rather than `<=`, on purpose: this must also reject NaN, and every
        // comparison against NaN is false, so `s <= 0.0` would let it through and the caller
        // would take a NaN pivot for a valid one.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(s > 0.0) {
            return 0.0;
        }
        d[i] = s;
        minp = minp.min(s);
        l[i][i] = 1.0;
        for j in (i + 1)..n {
            let mut t = a[j][i];
            for k in 0..i {
                t -= l[j][k] * l[i][k] * d[k];
            }
            l[j][i] = t / d[i];
        }
    }
    minp
}

/// Below this `LDLᵀ` pivot the normalised Gram is treated as rank-deficient and the oldest
/// DIIS vector is dropped.
const CPHF_DIIS_RANK_TOL: f64 = 1.0e-9;

fn cphf_diis_coeffs_from_gram(gram: &[Vec<f64>]) -> Option<Vec<f64>> {
    let n_full = gram.len();
    if n_full < 2 {
        return None;
    }
    let norms: Vec<f64> = (0..n_full)
        .map(|i| gram[i][i].max(0.0).sqrt().max(1.0e-300))
        .collect();

    // Shrink the window from the newest end until the normalised Gram has full numerical
    // rank. A stiff single-mode fixed point makes successive residuals collinear within two
    // or three iterations; the resulting Gram is rank one, and a ridged solve of it returns
    // the *uniform* vector -- DIIS silently degenerates into averaging the history, which
    // leaves the plain fixed-point rate untouched. That is exactly what this solver did: most
    // perturbations converged in five iterations while one crawled at ratio 0.87 and hit the
    // iteration cap. Dropping the stale vectors restores the two-point geometric
    // extrapolation that a single mode actually needs.
    for n in (2..=n_full).rev() {
        let off = n_full - n;
        let sub: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| gram[off + i][off + j] / (norms[off + i] * norms[off + j]))
                    .collect()
            })
            .collect();
        if min_ldl_pivot(&sub) < CPHF_DIIS_RANK_TOL {
            continue;
        }

        let dim = n + 1;
        let mut b = Matrix::zeros(dim, dim);
        for i in 0..n {
            for j in 0..n {
                let mut v = sub[i][j];
                if i == j {
                    v += 1.0e-12; // token ridge; the rank test above does the real work
                }
                b[(i, j)] = v;
            }
            b[(i, n)] = -1.0;
            b[(n, i)] = -1.0;
        }
        let mut rhs = vec![0.0; dim];
        rhs[n] = -1.0;
        let Some(raw) = crate::scf::solve_bordered_small(&b, &rhs) else {
            continue;
        };

        // Undo the normalisation and re-impose Σc = 1.
        let mut c: Vec<f64> = (0..n).map(|i| raw[i] / norms[off + i]).collect();
        let sum: f64 = c.iter().sum();
        if !sum.is_finite() || sum.abs() < 1.0e-12 {
            continue;
        }
        for v in &mut c {
            *v /= sum;
        }
        if c.iter().any(|v| !v.is_finite() || v.abs() > 1.0e4) {
            continue;
        }
        // Pad with zeros so the coefficients line up with the full trial history.
        let mut full = vec![0.0; n_full];
        full[off..].copy_from_slice(&c);
        return Some(full);
    }
    None
}

fn cphf_diis_coeffs(errors: &[Matrix]) -> Option<Vec<f64>> {
    let n = errors.len();
    let gram: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| errors[i].frobenius_dot(&errors[j]))
                .collect()
        })
        .collect();
    cphf_diis_coeffs_from_gram(&gram)
}

/// DIIS coefficients for the coupled α/β UCPHF residuals: the two spin blocks are one vector,
/// so their Gram contributions add.
fn ucphf_diis_coeffs(errors_a: &[Matrix], errors_b: &[Matrix]) -> Option<Vec<f64>> {
    let n = errors_a.len();
    let gram: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    errors_a[i].frobenius_dot(&errors_a[j])
                        + errors_b[i].frobenius_dot(&errors_b[j])
                })
                .collect()
        })
        .collect();
    cphf_diis_coeffs_from_gram(&gram)
}

/// Linear combination `Σ c_i M_i` over a DIIS history.
fn diis_combine(coeffs: &[f64], history: &[Matrix]) -> Matrix {
    let mut acc = Matrix::zeros(history[0].rows, history[0].cols);
    for (c, m) in coeffs.iter().zip(history.iter()) {
        for (av, mv) in acc.as_mut_slice().iter_mut().zip(m.as_slice()) {
            *av += c * mv;
        }
    }
    acc
}

/// Apply the orbital Hessian `B(u) = (ε_a − ε_i)∘u + [G(∂P(u))]_ov` to one perturbation.
///
/// One call is one full Fock build, which is what the whole CPHF solve costs — on a 150-atom
/// cluster the Fock builds inside CPHF are 65 % of an entire frequency calculation. So the
/// figure of merit for a solver here is simply how many times it calls this.
// The perturbation and the MO/AO context it has to be contracted against.
/// Everything a CPHF solve needs that does **not** change from one perturbation to the next.
///
/// The three functions below — the operator application, the conjugate-gradient solve and the
/// fixed-point fallback — each took these six arguments positionally and each carried an
/// `#[allow(clippy::too_many_arguments)]` to say so. Bundling them says which arguments are the
/// *problem* (`g_ov`, `denom`) and which are the setting, and makes it impossible to hand the
/// fallback a different Hamiltonian from the solver that gave up.
struct CphfContext<'a> {
    /// Virtual MO coefficients, `nao × n_vir`.
    cv: &'a Matrix,
    /// Occupied MO coefficients, `nao × n_occ`.
    co: &'a Matrix,
    molecule: &'a Molecule,
    params: &'a Am1Parameters,
    basis: &'a crate::basis::Basis,
    core: &'a crate::hamiltonian::CoreHamiltonian,
}

fn apply_orbital_hessian(ctx: &CphfContext<'_>, u: &Matrix, neg_denom: &Matrix) -> Result<Matrix> {
    let r = {
        let _t = crate::timing::Timer::start("cphf:to_ao");
        ao_response_density(u, ctx.cv, ctx.co)
    };
    // The two-electron matrix directly, not `F − H_core`: see `fock::build_g_matrix`. Sequential
    // pair loop because the caller is already parallel over the `3N` perturbations.
    let g_full = {
        let _t = crate::timing::Timer::start("cphf:fock");
        crate::fock::build_g_matrix(
            ctx.molecule,
            ctx.basis,
            ctx.params,
            ctx.core,
            &r,
            crate::fock::PairLoop::Sequential,
        )?
    };
    let mut out = {
        let _t = crate::timing::Timer::start("cphf:to_mo");
        project_ov(&g_full, ctx.cv, ctx.co)
    };
    for (ov, (uv, dv)) in out
        .as_mut_slice()
        .iter_mut()
        .zip(u.as_slice().iter().zip(neg_denom.as_slice()))
    {
        *ov += dv * uv;
    }
    Ok(out)
}

/// Solve the CPHF equations for one perturbation, in the MO occ–virt block.
///
/// # Why conjugate gradient
///
/// The CPHF equations are **linear**: `(ε_a − ε_i)∘U + [G(∂P(U))]_ov = −G_skel`, or `B(U) = b`
/// with `B` the orbital Hessian. At a stable SCF solution `B` is symmetric and positive
/// definite, which is exactly the setting conjugate gradient is for.
///
/// The previous solver iterated `U ← (G_skel + [G(∂P(U))]_ov)/(ε_i − ε_a)` to a fixed point,
/// accelerated by DIIS. That is preconditioned Richardson with `M = diag(ε_a − ε_i)` — the same
/// preconditioner used here — and it needed about 14 applications of `B` per perturbation.
/// Conjugate gradient builds a Krylov space over `B` instead of taking fixed steps along it, and
/// each application of `B` is a full Fock build.
///
/// The convergence test is deliberately the *same quantity* the fixed-point solver used, so the
/// two are directly comparable and the tolerance did not have to be retuned: the fixed point's
/// step `‖U_{n+1} − U_n‖` is identically `‖M⁻¹r‖`, the preconditioned residual, which is already
/// computed each CG iteration.
///
/// If `B` turns out not to be positive definite along the search direction — an unstable SCF
/// solution, which this code cannot rule out — CG cannot proceed, and the solve falls back to
/// the fixed-point iteration rather than returning something wrong.
fn cphf_ov(ctx: &CphfContext<'_>, g_ov: &Matrix, denom: &Matrix) -> Result<(Matrix, CphfOutcome)> {
    let mut neg_denom = denom.clone();
    for v in neg_denom.as_mut_slice() {
        *v = -*v;
    }
    // M⁻¹: divide by (ε_a − ε_i), with the near-degenerate guard the fixed-point solver used.
    let precondition = |x: &Matrix| -> Matrix {
        let mut z = x.clone();
        for (zv, dv) in z.as_mut_slice().iter_mut().zip(neg_denom.as_slice()) {
            *zv = if dv.abs() < 1.0e-10 { 0.0 } else { *zv / *dv };
        }
        z
    };
    let axpy = |y: &mut Matrix, a: f64, x: &Matrix| {
        for (yv, xv) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *yv += a * xv;
        }
    };

    // b = −G_skel, so that the operator is the positive-definite orbital Hessian.
    let mut b = g_ov.clone();
    for v in b.as_mut_slice() {
        *v = -*v;
    }

    // Start from the uncoupled solution, which is what the fixed-point solver also used.
    let mut u = precondition(&b);
    let mut r = b.clone();
    let bu = apply_orbital_hessian(ctx, &u, &neg_denom)?;
    axpy(&mut r, -1.0, &bu);
    let mut z = precondition(&r);
    let mut p = z.clone();
    let mut rz = r.frobenius_dot(&z);
    let mut outcome = CphfOutcome {
        iterations: 1,
        residual: z.frobenius_dot(&z).sqrt(),
        converged: false,
    };
    if outcome.residual < CPHF_TOL {
        outcome.converged = true;
        return Ok((u, outcome));
    }

    for iter in 2..=CPHF_MAX_ITER {
        let bp = apply_orbital_hessian(ctx, &p, &neg_denom)?;
        let pbp = p.frobenius_dot(&bp);
        // Negated `>` rather than `<=`: a NaN curvature is exactly the case that must fall back,
        // and `NaN <= 0.0` is false. Clippy objects to the idiom precisely because the two
        // differ on unordered values, which is the reason for writing it this way.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(pbp > 0.0) {
            // Not positive definite along this direction. Rather than press on with a step that
            // has no variational meaning, hand the problem to the fixed-point solver.
            return cphf_ov_fixed_point(ctx, g_ov, denom);
        }
        let alpha = rz / pbp;
        axpy(&mut u, alpha, &p);
        axpy(&mut r, -alpha, &bp);
        z = precondition(&r);
        let residual = z.frobenius_dot(&z).sqrt();
        if cphf_debug() {
            eprintln!("      cphf cg iter {iter:3}  residual {residual:.6e}");
        }
        outcome = CphfOutcome {
            iterations: iter,
            residual,
            converged: residual < CPHF_TOL,
        };
        if outcome.converged {
            break;
        }
        let rz_new = r.frobenius_dot(&z);
        let beta = rz_new / rz;
        rz = rz_new;
        // p ← z + β p
        for (pv, zv) in p.as_mut_slice().iter_mut().zip(z.as_slice()) {
            *pv = zv + beta * *pv;
        }
    }
    Ok((u, outcome))
}

/// The original preconditioned-Richardson solve with DIIS, kept as the fallback for the case
/// conjugate gradient cannot handle: an orbital Hessian that is not positive definite.
fn cphf_ov_fixed_point(
    ctx: &CphfContext<'_>,
    g_ov: &Matrix,
    denom: &Matrix,
) -> Result<(Matrix, CphfOutcome)> {
    // Uncoupled start: U0 = G / (ε_i − ε_a).
    let elem_div = |num: &Matrix| -> Matrix {
        let mut u = num.clone();
        for (uv, dv) in u.as_mut_slice().iter_mut().zip(denom.as_slice()) {
            *uv = if dv.abs() < 1.0e-10 { 0.0 } else { *uv / *dv };
        }
        u
    };
    let mut u = elem_div(g_ov);
    let mut trials: Vec<Matrix> = Vec::with_capacity(CPHF_DIIS_DEPTH);
    let mut errors: Vec<Matrix> = Vec::with_capacity(CPHF_DIIS_DEPTH);
    let mut outcome = CphfOutcome {
        iterations: 0,
        residual: f64::INFINITY,
        converged: false,
    };

    for iter in 1..=CPHF_MAX_ITER {
        // Three phases, timed separately because which one dominates is not obvious and the
        // answer decides where optimization effort goes. The Fock build looks like the expensive
        // one; the two MO<->AO transforms around it are `nao² × n_occ` matmuls and are not.
        let r = {
            let _t = crate::timing::Timer::start("cphf:to_ao");
            ao_response_density(&u, ctx.cv, ctx.co)
        };
        // The response kernel is the two-electron matrix `G(∂P)` alone, built directly rather
        // than as `F(∂P) − H_core`, and with a sequential pair loop because this whole function
        // already runs under rayon across the `3N` perturbations.
        let g_resp_full = {
            let _t = crate::timing::Timer::start("cphf:fock");
            crate::fock::build_g_matrix(
                ctx.molecule,
                ctx.basis,
                ctx.params,
                ctx.core,
                &r,
                crate::fock::PairLoop::Sequential,
            )?
        };
        let g_resp = {
            let _t = crate::timing::Timer::start("cphf:to_mo");
            project_ov(&g_resp_full, ctx.cv, ctx.co)
        };
        let mut rhs = g_ov.clone();
        for (rv, gv) in rhs.as_mut_slice().iter_mut().zip(g_resp.as_slice()) {
            *rv += *gv;
        }
        let u_new = elem_div(&rhs);

        let mut err = u_new.clone();
        for (ev, ov) in err.as_mut_slice().iter_mut().zip(u.as_slice()) {
            *ev -= *ov;
        }
        let residual = err.frobenius_dot(&err).sqrt();
        if cphf_debug() {
            eprintln!("      cphf iter {iter:3}  residual {residual:.6e}");
        }
        outcome = CphfOutcome {
            iterations: iter,
            residual,
            converged: residual < CPHF_TOL,
        };
        if outcome.converged {
            u = u_new;
            break;
        }

        if trials.len() == CPHF_DIIS_DEPTH {
            trials.remove(0);
            errors.remove(0);
        }
        trials.push(u_new.clone());
        errors.push(err);

        let coeffs = cphf_diis_coeffs(&errors);
        if cphf_debug() {
            match &coeffs {
                Some(c) => eprintln!("        diis n={} c={:?}", c.len(), c),
                None => eprintln!("        diis rejected (n={})", errors.len()),
            }
        }
        u = match coeffs {
            Some(c) => diis_combine(&c, &trials),
            None => u_new,
        };
    }
    Ok((u, outcome))
}

/// Fail if any perturbation's CPHF solve hit the iteration limit.
fn check_cphf(outcomes: &[CphfOutcome]) -> Result<()> {
    let failed = outcomes.iter().filter(|o| !o.converged).count();
    if failed == 0 {
        return Ok(());
    }
    let residual = outcomes
        .iter()
        .filter(|o| !o.converged)
        .map(|o| o.residual)
        .fold(0.0_f64, f64::max);
    Err(crate::error::Am1Error::CphfNotConverged {
        perturbations: failed,
        iterations: CPHF_MAX_ITER,
        residual,
    })
}

/// **Analytic open-shell (UHF) Cartesian Hessian** (eV/Bohr²). Same structure as the RHF path
/// but spin-resolved: the skeleton second derivative uses same-spin exchange
/// `−[Pα_μλ Pα_νσ + Pβ_μλ Pβ_νσ]`, and the response solves the **coupled** α/β CPHF equations
/// (the α and β responses are coupled through the total-density Coulomb term). Everything stays
/// in the per-spin MO occ–virt blocks — memory-lean, rayon-parallel. No finite differences.
fn analytic_hessian_uhf(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    scf: crate::scf::Am1Result,
) -> Result<HessianResponse> {
    use crate::dual2::Dual2;
    use crate::fock::build_fock_spin;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = crate::basis::Basis::build(molecule, params)?;
    let nao = basis.nao;

    // This path is structurally molecular: the skeleton loop below walks `(u, v)` over
    // intramolecular pairs rather than the image pair list, so a periodic system would get a
    // molecular answer with no sign that anything was dropped. It used to build a molecular
    // `H_core` here unconditionally, which hid the same mismatch inside the CPHF kernel as well.
    // Refusing is the honest form of the same limitation.
    if molecule.cell.map(|c| c.n_periodic() > 0).unwrap_or(false) {
        return Err(crate::error::Am1Error::InvalidInput(
            "the analytic UHF Hessian is molecular only: its skeleton second derivative walks \
             intramolecular pairs, not the periodic image pair list. Use a closed-shell \
             reference under a cell, or `numerical_hessian`."
                .into(),
        ));
    }
    if options.multipole_cutoff.is_some() {
        return Err(crate::error::Am1Error::InvalidInput(
            "the analytic UHF Hessian does not implement the far-field monopole screening: its \
             skeleton loop visits every pair while a screened `H_core` would not, so the two \
             halves would describe different Hamiltonians. Clear `multipole_cutoff` for an \
             open-shell Hessian."
                .into(),
        ));
    }
    // The same core the SCF built, so the CPHF kernel and the density it acts on agree. Only the
    // field survives from `options` here; the periodic and screening knobs are refused above.
    let core = crate::hamiltonian::build_core_with_neighbors(
        molecule,
        &basis,
        params,
        &crate::neighbors::NeighborList::molecular(molecule),
        crate::hamiltonian::CoreBuildOptions {
            electric_field: options.electric_field,
            ..crate::hamiltonian::CoreBuildOptions::molecular()
        },
    )?;

    // Spin densities Pα = (P_tot + S)/2, Pβ = (P_tot − S)/2.
    let pt = scf.density.clone();
    let spin = scf.spin_density.as_ref().ok_or_else(|| {
        crate::error::Am1Error::InvalidInput("UHF Hessian requires a spin density".into())
    })?;
    let mut pa = pt.clone();
    let mut pb = pt.clone();
    {
        let (pas, pbs) = (pa.as_mut_slice(), pb.as_mut_slice());
        let (pts, ss) = (pt.as_slice(), spin.as_slice());
        for i in 0..pts.len() {
            pas[i] = 0.5 * (pts[i] + ss[i]);
            pbs[i] = 0.5 * (pts[i] - ss[i]);
        }
    }
    // Recover both spin orbital sets by diagonalizing the converged spin Fock matrices.
    let fa = build_fock_spin(molecule, &basis, params, &core, &pt, &pa)?;
    let fb = build_fock_spin(molecule, &basis, params, &core, &pt, &pb)?;
    let (eps_a, ca) = symmetric_eigen(&fa)?;
    let (eps_b, cb) = symmetric_eigen(&fb)?;
    let n_alpha = scf.n_occ;
    let n_beta = scf.n_occ - (options.multiplicity - 1);

    // 1) Skeleton (fixed-density) second derivative — spin-resolved exchange.
    let mut hess = Matrix::zeros(ndof, ndof);
    let beta = |elem: &crate::params::Am1Element, orb: u8| {
        if orb == 0 {
            elem.beta_s
        } else {
            elem.beta_p
        }
    };
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|u| ((u + 1)..nat).map(move |v| (u, v)))
        .collect();
    let blocks: Vec<Result<HessianBlock>> = {
        let _t = crate::timing::Timer::start("hess:skeleton");
        use rayon::prelude::*;
        pairs
            .par_iter()
            .map(|&(u, v)| -> Result<HessianBlock> {
                let eu = params.element(molecule.atoms[u].z)?;
                let ev = params.element(molecule.atoms[v].z)?;
                let (a, b) = if eu.has_p() || !ev.has_p() {
                    (u, v)
                } else {
                    (v, u)
                };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let (posa, posb) = (molecule.atoms[a].position, molecule.atoms[b].position);
                let dvec = [
                    Dual2::var(posb.x - posa.x, 0),
                    Dual2::var(posb.y - posa.y, 1),
                    Dual2::var(posb.z - posa.z, 2),
                ];
                let te = crate::integrals::pair_two_electron_g::<Dual2>(ea, eb, dvec);
                let s = crate::overlap::diatom_overlap_dual2(ea, posa, eb, posb)?;
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                let mut epair = Dual2::constant(0.0);
                for i in 0..na {
                    let bi = beta(ea, basis.aos[oa + i].orb);
                    for j in 0..nb {
                        let bj = beta(eb, basis.aos[ob + j].orb);
                        let coef = pt[(oa + i, ob + j)] * (bi + bj);
                        epair = epair + s[i][j] * coef;
                    }
                }
                for i in 0..na {
                    for j in 0..na {
                        epair = epair + te.e1b[i][j] * pt[(oa + i, oa + j)];
                    }
                }
                for k in 0..nb {
                    for l in 0..nb {
                        epair = epair + te.e2a[k][l] * pt[(ob + k, ob + l)];
                    }
                }
                for mu in 0..na {
                    for nu in 0..na {
                        for la in 0..nb {
                            for si in 0..nb {
                                let coul = pt[(oa + mu, oa + nu)] * pt[(ob + la, ob + si)];
                                // Same-spin exchange: −(Pα_μλ Pα_νσ + Pβ_μλ Pβ_νσ).
                                let exch = -(pa[(oa + mu, ob + la)] * pa[(oa + nu, ob + si)]
                                    + pb[(oa + mu, ob + la)] * pb[(oa + nu, ob + si)]);
                                epair = epair + te.two_e(mu, nu, la, si) * (coul + exch);
                            }
                        }
                    }
                }
                let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
                epair = epair
                    + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                        ea,
                        eb,
                        molecule.atoms[a].z,
                        molecule.atoms[b].z,
                        r,
                    );
                Ok((a, b, epair.h))
            })
            .collect()
    };
    for blk in blocks {
        let (a, b, hb) = blk?;
        for i in 0..3 {
            for j in 0..3 {
                let val = hb[i][j];
                hess[(3 * a + i, 3 * a + j)] += val;
                hess[(3 * b + i, 3 * b + j)] += val;
                hess[(3 * a + i, 3 * b + j)] -= val;
                hess[(3 * b + i, 3 * a + j)] -= val;
            }
        }
    }

    // 2+3) Coupled UCPHF relaxation term: H_relax[a][b] = 2 Σ_σ Gσ^a · Uσ^b.
    let nva = nao - n_alpha;
    let nvb = nao - n_beta;
    let have_a = nva > 0 && n_alpha > 0;
    let have_b = nvb > 0 && n_beta > 0;
    let cva_out = submatrix_cols(&ca, n_alpha, nva);
    let coa_out = submatrix_cols(&ca, 0, n_alpha);
    let cvb_out = submatrix_cols(&cb, n_beta, nvb);
    let cob_out = submatrix_cols(&cb, 0, n_beta);
    // Each UHF spin channel carries one electron per orbital, not two.
    let mut channel_a = ResponseChannel {
        u_ov: vec![Matrix::zeros(nva, n_alpha); ndof],
        g_ov: vec![Matrix::zeros(nva, n_alpha); ndof],
        occupied: coa_out,
        virtuals: cva_out,
        occupation: 1.0,
    };
    let mut channel_b = ResponseChannel {
        u_ov: vec![Matrix::zeros(nvb, n_beta); ndof],
        g_ov: vec![Matrix::zeros(nvb, n_beta); ndof],
        occupied: cob_out,
        virtuals: cvb_out,
        occupation: 1.0,
    };
    let mut outcomes_out: Vec<CphfOutcome> = Vec::new();
    if have_a || have_b {
        let cva = submatrix_cols(&ca, n_alpha, nva);
        let coa = submatrix_cols(&ca, 0, n_alpha);
        let cvb = submatrix_cols(&cb, n_beta, nvb);
        let cob = submatrix_cols(&cb, 0, n_beta);
        let denom_a = ov_denominators(&eps_a, n_alpha, nva);
        let denom_b = ov_denominators(&eps_b, n_beta, nvb);

        // The molecular pair list, matching the `core` built above. The periodic and
        // far-field-screened cases were refused at the top of this function, so "molecular" here is
        // a guarantee rather than an assumption — and putting the list in the shared context is
        // what makes that visible instead of implicit in a shorter argument list.
        let neighbors = crate::neighbors::NeighborList::molecular(molecule);
        let (gova, govb) = skeleton_fock_ov_spin(
            &SkeletonContext {
                molecule,
                params,
                basis: &basis,
                neighbors: &neighbors,
                exchange_cutoff: options.exchange_cutoff,
                use_ewald: options.ewald,
                electric_field: options.electric_field,
            },
            SpinDensities {
                total: &pt,
                alpha: &pa,
                beta: &pb,
            },
            OvBlocks { cv: &cva, co: &coa },
            OvBlocks { cv: &cvb, co: &cob },
        )?;

        let solved: Vec<(Matrix, Matrix, CphfOutcome)> = {
            use rayon::prelude::*;
            (0..ndof)
                .into_par_iter()
                .map(|t| {
                    ucphf_ov(
                        &UcphfContext {
                            alpha: OvBlocks { cv: &cva, co: &coa },
                            beta: OvBlocks { cv: &cvb, co: &cob },
                            molecule,
                            params,
                            basis: &basis,
                            core: &core,
                        },
                        &gova[t],
                        &govb[t],
                        &denom_a,
                        &denom_b,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };
        let outcomes: Vec<CphfOutcome> = solved.iter().map(|s| s.2).collect();
        check_cphf(&outcomes)?;
        let uovs: Vec<(Matrix, Matrix)> = solved.into_iter().map(|(a, b, _)| (a, b)).collect();

        use rayon::prelude::*;
        let rows: Vec<Vec<f64>> = (0..ndof)
            .into_par_iter()
            .map(|a| {
                (0..ndof)
                    .map(|b| {
                        2.0 * (gova[a].frobenius_dot(&uovs[b].0)
                            + govb[a].frobenius_dot(&uovs[b].1))
                    })
                    .collect()
            })
            .collect();
        for (a, row) in rows.into_iter().enumerate() {
            for (b, v) in row.into_iter().enumerate() {
                hess[(a, b)] += v;
            }
        }
        let (ua, ub): (Vec<Matrix>, Vec<Matrix>) = uovs.into_iter().unzip();
        channel_a.u_ov = ua;
        channel_a.g_ov = gova;
        channel_b.u_ov = ub;
        channel_b.g_ov = govb;
        outcomes_out = outcomes;
    }

    // Symmetrize.
    let mut sym = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            sym[(i, j)] = 0.5 * (hess[(i, j)] + hess[(j, i)]);
        }
    }
    Ok(HessianResponse {
        hessian: sym,
        scf,
        alpha: channel_a,
        beta: Some(channel_b),
        cphf: outcomes_out,
    })
}

/// Spin-resolved skeleton derivative Fock ov-blocks: returns `(Gα, Gβ)` per DOF. The resonance,
/// electron–core, and Coulomb `J(P_tot)` parts are spin-independent (shared); the exchange
/// differs — `Kα(Pα)` into the α Fock, `Kβ(Pβ)` into the β Fock. Built one atom at a time and
/// projected to each spin's occ–virt block (peak memory `O(nao²)`).
/// The three densities an unrestricted skeleton contracts against: the total, and one per spin.
///
/// `Pα` and `Pβ` are reconstructed from the total and the spin density at the call site, and the
/// three then travel together everywhere. Loose, they are three same-typed arguments in a row.
#[derive(Clone, Copy)]
struct SpinDensities<'a> {
    total: &'a Matrix,
    alpha: &'a Matrix,
    beta: &'a Matrix,
}

fn skeleton_fock_ov_spin(
    ctx: &SkeletonContext<'_>,
    densities: SpinDensities<'_>,
    alpha_blocks: OvBlocks<'_>,
    beta_blocks: OvBlocks<'_>,
) -> Result<(Vec<Matrix>, Vec<Matrix>)> {
    let (molecule, params, basis, electric_field) =
        (ctx.molecule, ctx.params, ctx.basis, ctx.electric_field);
    let (pt, pa, pb) = (densities.total, densities.alpha, densities.beta);
    let (cva, coa) = (alpha_blocks.cv, alpha_blocks.co);
    let (cvb, cob) = (beta_blocks.cv, beta_blocks.co);
    use crate::integrals::pair_two_electron_dual;
    use crate::overlap::diatom_overlap_dual;
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let nao = basis.nao;
    let beta = |e: &crate::params::Am1Element, orb: u8| if orb == 0 { e.beta_s } else { e.beta_p };

    let per_atom: Vec<Result<[(Matrix, Matrix); 3]>> = (0..nat)
        .into_par_iter()
        .map(|c| -> Result<[(Matrix, Matrix); 3]> {
            let mut fa = [
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
            ];
            let mut fb = [
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
            ];
            for x in 0..nat {
                if x == c {
                    continue;
                }
                let (u, v) = (c.min(x), c.max(x));
                let eu = params.element(molecule.atoms[u].z)?;
                let ev = params.element(molecule.atoms[v].z)?;
                let (a, b) = if eu.has_p() || !ev.has_p() {
                    (u, v)
                } else {
                    (v, u)
                };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let (posa, posb) = (molecule.atoms[a].position, molecule.atoms[b].position);
                let sign = if c == b { 1.0 } else { -1.0 };
                let te = pair_two_electron_dual(ea, eb, posb - posa);
                let s = diatom_overlap_dual(ea, posa, eb, posb)?;
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                for axis in 0..3 {
                    // Borrow both spin Fock matrices for this atom (distinct arrays fa/fb).
                    let fma = &mut fa[axis];
                    let fmb = &mut fb[axis];
                    // Shared (spin-independent): resonance, e-core, Coulomb J(P_tot) → into BOTH,
                    // per-pair (do NOT copy the running-accumulated matrix, which would
                    // re-add earlier neighbours' contributions).
                    for i in 0..na {
                        let bi = beta(ea, basis.aos[oa + i].orb);
                        for j in 0..nb {
                            let bj = beta(eb, basis.aos[ob + j].orb);
                            let val = sign * 0.5 * (bi + bj) * s[i][j].d[axis];
                            fma[(oa + i, ob + j)] += val;
                            fma[(ob + j, oa + i)] += val;
                            fmb[(oa + i, ob + j)] += val;
                            fmb[(ob + j, oa + i)] += val;
                        }
                    }
                    for i in 0..na {
                        for j in 0..na {
                            let val = sign * te.e1b[i][j].d[axis];
                            fma[(oa + i, oa + j)] += val;
                            fmb[(oa + i, oa + j)] += val;
                        }
                    }
                    for k in 0..nb {
                        for l in 0..nb {
                            let val = sign * te.e2a[k][l].d[axis];
                            fma[(ob + k, ob + l)] += val;
                            fmb[(ob + k, ob + l)] += val;
                        }
                    }
                    for mu in 0..na {
                        for nu in 0..na {
                            let mut acc = 0.0;
                            for la in 0..nb {
                                for si in 0..nb {
                                    acc +=
                                        pt[(ob + la, ob + si)] * te.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            let val = sign * acc;
                            fma[(oa + mu, oa + nu)] += val;
                            fmb[(oa + mu, oa + nu)] += val;
                        }
                    }
                    for la in 0..nb {
                        for si in 0..nb {
                            let mut acc = 0.0;
                            for mu in 0..na {
                                for nu in 0..na {
                                    acc +=
                                        pt[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            let val = sign * acc;
                            fma[(ob + la, ob + si)] += val;
                            fmb[(ob + la, ob + si)] += val;
                        }
                    }
                    // Same-spin exchange Kσ (coefficient −1): Kα(Pα) → fa, Kβ(Pβ) → fb.
                    for mu in 0..na {
                        for la in 0..nb {
                            let mut acca = 0.0;
                            let mut accb = 0.0;
                            for nu in 0..na {
                                for si in 0..nb {
                                    let dw = te.two_e(mu, nu, la, si).d[axis];
                                    acca += pa[(oa + nu, ob + si)] * dw;
                                    accb += pb[(oa + nu, ob + si)] * dw;
                                }
                            }
                            let va = sign * (-acca);
                            let vb = sign * (-accb);
                            fma[(oa + mu, ob + la)] += va;
                            fma[(ob + la, oa + mu)] += va;
                            fmb[(oa + mu, ob + la)] += vb;
                            fmb[(ob + la, oa + mu)] += vb;
                        }
                    }
                }
            }
            // The external field's derivative Fock, `∂h^F/∂R_{c,axis} = F_axis` on atom `c`'s own
            // diagonal block. It is a one-electron term, so it is identical for both spins.
            if let Some(field) = electric_field {
                let off = basis.atom_offset[c];
                for axis in 0..3 {
                    let f = field.get(axis);
                    if f == 0.0 {
                        continue;
                    }
                    for k in 0..basis.atom_norb[c] {
                        fa[axis][(off + k, off + k)] += f;
                        fb[axis][(off + k, off + k)] += f;
                    }
                }
            }

            Ok([
                (project_ov(&fa[0], cva, coa), project_ov(&fb[0], cvb, cob)),
                (project_ov(&fa[1], cva, coa), project_ov(&fb[1], cvb, cob)),
                (project_ov(&fa[2], cva, coa), project_ov(&fb[2], cvb, cob)),
            ])
        })
        .collect();

    let mut gova: Vec<Matrix> = Vec::with_capacity(3 * nat);
    let mut govb: Vec<Matrix> = Vec::with_capacity(3 * nat);
    for res in per_atom {
        let arr = res?;
        for (ga, gb) in arr {
            gova.push(ga);
            govb.push(gb);
        }
    }
    Ok((gova, govb))
}

/// Coupled α/β CPHF solve for one perturbation (MO occ–virt blocks). Iterate
/// `Uσ = (Gσ_skel + [J(ΔP_tot) − Kσ(ΔPσ)]_ov) / (εσ_i − εσ_a)` to self-consistency; the α and β
/// channels couple through the total response density `ΔP_tot = ΔPα + ΔPβ` in the Coulomb term.
/// The unrestricted counterpart of [`CphfContext`]: what does not change from one perturbation to
/// the next, for both spin channels at once.
///
/// The two channels couple through the total response density in the Coulomb term, so they have to
/// be solved together and the solver has to hold both channels' orbitals. Bundling them is what
/// stops `cva`/`coa`/`cvb`/`cob` from being four interchangeable-looking arguments in a row.
struct UcphfContext<'a> {
    alpha: OvBlocks<'a>,
    beta: OvBlocks<'a>,
    molecule: &'a Molecule,
    params: &'a Am1Parameters,
    basis: &'a crate::basis::Basis,
    core: &'a crate::hamiltonian::CoreHamiltonian,
}

fn ucphf_ov(
    ctx: &UcphfContext<'_>,
    ga: &Matrix,
    gb: &Matrix,
    denom_a: &Matrix,
    denom_b: &Matrix,
) -> Result<(Matrix, Matrix, CphfOutcome)> {
    let (cva, coa) = (ctx.alpha.cv, ctx.alpha.co);
    let (cvb, cob) = (ctx.beta.cv, ctx.beta.co);
    let (molecule, params, basis, core) = (ctx.molecule, ctx.params, ctx.basis, ctx.core);
    let div = |num: &Matrix, denom: &Matrix| -> Matrix {
        let mut u = num.clone();
        for (uv, dv) in u.as_mut_slice().iter_mut().zip(denom.as_slice()) {
            *uv = if dv.abs() < 1.0e-10 { 0.0 } else { *uv / *dv };
        }
        u
    };
    let mut ua = div(ga, denom_a);
    let mut ub = div(gb, denom_b);
    let mut trials_a: Vec<Matrix> = Vec::with_capacity(CPHF_DIIS_DEPTH);
    let mut trials_b: Vec<Matrix> = Vec::with_capacity(CPHF_DIIS_DEPTH);
    let mut errors_a: Vec<Matrix> = Vec::with_capacity(CPHF_DIIS_DEPTH);
    let mut errors_b: Vec<Matrix> = Vec::with_capacity(CPHF_DIIS_DEPTH);
    let mut outcome = CphfOutcome {
        iterations: 0,
        residual: f64::INFINITY,
        converged: false,
    };
    for iter in 1..=CPHF_MAX_ITER {
        let dpa = ao_response_density_w(&ua, cva, coa, 1.0);
        let dpb = ao_response_density_w(&ub, cvb, cob, 1.0);
        let mut dpt = dpa.clone();
        for (t, x) in dpt.as_mut_slice().iter_mut().zip(dpb.as_slice()) {
            *t += *x;
        }
        // Response two-electron Focks Gσ(ΔP) = build_fock_spin(ΔP_tot, ΔPσ) − H_core.
        let mut fa_r = crate::fock::build_fock_spin(molecule, basis, params, core, &dpt, &dpa)?;
        let mut fb_r = crate::fock::build_fock_spin(molecule, basis, params, core, &dpt, &dpb)?;
        for (xv, hv) in fa_r.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
            *xv -= *hv;
        }
        for (xv, hv) in fb_r.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
            *xv -= *hv;
        }
        let ga_resp = project_ov(&fa_r, cva, coa);
        let gb_resp = project_ov(&fb_r, cvb, cob);
        let mut rhs_a = ga.clone();
        for (rv, gv) in rhs_a.as_mut_slice().iter_mut().zip(ga_resp.as_slice()) {
            *rv += *gv;
        }
        let mut rhs_b = gb.clone();
        for (rv, gv) in rhs_b.as_mut_slice().iter_mut().zip(gb_resp.as_slice()) {
            *rv += *gv;
        }
        let ua_new = div(&rhs_a, denom_a);
        let ub_new = div(&rhs_b, denom_b);

        let mut err_a = ua_new.clone();
        for (ev, ov) in err_a.as_mut_slice().iter_mut().zip(ua.as_slice()) {
            *ev -= *ov;
        }
        let mut err_b = ub_new.clone();
        for (ev, ov) in err_b.as_mut_slice().iter_mut().zip(ub.as_slice()) {
            *ev -= *ov;
        }
        let residual = (err_a.frobenius_dot(&err_a) + err_b.frobenius_dot(&err_b)).sqrt();
        outcome = CphfOutcome {
            iterations: iter,
            residual,
            converged: residual < CPHF_TOL,
        };
        if outcome.converged {
            ua = ua_new;
            ub = ub_new;
            break;
        }

        if trials_a.len() == CPHF_DIIS_DEPTH {
            trials_a.remove(0);
            trials_b.remove(0);
            errors_a.remove(0);
            errors_b.remove(0);
        }
        trials_a.push(ua_new.clone());
        trials_b.push(ub_new.clone());
        errors_a.push(err_a);
        errors_b.push(err_b);

        match ucphf_diis_coeffs(&errors_a, &errors_b) {
            Some(c) => {
                ua = diis_combine(&c, &trials_a);
                ub = diis_combine(&c, &trials_b);
            }
            None => {
                ua = ua_new;
                ub = ub_new;
            }
        }
    }
    Ok((ua, ub, outcome))
}

#[inline]
fn displace(p: &mut Vec3, k: usize, d: f64) {
    match k {
        0 => p.x += d,
        1 => p.y += d,
        _ => p.z += d,
    }
}
#[inline]
fn component(v: &Vec3, k: usize) -> f64 {
    match k {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{optimize, OptOptions};

    #[test]
    fn analytic_hessian_matches_numerical() {
        // The CPHF analytic Hessian must match the finite-difference Hessian (FD of the
        // full-SCF gradient) — the independent ground truth.
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.97 0.02 0.0\nH -0.25 0.94 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options::default();
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let ndof = ha.rows;
        let mut max_delta = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                max_delta = max_delta.max((ha[(i, j)] - hn[(i, j)]).abs());
            }
        }
        eprintln!("analytic-vs-numerical Hessian max delta = {max_delta:.2e} eV/Bohr^2");
        assert!(max_delta < 1.0e-3, "Hessian mismatch {max_delta:.3e}");
    }

    #[test]
    fn analytic_hessian_heavy_element() {
        // HBr (Br is n = 4): the analytic Hessian now runs the second-order AD through the
        // numerical Slater overlap quadrature (no numerical-Hessian fallback for heavy atoms),
        // and must match the finite-difference Hessian.
        let mol = Molecule::from_xyz_str("2\nHBr\nH 0.0 0.0 0.0\nBr 0.0 0.0 1.48\n", 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options::default();
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let ndof = ha.rows;
        let mut max_delta = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                max_delta = max_delta.max((ha[(i, j)] - hn[(i, j)]).abs());
            }
        }
        eprintln!("heavy-element analytic-vs-numerical Hessian max delta = {max_delta:.2e}");
        assert!(max_delta < 2.0e-3, "heavy Hessian mismatch {max_delta:.3e}");
    }

    #[test]
    fn analytic_hessian_uhf_radical() {
        // Methyl radical (doublet, UHF): the coupled α/β CPHF (UCPHF) analytic Hessian must
        // match the finite-difference Hessian (FD of the analytic UHF gradient).
        let mol = Molecule::from_xyz_str(
            "4\nmethyl\nC 0.0 0.0 0.05\nH 1.09 0.0 0.0\nH -0.545 0.944 0.0\nH -0.545 -0.944 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options {
            multiplicity: 2,
            ..Am1Options::default()
        };
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let ndof = ha.rows;
        let mut max_delta = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                max_delta = max_delta.max((ha[(i, j)] - hn[(i, j)]).abs());
            }
        }
        eprintln!("UHF analytic-vs-numerical Hessian max delta = {max_delta:.2e}");
        assert!(max_delta < 2.0e-3, "UHF Hessian mismatch {max_delta:.3e}");
    }

    #[test]
    fn water_vibrations() {
        // Optimize water, then compute harmonic frequencies. Expect 3 real modes
        // (bend ~1600–1800, two stretches ~3700–3900 cm⁻¹ for AM1) plus ~6 near-zero.
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options::default();
        let relaxed = optimize(&mol, &params, &opts, &OptOptions::default()).unwrap();
        let vib = vibrational_analysis(&relaxed.molecule, &params, &opts, 1.0e-3).unwrap();
        let freqs = &vib.frequencies_cm;
        eprintln!(
            "H2O frequencies (cm^-1): {:?}",
            freqs.iter().map(|f| f.round()).collect::<Vec<_>>()
        );
        // The three highest are the real vibrational modes.
        let n = freqs.len();
        let high = &freqs[n - 3..];
        assert!(high[0] > 1200.0 && high[0] < 2200.0, "bend {}", high[0]);
        assert!(high[1] > 3000.0 && high[2] > 3000.0, "stretches {high:?}");
        // The six lowest (trans/rot) should be small in magnitude.
        let six_low_max = freqs[..6].iter().map(|f| f.abs()).fold(0.0_f64, f64::max);
        assert!(
            six_low_max < 300.0,
            "trans/rot not near zero: {six_low_max}"
        );
    }
}
