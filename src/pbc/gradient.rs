// SPDX-License-Identifier: GPL-3.0-or-later

//! Analytic nuclear gradient and stress tensor for a periodic cell.
//!
//! Because NDDO treats the AO basis as orthonormal, the energy is stationary with respect to
//! the density and there is no Pulay term: the gradient is the derivative of the energy
//! expression at the converged density. That carries over to a periodic cell unchanged — the
//! only difference is which density block each term contracts against, which is exactly the
//! block structure the k-point SCF already builds:
//!
//! * resonance and exchange pair `μ` in cell 0 with `λ` in cell `T`, so they use `P(0,T)`;
//! * the core attraction and the Coulomb terms are on-site, so they use `P(0,0)`.
//!
//! The exchange taper is differentiated too. The exchange energy is `taper(r)·C(δ)`, so its
//! force has a `taper'(r)` piece; leaving it out gives a force that is not the gradient of the
//! energy actually being reported, and the first thing that notices is an MD run failing to
//! conserve.
//!
//! # Stress
//!
//! Under a homogeneous strain `F = I + ε` applied to the lattice vectors and the atomic
//! positions alike, every pair separation transforms as `δ → (I + ε) δ`, so
//!
//! ```text
//! σ_αβ = (1/Ω) Σ_pairs (∂E/∂δ)_α δ_β
//! ```
//!
//! with `Ω` the cell measure — volume in 3D, area in 2D, length in 1D. Every term in this
//! model is a function of pair separations alone, which is what makes the pair virial the
//! whole stress here; there is no reciprocal-space piece to add because there is no Ewald in
//! this path yet.

use crate::basis::Basis;
use crate::dual::{Dual, Scalar};
use crate::error::{Am1Error, Result};
use crate::hamiltonian::exchange_taper_scalar;
use crate::integrals::pair_two_electron_dual;
use crate::lattice::ImageOffset;
use crate::linalg::Matrix;
use crate::math::{Mat3, Vec3};
use crate::neighbors::NeighborList;
use crate::overlap::diatom_overlap_dual;
use crate::params::Am1Parameters;
use crate::pbc::scf::{PbcOptions, PbcResult, RealSpaceBlocks};
use crate::system::Molecule;

/// Forces and stress of a periodic calculation.
#[derive(Clone, Debug)]
pub struct PbcGradient {
    /// `dE/dR_A` per atom, eV/Bohr.
    pub gradient: Vec<Vec3>,
    /// `−dE/dR_A`, eV/Bohr.
    pub forces: Vec<Vec3>,
    /// Stress tensor, eV per Bohr^d where `d` is the number of periodic directions.
    pub stress: Mat3,
    pub max_gradient: f64,
}

impl PbcGradient {
    /// Voigt ordering `xx, yy, zz, yz, xz, xy`.
    pub fn stress_voigt(&self) -> [f64; 6] {
        let s = &self.stress;
        [
            s.col[0].x,
            s.col[1].y,
            s.col[2].z,
            0.5 * (s.col[1].z + s.col[2].y),
            0.5 * (s.col[0].z + s.col[2].x),
            0.5 * (s.col[0].y + s.col[1].x),
        ]
    }

    /// `−Tr(σ)/d` over the periodic directions.
    pub fn pressure(&self, n_periodic: usize) -> f64 {
        if n_periodic == 0 {
            return 0.0;
        }
        let s = &self.stress;
        -(s.col[0].x + s.col[1].y + s.col[2].z) / n_periodic as f64
    }
}

/// Analytic gradient and stress at a converged periodic density.
pub fn pbc_gradient(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    scf: &PbcResult,
) -> Result<PbcGradient> {
    let cell = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a periodic gradient needs a cell".into()))?;
    let basis = Basis::build(molecule, params)?;
    let neighbors = NeighborList::build(molecule, options.realspace_cutoff);
    let nat = molecule.atoms.len();

    let p_total = &scf.density;
    let p0 = p_total
        .get(ImageOffset::origin())
        .ok_or_else(|| Am1Error::InvalidInput("density is missing the origin block".into()))?;

    // Same-spin density blocks the exchange contracts against. Restricted: half the total.
    // Unrestricted: alpha and beta separately, reconstructed from total and spin densities.
    let spin_blocks: Vec<(RealSpaceBlocks, f64)> = match &scf.spin_density {
        None => vec![(p_total.clone(), 0.5)],
        Some(sd) => {
            let mut alpha = p_total.clone();
            let mut beta = p_total.clone();
            for ((a, b), s) in alpha
                .blocks
                .iter_mut()
                .zip(beta.blocks.iter_mut())
                .zip(&sd.blocks)
            {
                for ((av, bv), sv) in a
                    .as_mut_slice()
                    .iter_mut()
                    .zip(b.as_mut_slice())
                    .zip(s.as_slice())
                {
                    let tot = *av;
                    *av = 0.5 * (tot + *sv);
                    *bv = 0.5 * (tot - *sv);
                }
            }
            vec![(alpha, 1.0), (beta, 1.0)]
        }
    };

    let mut gradient = vec![Vec3::zero(); nat];
    let mut virial = [[0.0_f64; 3]; 3];

    for p in &neighbors.pairs {
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
        let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
        let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);
        let pos_a = molecule.atoms[a].position;

        let te = pair_two_electron_dual(ea, eb, delta);
        let s = diatom_overlap_dual(ea, pos_a, eb, pos_a + delta)?;
        let pt = p_total.get(offset);

        let mut f = [0.0_f64; 3];

        // Resonance, against P(0,T).
        if let Some(pt) = pt {
            for i in 0..na {
                let bi = beta_of(ea, basis.aos[oa + i].orb);
                for j in 0..nb {
                    let bj = beta_of(eb, basis.aos[ob + j].orb);
                    // The pair list holds one representative, so this stands for both
                    // orientations; the factor 2 is the mirror's equal contribution.
                    let coef = 2.0 * pt[(oa + i, ob + j)] * (bi + bj) * 0.5;
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * s[i][j].d[ax];
                    }
                }
            }
        }

        // Electron-core attraction, on-site blocks.
        for i in 0..na {
            for j in 0..na {
                let coef = p0[(oa + i, oa + j)];
                for (ax, fx) in f.iter_mut().enumerate() {
                    *fx += coef * te.e1b[i][j].d[ax];
                }
            }
        }
        for k in 0..nb {
            for l in 0..nb {
                let coef = p0[(ob + k, ob + l)];
                for (ax, fx) in f.iter_mut().enumerate() {
                    *fx += coef * te.e2a[k][l].d[ax];
                }
            }
        }

        // Coulomb, on-site x on-site.
        for mu in 0..na {
            for nu in 0..na {
                let pa = p0[(oa + mu, oa + nu)];
                if pa == 0.0 {
                    continue;
                }
                for la in 0..nb {
                    for si in 0..nb {
                        let coef = pa * p0[(ob + la, ob + si)];
                        if coef == 0.0 {
                            continue;
                        }
                        let dw = te.two_e(mu, nu, la, si).d;
                        for (ax, fx) in f.iter_mut().enumerate() {
                            *fx += coef * dw[ax];
                        }
                    }
                }
            }
        }

        // Exchange, against the same-spin P(0,T), with the taper differentiated.
        if let Some(rc) = options.exchange_cutoff {
            let r_dual = Dual {
                v: p.r,
                d: [delta.x / p.r, delta.y / p.r, delta.z / p.r],
            };
            let taper = exchange_taper_scalar::<Dual>(r_dual, rc);
            if taper.v != 0.0 || taper.d.iter().any(|v| *v != 0.0) {
                for (blocks, scale) in &spin_blocks {
                    let Some(ps) = blocks.get(offset) else {
                        continue;
                    };
                    // C(delta) = -sum P_{mu la} P_{nu si} (mu nu | la si), and the energy
                    // contribution is taper * C. Accumulate both C and dC/ddelta.
                    let mut c_value = 0.0;
                    let mut c_grad = [0.0_f64; 3];
                    for mu in 0..na {
                        for la in 0..nb {
                            let pml = ps[(oa + mu, ob + la)];
                            if pml == 0.0 {
                                continue;
                            }
                            for nu in 0..na {
                                for si in 0..nb {
                                    let coef = -scale * pml * ps[(oa + nu, ob + si)];
                                    if coef == 0.0 {
                                        continue;
                                    }
                                    let w = te.two_e(mu, nu, la, si);
                                    c_value += coef * w.v;
                                    for ax in 0..3 {
                                        c_grad[ax] += coef * w.d[ax];
                                    }
                                }
                            }
                        }
                    }
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += taper.d[ax] * c_value + taper.v * c_grad[ax];
                    }
                }
            }
        } else {
            for (blocks, scale) in &spin_blocks {
                let Some(ps) = blocks.get(offset) else {
                    continue;
                };
                for mu in 0..na {
                    for la in 0..nb {
                        let pml = ps[(oa + mu, ob + la)];
                        if pml == 0.0 {
                            continue;
                        }
                        for nu in 0..na {
                            for si in 0..nb {
                                let coef = -scale * pml * ps[(oa + nu, ob + si)];
                                let dw = te.two_e(mu, nu, la, si).d;
                                for (ax, fx) in f.iter_mut().enumerate() {
                                    *fx += coef * dw[ax];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Core-core.
        {
            let dv = [
                Dual::var(delta.x, 0),
                Dual::var(delta.y, 1),
                Dual::var(delta.z, 2),
            ];
            let r = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            let e = crate::repulsion::pair_core_energy_scalar::<Dual>(
                ea,
                eb,
                molecule.atoms[a].z,
                molecule.atoms[b].z,
                r,
            );
            for (ax, fx) in f.iter_mut().enumerate() {
                *fx += e.d[ax];
            }
        }

        // Scatter: `f` is dE/d(delta) with delta = R_b + T - R_a.
        let force = Vec3::new(f[0], f[1], f[2]);
        gradient[b] += force;
        gradient[a] -= force;

        // Pair virial. Uses the *full* separation including the lattice translation, which is
        // what makes this the periodic stress rather than the molecular one.
        let d = [delta.x, delta.y, delta.z];
        for (alpha, row) in virial.iter_mut().enumerate() {
            for (beta, v) in row.iter_mut().enumerate() {
                *v += f[alpha] * d[beta];
            }
        }
    }

    // Long-range monopole force. Hellmann–Feynman at the converged density: the explicit
    // position derivative of `½ Σ_ab Q_a Q_b Δ_ab` only, because the density-response term
    // vanishes at a stationary point. See `crate::pbc::ewald::LongRangeMonopole::energy_gradient`.
    // The Klopman–Ohno tail is included exactly when the SCF included it: a gradient taken against
    // a different `Δ` than the density was converged with is a gradient of a different energy.
    if let Some((monopole, ewald)) = crate::pbc::ewald::LongRangeMonopole::for_molecule_with(
        molecule,
        options
            .klopman_ohno_tail
            .then_some((params, options.realspace_cutoff)),
        &neighbors,
        options.ewald,
    )? {
        let mut charges = Vec::with_capacity(nat);
        for (a, atom) in molecule.atoms.iter().enumerate() {
            let mut population = 0.0;
            let off = basis.atom_offset[a];
            for k in 0..basis.atom_norb[a] {
                population += p0[(off + k, off + k)];
            }
            charges.push(params.element(atom.z)?.core_charge - population);
        }
        let extra = crate::pbc::ewald::LongRangeMonopole::energy_gradient(
            molecule, &neighbors, &ewald, &charges,
        )?;
        for (g, e) in gradient.iter_mut().zip(&extra) {
            *g += *e;
        }
        // No force term for the tail: it has no pair-separation dependence, which the gradient
        // finite difference confirms rather than assumes.
        let strain = crate::pbc::ewald::LongRangeMonopole::energy_strain(
            molecule, &neighbors, &ewald, &charges,
        )?;
        // The tail's own strain, from its `1/V` scaling. Separate because the pair virial above
        // has no term for a separation-independent contribution — see `klopman_ohno_strain`.
        let tail_strain = monopole.klopman_ohno_strain(&charges);
        for i in 0..3 {
            for j in 0..3 {
                virial[i][j] += strain[i][j] + tail_strain[i][j];
            }
        }
    }

    let measure = cell.measure();
    let mut stress = Mat3::zero();
    for alpha in 0..3 {
        for beta in 0..3 {
            let v = virial[alpha][beta] / measure;
            match beta {
                0 => stress.col[0] = set_component(stress.col[0], alpha, v),
                1 => stress.col[1] = set_component(stress.col[1], alpha, v),
                _ => stress.col[2] = set_component(stress.col[2], alpha, v),
            }
        }
    }
    // A non-periodic direction has no stress: there is no cell length to differentiate.
    for axis in 0..3 {
        if !cell.periodic[axis] {
            for other in 0..3 {
                stress.col[axis] = set_component(stress.col[axis], other, 0.0);
                stress.col[other] = set_component(stress.col[other], axis, 0.0);
            }
        }
    }

    // The external field's force, `+Q_a F` on atom `a`.
    //
    // The field is already in `H_core` through `build_realspace_core`, so the energy carries it;
    // without this the force would be the derivative of a different function and dynamics in a
    // field would not conserve. Only the **nuclear** half appears here for the same reason it does
    // in the molecular path: the electronic half is Hellmann-Feynman and already in the terms
    // above, since the dipole operator's dependence on the nuclear positions is linear.
    if let Some(field) = options.electric_field {
        let charges = crate::pbc::ewald::net_charges(molecule, &basis, params, p_total.origin()?)?;
        for (g, e) in gradient
            .iter_mut()
            .zip(&crate::dipole::field_gradient(field, &charges))
        {
            *g += *e;
        }
    }

    let max_gradient = gradient
        .iter()
        .map(|g| g.x.abs().max(g.y.abs()).max(g.z.abs()))
        .fold(0.0, f64::max);
    let forces = gradient.iter().map(|g| *g * -1.0).collect();

    Ok(PbcGradient {
        gradient,
        forces,
        stress,
        max_gradient,
    })
}

#[inline]
fn set_component(mut v: Vec3, index: usize, value: f64) -> Vec3 {
    match index {
        0 => v.x = value,
        1 => v.y = value,
        _ => v.z = value,
    }
    v
}

#[inline]
fn beta_of(elem: &crate::params::Am1Element, orb: u8) -> f64 {
    if orb == 0 {
        elem.beta_s
    } else {
        elem.beta_p
    }
}

/// Convenience: run the SCF and then the gradient.
pub fn pbc_energy_and_gradient(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<(PbcResult, PbcGradient)> {
    let scf = crate::pbc::scf::run_pbc_scf(molecule, params, options)?;
    let grad = pbc_gradient(molecule, params, options, &scf)?;
    Ok((scf, grad))
}

/// Unused import guard: `Matrix` is referenced through the block types.
const _: Option<&Matrix> = None;
