// SPDX-License-Identifier: GPL-3.0-or-later

//! Nuclear gradients of the AM1 total energy.
//!
//! Because NDDO works in an orthonormal AO basis, the SCF energy is stationary with respect to
//! the density, so the nuclear gradient is the derivative of the energy expression at the
//! **fixed converged density** (there is no Pulay/overlap-constraint term). Three routines are
//! provided:
//!
//! * [`closed_form_gradient`] — the primary, **fully closed-form** gradient (forward-mode
//!   dual-number AD of every integral kernel; radial *and* angular overlap analytic for
//!   `n ≤ 3`). No SCF re-runs and no finite differences. This is what the optimizer uses.
//! * [`analytic_gradient`] — the same Hellmann–Feynman gradient with the electronic term taken
//!   by fixed-density central differences (the core-core term stays closed-form). Kept for the
//!   open-shell path and as a cross-check.
//! * [`numerical_gradient`] — a full-SCF central-difference gradient, kept as an independent
//!   correctness reference (each Cartesian component re-runs the SCF twice).

use crate::basis::Basis;
use crate::error::Result;
use crate::fock::build_fock;
use crate::hamiltonian::build_core;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Am1Parameters;
use crate::repulsion::core_core_energy;
use crate::scf::{run_am1, Am1Options, Am1Result};
use crate::system::Molecule;

/// Electronic energy (eV) at a **fixed density** matrix (no SCF, no core-core term).
pub fn electronic_energy_at_fixed_density(
    molecule: &Molecule,
    params: &Am1Parameters,
    density: &Matrix,
) -> Result<f64> {
    let basis = Basis::build(molecule, params)?;
    let core = build_core(molecule, &basis, params)?;
    let f = build_fock(molecule, &basis, params, &core, density)?;
    Ok(0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f)))
}

/// Total AM1 energy (eV) at a **fixed density** (electronic + core-core).
pub fn energy_at_fixed_density(
    molecule: &Molecule,
    params: &Am1Parameters,
    density: &Matrix,
) -> Result<f64> {
    Ok(
        electronic_energy_at_fixed_density(molecule, params, density)?
            + core_core_energy(molecule, params)?,
    )
}

/// Hellmann–Feynman nuclear gradient. `step` is the displacement in Bohr (default 5e-4).
pub fn analytic_gradient(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    step: f64,
) -> Result<GradientResult> {
    use rayon::prelude::*;

    let scf = run_am1(molecule, params, options)?;
    let energy_ev = scf.total_ev;
    let nat = molecule.atoms.len();
    let density = scf.density.clone();

    // Core-core repulsion: exact closed-form derivative.
    let mut gradient = crate::repulsion::core_core_gradient(molecule, params)?;

    // Electronic term: Hellmann-Feynman (fixed converged density) central difference of the
    // electronic energy only — the 3N components are independent, so run them on rayon.
    let comps: Vec<(usize, usize)> = (0..nat).flat_map(|a| (0..3).map(move |k| (a, k))).collect();
    let electronic: Vec<(usize, usize, f64)> = comps
        .par_iter()
        .map(|&(a, k)| -> Result<(usize, usize, f64)> {
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            displace(&mut plus.atoms[a].position, k, step);
            displace(&mut minus.atoms[a].position, k, -step);
            let ep = electronic_energy_at_fixed_density(&plus, params, &density)?;
            let em = electronic_energy_at_fixed_density(&minus, params, &density)?;
            Ok((a, k, (ep - em) / (2.0 * step)))
        })
        .collect::<Result<Vec<_>>>()?;
    for (a, k, g) in electronic {
        match k {
            0 => gradient[a].x += g,
            1 => gradient[a].y += g,
            _ => gradient[a].z += g,
        }
    }
    // The field term in full. `electronic_energy_at_fixed_density` is deliberately field-free —
    // it is the skeleton energy — so the finite difference above contains no part of it, and this
    // supplies both halves at once rather than half here and half there.
    add_external_field_force(molecule, params, options, &density, &mut gradient)?;

    let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    Ok(GradientResult {
        scf,
        energy_ev,
        gradient,
        forces,
        max_gradient,
    })
}

#[derive(Clone, Debug)]
pub struct GradientResult {
    /// Converged SCF result at the input geometry.
    pub scf: Am1Result,
    /// Total energy (eV).
    pub energy_ev: f64,
    /// Gradient dE/dR in eV/Bohr (atomic-unit length).
    pub gradient: Vec<Vec3>,
    /// Forces = −gradient (eV/Bohr).
    pub forces: Vec<Vec3>,
    /// Largest gradient component magnitude (eV/Bohr).
    pub max_gradient: f64,
}

/// Finite-difference nuclear gradient. `step` is the displacement in Bohr (default 5e-4).
pub fn numerical_gradient(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    step: f64,
) -> Result<GradientResult> {
    let scf = run_am1(molecule, params, options)?;
    let energy_ev = scf.total_ev;
    let nat = molecule.atoms.len();
    let mut gradient = vec![Vec3::zero(); nat];

    let energy_at = |m: &Molecule| -> Result<f64> { Ok(run_am1(m, params, options)?.total_ev) };

    for a in 0..nat {
        for k in 0..3 {
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            displace(&mut plus.atoms[a].position, k, step);
            displace(&mut minus.atoms[a].position, k, -step);
            let ep = energy_at(&plus)?;
            let em = energy_at(&minus)?;
            let g = (ep - em) / (2.0 * step);
            set_component(&mut gradient[a], k, g);
        }
    }

    let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));

    Ok(GradientResult {
        scf,
        energy_ev,
        gradient,
        forces,
        max_gradient,
    })
}

/// Fully closed-form (dual-number) Hellmann–Feynman gradient. The two-electron and
/// core-attraction integral derivatives, the overlap (radial *and* angular, for valence shells
/// `n ≤ 3`), and the core-core term are all exact forward-mode AD — no SCF re-runs and no
/// finite differences. (Heavy elements, `n ≥ 4`, keep a tight 1-D radial overlap difference.)
/// Falls back to the fixed-density gradient for open-shell (UHF) systems.
pub fn closed_form_gradient(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
) -> Result<GradientResult> {
    let scf = run_am1(molecule, params, options)?;
    if scf.unrestricted {
        // Open-shell: spin-resolved closed-form fixed-density (Hellmann–Feynman) gradient.
        let energy_ev = scf.total_ev;
        let mut gradient = fixed_density_gradient_uhf(molecule, params, options, &scf)?;
        add_long_range_force(molecule, params, options, &scf.density, &mut gradient)?;
        add_external_field_force(molecule, params, options, &scf.density, &mut gradient)?;
        let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
        let max_gradient = gradient
            .iter()
            .flat_map(|g| g.to_array())
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        return Ok(GradientResult {
            scf,
            energy_ev,
            gradient,
            forces,
            max_gradient,
        });
    }
    let energy_ev = scf.total_ev;
    let neighbors = crate::neighbors::NeighborList::build_screened(
        molecule,
        options.realspace_cutoff,
        options.multipole_cutoff,
    );
    let mut gradient = fixed_density_gradient(
        molecule,
        params,
        &neighbors,
        options.exchange_cutoff,
        &scf.density,
    )?;
    add_long_range_force(molecule, params, options, &scf.density, &mut gradient)?;
    add_external_field_force(molecule, params, options, &scf.density, &mut gradient)?;
    let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    Ok(GradientResult {
        scf,
        energy_ev,
        gradient,
        forces,
        max_gradient,
    })
}

/// Add the long-range monopole force to `gradient`, when the cell has one.
///
/// A no-op for a molecule and whenever `options.ewald` is off.
///
/// # Why this is separate from [`fixed_density_gradient`]
///
/// That function takes a density and no options, so it cannot know whether the long-range
/// correction is in play — and it is also what the fixed-density (skeleton) path uses, which
/// must *not* include a term that depends on the self-consistent charges.
///
/// Keeping it separate is also how this was missed for long enough to matter: the correction
/// went into the energy and into [`crate::pbc::pbc_gradient`], but the molecular gradient path
/// that `run_am1` pairs with kept returning forces from an energy expression it no longer used.
/// The periodic finite-difference tests all went through the other path and stayed green. It
/// took a one- and two-dimensional test — where the correction is large — to show it, as an
/// 0.9 eV/Bohr disagreement.
/// Add the uniform external field's force to `gradient`. A no-op without a field.
///
/// The whole contribution is `∂E/∂R_a = −F Q_a`, i.e. a force `+Q_a F`, because the dipole
/// operator is **linear** in the nuclear positions: only its diagonal `R_a` term moves, and the
/// `s`–`p` hybridization term does not depend on position at all. That linearity is also why the
/// field contributes nothing to the fixed-density second derivative — see
/// [`crate::hessian::analytic_hessian`].
///
/// Separate from [`fixed_density_gradient`] for the same reason the long-range force is: that
/// function takes a density and no options, so it cannot know whether a field is in play, and it
/// is also the skeleton path, which must not carry a term that depends on the self-consistent
/// charges.
pub fn add_external_field_force(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    density: &Matrix,
    gradient: &mut [Vec3],
) -> Result<()> {
    let Some(field) = options.electric_field else {
        return Ok(());
    };
    let basis = Basis::build(molecule, params)?;
    let charges = crate::pbc::ewald::net_charges(molecule, &basis, params, density)?;
    for (g, e) in gradient
        .iter_mut()
        .zip(&crate::dipole::field_gradient(field, &charges))
    {
        *g += *e;
    }
    Ok(())
}

pub fn add_long_range_force(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    density: &Matrix,
    gradient: &mut [Vec3],
) -> Result<()> {
    let neighbors = crate::neighbors::NeighborList::build_screened(
        molecule,
        options.realspace_cutoff,
        options.multipole_cutoff,
    );
    // Far-field monopole force, when distant pairs were screened out of the pair list.
    if let Some(far) =
        crate::farfield::FarField::new(molecule, params, options.multipole_cutoff.unwrap_or(0.0))?
    {
        let basis = Basis::build(molecule, params)?;
        let charges = crate::pbc::ewald::net_charges(molecule, &basis, params, density)?;
        for (g, e) in gradient.iter_mut().zip(&far.gradient(&charges)) {
            *g += *e;
        }
    }
    let Some((_, kernel)) =
        crate::pbc::ewald::LongRangeMonopole::for_molecule(molecule, &neighbors, options.ewald)?
    else {
        return Ok(());
    };
    let basis = Basis::build(molecule, params)?;
    let charges = crate::pbc::ewald::net_charges(molecule, &basis, params, density)?;
    let extra = crate::pbc::ewald::LongRangeMonopole::energy_gradient(
        molecule, &neighbors, &kernel, &charges,
    )?;
    for (g, e) in gradient.iter_mut().zip(&extra) {
        *g += *e;
    }
    Ok(())
}

/// Total closed-form gradient (core-core + electronic) at an **arbitrary fixed density** `p`
/// (no SCF solve). Finite-differencing this over the nuclei at fixed `p` gives the skeleton
/// (fixed-density) second derivative used by the analytic Hessian.
pub fn fixed_density_gradient(
    molecule: &Molecule,
    params: &Am1Parameters,
    neighbors: &crate::neighbors::NeighborList,
    exchange_cutoff: Option<f64>,
    p: &Matrix,
) -> Result<Vec<Vec3>> {
    let basis = Basis::build(molecule, params)?;
    let mut gradient =
        crate::repulsion::core_core_gradient_with_neighbors(molecule, params, neighbors)?;
    let elec =
        electronic_gradient_fixed_density(molecule, params, &basis, neighbors, exchange_cutoff, p)?;
    for (g, e) in gradient.iter_mut().zip(&elec) {
        *g += *e;
    }
    Ok(gradient)
}

/// Electronic part of the closed-form gradient at fixed density `p` (dual-number contraction).
pub fn electronic_gradient_fixed_density(
    molecule: &Molecule,
    params: &Am1Parameters,
    basis: &Basis,
    neighbors: &crate::neighbors::NeighborList,
    exchange_cutoff: Option<f64>,
    p: &Matrix,
) -> Result<Vec<Vec3>> {
    Ok(electronic_gradient_and_virial_fixed_density(
        molecule,
        params,
        basis,
        neighbors,
        exchange_cutoff,
        p,
    )?
    .0)
}

/// [`electronic_gradient_fixed_density`] together with the **pair virial** `Σ f_α δ_β`.
///
/// Every term in this model is a function of pair separations alone, so the pair virial is the
/// whole electronic stress — there is no separate reciprocal-space piece here, because the
/// long-range correction carries its own strain derivative
/// ([`crate::pbc::ewald::LongRangeMonopole::energy_strain`]).
///
/// Returning both from one pass matters: a stress assembled from a second, separately written
/// loop is a second chance to disagree with the gradient about which pairs exist and what the
/// taper does to them.
#[allow(clippy::type_complexity)]
pub fn electronic_gradient_and_virial_fixed_density(
    molecule: &Molecule,
    params: &Am1Parameters,
    basis: &Basis,
    neighbors: &crate::neighbors::NeighborList,
    exchange_cutoff: Option<f64>,
    p: &Matrix,
) -> Result<(Vec<Vec3>, [[f64; 3]; 3])> {
    use crate::dual::Dual;
    use crate::hamiltonian::exchange_taper_scalar;
    use crate::integrals::pair_two_electron_dual;
    use crate::overlap::diatom_overlap_dual;
    use rayon::prelude::*;
    let _t = crate::timing::Timer::start("grad:electronic");
    let nat = molecule.atoms.len();
    let beta = |elem: &crate::params::Am1Element, orb: u8| {
        if orb == 0 {
            elem.beta_s
        } else {
            elem.beta_p
        }
    };

    // The pair loop is embarrassingly parallel: each pair produces one force vector that is
    // added to `b` and subtracted from `a`. Compute the pairs in parallel, then scatter
    // serially — the scatter is O(N²) adds and is not what costs.
    let contributions: Vec<(usize, usize, [f64; 3], Vec3)> = neighbors
        .pairs
        .par_iter()
        .map(|pair| {
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
            let te = pair_two_electron_dual(ea, eb, delta);
            let s = diatom_overlap_dual(ea, pa, eb, pb)?;
            // Image exchange is tapered off, and the taper is differentiated: the exchange
            // energy is `taper(r)·C(δ)`, so leaving out the `taper'(r)` piece gives a force that
            // is not the gradient of the energy being reported.
            let r_dual = crate::dual::Dual {
                v: pair.r,
                d: [delta.x / pair.r, delta.y / pair.r, delta.z / pair.r],
            };
            let taper = match exchange_cutoff {
                Some(rc) => exchange_taper_scalar::<Dual>(r_dual, rc),
                None => Dual::constant(1.0),
            };
            let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
            let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

            let mut f = [0.0_f64; 3];
            for i in 0..na {
                let bi = beta(ea, basis.aos[oa + i].orb);
                for j in 0..nb {
                    let bj = beta(eb, basis.aos[ob + j].orb);
                    let coef = p[(oa + i, ob + j)] * (bi + bj);
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * s[i][j].d[ax];
                    }
                }
            }
            for i in 0..na {
                for j in 0..na {
                    let coef = p[(oa + i, oa + j)];
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * te.e1b[i][j].d[ax];
                    }
                }
            }
            for k in 0..nb {
                for l in 0..nb {
                    let coef = p[(ob + k, ob + l)];
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * te.e2a[k][l].d[ax];
                    }
                }
            }
            for mu in 0..na {
                for nu in 0..na {
                    for la in 0..nb {
                        for si in 0..nb {
                            let w = te.two_e(mu, nu, la, si);
                            let coul = p[(oa + mu, oa + nu)] * p[(ob + la, ob + si)];
                            let exch = -0.5 * p[(oa + mu, ob + la)] * p[(oa + nu, ob + si)];
                            for (ax, fx) in f.iter_mut().enumerate() {
                                // d/dδ [ coul·w + exch·taper·w ]
                                *fx +=
                                    coul * w.d[ax] + exch * (taper.v * w.d[ax] + taper.d[ax] * w.v);
                            }
                        }
                    }
                }
            }
            Ok((a, b, f, delta))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut gradient = vec![Vec3::zero(); nat];
    let mut virial = [[0.0_f64; 3]; 3];
    for (a, b, f, delta) in contributions {
        let force = Vec3::new(f[0], f[1], f[2]);
        gradient[b] += force;
        gradient[a] -= force;
        // Uses the **full** separation including the lattice translation, which is what makes
        // this the periodic virial rather than the molecular one.
        let d = [delta.x, delta.y, delta.z];
        for (alpha, row) in virial.iter_mut().enumerate() {
            for (beta, v) in row.iter_mut().enumerate() {
                *v += f[alpha] * d[beta];
            }
        }
    }
    Ok((gradient, virial))
}

/// Total closed-form UHF gradient (core-core + spin-resolved electronic) at the converged
/// open-shell density. `Pα = (P_tot + S)/2`, `Pβ = (P_tot − S)/2` are reconstructed from the
/// total density and the spin density `S = Pα − Pβ`. Hellmann–Feynman (orthonormal basis).
pub fn fixed_density_gradient_uhf(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    scf: &Am1Result,
) -> Result<Vec<Vec3>> {
    let neighbors = crate::neighbors::NeighborList::build_screened(
        molecule,
        options.realspace_cutoff,
        options.multipole_cutoff,
    );
    let basis = Basis::build(molecule, params)?;
    let pt = &scf.density;
    let spin = scf.spin_density.as_ref().ok_or_else(|| {
        crate::error::Am1Error::InvalidInput("UHF gradient requires a spin density".into())
    })?;
    let mut pa = pt.clone();
    let mut pb = pt.clone();
    {
        let n = pt.as_slice().len();
        let (pas, pbs) = (pa.as_mut_slice(), pb.as_mut_slice());
        let (pts, ss) = (pt.as_slice(), spin.as_slice());
        for i in 0..n {
            pas[i] = 0.5 * (pts[i] + ss[i]);
            pbs[i] = 0.5 * (pts[i] - ss[i]);
        }
    }
    let mut gradient =
        crate::repulsion::core_core_gradient_with_neighbors(molecule, params, &neighbors)?;
    let elec = electronic_gradient_fixed_density_spin(
        molecule,
        params,
        &basis,
        &neighbors,
        options.exchange_cutoff,
        pt,
        &pa,
        &pb,
    )?;
    for (g, e) in gradient.iter_mut().zip(&elec) {
        *g += *e;
    }
    Ok(gradient)
}

/// Spin-resolved electronic part of the closed-form gradient at fixed densities: resonance,
/// electron–core attraction, and Coulomb use the **total** density `P_tot`; exchange uses the
/// **same-spin** densities `Pα`, `Pβ` (`−[Pα_μλ Pα_νσ + Pβ_μλ Pβ_νσ](μν|λσ)`). Reduces to the
/// RHF form when `Pα = Pβ = P_tot/2`.
// Molecule, basis, parameters, core, the two spin densities, the neighbour list and the options.
#[allow(clippy::too_many_arguments)]
pub fn electronic_gradient_fixed_density_spin(
    molecule: &Molecule,
    params: &Am1Parameters,
    basis: &Basis,
    neighbors: &crate::neighbors::NeighborList,
    exchange_cutoff: Option<f64>,
    pt: &Matrix,
    pa: &Matrix,
    pb: &Matrix,
) -> Result<Vec<Vec3>> {
    Ok(electronic_gradient_and_virial_fixed_density_spin(
        molecule,
        params,
        basis,
        neighbors,
        exchange_cutoff,
        pt,
        pa,
        pb,
    )?
    .0)
}

/// [`electronic_gradient_fixed_density_spin`] together with the **pair virial** `Σ f_α δ_β`.
///
/// The open-shell counterpart of [`electronic_gradient_and_virial_fixed_density`], and the reason
/// the divide-and-conquer stress no longer refuses an open shell. Every term is a function of pair
/// separations alone, so the pair virial is the whole electronic stress; the only thing that
/// differs from the restricted form is the exchange coefficient, which reads the same-spin
/// densities instead of half the total.
///
/// It is written as one loop that returns both, for the same reason the restricted version is: a
/// stress assembled from a separately written second loop is a second chance to disagree with the
/// gradient about which pairs exist and what the taper does to them.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn electronic_gradient_and_virial_fixed_density_spin(
    molecule: &Molecule,
    params: &Am1Parameters,
    basis: &Basis,
    neighbors: &crate::neighbors::NeighborList,
    exchange_cutoff: Option<f64>,
    pt: &Matrix,
    pa: &Matrix,
    pb: &Matrix,
) -> Result<(Vec<Vec3>, [[f64; 3]; 3])> {
    use crate::dual::Dual;
    use crate::hamiltonian::exchange_taper_scalar;
    use crate::integrals::pair_two_electron_dual;
    use crate::overlap::diatom_overlap_dual;
    let nat = molecule.atoms.len();
    let mut gradient = vec![Vec3::zero(); nat];
    let mut virial = [[0.0_f64; 3]; 3];
    let beta = |elem: &crate::params::Am1Element, orb: u8| {
        if orb == 0 {
            elem.beta_s
        } else {
            elem.beta_p
        }
    };
    {
        for pair in &neighbors.pairs {
            let eu = params.element(molecule.atoms[pair.i].z)?;
            let ev = params.element(molecule.atoms[pair.j].z)?;
            let (a, b, delta) = if eu.has_p() || !ev.has_p() {
                (pair.i, pair.j, pair.delta)
            } else {
                (pair.j, pair.i, pair.delta * -1.0)
            };
            let ea = params.element(molecule.atoms[a].z)?;
            let eb = params.element(molecule.atoms[b].z)?;
            let pos_a = molecule.atoms[a].position;
            let pos_b = pos_a + delta;
            let te = pair_two_electron_dual(ea, eb, delta);
            let s = diatom_overlap_dual(ea, pos_a, eb, pos_b)?;
            let r_dual = Dual {
                v: pair.r,
                d: [delta.x / pair.r, delta.y / pair.r, delta.z / pair.r],
            };
            let taper = match exchange_cutoff {
                Some(rc) => exchange_taper_scalar::<Dual>(r_dual, rc),
                None => Dual::constant(1.0),
            };
            let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
            let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

            let mut f = [0.0_f64; 3];
            // Resonance β·S (total density).
            for i in 0..na {
                let bi = beta(ea, basis.aos[oa + i].orb);
                for j in 0..nb {
                    let bj = beta(eb, basis.aos[ob + j].orb);
                    let coef = pt[(oa + i, ob + j)] * (bi + bj);
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * s[i][j].d[ax];
                    }
                }
            }
            // Electron–core attraction (total density).
            for i in 0..na {
                for j in 0..na {
                    let coef = pt[(oa + i, oa + j)];
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * te.e1b[i][j].d[ax];
                    }
                }
            }
            for k in 0..nb {
                for l in 0..nb {
                    let coef = pt[(ob + k, ob + l)];
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * te.e2a[k][l].d[ax];
                    }
                }
            }
            // Two-electron: Coulomb from P_tot, exchange from same-spin Pα/Pβ.
            for mu in 0..na {
                for nu in 0..na {
                    for la in 0..nb {
                        for si in 0..nb {
                            let w = te.two_e(mu, nu, la, si);
                            let coul = pt[(oa + mu, oa + nu)] * pt[(ob + la, ob + si)];
                            let exch = -(pa[(oa + mu, ob + la)] * pa[(oa + nu, ob + si)]
                                + pb[(oa + mu, ob + la)] * pb[(oa + nu, ob + si)]);
                            for (ax, fx) in f.iter_mut().enumerate() {
                                *fx +=
                                    coul * w.d[ax] + exch * (taper.v * w.d[ax] + taper.d[ax] * w.v);
                            }
                        }
                    }
                }
            }
            gradient[b] += Vec3::new(f[0], f[1], f[2]);
            gradient[a] -= Vec3::new(f[0], f[1], f[2]);
            // The **full** separation including the lattice translation, exactly as the restricted
            // virial does — that is what makes this the periodic virial and not the molecular one.
            let d = [delta.x, delta.y, delta.z];
            for (alpha, row) in virial.iter_mut().enumerate() {
                for (bta, v) in row.iter_mut().enumerate() {
                    *v += f[alpha] * d[bta];
                }
            }
        }
    }
    Ok((gradient, virial))
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
fn set_component(v: &mut Vec3, k: usize, val: f64) {
    match k {
        0 => v.x = val,
        1 => v.y = val,
        _ => v.z = val,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_matches_full_scf_gradient() {
        // Hellmann–Feynman (fixed-density) gradient must match the full-SCF finite
        // difference on a molecule displaced away from equilibrium (nonzero forces).
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 1.02 0.0 0.0\nH -0.28 0.96 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options::default();
        let a = analytic_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        let n = numerical_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        let mut max_delta = 0.0_f64;
        for (ga, gn) in a.gradient.iter().zip(&n.gradient) {
            for k in 0..3 {
                max_delta = max_delta.max((ga.get(k) - gn.get(k)).abs());
            }
        }
        eprintln!("analytic-vs-numerical gradient max delta = {max_delta:.3e} eV/Bohr");
        assert!(max_delta < 1.0e-4, "gradient mismatch {max_delta:.3e}");
        // Forces must be nonzero for this distorted geometry.
        assert!(a.max_gradient > 1.0e-2);
    }

    #[test]
    fn closed_form_matches_numerical_gradient() {
        // The fully closed-form (dual-number) gradient must match the full-SCF finite
        // difference on a molecule with s and p atoms displaced from equilibrium.
        let mol = Molecule::from_xyz_str(
            "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.03 0.0 1.25\nH 0.95 0.02 -0.55\nH -0.94 -0.03 -0.52\n",
            0.0,
        )
        .unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options::default();
        let cf = closed_form_gradient(&mol, &params, &opts).unwrap();
        let n = numerical_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        let mut max_delta = 0.0_f64;
        for (gc, gn) in cf.gradient.iter().zip(&n.gradient) {
            for k in 0..3 {
                max_delta = max_delta.max((gc.get(k) - gn.get(k)).abs());
            }
        }
        eprintln!("closed-form-vs-numerical gradient max delta = {max_delta:.3e} eV/Bohr");
        assert!(
            max_delta < 5.0e-5,
            "closed-form gradient mismatch {max_delta:.3e}"
        );
    }

    #[test]
    fn closed_form_gradient_heavy_element() {
        // Bromomethane (Br is n = 4): the closed-form gradient now differentiates the numerical
        // Slater overlap analytically (AD through the quadrature), so it must match the full-SCF
        // finite difference — no 1-D radial FD anywhere.
        let mol = Molecule::from_xyz_str(
            "5\nCH3Br\nC 0.0 0.0 0.0\nBr 0.0 0.0 -2.10\nH 1.03 0.0 0.40\nH -0.515 0.892 0.40\nH -0.515 -0.892 0.40\n",
            0.0,
        )
        .unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options::default();
        let cf = closed_form_gradient(&mol, &params, &opts).unwrap();
        let n = numerical_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        let mut max_delta = 0.0_f64;
        for (gc, gn) in cf.gradient.iter().zip(&n.gradient) {
            for k in 0..3 {
                max_delta = max_delta.max((gc.get(k) - gn.get(k)).abs());
            }
        }
        eprintln!("heavy-element closed-form-vs-numerical gradient max delta = {max_delta:.3e}");
        assert!(
            max_delta < 5.0e-4,
            "heavy gradient mismatch {max_delta:.3e}"
        );
        assert!(cf.max_gradient > 1.0e-2);
    }

    #[test]
    fn closed_form_gradient_uhf_radical() {
        // Methyl radical (doublet, UHF), distorted from planar: the spin-resolved closed-form
        // gradient must match the full-SCF finite difference (no fixed-density FD fallback).
        let mol = Molecule::from_xyz_str(
            "4\nmethyl\nC 0.0 0.0 0.05\nH 1.12 0.0 0.0\nH -0.55 0.95 0.0\nH -0.55 -0.95 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options {
            multiplicity: 2,
            ..Am1Options::default()
        };
        let cf = closed_form_gradient(&mol, &params, &opts).unwrap();
        let n = numerical_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        assert!(cf.scf.unrestricted);
        let mut max_delta = 0.0_f64;
        for (gc, gn) in cf.gradient.iter().zip(&n.gradient) {
            for k in 0..3 {
                max_delta = max_delta.max((gc.get(k) - gn.get(k)).abs());
            }
        }
        eprintln!("UHF closed-form-vs-numerical gradient max delta = {max_delta:.3e}");
        assert!(max_delta < 5.0e-5, "UHF gradient mismatch {max_delta:.3e}");
        assert!(cf.max_gradient > 1.0e-2);
    }
}
