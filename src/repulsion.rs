// SPDX-License-Identifier: GPL-3.0-or-later

//! AM1 core–core repulsion energy.
//!
//! `E_core(A,B) = Z_A Z_B γ_AB [1 + f_A + f_B]` where `γ_AB = (s_A s_A | s_B s_B)` is the
//! screened monopole integral, `f = e^{-α R}` (with the MNDO N–H / O–H `R·e^{-α R}`
//! special cases), plus the defining AM1 Gaussian corrections
//! `(Z_A Z_B / R) Σ_k K_k e^{-L_k (R - M_k)²}`. Distances `R` are in Ångström.

use crate::constants::{AM1_A0, AM1_EV};
use crate::dual::{Dual, Scalar};
use crate::error::Result;
use crate::math::Vec3;
use crate::neighbors::NeighborList;
use crate::params::{Am1Element, Am1Parameters};
use crate::system::Molecule;

pub fn core_core_energy(molecule: &Molecule, params: &Am1Parameters) -> Result<f64> {
    core_core_energy_with_neighbors(molecule, params, &NeighborList::molecular(molecule))
}

/// Core–core repulsion over an explicit pair list (eV).
///
/// For a periodic list this is the energy **per cell**: the list holds each physical pair
/// once, which is exactly the `½ Σ_{i,j,T}` of the lattice sum with the double counting
/// already removed.
///
/// Every term here — the screened monopole, the `e^{−αR}` factors, the N–H/O–H special case
/// and the AM1 Gaussian corrections — is summed over images with no change of form, because
/// they are all functions of one interatomic distance. That is what makes the AM1 core–core
/// corrections periodic without a separate implementation.
///
/// The Gaussians and exponentials are short-ranged, but the `Z_A Z_B γ_AB` monopole term is
/// not: it decays as `1/R` and this real-space sum is therefore only conditionally convergent.
/// Truncating it at a cutoff is an approximation whose error grows with the cutoff for a charged
/// cell; the correction that removes it lives in [`crate::pbc::ewald`] and is applied through the
/// net charges in [`crate::fock::long_range_potential`], **not** here.
pub fn core_core_energy_with_neighbors(
    molecule: &Molecule,
    params: &Am1Parameters,
    neighbors: &NeighborList,
) -> Result<f64> {
    let mut energy = 0.0;
    for p in &neighbors.pairs {
        let ei = params.element(molecule.atoms[p.i].z)?;
        let ej = params.element(molecule.atoms[p.j].z)?;
        energy += pair_core_energy_scalar::<f64>(
            ei,
            ej,
            molecule.atoms[p.i].z,
            molecule.atoms[p.j].z,
            p.r,
        );
    }
    Ok(energy)
}

/// Analytic Cartesian gradient of the core–core repulsion energy (eV/Bohr), returned per
/// atom. Fully closed-form (no finite differences).
pub fn core_core_gradient(molecule: &Molecule, params: &Am1Parameters) -> Result<Vec<Vec3>> {
    core_core_gradient_with_neighbors(molecule, params, &NeighborList::molecular(molecule))
}

/// [`core_core_gradient`] over an explicit pair list, so it works under a cell.
///
/// A self-image pair (`i == j`, `T ≠ 0`) contributes nothing and the scatter below produces
/// that automatically: moving the atom moves its image with it, so the separation — and the
/// energy — does not change, and `grad[i] += g; grad[j] -= g` with `i == j` cancels exactly.
pub fn core_core_gradient_with_neighbors(
    molecule: &Molecule,
    params: &Am1Parameters,
    neighbors: &NeighborList,
) -> Result<Vec<Vec3>> {
    Ok(core_core_gradient_and_virial(molecule, params, neighbors)?.0)
}

/// [`core_core_gradient_with_neighbors`] together with its pair virial `Σ f_α δ_β`.
///
/// The core–core energy depends only on `|δ|`, so its virial is `(dE/dr) δ_α δ_β / r`.
#[allow(clippy::type_complexity)]
pub fn core_core_gradient_and_virial(
    molecule: &Molecule,
    params: &Am1Parameters,
    neighbors: &NeighborList,
) -> Result<(Vec<Vec3>, [[f64; 3]; 3])> {
    let _t = crate::timing::Timer::start("grad:core_core");
    let mut grad = vec![Vec3::zero(); molecule.atoms.len()];
    let mut virial = [[0.0_f64; 3]; 3];
    for p in &neighbors.pairs {
        let ei = params.element(molecule.atoms[p.i].z)?;
        let ej = params.element(molecule.atoms[p.j].z)?;
        let e = pair_core_energy_scalar::<Dual>(
            ei,
            ej,
            molecule.atoms[p.i].z,
            molecule.atoms[p.j].z,
            Dual::var(p.r, 0),
        );
        let dedr = e.d[0];
        // r = |δ|; dr/dR_i = −δ̂, dr/dR_j = +δ̂, with δ the stored displacement.
        let unit = p.delta / p.r;
        let gi = unit * (-dedr);
        grad[p.i] += gi;
        grad[p.j] -= gi;
        // f = dE/d(delta) = dedr * unit; virial = f_alpha * delta_beta.
        let f = [unit.x * dedr, unit.y * dedr, unit.z * dedr];
        let d = [p.delta.x, p.delta.y, p.delta.z];
        for (alpha, row) in virial.iter_mut().enumerate() {
            for (beta, v) in row.iter_mut().enumerate() {
                *v += f[alpha] * d[beta];
            }
        }
    }
    Ok((grad, virial))
}

/// Core–core pair energy (eV) and its radial derivative `dE/dr` (eV/Bohr), closed form.
///
/// Forward-mode AD of [`pair_core_energy_scalar`], so the derivative is by construction the
/// derivative *of the energy this crate actually evaluates*. This used to be a separate
/// hand-written transcription: the two agreed, but nothing enforced that, and the gradient
/// and the Hessian were reading different copies of the same formula.
pub fn pair_core_energy_and_dr(
    zi: u8,
    zj: u8,
    pos_i: Vec3,
    pos_j: Vec3,
    params: &Am1Parameters,
) -> Result<(f64, f64)> {
    let ei = params.element(zi)?;
    let ej = params.element(zj)?;
    let r = (pos_j - pos_i).norm(); // Bohr
    let e = pair_core_energy_scalar::<Dual>(ei, ej, zi, zj, Dual::var(r, 0));
    Ok((e.v, e.d[0]))
}

/// Core–core pair energy (eV) as a generic scalar of the interatomic distance `r` (Bohr).
///
/// **This is the single definition of the AM1 core–core term.** The energy instantiates it at
/// `f64`, the gradient at [`crate::dual::Dual`], and the analytic Hessian at
/// [`crate::dual2::Dual2`] (seeding `r` on the interatomic displacement, so the Cartesian
/// chain rule runs through `sqrt`). One expression means the derivatives cannot drift from
/// the energy — and it is what lets a periodic lattice sum reuse the term unchanged.
pub fn pair_core_energy_scalar<S: Scalar>(
    ei: &Am1Element,
    ej: &Am1Element,
    zi: u8,
    zj: u8,
    r: S,
) -> S {
    let s = r * AM1_A0; // Ångström
    let rho = ei.rho0 + ej.rho0; // f64
    let gam = (r * r + rho * rho).sqrt().recip() * AM1_EV;
    let zz = ei.core_charge * ej.core_charge; // f64

    let i_special = matches!(zi, 7 | 8) && zj == 1;
    let j_special = matches!(zj, 7 | 8) && zi == 1;
    let fi = exp_term_scalar(ei.alpha, s, i_special);
    let fj = exp_term_scalar(ej.alpha, s, j_special);

    // Z_A Z_B γ (1 + f_A + f_B).
    let term1 = gam * (fi + fj + 1.0) * zz;
    // AM1 Gaussian corrections: Z_A Z_B G / s.
    let gsum = gaussians_scalar(&ei.gauss, s) + gaussians_scalar(&ej.gauss, s);
    let term2 = gsum * zz / s;
    term1 + term2
}

/// Exponential core term `f`, generic scalar. `special` selects the N–H/O–H `s·e^{−αs}` form.
fn exp_term_scalar<S: Scalar>(alpha: f64, s: S, special: bool) -> S {
    let e = (s * (-alpha)).exp();
    if special {
        s * e
    } else {
        e
    }
}

/// Sum of AM1 Gaussians `G = Σ K exp(−L (s−M)²)`, generic scalar.
fn gaussians_scalar<S: Scalar>(gauss: &[(f64, f64, f64)], s: S) -> S {
    let mut g = S::cst(0.0);
    for &(k, l, m) in gauss {
        let d = s - m;
        let e = (d * d * (-l)).exp();
        g = g + e * k;
    }
    g
}

/// Core–core repulsion energy (eV) for one atom pair.
pub fn pair_core_energy(
    zi: u8,
    zj: u8,
    pos_i: Vec3,
    pos_j: Vec3,
    params: &Am1Parameters,
) -> Result<f64> {
    let ei = params.element(zi)?;
    let ej = params.element(zj)?;
    let r = (pos_j - pos_i).norm(); // Bohr
    Ok(pair_core_energy_scalar::<f64>(ei, ej, zi, zj, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Molecule;

    #[test]
    fn analytic_core_core_gradient_matches_fd() {
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 1.02 0.05 0.0\nH -0.28 0.96 0.10\n",
            0.0,
        )
        .unwrap();
        let params = Am1Parameters::standard().unwrap();
        let analytic = core_core_gradient(&mol, &params).unwrap();
        let step = 1.0e-5;
        let mut max_delta = 0.0_f64;
        for a in 0..mol.atoms.len() {
            for k in 0..3 {
                let mut plus = mol.clone();
                let mut minus = mol.clone();
                match k {
                    0 => {
                        plus.atoms[a].position.x += step;
                        minus.atoms[a].position.x -= step;
                    }
                    1 => {
                        plus.atoms[a].position.y += step;
                        minus.atoms[a].position.y -= step;
                    }
                    _ => {
                        plus.atoms[a].position.z += step;
                        minus.atoms[a].position.z -= step;
                    }
                }
                let fd = (core_core_energy(&plus, &params).unwrap()
                    - core_core_energy(&minus, &params).unwrap())
                    / (2.0 * step);
                let an = match k {
                    0 => analytic[a].x,
                    1 => analytic[a].y,
                    _ => analytic[a].z,
                };
                max_delta = max_delta.max((an - fd).abs());
            }
        }
        eprintln!("core-core analytic-vs-FD gradient max delta = {max_delta:.2e} eV/Bohr");
        assert!(
            max_delta < 1e-6,
            "core-core gradient mismatch {max_delta:.3e}"
        );
    }

    #[test]
    fn core_core_scalar_matches_energy_and_second_derivative() {
        use crate::dual2::Dual2;
        let params = Am1Parameters::standard().unwrap();
        // O–H pair (exercises the N/O–H special exponential case).
        let (zi, zj) = (8u8, 1u8);
        let ei = params.element(zi).unwrap();
        let ej = params.element(zj).unwrap();
        let pi = Vec3::new(0.0, 0.0, 0.0);
        let pj = Vec3::new(1.3, -0.7, 0.4);
        let r = (pj - pi).norm();

        // f64 scalar path must equal the reference pair energy.
        let e_scalar = pair_core_energy_scalar::<f64>(ei, ej, zi, zj, r);
        let e_ref = pair_core_energy(zi, zj, pi, pj, &params).unwrap();
        assert!(
            (e_scalar - e_ref).abs() < 1e-10,
            "scalar {e_scalar} vs ref {e_ref}"
        );

        // Dual2 second derivative (w.r.t. the 1-D distance) vs finite difference.
        let rd = Dual2::var(r, 0);
        let ed = pair_core_energy_scalar::<Dual2>(ei, ej, zi, zj, rd);
        let h = 1e-5;
        let ep = pair_core_energy_scalar::<f64>(ei, ej, zi, zj, r + h);
        let em = pair_core_energy_scalar::<f64>(ei, ej, zi, zj, r - h);
        let fd2 = (ep - 2.0 * e_ref + em) / (h * h);
        assert!(
            (ed.h[0][0] - fd2).abs() < 1e-3,
            "core-core d2E/dr2 {} vs FD {}",
            ed.h[0][0],
            fd2
        );
    }
}
