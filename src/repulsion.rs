// SPDX-License-Identifier: GPL-3.0-or-later

//! AM1 core–core repulsion energy.
//!
//! `E_core(A,B) = Z_A Z_B γ_AB [1 + f_A + f_B]` where `γ_AB = (s_A s_A | s_B s_B)` is the
//! screened monopole integral, `f = e^{-α R}` (with the MNDO N–H / O–H `R·e^{-α R}`
//! special cases), plus the defining AM1 Gaussian corrections
//! `(Z_A Z_B / R) Σ_k K_k e^{-L_k (R - M_k)²}`. Distances `R` are in Ångström.

use crate::constants::{AM1_A0, AM1_EV};
use crate::dual::Scalar;
use crate::error::Result;
use crate::math::Vec3;
use crate::params::{Am1Element, Am1Parameters};
use crate::system::Molecule;

pub fn core_core_energy(molecule: &Molecule, params: &Am1Parameters) -> Result<f64> {
    let mut energy = 0.0;
    let nat = molecule.atoms.len();
    for i in 0..nat {
        for j in (i + 1)..nat {
            energy += pair_core_energy(
                molecule.atoms[i].z,
                molecule.atoms[j].z,
                molecule.atoms[i].position,
                molecule.atoms[j].position,
                params,
            )?;
        }
    }
    Ok(energy)
}

/// Analytic Cartesian gradient of the core–core repulsion energy (eV/Bohr), returned per
/// atom. Fully closed-form (no finite differences).
pub fn core_core_gradient(molecule: &Molecule, params: &Am1Parameters) -> Result<Vec<Vec3>> {
    let nat = molecule.atoms.len();
    let mut grad = vec![Vec3::zero(); nat];
    for i in 0..nat {
        for j in (i + 1)..nat {
            let (_e, dedr) = pair_core_energy_and_dr(
                molecule.atoms[i].z,
                molecule.atoms[j].z,
                molecule.atoms[i].position,
                molecule.atoms[j].position,
                params,
            )?;
            // r = |R_j − R_i|; dr/dR_i = (R_i − R_j)/r = −û(i→j).
            let d = molecule.atoms[j].position - molecule.atoms[i].position;
            let r = d.norm();
            let unit = d / r;
            let gi = unit * (-dedr); // dE/dR_i
            grad[i] += gi;
            grad[j] -= gi;
        }
    }
    Ok(grad)
}

/// Core–core pair energy (eV) and its radial derivative `dE/dr` (eV/Bohr), closed form.
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
    let s = r * AM1_A0; // Ångström
    let rho = ei.rho0 + ej.rho0;
    let denom = r * r + rho * rho;
    let gam = AM1_EV / denom.sqrt();
    let dgam_dr = -gam * r / denom;
    let zz = ei.core_charge * ej.core_charge;

    let i_special = matches!(zi, 7 | 8) && zj == 1;
    let j_special = matches!(zj, 7 | 8) && zi == 1;

    // f and df/dr for each exponential term (s = r·a0, so d/dr = a0·d/ds).
    let (fi, dfi_dr) = exp_term(ei.alpha, s, i_special);
    let (fj, dfj_dr) = exp_term(ej.alpha, s, j_special);

    let term1 = zz * gam * (1.0 + fi + fj);
    let dterm1 = zz * (dgam_dr * (1.0 + fi + fj) + gam * (dfi_dr + dfj_dr));

    // Gaussian term: zz/s · G, with G = Σ_k K exp(−L (s−M)²).
    let (gi, dgi_ds) = gaussians(&ei.gauss, s);
    let (gj, dgj_ds) = gaussians(&ej.gauss, s);
    let gsum = gi + gj;
    let dgsum_ds = dgi_ds + dgj_ds;
    let term2 = zz * gsum / s;
    // d/dr[zz·G/s] = zz·a0·(G'·s − G)/s²
    let dterm2 = zz * AM1_A0 * (dgsum_ds * s - gsum) / (s * s);

    Ok((term1 + term2, dterm1 + dterm2))
}

/// Core–core pair energy (eV) as a generic scalar of the interatomic distance `r` (Bohr).
/// Instantiated at `f64` for the energy, and at [`crate::dual2::Dual2`] (seeding `r` on the
/// displacement) for the exact closed-form core–core contribution to the analytic Hessian.
/// Mirrors [`pair_core_energy`] term-for-term.
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

/// Exponential core term `f` and `df/dr`. `special` selects the MNDO N–H/O–H `R·e^{−αs}` form.
fn exp_term(alpha: f64, s: f64, special: bool) -> (f64, f64) {
    let e = (-alpha * s).exp();
    if special {
        // f = s·e^{−αs}; df/ds = e^{−αs}(1 − αs); df/dr = a0·df/ds
        (s * e, AM1_A0 * e * (1.0 - alpha * s))
    } else {
        // f = e^{−αs}; df/ds = −α e^{−αs}; df/dr = a0·(−α f)
        (e, AM1_A0 * (-alpha * e))
    }
}

/// Sum of AM1 Gaussians `G = Σ K exp(−L (s−M)²)` and `dG/ds`.
fn gaussians(gauss: &[(f64, f64, f64)], s: f64) -> (f64, f64) {
    let mut g = 0.0;
    let mut dg = 0.0;
    for &(k, l, m) in gauss {
        let e = (-l * (s - m).powi(2)).exp();
        g += k * e;
        dg += k * e * (-2.0 * l * (s - m));
    }
    (g, dg)
}

/// Core–core repulsion energy (eV) for one atom pair.
pub fn pair_core_energy(
    zi: u8,
    zj: u8,
    pos_i: crate::math::Vec3,
    pos_j: crate::math::Vec3,
    params: &Am1Parameters,
) -> Result<f64> {
    let ei = params.element(zi)?;
    let ej = params.element(zj)?;
    let r_bohr = (pos_j - pos_i).norm();
    let rija = r_bohr * AM1_A0; // Ångström
    let gam = AM1_EV / (r_bohr * r_bohr + (ei.rho0 + ej.rho0).powi(2)).sqrt();
    let t1 = ei.core_charge * ej.core_charge * gam;

    // N–H / O–H special case: the heavy-atom exponential carries an extra R factor.
    let i_special = matches!(zi, 7 | 8) && zj == 1;
    let j_special = matches!(zj, 7 | 8) && zi == 1;
    let mut fi = (-ei.alpha * rija).exp();
    if i_special {
        fi *= rija;
    }
    let mut fj = (-ej.alpha * rija).exp();
    if j_special {
        fj *= rija;
    }

    let mut e = t1 * (1.0 + fi + fj);

    // AM1 Gaussian corrections.
    let t4 = ei.core_charge * ej.core_charge / rija;
    let mut g_sum = 0.0;
    for &(k, l, m) in &ei.gauss {
        g_sum += k * (-l * (rija - m).powi(2)).exp();
    }
    for &(k, l, m) in &ej.gauss {
        g_sum += k * (-l * (rija - m).powi(2)).exp();
    }
    e += t4 * g_sum;

    Ok(e)
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
        assert!(max_delta < 1e-6, "core-core gradient mismatch {max_delta:.3e}");
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
        assert!((e_scalar - e_ref).abs() < 1e-10, "scalar {e_scalar} vs ref {e_ref}");

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
