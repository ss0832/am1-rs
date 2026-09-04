// SPDX-License-Identifier: GPL-3.0-or-later

//! The CPHF coefficients `U^j_{ai}` against a finite difference of the MO coefficients.
//!
//! Everything else that uses `U` contracts it into something else first — the Hessian sums
//! `4 G^a : U^b`, the atomic polar tensor takes `Tr[(∂P/∂R) M_α]` — and a contraction can be
//! right while its ingredient is wrong in a way the contraction is blind to. A sign error on a
//! single `(a, i)` pair changes both of those by an amount that looks like ordinary numerical
//! noise but is not.
//!
//! The definition being tested is what `U` *means*: to first order in a nuclear displacement the
//! occupied orbitals rotate into the virtual space as
//!
//! ```text
//! |i(R + h e_j)⟩ = |i⟩ + h Σ_a U^j_{ai} |a⟩ + O(h²)
//! ```
//!
//! so `U^j_{ai} = Cᵥᵀ · ∂C_i/∂R_j`, component by component.
//!
//! # Three things make the comparison delicate, and each is handled explicitly
//!
//! **Phase.** An eigenvector is defined only up to a sign, and a diagonalizer is free to return
//! `−C_i` at the displaced geometry. Differencing then gives `−2C_i/2h` — enormous and
//! meaningless. Each displaced orbital is sign-aligned before differencing.
//!
//! **Which reference to align against.** Against the *response channel's own* coefficients, not
//! against a separately re-run SCF. Two independent SCF solves of the same geometry may return
//! opposite signs for the same orbital, and aligning to the wrong one flips `U` wholesale: the
//! first version of this test did exactly that and measured `|Δ| = 2|U|` on the β channel.
//!
//! **The occupied–occupied block is not `U`.** A rotation among occupied orbitals leaves the
//! density unchanged, so the CPHF neither determines nor needs it, while the finite difference
//! does contain one. Only the virtual–occupied projection is compared, which is the block `U`
//! spans.

use am1_rs::hessian::{analytic_hessian_with_response, ResponseChannel};
use am1_rs::linalg::Matrix;
use am1_rs::math::Vec3;
use am1_rs::scf::{run_am1, Am1Options, Am1Result, ScfReference};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;
/// Bohr. The error is `ε/2h` from the SCF convergence floor plus `O(h²)` truncation; at `1e-4`
/// the first dominates and the test measures the SCF tolerance instead of the response.
const STEP: f64 = 1.0e-3;

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.9584 * ANG, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.2400 * ANG, 0.9278 * ANG, 0.0),
        },
    ])
}

fn methyl() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.079 * ANG, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.5395 * ANG, 0.9344 * ANG, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.5395 * ANG, -0.9344 * ANG, 0.0),
        },
    ])
}

fn tight() -> Am1Options {
    Am1Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 800,
        ..Am1Options::default()
    }
}

fn displaced(molecule: &Molecule, dof: usize, shift: f64) -> Molecule {
    let mut moved = molecule.clone();
    let (atom, axis) = (dof / 3, dof % 3);
    match axis {
        0 => moved.atoms[atom].position.x += shift,
        1 => moved.atoms[atom].position.y += shift,
        _ => moved.atoms[atom].position.z += shift,
    }
    moved
}

/// The occupied columns at a displaced geometry, sign-aligned to `reference` column by column.
///
/// Returns `None` if any orbital's overlap with its reference has fallen below `0.9`, which means
/// the level order changed under the displacement and column `i` is no longer the same orbital.
/// Silently differencing two different orbitals is the failure mode this guards against.
fn aligned_occupied(
    scf: &Am1Result,
    channel_of: fn(&Am1Result) -> &Matrix,
    reference: &Matrix,
) -> Option<Matrix> {
    let full = channel_of(scf);
    let (nao, n_occ) = (reference.rows, reference.cols);
    let mut out = Matrix::zeros(nao, n_occ);
    for i in 0..n_occ {
        let mut overlap = 0.0;
        for mu in 0..nao {
            overlap += full[(mu, i)] * reference[(mu, i)];
        }
        if overlap.abs() < 0.9 {
            return None;
        }
        let sign = if overlap < 0.0 { -1.0 } else { 1.0 };
        for mu in 0..nao {
            out[(mu, i)] = sign * full[(mu, i)];
        }
    }
    Some(out)
}

fn alpha_coeff(r: &Am1Result) -> &Matrix {
    &r.mo_coeff
}

fn beta_coeff(r: &Am1Result) -> &Matrix {
    &r.beta
        .as_ref()
        .expect("beta orbitals must be reported")
        .coeff
}

/// `U^j` against `Cᵥᵀ (∂C_occ/∂R_j)` from a central difference of two full SCF solves.
fn check(
    name: &str,
    molecule: &Molecule,
    options: &Am1Options,
    channel: &ResponseChannel,
    channel_of: fn(&Am1Result) -> &Matrix,
) {
    let params = Am1Parameters::standard().unwrap();
    let (n_vir, n_occ) = (channel.virtuals.cols, channel.occupied.cols);
    let nao = channel.occupied.rows;

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for dof in 0..3 * molecule.atoms.len() {
        let solve = |shift: f64| {
            let scf = run_am1(&displaced(molecule, dof, shift), &params, options).unwrap();
            aligned_occupied(&scf, channel_of, &channel.occupied)
                .unwrap_or_else(|| panic!("{name}: orbital order changed at dof {dof}"))
        };
        let (cp, cm) = (solve(STEP), solve(-STEP));

        // U_ai = Σ_μ Cᵥ[μ,a] · dC[μ,i]/dR — the derivative projected onto the virtual space.
        for a in 0..n_vir {
            for i in 0..n_occ {
                let mut fd = 0.0;
                for mu in 0..nao {
                    fd += channel.virtuals[(mu, a)] * (cp[(mu, i)] - cm[(mu, i)]) / (2.0 * STEP);
                }
                let analytic = channel.u_ov[dof][(a, i)];
                scale = scale.max(analytic.abs());
                worst = worst.max((analytic - fd).abs());
            }
        }
    }
    eprintln!(
        "    {name}: max |U analytic - finite difference| = {worst:.3e}, largest |U| = {scale:.3e}"
    );
    assert!(
        scale > 1.0e-3,
        "{name}: every U is ~zero ({scale:.3e}); the test would pass on a stub"
    );
    assert!(
        worst < 5.0e-5,
        "{name}: U disagrees with the finite difference by {worst:.3e}"
    );
}

#[test]
fn the_cphf_coefficients_are_the_derivative_of_the_orbitals() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water();
    let options = Am1Options {
        reference: ScfReference::Restricted,
        ..tight()
    };
    let r = analytic_hessian_with_response(&molecule, &params, &options, STEP).unwrap();
    check("water (RHF)", &molecule, &options, &r.alpha, alpha_coeff);
}

/// The β channel has its own `U`, its own denominators and its own coupling to α, so checking
/// only the restricted path would leave `ucphf_ov` unvalidated against anything external.
///
/// The system is the **water cation**, not methyl, and the reason is not incidental. Methyl is
/// `D₃ₕ` and its `e′` orbitals are degenerate; degenerate orbitals mix arbitrarily under a
/// displacement, so column `i` at `+h` is not the same orbital as column `i` at `−h`, and a
/// column-by-column finite difference of the *coefficients* has no well-defined limit. That is a
/// property of the comparison, not a defect in `U` — the response **density** is invariant under
/// such a mixing, and it is checked on methyl below, where it agrees to 1.7e-7. `H₂O⁺` is `C₂ᵥ`
/// with no degeneracy, so here the column-wise comparison means something.
#[test]
fn the_unrestricted_cphf_coefficients_are_too() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water();
    let options = Am1Options {
        charge: 1.0,
        multiplicity: 2,
        reference: ScfReference::Unrestricted,
        ..tight()
    };
    let r = analytic_hessian_with_response(&molecule, &params, &options, STEP).unwrap();
    check("H2O+ UHF alpha", &molecule, &options, &r.alpha, alpha_coeff);
    let beta = r.beta.as_ref().expect("an unrestricted run carries beta");
    check("H2O+ UHF beta", &molecule, &options, beta, beta_coeff);
}

/// The response density built from `U` must equal a finite difference of the density itself.
///
/// Separate, because `response_density` applies the occupation weight and the
/// `C_v U C_oᵀ + transpose` symmetrization on top of `U`. Either could be wrong while `U` is
/// right — the weight in particular is 2 for RHF and 1 per UHF channel, and swapping them would
/// leave every symmetry intact. The density is also phase-independent, so this half needs none
/// of the alignment machinery above and is a genuinely independent route.
#[test]
fn the_response_density_is_the_derivative_of_the_density() {
    let params = Am1Parameters::standard().unwrap();
    for (name, molecule, options) in [
        (
            "water (RHF)",
            water(),
            Am1Options {
                reference: ScfReference::Restricted,
                ..tight()
            },
        ),
        (
            "methyl (UHF)",
            methyl(),
            Am1Options {
                multiplicity: 2,
                reference: ScfReference::Unrestricted,
                ..tight()
            },
        ),
    ] {
        let r = analytic_hessian_with_response(&molecule, &params, &options, STEP).unwrap();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for dof in 0..3 * molecule.atoms.len() {
            let analytic = r.response_density(dof);
            let density = |shift: f64| {
                run_am1(&displaced(&molecule, dof, shift), &params, &options)
                    .unwrap()
                    .density
            };
            let (pp, pm) = (density(STEP), density(-STEP));
            for k in 0..analytic.as_slice().len() {
                let fd = (pp.as_slice()[k] - pm.as_slice()[k]) / (2.0 * STEP);
                scale = scale.max(analytic.as_slice()[k].abs());
                worst = worst.max((analytic.as_slice()[k] - fd).abs());
            }
        }
        eprintln!(
            "    {name}: max |dP/dR analytic - finite difference| = {worst:.3e}, \
             largest |dP/dR| = {scale:.3e}"
        );
        assert!(scale > 1.0e-3, "{name}: every dP/dR is ~zero ({scale:.3e})");
        assert!(
            worst < 5.0e-5,
            "{name}: the response density disagrees by {worst:.3e}"
        );
    }
}
