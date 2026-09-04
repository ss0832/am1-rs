// SPDX-License-Identifier: GPL-3.0-or-later

//! The atomic polar tensor and infrared intensities.
//!
//! The tensor `∂μ_α/∂R_{a,β}` is the whole content here — intensities are a projection of it onto
//! normal modes — so it is what gets checked, three independent ways:
//!
//! 1. the **translational sum rule** `Σ_a ∂μ_α/∂R_{a,β} = q δ_αβ`, which is `3 × 3` exact
//!    constraints following from charge conservation alone;
//! 2. a **finite difference of the dipole** over nuclear displacements, at full SCF — sharing no
//!    code with the CPHF path;
//! 3. the **interchange theorem**, `∂μ_α/∂R_j = −∂²E/∂F_α∂R_j`, obtained by differencing the
//!    *analytic gradient* with respect to an applied field. This one goes through the
//!    field-perturbed SCF rather than the nuclear response, so it exercises a different half of
//!    the code than (2) does.
//!
//! A per-mode "this should be dark by symmetry" check is included, but last and with the
//! displacement written out explicitly rather than guessed from the diagonalizer's ordering.

use am1_rs::gradient::closed_form_gradient;
use am1_rs::ir::{dipole_derivatives, ir_spectrum};
use am1_rs::math::Vec3;
use am1_rs::{run_am1, Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;
const AU_DIPOLE_TO_DEBYE: f64 = 2.541_746_473;

fn tight(charge: f64, multiplicity: usize, field: Option<Vec3>) -> Am1Options {
    Am1Options {
        charge,
        multiplicity,
        electric_field: field,
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 500,
        ..Am1Options::default()
    }
}

fn water() -> Molecule {
    Molecule::from_xyz_str(
        "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n",
        0.0,
    )
    .unwrap()
}

/// Linear, centrosymmetric CO₂ at the AM1 optimized bond length. Exactly `D∞h` by construction,
/// which is what makes the symmetry statement below rigorous rather than approximate.
fn co2() -> Molecule {
    let b = 1.189_308_342 * ANG;
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 8,
            position: Vec3::new(b, 0.0, 0.0),
        },
        Atom {
            z: 8,
            position: Vec3::new(-b, 0.0, 0.0),
        },
    ])
}

/// Dipole in e·Bohr from a converged SCF.
fn dipole(mol: &Molecule, params: &Am1Parameters, opts: &Am1Options) -> Vec3 {
    run_am1(mol, params, opts).unwrap().dipole_debye / AU_DIPOLE_TO_DEBYE
}

/// `Σ_a ∂μ_α/∂R_{a,β} = q δ_αβ`. Translating the whole molecule moves its net charge and nothing
/// else — a statement about charge conservation, so a violation is a defect in the response.
#[test]
fn the_atomic_polar_tensor_obeys_the_translational_sum_rule() {
    let params = Am1Parameters::standard().unwrap();
    for (mol, charge, multiplicity) in [
        (water(), 0.0, 1),
        (water().with_charge(1.0), 1.0, 2),
        (co2(), 0.0, 1),
    ] {
        let apt = dipole_derivatives(&mol, &params, &tight(charge, multiplicity, None)).unwrap();
        let nat = mol.atoms.len();
        let mut worst = 0.0_f64;
        for alpha in 0..3 {
            for beta in 0..3 {
                let sum: f64 = (0..nat).map(|a| apt[(alpha, 3 * a + beta)]).sum();
                let expected = if alpha == beta { charge } else { 0.0 };
                worst = worst.max((sum - expected).abs());
            }
        }
        eprintln!("    charge {charge:+.1}, {nat} atoms: max |Σ_a APT − qδ| = {worst:.3e} e");
        assert!(
            worst < 1.0e-8,
            "the APT sum rule is violated by {worst:.3e} e"
        );
    }
}

/// The atomic polar tensor against a finite difference of the dipole, at full SCF.
///
/// The reference re-converges the SCF at every displaced geometry and reads the dipole the SCF
/// reports; the analytic side contracts the CPHF response against the dipole operator. They share
/// only the parameter table.
#[test]
fn the_atomic_polar_tensor_matches_a_dipole_finite_difference() {
    let params = Am1Parameters::standard().unwrap();
    let mol = water();
    let opts = tight(0.0, 1, None);
    let apt = dipole_derivatives(&mol, &params, &opts).unwrap();

    // The step is chosen at the minimum of (SCF noise)/2h + h²·M/6, not made as small as
    // possible. The dipole converges to roughly 1e-9 e·Bohr at these tolerances, so a 2e-4 step
    // amplifies that to ~2.5e-6 — which is what a first attempt measured, and it is noise in the
    // *reference*, not error in the tensor. At 1e-3 the two error terms are both a few 1e-7.
    let h = 1.0e-3; // Bohr
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for beta in 0..3 {
            let mut plus = mol.clone();
            let mut minus = mol.clone();
            match beta {
                0 => {
                    plus.atoms[a].position.x += h;
                    minus.atoms[a].position.x -= h;
                }
                1 => {
                    plus.atoms[a].position.y += h;
                    minus.atoms[a].position.y -= h;
                }
                _ => {
                    plus.atoms[a].position.z += h;
                    minus.atoms[a].position.z -= h;
                }
            }
            let dp = dipole(&plus, &params, &opts);
            let dm = dipole(&minus, &params, &opts);
            for alpha in 0..3 {
                let numeric = (dp.get(alpha) - dm.get(alpha)) / (2.0 * h);
                let analytic = apt[(alpha, 3 * a + beta)];
                worst = worst.max((numeric - analytic).abs());
                scale = scale.max(analytic.abs());
            }
        }
    }
    eprintln!(
        "    max |APT − FD(dipole)| = {worst:.3e} e, largest element {scale:.3e} e \
         ({:.2e} relative)",
        worst / scale
    );
    assert!(
        worst < 2.0e-6,
        "the APT disagrees with a dipole finite difference by {worst:.3e} e"
    );
}

/// The interchange theorem: `∂μ_α/∂R_j = −∂²E/∂F_α∂R_j`.
///
/// Obtained here by differencing the **analytic gradient** with respect to an applied field, so
/// the reference runs through the field-perturbed SCF and the field's own CPHF contribution —
/// a different half of the code from the nuclear response the analytic APT uses.
#[test]
fn the_atomic_polar_tensor_matches_the_mixed_field_nuclear_derivative() {
    let params = Am1Parameters::standard().unwrap();
    let mol = water();
    let apt = dipole_derivatives(&mol, &params, &tight(0.0, 1, None)).unwrap();

    let h = 1.0e-4 * 27.21; // eV per (e·Bohr)
    let mut worst = 0.0_f64;
    for alpha in 0..3 {
        let unit = match alpha {
            0 => Vec3::new(1.0, 0.0, 0.0),
            1 => Vec3::new(0.0, 1.0, 0.0),
            _ => Vec3::new(0.0, 0.0, 1.0),
        };
        let gp = closed_form_gradient(&mol, &params, &tight(0.0, 1, Some(unit * h))).unwrap();
        let gm = closed_form_gradient(&mol, &params, &tight(0.0, 1, Some(unit * -h))).unwrap();
        for a in 0..mol.atoms.len() {
            for beta in 0..3 {
                let numeric = -(gp.gradient[a].get(beta) - gm.gradient[a].get(beta)) / (2.0 * h);
                worst = worst.max((numeric - apt[(alpha, 3 * a + beta)]).abs());
            }
        }
    }
    eprintln!("    max |APT − (−∂²E/∂F∂R)| = {worst:.3e} e");
    assert!(
        worst < 1.0e-5,
        "the two routes to the APT disagree by {worst:.3e} e"
    );
}

/// Water's spectrum: three vibrations with real intensity, and the rigid-body modes identified as
/// such by their eigenvectors rather than by a frequency cutoff.
#[test]
fn water_has_three_infrared_active_vibrations() {
    let params = Am1Parameters::standard().unwrap();
    let opt = am1_rs::optimize(
        &water(),
        &params,
        &tight(0.0, 1, None),
        &am1_rs::OptOptions::default(),
    )
    .unwrap();
    assert!(opt.converged);

    let spectrum = ir_spectrum(&opt.molecule, &params, &tight(0.0, 1, None)).unwrap();
    let bands = spectrum.vibrational_bands(0.5);
    for (k, freq, intensity) in &bands {
        eprintln!("    mode {k}: {freq:9.2} cm^-1   {intensity:9.3} km/mol");
    }
    assert_eq!(
        bands.len(),
        3,
        "a bent triatomic has 3N−6 = 3 vibrations, got {}",
        bands.len()
    );
    // Water is strongly polar; every one of its three modes is infrared active.
    for (k, _, intensity) in &bands {
        assert!(
            *intensity > 1.0,
            "mode {k} has intensity {intensity:.3} km/mol, which is not an active band"
        );
    }
    // And the rigid-body modes really were the other six.
    assert_eq!(
        spectrum
            .modes
            .translation_rotation_overlap
            .iter()
            .filter(|o| **o > 0.5)
            .count(),
        6
    );
}

/// CO₂'s symmetric stretch is infrared inactive.
///
/// Asserted on the displacement written out by hand — `(0, +1, −1)` along the molecular axis —
/// rather than on whichever mode the diagonalizer happened to put where. The geometry is exactly
/// centrosymmetric by construction, so this is a symmetry statement and not an approximation.
#[test]
fn the_symmetric_stretch_of_carbon_dioxide_is_dark() {
    let params = Am1Parameters::standard().unwrap();
    let mol = co2();
    let apt = dipole_derivatives(&mol, &params, &tight(0.0, 1, None)).unwrap();

    // Symmetric stretch: carbon still, the two oxygens moving outwards together along x.
    let mut q = vec![0.0; 9];
    q[3] = 1.0; // O1, +x
    q[6] = -1.0; // O2, −x
                 // Antisymmetric stretch: the oxygens moving the same way, carbon opposing.
    let mut anti = vec![0.0; 9];
    anti[0] = -2.0;
    anti[3] = 1.0;
    anti[6] = 1.0;

    let project = |v: &[f64]| -> f64 {
        (0..3)
            .map(|alpha| {
                let d: f64 = (0..9).map(|j| apt[(alpha, j)] * v[j]).sum();
                d * d
            })
            .sum::<f64>()
            .sqrt()
    };
    let symmetric = project(&q);
    let antisymmetric = project(&anti);
    eprintln!("    |dmu/dQ| symmetric = {symmetric:.3e} e,  antisymmetric = {antisymmetric:.3e} e");
    assert!(
        symmetric < 1.0e-9,
        "the symmetric stretch carries a dipole derivative of {symmetric:.3e} e; it must vanish \
         by symmetry"
    );
    assert!(
        antisymmetric > 1.0e-2,
        "the antisymmetric stretch should be strongly active, got {antisymmetric:.3e} e"
    );
}

/// A linear molecule has five rigid-body modes, not six. The classification is by eigenvector
/// overlap, so it discovers that rather than assuming `3N − 6`.
#[test]
fn a_linear_molecule_has_five_rigid_body_modes() {
    let params = Am1Parameters::standard().unwrap();
    let spectrum = ir_spectrum(&co2(), &params, &tight(0.0, 1, None)).unwrap();
    let rigid = spectrum
        .modes
        .translation_rotation_overlap
        .iter()
        .filter(|o| **o > 0.5)
        .count();
    eprintln!(
        "    CO2: {rigid} rigid-body modes, {} vibrations",
        spectrum.vibrational_bands(0.5).len()
    );
    assert_eq!(rigid, 5, "a linear triatomic has 3N−5 = 4 vibrations");
    assert_eq!(spectrum.vibrational_bands(0.5).len(), 4);
}
