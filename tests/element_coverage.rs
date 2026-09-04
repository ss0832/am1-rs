// SPDX-License-Identifier: GPL-3.0-or-later

//! Every parameterized element, through the full SCF and its analytic gradient.
//!
//! The MOPAC reference tests (`tests/mopac_reference.rs`) pin the *values* down, but only for
//! the three elements CO₂ contains. This file has no external reference and does not need one:
//! it checks the analytic gradient against a finite difference of the full SCF energy, which is
//! a self-validating claim, and it does so for all 21 elements the parameter set covers.
//!
//! Two things make that worth its runtime:
//!
//! * **The heavy elements take a different code path.** For valence shells with `n ≥ 4` (Zn, Ge,
//!   As, Se, Br, Sb, Te, I, Hg) there is no tabulated closed-form Slater overlap, so the overlap
//!   is a Gauss–Legendre quadrature and the derivative is taken *through* it. Nothing else in
//!   the suite differentiates that path for most of those elements.
//! * **Half of these are open shells.** A hydride of an even-valence element has an odd electron
//!   count, so the doublets here exercise the UHF gradient — element by element — which the
//!   closed-shell tests never touch.
//!
//! Each element is checked in two orientations: a generic skewed one, and exactly along `+x`.
//! The second is the configuration that used to zero the derivatives through the local-frame
//! construction (see `tests/axis_alignment.rs`), and it costs nothing to confirm the fix holds
//! for every element rather than for the two that file happens to use.

use am1_rs::constants::covalent_radius_angstrom;
use am1_rs::{closed_form_gradient, run_am1, Am1Options, Am1Parameters, Atom, Molecule, Vec3};

const ANG: f64 = 1.0 / 0.529167;

/// Every element in the bundled AM1 parameter set.
const ELEMENTS: [u8; 21] = [
    1, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 30, 32, 33, 34, 35, 51, 52, 53, 80,
];

/// A diatomic hydride `Z–H`, oriented along `direction`.
fn hydride(z: u8, direction: Vec3) -> Molecule {
    let bond = (covalent_radius_angstrom(z) + covalent_radius_angstrom(1)) * ANG;
    let d = direction.normalized() * bond;
    Molecule::new(vec![
        Atom {
            z,
            position: Vec3::zero(),
        },
        Atom { z: 1, position: d },
    ])
}

/// Valence electrons of the hydride, to pick the multiplicity.
fn electron_count(z: u8, params: &Am1Parameters) -> f64 {
    params.element(z).unwrap().core_charge + params.element(1).unwrap().core_charge
}

fn options(z: u8, params: &Am1Parameters) -> Am1Options {
    let n = electron_count(z, params).round() as i64;
    Am1Options {
        // An odd electron count cannot be a singlet; those cases run UHF.
        multiplicity: if n % 2 == 0 { 1 } else { 2 },
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 500,
        ..Am1Options::default()
    }
}

fn component(v: &Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Worst |analytic − finite difference| over every Cartesian degree of freedom, eV/Bohr.
fn gradient_error(molecule: &Molecule, params: &Am1Parameters, opts: &Am1Options) -> f64 {
    let analytic = closed_form_gradient(molecule, params, opts).expect("gradient failed");
    // `h = 1e-4`: a central difference of a ~10-100 eV energy has a roundoff floor near
    // `eps * E / h`, so a smaller step would be measuring arithmetic rather than the gradient.
    let h = 1.0e-4;
    let mut worst = 0.0_f64;
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let shift = |m: &Molecule, d: f64| {
                let mut m = m.clone();
                let p = &mut m.atoms[atom].position;
                match axis {
                    0 => p.x += d,
                    1 => p.y += d,
                    _ => p.z += d,
                }
                m
            };
            let plus = run_am1(&shift(molecule, h), params, opts).expect("SCF failed");
            let minus = run_am1(&shift(molecule, -h), params, opts).expect("SCF failed");
            let fd = (plus.total_ev - minus.total_ev) / (2.0 * h);
            worst = worst.max((component(&analytic.gradient[atom], axis) - fd).abs());
        }
    }
    worst
}

#[test]
fn every_parameterized_element_converges_and_its_gradient_matches_finite_differences() {
    let params = Am1Parameters::standard().unwrap();
    let skewed = Vec3::new(0.41, -0.63, 0.66);
    let along_x = Vec3::new(1.0, 0.0, 0.0);

    eprintln!("     Z  sym  mult   shell |  skewed        along +x");
    eprintln!("    {}", "-".repeat(56));

    let mut worst_overall = 0.0_f64;
    let mut worst_element = 0u8;
    for &z in &ELEMENTS {
        let opts = options(z, &params);
        let element = params.element(z).unwrap();
        // The principal quantum number decides whether the overlap is the closed form or the
        // quadrature; `n >= 4` is the numerical path.
        let heavy = matches!(z, 30 | 32 | 33 | 34 | 35 | 51 | 52 | 53 | 80);

        let molecule = hydride(z, skewed);
        let scf = run_am1(&molecule, &params, &opts).unwrap_or_else(|e| {
            panic!("Z = {z}: SCF failed: {e}");
        });
        assert!(scf.converged, "Z = {z}: SCF did not converge");

        let e_skewed = gradient_error(&molecule, &params, &opts);
        let e_axis = gradient_error(&hydride(z, along_x), &params, &opts);

        eprintln!(
            "    {z:3}  {:<3}  {:4}   {:5} |  {e_skewed:.3e}     {e_axis:.3e}",
            am1_rs::z_to_symbol(z).unwrap_or("?"),
            opts.multiplicity,
            if heavy { "n>=4" } else { "n<=3" },
        );
        let _ = element;

        for e in [e_skewed, e_axis] {
            if e > worst_overall {
                worst_overall = e;
                worst_element = z;
            }
        }
    }

    eprintln!(
        "\n    worst over all 21 elements and both orientations: {worst_overall:.3e} eV/Bohr \
         (Z = {worst_element})"
    );
    // Set from the measurement, not from caution. Every element comes in between 1e-9 and 1e-7,
    // the top of that range being the finite difference's own roundoff floor rather than the
    // gradient.
    //
    // Worth noting what this shows about the heavy elements: their overlap is a quadrature
    // accurate only to ~5e-4 in *value*, yet their gradients are as accurate here as carbon's.
    // That is the point of differentiating through the quadrature rather than around it — the
    // result is the exact derivative of the quantity actually being used, so the quadrature error
    // shifts the energy surface without making the gradient inconsistent with it. It is also why
    // molecular dynamics on those elements conserves energy despite the value error.
    assert!(
        worst_overall < 1.0e-6,
        "Z = {worst_element}: gradient off by {worst_overall:.3e} eV/Bohr"
    );
}

#[test]
fn an_element_outside_the_parameter_set_is_refused_by_name() {
    // Silence would mean a wrong answer for a system the model has nothing to say about.
    let params = Am1Parameters::standard().unwrap();
    let neon = Molecule::new(vec![
        Atom {
            z: 10,
            position: Vec3::zero(),
        },
        Atom {
            z: 10,
            position: Vec3::new(3.0, 0.0, 0.0),
        },
    ]);
    let err = run_am1(&neon, &params, &Am1Options::default()).unwrap_err();
    let message = err.to_string();
    eprintln!("    neon: {message}");
    assert!(
        message.contains("10") || message.to_lowercase().contains("ne"),
        "the error should name the element, got: {message}"
    );
}
