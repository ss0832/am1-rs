// SPDX-License-Identifier: GPL-3.0-or-later

//! Orientation-dependence regression tests.
//!
//! The diatomic frame used by the two-centre kernels is built by rotating the pair
//! vector onto +x. That construction is singular when the pair vector is anti-parallel
//! to the reference axis, so these tests place bonds *exactly* on the Cartesian axes --
//! the configurations a periodic lattice produces constantly -- and check the analytic
//! derivatives against finite differences there.
//!
//! Physics is rotationally invariant, so every one of these must hold for every
//! orientation. A failure that depends on the orientation is a bug in the frame
//! construction, not in the model.

use am1_rs::{
    analytic_hessian, closed_form_gradient, numerical_gradient, numerical_hessian, Am1Options,
    Am1Parameters, Atom, Molecule, Vec3,
};

const ANG: f64 = 1.0 / 0.529167; // Angstrom -> Bohr, MOPAC7 a0

fn mol(spec: &[(u8, [f64; 3])]) -> Molecule {
    Molecule::new(
        spec.iter()
            .map(|(z, r)| Atom {
                z: *z,
                position: Vec3::new(r[0], r[1], r[2]) * ANG,
            })
            .collect(),
    )
}

/// CO2, linear, laid along the given axis (0 = x, 1 = y, 2 = z).
fn co2_along(axis: usize) -> Molecule {
    let place = |t: f64| {
        let mut r = [0.0; 3];
        r[axis] = t;
        r
    };
    mol(&[(8, place(-1.16)), (6, place(0.0)), (8, place(1.16))])
}

fn max_abs_diff(a: &am1_rs::Matrix, b: &am1_rs::Matrix) -> (f64, usize, usize) {
    let mut best = (0.0, 0, 0);
    for i in 0..a.rows {
        for j in 0..a.cols {
            let d = (a[(i, j)] - b[(i, j)]).abs();
            if d > best.0 {
                best = (d, i, j);
            }
        }
    }
    best
}

fn hessian_mismatch(m: &Molecule) -> f64 {
    let params = Am1Parameters::standard().unwrap();
    let opts = Am1Options::default();
    let ana = analytic_hessian(m, &params, &opts, 1.0e-3).unwrap();
    let num = numerical_hessian(m, &params, &opts, 1.0e-3).unwrap();
    let (d, i, j) = max_abs_diff(&ana, &num);
    eprintln!(
        "    max |analytic - numerical| = {:.3e} eV/Bohr^2 at ({}, {})  [ana {:.6}, num {:.6}]",
        d,
        i,
        j,
        ana[(i, j)],
        num[(i, j)]
    );
    d
}

fn gradient_mismatch(m: &Molecule) -> f64 {
    let params = Am1Parameters::standard().unwrap();
    let opts = Am1Options::default();
    let ana = closed_form_gradient(m, &params, &opts).unwrap();
    let num = numerical_gradient(m, &params, &opts, 1.0e-4).unwrap();
    let mut worst: f64 = 0.0;
    for (a, n) in ana.gradient.iter().zip(num.gradient.iter()) {
        for k in 0..3 {
            worst = worst.max((a.get(k) - n.get(k)).abs());
        }
    }
    eprintln!("    max |analytic - numerical| = {worst:.3e} eV/Bohr");
    worst
}

#[test]
fn co2_hessian_is_orientation_independent() {
    // Tolerance is set by the finite-difference reference's own truncation error,
    // which the existing z-aligned tests already meet at 1e-3.
    const TOL: f64 = 1.0e-3;

    eprintln!("CO2 along z (control -- the orientation existing tests use):");
    let dz = hessian_mismatch(&co2_along(2));
    eprintln!("CO2 along y:");
    let dy = hessian_mismatch(&co2_along(1));
    eprintln!("CO2 along x (the singular orientation):");
    let dx = hessian_mismatch(&co2_along(0));

    assert!(dz < TOL, "z-aligned CO2 Hessian off by {dz:.3e}");
    assert!(dy < TOL, "y-aligned CO2 Hessian off by {dy:.3e}");
    assert!(
        dx < TOL,
        "x-aligned CO2 Hessian off by {dx:.3e} (z-aligned is {dz:.3e}) -- \
         the analytic Hessian depends on how the molecule is oriented in space"
    );
}

#[test]
fn co2_gradient_is_orientation_independent() {
    const TOL: f64 = 5.0e-5;
    eprintln!("CO2 along z:");
    let dz = gradient_mismatch(&co2_along(2));
    eprintln!("CO2 along x:");
    let dx = gradient_mismatch(&co2_along(0));
    assert!(dz < TOL, "z-aligned CO2 gradient off by {dz:.3e}");
    assert!(dx < TOL, "x-aligned CO2 gradient off by {dx:.3e}");
}

#[test]
fn water_hessian_is_orientation_independent() {
    const TOL: f64 = 1.0e-3;
    // One O-H bond placed exactly on +x.
    let on_x = mol(&[
        (8, [0.0, 0.0, 0.0]),
        (1, [0.9584, 0.0, 0.0]),
        (1, [-0.2400, 0.9278, 0.0]),
    ]);
    // The same molecule rotated so that no bond touches an axis.
    let skewed = mol(&[
        (8, [0.0, 0.0, 0.0]),
        (1, [0.5534, 0.5534, 0.5534]),
        (1, [-0.1386, 0.2611, 0.9033]),
    ]);
    eprintln!("H2O skewed (control):");
    let ds = hessian_mismatch(&skewed);
    eprintln!("H2O with an O-H bond on +x:");
    let dx = hessian_mismatch(&on_x);
    assert!(ds < TOL, "skewed water Hessian off by {ds:.3e}");
    assert!(dx < TOL, "x-aligned water Hessian off by {dx:.3e}");
}

/// The 24 proper rotations of the cube: signed permutations of the axes with determinant +1.
/// These are exactly the orientations that put bonds on Cartesian axes, which is where the
/// old quaternion frame lost its derivatives.
fn octahedral_rotations() -> Vec<[[f64; 3]; 3]> {
    let perms = [
        [0usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut out = Vec::with_capacity(24);
    for p in perms {
        for bits in 0..8u8 {
            let s = [
                if bits & 1 != 0 { -1.0 } else { 1.0 },
                if bits & 2 != 0 { -1.0 } else { 1.0 },
                if bits & 4 != 0 { -1.0 } else { 1.0 },
            ];
            let mut m = [[0.0f64; 3]; 3];
            for (row, &col) in p.iter().enumerate() {
                m[row][col] = s[row];
            }
            let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            if det > 0.0 {
                out.push(m);
            }
        }
    }
    out
}

fn rotated(m: &Molecule, rot: &[[f64; 3]; 3]) -> Molecule {
    let atoms = m
        .atoms
        .iter()
        .map(|a| {
            let v = a.position;
            Atom {
                z: a.z,
                position: Vec3::new(
                    rot[0][0] * v.x + rot[0][1] * v.y + rot[0][2] * v.z,
                    rot[1][0] * v.x + rot[1][1] * v.y + rot[1][2] * v.z,
                    rot[2][0] * v.x + rot[2][1] * v.y + rot[2][2] * v.z,
                ),
            }
        })
        .collect();
    Molecule::new(atoms)
}

#[test]
fn harmonic_frequencies_are_rotationally_invariant() {
    // Frequencies are a property of the molecule, not of how it sits in the coordinate
    // system. Any orientation dependence is a bug in the two-centre frame construction.
    // Methane is the sharpest probe here: four C-H bonds, so several land on axes under the
    // octahedral rotations.
    let params = Am1Parameters::standard().unwrap();
    let opts = Am1Options::default();
    let ch4 = mol(&[
        (6, [0.0, 0.0, 0.0]),
        (1, [0.6276, 0.6276, 0.6276]),
        (1, [-0.6276, -0.6276, 0.6276]),
        (1, [-0.6276, 0.6276, -0.6276]),
        (1, [0.6276, -0.6276, -0.6276]),
    ]);

    let reference = am1_rs::vibrational_analysis(&ch4, &params, &opts, 1.0e-3).unwrap();
    let mut worst = 0.0_f64;
    let mut worst_rot = 0usize;

    // The 24 octahedral rotations, plus a spread of generic orientations.
    let mut rotations = octahedral_rotations();
    for i in 1..=20 {
        let t = i as f64 * 0.31415926;
        let (c, s) = (t.cos(), t.sin());
        // A rotation about (1,1,1)/sqrt(3) by angle t (Rodrigues, u = (1,1,1)/sqrt(3)).
        let u = 1.0 / 3.0_f64.sqrt();
        let (ux, uy, uz) = (u, u, u);
        let mc = 1.0 - c;
        rotations.push([
            [
                c + ux * ux * mc,
                ux * uy * mc - uz * s,
                ux * uz * mc + uy * s,
            ],
            [
                uy * ux * mc + uz * s,
                c + uy * uy * mc,
                uy * uz * mc - ux * s,
            ],
            [
                uz * ux * mc - uy * s,
                uz * uy * mc + ux * s,
                c + uz * uz * mc,
            ],
        ]);
    }

    for (idx, rot) in rotations.iter().enumerate() {
        let modes =
            am1_rs::vibrational_analysis(&rotated(&ch4, rot), &params, &opts, 1.0e-3).unwrap();
        for (a, b) in reference
            .frequencies_cm
            .iter()
            .zip(modes.frequencies_cm.iter())
        {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
                worst_rot = idx;
            }
        }
    }
    eprintln!(
        "    max frequency spread over {} orientations = {:.3e} cm^-1 (worst rotation #{})",
        rotations.len(),
        worst,
        worst_rot
    );
    // The floor here is not the frame, and it is not the CPHF either: tightening the CPHF
    // from a truncated 1e-7 residual to a converged 1e-9 left it at 6.0e-5, unchanged. What
    // remains is eigenvector scatter across methane's degenerate E and T2 manifolds plus
    // ordinary linear-algebra roundoff, and the worst case is a generic rotation rather than
    // an axis-aligned one. The frame bug this test was written for moved force constants by
    // ~4e-1 eV/Bohr^2, i.e. hundreds of cm^-1, so 1e-3 cm^-1 still separates the two by five
    // orders of magnitude.
    assert!(
        worst < 1.0e-3,
        "harmonic frequencies vary by {worst:.3e} cm^-1 with orientation"
    );
}
