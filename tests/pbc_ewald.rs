// SPDX-License-Identifier: GPL-3.0-or-later

//! Ewald summation, validated three independent ways before it is trusted anywhere.
//!
//! 1. **Independence of the splitting parameter.** The division between real and reciprocal
//!    space is arbitrary, so the answer must not depend on where it is made. This catches almost
//!    any error in the construction — a wrong prefactor, a missing self term, a dropped `G = 0`
//!    contribution — because each of those breaks the cancellation between the two halves.
//! 2. **The Madelung constant of rock salt**, 1.747 564 594 6. A number from the literature,
//!    computed here from nothing but the lattice sum, and independent of anything NDDO.
//! 3. **Agreement with a directly summed lattice**, for a *neutral* arrangement where the direct
//!    sum converges on its own. Ewald and brute force share no code.
//!
//! Only after those does the correction get applied to a real calculation.

use am1_rs::lattice::{ImageOffset, Lattice};
use am1_rs::math::Vec3;
use am1_rs::neighbors::NeighborList;
use am1_rs::pbc::ewald::{default_alpha, EwaldSum, LongRangeKernel, LongRangeMonopole};
use am1_rs::{Atom, Molecule};

#[test]
fn the_result_does_not_depend_on_the_splitting_parameter() {
    let lattice = Lattice::cubic(12.0).unwrap();
    let alpha0 = default_alpha(lattice.volume());
    let probes = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.7, 0.0, 0.0),
        Vec3::new(2.3, -3.1, 1.4),
        Vec3::new(6.0, 6.0, 6.0),
    ];

    eprintln!("      alpha/alpha0 |  potential at each probe (Bohr^-1)");
    let mut reference = Vec::new();
    for scale in [1.0_f64, 0.5, 0.75, 1.5, 2.5] {
        let ewald = EwaldSum::new(&lattice, scale * alpha0, 1.0e-14).unwrap();
        let values: Vec<f64> = probes.iter().map(|r| ewald.pair_potential(*r)).collect();
        eprintln!(
            "      {scale:11.2}  |  {}",
            values
                .iter()
                .map(|v| format!("{v:+.10}"))
                .collect::<Vec<_>>()
                .join("  ")
        );
        if reference.is_empty() {
            reference = values;
        } else {
            let worst = values
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                worst < 1.0e-9,
                "alpha scale {scale} moved the potential by {worst:.3e}; the split between real \
                 and reciprocal space is not cancelling"
            );
        }
    }
}

#[test]
fn the_madelung_constant_of_rock_salt_is_reproduced() {
    // The literature value, 1.7475645946, and a check that owes nothing to this project: a
    // conventional NaCl cell of side `a` holds eight ions at the corners of a half-cell lattice,
    // alternating in sign. The Madelung energy per ion pair is `−M e²/r₀` with `r₀ = a/2`.
    let a = 4.0_f64;
    let lattice = Lattice::cubic(a).unwrap();
    let ewald = EwaldSum::new(&lattice, default_alpha(lattice.volume()), 1.0e-15).unwrap();

    // Eight sites of the conventional cell, with alternating charges.
    let mut sites = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let sign = if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 };
                sites.push((Vec3::new(i as f64, j as f64, k as f64) * (a / 2.0), sign));
            }
        }
    }

    // E = ½ Σ_ab q_a q_b φ(R_ab). The cell is neutral, so the background term drops out.
    let mut energy = 0.0;
    for (ra, qa) in &sites {
        for (rb, qb) in &sites {
            energy += qa * qb * ewald.pair_potential(*rb - *ra);
        }
    }
    energy *= 0.5;

    // Per ion pair: four pairs in the conventional cell, nearest-neighbour distance a/2.
    let madelung = -energy / 4.0 * (a / 2.0);
    eprintln!("    Madelung constant of rock salt: {madelung:.10}  (literature 1.7475645946)");
    assert!(
        (madelung - 1.747_564_594_6).abs() < 1.0e-8,
        "got {madelung:.10}, expected 1.7475645946"
    );
}

#[test]
fn a_directly_summed_lattice_differs_by_exactly_the_dipole_surface_term() {
    // Ewald against brute force — and they must **not** agree, by a precisely known amount.
    //
    // A lattice of dipoles has a conditionally convergent sum: the dipole–dipole interaction
    // falls as `R⁻³`, so the answer depends on the shape of the region summed and on what is
    // assumed outside it. Ewald with a neutralizing background is the **tin-foil** (conducting)
    // boundary condition; a spherical direct sum is the vacuum one. The two differ by the
    // classic surface term
    //
    //     E_sphere − E_tinfoil = 2π |p|² / 3V,   p = Σ_a q_a r_a
    //
    // Asserting that identity is far stronger than asserting agreement: agreement to some
    // tolerance could hide a small error, whereas reproducing a specific non-zero difference to
    // six digits cannot. It also pins down which boundary condition this module implements,
    // which matters for anyone comparing against another code.
    let a = 10.0_f64;
    let lattice = Lattice::cubic(a).unwrap();
    let ewald = EwaldSum::new(&lattice, default_alpha(lattice.volume()), 1.0e-14).unwrap();

    // A neutral pair of opposite charges: a dipole, whose lattice sum converges absolutely.
    let sites = [
        (Vec3::new(0.0, 0.0, 0.0), 1.0_f64),
        (Vec3::new(1.6, 0.9, -0.4), -1.0_f64),
    ];

    let mut by_ewald = 0.0;
    for (ra, qa) in &sites {
        for (rb, qb) in &sites {
            by_ewald += qa * qb * ewald.pair_potential(*rb - *ra);
        }
    }
    by_ewald *= 0.5;

    // Direct sum over a large sphere of images. Neutral, so this converges — slowly.
    let mut by_direct = 0.0;
    let reach = 40;
    for i in -reach..=reach {
        for j in -reach..=reach {
            for k in -reach..=reach {
                let offset = ImageOffset { n: [i, j, k] };
                let t = lattice.translation(offset);
                if t.norm() > reach as f64 * a * 0.5 {
                    continue; // spherical truncation, which is what the convergence needs
                }
                for (ra, qa) in &sites {
                    for (rb, qb) in &sites {
                        let d = (*rb + t) - *ra;
                        let dist = d.norm();
                        if dist < 1.0e-10 {
                            continue;
                        }
                        by_direct += 0.5 * qa * qb / dist;
                    }
                }
            }
        }
    }

    // The dipole moment of the cell, and the surface term it implies.
    let p = sites.iter().fold(Vec3::zero(), |acc, (r, q)| acc + *r * *q);
    let surface = 2.0 * std::f64::consts::PI * p.norm2() / (3.0 * lattice.volume());

    let measured = by_direct - by_ewald;
    eprintln!(
        "    dipole lattice: Ewald (tin-foil) {by_ewald:.10}, spherical direct sum \
         {by_direct:.10}\n    difference {measured:.10}, surface term 2π|p|²/3V = {surface:.10}"
    );
    assert!(
        (measured - surface).abs() < 1.0e-6,
        "the difference between the two boundary conditions is {measured:.8}, but the dipole \
         surface term is {surface:.8}"
    );
}

#[test]
fn the_gradient_matches_a_finite_difference() {
    // The gradient is what the periodic forces will need, and it has its own real and reciprocal
    // halves to get wrong.
    let lattice = Lattice::cubic(9.0).unwrap();
    let ewald = EwaldSum::new(&lattice, default_alpha(lattice.volume()), 1.0e-14).unwrap();
    let r = Vec3::new(1.9, -2.4, 0.8);
    let analytic = ewald.pair_potential_gradient(r);

    let h = 1.0e-5;
    let numeric = Vec3::new(
        (ewald.pair_potential(r + Vec3::new(h, 0.0, 0.0))
            - ewald.pair_potential(r - Vec3::new(h, 0.0, 0.0)))
            / (2.0 * h),
        (ewald.pair_potential(r + Vec3::new(0.0, h, 0.0))
            - ewald.pair_potential(r - Vec3::new(0.0, h, 0.0)))
            / (2.0 * h),
        (ewald.pair_potential(r + Vec3::new(0.0, 0.0, h))
            - ewald.pair_potential(r - Vec3::new(0.0, 0.0, h)))
            / (2.0 * h),
    );
    let worst = (analytic - numeric).norm();
    eprintln!(
        "    gradient: analytic ({:+.8}, {:+.8}, {:+.8}), finite difference \
         ({:+.8}, {:+.8}, {:+.8}), |difference| {worst:.3e}",
        analytic.x, analytic.y, analytic.z, numeric.x, numeric.y, numeric.z
    );
    assert!(worst < 1.0e-7, "Ewald gradient off by {worst:.3e}");
}

#[test]
fn the_periodic_stress_does_not_depend_on_the_splitting_parameter() {
    // The single sharpest check on the Ewald strain derivative.
    //
    // `alpha` only decides how the same sum is split between real and reciprocal space, so every
    // strain-dependent piece — the `1/V` in the reciprocal prefactor, the `exp(-G^2/4a^2)`, the
    // `G^2` denominator, and the `-pi/(a^2 V)` background — must cancel in the total. Drop any
    // one of them and the stress acquires an `alpha` dependence immediately, whereas a
    // finite-difference comparison at a single `alpha` would still look fine to several digits.
    //
    // `alpha` is deliberately held fixed under strain in `pair_potential_strain`. This is what
    // makes that legitimate.
    let cell = Lattice::cubic(9.0).unwrap();
    let molecule = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.4, 0.3, 0.2),
        },
        Atom {
            z: 1,
            position: Vec3::new(2.2, 0.5, 0.1),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.3, 2.0, 0.4),
        },
    ])
    .with_cell(cell);
    let neighbors = NeighborList::build(&molecule, 30.0);
    let charges = [-0.4_f64, 0.2, 0.2];

    let mut reference: Option<[[f64; 3]; 3]> = None;
    let base = default_alpha(cell.volume());
    eprintln!("    alpha (1/Bohr)      stress_xx        stress_xy        stress_zz");
    for scale in [0.6_f64, 1.0, 1.7, 2.5] {
        let alpha = base * scale;
        let kernel = LongRangeKernel::Bulk(EwaldSum::new(&cell, alpha, 1.0e-13).unwrap());
        let s = LongRangeMonopole::energy_strain(&molecule, &neighbors, &kernel, &charges).unwrap();
        eprintln!(
            "    {alpha:12.6}  {:15.9}  {:15.9}  {:15.9}",
            s[0][0], s[0][1], s[2][2]
        );
        match &reference {
            None => reference = Some(s),
            Some(r) => {
                let mut worst = 0.0_f64;
                for i in 0..3 {
                    for j in 0..3 {
                        worst = worst.max((s[i][j] - r[i][j]).abs());
                    }
                }
                assert!(
                    worst < 1.0e-7,
                    "the strain derivative moved by {worst:.3e} eV when alpha changed by {scale}x, \
                     so a strain-dependent term is missing"
                );
            }
        }
    }
}
