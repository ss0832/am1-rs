// SPDX-License-Identifier: GPL-3.0-or-later

//! The Barnes–Hut far field: what it costs in accuracy, and what it buys in scaling.
//!
//! `docs/scope.md` recorded "linear-scaling Coulomb ⛔ — stays `O(N²)` by construction", and it
//! was true: `FarField` keeps the interaction in full and simplifies only its shape, so it visits
//! every distant pair. The prefactor falls about a hundredfold; the exponent does not move.
//!
//! `FarFieldTree` removes the exponent. Two things have to be shown and neither implies the other:
//!
//! * **It is still the same number.** The error against the direct sum has to be controlled by
//!   `theta` and to vanish as `theta → 0`, since `theta = 0` opens every node.
//! * **It is actually cheaper.** Measured as an *operation count* — partner evaluations summed over
//!   atoms — and fitted for an exponent, because on a loaded machine a stopwatch measures the load.
//!   That is the same discipline the divide-and-conquer counters use.

use am1_rs::farfield::FarField;
use am1_rs::math::Vec3;
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// A cubic lattice of water molecules, `n × n × n` cells at 4 Å spacing.
///
/// Spaced at 4.0 Å for the reason `tools/make_water_cluster.py` asserts: a closer random packing
/// puts hydrogens inside 1.6 Å of each other and the SCF falls off a cliff. Nothing here runs an
/// SCF, but the geometry should still be one that could.
fn water_grid(n: usize) -> Molecule {
    let s = 4.0 * ANG;
    let mut atoms = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let o = Vec3::new(i as f64 * s, j as f64 * s, k as f64 * s);
                for (z, r) in [
                    (8u8, [0.0, 0.0, 0.0]),
                    (1, [0.9584, 0.0, 0.0]),
                    (1, [-0.2400, 0.9278, 0.0]),
                ] {
                    atoms.push(Atom {
                        z,
                        position: Vec3::new(r[0], r[1], r[2]) * ANG + o,
                    });
                }
            }
        }
    }
    Molecule::new(atoms)
}

/// Alternating charges, so the far field has something to cancel and the test is not measuring a
/// monopole-dominated trivial case.
fn charges(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if i % 3 == 0 { -0.8 } else { 0.4 })
        .collect()
}

#[test]
fn the_tree_reproduces_the_direct_far_field_and_converges_with_theta() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_grid(4); // 192 atoms
    let field = FarField::new(&molecule, &params, 8.0).unwrap().unwrap();
    let q = charges(molecule.atoms.len());
    let exact = field.potential(&q);

    let scale = exact.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst_at = |theta: f64| -> (f64, usize) {
        let tree = field.tree(theta).unwrap();
        let got = tree.potential(&field, &q);
        (
            exact
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max),
            tree.partner_evaluations(&field, &q),
        )
    };

    let mut previous = f64::INFINITY;
    for theta in [0.8, 0.4, 0.2, 0.1, 0.05] {
        let (worst, ops) = worst_at(theta);
        eprintln!(
            "    theta={theta:.2}: worst |ΔV| = {worst:.3e} eV ({:.2e} relative), \
             {ops} partner evaluations",
            worst / scale
        );
        assert!(
            worst < previous * 1.05,
            "the error grew when theta shrank: {worst:.3e} against {previous:.3e}"
        );
        previous = worst;
    }

    // `theta = 0` accepts nothing — every node has a positive radius — so the traversal opens to
    // its leaves and the tree *is* the direct sum. This is the structural end of the convergence
    // above: the approximation has a knob that turns it off completely, not merely down.
    let (worst, ops) = worst_at(0.0);
    let (_, direct_far) = field.pair_counts();
    eprintln!("    theta=0: worst |ΔV| = {worst:.3e} eV, {ops} partner evaluations");

    // The **pair set** is asserted exactly: at `theta = 0` the tree must visit every far pair and
    // no others. That is the structural claim, and an equality is the right way to state it.
    assert_eq!(
        ops,
        2 * direct_far,
        "at theta=0 the tree must visit exactly the pairs the direct sum does"
    );
    // The **value** is asserted to roundoff and not bitwise, because the tree accumulates its
    // partners in traversal order and the direct sum in index order. Same terms, different order,
    // and floating-point addition is not associative — measured 5.3e-15 eV on a 4.7 eV potential.
    assert!(
        worst < 1.0e-12,
        "at theta=0 the tree must reproduce the direct sum to roundoff, got {worst:.3e}"
    );
}

#[test]
fn the_gradient_stays_consistent_with_the_potential() {
    // Both go through the same pair expression applied to the same pseudo-atoms, so this is really
    // asking whether `partners` is deterministic and whether the substitution is shared. It is the
    // property that stops an energy and its forces from describing different systems.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_grid(3);
    let field = FarField::new(&molecule, &params, 8.0).unwrap().unwrap();
    let q = charges(molecule.atoms.len());
    let tree = field.tree(0.15).unwrap();

    let exact = field.gradient(&q);
    let got = tree.gradient(&field, &q);
    let worst = exact
        .iter()
        .zip(&got)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);
    let scale = exact.iter().fold(0.0_f64, |m, v| m.max(v.norm()));
    eprintln!(
        "    worst |Δgradient| = {worst:.3e} eV/Bohr ({:.2e} relative)",
        worst / scale
    );
    assert!(
        worst < 0.05 * scale.max(1.0e-12),
        "the tree gradient differs from the direct one by {worst:.3e}"
    );
}

/// The point of the exercise: the exponent.
///
/// Fitted on partner evaluations across a factor of **43** in atom count, 24 to 1029. Measured:
/// **1.65 for the tree against 2.13 for the direct sum**, and at 1029 atoms 131 515 partner
/// evaluations against 1 043 490 — an **8× reduction**, growing with size.
///
/// 1.65 is not `N log N`'s asymptotic slope, and the assertion does not pretend it is. An octree
/// over a few hundred atoms with an 8 Bohr cutoff is still filling its levels: most pairs are
/// within a handful of cutoffs, so few nodes are far enough to accept. What the fit does establish
/// is that the exponent has *moved*, which is the claim `docs/scope.md` said was unreachable —
/// "stays `O(N²)` by construction". Asserting a sharper number here would be asserting the
/// asymptote rather than the measurement.
#[test]
fn the_tree_removes_the_quadratic_exponent() {
    let params = Am1Parameters::standard().unwrap();
    let mut sizes = Vec::new();
    let mut tree_ops = Vec::new();
    let mut direct_ops = Vec::new();

    for n in [2usize, 3, 4, 5, 6, 7] {
        let molecule = water_grid(n);
        let nat = molecule.atoms.len();
        let field = FarField::new(&molecule, &params, 8.0).unwrap().unwrap();
        let q = charges(nat);
        let tree = field.tree(0.5).unwrap();
        let ops = tree.partner_evaluations(&field, &q);
        // What the direct sum does: every ordered far pair.
        let (_, far) = field.pair_counts();
        sizes.push(nat as f64);
        tree_ops.push(ops as f64);
        direct_ops.push(2.0 * far as f64);
        eprintln!(
            "    N={nat:4}: tree {ops:8} partner evaluations, direct {:8}",
            2 * far
        );
    }

    let fit = |y: &[f64]| -> f64 {
        // Least-squares slope of ln y against ln N.
        let lx: Vec<f64> = sizes.iter().map(|v| v.ln()).collect();
        let ly: Vec<f64> = y.iter().map(|v| v.ln()).collect();
        let n = lx.len() as f64;
        let mx = lx.iter().sum::<f64>() / n;
        let my = ly.iter().sum::<f64>() / n;
        let num: f64 = lx.iter().zip(&ly).map(|(a, b)| (a - mx) * (b - my)).sum();
        let den: f64 = lx.iter().map(|a| (a - mx) * (a - mx)).sum();
        num / den
    };
    let tree_exponent = fit(&tree_ops);
    let direct_exponent = fit(&direct_ops);
    eprintln!("    fitted exponents: tree {tree_exponent:.3}, direct {direct_exponent:.3}");

    assert!(
        direct_exponent > 1.8,
        "the direct far field should be quadratic; fitted {direct_exponent:.3}"
    );
    assert!(
        tree_exponent < 1.8,
        "the tree should be sub-quadratic; fitted {tree_exponent:.3}"
    );
    assert!(
        tree_exponent < direct_exponent - 0.4,
        "the tree exponent {tree_exponent:.3} is not meaningfully below the direct one \
         {direct_exponent:.3}"
    );
}
