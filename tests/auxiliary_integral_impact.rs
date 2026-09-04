// SPDX-License-Identifier: GPL-3.0-or-later

//! What the `B_k` auxiliary-integral rewrite is worth, in heats of formation.
//!
//! `src/overlap.rs::bintgs` used to follow MOPAC in switching to a closed-form recurrence above
//! `|x| = 0.5`, where that recurrence is unstable. The argument is `x = r(ζ_a − ζ_b)/2`, so the
//! molecules that sit near the switch are ordinary ones — a C–N bond lands at `x ≈ 0.7`, not in
//! some exotic corner. These five were chosen to span it, with water as a control well past it.
//!
//! **The answer this test records is that nothing moved.** Every heat of formation here is
//! identical before and after the rewrite to four decimal places. The instability grows with the
//! integral's index and `overlap_locals` never reads above `B_6`, so what reached the energy was
//! a `4 × 10⁻¹¹` error, not the `3 × 10⁻³` visible at `B_12`. The rewrite is worth having for the
//! discontinuity it removes — up to `1.7 × 10⁻³` in apparent slope at the switch — and not for
//! any change in energy.
//!
//! Recording that explicitly is the point. A future change to the auxiliary integrals that
//! *does* move one of these numbers has to say so.

use am1_rs::{run_am1, Am1Options, Am1Parameters, Molecule};

/// `(name, xyz, ΔHf in kcal/mol after the rewrite)`.
const CASES: &[(&str, &str, f64)] = &[
    (
        "HCN (C≡N)",
        "3\n\nH 0.0000 0.0 0.0\nC 1.0655 0.0 0.0\nN 2.2261 0.0 0.0\n",
        31.0125,
    ),
    (
        "CH3NH2 (C–N)",
        "7\n\nC -0.7000 0.0000 0.0000\nN 0.7000 0.0000 0.0000\nH -1.0800 1.0180 0.0000\n\
         H -1.0800 -0.5090 0.8815\nH -1.0800 -0.5090 -0.8815\nH 1.0400 -0.4680 0.8300\n\
         H 1.0400 -0.4680 -0.8300\n",
        -2.4802,
    ),
    (
        "CH3OH (C–O)",
        "6\n\nC -0.7000 0.0000 0.0000\nO 0.7000 0.0000 0.0000\nH -1.0800 1.0180 0.0000\n\
         H -1.0800 -0.5090 0.8815\nH -1.0800 -0.5090 -0.8815\nH 0.9800 -0.8900 0.2000\n",
        -54.7464,
    ),
    (
        "CH3Cl (C–Cl)",
        "5\n\nC 0.0000 0.0000 0.0000\nCl 1.7810 0.0000 0.0000\nH -0.3600 1.0270 0.0000\n\
         H -0.3600 -0.5135 0.8894\nH -0.3600 -0.5135 -0.8894\n",
        -17.5923,
    ),
    (
        "H2O (control, x well past the switch)",
        "3\n\nO 0.0000 0.0000 0.0\nH 0.9614 0.0000 0.0\nH -0.2246 0.9348 0.0\n",
        -59.2408,
    ),
];

#[test]
fn heats_of_formation_across_the_auxiliary_integral_switch() {
    let params = Am1Parameters::standard().unwrap();
    eprintln!("    molecule                                 dHf (kcal/mol)   recorded");
    let mut worst = 0.0_f64;
    for (name, xyz, recorded) in CASES {
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let r = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        assert!(r.converged, "{name}: SCF did not converge");
        eprintln!(
            "    {name:40}  {:14.4}   {recorded:8.2}",
            r.heat_of_formation_kcal
        );
        worst = worst.max((r.heat_of_formation_kcal - recorded).abs());
    }
    eprintln!("    worst drift from the recorded values: {worst:.4} kcal/mol");
    assert!(
        worst < 0.01,
        "a heat of formation moved by {worst:.4} kcal/mol from the recorded value; if that was \
         intended, update the table and say what changed"
    );
}
