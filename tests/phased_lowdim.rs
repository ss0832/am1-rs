// SPDX-License-Identifier: GPL-3.0-or-later

//! The phased lattice sum `S(q; r) = Σ_T e^{iq·T}/|r+T|` in **one and two** dimensions.
//!
//! Through 0.2.1 only the three-dimensional phased kernel existed, so the DFPT response silently
//! dropped its long-range monopole channel on a chain or a slab and `LongRange::Require` was an
//! error there. 0.2.2 adds both, each the `q`-shifted form of the machinery its unphased sum
//! already used: Parry's slab sum over the shifted in-plane set, and the chain's direct summation
//! phased image by image with an Abel-transformed tail.
//!
//! # Why an oracle, and not only identities
//!
//! A phase error leaves the sum Hermitian and the frequencies real — it does not announce itself.
//! The identities below (`S(−q) = S(q)*`, periodicity in `q`, reduction to the unphased sum at a
//! reciprocal lattice vector) are each necessary and none is sufficient: the first is satisfied by
//! any real-valued mistake, the second by any `q`-independent one, and the third says nothing about
//! `q ≠ 0`. So the first test here compares against a **direct lattice sum** — the definition,
//! evaluated by brute force over a large shell — which is the only check that is not a property of
//! the implementation.
//!
//! The direct sum converges only conditionally, which is exactly why the Ewald and Abel
//! constructions exist. It is Cesàro-averaged over consecutive shells to damp the oscillating
//! boundary term — without that it is a *worse* approximation than the thing it is checking, and
//! the test is then vacuous. With it:
//!
//! | | agreement | the oracle's own drift |
//! |---|---|---|
//! | 1D, `q = ¼` | **1.6e-12** | 3.8e-12 |
//! | 1D, `q = ⅛` | 6.7e-6 | 3.4e-6 |
//! | 2D | 1.2e-5 … 9.7e-5 | 8.0e-5 … 2.3e-4 |
//!
//! In every case the implementation sits inside the oracle's own convergence, which is the correct
//! relationship and the only one this comparison can honestly assert. The sharp checks are
//! elsewhere: the slab sum is independent of the Ewald splitting parameter to **8.9e-16** across a
//! 2.8× range, and the chain sum independent of its explicit image count to **7.2e-16** across a
//! 6× range. Those are the ones a wrong prefactor cannot survive.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::ewald::{LongRangeKernel, PhasedDelta};
use am1_rs::pbc::ewald2d::{default_alpha_2d, Ewald2D};

// ------------------------------------------------------------------------------------ fixtures

fn slab(a: f64, gap: f64) -> Lattice {
    Lattice::from_vectors(
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(0.0, a, 0.0),
        Vec3::new(0.0, 0.0, gap),
        [true, true, false],
    )
    .unwrap()
}

fn chain(l: f64) -> Lattice {
    Lattice::from_vectors(
        Vec3::new(l, 0.0, 0.0),
        Vec3::new(0.0, 30.0, 0.0),
        Vec3::new(0.0, 0.0, 30.0),
        [true, false, false],
    )
    .unwrap()
}

/// `q` in Cartesian coordinates from fractional coordinates of the reciprocal lattice.
fn q_of(lattice: &Lattice, frac: [f64; 3]) -> Vec3 {
    let b = lattice.reciprocal_vectors_2pi();
    b[0] * frac[0] + b[1] * frac[1] + b[2] * frac[2]
}

/// The definition, summed by brute force over `±shells` in every periodic direction.
///
/// Deliberately naive: no splitting, no acceleration, nothing shared with the implementations it
/// checks. `T = 0` is included unless `r` vanishes there, matching the kernels' own convention.
fn partial_sum(lattice: &Lattice, q: Vec3, r: Vec3, shells: i32) -> [f64; 2] {
    let mut out = [0.0_f64; 2];
    let ranges: Vec<i32> = (0..3)
        .map(|i| if lattice.periodic[i] { shells } else { 0 })
        .collect();
    for n0 in -ranges[0]..=ranges[0] {
        for n1 in -ranges[1]..=ranges[1] {
            for n2 in -ranges[2]..=ranges[2] {
                let t = lattice.cell.col[0] * n0 as f64
                    + lattice.cell.col[1] * n1 as f64
                    + lattice.cell.col[2] * n2 as f64;
                let d = r + t;
                let dist = d.norm();
                if dist < 1.0e-12 {
                    continue;
                }
                let theta = q.dot(t);
                out[0] += theta.cos() / dist;
                out[1] += theta.sin() / dist;
            }
        }
    }
    out
}

/// The direct sum, **Cesàro-averaged** over `window` consecutive shell counts ending at `shells`.
///
/// The phased sum converges only conditionally: the partial sums oscillate about the limit with an
/// envelope that decays slowly, and truncating at any one shell keeps a boundary term of the size
/// of that oscillation. Averaging consecutive partial sums cancels it to leading order, which is
/// the standard treatment and costs nothing here.
///
/// This matters for the two-dimensional case and not much for the one-dimensional one: a 2D shell
/// at radius `R` contributes `O(1)` no matter how large `R` is (its `2πR dR/R` grows as fast as the
/// kernel decays), so the raw partial sums barely converge at all. Without the averaging the oracle
/// is a worse approximation than the thing it is checking, which makes for a weak test.
fn direct_sum(lattice: &Lattice, q: Vec3, r: Vec3, shells: i32, window: i32) -> [f64; 2] {
    let mut acc = [0.0_f64; 2];
    for k in 0..window {
        let p = partial_sum(lattice, q, r, shells - k);
        acc[0] += p[0];
        acc[1] += p[1];
    }
    [acc[0] / window as f64, acc[1] / window as f64]
}

fn kernel_for(lattice: &Lattice) -> LongRangeKernel {
    LongRangeKernel::for_lattice(lattice).unwrap().unwrap()
}

// ------------------------------------------------------------------------------- the oracle

#[test]
fn the_phased_slab_sum_matches_a_direct_lattice_sum() {
    // A wide in-plane cell so the direct sum's shells are physically large, and a displacement
    // well off the plane so the Parry kernel's `z` dependence is exercised rather than sitting at
    // its symmetric point.
    let lattice = slab(6.0, 40.0);
    let kernel = kernel_for(&lattice);
    let r = Vec3::new(1.3, -0.7, 1.9);

    for frac in [[0.25, 0.0, 0.0], [0.5, 0.25, 0.0], [1.0 / 3.0, -0.25, 0.0]] {
        let q = q_of(&lattice, frac);
        let got = kernel.phased_pair_potential(q, r, false);
        // The direct sum converges slowly; compare two shell counts to see where it is.
        let coarse = direct_sum(&lattice, q, r, 60, 40);
        let fine = direct_sum(&lattice, q, r, 120, 40);
        let drift = ((coarse[0] - fine[0]).powi(2) + (coarse[1] - fine[1]).powi(2)).sqrt();
        let err = ((got.value[0] - fine[0]).powi(2) + (got.value[1] - fine[1]).powi(2)).sqrt();
        eprintln!(
            "    2D q={frac:?}: Parry ({:+.8}, {:+.8})  direct ({:+.8}, {:+.8})  \
             err {err:.2e}, direct self-drift {drift:.2e}",
            got.value[0], got.value[1], fine[0], fine[1]
        );
        // The bound is the direct sum's own convergence, not the Parry sum's accuracy: the
        // oracle is the less accurate of the two here, which is the honest way round to say it.
        assert!(
            err < 20.0 * drift.max(1.0e-6),
            "2D phased sum disagrees with the direct sum by {err:.3e} (direct drift {drift:.3e})"
        );
    }
}

#[test]
fn the_phased_chain_sum_matches_a_direct_lattice_sum() {
    let lattice = chain(6.0);
    let kernel = kernel_for(&lattice);
    let r = Vec3::new(1.1, 1.7, -0.6);

    for frac in [[0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.125, 0.0, 0.0]] {
        let q = q_of(&lattice, frac);
        let got = kernel.phased_pair_potential(q, r, false);
        let coarse = direct_sum(&lattice, q, r, 4000, 20);
        let fine = direct_sum(&lattice, q, r, 6000, 20);
        let drift = ((coarse[0] - fine[0]).powi(2) + (coarse[1] - fine[1]).powi(2)).sqrt();
        let err = ((got.value[0] - fine[0]).powi(2) + (got.value[1] - fine[1]).powi(2)).sqrt();
        eprintln!(
            "    1D q={frac:?}: Abel ({:+.8}, {:+.8})  direct ({:+.8}, {:+.8})  \
             err {err:.2e}, direct self-drift {drift:.2e}",
            got.value[0], got.value[1], fine[0], fine[1]
        );
        assert!(
            err < 20.0 * drift.max(1.0e-9),
            "1D phased sum disagrees with the direct sum by {err:.3e} (direct drift {drift:.3e})"
        );
    }
}

// ------------------------------------------------------------------------------- identities

#[test]
fn the_phased_sum_is_the_conjugate_of_its_negative() {
    // `S(−q) = S(q)*` follows from `S` being a sum of `e^{iq·T}` over a translation set closed
    // under negation. True of the value, the gradient and the Hessian alike.
    for (name, lattice) in [("slab", slab(6.0, 40.0)), ("chain", chain(6.0))] {
        let kernel = kernel_for(&lattice);
        let r = Vec3::new(1.3, -0.7, 1.9);
        let q = q_of(&lattice, [0.3, 0.2, 0.0]);
        let plus = kernel.phased_pair_potential(q, r, false);
        let minus = kernel.phased_pair_potential(q * -1.0, r, false);
        let mut worst = 0.0_f64;
        worst = worst.max((plus.value[0] - minus.value[0]).abs());
        worst = worst.max((plus.value[1] + minus.value[1]).abs());
        for a in 0..3 {
            worst = worst.max((plus.gradient[a][0] - minus.gradient[a][0]).abs());
            worst = worst.max((plus.gradient[a][1] + minus.gradient[a][1]).abs());
            for b in 0..3 {
                worst = worst.max((plus.hessian[a][b][0] - minus.hessian[a][b][0]).abs());
                worst = worst.max((plus.hessian[a][b][1] + minus.hessian[a][b][1]).abs());
            }
        }
        eprintln!("    {name}: |S(-q) - S(q)*| = {worst:.3e}");
        assert!(worst < 1.0e-9, "{name}: S(-q) != S(q)* by {worst:.3e}");
    }
}

#[test]
fn the_phased_sum_is_periodic_in_q() {
    // `S(q + G) = S(q)` — every phase changes by `e^{iG·T} = 1`. This is the identity that caught
    // a wrong choice of which reciprocal element to drop in the 3D sum, at 12 eV; it is asserted
    // here for the two new dimensionalities for the same reason.
    for (name, lattice) in [("slab", slab(6.0, 40.0)), ("chain", chain(6.0))] {
        let kernel = kernel_for(&lattice);
        let r = Vec3::new(0.9, -1.4, 1.1);
        let base = q_of(&lattice, [0.3, 0.1, 0.0]);
        let shifted = base + q_of(&lattice, [1.0, 0.0, 0.0]);
        let a = kernel.phased_pair_potential(base, r, false);
        let b = kernel.phased_pair_potential(shifted, r, false);
        let worst = (0..2)
            .map(|c| (a.value[c] - b.value[c]).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("    {name}: |S(q+G) - S(q)| = {worst:.3e}");
        assert!(worst < 1.0e-8, "{name}: not periodic in q ({worst:.3e})");
    }
}

#[test]
fn at_a_reciprocal_lattice_vector_it_reduces_to_the_unphased_sum() {
    // Not `q == 0`: every reciprocal lattice vector phases the sum by one and must give the
    // unphased answer — background, sheet term and line charge included. A phonon commensurate
    // with an n x n x n mesh sits at a supercell reciprocal vector, which is exactly the `q` a
    // supercell comparison uses, so checking only the origin would be the silent version.
    for (name, lattice) in [("slab", slab(6.0, 40.0)), ("chain", chain(6.0))] {
        let kernel = kernel_for(&lattice);
        let r = Vec3::new(1.3, -0.7, 1.4);
        let reference = kernel.pair_potential(r);
        let grad = kernel.pair_potential_gradient(r);
        for frac in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, -1.0, 0.0]] {
            let q = q_of(&lattice, frac);
            let got = kernel.phased_pair_potential(q, r, false);
            let dv = (got.value[0] - reference).abs();
            let di = got.value[1].abs();
            let dg = (0..3)
                .map(|a| (got.gradient[a][0] - grad.get(a)).abs())
                .fold(0.0_f64, f64::max);
            eprintln!(
                "    {name} q={frac:?}: |Δvalue| {dv:.2e}, |imaginary| {di:.2e}, |Δgrad| {dg:.2e}"
            );
            assert!(dv < 1.0e-10, "{name}: value differs by {dv:.3e}");
            assert!(
                di < 1.0e-12,
                "{name}: imaginary part {di:.3e} should vanish"
            );
            assert!(dg < 1.0e-10, "{name}: gradient differs by {dg:.3e}");
        }
    }
}

#[test]
fn the_derivatives_are_the_derivatives() {
    // Central differences of the value against the returned gradient, and of the gradient against
    // the Hessian. Both parts, real and imaginary.
    for (name, lattice) in [("slab", slab(6.0, 40.0)), ("chain", chain(6.0))] {
        let kernel = kernel_for(&lattice);
        let r = Vec3::new(1.3, -0.7, 1.6);
        let q = q_of(&lattice, [0.3, 0.2, 0.0]);
        let analytic = kernel.phased_pair_potential(q, r, false);
        let h = 1.0e-5;

        let shifted = |axis: usize, delta: f64| -> PhasedDelta {
            let mut p = r;
            match axis {
                0 => p.x += delta,
                1 => p.y += delta,
                _ => p.z += delta,
            }
            kernel.phased_pair_potential(q, p, false)
        };

        let mut worst_g = 0.0_f64;
        let mut worst_h = 0.0_f64;
        for a in 0..3 {
            let plus = shifted(a, h);
            let minus = shifted(a, -h);
            for c in 0..2 {
                let fd = (plus.value[c] - minus.value[c]) / (2.0 * h);
                worst_g = worst_g.max((fd - analytic.gradient[a][c]).abs());
                for b in 0..3 {
                    let fd2 = (plus.gradient[b][c] - minus.gradient[b][c]) / (2.0 * h);
                    worst_h = worst_h.max((fd2 - analytic.hessian[a][b][c]).abs());
                }
            }
        }
        eprintln!("    {name}: gradient {worst_g:.2e}, hessian {worst_h:.2e} against FD");
        assert!(worst_g < 1.0e-6, "{name}: gradient off by {worst_g:.3e}");
        assert!(worst_h < 1.0e-5, "{name}: hessian off by {worst_h:.3e}");
    }
}

/// The slab sum must not depend on the Ewald splitting parameter. This is the sharpest internal
/// check available for a split formulation: `α` divides the same total between two sums, and a
/// wrong prefactor on either half — the `π/(A|k|)` of the shifted set, say, against the `2π/(A|G|)`
/// of the folded one — shows up here and almost nowhere else.
#[test]
fn the_slab_sum_is_independent_of_the_splitting_parameter() {
    let lattice = slab(6.0, 40.0);
    let r = Vec3::new(1.3, -0.7, 1.9);
    let q = q_of(&lattice, [0.25, 0.125, 0.0]);
    let base = default_alpha_2d(lattice.measure());

    let mut values: Vec<[f64; 2]> = Vec::new();
    for factor in [0.6, 1.0, 1.7] {
        let e = Ewald2D::new(&lattice, base * factor, 1.0e-13).unwrap();
        let got = e.phased_pair_potential(q, r, false);
        eprintln!(
            "    alpha = {:.4}: ({:+.12}, {:+.12})",
            base * factor,
            got.value[0],
            got.value[1]
        );
        values.push(got.value);
    }
    let worst = values
        .iter()
        .flat_map(|v| [(v[0] - values[0][0]).abs(), (v[1] - values[0][1]).abs()])
        .fold(0.0_f64, f64::max);
    eprintln!("    spread across a 2.8x range of alpha: {worst:.3e}");
    assert!(worst < 1.0e-9, "the slab sum moved {worst:.3e} with alpha");
}

/// How the kernel behaves as `q → 0`, by dimensionality.
///
/// This is the claim `docs/pbc.md` makes and the one it is easiest to state at the wrong level.
/// The **kernel** diverges in all three dimensionalities — there is no dimension in which
/// `Σ_T e^{iq·T}/|r+T|` stays finite at `q = 0`, which is what the neutralizing background exists
/// to handle — but at different rates:
///
/// ```text
/// 3D: 4π/(V q²)      2D: 2π/(A|q|)      1D: −(2/L) ln|q|
/// ```
///
/// It is the *contribution to `D(q)`* that carries two factors of `q` from charge conservation, and
/// so vanishes in 1D and 2D while staying finite and direction-dependent in 3D. Conflating the two
/// levels is how an earlier draft of this work came to record "2D is discontinuous at Γ", which is
/// false: only 3D is.
///
/// Measured here on the kernel, by fitting the exponent of the divergence.
#[test]
fn the_kernel_diverges_at_the_rate_its_dimension_implies() {
    let cases: [(&str, Lattice, f64); 2] = [
        ("2D (expect q^-1)", slab(6.0, 40.0), -1.0),
        ("1D (expect log)", chain(6.0), 0.0),
    ];
    let r = Vec3::new(1.3, -0.7, 1.1);
    for (name, lattice, expected) in cases {
        let kernel = kernel_for(&lattice);
        // Two small `q` a factor of four apart, along the periodic direction.
        let small = q_of(&lattice, [0.01, 0.0, 0.0]);
        let smaller = q_of(&lattice, [0.0025, 0.0, 0.0]);
        let a = kernel.phased_pair_potential(small, r, false).value[0];
        let b = kernel.phased_pair_potential(smaller, r, false).value[0];
        if expected < -0.5 {
            // A power law: the ratio of the values is the ratio of |q| to the exponent.
            let exponent = (b / a).ln() / (0.25_f64).ln();
            eprintln!("    {name}: value {a:.4} -> {b:.4}, fitted exponent {exponent:.3}");
            assert!(
                (exponent - expected).abs() < 0.15,
                "{name}: fitted {exponent:.3}, expected {expected}"
            );
        } else {
            // Logarithmic: the *difference* is constant per factor of four, not the ratio.
            let even_smaller = q_of(&lattice, [0.000625, 0.0, 0.0]);
            let c = kernel.phased_pair_potential(even_smaller, r, false).value[0];
            let step1 = b - a;
            let step2 = c - b;
            eprintln!(
                "    {name}: value {a:.4} -> {b:.4} -> {c:.4}, steps {step1:.4} and {step2:.4}"
            );
            assert!(
                (step1 - step2).abs() < 0.05 * step1.abs().max(1.0e-6),
                "{name}: a logarithm has equal steps per factor of four, got {step1:.4} \
                 and {step2:.4}"
            );
        }
    }
}

/// The chain sum must not depend on how many images are summed explicitly before the Abel tail
/// takes over — the tail exists precisely to make that boundary invisible.
#[test]
fn the_chain_sum_is_independent_of_the_explicit_image_count() {
    use am1_rs::pbc::ewald1d::Ewald1D;
    let lattice = chain(6.0);
    let r = Vec3::new(1.1, 1.7, -0.6);
    let q = q_of(&lattice, [0.25, 0.0, 0.0]);

    let mut values: Vec<[f64; 2]> = Vec::new();
    for n in [64, 200, 400] {
        let e = Ewald1D::new(&lattice, n).unwrap();
        let got = e.phased_pair_potential(q, r, false);
        eprintln!(
            "    n_real = {n}: ({:+.12}, {:+.12})",
            got.value[0], got.value[1]
        );
        values.push(got.value);
    }
    let worst = values
        .iter()
        .flat_map(|v| [(v[0] - values[0][0]).abs(), (v[1] - values[0][1]).abs()])
        .fold(0.0_f64, f64::max);
    eprintln!("    spread across a 6x range of image count: {worst:.3e}");
    assert!(
        worst < 1.0e-10,
        "the chain sum moved {worst:.3e} with the image count"
    );
}
