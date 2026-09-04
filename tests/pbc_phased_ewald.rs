// SPDX-License-Identifier: GPL-3.0-or-later

//! The phased long-range correction `Δ(q; r)`, before anything is built on it.
//!
//! A phase error here would reach the dynamical matrix and leave it Hermitian with real
//! frequencies, so it has to be caught at this level, against things that cannot both be wrong:
//!
//! 1. at `q = 0` it must reproduce the existing unphased `Δ` and its derivatives — different code
//!    paths, same number;
//! 2. its `r`-derivatives must match finite differences of its own value, at `q ≠ 0`;
//! 3. `Δ(−q; r) = conj(Δ(q; r))`, which follows from the summand being real; and
//! 4. it must be periodic in `q` under a reciprocal lattice vector, since `e^{iq·T}` is.

use am1_rs::lattice::{ImageOffset, Lattice};
use am1_rs::math::Vec3;
use am1_rs::pbc::ewald::{
    delta_gradient, delta_hessian, phased_delta, q_cartesian, EwaldSum, LongRangeKernel,
};

const ANG: f64 = 1.0 / 0.529167;

fn cell() -> Lattice {
    Lattice::cubic(5.0 * ANG).unwrap()
}

/// The translation set a pair list would have used.
fn translations(lattice: &Lattice) -> Vec<ImageOffset> {
    lattice.image_offsets(25.0)
}

/// Wrapped in `LongRangeKernel::Bulk` because `phased_delta` dispatches on dimensionality since
/// 0.2.2 — the 2D and 1D phased kernels now exist and go through the same entry point. The sum
/// this builds is the same three-dimensional one it always was.
fn ewald(lattice: &Lattice) -> LongRangeKernel {
    LongRangeKernel::Bulk(
        EwaldSum::new(
            lattice,
            am1_rs::pbc::default_alpha(lattice.volume()),
            1.0e-12,
        )
        .unwrap(),
    )
}

/// Separations chosen to include a short one, a long one, and an off-axis one — a phase bug that
/// cancels along a symmetry direction would survive a single axis-aligned probe.
fn probes() -> Vec<Vec3> {
    vec![
        Vec3::new(1.3, 0.0, 0.0),
        Vec3::new(0.7, -1.9, 2.4),
        Vec3::new(3.1, 2.2, -0.5),
    ]
}

#[test]
fn at_gamma_it_reproduces_the_unphased_correction() {
    // Exactly, value and both derivatives — the phased form keeps the neutralizing background on
    // the same terms the unphased one does, so there is nothing left over to explain away.
    let lattice = cell();
    let trans = translations(&lattice);
    let e = ewald(&lattice);
    let kernel = e.clone();
    let q = Vec3::zero();

    let background = 0.0;

    let mut worst_value = 0.0_f64;
    let mut worst_grad = 0.0_f64;
    let mut worst_hess = 0.0_f64;
    let mut worst_imag = 0.0_f64;
    for r in probes() {
        let p = phased_delta(q, r, &lattice, &trans, &e, false);
        let g = delta_gradient(r, &lattice, &trans, &kernel);
        let h = delta_hessian(r, &lattice, &trans, &kernel);

        // `LongRangeMonopole::new` builds the value as `AM1_EV*(pair_potential − truncated)`;
        // reconstruct it here rather than building a whole molecule for one number.
        let mut truncated = 0.0;
        for offset in &trans {
            let d = r + lattice.translation(*offset);
            if d.norm() > 1.0e-10 {
                truncated += 1.0 / d.norm();
            }
        }
        let unphased = am1_rs::constants::AM1_EV * (e.pair_potential(r) - truncated);

        worst_value = worst_value.max((p.value[0] - (unphased + background)).abs());
        worst_imag = worst_imag.max(p.value[1].abs());
        for (i, hi) in h.iter().enumerate() {
            worst_grad = worst_grad.max((p.gradient[i][0] - g.get(i)).abs());
            worst_imag = worst_imag.max(p.gradient[i][1].abs());
            for (j, hij) in hi.iter().enumerate() {
                worst_hess = worst_hess.max((p.hessian[i][j][0] - hij).abs());
                worst_imag = worst_imag.max(p.hessian[i][j][1].abs());
            }
        }
    }
    eprintln!(
        "    q = 0: |Δ − (Δ_unphased + background)| = {worst_value:.3e} eV, \
         |∇Δ − ∇Δ_unphased| = {worst_grad:.3e}, |∇∇Δ − ∇∇Δ_unphased| = {worst_hess:.3e}"
    );
    eprintln!("    largest imaginary part at q = 0: {worst_imag:.3e}");
    assert!(worst_value < 1.0e-9, "value off by {worst_value:.3e}");
    assert!(worst_grad < 1.0e-9, "gradient off by {worst_grad:.3e}");
    assert!(worst_hess < 1.0e-9, "hessian off by {worst_hess:.3e}");
    assert!(
        worst_imag < 1.0e-12,
        "the phased sum must be real at q = 0, largest imaginary part {worst_imag:.3e}"
    );
}

#[test]
fn the_derivatives_match_finite_differences_at_finite_q() {
    // The test that actually exercises the phases: a wrong `e^{iq·T}` anywhere changes the value
    // and its derivatives consistently *within* the analytic path, so only differencing the value
    // catches it.
    let lattice = cell();
    let trans = translations(&lattice);
    let e = ewald(&lattice);

    for frac in [[0.25, 0.0, 0.0], [0.3, -0.2, 0.15], [0.5, 0.5, 0.0]] {
        let q = q_cartesian(&lattice, frac);
        let mut worst_g = 0.0_f64;
        let mut worst_h = 0.0_f64;
        let mut scale = 0.0_f64;
        for r in probes() {
            let analytic = phased_delta(q, r, &lattice, &trans, &e, false);
            let step = 1.0e-5;
            for i in 0..3 {
                let mut plus = r;
                let mut minus = r;
                match i {
                    0 => {
                        plus.x += step;
                        minus.x -= step;
                    }
                    1 => {
                        plus.y += step;
                        minus.y -= step;
                    }
                    _ => {
                        plus.z += step;
                        minus.z -= step;
                    }
                }
                let p = phased_delta(q, plus, &lattice, &trans, &e, false);
                let m = phased_delta(q, minus, &lattice, &trans, &e, false);
                for part in 0..2 {
                    let fd = (p.value[part] - m.value[part]) / (2.0 * step);
                    worst_g = worst_g.max((analytic.gradient[i][part] - fd).abs());
                    scale = scale.max(analytic.gradient[i][part].abs());
                    for j in 0..3 {
                        let fdh = (p.gradient[j][part] - m.gradient[j][part]) / (2.0 * step);
                        worst_h = worst_h.max((analytic.hessian[i][j][part] - fdh).abs());
                    }
                }
            }
        }
        eprintln!(
            "    q = {frac:?}: max |∇Δ − FD| = {worst_g:.3e}, max |∇∇Δ − FD(∇Δ)| = {worst_h:.3e} \
             (gradients up to {scale:.3e})"
        );
        assert!(
            worst_g < 1.0e-6,
            "gradient off by {worst_g:.3e} at q = {frac:?}"
        );
        assert!(
            worst_h < 1.0e-5,
            "hessian off by {worst_h:.3e} at q = {frac:?}"
        );
    }
}

#[test]
fn the_correction_is_conjugate_symmetric_in_q() {
    // `Δ(−q) = conj(Δ(q))`, because the summand is real. Holds whatever the physics is, so a
    // violation is a defect in the construction rather than a property of the system.
    let lattice = cell();
    let trans = translations(&lattice);
    let e = ewald(&lattice);

    let mut worst = 0.0_f64;
    for frac in [[0.25, 0.0, 0.0], [0.1, 0.4, -0.3]] {
        let qp = q_cartesian(&lattice, frac);
        let qm = q_cartesian(&lattice, [-frac[0], -frac[1], -frac[2]]);
        for r in probes() {
            let a = phased_delta(qp, r, &lattice, &trans, &e, false);
            let b = phased_delta(qm, r, &lattice, &trans, &e, false);
            worst = worst.max((a.value[0] - b.value[0]).abs());
            worst = worst.max((a.value[1] + b.value[1]).abs());
            for i in 0..3 {
                worst = worst.max((a.gradient[i][0] - b.gradient[i][0]).abs());
                worst = worst.max((a.gradient[i][1] + b.gradient[i][1]).abs());
                for j in 0..3 {
                    worst = worst.max((a.hessian[i][j][0] - b.hessian[i][j][0]).abs());
                    worst = worst.max((a.hessian[i][j][1] + b.hessian[i][j][1]).abs());
                }
            }
        }
    }
    eprintln!("    max |Δ(−q) − conj(Δ(q))| = {worst:.3e}");
    assert!(worst < 1.0e-9, "conjugate symmetry violated by {worst:.3e}");
}

#[test]
fn it_is_periodic_in_q_under_a_reciprocal_lattice_vector() {
    // `e^{iq·T}` is unchanged by `q → q + G`, so `Δ` must be too. This is what pins down that the
    // reciprocal sum really is over the shifted lattice `{G − q}` and not something that only
    // looks right at small `q`.
    let lattice = cell();
    let trans = translations(&lattice);
    let e = ewald(&lattice);

    let mut worst = 0.0_f64;
    for frac in [[0.25, 0.0, 0.0], [0.1, 0.4, -0.3]] {
        let a = phased_delta(
            q_cartesian(&lattice, frac),
            probes()[1],
            &lattice,
            &trans,
            &e,
            false,
        );
        let shifted = [frac[0] + 1.0, frac[1] - 1.0, frac[2] + 2.0];
        let b = phased_delta(
            q_cartesian(&lattice, shifted),
            probes()[1],
            &lattice,
            &trans,
            &e,
            false,
        );
        worst = worst.max((a.value[0] - b.value[0]).abs());
        worst = worst.max((a.value[1] - b.value[1]).abs());
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max((a.hessian[i][j][0] - b.hessian[i][j][0]).abs());
                worst = worst.max((a.hessian[i][j][1] - b.hessian[i][j][1]).abs());
            }
        }
    }
    eprintln!("    max |Δ(q + G) − Δ(q)| = {worst:.3e}");
    assert!(worst < 1.0e-8, "not periodic in q: {worst:.3e}");
}

/// **What the correction is for**, checked in isolation: the truncated sum plus the correction is
/// the exact sum, so the total must not depend on where the truncation was put.
///
/// `Δ` is defined as *the exact phased lattice sum minus what the pair list already counted*, so
/// adding back that counted part has to reproduce the exact sum whatever the cutoff was. At
/// `q = 0` the raw sum converges quickly on its own and this says little; at finite `q` the raw
/// sum converges slowly, and the whole value of the correction is that the total does not.
///
/// This is the sharpest available test of the phases at finite `q`. A supercell cannot provide
/// one — a truncated `Φ(T)` structurally cannot carry the long-range tail, which is why the
/// correction exists — and the `q = 0` identity only exercises `e^{iq·T} = 1`.
#[test]
fn the_truncated_sum_plus_the_correction_is_independent_of_the_cutoff() {
    let lattice = cell();
    let e = ewald(&lattice);

    for frac in [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.3, -0.2, 0.15]] {
        let q = q_cartesian(&lattice, frac);
        let mut totals: Vec<[[f64; 2]; 3]> = Vec::new();
        let mut raw: Vec<[[f64; 2]; 3]> = Vec::new();
        for cutoff in [15.0_f64, 22.0, 30.0] {
            let trans = lattice.image_offsets(cutoff);
            let corr = phased_delta(q, probes()[1], &lattice, &trans, &e, false);

            // The part the pair list would have summed, with the same phases: the second
            // derivative of `Σ_T e^{iq·T}/|d+T|` over the retained translations.
            let mut counted = [[0.0_f64; 2]; 3];
            for offset in &trans {
                let t = lattice.translation(*offset);
                let d = probes()[1] + t;
                let dist = d.norm();
                if dist < 1.0e-10 {
                    continue;
                }
                let theta = q.dot(t);
                let (c, s) = (theta.cos(), theta.sin());
                let u = [d.x / dist, d.y / dist, d.z / dist];
                let inv3 = 1.0 / (dist * dist * dist);
                for i in 0..3 {
                    // The `xx`, `yy`, `zz` diagonal is enough to catch a phase error.
                    let hij = (3.0 * u[i] * u[i] - 1.0) * inv3 * am1_rs::constants::AM1_EV;
                    counted[i][0] += c * hij;
                    counted[i][1] += s * hij;
                }
            }
            let mut total = [[0.0_f64; 2]; 3];
            for i in 0..3 {
                for part in 0..2 {
                    total[i][part] = corr.hessian[i][i][part] + counted[i][part];
                }
            }
            totals.push(total);
            raw.push(counted);
        }

        let spread = |v: &[[[f64; 2]; 3]]| {
            let mut worst = 0.0_f64;
            for a in v {
                for b in v {
                    for i in 0..3 {
                        for part in 0..2 {
                            worst = worst.max((a[i][part] - b[i][part]).abs());
                        }
                    }
                }
            }
            worst
        };
        let with_correction = spread(&totals);
        let without = spread(&raw);
        eprintln!(
            "    q = {frac:?}: cutoff 15/22/30 Bohr — truncated sum alone varies by \
             {without:.3e}, plus the correction {with_correction:.3e} eV/Bohr²"
        );
        assert!(
            with_correction < 1.0e-9,
            "the correction should make the total cutoff-independent at q = {frac:?}; it still \
             varies by {with_correction:.3e}"
        );
    }
}

/// The self case: an atom with its own images. The `T = 0` term is excluded and the Ewald
/// self-energy applied, exactly as the unphased path does.
#[test]
fn the_self_interaction_matches_the_unphased_path_at_gamma() {
    let lattice = cell();
    let trans = translations(&lattice);
    let e = ewald(&lattice);

    let p = phased_delta(Vec3::zero(), Vec3::zero(), &lattice, &trans, &e, true);
    let mut truncated = 0.0;
    for offset in &trans {
        let d = lattice.translation(*offset);
        if d.norm() > 1.0e-10 {
            truncated += 1.0 / d.norm();
        }
    }
    let unphased = am1_rs::constants::AM1_EV * (e.pair_potential(Vec3::zero()) - truncated);
    let diff = (p.value[0] - unphased).abs();
    eprintln!(
        "    self term: phased {:.8} vs unphased {:.8} eV",
        p.value[0], unphased
    );
    assert!(diff < 1.0e-9, "self interaction off by {diff:.3e}");
    assert!(p.value[1].abs() < 1.0e-12);
}
