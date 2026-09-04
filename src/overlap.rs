// SPDX-License-Identifier: GPL-3.0-or-later

//! Analytic Slater-type diatomic overlap integrals (`diat.f`/`diat2.f`), ported from the
//! reference implementation. Supports valence shells with principal quantum number 1–3
//! (H through Ar), which covers the core organic + main-group AM1 elements. Returns the
//! 4×4 block `S[a_i][b_j]` in the global frame with orbital order `s, px, py, pz`.

use crate::dual::{Dual, Scalar};
use crate::dual2::Dual2;
use crate::error::Result;
use crate::integrals::transverse_projector;
use crate::math::Vec3;
use crate::params::Am1Element;

/// Overlap block between atom `i` (rows) and atom `j` (cols), both in the global frame.
pub fn diatom_overlap(
    ei: &Am1Element,
    pos_i: Vec3,
    ej: &Am1Element,
    pos_j: Vec3,
) -> Result<[[f64; 4]; 4]> {
    let d = pos_j - pos_i;
    let r = d.norm();
    if r < 1.0e-8 {
        let mut m = [[0.0; 4]; 4];
        for k in 0..4 {
            m[k][k] = 1.0;
        }
        return Ok(m);
    }
    let xij = d / r;
    // The kernel requires the first atom to have the higher principal quantum number.
    let (locals, dir, swap) = if ei.n >= ej.n {
        (overlap_locals(ei, ej, r)?, xij, false)
    } else {
        (overlap_locals(ej, ei, r)?, xij * -1.0, true)
    };
    let di = build_di_g::<f64>(locals, [dir.x, dir.y, dir.z]);
    Ok(if swap { transpose4(di) } else { di })
}

/// Dual-valued overlap block, seeded on the displacement `R_j − R_i`, giving the exact
/// derivative `∂S/∂R_j`. Both the radial and angular dependence differentiate in **closed form**
/// for all valence shells — `n ≤ 3` uses the analytic Slater kernel, `n ≥ 4` uses forward-mode
/// AD through the numerical Slater quadrature. No finite differences.
pub fn diatom_overlap_dual(
    ei: &Am1Element,
    pos_i: Vec3,
    ej: &Am1Element,
    pos_j: Vec3,
) -> Result<[[Dual; 4]; 4]> {
    let d = pos_j - pos_i; // i→j displacement; all derivatives are w.r.t. this vector
    let swap = ei.n < ej.n;
    let (ea, eb) = if swap { (ej, ei) } else { (ei, ej) };

    // a→b direction: +dvec/r if not swapped, else −dvec/r (j→i). Seed on the raw displacement.
    let dvec = [Dual::var(d.x, 0), Dual::var(d.y, 1), Dual::var(d.z, 2)];
    let r_dual = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    let inv_r = r_dual.recip();
    let s = if swap { -1.0 } else { 1.0 };
    let dir = [
        dvec[0] * inv_r * s,
        dvec[1] * inv_r * s,
        dvec[2] * inv_r * s,
    ];

    // Radial local scalars, differentiated in closed form for all n.
    let locals = overlap_locals::<Dual>(ea, eb, r_dual)?;

    let di = build_di_g::<Dual>(locals, dir);
    Ok(if swap { transpose4(di) } else { di })
}

/// Second-derivative overlap block: `[[Dual2;4];4]` seeded on the displacement `R_j − R_i`,
/// giving the exact `∂²S/∂R∂R` (value + gradient + 3×3 Hessian per orbital pair). The radial
/// *and* angular dependence are both differentiated in closed form (no finite differences), so
/// this feeds the fully analytic skeleton Hessian — for all valence shells (`n ≤ 3` analytic
/// kernel, `n ≥ 4` via second-order AD through the numerical Slater quadrature).
pub fn diatom_overlap_dual2(
    ei: &Am1Element,
    pos_i: Vec3,
    ej: &Am1Element,
    pos_j: Vec3,
) -> Result<[[Dual2; 4]; 4]> {
    let d = pos_j - pos_i; // i→j displacement; all derivatives are w.r.t. this vector
    let swap = ei.n < ej.n;
    let (ea, eb) = if swap { (ej, ei) } else { (ei, ej) };

    let dvec = [Dual2::var(d.x, 0), Dual2::var(d.y, 1), Dual2::var(d.z, 2)];
    let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    // Radial locals, exact first+second derivatives w.r.t. the displacement (via `r`).
    let locals = overlap_locals::<Dual2>(ea, eb, r)?;
    // a→b unit direction: +dvec/r if not swapped, else −dvec/r (j→i).
    let inv_r = r.recip();
    let sgn = if swap { -1.0 } else { 1.0 };
    let dir = [
        dvec[0] * inv_r * sgn,
        dvec[1] * inv_r * sgn,
        dvec[2] * inv_r * sgn,
    ];

    let di = build_di_g::<Dual2>(locals, dir);
    Ok(if swap { transpose4(di) } else { di })
}

fn transpose4<T: Copy>(m: [[T; 4]; 4]) -> [[T; 4]; 4] {
    let mut out = m;
    for a in 0..4 {
        for b in 0..4 {
            out[a][b] = m[b][a];
        }
    }
    out
}

/// `ea` is the higher-(or equal-)quantum-number atom; `r` is the interatomic distance (Bohr).
/// Generic over the scalar type: instantiated at `f64` for the energy/overlap path, and at
/// [`Dual`]/[`crate::dual2::Dual2`] (seeding `r` on the displacement) for the exact first/second
/// radial derivatives. Products are ordered to keep the scalar on the left of any `f64` factor.
fn overlap_locals<S: Scalar>(ea: &Am1Element, eb: &Am1Element, r: S) -> Result<[S; 5]> {
    let jcall = match (ea.n, eb.n) {
        (1, 1) => 2,
        (2, 1) => 3,
        (2, 2) => 4,
        (3, 1) => 431,
        (3, 2) => 5,
        (3, 3) => 6,
        // n >= 4 (heavy AM1 elements) have no tabulated closed form here; use the general
        // numerical Slater overlap. Sentinel 0 selects it below.
        _ => 0,
    };

    let za = [ea.zeta_s, ea.zeta_p];
    let zb = [eb.zeta_s, eb.zeta_p];

    // A/B auxiliary integrals for each (zeta_a-shell, zeta_b-shell) pair.
    let a111 = aintgs(r * (0.5 * (za[0] + zb[0])));
    let b111 = bintgs(r * (0.5 * (za[0] - zb[0])));
    let a211 = aintgs(r * (0.5 * (za[1] + zb[0])));
    let b211 = bintgs(r * (0.5 * (za[1] - zb[0])));
    let a121 = aintgs(r * (0.5 * (za[0] + zb[1])));
    let b121 = bintgs(r * (0.5 * (za[0] - zb[1])));
    let a22 = aintgs(r * (0.5 * (za[1] + zb[1])));
    let b22 = bintgs(r * (0.5 * (za[1] - zb[1])));

    // `s111` is set by every (reachable) arm; the others default to 0 for arms that only
    // populate the sigma channels.
    let s111;
    let (mut s211, mut s121, mut s221, mut s222) =
        (S::cst(0.0), S::cst(0.0), S::cst(0.0), S::cst(0.0));
    let sq3 = 3.0_f64.sqrt();

    match jcall {
        0 => {
            // Heavy elements (n >= 4): the general numerical Slater overlap, now differentiated
            // analytically through the quadrature (r is carried as the scalar type S).
            let (a, b, c, d, e) = crate::overlap_numeric::slater_locals_numeric::<S>(
                ea.n, ea.zeta_s, ea.zeta_p, eb.n, eb.zeta_s, eb.zeta_p, r,
            );
            s111 = a;
            s211 = b;
            s121 = c;
            s221 = d;
            s222 = e;
        }
        2 => {
            s111 =
                (r * r * (za[0] * zb[0])).powf(1.5) * (a111[2] * b111[0] - b111[2] * a111[0]) / 4.0;
        }
        3 => {
            s111 = r.powi(4)
                * (zb[0].powf(1.5) * za[0].powf(2.5))
                * (a111[3] * b111[0] - b111[3] * a111[0] + a111[2] * b111[1] - b111[2] * a111[1])
                / (sq3 * 8.0);
            s211 = r.powi(4)
                * (zb[0].powf(1.5) * za[1].powf(2.5))
                * (a211[2] * b211[0] - b211[2] * a211[0] + a211[3] * b211[1] - b211[3] * a211[1])
                / 8.0;
        }
        4 => {
            s111 = r.powi(5)
                * (zb[0] * za[0]).powf(2.5)
                * (a111[4] * b111[0] + b111[4] * a111[0] - a111[2] * b111[2] * 2.0)
                / 48.0;
            s211 = r.powi(5)
                * (zb[0] * za[1]).powf(2.5)
                * (a211[3] * (b211[0] - b211[2]) - a211[1] * (b211[2] - b211[4])
                    + b211[3] * (a211[0] - a211[2])
                    - b211[1] * (a211[2] - a211[4]))
                / (16.0 * sq3);
            s121 = r.powi(5)
                * (zb[1] * za[0]).powf(2.5)
                * (a121[3] * (b121[0] - b121[2])
                    - a121[1] * (b121[2] - b121[4])
                    - b121[3] * (a121[0] - a121[2])
                    + b121[1] * (a121[2] - a121[4]))
                / (16.0 * sq3);
            let w = r.powi(5) * (zb[1] * za[1]).powf(2.5) / 16.0;
            s221 = -(w * (b22[2] * (a22[4] + a22[0]) - a22[2] * (b22[4] + b22[0])));
            s222 = w
                * (a22[4] * (b22[0] - b22[2]) - b22[4] * (a22[0] - a22[2]) - a22[2] * b22[0]
                    + b22[2] * a22[0])
                * 0.5;
        }
        431 => {
            s111 = r.powi(5)
                * (zb[0].powf(1.5) * za[0].powf(3.5))
                * (a111[4] * b111[0] + b111[1] * a111[3] * 2.0
                    - a111[1] * b111[3] * 2.0
                    - b111[4] * a111[0])
                / (10.0_f64.sqrt() * 24.0);
            s211 = r.powi(5)
                * (zb[0].powf(1.5) * za[1].powf(3.5))
                * (a211[3] * (b211[0] + b211[2]) - a211[1] * (b211[4] + b211[2])
                    + b211[1] * (a211[2] + a211[4])
                    - b211[3] * (a211[2] + a211[0]))
                / (8.0 * 30.0_f64.sqrt());
        }
        5 => {
            s111 = r.powi(6)
                * (zb[0].powf(2.5) * za[0].powf(3.5))
                * (a111[5] * b111[0] + b111[1] * a111[4]
                    - b111[2] * a111[3] * 2.0
                    - a111[2] * b111[3] * 2.0
                    + b111[4] * a111[1]
                    + b111[5] * a111[0])
                / (30.0_f64.sqrt() * 48.0);
            s211 = r.powi(6)
                * (zb[0].powf(2.5) * za[1].powf(3.5))
                * (a211[4] * b211[0] + b211[1] * a211[5]
                    - b211[3] * a211[3] * 2.0
                    - a211[2] * b211[2] * 2.0
                    + a211[1] * b211[5]
                    + a211[0] * b211[4])
                / (48.0 * 10.0_f64.sqrt());
            s121 = r.powi(6)
                * (zb[1].powf(2.5) * za[0].powf(3.5))
                * ((a121[4] * b121[0] - a121[5] * b121[1])
                    + (a121[3] * b121[1] - a121[4] * b121[2]) * 2.0
                    - (a121[1] * b121[3] - a121[2] * b121[4]) * 2.0
                    - (a121[0] * b121[4] - a121[1] * b121[5]))
                / (48.0 * 10.0_f64.sqrt());
            s221 = r.powi(6)
                * (zb[1].powf(2.5) * za[1].powf(3.5))
                * ((a22[3] * b22[0] - a22[5] * b22[2]) + (a22[2] * b22[1] - a22[4] * b22[3])
                    - (a22[1] * b22[2] - a22[3] * b22[4])
                    - (a22[0] * b22[3] - a22[2] * b22[5]))
                / (16.0 * 30.0_f64.sqrt());
            s222 = r.powi(6)
                * (zb[1].powf(2.5) * za[1].powf(3.5))
                * ((a22[5] - a22[3]) * (b22[0] - b22[2]) + (a22[4] - a22[2]) * (b22[1] - b22[3])
                    - (a22[3] - a22[1]) * (b22[2] - b22[4])
                    - (a22[2] - a22[0]) * (b22[3] - b22[5]))
                / (32.0 * 30.0_f64.sqrt());
        }
        6 => {
            s111 = r.powi(7)
                * (zb[0] * za[0]).powf(3.5)
                * (a111[6] * b111[0] - b111[2] * a111[4] * 3.0 + a111[2] * b111[4] * 3.0
                    - a111[0] * b111[6])
                / 1440.0;
            s211 = r.powi(7)
                * (zb[0] * za[1]).powf(3.5)
                * ((a211[5] * b211[0] + a211[6] * b211[1])
                    + (-a211[4] * b211[1] - a211[5] * b211[2])
                    - (a211[3] * b211[2] + a211[4] * b211[3]) * 2.0
                    - (-a211[2] * b211[3] - a211[3] * b211[4]) * 2.0
                    + (a211[1] * b211[4] + a211[2] * b211[5])
                    + (-a211[0] * b211[5] - a211[1] * b211[6]))
                / (480.0 * sq3);
            s121 = r.powi(7)
                * (zb[1] * za[0]).powf(3.5)
                * ((a121[5] * b121[0] - a121[6] * b121[1])
                    + (a121[4] * b121[1] - a121[5] * b121[2])
                    - (a121[3] * b121[2] - a121[4] * b121[3]) * 2.0
                    - (a121[2] * b121[3] - a121[3] * b121[4]) * 2.0
                    + (a121[1] * b121[4] - a121[2] * b121[5])
                    + (a121[0] * b121[5] - a121[1] * b121[6]))
                / (480.0 * sq3);
            s221 = r.powi(7)
                * (zb[1].powf(3.5) * za[1].powf(3.5))
                * ((a22[4] * b22[0] - a22[6] * b22[2]) - (a22[2] * b22[2] - a22[4] * b22[4]) * 2.0
                    + (a22[0] * b22[4] - a22[2] * b22[6]))
                / 480.0;
            s222 = r.powi(7)
                * (zb[1].powf(3.5) * za[1].powf(3.5))
                * ((a22[6] - a22[4]) * (b22[0] - b22[2])
                    - (a22[4] - a22[2]) * (b22[2] - b22[4]) * 2.0
                    + (a22[2] - a22[0]) * (b22[4] - b22[6]))
                / 960.0;
        }
        _ => unreachable!(),
    }

    Ok([s111, s211, s121, s221, s222])
}

/// Assemble the molecular-frame 4×4 overlap block from the local-frame quantities and the
/// a→b direction, generic over the scalar type (so the direction dependence differentiates
/// exactly). Shared by the analytic (n ≤ 3) and numerical (n ≥ 4) overlap paths.
fn build_di_g<S: Scalar>(s: [S; 5], dir: [S; 3]) -> [[S; 4]; 4] {
    let (s111, s211, s121, s221, s222) = (s[0], s[1], s[2], s[3], s[4]);
    // The p-block is `R^T diag(-s221, s222, s222) R`, whose first axis is `dir`. Because the
    // two transverse eigenvalues are equal, the transverse frame cancels and the block
    // collapses onto the axis and the transverse projector:
    //     di[a][b] = -s221 * n_a n_b + s222 * (delta_ab - n_a n_b).
    // Written this way there is no frame to be singular, so the direction dependence
    // differentiates exactly at every orientation (see `integrals::transverse_projector`).
    let p = transverse_projector(dir);
    let mut di = [[S::cst(0.0); 4]; 4];
    di[0][0] = s111;
    for a in 0..3 {
        di[1 + a][0] = s211 * dir[a];
        di[0][1 + a] = -s121 * dir[a];
    }
    for a in 0..3 {
        for b in 0..3 {
            di[1 + a][1 + b] = -s221 * (dir[a] * dir[b]) + s222 * p[a][b];
        }
    }
    di
}

/// Test-only: overlap block built with the *numeric* local overlaps (forces the general
/// path even for n ≤ 3), used to validate the numeric kernel against the analytic one.
#[cfg(test)]
pub(crate) fn diatom_overlap_forced_numeric(
    ei: &Am1Element,
    pos_i: Vec3,
    ej: &Am1Element,
    pos_j: Vec3,
) -> [[f64; 4]; 4] {
    let d = pos_j - pos_i;
    let r = d.norm();
    let dir = d / r;
    let (ea, eb, direction, swap) = if ei.n >= ej.n {
        (ei, ej, dir, false)
    } else {
        (ej, ei, dir * -1.0, true)
    };
    let (s111, s211, s121, s221, s222) = crate::overlap_numeric::slater_locals_numeric(
        ea.n, ea.zeta_s, ea.zeta_p, eb.n, eb.zeta_s, eb.zeta_p, r,
    );
    let di = build_di_g::<f64>(
        [s111, s211, s121, s221, s222],
        [direction.x, direction.y, direction.z],
    );
    if !swap {
        di
    } else {
        transpose4(di)
    }
}

/// A auxiliary integrals `A_k(x) = ∫₁^∞ t^k e^{-xt} dt`, returned as `[A_0 … A_12]`.
/// Generic over the scalar type so the radial (`x ∝ r`) dependence differentiates exactly.
fn aintgs<S: Scalar>(x: S) -> [S; 13] {
    let mut a = [S::cst(0.0); 13];
    if x.val().abs() < 1.0e-12 {
        return a; // vanishing exponent (e.g. zeta_p of H); those channels are unused
    }
    let inv = x.recip();
    a[0] = (-x).exp() * inv;
    for i in 1..13 {
        a[i] = a[0] + a[i - 1] * inv * (i as f64);
    }
    a
}

/// Argument below which [`bintgs`] uses its Taylor series rather than the closed form.
///
/// Chosen by measurement, not by convention — see [`bintgs`]. Below this the closed form's
/// recurrence has lost accuracy; above it, it is at machine precision and the series would need
/// an ever-growing number of terms. At `x = 5` the two agree to `3e-15`, so the switch is
/// invisible in both value and every derivative.
const BINTGS_SERIES_SWITCH: f64 = 5.0;

/// Terms of the Taylor series [`bintgs`] will use before giving up.
///
/// At the switch point 36 are needed for double precision and the loop exits on its own well
/// before this; the cap only bounds the worst case.
const BINTGS_MAX_TERMS: usize = 64;

/// B auxiliary integrals `B_k(x) = ∫₋₁^¹ t^k e^{-xt} dt`, returned as `[B_0 … B_12]`.
/// Generic over the scalar type so the radial (`x ∝ r`) dependence differentiates exactly.
///
/// # Why this is not MOPAC's `BINTGS`
///
/// MOPAC evaluates these from the closed form
///
/// ```text
/// B_k(x) = (−1)ᵏ eˣ/x − e^{−x}/x + (k/x) B_{k−1}(x)
/// ```
///
/// and falls back to a three-term power series below `|x| = 0.5`. The fallback exists because
/// that **upward** recurrence is unstable: the recurrence term is `O(1/x)` while `B_k` itself is
/// `O(1)`, so each of the twelve steps cancels away roughly `log₁₀(k/x)` digits. Every NDDO code
/// in this lineage carries the same branch.
///
/// The branch does not put the problem far enough away. Measured against the series carried to
/// convergence, the closed form is only accurate to **3.3 × 10⁻³** at `x = 0.5` where MOPAC
/// starts using it, and the three-term series on the other side is truncated at `x⁶`.
///
/// **How much of that reached an answer, precisely.** That 3.3 × 10⁻³ is at `B_12`, and
/// [`overlap_locals`] never reads above `B_6` — the error grows with the index because each
/// recurrence step amplifies it again. At the indices actually consumed the old closed form was
/// accurate to between 10⁻¹⁵ and 4 × 10⁻¹¹, and the heats of formation of HCN, CH₃NH₂, CH₃OH and
/// CH₃Cl are **unchanged to four decimals** by this rewrite (`tests/auxiliary_integral_impact.rs`
/// records that). So this was not a wrong-energy bug, and it is not claimed as one.
///
/// What it *was* is a **discontinuity**. The two branches disagreed where they met, and at the
/// consumed indices the mismatch was up to `3.5 × 10⁻⁷` in value — which a gradient evaluation
/// straddling the switch sees as a slope error of up to **1.7 × 10⁻³**, the same order as the
/// analytic-Hessian tolerance. `x = r(ζ_a − ζ_b)/2` puts a C–N bond at `x ≈ 0.7`, so ordinary
/// molecules sit near that switch rather than far from it.
///
/// # The fix
///
/// The Taylor series is not an approximation to be used only near zero. `B_k` is **entire**, so
///
/// ```text
/// B_k(x) = Σ_n (−x)ⁿ/n! ∫₋₁¹ t^{k+n} dt = 2 Σ_{n : k+n even} (−x)ⁿ / (n! (k+n+1))
/// ```
///
/// converges everywhere — and, because only terms of one parity survive, **every term carries
/// the same sign**. There is no cancellation in it at any argument. The only cost is the number
/// of terms, which grows with `|x|`, and the only reason not to use it everywhere is that cost.
///
/// So the series runs wherever it is cheap and the closed form runs wherever it is accurate, and
/// the two regions overlap. Measured worst-case relative error of the closed form, and terms the
/// series needs, as a function of `x`:
///
/// ```text
///     x     closed form      series terms
///   0.5       3.3e-3              16
///   1.0       1.8e-7              20
///   3.0       4.0e-12             29
///   5.0       3.4e-15             36        <- the switch
///  12.0       7.7e-16             55
/// ```
///
/// At `x = 5` both are exact, so the join is seamless: no jump in value, slope or curvature.
/// `tests::the_b_integrals_are_continuous_and_accurate_across_the_branch_point` asserts that,
/// and measures the whole range at 7 × 10⁻¹⁴ relative against 7.6 × 10⁻³ before.
///
/// The extra terms cost nothing measurable — a 102-atom Hessian takes 0.37 s either way, because
/// the overlap was never the bottleneck.
///
/// This also removes MOPAC's third branch, the `|x| < 10⁻⁶` one that returned the limiting
/// values as **constants**. Under forward-mode AD a constant carries a zero derivative, so that
/// branch reported `dB_k/dx = 0` when the true value is `−2/(k+2)` for odd `k`. It happened to
/// be harmless — `x = r(ζ_a − ζ_b)/2` is that small only when the two exponents are equal, and
/// then `x ≡ 0` for every `r`, so `dB/dr` really is zero — but it was harmless by accident, and
/// the series is simply correct there.
fn bintgs<S: Scalar>(x: S) -> [S; 13] {
    if x.val().abs() > BINTGS_SERIES_SWITCH {
        let inv = x.recip();
        let tx = x.exp() * inv;
        let tmx = (-x).exp() * inv * (-1.0);
        let mut b = [S::cst(0.0); 13];
        b[0] = tx + tmx;
        for i in 1..13 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            b[i] = tx * sign + tmx + b[i - 1] * inv * (i as f64);
        }
        return b;
    }

    let mut b = [S::cst(0.0); 13];
    let mut term = S::cst(1.0); // (−x)ⁿ / n!
    let mut magnitude = 1.0_f64; // |x|ⁿ / n!, tracked in f64 for the stopping test
    let absx = x.val().abs();
    for n in 0..BINTGS_MAX_TERMS {
        // Only `k ≡ n (mod 2)` receives this term — the other parity integrates to zero.
        let mut k = n % 2;
        while k < 13 {
            b[k] = b[k] + term * (2.0 / (k + n + 1) as f64);
            k += 2;
        }
        term = term * x * (-1.0 / (n + 1) as f64);
        magnitude *= absx / (n + 1) as f64;
        // `B_0 ≥ 2` for every argument, so an absolute bound here is a relative one.
        if n >= 4 && magnitude < 1.0e-18 {
            break;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Am1Parameters;

    /// `B_k(x) = ∫₋₁¹ tᵏ e^{−xt} dt` from its Taylor series, carried far past convergence.
    ///
    /// `B_k` is an **entire** function of `x`, so this series converges for every argument; at
    /// `|x| ≤ 1` forty terms is far beyond double precision. That makes it an independent
    /// reference for the production routine rather than a reimplementation of it.
    ///
    /// ```text
    /// B_k(x) = Σ_n (−x)ⁿ/n! ∫₋₁¹ t^{k+n} dt = 2 Σ_{n : k+n even} (−x)ⁿ / (n! (k+n+1))
    /// ```
    /// The series has **no cancellation at any `x`**: for `k` even only even `n` contribute and
    /// `(−x)ⁿ = |x|ⁿ`; for `k` odd only odd `n`, and every term carries the same sign. So the
    /// only thing limiting it is the term count, and running to convergence makes it exact.
    fn bintgs_reference(k: usize, x: f64) -> f64 {
        let mut total = 0.0;
        let mut term = 1.0; // (−x)ⁿ/n!
        for n in 0..400 {
            if (k + n) % 2 == 0 {
                total += 2.0 * term / (k + n + 1) as f64;
            }
            term *= -x / (n + 1) as f64;
            if n > 4 && term.abs() < 1.0e-18 * total.abs() {
                break;
            }
        }
        total
    }

    #[test]
    #[ignore = "diagnostic: prints where each B-integral branch is trustworthy"]
    fn survey_where_the_closed_form_recurrence_is_stable() {
        eprintln!("       x    worst rel. err (closed form)   at k    series terms needed");
        for &x in &[
            0.5_f64, 0.75, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 30.0, 50.0,
        ] {
            // The closed form, isolated from the branch logic.
            let inv = 1.0 / x;
            let tx = x.exp() * inv;
            let tmx = -(-x).exp() * inv;
            let mut b = [0.0_f64; 13];
            b[0] = tx + tmx;
            for i in 1..13 {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                b[i] = tx * sign + tmx + b[i - 1] * inv * (i as f64);
            }
            let mut worst = 0.0_f64;
            let mut at = 0;
            for (k, bk) in b.iter().enumerate() {
                let r = bintgs_reference(k, x);
                let e = (bk - r).abs() / r.abs().max(1.0e-300);
                if e > worst {
                    worst = e;
                    at = k;
                }
            }
            // How many series terms the largest k needs.
            let mut terms = 0;
            {
                let mut total = 0.0;
                let mut term = 1.0_f64;
                for n in 0..400 {
                    if (12 + n) % 2 == 0 {
                        total += 2.0 * term / (12 + n + 1) as f64;
                    }
                    term *= -x / (n + 1) as f64;
                    if n > 4 && term.abs() < 1.0e-18 * total.abs() {
                        terms = n;
                        break;
                    }
                }
            }
            eprintln!("  {x:6.1}    {worst:22.3e}   {at:5}    {terms:12}");
        }

        // The error grows with `k` because each recurrence step amplifies it by `k/x`. Only the
        // low indices are actually consumed by `overlap_locals` (the highest is `B_6`, in the
        // `jcall = 6` arm), so the per-index breakdown is what decides whether any of this
        // reaches an energy.
        eprintln!("\n    relative error of the closed form, by index, just past |x| = 0.5:");
        for &x in &[0.51_f64, 0.75, 1.0] {
            let inv = 1.0 / x;
            let tx = x.exp() * inv;
            let tmx = -(-x).exp() * inv;
            let mut b = [0.0_f64; 13];
            b[0] = tx + tmx;
            for i in 1..13 {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                b[i] = tx * sign + tmx + b[i - 1] * inv * (i as f64);
            }
            let errs: Vec<String> = (0..8)
                .map(|k| {
                    let r = bintgs_reference(k, x);
                    format!("{:.0e}", (b[k] - r).abs() / r.abs().max(1.0e-300))
                })
                .collect();
            eprintln!("      x = {x:.2}:  B_0..B_7 = {}", errs.join("  "));
        }

        // What the *old* MOPAC branch structure did at the indices `overlap_locals` reads.
        // Below `|x| = 0.5` it used a three-term series truncated at `x⁶`; above, the closed
        // form. The jump where they met is the sum of both errors, and it is a genuine
        // discontinuity in `S(r)` and `∂S/∂r`.
        let old_series = |k: usize, x: f64| -> f64 {
            let (x2, x3, x4, x5, x6) = (x * x, x * x * x, x.powi(4), x.powi(5), x.powi(6));
            match k {
                0 => 2.0 + x2 / 3.0 + x4 / 60.0 + x6 / 2520.0,
                1 => -2.0 * x / 3.0 - x3 / 15.0 - x5 / 420.0,
                2 => 2.0 / 3.0 + x2 / 5.0 + x4 / 84.0 + x6 / 3240.0,
                3 => -2.0 * x / 5.0 - x3 / 21.0 - x5 / 540.0,
                4 => 2.0 / 5.0 + x2 / 7.0 + x4 / 108.0 + x6 / 3960.0,
                5 => -2.0 * x / 7.0 - x3 / 27.0 - x5 / 660.0,
                _ => 2.0 / 7.0 + x2 / 9.0 + x4 / 132.0 + x6 / 4680.0,
            }
        };
        let h = 1.0e-4;
        eprintln!("\n    the OLD branch structure, at the indices overlap_locals reads:");
        eprintln!("      k    value jump at |x|=0.5    apparent slope error");
        for k in 0..7 {
            let (xm, xp) = (0.5 - h, 0.5 + h);
            let below = old_series(k, xm);
            let inv = 1.0 / xp;
            let (tx, tmx) = (xp.exp() * inv, -(-xp).exp() * inv);
            let mut b = [0.0_f64; 13];
            b[0] = tx + tmx;
            for i in 1..13 {
                let s = if i % 2 == 0 { 1.0 } else { -1.0 };
                b[i] = tx * s + tmx + b[i - 1] * inv * (i as f64);
            }
            let true_change = bintgs_reference(k, xp) - bintgs_reference(k, xm);
            let jump = ((b[k] - below) - true_change).abs();
            eprintln!("      {k}    {jump:20.3e}    {:19.3e}", jump / (2.0 * h));
        }
    }

    #[test]
    fn the_b_integrals_are_continuous_and_accurate_across_the_branch_point() {
        // `bintgs` is MOPAC's, and MOPAC switches at `|x| = 0.5` between a closed form and a
        // truncated power series. Both the switch and the truncation are visible in the answer,
        // and this test measures them rather than assuming they are negligible.
        //
        // Why the switch exists at all: the closed form runs an **upward** recurrence
        // `B_k = ±eˣ/x − e^{−x}/x + (k/x)·B_{k−1}`, in which the recurrence term is `O(1/x)`
        // while the answer is `O(1)`. As `x → 0` those cancel to arbitrary precision loss, so
        // some series branch is unavoidable. What *is* avoidable is truncating that series so
        // early that it becomes the accuracy bottleneck, and letting the two branches disagree
        // where they meet.
        let step = 1.0e-4;
        let mut worst_value = 0.0_f64;
        let mut worst_slope = 0.0_f64;
        let mut worst_at = (0usize, 0.0_f64);

        // Accuracy of each branch against the reference, over the whole range the code uses.
        for i in 0..=600 {
            let x = -6.0 + 0.02 * i as f64;
            if x.abs() < 1.0e-9 {
                continue;
            }
            let b = bintgs::<f64>(x);
            for (k, bk) in b.iter().enumerate() {
                let reference = bintgs_reference(k, x);
                let err = (bk - reference).abs() / reference.abs().max(1.0e-3);
                if err > worst_value {
                    worst_value = err;
                    worst_at = (k, x);
                }
            }
        }

        // The jump across the branch point, in value and in slope.
        let mut worst_jump = 0.0_f64;
        for &sign in &[1.0_f64, -1.0] {
            let inside = bintgs::<f64>(sign * (BINTGS_SERIES_SWITCH - step));
            let outside = bintgs::<f64>(sign * (BINTGS_SERIES_SWITCH + step));
            let ref_in: Vec<f64> = (0..13)
                .map(|k| bintgs_reference(k, sign * (BINTGS_SERIES_SWITCH - step)))
                .collect();
            let ref_out: Vec<f64> = (0..13)
                .map(|k| bintgs_reference(k, sign * (BINTGS_SERIES_SWITCH + step)))
                .collect();
            for k in 0..13 {
                // The true change across the interval, removed, leaves only the branch mismatch.
                let jump = ((outside[k] - inside[k]) - (ref_out[k] - ref_in[k])).abs();
                worst_jump = worst_jump.max(jump);
                let slope_error = jump / (2.0 * step);
                worst_slope = worst_slope.max(slope_error);
            }
        }

        eprintln!(
            "    worst relative error vs the series reference: {worst_value:.3e}  \
             (B_{} at x = {:.3})",
            worst_at.0, worst_at.1
        );
        eprintln!(
            "    discontinuity at the switch:  value {worst_jump:.3e}, slope {worst_slope:.3e}"
        );

        // Before the rewrite these measured 7.6e-3, 4.5e-4 and 2.3 respectively.
        //
        // What is left is not a branch artifact: it is the rounding the closed form's twelve
        // recurrence steps accumulate above the switch, about 30 ulp, which is what that
        // recurrence costs when it is used only where it is stable. The series side is exact.
        assert!(
            worst_value < 1.0e-12,
            "the B integrals are only accurate to {worst_value:.3e} relative, which then caps \
             every overlap and every overlap derivative built on them"
        );
        assert!(
            worst_jump < 1.0e-12,
            "the two branches disagree by {worst_jump:.3e} where they meet, so the overlap is \
             discontinuous in the geometry at the switch"
        );
        // The slope figure is the value jump divided by the `2·step` probe interval, so it is
        // not an independent measurement — it is what a gradient evaluation would *see* if a
        // pair happened to straddle the switch. That framing is the point: it was **2.3**, an
        // order-one error in `∂S/∂r`, and it is now 3e-9.
        assert!(
            worst_slope < 1.0e-8,
            "a gradient evaluation straddling the switch would see a slope error of \
             {worst_slope:.3e}"
        );
    }

    #[test]
    fn numeric_reproduces_analytic_overlap() {
        // The general numerical overlap must match the MOPAC-validated analytic kernel for
        // n <= 3 pairs, across several orientations, for every AO of the block.
        let p = Am1Parameters::standard().unwrap();
        let a0 = crate::constants::ANGSTROM_TO_BOHR;
        let cases = [(6u8, 1u8), (6, 6), (6, 8), (7, 8), (1, 8), (16, 1), (17, 6)];
        let dirs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, -0.5, 0.8),
            Vec3::new(-0.6, 0.2, -0.1),
        ];
        let mut max_delta = 0.0_f64;
        for (za, zb) in cases {
            let ea = p.element(za).unwrap();
            let eb = p.element(zb).unwrap();
            for d in dirs {
                let r_ang = 1.5;
                let pos_i = Vec3::zero();
                let pos_j = d.normalized() * (r_ang * a0);
                let ana = diatom_overlap(ea, pos_i, eb, pos_j).unwrap();
                let num = diatom_overlap_forced_numeric(ea, pos_i, eb, pos_j);
                for a in 0..4 {
                    for b in 0..4 {
                        max_delta = max_delta.max((ana[a][b] - num[a][b]).abs());
                    }
                }
            }
        }
        eprintln!("numeric-vs-analytic overlap max delta = {max_delta:.2e}");
        assert!(max_delta < 5e-5, "overlap mismatch {max_delta:.3e}");
    }
}
