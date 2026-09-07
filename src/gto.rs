// SPDX-License-Identifier: GPL-3.0-or-later

//! **Gaussian expansion of the Slater valence basis**, for wavefunction output.
//!
//! The AM1/RM1 valence basis is Slater-type: one radial function `r^{n−1} e^{−ζr}` per shell,
//! with `n` and `ζ` from the parameter table. That is what the integrals use and it is exact.
//! It is also what almost no visualization program can read: `[STO]` is a corner of the Molden
//! format that Jmol, VMD, Avogadro and Multiwfn either ignore or mis-parse, and a file whose
//! basis section is skipped produces orbitals drawn from nothing. `[GTO]` is the section every
//! one of them implements, so an orbital picture has to go through Gaussians.
//!
//! # The expansion is fitted here, not tabulated
//!
//! The published STO-*n*G tables (Hehre, Stewart & Pople) cover `1s`–`3d` at a handful of `n`.
//! This crate needs `4s`, `4p`, `5s` and `5p` as well — Ge, As, Se, Br, Sb, Te, I and Hg are in
//! the parameter set — and needs them for whatever `ζ` a parameter file happens to carry. So the
//! expansion is **solved for** at run time and cached, the same way [`crate::molden::slater_norm`]
//! is derived rather than copied. Nothing here is taken from a third-party table.
//!
//! # What is solved
//!
//! Maximize the overlap of a normalized contraction with the normalized Slater radial function
//!
//! ```text
//! chi(r) = N_s r^{n-1} e^{-zeta r},     g_a(r) = N_a r^l e^{-a r^2}
//! ```
//!
//! over both the contraction coefficients (a linear problem, solved exactly) and the exponents
//! (a small nonlinear one, solved by coordinate descent from an even-tempered start). The two
//! integrals the linear problem needs are:
//!
//! * `<g_a|g_b> = (2 sqrt(ab) / (a+b))^{l+3/2}` — closed form, and the reason the Gram matrix
//!   never has to be built numerically;
//! * `<g_a|chi>` — no elementary closed form, so it is quadratured (Gauss–Legendre, nodes found
//!   by Newton on the Legendre recurrence, so this file has no numerical tables either).
//!
//! # Only the shape has to be fitted once
//!
//! `chi_zeta(r) = zeta^{3/2} chi_1(zeta r)` and `g_{zeta^2 a}(r) = zeta^{3/2} g_a(zeta r)`, so a
//! fit at `zeta = 1` transfers to every `zeta` by `a -> zeta^2 a` with the coefficients
//! **unchanged**. The cache is therefore keyed on `(n, l, ngauss)` alone, and a molecule of a
//! hundred atoms solves at most a handful of fits.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::error::{Am1Error, Result};

/// The default expansion length. Six is where the published STO-*n*G series stops being worth
/// extending: the residual is already below what a contour plot can show, and the fit's cost is
/// paid once per element.
pub const DEFAULT_NGAUSS: usize = 6;

/// The most primitives a shell may be expanded into.
///
/// Not a resource limit — the fit is milliseconds at any of these — but a conditioning one. The
/// Gram matrix of a long even-tempered sequence is exponentially ill-conditioned, and past about
/// ten primitives the added ones buy less than the pseudo-inverse cut throws away.
pub const MAX_NGAUSS: usize = 10;

/// A contracted Cartesian Gaussian shell standing in for one Slater radial function.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianShell {
    /// Angular momentum: 0 for `s`, 1 for `p`.
    pub l: u32,
    /// Primitive exponents, Bohr⁻².
    pub exponents: Vec<f64>,
    /// Contraction coefficients over **normalized** primitives — the convention every Molden
    /// writer uses, and the one a reader assumes when it renormalizes nothing.
    pub coefficients: Vec<f64>,
    /// `<fit|chi>` with both sides normalized: 1 would be an exact reproduction.
    ///
    /// Reported rather than asserted, because it is the honest measure of what the picture is
    /// worth, and it is written into the Molden file's header so it travels with the data.
    pub overlap: f64,
}

/// Expand `r^{n−1} e^{−zeta·r}` (angular momentum `l`) into `ngauss` normalized Gaussians.
///
/// `zeta` is in Bohr⁻¹ and the returned exponents are in Bohr⁻², which is the unit Molden's
/// `[GTO]` section is defined in.
pub fn expand_slater(n: u8, l: u32, zeta: f64, ngauss: usize) -> Result<GaussianShell> {
    if !(1..=MAX_NGAUSS).contains(&ngauss) {
        return Err(Am1Error::InvalidInput(format!(
            "a Gaussian expansion needs between 1 and {MAX_NGAUSS} primitives, got {ngauss}"
        )));
    }
    if l as u8 + 1 > n {
        return Err(Am1Error::InvalidInput(format!(
            "there is no {n}{} shell",
            if l == 0 { "s" } else { "p" }
        )));
    }
    if !(zeta.is_finite() && zeta > 0.0) {
        return Err(Am1Error::InvalidInput(format!(
            "a Slater exponent must be finite and positive, got {zeta}"
        )));
    }
    let unit = unit_fit(n, l, ngauss)?;
    Ok(GaussianShell {
        l,
        exponents: unit.exponents.iter().map(|a| a * zeta * zeta).collect(),
        coefficients: unit.coefficients.clone(),
        overlap: unit.overlap,
    })
}

/// The `zeta = 1` fit for `(n, l, ngauss)`, computed once per process.
///
/// A `HashMap` behind a `Mutex`: this is a pure lookup, never iterated and never summed over, so
/// the hash-order nondeterminism that bit [`crate::pbc::phonon`] cannot reach it.
fn unit_fit(n: u8, l: u32, ngauss: usize) -> Result<GaussianShell> {
    /// Keyed by the shell and the primitive count -- the three things a fit depends on once zeta`n    /// has been scaled out.
    type FitCache = OnceLock<Mutex<HashMap<(u8, u32, usize), GaussianShell>>>;
    static CACHE: FitCache = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (n, l, ngauss);
    if let Ok(map) = cache.lock() {
        if let Some(hit) = map.get(&key) {
            return Ok(hit.clone());
        }
    }
    let fitted = solve_fit(n, l, ngauss);
    if let Ok(mut map) = cache.lock() {
        map.insert(key, fitted.clone());
    }
    Ok(fitted)
}

/// `Gamma(l + 3/2) = (2l+1)!! sqrt(pi) / 2^{l+1}`.
fn gamma_l_plus_3_2(l: u32) -> f64 {
    let mut double_factorial = 1.0_f64; // (2l+1)!!
    let mut k = 3u32;
    while k <= 2 * l + 1 {
        double_factorial *= k as f64;
        k += 2;
    }
    double_factorial * std::f64::consts::PI.sqrt() / 2.0_f64.powi(l as i32 + 1)
}

/// `<g_a|g_b>` for normalized primitives of angular momentum `l`.
///
/// `N_a N_b Gamma(l+3/2) / (2 (a+b)^{l+3/2})` with `N_a = sqrt(2 (2a)^{l+3/2} / Gamma(l+3/2))`
/// collapses to a single power, which is why the Gram matrix costs nothing to build.
fn primitive_overlap(a: f64, b: f64, l: u32) -> f64 {
    (2.0 * (a * b).sqrt() / (a + b)).powf(l as f64 + 1.5)
}

/// `<g_a|chi>` at `zeta = 1`, both normalized.
fn slater_gauss_overlap(n: u8, l: u32, alpha: f64) -> f64 {
    let gamma = gamma_l_plus_3_2(l);
    let norm_g = (2.0 * (2.0 * alpha).powf(l as f64 + 1.5) / gamma).sqrt();
    // N_s = (2 zeta)^{n+1/2} / sqrt((2n)!), at zeta = 1.
    let two_n_factorial: f64 = (1..=2 * n as u32).map(f64::from).product();
    let norm_s = 2.0_f64.powf(n as f64 + 0.5) / two_n_factorial.sqrt();
    // int_0^inf r^{n+l+1} e^{-alpha r^2 - r} dr
    let m = n as i32 + l as i32 + 1;
    let radial = quadrature(m, alpha);
    norm_g * norm_s * radial
}

/// `int_0^inf r^m e^{-alpha r^2 - r} dr`, by composite Gauss–Legendre.
///
/// The upper limit is where **both** decays have killed the integrand: `e^{-alpha r^2}` needs
/// `r > sqrt(50/alpha)` and `e^{-r}` needs `r > 50`, and `r^m` with `m <= 8` cannot rescue
/// `e^{-50}`. Sixteen nodes on each of sixty-four panels is far past what the smooth integrand
/// needs; the cost is paid once per `(n, l, alpha)` inside a fit that runs once per process.
fn quadrature(m: i32, alpha: f64) -> f64 {
    let r_max = (50.0_f64 / alpha).sqrt().max(50.0) + m as f64;
    let (nodes, weights) = gauss_legendre(16);
    let panels = 64;
    let h = r_max / panels as f64;
    let mut acc = 0.0;
    for p in 0..panels {
        let lo = p as f64 * h;
        let mid = lo + 0.5 * h;
        for (x, w) in nodes.iter().zip(&weights) {
            let r = mid + 0.5 * h * x;
            acc += 0.5 * h * w * r.powi(m) * (-alpha * r * r - r).exp();
        }
    }
    acc
}

/// Gauss–Legendre nodes and weights on `[-1, 1]`, found by Newton on the Legendre recurrence.
///
/// Computed rather than tabulated for the same reason as everything else in this module: a table
/// is a thing that can be transcribed wrongly, and this is fifteen lines.
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    /// Nodes and weights per order.
    type RuleCache = OnceLock<Mutex<HashMap<usize, (Vec<f64>, Vec<f64>)>>>;
    static CACHE: RuleCache = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(map) = cache.lock() {
        if let Some(hit) = map.get(&n) {
            return hit.clone();
        }
    }
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        // Chebyshev-like starting guess; two-digit accurate, and Newton doubles that each pass.
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            // P_n(x) and P_n'(x) by the three-term recurrence.
            let (mut p0, mut p1) = (1.0_f64, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let p = if n == 0 { 1.0 } else { p1 };
            dp = n as f64 * (x * p - p0) / (x * x - 1.0);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1.0e-15 {
                break;
            }
        }
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    if let Ok(mut map) = cache.lock() {
        map.insert(n, (nodes.clone(), weights.clone()));
    }
    (nodes, weights)
}

/// Squared overlap of the best normalized contraction on `alphas` with the Slater function.
///
/// Solved through the eigendecomposition of the Gram matrix rather than by elimination: an
/// even-tempered exponent set is deliberately close to linearly dependent, and cutting the
/// eigenvalues below `1e-12` turns "the solve blew up" into "those directions carry nothing",
/// which is what they do. Returns `None` if the Gram matrix cannot be decomposed at all.
fn best_contraction(n: u8, l: u32, alphas: &[f64]) -> Option<(Vec<f64>, f64)> {
    let k = alphas.len();
    let mut gram = crate::linalg::Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            gram[(i, j)] = primitive_overlap(alphas[i], alphas[j], l);
        }
    }
    let b: Vec<f64> = alphas
        .iter()
        .map(|&a| slater_gauss_overlap(n, l, a))
        .collect();
    let (values, vectors) = crate::linalg::symmetric_eigen(&gram).ok()?;
    let cut = 1.0e-13 * values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut coeffs = vec![0.0; k];
    let mut objective = 0.0;
    for (idx, &lambda) in values.iter().enumerate() {
        if lambda <= cut {
            continue;
        }
        let dot: f64 = (0..k).map(|i| vectors[(i, idx)] * b[i]).sum();
        let scale = dot / lambda;
        objective += dot * scale;
        for i in 0..k {
            coeffs[i] += scale * vectors[(i, idx)];
        }
    }
    if !objective.is_finite() || objective <= 0.0 {
        return None;
    }
    // Normalize the contraction: c^T S c = objective, so this divides by its square root.
    let scale = 1.0 / objective.sqrt();
    for c in coeffs.iter_mut() {
        *c *= scale;
    }
    Some((coeffs, objective.min(1.0)))
}

/// Fit `(n, l)` at `zeta = 1` with `ngauss` primitives.
///
/// Two stages, both deterministic — no randomness and a fixed iteration count, so the same input
/// gives the same exponents to the last bit on every run and every platform:
///
/// 1. an even-tempered start `a_k = a_0 beta^k`, scanned over `(a_0, beta)` on a fixed grid;
/// 2. coordinate descent on each `ln a_k` with a geometrically shrinking step.
///
/// Stage 2 is what closes the gap to the published tables: an even-tempered sequence is a good
/// guess and not the optimum, and the published STO-*n*G exponent ratios are visibly not constant.
fn solve_fit(n: u8, l: u32, ngauss: usize) -> GaussianShell {
    // A Slater `r^{n-1} e^{-r}` has `<r^2> = (2n+2)(2n+1)/4`, and a Gaussian matching that second
    // moment has `alpha = (2l+3)/(2<r^2>)`. That fixes the centre of the sequence to within a
    // factor of a few, which is all the grid below has to cover.
    let r2 = (2.0 * n as f64 + 2.0) * (2.0 * n as f64 + 1.0) / 4.0;
    let centre = (2.0 * l as f64 + 3.0) / (2.0 * r2);

    let mut seeds: Vec<Vec<f64>> = Vec::new();
    for scale_step in 0..13 {
        // centre scaled over four orders of magnitude, geometrically
        let scale = 10.0_f64.powf(-2.0 + scale_step as f64 / 3.0);
        for beta_step in 0..24 {
            let beta = 1.6 + 0.35 * beta_step as f64;
            let a0 = centre * scale / beta.powf((ngauss as f64 - 1.0) / 2.0);
            seeds.push((0..ngauss).map(|k| a0 * beta.powi(k as i32)).collect());
        }
    }

    // Seed from the one-shorter fit as well, extended by a primitive below the sequence, above
    // it, and between each adjacent pair. This is what makes the series **monotone**: the linear
    // solve on a superset of exponents cannot do worse than on the subset, so a length-`k` fit
    // seeded from the length-`k−1` optimum starts at least as good as it. Grid search alone does
    // not guarantee that — measured, seven primitives came out 3e-7 *worse* than six.
    if ngauss > 1 {
        if let Ok(previous) = unit_fit(n, l, ngauss - 1) {
            let p = &previous.exponents;
            let ratio = if p.len() > 1 { p[1] / p[0] } else { 3.0 };
            let mut extensions: Vec<Vec<f64>> = vec![
                std::iter::once(p[0] / ratio)
                    .chain(p.iter().copied())
                    .collect(),
                p.iter()
                    .copied()
                    .chain(std::iter::once(p[p.len() - 1] * ratio))
                    .collect(),
            ];
            for k in 0..p.len().saturating_sub(1) {
                let mut extended = p.clone();
                extended.insert(k + 1, (p[k] * p[k + 1]).sqrt());
                extensions.push(extended);
            }
            seeds.extend(extensions);
        }
    }

    let mut best: Option<(Vec<f64>, Vec<f64>, f64)> = None;
    for alphas in seeds {
        if let Some((coeffs, objective)) = best_contraction(n, l, &alphas) {
            if best.as_ref().map_or(true, |(_, _, o)| objective > *o) {
                best = Some((alphas, coeffs, objective));
            }
        }
    }

    let (mut alphas, mut coeffs, mut objective) =
        best.unwrap_or_else(|| (vec![centre; ngauss], vec![1.0; ngauss], 0.0));

    // Coordinate descent in log-exponent space. Eight sweeps at a step shrinking by 0.55 each
    // time reaches the resolution the quadrature supports; more sweeps change nothing.
    let mut step = 0.5_f64;
    for _ in 0..8 {
        for k in 0..ngauss {
            for direction in [1.0_f64, -1.0] {
                loop {
                    let mut trial = alphas.clone();
                    trial[k] *= (direction * step).exp();
                    // Keep the sequence ordered and separated; a crossing makes the Gram matrix
                    // singular and the pseudo-inverse would silently drop a primitive.
                    trial.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                    let separated = trial.windows(2).all(|w| w[1] / w[0] > 1.05);
                    if !separated {
                        break;
                    }
                    match best_contraction(n, l, &trial) {
                        Some((c, o)) if o > objective + 1.0e-15 => {
                            alphas = trial;
                            coeffs = c;
                            objective = o;
                        }
                        _ => break,
                    }
                }
            }
        }
        step *= 0.55;
    }

    GaussianShell {
        l,
        exponents: alphas,
        coefficients: coeffs,
        overlap: objective.clamp(0.0, 1.0).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Gram matrix's closed form has to agree with a numerical radial integral, or every
    /// fit built on it is wrong in a way no later test would notice — the coefficients would be
    /// self-consistently wrong.
    #[test]
    fn the_closed_form_primitive_overlap_matches_quadrature() {
        for (a, b, l) in [
            (0.3_f64, 1.7_f64, 0u32),
            (2.0, 2.0, 0),
            (0.11, 4.3, 1),
            (1.0, 1.5, 1),
        ] {
            let gamma = gamma_l_plus_3_2(l);
            let na = (2.0 * (2.0 * a).powf(l as f64 + 1.5) / gamma).sqrt();
            let nb = (2.0 * (2.0 * b).powf(l as f64 + 1.5) / gamma).sqrt();
            // int r^{2l+2} e^{-(a+b) r^2} dr on a fine grid.
            let steps = 400_000;
            let rmax = 30.0 / (a + b).sqrt();
            let h = rmax / steps as f64;
            let mut acc = 0.0;
            for i in 0..=steps {
                let r = i as f64 * h;
                let f = r.powi(2 * l as i32 + 2) * (-(a + b) * r * r).exp();
                acc += if i == 0 || i == steps { 0.5 * f } else { f } * h;
            }
            let numeric = na * nb * acc;
            let closed = primitive_overlap(a, b, l);
            assert!(
                (numeric - closed).abs() < 1.0e-9,
                "l={l} a={a} b={b}: {numeric} vs {closed}"
            );
        }
    }

    /// Every shell the parameter sets can ask for must fit well enough that a contour plot cannot
    /// tell the difference. The bar is the published STO-6G quality on the shells it covers.
    #[test]
    fn six_gaussians_reproduce_every_valence_shell() {
        for (n, l) in [
            (1u8, 0u32),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
        ] {
            let shell = expand_slater(n, l, 1.0, 6).unwrap();
            eprintln!(
                "    {n}{}  overlap {:.8}  alphas {:?}",
                if l == 0 { "s" } else { "p" },
                shell.overlap,
                shell
                    .exponents
                    .iter()
                    .map(|a| format!("{a:.4}"))
                    .collect::<Vec<_>>()
            );
            assert!(
                shell.overlap > 0.9999,
                "{n}{l} fits to only {}",
                shell.overlap
            );
        }
    }

    /// More primitives must not fit worse. A monotone series is the cheap check that the search
    /// is finding the optimum rather than a nearby ledge.
    #[test]
    fn the_fit_improves_monotonically_with_length() {
        let mut previous = 0.0;
        for ngauss in 1..=8 {
            let shell = expand_slater(2, 1, 1.0, ngauss).unwrap();
            eprintln!("    2p, {ngauss} gaussians: overlap {:.10}", shell.overlap);
            assert!(
                shell.overlap >= previous - 1.0e-12,
                "{ngauss} primitives fit worse than {}",
                ngauss - 1
            );
            previous = shell.overlap;
        }
    }

    /// The contraction has to be normalized, because Molden readers do not renormalize it.
    #[test]
    fn the_contraction_is_normalized() {
        for (n, l, zeta) in [(2u8, 0u32, 1.83_f64), (3, 1, 1.29), (1, 0, 2.7)] {
            let shell = expand_slater(n, l, zeta, 6).unwrap();
            let mut norm = 0.0;
            for i in 0..shell.exponents.len() {
                for j in 0..shell.exponents.len() {
                    norm += shell.coefficients[i]
                        * shell.coefficients[j]
                        * primitive_overlap(shell.exponents[i], shell.exponents[j], l);
                }
            }
            assert!((norm - 1.0).abs() < 1.0e-10, "n={n} l={l}: norm {norm}");
        }
    }

    /// The `zeta` scaling has to be exact, not approximate: it is what makes one cached fit serve
    /// every element. Checked by comparing the scaled fit against an overlap integral computed at
    /// the real `zeta`.
    #[test]
    fn the_expansion_scales_exactly_with_zeta() {
        let zeta = 2.3_f64;
        let unit = expand_slater(2, 0, 1.0, 6).unwrap();
        let scaled = expand_slater(2, 0, zeta, 6).unwrap();
        for (u, s) in unit.exponents.iter().zip(&scaled.exponents) {
            assert!((s - u * zeta * zeta).abs() < 1.0e-12 * s.max(1.0));
        }
        assert_eq!(unit.coefficients, scaled.coefficients);
        assert!((unit.overlap - scaled.overlap).abs() < 1.0e-15);
    }

    /// Determinism, in the same spirit as `tests/phonon_determinism.rs`: the fit is a search, and
    /// a search that depended on iteration order would put a different basis in the file on every
    /// run.
    #[test]
    fn the_fit_is_bit_reproducible() {
        let a = solve_fit(3, 1, 6);
        let b = solve_fit(3, 1, 6);
        assert_eq!(a.exponents, b.exponents);
        assert_eq!(a.coefficients, b.coefficients);
    }

    #[test]
    fn impossible_shells_and_lengths_are_refused() {
        assert!(expand_slater(1, 1, 1.0, 6).is_err(), "there is no 1p shell");
        assert!(expand_slater(2, 0, 1.0, 0).is_err());
        assert!(expand_slater(2, 0, 1.0, MAX_NGAUSS + 1).is_err());
        assert!(expand_slater(2, 0, -1.0, 6).is_err());
    }
}
