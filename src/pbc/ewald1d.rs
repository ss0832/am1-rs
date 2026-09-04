// SPDX-License-Identifier: GPL-3.0-or-later

//! Lattice sums for a **one-dimensional** periodic system — a chain.
//!
//! # Why there is no `erfc` here
//!
//! The other two dimensionalities need an Ewald split because their real-space sums converge too
//! slowly to truncate: `Σ_T |T|⁻¹` over a plane or a volume diverges. Along a line it also
//! diverges, but only *logarithmically*, and — this is the part that changes the design — every
//! channel above the monopole converges **absolutely**: `Σ_n n⁻²` and beyond need no treatment
//! at all. So a chain needs exactly one thing handled, `1/R`, and handling it does not require
//! splitting the sum.
//!
//! What it requires instead is the tail. Summing `Σ_{|n|≤N} 1/|r + nLê|` directly and stopping
//! leaves an error of order `Σ_{n>N} 1/(nL)`, which does not go away with a larger `N` — it goes
//! like `ln`. But the same expansion that shows the divergence also gives everything else in
//! closed form. Writing `u = 1/(nL)` and `s = z² + ρ²`, the two images `±n` together contribute
//!
//! ```text
//! (2/(nL)) [ 1 + c₂ u² + c₄ u⁴ + … ],    c₂ = z² − ρ²/2,
//!                                         c₄ = z⁴ − 3z²ρ² + 3ρ⁴/8
//! ```
//!
//! — the **odd** powers cancel between `+n` and `−n`, which is why the series is in `u²` and why
//! it converges so fast. Summed over `n > N` those become Hurwitz zeta values `ζ(3, N+1)`,
//! `ζ(5, N+1)`, …, computed once per lattice. What is left is the `u⁰` term, `(2/L) Σ_{n>N} 1/n`,
//! and that is the logarithm.
//!
//! The result: no special functions, no splitting parameter, and an expression that is a finite
//! sum of `1/|r + nLê|` plus polynomials in `z²` and `ρ²`. It is therefore **smooth everywhere,
//! including on the chain axis** (`ρ = 0`), which a Bessel-function formulation is not — and it
//! differentiates exactly, so forces and the axial stress come out of the same code.
//!
//! # The convention, stated
//!
//! The logarithm has to be regularized, and this module does it as
//!
//! ```text
//! φ(r) = lim_{M→∞} [ Σ_{|n|≤M} 1/|r + nLê| − (2/L) ln M ]
//! ```
//!
//! which exists. Subtracting `(2/L) ln M` adds a term proportional to `(Σ_a q_a)²`, so for a
//! **neutral** cell it cancels identically and the answer is convention-free. For a charged
//! chain it does not cancel, and the number then means "relative to a neutralizing charge at the
//! stated reference" — see [`AxisConvention`]. A charged chain is refused unless the caller says
//! which reference it wants, because there is no natural one.

use crate::dual::Scalar;
use crate::error::{Am1Error, Result};
use crate::lattice::Lattice;
use crate::math::Vec3;

/// Euler–Mascheroni constant, from the `γ − H_N` regularization constant.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// How a **charged** chain's compensating background is referenced.
///
/// The potential of a charged line is `−2λ ln ρ`: it diverges logarithmically with distance and
/// has no finite value without a reference radius. This is the one-dimensional counterpart of
/// [`crate::pbc::ewald2d::SheetConvention`], and like it, it is never inferred.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AxisConvention {
    /// Refuse. The default: a charged chain returns an error naming this enum.
    #[default]
    Undefined,
    /// A neutralizing line charge, with the potential referenced to the cylindrical radius
    /// `radius` (Bohr). The reported energy is then "per cell, relative to that reference".
    BackgroundRadius { radius: f64 },
}

/// A one-dimensional lattice sum for one chain.
#[derive(Clone, Debug)]
pub struct Ewald1D {
    /// Chain repeat vector (Bohr).
    axis: Vec3,
    /// Unit vector along the chain. Kept for callers that need the chain direction; the
    /// potential itself derives it from the (possibly strained) repeat vector.
    #[allow(dead_code)]
    unit: Vec3,
    /// Repeat length `L` (Bohr).
    pub period: f64,
    /// Number of images summed explicitly on each side.
    pub n_real: i32,
    /// `γ − H_N`, the numerator of the regularization constant. Stored bare rather than divided
    /// by `L`, because `L` strains and this does not.
    gamma_minus_harmonic: f64,
    /// `ζ(3, N+1)`, `ζ(5, N+1)`, `ζ(7, N+1)` — the tail sums, likewise independent of `L`.
    zeta3: f64,
    zeta5: f64,
    zeta7: f64,
}

/// `ζ(s, a) = Σ_{n ≥ a} n^{−s}` for integer `a ≥ 1` and `s ≥ 3`.
///
/// Direct summation with an Euler–Maclaurin tail. Evaluated once per lattice, so the cost of
/// being generous with the term count is irrelevant next to being sure it is converged.
fn hurwitz_zeta(s: i32, a: i32) -> f64 {
    let mut total = 0.0;
    let cut = a + 200_000;
    for n in a..cut {
        total += (n as f64).powi(-s);
    }
    // ∫_cut^∞ x^{−s} dx + ½ cut^{−s}, the leading Euler–Maclaurin terms.
    let x = cut as f64;
    total += x.powi(-(s - 1)) / (s - 1) as f64 + 0.5 * x.powi(-s);
    total
}

impl Ewald1D {
    /// Build the sum for `lattice`, which must have exactly one periodic direction.
    ///
    /// `n_real` sets how many images are summed explicitly before the analytic tail takes over.
    /// The answer must not depend on it — that is the strongest check available here, and it is
    /// what the `α`-independence test is for the other two dimensionalities.
    pub fn new(lattice: &Lattice, n_real: i32) -> Result<Self> {
        let axes: Vec<usize> = (0..3).filter(|&i| lattice.periodic[i]).collect();
        if axes.len() != 1 {
            return Err(Am1Error::InvalidInput(format!(
                "the one-dimensional lattice sum needs exactly one periodic direction, got {}",
                lattice.n_periodic()
            )));
        }
        if n_real < 4 {
            return Err(Am1Error::InvalidInput(
                "the one-dimensional lattice sum needs at least 4 explicit images; the analytic \
                 tail is an expansion in 1/(N L) and is not accurate for a very small N"
                    .into(),
            ));
        }
        let axis = lattice.cell.col[axes[0]];
        let period = axis.norm();
        if period < 1.0e-12 {
            return Err(Am1Error::InvalidInput(
                "the chain repeat vector is degenerate".into(),
            ));
        }
        let unit = axis / period;

        // (2/L)(γ − H_N): what is left after subtracting (2/L) ln M as M → ∞.
        let harmonic: f64 = (1..=n_real).map(|n| 1.0 / n as f64).sum();

        Ok(Self {
            axis,
            unit,
            period,
            n_real,
            gamma_minus_harmonic: EULER_GAMMA - harmonic,
            zeta3: hurwitz_zeta(3, n_real + 1),
            zeta5: hurwitz_zeta(5, n_real + 1),
            zeta7: hurwitz_zeta(7, n_real + 1),
        })
    }

    /// The regularized chain potential at displacement `r`, in atomic units (inverse Bohr).
    ///
    /// Generic over the scalar type, so the gradient and the second derivative come from the
    /// same expression by forward-mode AD rather than from a transcription of it.
    ///
    /// `r = 0` is the self term: the `n = 0` image is skipped, which is the whole of the
    /// treatment — unlike the split sums there is no Gaussian self-energy to remove, because no
    /// Gaussian was ever introduced.
    pub fn pair_potential_scalar<S: Scalar>(&self, r: [S; 3]) -> S {
        self.pair_potential_strained(r, &[[S::cst(0.0); 3]; 3])
    }

    /// [`Self::pair_potential_scalar`] with a homogeneous strain applied to the displacement and
    /// to the chain repeat vector.
    ///
    /// Everything that depends on the repeat length has to move with it, and here that is more
    /// than the obvious `1/|r + nLê|`: the tail coefficients go as `L⁻³`, `L⁻⁵`, `L⁻⁷` and the
    /// regularization constant as `L⁻¹`. Seeding the strain through the same expression means
    /// none of those can be forgotten, which is exactly the kind of term a hand-derived stress
    /// drops silently.
    pub fn pair_potential_strained<S: Scalar>(&self, r: [S; 3], eps: &[[S; 3]; 3]) -> S {
        let deform = |v: [S; 3]| -> [S; 3] {
            let mut out = v;
            for (i, o) in out.iter_mut().enumerate() {
                for (j, vj) in v.iter().enumerate() {
                    *o = *o + eps[i][j] * *vj;
                }
            }
            out
        };
        let rd = deform(r);
        let a0 = [
            S::cst(self.axis.x),
            S::cst(self.axis.y),
            S::cst(self.axis.z),
        ];
        let ad = deform(a0);
        let period = (ad[0] * ad[0] + ad[1] * ad[1] + ad[2] * ad[2]).sqrt();
        let inv_l = period.recip();
        let unit = [ad[0] * inv_l, ad[1] * inv_l, ad[2] * inv_l];

        let mut total = S::cst(0.0);
        for n in -self.n_real..=self.n_real {
            let f = n as f64;
            let d = [rd[0] + ad[0] * f, rd[1] + ad[1] * f, rd[2] + ad[2] * f];
            let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if dist2.val() < 1.0e-20 {
                continue; // the atom itself, at n = 0
            }
            total = total + dist2.sqrt().recip();
        }

        // z along the chain, ρ² perpendicular to it.
        let z = rd[0] * unit[0] + rd[1] * unit[1] + rd[2] * unit[2];
        let r2 = rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2];
        let z2 = z * z;
        let rho2 = r2 - z2;

        // c₂ = z² − ρ²/2,  c₄ = z⁴ − 3z²ρ² + 3ρ⁴/8,
        // c₆ = z⁶ − (15/2) z⁴ρ² + (45/8) z²ρ⁴ − (5/16) ρ⁶
        let c2 = z2 - rho2 * 0.5;
        let c4 = z2 * z2 - z2 * rho2 * 3.0 + rho2 * rho2 * 0.375;
        let c6 = z2 * z2 * z2 - z2 * z2 * rho2 * 7.5 + z2 * rho2 * rho2 * 5.625
            - rho2 * rho2 * rho2 * 0.3125;

        let inv_l3 = inv_l * inv_l * inv_l;
        let inv_l5 = inv_l3 * inv_l * inv_l;
        let inv_l7 = inv_l5 * inv_l * inv_l;
        total
            + c2 * inv_l3 * (2.0 * self.zeta3)
            + c4 * inv_l5 * (2.0 * self.zeta5)
            + c6 * inv_l7 * (2.0 * self.zeta7)
            + inv_l * (2.0 * self.gamma_minus_harmonic)
    }

    /// `∂φ/∂ε_αβ` at `ε = 0`, by seeding the strain instead of the position. See
    /// [`crate::pbc::ewald2d::Ewald2D::pair_potential_strain`].
    pub fn pair_potential_strain(&self, r: Vec3) -> [[f64; 3]; 3] {
        use crate::dual::Dual;
        const PASSES: [[(usize, usize); 3]; 2] =
            [[(0, 0), (1, 1), (2, 2)], [(0, 1), (0, 2), (1, 2)]];
        let mut out = [[0.0_f64; 3]; 3];
        for pass in &PASSES {
            let mut eps = [[Dual::constant(0.0); 3]; 3];
            for (slot, (a, b)) in pass.iter().enumerate() {
                let v = Dual::var(0.0, slot);
                eps[*a][*b] = v;
                if a != b {
                    eps[*b][*a] = v;
                }
            }
            let v = self.pair_potential_strained::<Dual>(
                [
                    Dual::constant(r.x),
                    Dual::constant(r.y),
                    Dual::constant(r.z),
                ],
                &eps,
            );
            for (slot, (a, b)) in pass.iter().enumerate() {
                let d = if a == b { v.d[slot] } else { 0.5 * v.d[slot] };
                out[*a][*b] = d;
                out[*b][*a] = d;
            }
        }
        out
    }

    /// Value of [`Self::pair_potential_scalar`].
    pub fn pair_potential(&self, r: Vec3) -> f64 {
        self.pair_potential_scalar::<f64>([r.x, r.y, r.z])
    }

    /// Gradient of [`Self::pair_potential`] with respect to `r`, by forward-mode AD.
    pub fn pair_potential_gradient(&self, r: Vec3) -> Vec3 {
        use crate::dual::Dual;
        let v = self.pair_potential_scalar::<Dual>([
            Dual::var(r.x, 0),
            Dual::var(r.y, 1),
            Dual::var(r.z, 2),
        ]);
        Vec3::new(v.d[0], v.d[1], v.d[2])
    }

    /// Second derivative of [`Self::pair_potential`] with respect to `r`, by forward-mode AD.
    pub fn pair_potential_hessian(&self, r: Vec3) -> [[f64; 3]; 3] {
        use crate::dual2::Dual2;
        let v = self.pair_potential_scalar::<Dual2>([
            Dual2::var(r.x, 0),
            Dual2::var(r.y, 1),
            Dual2::var(r.z, 2),
        ]);
        v.h
    }

    /// `S(q; r) = Σ_n e^{iqnL} / |r + n·a|` for a **chain**, and its first two `r`-derivatives.
    ///
    /// The one-dimensional counterpart of [`crate::pbc::ewald::EwaldSum::phased_pair_potential`],
    /// and what lets the DFPT response carry its long-range monopole channel on a chain.
    ///
    /// # `q ≠ 0` is the easy case, and it is a different case
    ///
    /// This path keeps the chain sum's no-splitting character — there is no `α` here to get wrong —
    /// but what changes with `q` is the **tail**, and it changes character rather than size.
    ///
    /// At a reciprocal lattice vector every phase is one, `Σ_n 1/|r + na|` diverges
    /// logarithmically, and the value is only defined by the neutralizing line-charge convention
    /// [`Self::pair_potential`] carries. That case delegates, so the convention has one home.
    ///
    /// Away from the reciprocal lattice the phase oscillates and the sum converges on its own, and
    /// the line charge and the `1/n³` dipole tail are `q = 0` artefacts that must **not** be
    /// carried over — doing so would make `S(q)` discontinuous as `q → 0` in a way no `q = 0` test
    /// could catch, since `Σ_n e^{iqnL}/(nL)` is finite while `Σ_n 1/(nL)` is the thing the line
    /// charge exists to cancel.
    ///
    /// # Why the tail is summed by parts
    ///
    /// Truncating the oscillating sum is only `O(1/N)`. Dirichlet's test converges it because the
    /// partial sums of `z^n` are bounded — by `|w| = 1/|1−z| = 1/(2|sin(θ/2)|)` — but that same
    /// bound multiplies the first neglected term and blows up as `q → 0`. [`abel_tail`] transforms
    /// it into a sum over *differences* of the kernel, which is an asymptotic series in `1/N` with
    /// conditioning `|w|`, truncated at its smallest term. [`CHAIN_TAIL_CONDITIONING`] widens the
    /// direct sum where that would otherwise be ill-conditioned.
    pub fn phased_pair_potential(
        &self,
        q_cart: Vec3,
        r: Vec3,
        _exclude_self: bool,
    ) -> crate::pbc::ewald::PhasedDelta {
        use crate::pbc::ewald::PhasedDelta;
        let mut out = PhasedDelta::default();

        // At a reciprocal lattice vector this is the unphased sum, line-charge convention and all.
        //
        // The one-dimensional form of `ewald::q_folds_to_gamma`, written against the axis because
        // a chain keeps no lattice: only the periodic direction carries a translation, so only
        // `q·a/2π` can make a phase, and a component off the axis is never read.
        let turns = q_cart.dot(self.axis) / (2.0 * std::f64::consts::PI);
        if (turns - turns.round()).abs() < 1.0e-9 {
            out.value[0] = self.pair_potential(r);
            let g = self.pair_potential_gradient(r);
            out.gradient[0][0] = g.x;
            out.gradient[1][0] = g.y;
            out.gradient[2][0] = g.z;
            let h = self.pair_potential_hessian(r);
            for i in 0..3 {
                for j in 0..3 {
                    out.hessian[i][j][0] = h[i][j];
                }
            }
            return out;
        }

        let theta = q_cart.dot(self.axis); // radians per cell
                                           // Widen the direct sum where the Abel tail would be ill-conditioned.
        let w_magnitude = 0.5 / (0.5 * theta).sin().abs();
        let images = (self.n_real as usize)
            .max((CHAIN_TAIL_CONDITIONING * w_magnitude).ceil() as usize)
            .min(CHAIN_IMAGES_CEILING);

        // `n = 0`, phase one at every `q`.
        add_bare(&mut out, [1.0, 0.0], r);
        // ±n together, so the cancellation happens before rounding — the discipline the unphased
        // sum keeps for the same reason.
        for n in 1..=images {
            let shift = self.axis * n as f64;
            let angle = theta * n as f64;
            let (sin, cos) = angle.sin_cos();
            add_bare(&mut out, [cos, sin], r + shift);
            add_bare(&mut out, [cos, -sin], r - shift);
        }

        // The two half-line tails, `n > N` and `n < −N`.
        for tail in [
            abel_tail(r, self.axis, theta, images + 1),
            abel_tail(r, self.axis * -1.0, -theta, images + 1),
        ] {
            out.value[0] += tail[0][0];
            out.value[1] += tail[0][1];
            for a in 0..3 {
                out.gradient[a][0] += tail[1 + a][0];
                out.gradient[a][1] += tail[1 + a][1];
                for b in 0..3 {
                    out.hessian[a][b][0] += tail[4 + 3 * a + b][0];
                    out.hessian[a][b][1] += tail[4 + 3 * a + b][1];
                }
            }
        }
        out
    }
}

/// `N/|w|` at least this large: at 40 the truncation floor is `~e^{−40} ≈ 4·10⁻¹⁸` relative, below
/// everything else in this crate. Each extra image is a dozen flops, so this is cheap except
/// within `~1/40` of a reciprocal lattice vector, where the sum is near its physical divergence.
const CHAIN_TAIL_CONDITIONING: f64 = 40.0;

/// Ceiling on the widened image count, so a `q` pathologically close to — but not at — a reciprocal
/// lattice vector degrades in accuracy rather than in run time.
const CHAIN_IMAGES_CEILING: usize = 100_000;

/// Terms kept in the Abel series before optimal truncation takes over.
const ABEL_ORDERS: usize = 20;

/// Value, three gradient components, nine Hessian entries row-major.
type Bundle = [f64; 13];

/// The bare kernel bundle at one point: `1/r`, `∇(1/r)`, `∇∇(1/r)`.
fn bare_bundle(v: Vec3) -> Bundle {
    let mut out = [0.0; 13];
    let r = v.norm();
    if r < 1.0e-12 {
        return out;
    }
    let i1 = 1.0 / r;
    let i3 = i1 * i1 * i1;
    let i5 = i3 * i1 * i1;
    let dv = [v.x, v.y, v.z];
    out[0] = i1;
    for a in 0..3 {
        out[1 + a] = -dv[a] * i3;
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            out[4 + 3 * a + b] = 3.0 * dv[a] * dv[b] * i5 - delta * i3;
        }
    }
    out
}

/// Add `phase · (1/|v|)` and its derivatives to `out`.
fn add_bare(out: &mut crate::pbc::ewald::PhasedDelta, phase: [f64; 2], v: Vec3) {
    let bundle = bare_bundle(v);
    out.value[0] += phase[0] * bundle[0];
    out.value[1] += phase[1] * bundle[0];
    for a in 0..3 {
        out.gradient[a][0] += phase[0] * bundle[1 + a];
        out.gradient[a][1] += phase[1] * bundle[1 + a];
        for b in 0..3 {
            out.hessian[a][b][0] += phase[0] * bundle[4 + 3 * a + b];
            out.hessian[a][b][1] += phase[1] * bundle[4 + 3 * a + b];
        }
    }
}

#[inline]
fn complex_mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

/// `Σ_{n ≥ first} z^n f(n)` for one half-line, by repeated Abel transformation.
///
/// With `z = e^{iθ}` and `w = 1/(1−z)`, telescoping the partial sums gives
///
/// ```text
/// Σ_{n≥N} z^n f(n) = w z^N f(N) − z w Σ_{n≥N} z^n Δf(n),     Δf(n) = f(n) − f(n+1)
/// ```
///
/// and applying it to its own remainder `K` times over,
///
/// ```text
/// Σ_{n≥N} z^n f(n) = Σ_{j<K} (−zw)^j · w z^N · Δʲf(N)  +  (−zw)^K R_K,   |R_K| ≲ K!/(K N^K L)
/// ```
///
/// The differences are **exact** forward differences of samples at `N … N+K`, not an asymptotic
/// formula. The terms decay like `(|w|/N)^{j+1} j!` — an asymptotic series, best truncated at its
/// smallest term, which is what the loop does: it stops before the first term that grows, so the
/// error is the size of the first omitted one.
fn abel_tail(d: Vec3, step: Vec3, theta: f64, first: usize) -> [[f64; 2]; 13] {
    let mut table: Vec<Bundle> = (0..=ABEL_ORDERS)
        .map(|j| bare_bundle(d + step * (first + j) as f64))
        .collect();
    let mut differences: Vec<Bundle> = Vec::with_capacity(ABEL_ORDERS + 1);
    differences.push(table[0]);
    for level in 1..=ABEL_ORDERS {
        for i in 0..=(ABEL_ORDERS - level) {
            let next = table[i + 1];
            for (slot, sub) in table[i].iter_mut().zip(next.iter()) {
                *slot -= sub;
            }
        }
        differences.push(table[0]);
    }

    let (sin, cos) = theta.sin_cos();
    let z = [cos, sin];
    // w = 1/(1 − z). Nonzero away from the reciprocal lattice, the only place this is called.
    let denom2 = (1.0 - cos) * (1.0 - cos) + sin * sin;
    let w = [(1.0 - cos) / denom2, sin / denom2];
    let ratio = {
        let zw = complex_mul(z, w);
        [-zw[0], -zw[1]]
    };
    let (sin_first, cos_first) = (theta * first as f64).sin_cos();
    let mut coefficient = complex_mul(w, [cos_first, sin_first]);

    let mut out = [[0.0; 2]; 13];
    let mut previous = f64::INFINITY;
    for difference in differences.iter() {
        let weight = (coefficient[0] * coefficient[0] + coefficient[1] * coefficient[1]).sqrt();
        let largest = difference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let magnitude = weight * largest;
        // Optimal truncation: stop *before* the first growing term.
        if magnitude > previous {
            break;
        }
        for (slot, value) in out.iter_mut().zip(difference.iter()) {
            slot[0] += coefficient[0] * value;
            slot[1] += coefficient[1] * value;
        }
        if magnitude < 1.0e-18 {
            break;
        }
        previous = magnitude;
        coefficient = complex_mul(coefficient, ratio);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(l: f64) -> Lattice {
        Lattice::from_vectors(
            Vec3::new(l, 0.0, 0.0),
            Vec3::new(0.0, 60.0, 0.0),
            Vec3::new(0.0, 0.0, 60.0),
            [true, false, false],
        )
        .unwrap()
    }

    #[test]
    fn a_slab_or_a_bulk_is_refused() {
        assert!(Ewald1D::new(&Lattice::cubic(10.0).unwrap(), 40).is_err());
        let slab = Lattice::from_vectors(
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 8.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap();
        assert!(Ewald1D::new(&slab, 40).is_err());
    }

    #[test]
    fn the_potential_does_not_depend_on_how_many_images_are_summed_explicitly() {
        // The counterpart of the `α`-independence test for the split sums. `n_real` decides only
        // where the explicit sum stops and the analytic tail starts, so the total must not move.
        // Every coefficient of the tail expansion is checked at once by this.
        let lattice = chain(5.0);
        let probes = [
            Vec3::new(1.3, 0.7, 0.2),
            Vec3::new(2.5, 0.0, 0.0),  // on the chain axis, rho = 0
            Vec3::new(0.0, 3.1, -1.4), // purely transverse
            Vec3::zero(),              // the self term
        ];
        let mut reference: Option<Vec<f64>> = None;
        eprintln!("     n_real |  potential at each probe (1/Bohr)");
        for n in [8_i32, 16, 32, 64, 128] {
            let e = Ewald1D::new(&lattice, n).unwrap();
            let vals: Vec<f64> = probes.iter().map(|p| e.pair_potential(*p)).collect();
            eprintln!(
                "    {n:7} | {}",
                vals.iter()
                    .map(|v| format!("{v:15.11}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            match &reference {
                None => reference = Some(vals),
                Some(r) => {
                    for (a, b) in vals.iter().zip(r) {
                        assert!(
                            (a - b).abs() < 1.0e-9,
                            "the chain sum moved by {:.3e} when n_real became {n}",
                            (a - b).abs()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_alternating_chain_madelung_constant_is_two_ln_two() {
        // An alternating ±1 chain has Madelung constant `2 ln 2` exactly, from
        // `Σ_j q_j/r_j = (2/a) Σ_{n≥1} (−1)^{n+1}/n = (2/a) ln 2`.
        //
        // An exact closed-form reference with nothing to do with this implementation — the
        // one-dimensional counterpart of the rock-salt and square-lattice constants used for the
        // other two dimensionalities. It checks the explicit sum, the tail coefficients and the
        // regularization constant together, because a chain of alternating charges is neutral
        // and the convention-dependent term has to cancel exactly for this to come out right.
        let a = 1.0_f64;
        let lattice = chain(2.0 * a);
        let ewald = Ewald1D::new(&lattice, 64).unwrap();
        let sites = [(Vec3::zero(), 1.0), (Vec3::new(a, 0.0, 0.0), -1.0)];
        let (r0, q0) = sites[0];
        let mut potential = 0.0;
        for (rj, qj) in &sites {
            potential += qj * ewald.pair_potential(*rj - r0);
        }
        let madelung = -potential * a / q0;
        let reference = 2.0 * 2.0_f64.ln();
        eprintln!(
            "    1D alternating-chain Madelung constant: {madelung:.12} (exact 2 ln 2 = {reference:.12})"
        );
        assert!(
            (madelung - reference).abs() < 1.0e-9,
            "1D Madelung constant came out {madelung:.12}, expected {reference:.12}"
        );
    }

    #[test]
    fn the_strain_derivative_matches_a_strain_finite_difference() {
        // Only the axial component is a real deformation of a chain: there is no cell length in
        // the other two directions to strain, and the caller zeroes those components.
        //
        // This is the test that catches a forgotten `L` dependence — the tail coefficients go as
        // `L⁻³`, `L⁻⁵`, `L⁻⁷` and the regularization constant as `L⁻¹`, and a hand-derived
        // stress that differentiated only the explicit `1/|r + nLê|` sum would pass every other
        // test in this file.
        let lattice = chain(5.0);
        let ewald = Ewald1D::new(&lattice, 48).unwrap();
        let probe = Vec3::new(1.9, 1.2, -0.6);
        let analytic = ewald.pair_potential_strain(probe);

        let step = 1.0e-6;
        let shifted = |sign: f64| {
            let mut eps = [[0.0_f64; 3]; 3];
            eps[0][0] = sign * step;
            let l = lattice.strained(&eps).unwrap();
            let e = Ewald1D::new(&l, 48).unwrap();
            let mut p = probe;
            p.x += eps[0][0] * probe.x;
            e.pair_potential(p)
        };
        let fd = (shifted(1.0) - shifted(-1.0)) / (2.0 * step);
        let delta = (analytic[0][0] - fd).abs();
        eprintln!(
            "    1D axial strain: analytic {:.10}, FD {fd:.10}, delta {delta:.3e}",
            analytic[0][0]
        );
        assert!(delta < 1.0e-7, "1D strain derivative mismatch {delta:.3e}");
    }

    #[test]
    fn the_gradient_and_hessian_match_finite_differences() {
        let ewald = Ewald1D::new(&chain(5.0), 48).unwrap();
        let step = 1.0e-5;
        let mut worst_g = 0.0_f64;
        let mut worst_h = 0.0_f64;
        for probe in [
            Vec3::new(1.6, 0.8, 2.1),
            Vec3::new(2.4, 0.0, 0.0), // on the axis: the case a Bessel formulation cannot do
            Vec3::new(0.3, 3.0, -1.0),
        ] {
            let g = ewald.pair_potential_gradient(probe);
            let hess = ewald.pair_potential_hessian(probe);
            for j in 0..3 {
                let mut plus = probe;
                let mut minus = probe;
                match j {
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
                let fd = (ewald.pair_potential(plus) - ewald.pair_potential(minus)) / (2.0 * step);
                worst_g = worst_g.max(([g.x, g.y, g.z][j] - fd).abs());
                let gp = ewald.pair_potential_gradient(plus);
                let gm = ewald.pair_potential_gradient(minus);
                let fdh = (gp - gm) / (2.0 * step);
                for (i, f) in [fdh.x, fdh.y, fdh.z].iter().enumerate() {
                    worst_h = worst_h.max((hess[i][j] - f).abs());
                }
            }
        }
        eprintln!("    1D chain vs FD: gradient {worst_g:.3e}, hessian {worst_h:.3e}");
        assert!(worst_g < 1.0e-8, "1D gradient mismatch {worst_g:.3e}");
        assert!(worst_h < 1.0e-6, "1D hessian mismatch {worst_h:.3e}");
    }
}
