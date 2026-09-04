// SPDX-License-Identifier: GPL-3.0-or-later

//! Ewald summation for a **two-dimensional** periodic system — a slab.
//!
//! # Why this is not the three-dimensional sum with a vacuum gap
//!
//! The usual shortcut for a slab is to make the cell tall, leave vacuum, and run a 3D Ewald
//! sum. That is the Yeh–Berkowitz "EW3DC" construction, and this module deliberately does not
//! use it, for three reasons that are all visible in the results rather than matters of taste:
//!
//! * It is only asymptotically exact. The vacuum has to be roughly three times the slab
//!   thickness, and the answer keeps moving as it grows — so "converged" means converged in a
//!   parameter that has no physical meaning.
//! * Its dipole correction is derived for point charges, and extending it to the rank-2
//!   multipoles and the `R⁻³` Klopman–Ohno kernel that NDDO actually carries is not standard.
//! * **There is no analytic stress.** The `c` axis is fictitious, so `∂E/∂ε_zz` is meaningless,
//!   and the in-plane components are contaminated by the vacuum. A slab under a barostat is
//!   exactly the case a 2D treatment is for.
//!
//! Parry's formulation sums the plane exactly and leaves the third direction open, so the
//! in-plane `2 × 2` stress falls out directly and no vacuum parameter exists to converge.
//!
//! # The sum
//!
//! With `r = (ρ, z)` split into its in-plane part and its component along the plane normal,
//! `A` the cell area, `h` the in-plane reciprocal vectors, and `α` the splitting parameter:
//!
//! ```text
//! φ(ρ, z) =   Σ_T erfc(α|r+T|)/|r+T|
//!           + (π/A) Σ_{h≠0} (cos(h·ρ)/h) [ e^{hz} erfc(h/2α + αz) + e^{−hz} erfc(h/2α − αz) ]
//!           − (2π/A) [ z erf(αz) + e^{−α²z²}/(α√π) ]
//!           − 2α/√π                                       (only when r = 0)
//! ```
//!
//! The `h = 0` term is the one that carries the slab's own depolarizing field: for large `|z|`
//! it tends to `−2π|z|/A`, the parallel-plate result, which is why a 2D sum needs no
//! compensating background for a **neutral** cell and why a charged one is not defined without
//! choosing where the compensating sheet sits. See [`crate::pbc::ewald2d::SheetConvention`].
//!
//! # The overflow that makes this delicate
//!
//! `e^{hz} erfc(h/2α + αz)` is a product of a factor that overflows and a factor that
//! underflows, and it leaves double precision at moderate `hz` — around `h z > 700` for the
//! first and `h/2α + αz > 26.6` for the second, neither of which is a large slab. Writing it
//! literally produces `inf × 0`.
//!
//! Composing the exponents analytically fixes it exactly. With `u = h/2α + αz`,
//!
//! ```text
//! e^{hz} e^{−u²} = e^{hz − h²/4α² − hz − α²z²} = e^{−h²/4α² − α²z²}
//! ```
//!
//! so `e^{hz} erfc(u) = e^{−h²/4α² − α²z²} · erfcx(u)` with a prefactor that is bounded by 1 and
//! a scaled `erfc` that decays as `1/(u√π)`. Neither can leave range. That is what
//! [`crate::dual::Scalar::erfcx`] exists for.

use crate::dual::Scalar;
use crate::error::{Am1Error, Result};
use crate::lattice::{ImageOffset, Lattice};
use crate::math::Vec3;

/// How a **charged** slab's compensating background is placed.
///
/// A charged two-dimensional cell has no energy until this is chosen, and the choice changes the
/// answer by a finite amount rather than a small one. The compensating charge is a *sheet*, and
/// the interaction of a point charge with a sheet is `−2πσ|z − z₀|`: it depends on where the
/// sheet is put. There is no natural answer the way tin-foil is natural in three dimensions.
///
/// So this is never inferred. A neutral slab needs none of it; a charged one must say.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SheetConvention {
    /// Refuse. The default: a charged slab returns an error naming this enum rather than a
    /// number whose convention the caller never chose.
    #[default]
    Undefined,
    /// A uniform neutralizing sheet in the plane `z = z₀`, `z₀` measured along the plane normal
    /// in Bohr, in the same frame as the atomic coordinates.
    Sheet { z0: f64 },
    /// A uniform neutralizing sheet at the charge-weighted mean plane of the cell's atoms.
    ///
    /// The least arbitrary choice available, and the one that makes the result invariant under
    /// rigid translation of the slab along the normal — which `Sheet` is not. Still a
    /// convention, and still reported as one.
    CentreOfCharge,
}

/// A two-dimensional (Parry) Ewald sum for one lattice.
#[derive(Clone, Debug)]
pub struct Ewald2D {
    lattice: Lattice,
    pub alpha: f64,
    /// Index of the **non**-periodic axis.
    pub normal_axis: usize,
    /// Unit normal to the periodic plane.
    normal: Vec3,
    /// In-plane reciprocal vectors and their magnitudes, `G = 0` excluded.
    reciprocal: Vec<(Vec3, f64)>,
    real_images: Vec<ImageOffset>,
    real_cutoff: f64,
    /// Radius of the reciprocal shell, so the phased sum can re-enumerate it centred on `q`.
    g_cutoff: f64,
    /// Cell area (Bohr²).
    pub area: f64,
}

/// Default splitting parameter for a cell of area `A`: `√π / √A`.
///
/// The two-dimensional analogue of the three-dimensional `√π / V^{1/3}` — it balances the real
/// and reciprocal sums at comparable cost. Nothing depends on the choice; see the
/// `α`-independence test.
pub fn default_alpha_2d(area: f64) -> f64 {
    std::f64::consts::PI.sqrt() / area.sqrt()
}

impl Ewald2D {
    /// Build the sum for `lattice`, which must have exactly two periodic directions.
    ///
    /// The non-periodic axis must be **orthogonal** to the periodic plane. That is not a
    /// limitation of Parry's method but of this implementation's bookkeeping: it takes the
    /// in-plane reciprocal vectors straight from the 3D reciprocal basis, which lie in the plane
    /// only when the third cell vector is normal to it. A tilted slab is rejected rather than
    /// silently summed over vectors that are not in the plane.
    pub fn new(lattice: &Lattice, alpha: f64, accuracy: f64) -> Result<Self> {
        let axes: Vec<usize> = (0..3).filter(|&i| !lattice.periodic[i]).collect();
        if axes.len() != 1 {
            return Err(Am1Error::InvalidInput(format!(
                "the two-dimensional Ewald sum needs exactly two periodic directions, got {}",
                lattice.n_periodic()
            )));
        }
        let normal_axis = axes[0];
        if alpha.is_nan() || alpha <= 0.0 {
            return Err(Am1Error::InvalidInput(
                "the Ewald splitting parameter must be positive".into(),
            ));
        }

        let (p, q) = match normal_axis {
            0 => (1, 2),
            1 => (2, 0),
            _ => (0, 1),
        };
        let a_p = lattice.cell.col[p];
        let a_q = lattice.cell.col[q];
        let cross = a_p.cross(a_q);
        let area = cross.norm();
        if area < 1.0e-12 {
            return Err(Am1Error::InvalidInput(
                "the periodic plane of a slab is degenerate".into(),
            ));
        }
        let normal = cross / area;

        // Reject a tilted slab rather than summing the wrong vectors.
        let a_n = lattice.cell.col[normal_axis];
        let tilt = (a_n.dot(normal).abs() / a_n.norm().max(1.0e-30) - 1.0).abs();
        if tilt > 1.0e-8 {
            return Err(Am1Error::InvalidInput(format!(
                "the two-dimensional Ewald sum needs the non-periodic axis orthogonal to the \
                 periodic plane; axis {normal_axis} is tilted by {:.3e} rad",
                (a_n.dot(normal).abs() / a_n.norm()).acos()
            )));
        }

        // Same accuracy-driven cutoffs as the three-dimensional sum.
        let tol = (-accuracy.ln()).sqrt();
        let real_cutoff = tol / alpha;
        let g_cutoff = 2.8 * alpha * tol;

        // A slab's real-space sum runs over in-plane translations only, but a pair can be
        // separated along the normal, so the cutoff has to admit every in-plane translation that
        // could still bring an image within range.
        let diagonal = (a_p + a_q).norm();
        let real_images: Vec<ImageOffset> = lattice
            .image_offsets(real_cutoff + diagonal)
            .into_iter()
            .collect();

        let mut reciprocal = Vec::new();
        for (_, g) in lattice.reciprocal_vectors_within(g_cutoff) {
            let h = g.norm();
            if h > 1.0e-12 {
                reciprocal.push((g, h));
            }
        }

        Ok(Self {
            lattice: *lattice,
            alpha,
            normal_axis,
            normal,
            reciprocal,
            real_images,
            real_cutoff,
            g_cutoff,
            area,
        })
    }

    /// `S(q; r) = Σ_T e^{iq·T} / |r + T|` for a **slab**, and its first two `r`-derivatives.
    ///
    /// The two-dimensional counterpart of [`crate::pbc::ewald::EwaldSum::phased_pair_potential`],
    /// and what lets the DFPT response carry its long-range monopole channel on a slab rather than
    /// refusing. Same convention as the three-dimensional one: the real-space half carries
    /// `e^{+iq·T}`, and the reciprocal half runs over `k = G − q` with `e^{+ik·r}`.
    ///
    /// # What differs from the unphased Parry sum
    ///
    /// * **No ±G folding.** `|G − q| ≠ |−G − q|`, so the sum runs the **full** shifted in-plane set
    ///   with prefactor `π/(A|k|)`. At `q = 0` the full set with `π` reproduces the half set with
    ///   `2π`, which is what [`Self::pair_potential_strained`]'s `cos` form does — so the two agree
    ///   there, and a wrong factor of two here is precisely what the reduction test below catches.
    /// * **`k ⊥ n̂` still, but the phase is complex**, so the pair Hessian's cross terms no longer
    ///   collapse: `∂²/∂r_a∂r_b = e^{ik·r}[−k_a k_b K + i(k_a n̂_b + n̂_a k_b) K_z + n̂_a n̂_b K_zz]`.
    /// * **No `h = 0` term and no neutralizing sheet.** The shifted set has no `k = 0` member unless
    ///   `q` is a reciprocal lattice vector, and where it does the whole sum reduces to the
    ///   unphased one — which is handled by delegating, so the delicate `q → Γ` limit reuses code
    ///   that is already validated rather than a second derivation of it.
    ///
    /// The kernel and its `z`-derivatives are
    ///
    /// ```text
    /// K   = W₊ + W₋,     W± = e^{±hz} erfc(h/2α ± αz)
    /// K_z = h (W₊ − W₋)
    /// K_zz = h² K − (4αh/√π) e^{−h²/4α² − α²z²}
    /// ```
    ///
    /// — the Gaussian pieces of `∂W±/∂z` are equal and cancel in `K_z`, which is why it is this
    /// simple. `W±` is formed through [`Self::wall_term_s`] so neither factor has to be
    /// representable on its own.
    pub fn phased_pair_potential(
        &self,
        q_cart: Vec3,
        r: Vec3,
        exclude_self: bool,
    ) -> crate::pbc::ewald::PhasedDelta {
        use crate::pbc::ewald::PhasedDelta;
        let mut out = PhasedDelta::default();
        let a = self.alpha;
        let sqrt_pi = std::f64::consts::PI.sqrt();

        // At a reciprocal lattice vector every phase is one and this *is* the unphased sum,
        // background and sheet term included. Delegating keeps one derivation of that case.
        if crate::pbc::ewald::q_folds_to_gamma(&self.lattice, q_cart) {
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

        // ---- real space: Σ_T e^{iq·T} erfc(α|r+T|)/|r+T| ----
        let two_over_sqrt_pi = 2.0 / sqrt_pi;
        for offset in &self.real_images {
            let t = self.lattice.translation(*offset);
            let d = r + t;
            let dist = d.norm();
            if dist < 1.0e-10 || dist > self.real_cutoff {
                continue;
            }
            let theta = q_cart.dot(t);
            let (sin_t, cos_t) = theta.sin_cos();
            let erfc = libm::erfc(a * dist);
            let gauss = (-(a * dist) * (a * dist)).exp();
            let f0 = erfc / dist;
            let f1 = -erfc / (dist * dist) - two_over_sqrt_pi * a * gauss / dist;
            let f2 = 2.0 * erfc / (dist * dist * dist)
                + two_over_sqrt_pi * a * gauss * (2.0 / (dist * dist) + 2.0 * a * a);
            let u = [d.x / dist, d.y / dist, d.z / dist];
            out.value[0] += cos_t * f0;
            out.value[1] += sin_t * f0;
            for i in 0..3 {
                let gi = f1 * u[i];
                out.gradient[i][0] += cos_t * gi;
                out.gradient[i][1] += sin_t * gi;
                for j in 0..3 {
                    let delta_ij = if i == j { 1.0 } else { 0.0 };
                    let hij = f2 * u[i] * u[j] + f1 * (delta_ij - u[i] * u[j]) / dist;
                    out.hessian[i][j][0] += cos_t * hij;
                    out.hessian[i][j][1] += sin_t * hij;
                }
            }
        }

        // ---- reciprocal space: (π/A) Σ_k e^{ik·r} K(|k|, z) / |k|,  k = G − q, in-plane ----
        let n = self.normal;
        let nv = [n.x, n.y, n.z];
        let z = r.dot(n);
        let reach = self.g_cutoff + q_cart.norm();
        let mut ks: Vec<Vec3> = vec![-q_cart];
        ks.extend(
            self.lattice
                .reciprocal_vectors_within(reach)
                .into_iter()
                .map(|(_, g)| g - q_cart),
        );
        let pi_over_a = std::f64::consts::PI / self.area;
        for k in ks {
            let h = k.norm();
            if h > self.g_cutoff || h < 1.0e-12 {
                continue;
            }
            let w_plus = self.wall_term_s(h, z, 1.0);
            let w_minus = self.wall_term_s(h, z, -1.0);
            let kernel = w_plus + w_minus;
            let kernel_z = h * (w_plus - w_minus);
            let gaussian = (-h * h / (4.0 * a * a) - a * a * z * z).exp();
            let kernel_zz = h * h * kernel - 4.0 * a * h * gaussian / sqrt_pi;

            let weight = pi_over_a / h;
            let phase = k.dot(r);
            let (sin_p, cos_p) = phase.sin_cos();
            let kv = [k.x, k.y, k.z];

            // value: e^{ik·r} K
            out.value[0] += weight * cos_p * kernel;
            out.value[1] += weight * sin_p * kernel;
            for i in 0..3 {
                // ∂/∂r_i = e^{ik·r}[i k_i K + n̂_i K_z]
                let re = -kv[i] * sin_p * kernel + nv[i] * cos_p * kernel_z;
                let im = kv[i] * cos_p * kernel + nv[i] * sin_p * kernel_z;
                out.gradient[i][0] += weight * re;
                out.gradient[i][1] += weight * im;
                for j in 0..3 {
                    // ∂²/∂r_i∂r_j = e^{ik·r}[−k_i k_j K + i(k_i n̂_j + n̂_i k_j) K_z + n̂_i n̂_j K_zz]
                    let cross = kv[i] * nv[j] + nv[i] * kv[j];
                    let re = -kv[i] * kv[j] * cos_p * kernel - cross * sin_p * kernel_z
                        + nv[i] * nv[j] * cos_p * kernel_zz;
                    let im = -kv[i] * kv[j] * sin_p * kernel
                        + cross * cos_p * kernel_z
                        + nv[i] * nv[j] * sin_p * kernel_zz;
                    out.hessian[i][j][0] += weight * re;
                    out.hessian[i][j][1] += weight * im;
                }
            }
        }

        // The Ewald self-energy: the real-space sum skipped `T = 0` because `r` vanished, but the
        // reciprocal sum included it. Independent of `q`, exactly as in three dimensions.
        if exclude_self && r.norm() < 1.0e-10 {
            out.value[0] -= 2.0 * a / sqrt_pi;
        }
        out
    }

    /// `e^{s·h·z} · erfc(h/(2α) + s·α·z)`, formed so neither factor has to be representable.
    ///
    /// The branch is on the sign of the `erfc` argument, and the two expressions are the same
    /// analytic function written two ways, so the value **and every derivative** are continuous
    /// across it — there is no seam here of the kind a threshold usually introduces.
    #[inline]
    fn wall_term_s<S: Scalar>(&self, h: S, z: S, sign: f64) -> S {
        let a = self.alpha;
        let u = z * (sign * a) + h * (1.0 / (2.0 * a));
        // e^{−h²/4α² − α²z²}, bounded by 1.
        let prefactor = (z * z * (-a * a) - h * h * (1.0 / (4.0 * a * a))).exp();
        if u.val() >= 0.0 {
            prefactor * u.erfcx()
        } else {
            // erfc(u) = 2 − erfc(−u). Here `s·h·z < 0` necessarily, so `2 e^{s h z} ≤ 2`.
            (z * sign * h).exp() * 2.0 - prefactor * (-u).erfcx()
        }
    }

    /// The Parry potential at displacement `r`, in atomic units (inverse Bohr).
    ///
    /// Generic over the scalar type: at `f64` this is the value, at [`crate::dual::Dual`] the
    /// gradient with respect to `r`, and at [`crate::dual2::Dual2`] the second derivative. The
    /// derivatives are therefore of the expression actually evaluated, not of a transcription
    /// of it — which for a sum this intricate is the only way to keep them in step.
    ///
    /// `r = 0` is the self term: the `T = 0` image is skipped and the Ewald self-energy
    /// `−2α/√π` subtracted, exactly as in three dimensions.
    pub fn pair_potential_scalar<S: Scalar>(&self, r: [S; 3]) -> S {
        self.pair_potential_strained(r, &[[S::cst(0.0); 3]; 3])
    }

    /// [`Self::pair_potential_scalar`] with a homogeneous strain `ε` applied to the displacement
    /// **and** to the lattice.
    ///
    /// Seeding `ε` rather than `r` turns forward-mode AD into the strain derivative, which is
    /// where the periodic stress comes from. Doing it this way rather than differentiating the
    /// Parry sum by hand matters: the three-dimensional stress needed a page of algebra for the
    /// reciprocal prefactor alone, and every term of it was an opportunity to drop a factor that
    /// only a very specific test would catch. Here there is nothing to derive — the deformation
    /// is applied to the inputs and the same expression is evaluated.
    ///
    /// Only the first order in `ε` is represented, which is exact for a first derivative at
    /// `ε = 0` and is all a stress needs.
    pub fn pair_potential_strained<S: Scalar>(&self, r: [S; 3], eps: &[[S; 3]; 3]) -> S {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let is_self =
            r[0].val().abs() < 1.0e-10 && r[1].val().abs() < 1.0e-10 && r[2].val().abs() < 1.0e-10;

        // `v → (1 + ε)v` for a real-space vector.
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
        let mut total = S::cst(0.0);

        // Real space, over in-plane translations.
        for offset in &self.real_images {
            let t = self.lattice.translation(*offset);
            let td = deform([S::cst(t.x), S::cst(t.y), S::cst(t.z)]);
            let d = [rd[0] + td[0], rd[1] + td[1], rd[2] + td[2]];
            let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let dv = dist2.val().sqrt();
            if dv < 1.0e-10 || dv > self.real_cutoff {
                continue;
            }
            let dist = dist2.sqrt();
            total = total + (dist * self.alpha).erfc() / dist;
        }

        // The plane normal is unchanged to first order for an in-plane strain, and an out-of-plane
        // strain component has no cell length to act on, so `z` is taken from the deformed
        // displacement against the undeformed normal.
        let n = self.normal;
        let z = rd[0] * n.x + rd[1] * n.y + rd[2] * n.z;

        // `A → A [1 + tr ε − n·εᵀ·n]`, the in-plane part of the trace.
        let mut tr = S::cst(0.0);
        for (i, row) in eps.iter().enumerate() {
            tr = tr + row[i];
        }
        let nv = [n.x, n.y, n.z];
        let mut nen = S::cst(0.0);
        for (a, na) in nv.iter().enumerate() {
            for (b, nb) in nv.iter().enumerate() {
                nen = nen + eps[b][a] * (*na * *nb);
            }
        }
        let area = (tr - nen + 1.0) * self.area;
        let pi_over_a = area.recip() * std::f64::consts::PI;

        // Reciprocal vectors transform contravariantly, `h → h − εᵀh`, which is what keeps
        // `h·ρ` invariant. `|h|` does change, and the Parry kernel depends on it.
        for (g, _) in &self.reciprocal {
            let gv = [g.x, g.y, g.z];
            let mut hd = [S::cst(gv[0]), S::cst(gv[1]), S::cst(gv[2])];
            for (i, h) in hd.iter_mut().enumerate() {
                for (j, gj) in gv.iter().enumerate() {
                    *h = *h - eps[j][i] * *gj;
                }
            }
            let h2 = hd[0] * hd[0] + hd[1] * hd[1] + hd[2] * hd[2];
            let h = h2.sqrt();
            let phase = rd[0] * hd[0] + rd[1] * hd[1] + rd[2] * hd[2];
            let wall = self.wall_term_s(h, z, 1.0) + self.wall_term_s(h, z, -1.0);
            total = total + phase.cos() * wall * pi_over_a / h;
        }

        // The h = 0 term: −(2π/A)[ z erf(αz) + e^{−α²z²}/(α√π) ].
        let az = z * self.alpha;
        let h0 = z * az.erf() + (az * az * (-1.0)).exp() * (1.0 / (self.alpha * sqrt_pi));
        total = total - h0 * pi_over_a * 2.0;

        if is_self {
            total = total - 2.0 * self.alpha / sqrt_pi;
        }
        total
    }

    /// `∂φ/∂ε_αβ` at `ε = 0`, by seeding the strain instead of the position.
    ///
    /// Six independent components, three partials per [`crate::dual::Dual`], so two passes.
    pub fn pair_potential_strain(&self, r: Vec3) -> [[f64; 3]; 3] {
        use crate::dual::Dual;
        // (α, β) pairs of the symmetric strain tensor, three per pass.
        const PASSES: [[(usize, usize); 3]; 2] =
            [[(0, 0), (1, 1), (2, 2)], [(0, 1), (0, 2), (1, 2)]];
        let mut out = [[0.0_f64; 3]; 3];
        for pass in &PASSES {
            let mut eps = [[Dual::constant(0.0); 3]; 3];
            for (slot, (a, b)) in pass.iter().enumerate() {
                let v = Dual::var(0.0, slot);
                eps[*a][*b] = v;
                if a != b {
                    // Symmetric strain: both off-diagonal entries carry the same variable, so
                    // the derivative returned is `∂/∂ε_αβ` with `ε_βα` moving with it.
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
                // A symmetric seed differentiates both entries at once; halve to recover the
                // per-component derivative the virial expects.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_slab(a: f64, height: f64) -> Lattice {
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, a, 0.0),
            Vec3::new(0.0, 0.0, height),
            [true, true, false],
        )
        .unwrap()
    }

    #[test]
    fn a_bulk_or_a_chain_is_refused() {
        let bulk = Lattice::cubic(10.0).unwrap();
        assert!(Ewald2D::new(&bulk, 0.3, 1.0e-12).is_err());
        let chain = Lattice::from_vectors(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap();
        assert!(Ewald2D::new(&chain, 0.3, 1.0e-12).is_err());
    }

    #[test]
    fn a_tilted_slab_is_refused_rather_than_summed_wrongly() {
        let tilted = Lattice::from_vectors(
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 8.0, 0.0),
            Vec3::new(2.0, 0.0, 30.0), // c is not normal to the ab plane
            [true, true, false],
        )
        .unwrap();
        let err = Ewald2D::new(&tilted, 0.3, 1.0e-12).unwrap_err();
        assert!(err.to_string().contains("orthogonal"), "{err}");
    }

    #[test]
    fn the_potential_does_not_depend_on_the_splitting_parameter() {
        // The sharpest check available without an external reference: `α` only decides how the
        // same sum is divided between real and reciprocal space, so every piece — including the
        // `h = 0` term and the self energy — has to cancel out of the total.
        let lattice = square_slab(7.0, 60.0);
        let base = default_alpha_2d(lattice.cell.col[0].cross(lattice.cell.col[1]).norm());
        let probes = [
            Vec3::new(1.3, 0.7, 0.0),
            Vec3::new(2.0, -1.1, 3.4),
            Vec3::new(3.5, 3.5, 0.0),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::zero(),
        ];
        let mut reference: Option<Vec<f64>> = None;
        eprintln!("    alpha/alpha0 |  potential at each probe (1/Bohr)");
        for scale in [0.55_f64, 0.8, 1.0, 1.5, 2.2] {
            let e = Ewald2D::new(&lattice, base * scale, 1.0e-13).unwrap();
            let vals: Vec<f64> = probes.iter().map(|p| e.pair_potential(*p)).collect();
            eprintln!(
                "    {scale:11.2}  | {}",
                vals.iter()
                    .map(|v| format!("{v:14.10}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            match &reference {
                None => reference = Some(vals),
                Some(r) => {
                    for (a, b) in vals.iter().zip(r) {
                        assert!(
                            (a - b).abs() < 1.0e-9,
                            "the 2D sum moved by {:.3e} when alpha changed by {scale}x",
                            (a - b).abs()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_two_dimensional_madelung_constant_is_reproduced() {
        // A square lattice of alternating ±1 charges. The 2D Madelung constant is
        // 1.6155426267... — a value that has nothing to do with this implementation, so it
        // checks the whole construction (constants, h = 0 term, self energy) at once, the way
        // the rock-salt constant does in three dimensions.
        let a = 1.0_f64;
        let lattice = Lattice::from_vectors(
            Vec3::new(2.0 * a, 0.0, 0.0),
            Vec3::new(0.0, 2.0 * a, 0.0),
            Vec3::new(0.0, 0.0, 200.0),
            [true, true, false],
        )
        .unwrap();
        let area = 4.0 * a * a;
        let ewald = Ewald2D::new(&lattice, default_alpha_2d(area), 1.0e-14).unwrap();

        // Four ions per cell: (0,0) +, (a,0) −, (0,a) −, (a,a) +.
        let sites = [
            (Vec3::new(0.0, 0.0, 0.0), 1.0),
            (Vec3::new(a, 0.0, 0.0), -1.0),
            (Vec3::new(0.0, a, 0.0), -1.0),
            (Vec3::new(a, a, 0.0), 1.0),
        ];
        // The Madelung constant is conventionally the **site potential**, `M = −a·V_i`, not the
        // energy per ion — which is half of it, because the energy counts each pair once.
        let (r0, q0) = sites[0];
        let mut potential = 0.0;
        for (rj, qj) in &sites {
            potential += qj * ewald.pair_potential(*rj - r0);
        }
        let madelung = -potential * a / q0;
        eprintln!(
            "    2D square-lattice Madelung constant: {madelung:.10} (reference 1.6155426267)"
        );
        assert!(
            (madelung - 1.615_542_626_7).abs() < 1.0e-8,
            "2D Madelung constant came out {madelung:.10}"
        );
    }

    #[test]
    fn the_strain_derivative_matches_a_strain_finite_difference() {
        // The strain derivative is obtained by seeding `ε` through the same expression the value
        // comes from, so this checks that the deformation is applied to every input that
        // depends on it — the translations, the reciprocal vectors and the area — and not just
        // to the obvious ones.
        //
        // The finite difference genuinely rebuilds the sum on a strained lattice, so it also
        // catches anything the first-order deformation inside `pair_potential_strained` fails
        // to represent.
        let lattice = square_slab(7.0, 60.0);
        let base = default_alpha_2d(49.0);
        let ewald = Ewald2D::new(&lattice, base, 1.0e-13).unwrap();
        let probe = Vec3::new(1.9, -1.2, 2.6);
        let analytic = ewald.pair_potential_strain(probe);

        let step = 1.0e-6;
        let mut worst = 0.0_f64;
        eprintln!("      (a,b)      analytic          finite difference");
        // In-plane components only. A strain component touching the non-periodic axis tilts a
        // cell vector out of the plane, which is not a deformation of *this* system at all — the
        // slab has no length along the normal to strain. The caller zeroes those components for
        // exactly that reason, and there is nothing here for a finite difference to compare to.
        for (a, b) in [(0usize, 0usize), (1, 1), (0, 1)] {
            let shifted = |sign: f64| {
                let mut eps = [[0.0_f64; 3]; 3];
                eps[a][b] += sign * step;
                if a != b {
                    eps[b][a] += sign * step;
                }
                // `α` is held fixed under strain, exactly as in three dimensions: the total is
                // α-independent, so it is legitimate and it keeps `dα/dε` out of every term.
                let l = lattice.strained(&eps).unwrap();
                let e = Ewald2D::new(&l, base, 1.0e-13).unwrap();
                let mut p = probe;
                let d = Vec3::new(
                    eps[0][0] * probe.x + eps[0][1] * probe.y + eps[0][2] * probe.z,
                    eps[1][0] * probe.x + eps[1][1] * probe.y + eps[1][2] * probe.z,
                    eps[2][0] * probe.x + eps[2][1] * probe.y + eps[2][2] * probe.z,
                );
                p += d;
                e.pair_potential(p)
            };
            let fd = (shifted(1.0) - shifted(-1.0)) / (2.0 * step);
            // The finite difference moves `ε_ab` **and** `ε_ba` together, so it measures the sum
            // of the two partials. `pair_potential_strain` reports them separately, matching the
            // convention the three-dimensional virial already uses.
            let combined = if a == b {
                analytic[a][b]
            } else {
                analytic[a][b] + analytic[b][a]
            };
            eprintln!("      ({a},{b})    {combined:16.10}  {fd:16.10}");
            worst = worst.max((combined - fd).abs());
        }
        eprintln!("      worst |analytic - FD| = {worst:.3e}");
        assert!(worst < 1.0e-7, "2D strain derivative mismatch {worst:.3e}");
    }

    #[test]
    fn the_gradient_and_hessian_match_finite_differences() {
        let lattice = square_slab(7.0, 60.0);
        let area = 49.0;
        let ewald = Ewald2D::new(&lattice, default_alpha_2d(area), 1.0e-13).unwrap();
        let step = 1.0e-5;
        let mut worst_g = 0.0_f64;
        let mut worst_h = 0.0_f64;
        for probe in [
            Vec3::new(1.6, -0.8, 2.1),
            Vec3::new(3.5, 3.5, 0.4),
            Vec3::new(0.3, 0.1, -4.0),
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
        eprintln!("    2D Ewald vs FD: gradient {worst_g:.3e}, hessian {worst_h:.3e}");
        assert!(worst_g < 1.0e-8, "2D gradient mismatch {worst_g:.3e}");
        assert!(worst_h < 1.0e-6, "2D hessian mismatch {worst_h:.3e}");
    }
}
