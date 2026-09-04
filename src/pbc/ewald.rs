// SPDX-License-Identifier: GPL-3.0-or-later

//! Ewald summation of the **monopole** channel, with a neutralizing background.
//!
//! # What this fixes, and what it does not
//!
//! The NDDO electrostatics is three `1/R`-like pieces that cancel for a neutral cell: the
//! electron–core attraction, the electron–electron Coulomb, and the core–core repulsion. Their
//! monopole parts combine to `Q_a Q_b γ_ab(R)` with the Klopman–Ohno kernel
//! `γ_η(R) = e²/√(R² + η²)`, `η = ρ⁰_a + ρ⁰_b`, and the *same* `γ` in all three — which is why
//! they cancel exactly when `Σ_a Q_a = 0`.
//!
//! When the cell carries a net charge they do not cancel, and `Σ_T Q²/|T|` diverges. Measured
//! before this module existed: a +1 water cell in an 8 Å cube gave −331 eV at a 20 Bohr
//! real-space cutoff and **+72 eV** at 130 Bohr, tracking the missing jellium term
//! `π Q² r_c²/V` to 1.2 %. That is what Ewald summation fixes.
//!
//! ## The split, and why `η` drops out of it
//!
//! Write the lattice sum as the `1/R` part plus what is left:
//!
//! ```text
//! Σ_T γ_η(|R+T|) = Σ_T 1/|R+T|  +  Σ_T [γ_η(|R+T|) − 1/|R+T|]
//! ```
//!
//! The first sum goes to Ewald. The second decays as `−η²/2R³` and stays in real space,
//! truncated at the same cutoff the pair list already uses. So the correction this module
//! computes is
//!
//! ```text
//! Δ_ab = φ^Ewald(R_ab) − Σ_{T ∈ pair list} 1/|R_ab + T|
//! ```
//!
//! and **`η` does not appear**: it cancels between what the pair list already summed and what
//! is being replaced. One reciprocal-space sum therefore serves every element pair, rather than
//! one per `(element, element)` combination.
//!
//! ## What remains divergent, honestly
//!
//! The `R⁻³` residual left in real space has a logarithmically divergent lattice sum in three
//! dimensions, and neutrality does **not** rescue it. Expanding `η_ab² = ρ_a² + 2ρ_aρ_b + ρ_b²`
//! and using `Σ_a Q_a = 0`:
//!
//! ```text
//! Σ_ab Q_a Q_b η_ab² = 2 (Σ_a Q_a ρ_a)²
//! ```
//!
//! which is a perfect square and generally non-zero. So a residual `ln(r_c)` dependence survives
//! — measured at roughly 3 × 10⁻⁴ eV per doubling of the cutoff on a water chain, against the
//! 400 eV this module removes. Removing it too would need the generalized Ewald machinery for an
//! `R⁻³` kernel, with a `G = 0` logarithmic term and its matching self-energy. That is not
//! implemented, and `docs/pbc.md` says so.
//!
//! # Dimensionality
//!
//! **Three-dimensional cells only.** For a slab or a chain the reciprocal sum below is simply
//! wrong — it assumes periodicity in all three directions — and a charged one is not even well
//! defined without an explicit convention: the compensating background for a charged slab is a
//! sheet whose energy depends on where it is placed, and the potential of a charged line
//! diverges logarithmically. [`EwaldSum::new`] refuses anything but 3D rather than returning a
//! number that looks like an answer.

use crate::constants::AM1_EV;
use crate::error::{Am1Error, Result};
use crate::lattice::{ImageOffset, Lattice};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::neighbors::NeighborList;
use crate::params::Am1Parameters;
use crate::system::Molecule;

/// A prepared 3D Ewald sum for one lattice.
#[derive(Clone, Debug)]
pub struct EwaldSum {
    lattice: Lattice,
    /// Splitting parameter, Bohr⁻¹.
    pub alpha: f64,
    /// Real-space translations kept, chosen so `erfc(α r)` is negligible beyond them.
    real_images: Vec<ImageOffset>,
    /// Separation beyond which `erfc(α r)/r` is below the requested accuracy.
    real_cutoff: f64,
    /// Reciprocal vectors kept, with `4π e^{−G²/4α²} / (V G²)` precomputed.
    reciprocal: Vec<(Vec3, f64)>,
    /// Radius of the reciprocal shell, so the phased sum can re-enumerate it centred on `q`.
    g_cutoff: f64,
    volume: f64,
}

/// Default splitting parameter for a cell of volume `V`.
///
/// `α = √π / V^{1/3}` balances the real and reciprocal sums for a roughly cubic cell. The final
/// answer must not depend on it — that is the sharpest test this module has, and
/// `tests/pbc_ewald.rs` asserts it over a factor of three in `α`.
pub fn default_alpha(volume: f64) -> f64 {
    std::f64::consts::PI.sqrt() / volume.cbrt()
}

/// One atom-pair block of the long-range correction's second derivative, tagged with the pair.
///
/// `(c, d, block)` means `block` is the `3x3` second derivative with respect to atoms `c`
/// and `d`; only `c <= d` is produced, and the caller mirrors it.
pub type HessianBlock = (usize, usize, [[f64; 3]; 3]);

impl EwaldSum {
    /// Prepare the sum. `accuracy` is the target relative precision (1e-12 is a good default).
    pub fn new(lattice: &Lattice, alpha: f64, accuracy: f64) -> Result<Self> {
        if !lattice.is_fully_periodic() {
            return Err(Am1Error::InvalidInput(
                "Ewald summation here is three-dimensional; a slab or a chain needs a different \
                 reciprocal sum, and a charged one needs an explicit convention for the \
                 compensating background. See docs/pbc.md."
                    .into(),
            ));
        }
        // Written to reject NaN as well as a non-positive value: `alpha <= 0.0` would let a NaN
        // through, and a NaN splitting parameter turns every energy into a NaN much later.
        if alpha.is_nan() || alpha <= 0.0 {
            return Err(Am1Error::InvalidInput(
                "the Ewald splitting parameter must be positive".into(),
            ));
        }
        let volume = lattice.volume();
        // `erfc(x) < e^{-x²}`, so `α r > √(−ln accuracy)` makes the real-space tail negligible.
        let tol = (-accuracy.ln()).sqrt();
        let real_cutoff = tol / alpha;
        // Likewise `e^{−G²/4α²} < accuracy` for `G > 2α√(−ln accuracy)`. The margin is for the
        // *number* of terms: a shell at that radius holds thousands, and each being individually
        // small does not make their sum negligible.
        let g_cutoff = 2.8 * alpha * tol;

        // The translation list is built with a margin of one cell diagonal, and the real cutoff
        // is then applied to `|r + T|` inside the sum rather than to `|T|` here.
        //
        // This distinction is a real bug if it is got wrong, not bookkeeping. The `erfc` decays
        // in the *separation* `|r + T|`, not in the translation length, so for a pair near
        // opposite corners of the cell the translations that bring the images closest are not
        // the shortest ones. Truncating on `|T|` therefore keeps some members of a set of
        // equidistant images and drops others — for a probe at the body centre of a cubic cell,
        // four of the eight nearest images survive and four do not, and the survivors do not
        // cancel against anything.
        let diagonal = (lattice.cell.col[0] + lattice.cell.col[1] + lattice.cell.col[2]).norm();
        let real_images = lattice.image_offsets(real_cutoff + diagonal);
        let four_pi = 4.0 * std::f64::consts::PI;
        let reciprocal = lattice
            .reciprocal_vectors_within(g_cutoff)
            .into_iter()
            .map(|(_, g)| {
                let g2 = g.norm2();
                (
                    g,
                    four_pi * (-g2 / (4.0 * alpha * alpha)).exp() / (volume * g2),
                )
            })
            .collect();

        Ok(Self {
            lattice: *lattice,
            alpha,
            real_images,
            real_cutoff,
            reciprocal,
            g_cutoff,
            volume,
        })
    }

    /// The lattice-summed Coulomb potential between two unit charges separated by `r`, with a
    /// neutralizing background.
    ///
    /// For `r = 0` this is the potential an atom feels from **its own images** — the `T = 0`
    /// term is excluded and replaced by the Ewald self-energy `−2α/√π`.
    pub fn pair_potential(&self, r: Vec3) -> f64 {
        let is_self = r.norm() < 1.0e-10;
        let mut total = 0.0;

        // Real space: Σ_T erfc(α|r+T|)/|r+T|, skipping T = 0 when r = 0.
        for offset in &self.real_images {
            let d = r + self.lattice.translation(*offset);
            let dist = d.norm();
            if dist < 1.0e-10 || dist > self.real_cutoff {
                continue;
            }
            total += libm::erfc(self.alpha * dist) / dist;
        }

        // Reciprocal space: (4π/V) Σ_{G≠0} e^{−G²/4α²} cos(G·r) / G².
        for (g, weight) in &self.reciprocal {
            total += weight * g.dot(r).cos();
        }

        // The G = 0 term of the reciprocal sum, i.e. the neutralizing background. It is what
        // makes a charged cell finite, and it is the whole reason this module exists.
        total -= std::f64::consts::PI / (self.alpha * self.alpha * self.volume);

        // Self-interaction correction, for the atom-with-its-own-images case only.
        if is_self {
            total -= 2.0 * self.alpha / std::f64::consts::PI.sqrt();
        }
        total
    }

    /// Gradient of [`Self::pair_potential`] with respect to `r`.
    pub fn pair_potential_gradient(&self, r: Vec3) -> Vec3 {
        let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
        let mut grad = Vec3::zero();

        for offset in &self.real_images {
            let d = r + self.lattice.translation(*offset);
            let dist = d.norm();
            if dist < 1.0e-10 || dist > self.real_cutoff {
                continue;
            }
            // d/dr [erfc(αr)/r] = −[erfc(αr)/r² + (2α/√π) e^{−α²r²}/r]
            let scale = -(libm::erfc(self.alpha * dist) / (dist * dist)
                + two_over_sqrt_pi * self.alpha * (-(self.alpha * dist).powi(2)).exp() / dist);
            grad += d * (scale / dist);
        }

        for (g, weight) in &self.reciprocal {
            grad += *g * (-weight * g.dot(r).sin());
        }
        grad
    }

    /// Second derivative of [`Self::pair_potential`] with respect to `r`, as a 3×3 block.
    ///
    /// Written out rather than obtained by AD because [`crate::dual::Scalar`] has no `erfc`. It
    /// is therefore checked against a finite difference of [`Self::pair_potential_gradient`] in
    /// this module's tests, which is the only thing that makes a hand-written derivative
    /// trustworthy.
    ///
    /// For `f(d) = erfc(αd)/d`:
    ///
    /// ```text
    /// f'(d)  = −[ erfc(αd)/d² + (2α/√π) e^{−α²d²}/d ]
    /// f''(d) = 2 erfc(αd)/d³ + (2α/√π) e^{−α²d²} (2/d² + 2α²)
    /// ```
    ///
    /// and `∂²f/∂r_i∂r_j = f''(d) d̂_i d̂_j + f'(d) (δ_ij − d̂_i d̂_j)/d`.
    pub fn pair_potential_hessian(&self, r: Vec3) -> [[f64; 3]; 3] {
        let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
        let mut h = [[0.0_f64; 3]; 3];

        for offset in &self.real_images {
            let d = r + self.lattice.translation(*offset);
            let dist = d.norm();
            if dist < 1.0e-10 || dist > self.real_cutoff {
                continue;
            }
            let a = self.alpha;
            let erfc = libm::erfc(a * dist);
            let gauss = (-(a * dist).powi(2)).exp();
            let f1 = -(erfc / (dist * dist) + two_over_sqrt_pi * a * gauss / dist);
            let f2 = 2.0 * erfc / (dist * dist * dist)
                + two_over_sqrt_pi * a * gauss * (2.0 / (dist * dist) + 2.0 * a * a);
            let u = [d.x / dist, d.y / dist, d.z / dist];
            for i in 0..3 {
                for j in 0..3 {
                    let delta_ij = if i == j { 1.0 } else { 0.0 };
                    h[i][j] += f2 * u[i] * u[j] + f1 * (delta_ij - u[i] * u[j]) / dist;
                }
            }
        }

        // ∂²/∂r_i∂r_j [w cos(G·r)] = −w G_i G_j cos(G·r).
        for (g, weight) in &self.reciprocal {
            let c = -weight * g.dot(r).cos();
            let gv = [g.x, g.y, g.z];
            for i in 0..3 {
                for j in 0..3 {
                    h[i][j] += c * gv[i] * gv[j];
                }
            }
        }
        h
    }
}

/// A complex scalar, `[real, imaginary]` — the same packing [`crate::pbc::complex`] uses.
type C = [f64; 2];

/// Whether `q` is a reciprocal lattice vector of `lattice`, so every `e^{iq·T}` is exactly one.
///
/// Tested on the fractional coordinates `f_i = q·a_i/2π` rather than by searching the reciprocal
/// lattice: exact in intent, and it costs three dot products. Only the **periodic** axes are
/// tested — a slab's translations never leave the plane, so a `q` component along the normal is
/// never read by any phase and cannot make the sum phased.
///
/// Shared by all three dimensionalities, because "does this `q` fold to Γ" has to mean the same
/// thing in each: it is what decides whether the neutralizing background belongs in the sum, and a
/// 2D or 1D kernel answering it differently from the 3D one would put the background in the wrong
/// place for exactly the `q` a supercell comparison uses.
pub fn q_folds_to_gamma(lattice: &Lattice, q_cart: Vec3) -> bool {
    let two_pi = 2.0 * std::f64::consts::PI;
    (0..3).filter(|&i| lattice.periodic[i]).all(|i| {
        let f = q_cart.dot(lattice.cell.col[i]) / two_pi;
        (f - f.round()).abs() < 1.0e-9
    })
}

/// The phased lattice sum and its first two `r`-derivatives, all complex.
///
/// Units follow [`delta_gradient`]: eV, eV/Bohr and eV/Bohr² once the `e²` factor is applied.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhasedDelta {
    pub value: C,
    /// `∂/∂r_i`.
    pub gradient: [C; 3],
    /// `∂²/∂r_i∂r_j`.
    pub hessian: [[C; 3]; 3],
}

impl EwaldSum {
    /// `S(q; r) = Σ_T e^{iq·T} / |r + T|` and its `r`-derivatives, by Ewald splitting.
    ///
    /// # The split
    ///
    /// With `f(x) = erf(α|x|)/|x|` and Poisson summation carrying the phase,
    ///
    /// ```text
    /// Σ_T e^{iq·T} f(r+T) = (4π/V) Σ_G e^{−|k|²/4α²} e^{ik·r} / |k|²,     k = G − q
    /// ```
    ///
    /// so the reciprocal sum runs over the lattice **shifted by `−q`**.
    ///
    /// # Which element is dropped, and why it is exactly `k = 0`
    ///
    /// Only the term with `k = 0`, which occurs when — and only when — `q` is itself a reciprocal
    /// lattice vector, i.e. when `q` folds to Γ. There it is the divergent term the neutralizing
    /// background cancels, and dropping it reproduces this module's tin-foil `Σ_{G≠0}` exactly.
    ///
    /// Dropping instead "the `G = 0` element" — the long-wavelength term, `k = −q` — is the
    /// tempting alternative, because it is what keeps the direction-dependent behaviour out of
    /// `D(q)` so a post-hoc LO–TO term can supply it. It was tried and rejected: that rule is
    /// **not periodic in `q`** (shifting `q` by a reciprocal vector changes which element is
    /// dropped) and it has no well-defined answer at a zone boundary, where several `k` tie for
    /// smallest and dropping one of them breaks the crystal symmetry of `D(q)`. `k = 0` has
    /// neither problem: it is a periodic condition, and it never ties.
    ///
    /// The consequence is that `D(q)` here is the **full** dynamical matrix at finite `q`,
    /// long-range monopole channel included, so its `q → 0` limit is direction dependent — which
    /// is the physics. It must therefore **not** be combined with `frequencies_with_lo_to`, whose
    /// job is to restore that same physics to the *supercell* route, where a truncated `Φ(T)`
    /// structurally cannot carry it. See `docs/pbc.md`.
    ///
    /// The `−π/(α²V)` background that [`Self::pair_potential`] carries is the regular remainder of
    /// the `k = 0` element and is included here on the same terms: at `q ≡ 0` only. Being a
    /// constant in `r` it cannot affect either derivative.
    ///
    /// `exclude_self` drops the `T = 0` image, for the case of an atom with its own images.
    ///
    /// # The shell is centred on `q`, not on the origin
    ///
    /// The Gaussian decays in `|G − q|`, so the shell that truncates it has to be centred there
    /// too. Reusing the precomputed `|G| ≤ g_cutoff` shell instead makes the sum depend on which
    /// representative of `q` was passed — measured at 12 eV between `q` and `q + G`, against the
    /// `10⁻⁸` the periodicity test now holds to.
    /// Whether `q` is a reciprocal lattice vector, i.e. whether it folds to Γ.
    ///
    /// Tested on the fractional coordinates `f_i = q·a_i/2π` rather than by searching the
    /// reciprocal lattice: exact-in-intent, and it costs three dot products.
    fn q_folds_to_gamma(&self, q_cart: Vec3) -> bool {
        q_folds_to_gamma(&self.lattice, q_cart)
    }

    pub fn phased_pair_potential(&self, q_cart: Vec3, r: Vec3, exclude_self: bool) -> PhasedDelta {
        let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
        let mut out = PhasedDelta::default();
        let a = self.alpha;

        // ---- real space: Σ_T e^{iq·T} erfc(α|r+T|)/|r+T| ----
        //
        // The phase depends on `T` and the radial functions on `|r+T|`, so each image contributes
        // its (real) derivative tensor multiplied by a complex scalar. That factorization is why
        // this can reuse the real-space algebra of `pair_potential_hessian` unchanged.
        for offset in &self.real_images {
            let t = self.lattice.translation(*offset);
            let d = r + t;
            let dist = d.norm();
            if dist < 1.0e-10 || dist > self.real_cutoff {
                continue;
            }
            if exclude_self && offset.is_origin() && r.norm() < 1.0e-10 {
                continue;
            }
            let theta = q_cart.dot(t);
            let (cos_t, sin_t) = (theta.cos(), theta.sin());

            let erfc = libm::erfc(a * dist);
            let gauss = (-(a * dist).powi(2)).exp();
            let f0 = erfc / dist;
            let f1 = -(erfc / (dist * dist) + two_over_sqrt_pi * a * gauss / dist);
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

        // ---- reciprocal space: (4π/V) Σ_{k ≠ 0} e^{−k²/4α²} e^{ik·r} / k², with k = G − q ----
        //
        // Enumerated around `q` and filtered on `|k|`, so the truncation follows the Gaussian
        // rather than the origin. The enumeration radius grows with `|q|` only because a caller
        // may hand over an unfolded `q`; for one inside the Brillouin zone it is a half-vector.
        let four_pi = 4.0 * std::f64::consts::PI;
        let reach = self.g_cutoff + q_cart.norm();
        // Whether `k = 0` belongs to the shifted set at all, i.e. whether `q` is itself a
        // reciprocal lattice vector. It cannot be discovered from the enumeration below, which
        // skips `G = 0` — so it is tested directly, on the fractional coordinates of `q`.
        let folds_to_gamma = self.q_folds_to_gamma(q_cart);
        // `reciprocal_vectors_within` skips `G = 0`, which is right for an unshifted sum and
        // wrong here: the `G = 0` element is `k = −q`, an ordinary long-wavelength term at finite
        // `q`. Leaving it to the enumeration would drop a *different physical* `k` depending on
        // which representative of `q` was passed — which is exactly how the periodicity test
        // caught it, at 12 eV between `q` and `q + G`.
        let mut ks: Vec<Vec3> = vec![-q_cart];
        ks.extend(
            self.lattice
                .reciprocal_vectors_within(reach)
                .into_iter()
                .map(|(_, g)| g - q_cart),
        );
        for k in ks {
            let k2 = k.norm2();
            if k2 > self.g_cutoff * self.g_cutoff || k2 < 1.0e-20 {
                continue;
            }
            let w = four_pi * (-k2 / (4.0 * a * a)).exp() / (self.volume * k2);
            let phase = k.dot(r);
            let (cos_p, sin_p) = (phase.cos(), phase.sin());
            out.value[0] += w * cos_p;
            out.value[1] += w * sin_p;
            let kv = [k.x, k.y, k.z];
            for i in 0..3 {
                // ∂/∂r_i e^{ik·r} = i k_i e^{ik·r}
                out.gradient[i][0] += w * (-kv[i] * sin_p);
                out.gradient[i][1] += w * (kv[i] * cos_p);
                for j in 0..3 {
                    // ∂²/∂r_i∂r_j e^{ik·r} = −k_i k_j e^{ik·r}
                    out.hessian[i][j][0] += w * (-kv[i] * kv[j] * cos_p);
                    out.hessian[i][j][1] += w * (-kv[i] * kv[j] * sin_p);
                }
            }
        }

        // The neutralizing background — the regular remainder of the `k = 0` element — on exactly
        // the terms that element is dropped: only when it was there to drop, i.e. when `q` folds
        // to Γ. A constant in `r`, so it touches the value alone.
        if folds_to_gamma {
            out.value[0] -= std::f64::consts::PI / (a * a * self.volume);
        }

        // The Ewald self-energy, for the atom-with-its-own-images case. Also a constant.
        if exclude_self && r.norm() < 1.0e-10 {
            out.value[0] -= 2.0 * a / std::f64::consts::PI.sqrt();
        }
        out
    }
}

/// `Δ(q; r)` — the phased long-range correction and its `r`-derivatives, in eV / eV·Bohr⁻ⁿ.
///
/// The `q ≠ 0` counterpart of [`delta_gradient`] and [`delta_hessian`], and defined the same way:
/// the exact phased lattice sum minus the phased truncated sum the pair list already counted.
/// `translations` must be the list the pair list was built against, for the same reason as there.
///
/// Every dimensionality since 0.2.2 — it takes a [`LongRangeKernel`] rather than an [`EwaldSum`],
/// so a slab and a chain go down the same path a crystal does.
pub fn phased_delta(
    q_cart: Vec3,
    r: Vec3,
    lattice: &Lattice,
    translations: &[ImageOffset],
    ewald: &LongRangeKernel,
    is_self: bool,
) -> PhasedDelta {
    let mut out = ewald.phased_pair_potential(q_cart, r, is_self);

    // Subtract what the pair list already summed, with the same phases it carried.
    for offset in translations {
        let t = lattice.translation(*offset);
        let d = r + t;
        let dist = d.norm();
        if dist < 1.0e-10 {
            continue;
        }
        let theta = q_cart.dot(t);
        let (cos_t, sin_t) = (theta.cos(), theta.sin());
        let inv = 1.0 / dist;
        let inv3 = inv * inv * inv;
        let u = [d.x * inv, d.y * inv, d.z * inv];

        out.value[0] -= cos_t * inv;
        out.value[1] -= sin_t * inv;
        for i in 0..3 {
            // ∇(1/d) = −d̂/d²
            let gi = -u[i] * inv * inv;
            out.gradient[i][0] -= cos_t * gi;
            out.gradient[i][1] -= sin_t * gi;
            for j in 0..3 {
                let delta_ij = if i == j { 1.0 } else { 0.0 };
                // ∂²(1/d) = (3 d̂_i d̂_j − δ_ij)/d³
                let hij = (3.0 * u[i] * u[j] - delta_ij) * inv3;
                out.hessian[i][j][0] -= cos_t * hij;
                out.hessian[i][j][1] -= sin_t * hij;
            }
        }
    }

    for c in out.value.iter_mut() {
        *c *= AM1_EV;
    }
    for g in out.gradient.iter_mut() {
        for c in g.iter_mut() {
            *c *= AM1_EV;
        }
    }
    for row in out.hessian.iter_mut() {
        for h in row.iter_mut() {
            for c in h.iter_mut() {
                *c *= AM1_EV;
            }
        }
    }
    out
}

/// Cartesian `q` from fractional coordinates, using the same `2π` convention
/// [`crate::pbc::kpoints::KPoint::phase`] does — so `e^{iq·T}` agrees between the two.
pub fn q_cartesian(lattice: &Lattice, fractional: [f64; 3]) -> Vec3 {
    let b = lattice.reciprocal_vectors_2pi();
    b[0] * fractional[0] + b[1] * fractional[1] + b[2] * fractional[2]
}

/// The long-range `1/R` lattice sum, whichever dimensionality the cell has.
///
/// The correction `Δ_ab` built on top of this has the same form in every case — the exact
/// lattice sum minus the truncated real-space sum the pair list already counted — so only the
/// kernel changes. Dispatching here rather than at every call site is what lets the SCF,
/// gradient, stress and Hessian paths stay dimension-agnostic.
///
/// Each variant has its own validation and its own limits; see [`EwaldSum`],
/// [`crate::pbc::ewald2d::Ewald2D`] and [`crate::pbc::ewald1d::Ewald1D`].
#[derive(Clone, Debug)]
pub enum LongRangeKernel {
    /// Three-dimensional Ewald summation, tin-foil boundary condition.
    Bulk(EwaldSum),
    /// Two-dimensional (Parry) summation for a slab.
    Slab(crate::pbc::ewald2d::Ewald2D),
    /// One-dimensional regularized summation for a chain.
    Chain(crate::pbc::ewald1d::Ewald1D),
}

impl LongRangeKernel {
    /// Build the kernel matching `lattice`'s dimensionality, or `None` for a molecule.
    pub fn for_lattice(lattice: &Lattice) -> Result<Option<Self>> {
        match lattice.n_periodic() {
            3 => Ok(Some(Self::Bulk(EwaldSum::new(
                lattice,
                default_alpha(lattice.volume()),
                1.0e-12,
            )?))),
            2 => {
                let area = lattice.measure();
                Ok(Some(Self::Slab(crate::pbc::ewald2d::Ewald2D::new(
                    lattice,
                    crate::pbc::ewald2d::default_alpha_2d(area),
                    1.0e-12,
                )?)))
            }
            1 => Ok(Some(Self::Chain(crate::pbc::ewald1d::Ewald1D::new(
                lattice,
                DEFAULT_CHAIN_IMAGES,
            )?))),
            _ => Ok(None),
        }
    }

    /// Lattice-summed `1/R` potential at displacement `r`, in inverse Bohr.
    pub fn pair_potential(&self, r: Vec3) -> f64 {
        match self {
            Self::Bulk(e) => e.pair_potential(r),
            Self::Slab(e) => e.pair_potential(r),
            Self::Chain(e) => e.pair_potential(r),
        }
    }

    /// `∂φ/∂r`.
    pub fn pair_potential_gradient(&self, r: Vec3) -> Vec3 {
        match self {
            Self::Bulk(e) => e.pair_potential_gradient(r),
            Self::Slab(e) => e.pair_potential_gradient(r),
            Self::Chain(e) => e.pair_potential_gradient(r),
        }
    }

    /// `∂²φ/∂r∂r`.
    pub fn pair_potential_hessian(&self, r: Vec3) -> [[f64; 3]; 3] {
        match self {
            Self::Bulk(e) => e.pair_potential_hessian(r),
            Self::Slab(e) => e.pair_potential_hessian(r),
            Self::Chain(e) => e.pair_potential_hessian(r),
        }
    }

    /// `∂φ/∂ε_αβ` under a homogeneous strain of the displacement and the lattice.
    pub fn pair_potential_strain(&self, r: Vec3) -> [[f64; 3]; 3] {
        match self {
            Self::Bulk(e) => pair_potential_strain(e, r),
            Self::Slab(e) => e.pair_potential_strain(r),
            Self::Chain(e) => e.pair_potential_strain(r),
        }
    }

    /// `S(q; r) = Σ_T e^{iq·T} / |r + T|` and its first two `r`-derivatives, in whichever
    /// dimensionality the cell has.
    ///
    /// Each variant is the `q`-shifted form of the machinery its unphased sum uses: the textbook
    /// split in 3D, Parry's slab sum over the shifted in-plane set in 2D, and the chain's direct
    /// summation phased image by image in 1D. All three share one convention — the real-space half
    /// carries `e^{+iq·T}` and the reciprocal half runs over `k = G − q` with `e^{+ik·r}` — and all
    /// three delegate to their unphased counterpart where `q` is a reciprocal lattice vector, so
    /// the neutralizing background, the sheet term and the chain's line charge each have exactly
    /// one derivation.
    ///
    /// Available in **every** dimensionality since 0.2.2. Through 0.2.1 only the 3D kernel existed,
    /// and `LongRange::Require` on a slab or a chain was an error for that reason alone.
    pub fn phased_pair_potential(&self, q_cart: Vec3, r: Vec3, exclude_self: bool) -> PhasedDelta {
        match self {
            Self::Bulk(e) => e.phased_pair_potential(q_cart, r, exclude_self),
            Self::Slab(e) => e.phased_pair_potential(q_cart, r, exclude_self),
            Self::Chain(e) => e.phased_pair_potential(q_cart, r, exclude_self),
        }
    }

    /// A short name for diagnostics and error messages.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Bulk(_) => "3D Ewald (tin-foil)",
            Self::Slab(_) => "2D Parry",
            Self::Chain(_) => "1D regularized chain sum",
        }
    }
}

/// Reference length (Bohr) at which the Klopman–Ohno `R⁻³` continuum tail is cut.
///
/// The lattice sum of `γ_η − 1/R` diverges logarithmically, so the tail has no value without a
/// stated reference — the three-dimensional counterpart of the chain's line-charge convention, and
/// declared for the same reason. See [`LongRangeMonopole::with_klopman_ohno_tail`].
///
/// What the choice does and does not affect: the `ln r_c` dependence cancels regardless, so a
/// *cutoff* sweep is convention-free. What the reference shifts is the absolute energy, by
/// `−(2π/V)(Σ_a Q_a ρ_a)² ln r_0` for a neutral cell. `1 Bohr` makes `ln r_0 = 0`, so this
/// particular choice adds **nothing** and the correction is purely the removal of the cutoff
/// dependence — which is why it is the one taken.
pub const KLOPMAN_OHNO_REFERENCE: f64 = 1.0;

/// The **Klopman–Ohno `R⁻³` tail** beyond the pair list, as a per-pair constant `ko[a][b]` in eV.
///
/// `None` when there is no cell, no volume, or no cutoff to sum beyond.
///
/// # What was missing
///
/// [`LongRangeMonopole::new`] corrects the `1/R` channel: the pair list summed it truncated, and
/// `Δ` replaces that with the exact Ewald sum. But the pair list summed the *full* NDDO kernel
/// `γ_η(R) = e²/√(R² + η²)`, not `1/R`, and the difference
///
/// ```text
/// γ_η(R) − 1/R = −η²/(2R³) + 3η⁴/(8R⁵) − …
/// ```
///
/// was left truncated at the real-space cutoff. `Σ_T |T|⁻³` diverges logarithmically in three
/// dimensions, so that residual does not converge — `docs/scope.md` recorded it as "⛔ real-space;
/// logarithmically divergent, 0.10 eV per unit `ln r_c`".
///
/// # It is not unfixable, and the coefficient is computable
///
/// Approximating the lattice beyond `r_c` by its continuum density `1/V`,
///
/// ```text
/// Σ_{|T|>r_c} [γ_η − 1/R] ≈ (1/V)[ −2πη² ln(R/r_c) + 3πη⁴/(4 r_c²) ] + O(r_c⁻⁴)
/// ```
///
/// The `ln R` is the divergence. In the **energy** its coefficient is
/// `−(π/V) Σ_ab Q_a Q_b η_ab²`, and with `η_ab = ρ_a + ρ_b` that expands to
/// `2 Q_tot Σ_a Q_a ρ_a² + 2 (Σ_a Q_a ρ_a)²` — so for a **neutral** cell the first term drops and
/// the whole divergence is `−(2π/V) (Σ_a Q_a ρ_a)² ln R`. One computable scalar, always negative,
/// which is why the measured drift had a consistent sign.
///
/// So this is the same situation as a charged chain, and it takes the same treatment: cut the
/// continuum tail at a **stated reference length** [`KLOPMAN_OHNO_REFERENCE`] instead of at
/// infinity. The `ln r_c` then cancels exactly between the truncated sum and the tail, and what
/// remains depends on the declared reference rather than on the cutoff — which is the property
/// worth having, and the one `tests/pbc_klopman_ohno_tail.rs` measures by sweeping `r_c`.
///
/// # Why it has no gradient
///
/// The tail's leading terms depend on `η_ab`, `r_c` and the cell volume — **not** on the pair
/// separation `d`, because the expansion is in `d/|T|` with `|T| > r_c`. Its `d`-dependence first
/// appears at `O(d²/r_c²)` relative, below the `O(r_c⁻⁴)` already dropped. So it shifts the energy
/// and the Fock diagonal and contributes nothing to the forces at this order, which is stated here
/// rather than left for a reader to infer from a zero.
///
/// # Why the continuum is not used all the way down to `r_c`
///
/// A first version replaced the *whole* sum beyond `r_c` by the integral above. That is only
/// legitimate when many lattice repeats lie beyond `r_c`, and it failed in two regimes that the
/// existing tests caught:
///
/// * **A dilute cell.** A water molecule in a 45 Bohr box with a 40 Bohr cutoff has its *nearest*
///   image outside the cutoff. "Beyond `r_c` the lattice is a uniform density" is then a statement
///   about six discrete neighbours, and it moved the image-dipole coefficient — a quantity with a
///   known closed form, `−2π|p|²/3` — by 25 %.
/// * **A chain or a slab.** `Σ_T |T|⁻³` *converges* in one and two dimensions; there is no
///   logarithm to cancel, and applying the three-dimensional form is the dimensionality error
///   `docs/scope.md` warns about elsewhere. A charged chain's energy moved by 3e-2 eV.
///
/// So the sum beyond `r_c` is taken **explicitly**, over the translations the pair list actually
/// dropped, out to [`TAIL_DISCRETE_REACH`] times the cutoff — using the exact `γ_η − 1/R`, not its
/// expansion — and only the remainder past that is continuum. The remainder is where the
/// dimensionality enters, and it is a different integrand for a crystal, a slab and a chain.
///
/// The handover is **tapered** rather than sharp. A sharp one would make the energy jump by
/// `γ_η(R) − 1/R` each time a translation crossed it under strain, which at `R = 120 Bohr` is
/// `1.6e-4` eV — five orders above the stress tolerance. The taper is quintic, so value, first and
/// second derivatives are continuous and the analytic stress and Hessian stay exact. At the *inner*
/// edge, `|T| = r_c`, no taper is needed: a translation crossing there leaves the pair list and
/// enters this sum carrying the same `γ_η`, and the `1/R` half is in the Ewald sum either way, so
/// the two cancel to the `O(d/|T|)` at which this expansion is taken anyway.
///
/// # Why the same constant is used at every `q`
///
/// The response ([`crate::pbc::dfpt`]) needs the tail at finite `q`, where the sum is
/// `Σ_{|T|>r_c} e^{iq·T}/|T|³` and converges. In the same continuum approximation,
///
/// ```text
/// (1/V) ∫_{r>r_c} e^{iq·r}/r³ d³r = (4π/V) ∫_{q r_c}^∞ sin x / x² dx
///                                 = (4π/V) [ sin(z)/z − Ci(z) ],   z = q r_c
///                                 = −(4π/V) ln r_c − (4π/V)[ ln q − 1 + γ ] + O(z²)
/// ```
///
/// The **cutoff-dependent part, `−(4π/V) ln r_c`, does not depend on `q`.** That is the part this
/// correction exists to remove, so adding the `q = 0` constant at every `q` removes the truncation
/// dependence at every `q`, and it is what makes `D(q = 0)` equal the `q = 0` Hessian — the two are
/// the same number and `tests/pbc_dfpt.rs` requires them to agree to `1e-6` relative.
///
/// What is *not* captured is the remaining `−(4π/V) ln(q r_0)`, a genuine but weak non-analyticity
/// at `q → 0`: weaker than the monopole's own `4π/(V q²)`, which is the LO–TO discontinuity and
/// *is* treated exactly. `docs/scope.md` records it as the tail's remaining approximation.
///
/// # Declared cutoff, not the largest retained translation
///
/// `cutoff` is the **declared** real-space cutoff. Using `max |T|` over the retained set instead is
/// tempting — it is the honest boundary of what was actually summed — but the retained set changes
/// discretely as the cell strains, so `max |T|` jitters and `ln(r_0 / max|T|)` is not a smooth
/// function of strain. Built that way the stress missed its finite difference by `4.5e-7`
/// eV/Bohr³ where the tolerance is `1e-8`. The declared cutoff is strain-independent, which makes
/// the tail exactly `∝ 1/V` and its strain derivative exactly
/// [`LongRangeMonopole::klopman_ohno_strain`].
pub(crate) fn klopman_ohno_tail_matrix(
    molecule: &Molecule,
    params: &Am1Parameters,
    cutoff: f64,
) -> Result<Option<KlopmanOhnoTail>> {
    let Some(lattice) = molecule.cell else {
        return Ok(None);
    };
    if lattice.n_periodic() == 0 || lattice.measure() < 1.0e-12 || cutoff < 1.0e-6 {
        return Ok(None);
    }

    // One entry per *element*, not per atom: the tail depends on the pair only through
    // `η_ab = ρ⁰_a + ρ⁰_b`, so a system of `n` atoms over `k` elements needs `k(k+1)/2` lattice
    // sums rather than `n²` of them. With `k ≤ 10` in practice this is what keeps an explicit sum
    // over ~10⁴ translations affordable at every geometry.
    let mut elements: Vec<u8> = molecule.atoms.iter().map(|a| a.z).collect();
    elements.sort_unstable();
    elements.dedup();
    let kind: Vec<usize> = molecule
        .atoms
        .iter()
        .map(|a| elements.binary_search(&a.z).expect("element was collected"))
        .collect();
    let rho: Vec<f64> = elements
        .iter()
        .map(|z| params.element(*z).map(|e| e.rho0))
        .collect::<Result<Vec<_>>>()?;

    // From the **declared** cutoff and nothing else — in particular not from the cell.
    //
    // Keying it to the cell instead is tempting, because how good the continuum remainder is
    // depends on how many lattice repeats it covers, not on how many cutoffs. But then the handover
    // radius moves under strain, translations slide through the taper window, and the analytic
    // strain derivative would have to carry all of that. Measured, with `r_outer` at ten repeats of
    // a water chain: the true `dE/dε` was 1.9 eV per unit strain and the analytic term reported
    // 8e-6, putting the open-shell stress 1.0e-1 eV/Bohr³ from its finite difference. A declared
    // length has no strain derivative, which is the same reason `cutoff` itself is the declared one.
    let r_outer = TAIL_DISCRETE_REACH * cutoff;
    let r_inner = TAIL_TAPER_START * r_outer;
    // The translations the pair list dropped, out to where the continuum takes over.
    let dropped: Vec<Vec3> = lattice
        .image_offsets(r_outer)
        .into_iter()
        .map(|o| lattice.translation(o))
        .filter(|t| t.norm() > cutoff)
        .collect();

    // `∫ (1 − w(r)) r^{-n} dr` from `r_inner` to infinity, for the powers each dimensionality
    // needs. Computed once — they carry no `η`.
    let moment = |n: i32| taper_complement_moment(n, r_inner, r_outer);
    let (m1, m3) = (moment(1), moment(3));
    let measure = lattice.measure();
    let pi = std::f64::consts::PI;
    // `C_η = c2·(−η²/2) + c4·(3η⁴/8)`, the continuum remainder past `r_inner`.
    let (c2, c4) = match lattice.n_periodic() {
        // `∫ … 4πr² dr`, and the `r⁻¹` moment is the logarithm the reference length regularizes.
        3 => (
            4.0 * pi * (m1 + (KLOPMAN_OHNO_REFERENCE / r_outer).ln()) / measure,
            4.0 * pi * m3 / measure,
        ),
        // `∫ … 2πr dr` — convergent, so no reference length appears.
        2 => (
            2.0 * pi * moment(2) / measure,
            2.0 * pi * moment(4) / measure,
        ),
        // `∫ … 2 dr` along the chain — convergent likewise.
        _ => (2.0 * m3 / measure, 2.0 * moment(5) / measure),
    };

    let n_kind = elements.len();
    let mut value = vec![0.0; n_kind * n_kind];
    let mut strain = vec![[[0.0; 3]; 3]; n_kind * n_kind];
    // `d(ln measure)/dε_αβ` is the projector onto the periodic directions: `δ_αβ` for a crystal,
    // the in-plane projector for a slab, `û_α û_β` for a chain.
    let projector = lattice.periodic_projector();

    for i in 0..n_kind {
        for j in i..n_kind {
            let eta2 = (rho[i] + rho[j]) * (rho[i] + rho[j]);
            let mut acc = c2 * (-0.5 * eta2) + c4 * (0.375 * eta2 * eta2);
            let mut acc_strain = [[0.0; 3]; 3];
            for (alpha, row) in acc_strain.iter_mut().enumerate() {
                for (beta, s) in row.iter_mut().enumerate() {
                    *s = -projector[alpha][beta] * acc;
                }
            }
            for t in &dropped {
                let r = t.norm();
                let (w, dw) = quintic_switch(r, r_inner, r_outer);
                if w == 0.0 {
                    continue;
                }
                // The exact residual `γ_η(R) − 1/R`, not its expansion: at the inner edge this has
                // to be what the pair list dropped, or the two do not cancel there.
                let g = 1.0 / (r * r + eta2).sqrt() - 1.0 / r;
                let dg = -r / (r * r + eta2).powf(1.5) + 1.0 / (r * r);
                acc += w * g;
                // `d|T|/dε_αβ = T_α T_β / |T|` under a homogeneous strain.
                let radial = (dw * g + w * dg) / r;
                let tv = [t.x, t.y, t.z];
                for (alpha, row) in acc_strain.iter_mut().enumerate() {
                    for (beta, s) in row.iter_mut().enumerate() {
                        *s += radial * tv[alpha] * tv[beta];
                    }
                }
            }
            value[i * n_kind + j] = AM1_EV * acc;
            value[j * n_kind + i] = AM1_EV * acc;
            for (alpha, row) in acc_strain.iter().enumerate() {
                for (beta, s) in row.iter().enumerate() {
                    strain[i * n_kind + j][alpha][beta] = AM1_EV * s;
                    strain[j * n_kind + i][alpha][beta] = AM1_EV * s;
                }
            }
        }
    }

    Ok(Some(KlopmanOhnoTail {
        kind,
        n_kind,
        value,
        strain,
    }))
}

/// The Klopman–Ohno tail for one geometry, stored per **element pair**.
///
/// `nat × nat` would be the obvious layout and is what `Δ` itself uses, but the tail depends on the
/// pair only through `η_ab = ρ⁰_a + ρ⁰_b`. Storing it per element pair keeps the strain derivative
/// — nine numbers per entry — from being `9 nat²` doubles, which at a thousand atoms would have
/// been 576 MB for a term worth a few meV.
#[derive(Clone, Debug)]
pub(crate) struct KlopmanOhnoTail {
    /// Element index of each atom, into the `n_kind × n_kind` tables.
    kind: Vec<usize>,
    n_kind: usize,
    /// `ko` for an element pair, in eV.
    value: Vec<f64>,
    /// `∂ko/∂ε_αβ` for an element pair, in eV.
    strain: Vec<[[f64; 3]; 3]>,
}

impl KlopmanOhnoTail {
    #[inline]
    fn at(&self, a: usize, b: usize) -> usize {
        self.kind[a] * self.n_kind + self.kind[b]
    }

    #[inline]
    pub(crate) fn value(&self, a: usize, b: usize) -> f64 {
        self.value[self.at(a, b)]
    }

    /// `½ Σ_ab Q_a Q_b ∂ko_ab/∂ε_αβ` — the tail's contribution to `∂E/∂ε`.
    pub(crate) fn strain_energy(&self, charges: &[f64]) -> [[f64; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for (a, qa) in charges.iter().enumerate() {
            for (b, qb) in charges.iter().enumerate() {
                let block = &self.strain[self.at(a, b)];
                let w = 0.5 * qa * qb;
                for (alpha, row) in out.iter_mut().enumerate() {
                    for (beta, s) in row.iter_mut().enumerate() {
                        *s += w * block[alpha][beta];
                    }
                }
            }
        }
        out
    }
}

/// Translations out to this multiple of the real-space cutoff are summed explicitly.
///
/// Three puts the handover at `3 r_c`, where `γ_η − 1/R` has fallen by a factor of 27 from its
/// value at the cutoff and the continuum approximation to the remainder is correspondingly better.
/// The cost is the number of lattice points inside that radius, which for a dense cell is `~10⁴`
/// and for a dilute one is a handful — the expensive case is the one where the approximation was
/// already good, and the cheap case is the one that needed the explicit sum.
const TAIL_DISCRETE_REACH: f64 = 3.0;

/// Fraction of the handover radius at which the taper onto the continuum begins.
const TAIL_TAPER_START: f64 = 0.8;

/// Quintic switch: `1` below `a`, `0` above `b`, and its derivative.
///
/// The same smoothstep as [`crate::hamiltonian::exchange_taper`] and for the same reason — value,
/// first and second derivatives continuous, so an analytic gradient, stress and Hessian all stay
/// valid across it — but returning the derivative, which the tail's strain term needs.
#[inline]
fn quintic_switch(r: f64, a: f64, b: f64) -> (f64, f64) {
    if r <= a {
        return (1.0, 0.0);
    }
    if r >= b {
        return (0.0, 0.0);
    }
    let width = b - a;
    let x = (r - a) / width;
    let s = x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
    let ds = 30.0 * x * x * (1.0 - x) * (1.0 - x) / width;
    (1.0 - s, -ds)
}

/// `∫_a^∞ (1 − w(r)) r^{-n} dr`, with `w` the switch of [`quintic_switch`].
///
/// Splits at `b`: above it `w = 0` and the integral is elementary (`ln` for `n = 1`, a power
/// otherwise); between `a` and `b` it is done by composite Simpson. The integrand there is a
/// quintic over a power, smooth and bounded, and this runs once per dimensionality rather than
/// once per pair, so the panel count is set for accuracy and not for speed.
///
/// For `n = 1` the elementary part diverges and only the `[a, b]` piece is returned; the caller
/// supplies `ln(r₀ / b)` from [`KLOPMAN_OHNO_REFERENCE`]. That is the one place a convention
/// enters, and it enters in exactly one dimensionality — three.
fn taper_complement_moment(n: i32, a: f64, b: f64) -> f64 {
    const PANELS: usize = 1024;
    let h = (b - a) / PANELS as f64;
    let f = |r: f64| (1.0 - quintic_switch(r, a, b).0) * r.powi(-n);
    let mut sum = f(a) + f(b);
    for k in 1..PANELS {
        let r = a + h * k as f64;
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(r);
    }
    let near = sum * h / 3.0;
    let far = if n == 1 {
        0.0 // the caller adds `ln(r₀ / b)`
    } else {
        b.powi(1 - n) / (n - 1) as f64
    };
    near + far
}

/// Images summed explicitly on each side of a chain before the analytic tail takes over.
///
/// The answer does not depend on it — that is asserted in [`crate::pbc::ewald1d`] — so this is a
/// cost/accuracy choice, not a convergence parameter. 64 makes the neglected `u⁸` term of the
/// tail expansion smaller than double precision for any repeat length a chain would have.
const DEFAULT_CHAIN_IMAGES: i32 = 64;

/// Net Mulliken charges `Q_a = Z_a − p_a` from a total density matrix.
///
/// The long-range correction is expressed entirely in these — see
/// [`crate::fock::long_range_potential`] — so every derivative of it needs them too, and having
/// one definition keeps the energy, gradient, stress and Hessian reading the same quantity.
pub fn net_charges(
    molecule: &Molecule,
    basis: &crate::basis::Basis,
    params: &Am1Parameters,
    p_tot: &Matrix,
) -> Result<Vec<f64>> {
    let mut charges = Vec::with_capacity(molecule.atoms.len());
    for (a, atom) in molecule.atoms.iter().enumerate() {
        let mut population = 0.0;
        let off = basis.atom_offset[a];
        for k in 0..basis.atom_norb[a] {
            population += p_tot[(off + k, off + k)];
        }
        charges.push(params.element(atom.z)?.core_charge - population);
    }
    Ok(charges)
}

/// `∂φ^Ewald(r)/∂ε_αβ`, the derivative under a uniform strain applied to **both** `r` and the
/// lattice, at fixed `α`.
///
/// # Why the reciprocal term looks the way it does
///
/// Under `r → (1+ε)r` the reciprocal vectors transform contravariantly, `G → (1+ε)^{−T}G`, so
/// `G·r` is **strain-invariant** and the `cos(G·r)` factor contributes nothing. All the strain
/// dependence sits in the prefactor `A(G) = (4π/V) e^{−G²/4α²}/G²`, through `∂V/∂ε_αβ = V δ_αβ`
/// and `∂G²/∂ε_αβ = −2 G_α G_β`:
///
/// ```text
/// ∂A/∂ε_αβ = A [ −δ_αβ + 2 G_α G_β (1/4α² + 1/G²) ]
/// ```
///
/// `α` is held fixed rather than tracking `V^{−1/3}`. That is legitimate precisely because the
/// total is `α`-independent, and it is checked as such — a stress that moved with `α` would mean
/// a term had been dropped.
pub fn pair_potential_strain(ewald: &EwaldSum, r: Vec3) -> [[f64; 3]; 3] {
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let mut s = [[0.0_f64; 3]; 3];

    // Real space: a plain pair term, `f'(d) d_α d_β / d`.
    for offset in &ewald.real_images {
        let d = r + ewald.lattice.translation(*offset);
        let dist = d.norm();
        if dist < 1.0e-10 || dist > ewald.real_cutoff {
            continue;
        }
        let a = ewald.alpha;
        let f1 = -(libm::erfc(a * dist) / (dist * dist)
            + two_over_sqrt_pi * a * (-(a * dist).powi(2)).exp() / dist);
        let dv = [d.x, d.y, d.z];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += f1 * dv[i] * dv[j] / dist;
            }
        }
    }

    let inv_four_alpha2 = 1.0 / (4.0 * ewald.alpha * ewald.alpha);
    for (g, weight) in &ewald.reciprocal {
        let g2 = g.dot(*g);
        let term = weight * g.dot(r).cos();
        let gv = [g.x, g.y, g.z];
        let factor = 2.0 * (inv_four_alpha2 + 1.0 / g2);
        for i in 0..3 {
            for j in 0..3 {
                let delta_ij = if i == j { 1.0 } else { 0.0 };
                s[i][j] += term * (-delta_ij + factor * gv[i] * gv[j]);
            }
        }
    }

    // Background `−π/(α²V)`: `∂/∂ε_αβ = +π/(α²V) δ_αβ`. The self term is strain-independent.
    let background = std::f64::consts::PI / (ewald.alpha * ewald.alpha * ewald.volume);
    for (i, row) in s.iter_mut().enumerate() {
        row[i] += background;
    }
    s
}

/// `∂Δ_ab/∂ε_αβ`, in eV. See [`pair_potential_strain`] and [`delta_gradient`].
///
/// Unlike the position derivative, the **diagonal** `Δ_aa` contributes here: an atom's
/// interaction with its own images does not change when the atom moves, but it very much
/// changes when the cell is strained.
pub fn delta_strain(
    r: Vec3,
    lattice: &Lattice,
    translations: &[ImageOffset],
    ewald: &LongRangeKernel,
) -> [[f64; 3]; 3] {
    let mut s = ewald.pair_potential_strain(r);
    // Subtract `∂/∂ε` of `Σ_T 1/|r+T|`, which is `−d_α d_β / d³` per image.
    for offset in translations {
        let d = r + lattice.translation(*offset);
        let dist = d.norm();
        if dist < 1.0e-10 {
            continue;
        }
        let dv = [d.x, d.y, d.z];
        let inv3 = 1.0 / (dist * dist * dist);
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += dv[i] * dv[j] * inv3;
            }
        }
    }
    for row in s.iter_mut() {
        for v in row.iter_mut() {
            *v *= AM1_EV;
        }
    }
    s
}

/// `∂Δ_ab/∂r` at `r = R_b − R_a`, in eV/Bohr — the derivative of one entry of
/// [`LongRangeMonopole::delta`].
///
/// `Δ` is the Ewald potential minus the truncated real-space sum, so its derivative is the
/// derivative of each. `translations` must be the same list [`LongRangeMonopole::new`] was built
/// against, for the same reason: the correction is defined *relative to* what was already summed.
pub fn delta_gradient(
    r: Vec3,
    lattice: &Lattice,
    translations: &[ImageOffset],
    ewald: &LongRangeKernel,
) -> Vec3 {
    let mut truncated = Vec3::zero();
    for offset in translations {
        let d = r + lattice.translation(*offset);
        let dist = d.norm();
        if dist < 1.0e-10 {
            continue;
        }
        // ∇_r (1/|r+T|) = −(r+T)/|r+T|³
        truncated -= d / (dist * dist * dist);
    }
    (ewald.pair_potential_gradient(r) - truncated) * AM1_EV
}

/// `∂²Δ_ab/∂r∂r`, in eV/Bohr². See [`delta_gradient`].
pub fn delta_hessian(
    r: Vec3,
    lattice: &Lattice,
    translations: &[ImageOffset],
    ewald: &LongRangeKernel,
) -> [[f64; 3]; 3] {
    let mut h = ewald.pair_potential_hessian(r);
    // Subtract ∂²/∂r² of Σ_T 1/|r+T|, which is (3 d̂_i d̂_j − δ_ij)/d³.
    for offset in translations {
        let d = r + lattice.translation(*offset);
        let dist = d.norm();
        if dist < 1.0e-10 {
            continue;
        }
        let u = [d.x / dist, d.y / dist, d.z / dist];
        let inv3 = 1.0 / (dist * dist * dist);
        for i in 0..3 {
            for j in 0..3 {
                let delta_ij = if i == j { 1.0 } else { 0.0 };
                h[i][j] -= (3.0 * u[i] * u[j] - delta_ij) * inv3;
            }
        }
    }
    for row in h.iter_mut() {
        for v in row.iter_mut() {
            *v *= AM1_EV;
        }
    }
    h
}

/// The long-range monopole correction `Δ_ab`, in **eV**, for every atom pair.
///
/// `Δ_ab = e² [ φ^Ewald(R_ab) − Σ_{T ∈ pair list} 1/|R_ab + T| ]`: what the Ewald sum says the
/// lattice-summed `1/R` interaction is, minus what the truncated real-space pair list already
/// counted. Adding this to the monopole channel therefore replaces the truncation with the exact
/// sum, and leaves every other channel untouched.
///
/// The diagonal `Δ_aa` is the interaction of an atom with **its own images**, which is a real
/// contribution and not zero.
#[derive(Clone, Debug)]
pub struct LongRangeMonopole {
    /// `nat × nat`, eV.
    pub delta: Matrix,
    /// The Klopman–Ohno `R⁻³` tail's share of [`Self::delta`], when it is included.
    ///
    /// Kept separately because its **strain** derivative is different from the rest's. The Ewald
    /// part is a function of the pair separation and its virial is the ordinary pair virial; the
    /// tail is not a function of separation at all, so the pair expression has no term for it. It
    /// strains through the cell measure and through the lengths of the translations it sums, which
    /// [`KlopmanOhnoTail::strain_energy`] carries.
    ///
    /// Storing it apart is what stops the stress from being wrong: a first attempt folded the tail
    /// into `delta` and left the virial to the pair expression. The result passed the *gradient*
    /// finite difference — correctly, since the tail has no position dependence — and failed the
    /// *stress* one by 4.5e-6 eV/Bohr³, which is `E_tail/V` on that cell.
    pub(crate) klopman_ohno: Option<KlopmanOhnoTail>,
}

impl LongRangeMonopole {
    /// Build the correction for `molecule` against the translations `neighbors` actually used.
    ///
    /// The pair list is passed in rather than rebuilt, because the correction is defined
    /// *relative to it*: it must subtract exactly the translations that were already summed, or
    /// the result double-counts or leaves a gap.
    pub fn new(
        molecule: &Molecule,
        neighbors: &NeighborList,
        ewald: &LongRangeKernel,
    ) -> Result<Self> {
        let nat = molecule.atoms.len();
        let lattice = molecule
            .cell
            .ok_or_else(|| Am1Error::InvalidInput("the Ewald correction needs a cell".into()))?;
        let mut delta = Matrix::zeros(nat, nat);

        use rayon::prelude::*;
        let rows: Vec<Vec<f64>> = (0..nat)
            .into_par_iter()
            .map(|a| {
                let mut row = vec![0.0; nat];
                for (b, value) in row.iter_mut().enumerate() {
                    let r = molecule.atoms[b].position - molecule.atoms[a].position;
                    let mut truncated = 0.0;
                    for offset in &neighbors.translations {
                        let d = r + lattice.translation(*offset);
                        let dist = d.norm();
                        if dist < 1.0e-10 {
                            continue; // the atom itself, at T = 0
                        }
                        truncated += 1.0 / dist;
                    }
                    *value = AM1_EV * (ewald.pair_potential(r) - truncated);
                }
                row
            })
            .collect();

        for (a, row) in rows.iter().enumerate() {
            for (b, v) in row.iter().enumerate() {
                delta[(a, b)] = *v;
            }
        }
        Ok(Self {
            delta,
            klopman_ohno: None,
        })
    }

    /// `∂/∂ε_αβ` of the Klopman–Ohno tail's energy (eV).
    ///
    /// Zero when the tail is not included. See `Self::klopman_ohno` for why this is separate from
    /// the pair virial.
    pub fn klopman_ohno_strain(&self, charges: &[f64]) -> [[f64; 3]; 3] {
        match &self.klopman_ohno {
            Some(ko) => ko.strain_energy(charges),
            None => [[0.0; 3]; 3],
        }
    }

    // (the tail itself is the free function `klopman_ohno_tail_matrix`, below the impl)

    /// Add the analytic **Klopman–Ohno `R⁻³` tail** to `Δ`, and keep it for the strain derivative.
    ///
    /// The tail itself, and why it exists, is `klopman_ohno_tail_matrix`. What is here is only
    /// where it is stored: separately from `Δ`, because `Δ`'s strain derivative is taken from the
    /// Ewald kernel and the tail's is not — see [`Self::klopman_ohno_strain`].
    pub fn with_klopman_ohno_tail(
        mut self,
        molecule: &Molecule,
        params: &Am1Parameters,
        cutoff: f64,
    ) -> Result<Self> {
        let Some(ko) = klopman_ohno_tail_matrix(molecule, params, cutoff)? else {
            return Ok(self);
        };
        for a in 0..self.delta.rows {
            for b in 0..self.delta.cols {
                self.delta[(a, b)] += ko.value(a, b);
            }
        }
        self.klopman_ohno = Some(ko);
        Ok(self)
    }

    /// Build the correction and its machinery for `molecule`, or `None` when it does not apply.
    ///
    /// Applies in **every** periodic dimensionality — [`LongRangeKernel::for_lattice`] dispatches
    /// to the 3D Ewald sum, the 2D Parry sum or the 1D regularized chain sum — and to a molecule
    /// not at all, since there is no lattice sum to correct. Returning `None` rather than erroring
    /// is what lets `ewald: true` stay the default without making a molecule an error the caller
    /// did not ask for.
    ///
    /// > **Corrected in 0.2.2.** This said "applies only to a fully three-dimensional cell ... a
    /// > slab or a chain would need a reciprocal sum this module does not implement", which
    /// > contradicted the code below and `LongRangeKernel`, both of which have handled 1D and 2D
    /// > since 0.2.0. What *is* three-dimensional only is the **phased** kernel
    /// > ([`EwaldSum::phased_pair_potential`]) that the DFPT response needs.
    pub fn for_molecule(
        molecule: &Molecule,
        neighbors: &NeighborList,
        enabled: bool,
    ) -> Result<Option<(Self, LongRangeKernel)>> {
        Self::for_molecule_with(molecule, None, neighbors, enabled)
    }

    /// [`Self::for_molecule`], optionally including the Klopman–Ohno `R⁻³` tail.
    ///
    /// `params` is `Some` when the tail is wanted — it needs each element's `ρ⁰` — and `None` to
    /// leave it out, which is what lets `tests/pbc_klopman_ohno_tail.rs` measure the cutoff drift
    /// with and without rather than assert it away.
    pub fn for_molecule_with(
        molecule: &Molecule,
        params: Option<(&Am1Parameters, f64)>,
        neighbors: &NeighborList,
        enabled: bool,
    ) -> Result<Option<(Self, LongRangeKernel)>> {
        match (enabled, molecule.cell) {
            (true, Some(cell)) if cell.n_periodic() >= 1 => {
                match LongRangeKernel::for_lattice(&cell)? {
                    Some(kernel) => {
                        let mut monopole = Self::new(molecule, neighbors, &kernel)?;
                        if let Some((p, cutoff)) = params {
                            monopole = monopole.with_klopman_ohno_tail(molecule, p, cutoff)?;
                        }
                        Ok(Some((monopole, kernel)))
                    }
                    None => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    /// `∂/∂R_c` of `½ Σ_ab Q_a Q_b Δ_ab`, in eV/Bohr, one entry per atom.
    ///
    /// # Derivation
    ///
    /// With `Δ_ab = D(R_b − R_a)` and `D` even (the lattice has inversion symmetry, so the
    /// pair derivative is antisymmetric, `g_ab = −g_ba`):
    ///
    /// ```text
    /// ∂E/∂R_c = ½ Σ_ab Q_a Q_b g_ab (δ_bc − δ_ac) = −Q_c Σ_b Q_b g_cb
    /// ```
    ///
    /// The diagonal drops out on its own: `Δ_aa` is an atom's interaction with its own images,
    /// which does not change when the atom moves (every image moves with it).
    ///
    /// This is the **Hellmann–Feynman** part only, which is the whole gradient at a converged
    /// SCF: the density-response term vanishes because the energy is stationary in `P`.
    pub fn energy_gradient(
        molecule: &Molecule,
        neighbors: &NeighborList,
        ewald: &LongRangeKernel,
        charges: &[f64],
    ) -> Result<Vec<Vec3>> {
        let lattice = molecule
            .cell
            .ok_or_else(|| Am1Error::InvalidInput("the Ewald correction needs a cell".into()))?;
        let nat = molecule.atoms.len();
        use rayon::prelude::*;
        let grad: Vec<Vec3> = (0..nat)
            .into_par_iter()
            .map(|c| {
                let mut acc = Vec3::zero();
                for b in 0..nat {
                    if b == c {
                        continue;
                    }
                    let r = molecule.atoms[b].position - molecule.atoms[c].position;
                    acc += delta_gradient(r, &lattice, &neighbors.translations, ewald) * charges[b];
                }
                acc * (-charges[c])
            })
            .collect();
        Ok(grad)
    }

    /// `∂/∂ε_αβ` of `½ Σ_ab Q_a Q_b Δ_ab`, in eV — the virial the periodic stress needs.
    ///
    /// Divide by the cell measure to get a stress. The `a = b` diagonal is included: strain
    /// changes an atom's interaction with its own images even though translation does not.
    pub fn energy_strain(
        molecule: &Molecule,
        neighbors: &NeighborList,
        ewald: &LongRangeKernel,
        charges: &[f64],
    ) -> Result<[[f64; 3]; 3]> {
        let lattice = molecule
            .cell
            .ok_or_else(|| Am1Error::InvalidInput("the Ewald correction needs a cell".into()))?;
        let nat = molecule.atoms.len();
        use rayon::prelude::*;
        let rows: Vec<[[f64; 3]; 3]> = (0..nat)
            .into_par_iter()
            .map(|a| {
                let mut acc = [[0.0_f64; 3]; 3];
                for b in 0..nat {
                    let r = molecule.atoms[b].position - molecule.atoms[a].position;
                    let s = delta_strain(r, &lattice, &neighbors.translations, ewald);
                    let w = 0.5 * charges[a] * charges[b];
                    for i in 0..3 {
                        for j in 0..3 {
                            acc[i][j] += w * s[i][j];
                        }
                    }
                }
                acc
            })
            .collect();
        let mut total = [[0.0_f64; 3]; 3];
        for r in rows {
            for i in 0..3 {
                for j in 0..3 {
                    total[i][j] += r[i][j];
                }
            }
        }
        Ok(total)
    }

    /// `∂²/∂R_c∂R_d` of `½ Σ_ab Q_a Q_b Δ_ab`, in eV/Bohr², as `(c, d, block)` triples.
    ///
    /// Off-diagonal `c ≠ d` is `−Q_c Q_d H_cd`; the diagonal is `+Q_c Σ_{b≠c} Q_b H_cb`, which is
    /// the same statement as the acoustic sum rule for this term. Only `c ≤ d` is returned.
    ///
    /// Like [`Self::energy_gradient`] this is the fixed-charge part. The response of the charges
    /// to the displacement enters through the CPHF, via the perturbed Fock matrix.
    pub fn energy_hessian(
        molecule: &Molecule,
        neighbors: &NeighborList,
        ewald: &LongRangeKernel,
        charges: &[f64],
    ) -> Result<Vec<HessianBlock>> {
        let lattice = molecule
            .cell
            .ok_or_else(|| Am1Error::InvalidInput("the Ewald correction needs a cell".into()))?;
        let _t = crate::timing::Timer::start("ewald:hessian");
        let nat = molecule.atoms.len();
        use rayon::prelude::*;

        // Each unordered pair's `Δ''` once, not twice.
        //
        // `delta_hessian` is even in `r`: it is built from `3 d̂_i d̂_j − δ_ij` over `1/d³` and
        // from the Ewald second derivative, both of which are unchanged by `r → −r`, and the
        // translation set is symmetric so `{r + T}` maps onto `−{r + T}`. The old loop ran over
        // all `nat²` ordered pairs and evaluated the *same* lattice sum for `(c,b)` and `(b,c)`.
        // Evaluating the upper triangle halves the work exactly, and the lattice sum is what
        // costs here — each call walks every translation and every reciprocal vector.
        // `the_pair_hessian_is_even_in_the_separation` pins the symmetry this rests on.
        let pairs: Vec<(usize, usize)> = (0..nat)
            .flat_map(|c| ((c + 1)..nat).map(move |b| (c, b)))
            .collect();
        let hessians: Vec<[[f64; 3]; 3]> = pairs
            .par_iter()
            .map(|&(c, b)| {
                let r = molecule.atoms[b].position - molecule.atoms[c].position;
                delta_hessian(r, &lattice, &neighbors.translations, ewald)
            })
            .collect();

        let mut diagonals = vec![[[0.0_f64; 3]; 3]; nat];
        let mut out: Vec<HessianBlock> = Vec::with_capacity(pairs.len() + nat);
        for (&(c, b), hb) in pairs.iter().zip(&hessians) {
            let w = charges[c] * charges[b];
            let mut off = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let v = w * hb[i][j];
                    // The same `Δ''` serves both diagonal blocks, by the evenness above.
                    diagonals[c][i][j] += v;
                    diagonals[b][i][j] += v;
                    off[i][j] = -v;
                }
            }
            out.push((c, b, off));
        }
        for (c, diagonal) in diagonals.into_iter().enumerate() {
            out.push((c, c, diagonal));
        }
        Ok(out)
    }

    /// Core–core contribution `½ Σ_ab Z_a Z_b Δ_ab`, eV.
    pub fn core_core_energy(&self, molecule: &Molecule, params: &Am1Parameters) -> Result<f64> {
        let nat = molecule.atoms.len();
        let mut charges = Vec::with_capacity(nat);
        for atom in &molecule.atoms {
            charges.push(params.element(atom.z)?.core_charge);
        }
        let mut total = 0.0;
        for a in 0..nat {
            for b in 0..nat {
                total += charges[a] * charges[b] * self.delta[(a, b)];
            }
        }
        Ok(0.5 * total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cubic lattice of side `a` Bohr.
    fn cubic(a: f64) -> Lattice {
        Lattice::cubic(a).unwrap()
    }

    #[test]
    fn the_pair_potential_does_not_depend_on_the_splitting_parameter() {
        // The defining property of an Ewald sum, and the one test that catches almost any error
        // in it: the split between real and reciprocal space is arbitrary, so the answer must be
        // independent of where it is made.
        let lattice = cubic(10.0);
        let r = Vec3::new(1.3, -2.1, 0.7);
        let reference = EwaldSum::new(&lattice, default_alpha(lattice.volume()), 1.0e-14)
            .unwrap()
            .pair_potential(r);
        for scale in [0.5, 0.75, 1.5, 2.0] {
            let alpha = scale * default_alpha(lattice.volume());
            let value = EwaldSum::new(&lattice, alpha, 1.0e-14)
                .unwrap()
                .pair_potential(r);
            let diff = (value - reference).abs();
            assert!(
                diff < 1.0e-9,
                "alpha scale {scale}: potential moved by {diff:.3e} ({value} vs {reference})"
            );
        }
    }

    /// `Δ'(−r) = −Δ'(r)`, which is what lets the perturbed-Fock long-range term tabulate one
    /// triangle of pairs and negate for the other.
    ///
    /// `Δ` is even in the separation, so its gradient is odd. Without this the table in
    /// `pbc::hessian`'s `skeleton_fock_ov` would carry the wrong sign on half its entries, and
    /// the CPHF would converge to a plausible wrong answer rather than fail.
    #[test]
    fn the_pair_gradient_is_odd_in_the_separation() {
        let lattice = cubic(9.0);
        let ewald = LongRangeKernel::for_lattice(&lattice).unwrap().unwrap();
        let translations = lattice.image_offsets(14.0);
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for r in [
            Vec3::new(1.3, -2.1, 0.7),
            Vec3::new(3.4, 0.0, 0.0),
            Vec3::new(-0.6, 2.9, -4.2),
        ] {
            let plus = delta_gradient(r, &lattice, &translations, &ewald);
            let minus = delta_gradient(r * -1.0, &lattice, &translations, &ewald);
            let sum = plus + minus;
            scale = scale.max(plus.norm());
            worst = worst.max(sum.norm());
        }
        eprintln!("    max |D'(-r) + D'(r)| = {worst:.3e} of {scale:.3e}");
        assert!(
            worst < 1.0e-12 * scale.max(1.0),
            "the pair gradient is not odd in r: {worst:.3e}"
        );
    }

    /// `Δ''(−r) = Δ''(r)`, which is what lets `energy_hessian` evaluate one triangle of pairs.
    ///
    /// Not an optimization detail: if this failed, halving the loop would silently use the wrong
    /// block for half the pairs, and the result would stay symmetric and plausible. The reason it
    /// holds is that every term is built from `d̂_i d̂_j` and even powers of `|d|`, and the
    /// translation set is symmetric so `{r + T}` maps onto `−{r + T}` under `r → −r`.
    #[test]
    fn the_pair_hessian_is_even_in_the_separation() {
        let lattice = cubic(9.0);
        let ewald = LongRangeKernel::for_lattice(&lattice).unwrap().unwrap();
        let translations = lattice.image_offsets(14.0);
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for r in [
            Vec3::new(1.3, -2.1, 0.7),
            Vec3::new(3.4, 0.0, 0.0),
            Vec3::new(-0.6, 2.9, -4.2),
        ] {
            let plus = delta_hessian(r, &lattice, &translations, &ewald);
            let minus = delta_hessian(r * -1.0, &lattice, &translations, &ewald);
            for i in 0..3 {
                for j in 0..3 {
                    scale = scale.max(plus[i][j].abs());
                    worst = worst.max((plus[i][j] - minus[i][j]).abs());
                }
            }
        }
        eprintln!("    max |D''(-r) - D''(r)| = {worst:.3e} of {scale:.3e}");
        assert!(
            worst < 1.0e-12 * scale.max(1.0),
            "the pair Hessian is not even in r: {worst:.3e}"
        );
    }

    #[test]
    fn the_hand_written_second_derivative_matches_a_finite_difference() {
        // `pair_potential_hessian` is transcribed by hand because `Scalar` has no `erfc`, so it
        // gets the check a hand-written derivative needs: against a central difference of
        // `pair_potential_gradient`, which is itself checked against `pair_potential`.
        let lattice = cubic(9.0);
        let ewald = EwaldSum::new(&lattice, default_alpha(lattice.volume()), 1.0e-12).unwrap();
        let step = 1.0e-5;
        let mut worst = 0.0_f64;
        for r in [
            Vec3::new(1.7, -0.9, 2.3),
            Vec3::new(4.5, 4.5, 4.5), // the body centre, where images are equidistant
            Vec3::new(0.4, 0.0, 0.0), // nearly on top of an image
        ] {
            let analytic = ewald.pair_potential_hessian(r);
            for j in 0..3 {
                let mut plus = r;
                let mut minus = r;
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
                let gp = ewald.pair_potential_gradient(plus);
                let gm = ewald.pair_potential_gradient(minus);
                let fd = (gp - gm) / (2.0 * step);
                for (i, f) in [fd.x, fd.y, fd.z].iter().enumerate() {
                    worst = worst.max((analytic[i][j] - f).abs());
                }
            }
        }
        eprintln!("Ewald d2phi/dr2 vs FD: worst {worst:.3e}");
        assert!(
            worst < 1.0e-7,
            "Ewald second derivative mismatch {worst:.3e}"
        );
    }

    #[test]
    fn a_slab_or_a_chain_is_refused() {
        let slab = Lattice::from_vectors(
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap();
        let err = EwaldSum::new(&slab, 0.3, 1.0e-12).unwrap_err();
        assert!(err.to_string().contains("three-dimensional"));
    }
}

#[cfg(test)]
mod tail_component_tests {
    use super::*;

    /// The quintic switch is `C²` at both ends, which is the whole reason it is a quintic.
    ///
    /// Value, first and second derivative all have to be continuous there: the energy is
    /// differentiated once for the force and twice for the Hessian and the stress, and a switch
    /// that is only `C¹` puts a step in the second derivative exactly where the handover is. The
    /// derivative is checked against a central difference of the value, so the analytic `dw` and
    /// the `w` it claims to differentiate cannot drift apart.
    #[test]
    fn the_quintic_switch_is_twice_differentiable_at_both_ends() {
        let (a, b) = (8.0_f64, 10.0);
        for edge in [a, b] {
            let h = 1.0e-4;
            let (w_in, d_in) = quintic_switch(edge - h, a, b);
            let (w_out, d_out) = quintic_switch(edge + h, a, b);
            let (w_at, d_at) = quintic_switch(edge, a, b);
            // Value.
            assert!(
                (w_in - w_at).abs() < 1.0e-8 && (w_out - w_at).abs() < 1.0e-8,
                "the switch jumps at {edge}: {w_in} / {w_at} / {w_out}"
            );
            // First derivative: zero at both ends, and continuous across them.
            assert!(
                d_at.abs() < 1.0e-12 && d_in.abs() < 1.0e-6 && d_out.abs() < 1.0e-6,
                "the switch's slope is not zero at {edge}: {d_in} / {d_at} / {d_out}"
            );
            // Second derivative. It vanishes *at* the edge and rises linearly away from it —
            // `s''(x) = 60x(1−x)(1−2x)` — so the statement that makes this a test is that the
            // measured curvature is `O(offset)`: halve the distance from the edge and it halves.
            // A `C¹`-only switch would leave a step there and the two would come out equal.
            let inward = if edge == a { 1.0 } else { -1.0 };
            let curvature = |offset: f64| {
                let r = edge + inward * offset;
                (quintic_switch(r + h, a, b).1 - quintic_switch(r - h, a, b).1) / (2.0 * h)
            };
            let (coarse, fine) = (curvature(0.02), curvature(0.01));
            assert!(
                coarse.abs() > 1.0e-12,
                "at {edge} the curvature is identically zero, so this measures nothing"
            );
            assert!(
                fine.abs() < 0.6 * coarse.abs(),
                "the switch's curvature does not vanish linearly at {edge}: {coarse:.3e} at 0.02 \
                 against {fine:.3e} at 0.01, so the second derivative has a step there"
            );
        }
        // And it is exactly 1 below and 0 above, not merely close.
        assert_eq!(quintic_switch(a - 1.0, a, b).0, 1.0);
        assert_eq!(quintic_switch(b + 1.0, a, b).0, 0.0);
        // The analytic slope is the derivative of the value everywhere in between.
        for r in [8.3_f64, 9.0, 9.7] {
            let h = 1.0e-6;
            let fd = (quintic_switch(r + h, a, b).0 - quintic_switch(r - h, a, b).0) / (2.0 * h);
            let analytic = quintic_switch(r, a, b).1;
            assert!(
                (fd - analytic).abs() < 1.0e-7,
                "at r = {r} the slope is {analytic} but the value differentiates to {fd}"
            );
        }
    }

    /// `taper_complement_moment` against an independent quadrature.
    ///
    /// It is `∫_a^∞ (1 − w(r)) r^{-n} dr`, done as Simpson over `[a, b]` plus a closed form beyond.
    /// Checked here against a much finer trapezoid taken out to a large radius — a different rule,
    /// a different truncation — because the closed-form tail and the numeric part meeting correctly
    /// is exactly the kind of seam that is wrong by one term and still looks plausible.
    #[test]
    fn the_taper_complement_moments_match_an_independent_quadrature() {
        let (a, b) = (24.0_f64, 30.0);
        for n in [2_i32, 3, 4, 5] {
            let mine = taper_complement_moment(n, a, b);
            // A brute-force trapezoid from `a` out to where `r^{-n}` is negligible.
            let far = b * 4000.0;
            let steps = 4_000_000_usize;
            let dr = (far - a) / steps as f64;
            let f = |r: f64| (1.0 - quintic_switch(r, a, b).0) * r.powi(-n);
            let mut acc = 0.5 * (f(a) + f(far));
            for i in 1..steps {
                acc += f(a + dr * i as f64);
            }
            let mut brute = acc * dr;
            // The trapezoid stops at `far`; the analytic remainder beyond it.
            brute += far.powi(1 - n) / (n - 1) as f64;
            assert!(
                (mine - brute).abs() < 1.0e-9 * mine.abs().max(1.0e-12),
                "moment n = {n}: Simpson-plus-closed-form gives {mine:.12e}, an independent \
                 quadrature gives {brute:.12e}"
            );
        }
    }

    /// The `n = 1` moment is the divergent one, and the caller supplies its logarithm.
    ///
    /// So `taper_complement_moment(1, ..)` must return **only** the finite `[a, b]` piece; if it
    /// silently added a tail the reference length would be counted twice and the cutoff dependence
    /// the whole tail exists to cancel would come back with the wrong sign.
    #[test]
    fn the_logarithmic_moment_stops_at_the_handover() {
        let (a, b) = (24.0_f64, 30.0);
        let mine = taper_complement_moment(1, a, b);
        let steps = 200_000_usize;
        let dr = (b - a) / steps as f64;
        let f = |r: f64| (1.0 - quintic_switch(r, a, b).0) / r;
        let mut acc = 0.5 * (f(a) + f(b));
        for i in 1..steps {
            acc += f(a + dr * i as f64);
        }
        let brute = acc * dr;
        assert!(
            (mine - brute).abs() < 1.0e-10,
            "the n = 1 moment is {mine:.12e}, but the finite part alone is {brute:.12e} — it must \
             not carry a tail of its own"
        );
    }

    /// The Klopman–Ohno tail reduces to its continuum form when the lattice is fine enough that
    /// the explicit sum has nothing left to correct.
    ///
    /// The implementation sums the dropped translations exactly out to the handover and takes only
    /// the remainder from the continuum. That extra work is invisible unless it is compared
    /// against the thing it replaced: for a **small** cell with a large cutoff, many shells lie
    /// beyond `r_c` and the continuum estimate `−2πη² ln(r₀/r_c)/V + 3πη⁴/(4 V r_c²)` is close, so
    /// the two must agree. For a coarse lattice they must **not** — that difference is what the
    /// explicit sum was added for, and it is asserted too.
    #[test]
    fn the_tail_meets_its_continuum_limit_where_the_continuum_is_valid() {
        let params = crate::params::Am1Parameters::standard().unwrap();
        let ang = 1.0 / crate::constants::AM1_A0;
        let make = |a_ang: f64| {
            crate::system::Molecule::new(vec![
                crate::system::Atom {
                    z: 9,
                    position: Vec3::zero(),
                },
                crate::system::Atom {
                    z: 1,
                    position: Vec3::new(0.94 * ang, 0.0, 0.0),
                },
            ])
            .with_cell(Lattice::cubic(a_ang * ang).unwrap())
        };
        // The continuum value the pre-explicit-sum version used, written out here rather than
        // called, so the comparison is against the formula and not against the code.
        let continuum =
            |molecule: &crate::system::Molecule, a: usize, b: usize, cutoff: f64| -> f64 {
                let cell = molecule.cell.unwrap();
                let rho = |i: usize| params.element(molecule.atoms[i].z).unwrap().rho0;
                let eta = rho(a) + rho(b);
                let (eta2, pi) = (eta * eta, std::f64::consts::PI);
                AM1_EV
                    * (-2.0 * pi * eta2 * (KLOPMAN_OHNO_REFERENCE / cutoff).ln()
                        + 3.0 * pi * eta2 * eta2 / (4.0 * cutoff * cutoff))
                    / cell.volume()
            };

        // Two cells, and the claim is the **trend**: the explicit sum's correction to the
        // continuum grows as the lattice coarsens relative to the cutoff. One cell cannot say
        // that — the continuum's leading term is the same logarithm either way, so the two agree
        // to a few percent even where the continuum is a poor description of the lattice, and a
        // threshold picked to "look like disagreement" would be a number with nothing behind it.
        let dense = make(4.5); // `r_c / L` = 7: many shells lie beyond the cutoff.
        let dense_cutoff = 60.0;
        let dense_rel = {
            let ko = klopman_ohno_tail_matrix(&dense, &params, dense_cutoff)
                .unwrap()
                .unwrap();
            let want = continuum(&dense, 0, 1, dense_cutoff);
            (ko.value(0, 1) - want).abs() / want.abs()
        };

        let dilute = make(24.0); // `r_c / L` = 0.66: the cutoff does not reach one repeat.
        let dilute_cutoff = 30.0;
        let dilute_rel = {
            let ko = klopman_ohno_tail_matrix(&dilute, &params, dilute_cutoff)
                .unwrap()
                .unwrap();
            let want = continuum(&dilute, 0, 1, dilute_cutoff);
            (ko.value(0, 1) - want).abs() / want.abs()
        };

        assert!(
            dense_rel < 0.02,
            "on a dense cell the explicit sum and the continuum should agree, but they are \
             {:.3} % apart",
            100.0 * dense_rel
        );
        assert!(
            dilute_rel > 5.0 * dense_rel,
            "the correction should grow as the lattice coarsens, but it is {:.3} % on a dense \
             cell against {:.3} % on a dilute one, which is not a trend",
            100.0 * dense_rel,
            100.0 * dilute_rel
        );
    }

    /// The tail depends on the pair only through `η = ρ⁰_a + ρ⁰_b`, which is what lets it be
    /// stored per element pair rather than per atom pair.
    ///
    /// Asserted rather than assumed, because the storage layout is built on it: two atoms of the
    /// same element must get identical entries, and the matrix must be symmetric.
    #[test]
    fn the_tail_depends_on_the_pair_only_through_eta() {
        let params = crate::params::Am1Parameters::standard().unwrap();
        let ang = 1.0 / crate::constants::AM1_A0;
        let molecule = crate::system::Molecule::new(vec![
            crate::system::Atom {
                z: 8,
                position: Vec3::zero(),
            },
            crate::system::Atom {
                z: 1,
                position: Vec3::new(0.96 * ang, 0.0, 0.0),
            },
            crate::system::Atom {
                z: 1,
                position: Vec3::new(-0.22 * ang, 0.93 * ang, 0.0),
            },
        ])
        .with_cell(Lattice::cubic(6.0 * ang).unwrap());
        let ko = klopman_ohno_tail_matrix(&molecule, &params, 40.0)
            .unwrap()
            .unwrap();
        // The two hydrogens are the same element, so every entry involving one equals the
        // corresponding entry involving the other.
        for other in 0..3 {
            assert_eq!(
                ko.value(1, other),
                ko.value(2, other),
                "the two hydrogens differ at ({other})"
            );
        }
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(ko.value(a, b), ko.value(b, a), "the tail is not symmetric");
            }
        }
    }
}
