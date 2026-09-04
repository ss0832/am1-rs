// SPDX-License-Identifier: GPL-3.0-or-later

//! Turning a low-dimensional polarizability into a dielectric constant, by **declaring the body**.
//!
//! # What was refused, and what is offered instead
//!
//! [`crate::pbc::dielectric_tensor`] returns `ε_∞ = 1 + 4πα/Ω`, and `Ω` has to be a volume.
//! [`crate::lattice::Lattice::measure`] gives an *area* for a slab and a *length* for a chain, so
//! `α/measure` is a length and an area respectively — not a susceptibility. 0.2.0 divided anyway
//! and reported numbers that were not dielectric constants; 0.2.1 refused; 0.2.2 supplies the
//! missing ingredient **as an argument** rather than inventing it.
//!
//! That ingredient is a thickness for a slab, or a cross-sectional area for a wire. It is a claim
//! about the material — where the layer stops — and nothing in a supercell fixes it, which is why
//! it is required and never defaulted. The same rule [`crate::pbc::ewald1d::AxisConvention`] and
//! `dielectric_function`'s `chain_radius` already follow.
//!
//! # The conversion is a depolarization problem, not a division
//!
//! The naive step is `ε = 1 + 4πα/(measure · extent)`. That is right along directions where the
//! induced polarization creates **no** macroscopic field, and wrong along the others, because the
//! `α` this crate computes is the response to the *external* field: the induced charges interact
//! through the same Coulomb operator the SCF uses, so for a slab polarized along its normal the
//! depolarizing field is already inside `α`. Dividing and adding 1 would then count the screening
//! once and the shape not at all.
//!
//! With `χ = α/(measure · extent)` the external-field susceptibility and `N` the depolarization
//! factor of the assumed body along that principal axis,
//!
//! ```text
//! ε = 1 + 4πχ / (1 − 4πNχ)
//! ```
//!
//! | body | axis | `N` | `ε` |
//! |---|---|---|---|
//! | slab | in plane | 0 | `1 + 4πχ` |
//! | slab | along the normal | 1 | `1/(1 − 4πχ)` |
//! | wire | along the axis | 0 | `1 + 4πχ` |
//! | wire | transverse (circular section) | 1/2 | `(1 + 2πχ)/(1 − 2πχ)` |
//! | crystal | any | 0 | `1 + 4πχ` — which is `dielectric_tensor` |
//!
//! The crystal row is not a special case bolted on: three-dimensional tin-foil boundary conditions
//! remove the macroscopic depolarizing field, so `α` there is already the response to the *internal*
//! macroscopic field and `N = 0` is the correct entry. That the table closes on the existing
//! function is what [`crate::pbc::extent::epsilon_from_polarizability`]'s tests check first.
//!
//! # What is model-dependent, and what is not
//!
//! The thickness is a choice, so `ε` is a choice. Two combinations of it are **not**:
//!
//! ```text
//! (ε_∥ − 1) · d = 4π α_∥ / A            (1 − 1/ε_⊥) · d = 4π α_⊥ / A
//! ```
//!
//! Both sides of each are thickness-free. They say the layer has a well-defined *sheet*
//! susceptibility in plane and a well-defined sheet **inverse** susceptibility out of plane, and
//! that everything else in `ε` is the convention. `2π α_∥/A` is the Rytova–Keldysh screening
//! length, the quantity the monolayer literature reports for exactly this reason.
//!
//! Read the other way round, those two identities are capacitor stacking: a layer of thickness `d₁`
//! and dielectric `ε(d₁)` padded with vacuum out to `d₂` is, in parallel, `d₂ε(d₂) = d₁ε(d₁) +
//! (d₂ − d₁)`, and in series, `d₂/ε(d₂) = d₁/ε(d₁) + (d₂ − d₁)`. So the two formulas above are
//! forced, not chosen, once the thickness is named — which is a second, independent derivation and
//! is tested as one.

use crate::error::{Am1Error, Result};
use crate::lattice::Lattice;
use crate::math::Vec3;
use crate::params::Am1Parameters;
use crate::pbc::hessian::DielectricTensors;
use crate::pbc::scf::PbcOptions;
use crate::system::Molecule;

use std::f64::consts::PI;

/// The physical extent assigned to a cell that is not periodic in three dimensions.
///
/// There is no default and no inference. A supercell says where the atoms are, not where the
/// material stops, and every choice here changes `ε` — so it is spelled out by the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtentConvention {
    /// A slab (two periodic directions) assigned a **thickness** in Bohr, measured along the cell
    /// normal `â₁ × â₂`. The assigned volume per cell is `A · d`.
    SlabThickness(f64),
    /// A chain (one periodic direction) assigned a **cross-sectional area** in Bohr², transverse
    /// to the periodic axis. The assigned volume per cell is `L · S`.
    ///
    /// An area rather than a radius, so that it multiplies `measure` to a volume the way a slab's
    /// thickness does; the section is taken to be **circular**, which is what fixes the transverse
    /// depolarization factor at `1/2`.
    WireCrossSection(f64),
}

impl ExtentConvention {
    /// The extent itself: Bohr for a slab, Bohr² for a wire.
    #[inline]
    pub fn value(&self) -> f64 {
        match self {
            Self::SlabThickness(d) => *d,
            Self::WireCrossSection(s) => *s,
        }
    }

    /// How many periodic directions this convention describes.
    #[inline]
    fn n_periodic(&self) -> usize {
        match self {
            Self::SlabThickness(_) => 2,
            Self::WireCrossSection(_) => 1,
        }
    }

    fn units(&self) -> &'static str {
        match self {
            Self::SlabThickness(_) => "Bohr",
            Self::WireCrossSection(_) => "Bohr^2",
        }
    }
}

/// An orthonormal frame `(e1, e2, e3)` with `e3` along `axis`.
fn frame(axis: Vec3) -> [Vec3; 3] {
    let e3 = axis / axis.norm();
    // Any vector not parallel to `e3` seeds the complement; picking the smallest component of
    // `e3` to zero out is the standard way to guarantee the cross product is well conditioned.
    let seed = if e3.x.abs() <= e3.y.abs() && e3.x.abs() <= e3.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if e3.y.abs() <= e3.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let mut e1 = seed - e3 * e3.dot(seed);
    e1 = e1 / e1.norm();
    let e2 = e3.cross(e1);
    [e1, e2, e3]
}

fn dot3(a: Vec3, m: &[[f64; 3]; 3], b: Vec3) -> f64 {
    let av = [a.x, a.y, a.z];
    let bv = [b.x, b.y, b.z];
    let mut acc = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            acc += av[i] * m[i][j] * bv[j];
        }
    }
    acc
}

/// `ε = 1 + 4πχ/(1 − 4πNχ)` for one principal value.
fn scalar_epsilon(chi: f64, n: f64) -> Result<f64> {
    let denom = 1.0 - 4.0 * PI * n * chi;
    if denom <= 1.0e-12 {
        return Err(Am1Error::InvalidInput(format!(
            "the assigned extent is too small for this cell's own response: the depolarization \
             denominator `1 - 4πNχ` came out {denom:.6e}, so ε would be infinite or negative. \
             χ = {chi:.6} is the polarizability spread over the volume you assigned, and a body \
             cannot be thinner than the polarization it carries. Assign more extent, or check \
             that the polarizability is the one you meant."
        )));
    }
    Ok(1.0 + 4.0 * PI * chi / denom)
}

/// Symmetric 2 × 2 eigendecomposition: returns `(λ₁, λ₂, cos θ, sin θ)` with the eigenvector of
/// `λ₁` at angle `θ`.
fn eig2(a: f64, b: f64, d: f64) -> (f64, f64, f64, f64) {
    if b.abs() < 1.0e-300 {
        return (a, d, 1.0, 0.0);
    }
    let tr = a + d;
    let diff = a - d;
    let disc = (diff * diff + 4.0 * b * b).sqrt();
    let l1 = 0.5 * (tr + disc);
    let l2 = 0.5 * (tr - disc);
    // Eigenvector for `l1`: `(b, l1 - a)`, normalized.
    let (vx, vy) = (b, l1 - a);
    let n = (vx * vx + vy * vy).sqrt();
    (l1, l2, vx / n, vy / n)
}

/// The dielectric tensor implied by a polarizability, an assigned extent, and the shape that goes
/// with it. **Pure algebra** — no SCF, no cell beyond `measure` and `axis`.
///
/// This is the whole of the model-dependent step, isolated so it can be tested against synthetic
/// input: the identities in this module's header are properties of *these fifteen lines*, not of
/// the CPHF that supplies `alpha`.
///
/// - `alpha` — the clamped-ion polarizability in Bohr³, Cartesian, as [`crate::pbc::polarizability`]
///   returns it.
/// - `axis` — the slab **normal**, or the wire **axis**. Need not be normalized.
/// - `measure` — the cell's periodic extent: area (Bohr²) for a slab, length (Bohr) for a wire.
/// - `extent` — the assigned thickness or cross-section.
///
/// Components mixing the distinguished axis with its complement are **dropped**: the depolarization
/// factor is defined per principal axis of the assumed body, and a body whose principal axes do not
/// line up with its own normal is not described by one factor. [`extent_axis_mixing`] measures what
/// was dropped so that the assumption is checkable rather than implicit; for a slab or a wire with
/// any symmetry at all it is zero.
pub fn epsilon_from_polarizability(
    alpha: &[[f64; 3]; 3],
    axis: Vec3,
    measure: f64,
    extent: ExtentConvention,
) -> Result<[[f64; 3]; 3]> {
    let e = extent.value();
    // `is_finite` first: it is what rejects NaN, which every comparison below would let through.
    if !e.is_finite() || e <= 0.0 {
        return Err(Am1Error::InvalidInput(format!(
            "the assigned extent must be a positive, finite number of {}; got {e}",
            extent.units()
        )));
    }
    if !measure.is_finite() || measure <= 0.0 {
        return Err(Am1Error::InvalidInput(
            "the cell's periodic measure is not positive, so there is no volume to spread the \
             polarizability over"
                .into(),
        ));
    }
    if axis.norm() < 1.0e-12 {
        return Err(Am1Error::InvalidInput(
            "the distinguished axis is degenerate: a slab normal or a wire axis is needed".into(),
        ));
    }

    // `measure · extent` is a volume in both conventions — area × thickness, or length × section.
    let volume = measure * e;
    let [e1, e2, e3] = frame(axis);

    // The depolarization factor along `e3` and in the `(e1, e2)` plane. This one line is the
    // entire difference between the two bodies.
    let (n_axis, n_plane) = match extent {
        ExtentConvention::SlabThickness(_) => (1.0, 0.0),
        ExtentConvention::WireCrossSection(_) => (0.0, 0.5),
    };

    // χ in the frame. Only the blocks that survive are formed.
    let chi = |u: Vec3, v: Vec3| dot3(u, alpha, v) / volume;
    let (c11, c12, c22, c33) = (chi(e1, e1), chi(e1, e2), chi(e2, e2), chi(e3, e3));

    // The 2 × 2 plane block, diagonalized so the scalar law applies per principal value. For
    // `N = 0` this is the identity `I + 4πχ` and the rotation is a no-op, but doing it the same
    // way in both cases keeps one code path.
    let (l1, l2, cx, sx) = eig2(c11, c12, c22);
    let (p1, p2) = (scalar_epsilon(l1, n_plane)?, scalar_epsilon(l2, n_plane)?);
    let p33 = scalar_epsilon(c33, n_axis)?;

    // Back to Cartesian: `ε = Σ_k p_k v_k v_kᵀ` over the three principal directions.
    let v1 = e1 * cx + e2 * sx;
    let v2 = e1 * (-sx) + e2 * cx;
    let mut out = [[0.0_f64; 3]; 3];
    for (p, v) in [(p1, v1), (p2, v2), (p33, e3)] {
        let c = [v.x, v.y, v.z];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] += p * c[i] * c[j];
            }
        }
    }
    Ok(out)
}

/// How much of `alpha` couples the distinguished axis to its complement, relative to the largest
/// diagonal entry.
///
/// [`epsilon_from_polarizability`] drops that coupling, because a depolarization factor is a
/// per-principal-axis quantity. Zero means the slab normal (or wire axis) *is* a principal axis of
/// the response and nothing was lost. This is the same kind of measurement
/// [`crate::pbc::dielectric_origin_sensitivity`] makes for the position operator: the assumption is
/// reported, not asserted.
pub fn extent_axis_mixing(alpha: &[[f64; 3]; 3], axis: Vec3) -> f64 {
    if axis.norm() < 1.0e-12 {
        return f64::NAN;
    }
    let [e1, e2, e3] = frame(axis);
    let scale = (0..3)
        .map(|i| alpha[i][i].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0e-30);
    dot3(e1, alpha, e3).abs().max(dot3(e2, alpha, e3).abs()) / scale
}

/// The distinguished axis of a cell under a given convention: the slab normal, or the wire axis.
fn distinguished_axis(cell: &Lattice, extent: ExtentConvention) -> Result<Vec3> {
    let ax = cell.periodic_axes();
    match extent {
        ExtentConvention::SlabThickness(_) => Ok(cell.cell.col[ax[0]].cross(cell.cell.col[ax[1]])),
        ExtentConvention::WireCrossSection(_) => Ok(cell.cell.col[ax[0]]),
    }
}

/// Clamped-ion polarizability `α` (Bohr³) and dielectric tensor `ε_∞` for a **slab or a chain**,
/// given the extent the caller assigns to it.
///
/// The companion to [`crate::pbc::dielectric_tensor`], which handles the three-dimensional case
/// where no extent has to be assigned. `α` is identical to what [`crate::pbc::polarizability`]
/// returns; all this adds is the conversion, and the conversion is
/// [`epsilon_from_polarizability`].
///
/// # Errors
///
/// - a fully periodic cell, or no cell — [`crate::pbc::dielectric_tensor`] is the function for the
///   first and there is no dielectric constant for the second;
/// - a convention that does not match the cell's dimensionality;
/// - an extent that does not fit inside the cell, which would make the assigned bodies overlap
///   their own images;
/// - an extent so small that the depolarization denominator goes non-positive — see
///   [`epsilon_from_polarizability`].
pub fn dielectric_tensor_with_extent(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    extent: ExtentConvention,
) -> Result<DielectricTensors> {
    let cell = molecule.cell.ok_or_else(|| {
        Am1Error::InvalidInput(
            "a dielectric tensor needs a cell; an isolated molecule has a polarizability and no \
             dielectric constant"
                .into(),
        )
    })?;
    let n = cell.n_periodic();
    if n == 3 {
        return Err(Am1Error::InvalidInput(
            "this cell is periodic in three dimensions, where the volume is the cell's own and \
             nothing has to be assigned. Call `pbc::dielectric_tensor`."
                .into(),
        ));
    }
    if n == 0 {
        return Err(Am1Error::InvalidInput(
            "this cell has no periodic direction. Use the molecular polarizability.".into(),
        ));
    }
    if n != extent.n_periodic() {
        let (want, got) = match extent {
            ExtentConvention::SlabThickness(_) => ("SlabThickness", "a chain"),
            ExtentConvention::WireCrossSection(_) => ("WireCrossSection", "a slab"),
        };
        return Err(Am1Error::InvalidInput(format!(
            "`ExtentConvention::{want}` describes a cell periodic in {} directions, but this is \
             {got} ({n} periodic). The two conventions carry different units and different \
             depolarization factors, so the mismatch is refused rather than reinterpreted.",
            extent.n_periodic()
        )));
    }

    // Height along the normal for a slab, transverse section for a wire — the same ratio in both
    // cases, because `measure` is exactly the part of the determinant the periodic axes span.
    let available = cell.volume().abs() / cell.measure();
    if extent.value() > available * (1.0 + 1.0e-9) {
        return Err(Am1Error::InvalidInput(format!(
            "the assigned extent {:.6} {} is larger than the {:.6} {} the cell leaves for it, so \
             the assumed bodies would overlap their own periodic images and the dielectric \
             constant would be counting the same matter twice. Assign at most the cell's own \
             extent, or enlarge the cell.",
            extent.value(),
            extent.units(),
            available,
            extent.units()
        )));
    }

    let alpha = crate::pbc::hessian::polarizability(molecule, params, options)?;
    let axis = distinguished_axis(&cell, extent)?;
    let epsilon = epsilon_from_polarizability(&alpha, axis, cell.measure(), extent)?;
    Ok((alpha, epsilon))
}

#[cfg(test)]
mod extent_component_tests {
    use super::*;

    fn iso(a: f64) -> [[f64; 3]; 3] {
        [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    }

    /// `N = 0` in every direction is `1 + 4πα/Ω`, which is what a crystal gets — so the
    /// depolarization form contains the existing three-dimensional relation rather than replacing
    /// it. Checked through the in-plane block of a slab, where `N` is 0.
    #[test]
    fn the_zero_depolarization_limit_is_the_three_dimensional_relation() {
        let alpha = iso(3.0);
        let (area, d) = (25.0, 4.0);
        let eps = epsilon_from_polarizability(
            &alpha,
            Vec3::new(0.0, 0.0, 1.0),
            area,
            ExtentConvention::SlabThickness(d),
        )
        .unwrap();
        let want = 1.0 + 4.0 * PI * 3.0 / (area * d);
        assert!(
            (eps[0][0] - want).abs() < 1.0e-14,
            "{} vs {want}",
            eps[0][0]
        );
        assert!((eps[1][1] - want).abs() < 1.0e-14);
    }

    /// The out-of-plane form is the **inverse** law, and the direction it runs in is worth
    /// pinning down, because both directions sound right until the algebra is written out.
    ///
    /// `α` here is the response to the **external** field. Out of plane the depolarizing field
    /// opposes the polarization that creates it, so:
    ///
    /// - at fixed `α`, `ε_⊥ > ε_∥` — the same external response needs a *stronger* intrinsic one
    ///   to overcome the depolarization;
    /// - at fixed `ε`, `α_⊥ < α_∥` — which is the observable statement, and the one that matters.
    ///
    /// Both are asserted, since holding only the first would pass with the law inverted.
    #[test]
    fn the_slab_normal_gets_the_inverse_law() {
        let alpha = iso(3.0);
        let (area, d) = (25.0, 4.0);
        let eps = epsilon_from_polarizability(
            &alpha,
            Vec3::new(0.0, 0.0, 1.0),
            area,
            ExtentConvention::SlabThickness(d),
        )
        .unwrap();
        let chi = 3.0 / (area * d);
        assert!((1.0 / eps[2][2] - (1.0 - 4.0 * PI * chi)).abs() < 1.0e-14);
        assert!(eps[2][2] > eps[0][0], "{} !> {}", eps[2][2], eps[0][0]);

        // Fixed `ε`, solved back for `α`: `4πα_∥/(Ad) = ε − 1` against `4πα_⊥/(Ad) = 1 − 1/ε`.
        //
        // The two are put in *separate* tensors rather than one isotropic α, because an in-plane
        // `α` large enough for `ε_∥ = 12` is far past the out-of-plane pole — which is the
        // saturation the next test is about, and would otherwise fail this one for the right
        // reason at the wrong place.
        let n = Vec3::new(0.0, 0.0, 1.0);
        let diag = |x: f64, y: f64, z: f64| [[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, z]];
        for target in [1.5, 3.0, 12.0] {
            let a_par = area * d * (target - 1.0) / (4.0 * PI);
            let a_perp = area * d * (1.0 - 1.0 / target) / (4.0 * PI);
            assert!(a_perp < a_par, "at eps={target}: {a_perp} !< {a_par}");
            let e_par = epsilon_from_polarizability(
                &diag(a_par, a_par, 0.0),
                n,
                area,
                ExtentConvention::SlabThickness(d),
            )
            .unwrap();
            let e_perp = epsilon_from_polarizability(
                &diag(0.0, 0.0, a_perp),
                n,
                area,
                ExtentConvention::SlabThickness(d),
            )
            .unwrap();
            assert!((e_par[0][0] - target).abs() < 1.0e-12, "{}", e_par[0][0]);
            assert!((e_perp[2][2] - target).abs() < 1.0e-12, "{}", e_perp[2][2]);
        }
    }

    /// The out-of-plane sheet response **saturates**: `4πα_⊥/A < d` for any material at all,
    /// because a slab cannot expel more field than a perfect conductor does.
    ///
    /// That is the physics behind the error [`scalar_epsilon`] raises. The bound is a statement
    /// about the *measured* `α`, so exceeding it means the assigned thickness is wrong — the
    /// computed response is real, and no slab of that thickness could produce it.
    #[test]
    fn the_out_of_plane_sheet_response_saturates_at_the_perfect_conductor() {
        let (area, d) = (25.0, 4.0);
        let ceiling = area * d / (4.0 * PI);
        // Approaching a perfect conductor from below: `ε_⊥ → ∞`, `α_⊥ → Ad/4π`.
        let mut last = 0.0;
        for eps in [10.0, 100.0, 10_000.0, 1.0e8] {
            let a = area * d * (1.0 - 1.0 / eps) / (4.0 * PI);
            assert!(a > last && a < ceiling, "{a} not in ({last}, {ceiling})");
            last = a;
        }
        assert!((last - ceiling).abs() / ceiling < 1.0e-7);
        // At the ceiling the law has no value, and says so rather than returning a negative one.
        assert!(epsilon_from_polarizability(
            &iso(ceiling * 1.01),
            Vec3::new(0.0, 0.0, 1.0),
            area,
            ExtentConvention::SlabThickness(d)
        )
        .is_err());
    }

    /// **Capacitor stacking.** Padding the layer with vacuum out to a larger thickness must give
    /// the same physical object: parallel in plane, series out of plane. This is an independent
    /// derivation of both formulas, so agreeing with it is a real check and not a restatement.
    #[test]
    fn the_thickness_convention_obeys_capacitor_stacking() {
        let alpha = [[2.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.8]];
        let area = 30.0;
        let n = Vec3::new(0.0, 0.0, 1.0);
        for (d1, d2) in [(2.0, 5.0), (1.0, 11.0), (4.0, 4.25)] {
            let e1 =
                epsilon_from_polarizability(&alpha, n, area, ExtentConvention::SlabThickness(d1))
                    .unwrap();
            let e2 =
                epsilon_from_polarizability(&alpha, n, area, ExtentConvention::SlabThickness(d2))
                    .unwrap();
            for a in 0..2 {
                let parallel = d2 * e2[a][a] - d1 * e1[a][a];
                assert!(
                    (parallel - (d2 - d1)).abs() < 1.0e-12,
                    "in-plane axis {a}: {parallel} vs {}",
                    d2 - d1
                );
            }
            let series = d2 / e2[2][2] - d1 / e1[2][2];
            assert!(
                (series - (d2 - d1)).abs() < 1.0e-12,
                "out of plane: {series} vs {}",
                d2 - d1
            );
        }
    }

    /// The two combinations that do **not** depend on the thickness, which is the whole of what a
    /// slab calculation can report without a convention.
    #[test]
    fn the_sheet_invariants_are_thickness_free() {
        let alpha = [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 0.8]];
        let area = 30.0;
        let n = Vec3::new(0.0, 0.0, 1.0);
        let mut inplane = Vec::new();
        let mut outplane = Vec::new();
        for d in [1.0, 3.0, 7.5, 20.0] {
            let e =
                epsilon_from_polarizability(&alpha, n, area, ExtentConvention::SlabThickness(d))
                    .unwrap();
            inplane.push((e[0][0] - 1.0) * d);
            outplane.push((1.0 - 1.0 / e[2][2]) * d);
        }
        for v in &inplane {
            assert!((v - 4.0 * PI * 2.5 / area).abs() < 1.0e-12, "{v}");
        }
        for v in &outplane {
            assert!((v - 4.0 * PI * 0.8 / area).abs() < 1.0e-12, "{v}");
        }
    }

    /// The wire's transverse law is the circular-cylinder one, `N = 1/2`, and its own invariant
    /// `S(ε−1)/(ε+1) = 2πα/L` is cross-section-free.
    #[test]
    fn the_wire_transverse_law_is_the_cylinder_one() {
        let alpha = [[4.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]];
        let length = 6.0;
        let axis = Vec3::new(1.0, 0.0, 0.0);
        for s in [3.0, 9.0, 25.0] {
            let e = epsilon_from_polarizability(
                &alpha,
                axis,
                length,
                ExtentConvention::WireCrossSection(s),
            )
            .unwrap();
            let chi_t = 1.2 / (length * s);
            let want = (1.0 + 2.0 * PI * chi_t) / (1.0 - 2.0 * PI * chi_t);
            assert!((e[1][1] - want).abs() < 1.0e-12, "{} vs {want}", e[1][1]);
            // Axial is the `N = 0` law.
            assert!((e[0][0] - (1.0 + 4.0 * PI * 4.0 / (length * s))).abs() < 1.0e-12);
            // Cross-section-free invariant.
            let inv = s * (e[1][1] - 1.0) / (e[1][1] + 1.0);
            assert!((inv - 2.0 * PI * 1.2 / length).abs() < 1.0e-12, "{inv}");
        }
    }

    /// Every law reduces to vacuum when the response does, and grows monotonically with it.
    #[test]
    fn no_polarizability_is_vacuum_and_more_of_it_screens_more() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        let zero =
            epsilon_from_polarizability(&iso(0.0), n, 10.0, ExtentConvention::SlabThickness(2.0))
                .unwrap();
        for a in 0..3 {
            for b in 0..3 {
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((zero[a][b] - want).abs() < 1.0e-15);
            }
        }
        let mut last = [1.0, 1.0];
        for a in [0.1, 0.5, 1.0, 2.0] {
            let e =
                epsilon_from_polarizability(&iso(a), n, 40.0, ExtentConvention::SlabThickness(3.0))
                    .unwrap();
            assert!(e[0][0] > last[0] && e[2][2] > last[1]);
            last = [e[0][0], e[2][2]];
        }
    }

    /// A response too large for the assigned thickness is refused, not returned negative. The
    /// out-of-plane law is the one with a pole, and this is where a silently wrong `ε < 0` would
    /// otherwise come from.
    #[test]
    fn a_thickness_the_response_does_not_fit_in_is_refused() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        // `4πχ = 1` exactly at `d = 4π α/A`.
        let (alpha_zz, area) = (1.0, 4.0);
        let critical = 4.0 * PI * alpha_zz / area;
        let ok = epsilon_from_polarizability(
            &iso(alpha_zz),
            n,
            area,
            ExtentConvention::SlabThickness(critical * 1.5),
        );
        assert!(ok.is_ok());
        let err = epsilon_from_polarizability(
            &iso(alpha_zz),
            n,
            area,
            ExtentConvention::SlabThickness(critical * 0.75),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("too small"), "{err}");
    }

    /// The result is a tensor, so it must not depend on which Cartesian axes the slab happens to
    /// lie along: rotate the cell and the polarizability together and `ε` follows.
    #[test]
    fn the_conversion_is_covariant_under_rotation() {
        let alpha = [[2.5, 0.3, 0.0], [0.3, 1.5, 0.0], [0.0, 0.0, 0.8]];
        let n = Vec3::new(0.0, 0.0, 1.0);
        let base =
            epsilon_from_polarizability(&alpha, n, 30.0, ExtentConvention::SlabThickness(3.0))
                .unwrap();

        // Rotate by 40° about x.
        let (c, s) = (0.4_f64.cos(), 0.4_f64.sin());
        let r = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]];
        let rot = |m: &[[f64; 3]; 3]| {
            let mut out = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        for l in 0..3 {
                            out[i][j] += r[i][k] * m[k][l] * r[j][l];
                        }
                    }
                }
            }
            out
        };
        let a_rot = rot(&alpha);
        let n_rot = Vec3::new(0.0, -s, c);
        let got =
            epsilon_from_polarizability(&a_rot, n_rot, 30.0, ExtentConvention::SlabThickness(3.0))
                .unwrap();
        let want = rot(&base);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (got[i][j] - want[i][j]).abs() < 1.0e-12,
                    "({i},{j}): {} vs {}",
                    got[i][j],
                    want[i][j]
                );
            }
        }
    }

    /// The mixing diagnostic is zero when the axis is principal and picks up the coupling when it
    /// is not — so a caller can tell whether the dropped block mattered.
    #[test]
    fn the_mixing_diagnostic_reports_what_is_dropped() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        let clean = [[2.5, 0.3, 0.0], [0.3, 1.5, 0.0], [0.0, 0.0, 0.8]];
        assert!(extent_axis_mixing(&clean, n) < 1.0e-15);
        let mixed = [[2.5, 0.0, 0.5], [0.0, 1.5, 0.0], [0.5, 0.0, 0.8]];
        assert!((extent_axis_mixing(&mixed, n) - 0.5 / 2.5).abs() < 1.0e-12);
    }

    /// A non-positive or non-finite extent is rejected before it can produce a division by zero.
    #[test]
    fn a_meaningless_extent_is_rejected() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(epsilon_from_polarizability(
                &iso(1.0),
                n,
                10.0,
                ExtentConvention::SlabThickness(bad)
            )
            .is_err());
        }
    }
}
