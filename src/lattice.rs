// SPDX-License-Identifier: GPL-3.0-or-later

//! Bravais lattice: fractional/Cartesian conversion, minimum image, and the direct- and
//! reciprocal-space image enumeration the periodic sums need.
//!
//! # Dimensionality
//!
//! A lattice carries a `periodic: [bool; 3]` flag per axis, so the same type describes a
//! molecule (nothing periodic), a polymer chain (one axis), a slab (two) and a crystal
//! (three). This is not cosmetic: **every enumeration below must respect those flags.**
//! Enumerating a three-dimensional box of reciprocal vectors for a slab silently computes a
//! three-dimensional answer for a two-dimensional system, and every 3D test still passes
//! while doing it. [`Lattice::reciprocal_index_ranges`] and
//! [`Lattice::reciprocal_vectors_within`] therefore collapse non-periodic axes, exactly as
//! [`Lattice::image_ranges`] does for the direct lattice.
//!
//! # Cell measure
//!
//! [`Lattice::measure`] returns the volume in 3D, the area in 2D and the length in 1D — the
//! quantity a stress or an energy density is per. Reduced dimensionality has no meaningful
//! "volume", so asking for one is a mistake the type system should not have to catch.

use crate::error::{Am1Error, Result};
use crate::math::{Mat3, Vec3};

/// Below this determinant magnitude a cell is treated as degenerate.
const DET_EPS: f64 = 1.0e-14;
/// Guard against a zero plane height when bounding image ranges.
const RANGE_EPS: f64 = 1.0e-12;

/// Integer translation `n₁a₁ + n₂a₂ + n₃a₃` identifying one periodic image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImageOffset {
    pub n: [i32; 3],
}

impl ImageOffset {
    #[inline]
    pub const fn origin() -> Self {
        Self { n: [0, 0, 0] }
    }
    #[inline]
    pub fn is_origin(self) -> bool {
        self.n == [0, 0, 0]
    }
    #[inline]
    pub fn negated(self) -> Self {
        Self {
            n: [-self.n[0], -self.n[1], -self.n[2]],
        }
    }
}

/// How many lattice directions are periodic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Periodicity {
    /// Molecular: no periodic direction.
    Molecular,
    /// A chain, periodic along one axis.
    Chain,
    /// A slab, periodic in two.
    Slab,
    /// Bulk, periodic in all three.
    Bulk,
}

impl Periodicity {
    pub fn count(self) -> usize {
        match self {
            Self::Molecular => 0,
            Self::Chain => 1,
            Self::Slab => 2,
            Self::Bulk => 3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Lattice {
    /// Lattice vectors as columns, in Bohr.
    pub cell: Mat3,
    inv_rows: [Vec3; 3],
    pub periodic: [bool; 3],
}

impl Lattice {
    pub fn new(cell: Mat3, periodic: [bool; 3]) -> Result<Self> {
        let inv_rows = cell.inverse_rows(DET_EPS).ok_or_else(|| {
            Am1Error::InvalidInput(
                "the cell vectors are linearly dependent (zero volume)".to_string(),
            )
        })?;
        Ok(Self {
            cell,
            inv_rows,
            periodic,
        })
    }

    pub fn from_vectors(a: Vec3, b: Vec3, c: Vec3, periodic: [bool; 3]) -> Result<Self> {
        Self::new(Mat3::from_columns(a, b, c), periodic)
    }

    /// Build from the crystallographic parameters, in Bohr and degrees.
    pub fn from_lengths_angles(
        a: f64,
        b: f64,
        c: f64,
        alpha_deg: f64,
        beta_deg: f64,
        gamma_deg: f64,
        periodic: [bool; 3],
    ) -> Result<Self> {
        let deg = std::f64::consts::PI / 180.0;
        let (alpha, beta, gamma) = (alpha_deg * deg, beta_deg * deg, gamma_deg * deg);
        if gamma.sin().abs() < 1.0e-12 {
            return Err(Am1Error::InvalidInput(
                "cell angle gamma cannot be 0 or 180 degrees".to_string(),
            ));
        }
        let avec = Vec3::new(a, 0.0, 0.0);
        let bvec = Vec3::new(b * gamma.cos(), b * gamma.sin(), 0.0);
        let cx = c * beta.cos();
        let cy = c * (alpha.cos() - beta.cos() * gamma.cos()) / gamma.sin();
        let cz2 = c * c - cx * cx - cy * cy;
        if cz2 <= 0.0 {
            return Err(Am1Error::InvalidInput(format!(
                "cell angles alpha={alpha_deg}, beta={beta_deg}, gamma={gamma_deg} do not \
                 describe a realizable triclinic lattice"
            )));
        }
        Self::from_vectors(avec, bvec, Vec3::new(cx, cy, cz2.sqrt()), periodic)
    }

    /// A cubic cell periodic in all three directions.
    pub fn cubic(a: f64) -> Result<Self> {
        Self::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, a, 0.0),
            Vec3::new(0.0, 0.0, a),
            [true, true, true],
        )
    }

    pub fn periodicity(&self) -> Periodicity {
        match self.periodic.iter().filter(|p| **p).count() {
            0 => Periodicity::Molecular,
            1 => Periodicity::Chain,
            2 => Periodicity::Slab,
            _ => Periodicity::Bulk,
        }
    }

    #[inline]
    pub fn n_periodic(&self) -> usize {
        self.periodic.iter().filter(|p| **p).count()
    }

    #[inline]
    pub fn is_fully_periodic(&self) -> bool {
        self.n_periodic() == 3
    }

    /// Indices of the periodic axes, in order.
    pub fn periodic_axes(&self) -> Vec<usize> {
        (0..3).filter(|&i| self.periodic[i]).collect()
    }

    /// Full 3×3 cell determinant (Bohr³), regardless of how many axes are periodic.
    ///
    /// For a slab or a chain this is the volume of the *box the cell vectors describe*, which
    /// includes the non-periodic padding and is therefore not a physical volume. Use
    /// [`Lattice::measure`] for the quantity a periodic energy is per.
    #[inline]
    pub fn volume(&self) -> f64 {
        self.cell.determinant().abs()
    }

    /// The orthogonal projector onto the span of the **periodic** cell vectors.
    ///
    /// `δ_αβ` for a crystal, the in-plane projector for a slab, `û_α û_β` for a chain, zero for a
    /// molecule. Two things want it and would otherwise each grow their own copy:
    ///
    /// * `∂(ln measure)/∂ε_αβ`, the strain derivative of the cell measure — the volume scales with
    ///   every direction, a slab's area only with the two in-plane ones.
    /// * Deciding whether a uniform field is a lattice-periodic perturbation. `F·R` is bounded
    ///   under translation by `T` exactly when `F·T = 0` for every lattice vector, i.e. when this
    ///   projector annihilates `F`.
    pub fn periodic_projector(&self) -> [[f64; 3]; 3] {
        let mut p = [[0.0; 3]; 3];
        // Gram–Schmidt over the periodic cell vectors: the projector onto their span does not
        // depend on which orthonormal basis of that span is used, so any will do.
        let mut basis: Vec<Vec3> = Vec::new();
        for axis in self.periodic_axes() {
            let mut v = self.cell.col[axis];
            for u in &basis {
                v = v - *u * u.dot(v);
            }
            let n = v.norm();
            if n > 1.0e-10 {
                basis.push(v / n);
            }
        }
        for u in &basis {
            let uv = [u.x, u.y, u.z];
            for (alpha, row) in p.iter_mut().enumerate() {
                for (beta, s) in row.iter_mut().enumerate() {
                    *s += uv[alpha] * uv[beta];
                }
            }
        }
        p
    }

    /// The part of `v` lying along the periodic directions. See [`Self::periodic_projector`].
    pub fn periodic_component(&self, v: Vec3) -> Vec3 {
        let p = self.periodic_projector();
        let x = [v.x, v.y, v.z];
        Vec3::new(
            p[0][0] * x[0] + p[0][1] * x[1] + p[0][2] * x[2],
            p[1][0] * x[0] + p[1][1] * x[1] + p[1][2] * x[2],
            p[2][0] * x[0] + p[2][1] * x[1] + p[2][2] * x[2],
        )
    }

    /// The extent of one cell along its periodic directions only: volume (Bohr³) in 3D, area
    /// (Bohr²) in 2D, length (Bohr) in 1D, and 1 for a molecule.
    pub fn measure(&self) -> f64 {
        match self.periodicity() {
            Periodicity::Molecular => 1.0,
            Periodicity::Chain => {
                let ax = self.periodic_axes()[0];
                self.cell.col[ax].norm()
            }
            Periodicity::Slab => {
                let ax = self.periodic_axes();
                self.cell.col[ax[0]].cross(self.cell.col[ax[1]]).norm()
            }
            Periodicity::Bulk => self.volume(),
        }
    }

    #[inline]
    pub fn frac_of(&self, cart: Vec3) -> Vec3 {
        Vec3::new(
            self.inv_rows[0].dot(cart),
            self.inv_rows[1].dot(cart),
            self.inv_rows[2].dot(cart),
        )
    }

    #[inline]
    pub fn cart_of(&self, frac: Vec3) -> Vec3 {
        self.cell.mul_vec(frac)
    }

    #[inline]
    pub fn inverse_rows(&self) -> [Vec3; 3] {
        self.inv_rows
    }

    /// Wrap fractional coordinates into `[0, 1)` on periodic axes.
    pub fn wrap_frac(&self, mut frac: Vec3) -> Vec3 {
        if self.periodic[0] {
            frac.x -= frac.x.floor();
        }
        if self.periodic[1] {
            frac.y -= frac.y.floor();
        }
        if self.periodic[2] {
            frac.z -= frac.z.floor();
        }
        frac
    }

    /// Wrap into `[-1/2, 1/2)` on periodic axes — the right convention for displacements,
    /// which should stay centred on zero.
    pub fn wrap_frac_centered(&self, mut frac: Vec3) -> Vec3 {
        if self.periodic[0] {
            frac.x -= (frac.x + 0.5).floor();
        }
        if self.periodic[1] {
            frac.y -= (frac.y + 0.5).floor();
        }
        if self.periodic[2] {
            frac.z -= (frac.z + 0.5).floor();
        }
        frac
    }

    pub fn wrap_cart(&self, cart: Vec3) -> Vec3 {
        self.cart_of(self.wrap_frac(self.frac_of(cart)))
    }

    pub fn minimum_image(&self, delta: Vec3) -> Vec3 {
        self.minimum_image_with_offset(delta).0
    }

    /// Minimum-image displacement, and the integer offset subtracted to reach it.
    ///
    /// Rounding each fractional component finds the nearest image only for a cell whose
    /// vectors are close to orthogonal; for a skew cell it can miss. Searching the 3×3×3
    /// stencil around the rounded offset covers the triclinic case.
    pub fn minimum_image_with_offset(&self, delta: Vec3) -> (Vec3, ImageOffset) {
        let frac = self.frac_of(delta);
        let centre = [
            if self.periodic[0] {
                frac.x.round() as i32
            } else {
                0
            },
            if self.periodic[1] {
                frac.y.round() as i32
            } else {
                0
            },
            if self.periodic[2] {
                frac.z.round() as i32
            } else {
                0
            },
        ];
        let mut best = delta;
        let mut best2 = delta.norm2();
        let mut best_off = [0_i32; 3];
        let span = |p: bool| if p { -1..=1 } else { 0..=0 };
        for ix in span(self.periodic[0]) {
            for iy in span(self.periodic[1]) {
                for iz in span(self.periodic[2]) {
                    let off = [centre[0] + ix, centre[1] + iy, centre[2] + iz];
                    let cart = self.cart_of(Vec3::new(
                        frac.x - off[0] as f64,
                        frac.y - off[1] as f64,
                        frac.z - off[2] as f64,
                    ));
                    let r2 = cart.norm2();
                    if r2 < best2 {
                        best = cart;
                        best2 = r2;
                        best_off = off;
                    }
                }
            }
        }
        (best, ImageOffset { n: best_off })
    }

    #[inline]
    pub fn translation(&self, offset: ImageOffset) -> Vec3 {
        self.cell.col[0] * offset.n[0] as f64
            + self.cell.col[1] * offset.n[1] as f64
            + self.cell.col[2] * offset.n[2] as f64
    }

    /// Perpendicular spacing between the lattice planes normal to `axis`.
    ///
    /// This, not the vector length, is what bounds how many images a cutoff sphere reaches:
    /// for a strongly skewed cell the two differ by a large factor.
    pub fn plane_height(&self, axis: usize) -> f64 {
        let (u, v, w) = match axis {
            0 => (self.cell.col[0], self.cell.col[1], self.cell.col[2]),
            1 => (self.cell.col[1], self.cell.col[2], self.cell.col[0]),
            _ => (self.cell.col[2], self.cell.col[0], self.cell.col[1]),
        };
        let normal = v.cross(w);
        let n = normal.norm();
        if n < RANGE_EPS {
            0.0
        } else {
            (u.dot(normal) / n).abs()
        }
    }

    /// Conservative integer image range reaching a sphere of radius `cutoff`. Non-periodic
    /// axes get range 0.
    pub fn image_ranges(&self, cutoff: f64) -> [i32; 3] {
        let mut range = [0_i32; 3];
        if cutoff <= 0.0 {
            return range;
        }
        for axis in 0..3 {
            if self.periodic[axis] {
                let h = self.plane_height(axis).max(RANGE_EPS);
                range[axis] = (cutoff / h).ceil() as i32 + 1;
            }
        }
        range
    }

    /// Every image offset whose translation lies within `cutoff`, nearest first.
    ///
    /// Sorted by distance so a real-space sum accumulates the largest contributions first and
    /// its convergence can be watched by truncating the list.
    pub fn image_offsets(&self, cutoff: f64) -> Vec<ImageOffset> {
        let range = self.image_ranges(cutoff);
        let mut out = Vec::new();
        for i in -range[0]..=range[0] {
            for j in -range[1]..=range[1] {
                for k in -range[2]..=range[2] {
                    let off = ImageOffset { n: [i, j, k] };
                    if off.is_origin() || self.translation(off).norm() <= cutoff {
                        out.push(off);
                    }
                }
            }
        }
        out.sort_by(|a, b| {
            let (ra, rb) = (self.translation(*a).norm2(), self.translation(*b).norm2());
            ra.partial_cmp(&rb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.n.cmp(&b.n))
        });
        out
    }

    /// Reciprocal lattice vectors including the `2π`, so `bᵢ · aⱼ = 2π δᵢⱼ`.
    ///
    /// Returned for all three axes regardless of periodicity; callers that enumerate must
    /// restrict themselves to the periodic ones (see [`Lattice::reciprocal_vectors_within`]).
    pub fn reciprocal_vectors_2pi(&self) -> [Vec3; 3] {
        let two_pi = 2.0 * std::f64::consts::PI;
        [
            self.inv_rows[0] * two_pi,
            self.inv_rows[1] * two_pi,
            self.inv_rows[2] * two_pi,
        ]
    }

    /// Integer ranges covering all reciprocal vectors of length `<= cutoff`.
    ///
    /// **Non-periodic axes collapse to 0.** A slab has no reciprocal direction normal to its
    /// surface, and enumerating one produces a three-dimensional sum for a two-dimensional
    /// system — which converges to the wrong answer while looking entirely healthy.
    pub fn reciprocal_index_ranges(&self, cutoff: f64) -> [i32; 3] {
        let mut range = [0_i32; 3];
        if cutoff <= 0.0 {
            return range;
        }
        let b = self.reciprocal_vectors_2pi();
        for axis in 0..3 {
            if !self.periodic[axis] {
                continue;
            }
            // Same plane-height argument as the direct lattice, in reciprocal space.
            let (u, v, w) = match axis {
                0 => (b[0], b[1], b[2]),
                1 => (b[1], b[2], b[0]),
                _ => (b[2], b[0], b[1]),
            };
            let normal = v.cross(w);
            let n = normal.norm();
            let h = if n < RANGE_EPS {
                u.norm()
            } else {
                (u.dot(normal) / n).abs()
            };
            range[axis] = (cutoff / h.max(RANGE_EPS)).ceil() as i32 + 1;
        }
        range
    }

    /// Reciprocal vectors with `0 < |G| <= cutoff`, nearest first, excluding `G = 0`.
    ///
    /// Restricted to the periodic axes, so a chain enumerates a line of `G` and a slab a
    /// plane.
    pub fn reciprocal_vectors_within(&self, cutoff: f64) -> Vec<(ImageOffset, Vec3)> {
        let range = self.reciprocal_index_ranges(cutoff);
        let b = self.reciprocal_vectors_2pi();
        let mut out = Vec::new();
        for i in -range[0]..=range[0] {
            for j in -range[1]..=range[1] {
                for k in -range[2]..=range[2] {
                    if i == 0 && j == 0 && k == 0 {
                        continue;
                    }
                    let g = b[0] * i as f64 + b[1] * j as f64 + b[2] * k as f64;
                    if g.norm() <= cutoff {
                        out.push((ImageOffset { n: [i, j, k] }, g));
                    }
                }
            }
        }
        out.sort_by(|a, b| {
            a.1.norm2()
                .partial_cmp(&b.1.norm2())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.n.cmp(&b.0.n))
        });
        out
    }

    /// Scale the periodic lattice vectors by `factor`, leaving non-periodic axes alone.
    /// Used by isotropic-strain finite differences.
    pub fn scaled(&self, factor: f64) -> Result<Self> {
        let mut cols = self.cell.col;
        for (axis, col) in cols.iter_mut().enumerate() {
            if self.periodic[axis] {
                *col = *col * factor;
            }
        }
        Self::from_vectors(cols[0], cols[1], cols[2], self.periodic)
    }

    /// Apply a homogeneous strain `F = I + ε` to the lattice vectors.
    pub fn strained(&self, eps: &[[f64; 3]; 3]) -> Result<Self> {
        let deform = |v: Vec3| {
            Vec3::new(
                v.x + eps[0][0] * v.x + eps[0][1] * v.y + eps[0][2] * v.z,
                v.y + eps[1][0] * v.x + eps[1][1] * v.y + eps[1][2] * v.z,
                v.z + eps[2][0] * v.x + eps[2][1] * v.y + eps[2][2] * v.z,
            )
        };
        Self::from_vectors(
            deform(self.cell.col[0]),
            deform(self.cell.col[1]),
            deform(self.cell.col[2]),
            self.periodic,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triclinic() -> Lattice {
        Lattice::from_vectors(
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(3.5, 9.2, 0.0),
            Vec3::new(1.7, -2.3, 8.4),
            [true, true, true],
        )
        .unwrap()
    }

    #[test]
    fn fractional_and_cartesian_round_trip() {
        let l = triclinic();
        for p in [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.5, 0.25, 7.0),
            Vec3::new(0.0, 0.0, 0.0),
        ] {
            let back = l.cart_of(l.frac_of(p));
            assert!((back - p).norm() < 1.0e-12, "round trip failed for {p:?}");
        }
    }

    #[test]
    fn reciprocal_basis_is_dual_to_the_direct_basis() {
        let l = triclinic();
        let b = l.reciprocal_vectors_2pi();
        let two_pi = 2.0 * std::f64::consts::PI;
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { two_pi } else { 0.0 };
                let got = b[i].dot(l.cell.col[j]);
                assert!(
                    (got - want).abs() < 1.0e-10,
                    "b{i}·a{j} = {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn minimum_image_beats_naive_rounding_in_a_skew_cell() {
        // A deliberately skewed cell where rounding each fractional coordinate does not give
        // the nearest image.
        let l = Lattice::from_vectors(
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(9.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
            [true, true, true],
        )
        .unwrap();
        let delta = Vec3::new(5.2, 1.9, 0.0);
        let (mi, off) = l.minimum_image_with_offset(delta);

        // Brute force over a generous stencil.
        let mut best = delta;
        for i in -3..=3 {
            for j in -3..=3 {
                for k in -3..=3 {
                    let t = l.translation(ImageOffset { n: [i, j, k] });
                    let cand = delta - t;
                    if cand.norm2() < best.norm2() {
                        best = cand;
                    }
                }
            }
        }
        assert!(
            (mi.norm() - best.norm()).abs() < 1.0e-10,
            "minimum image {:.6} is not the true nearest {:.6} (offset {:?})",
            mi.norm(),
            best.norm(),
            off.n
        );
    }

    #[test]
    fn image_ranges_use_plane_height_not_vector_length() {
        // For this cell |b| = sqrt(81+16) ~ 9.85 but the plane spacing along b is only 4.
        let l = Lattice::from_vectors(
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(9.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
            [true, true, true],
        )
        .unwrap();
        assert!((l.plane_height(1) - 4.0).abs() < 1.0e-10);
        let r = l.image_ranges(20.0);
        assert!(r[1] >= 6, "range along the skewed axis is only {}", r[1]);
    }

    #[test]
    fn non_periodic_axes_are_excluded_from_both_enumerations() {
        // The bug this guards: enumerating a 3-D box of images or reciprocal vectors for a
        // slab or a chain gives a 3-D answer for a lower-dimensional system, and every
        // fully-periodic test still passes.
        let slab = Lattice::from_vectors(
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 8.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap();
        assert_eq!(slab.periodicity(), Periodicity::Slab);
        assert_eq!(slab.image_ranges(20.0)[2], 0);
        assert!(slab.image_offsets(20.0).iter().all(|o| o.n[2] == 0));
        assert_eq!(slab.reciprocal_index_ranges(3.0)[2], 0);
        assert!(slab
            .reciprocal_vectors_within(3.0)
            .iter()
            .all(|(o, _)| o.n[2] == 0));

        let chain = Lattice::from_vectors(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap();
        assert_eq!(chain.periodicity(), Periodicity::Chain);
        assert!(chain
            .image_offsets(30.0)
            .iter()
            .all(|o| o.n[1] == 0 && o.n[2] == 0));
        assert!(chain
            .reciprocal_vectors_within(5.0)
            .iter()
            .all(|(o, _)| o.n[1] == 0 && o.n[2] == 0));
    }

    #[test]
    fn measure_is_volume_area_or_length_by_dimensionality() {
        let bulk = Lattice::cubic(8.0).unwrap();
        assert!((bulk.measure() - 512.0).abs() < 1.0e-10);

        let slab = Lattice::from_vectors(
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 6.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap();
        assert!((slab.measure() - 48.0).abs() < 1.0e-10, "slab area");
        // The full determinant is not the physical measure for a slab.
        assert!((slab.volume() - 1920.0).abs() < 1.0e-10);

        let chain = Lattice::from_vectors(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap();
        assert!((chain.measure() - 5.0).abs() < 1.0e-10, "chain length");
    }

    #[test]
    fn image_offsets_are_sorted_nearest_first_and_include_the_origin() {
        let l = triclinic();
        let offs = l.image_offsets(25.0);
        assert!(offs[0].is_origin(), "the origin should come first");
        let mut prev = -1.0;
        for o in &offs {
            let r = l.translation(*o).norm();
            assert!(
                r >= prev - 1.0e-9,
                "image offsets are not sorted by distance"
            );
            prev = r;
        }
    }

    #[test]
    fn a_degenerate_cell_is_rejected() {
        let err = Lattice::from_vectors(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            [true, true, true],
        )
        .unwrap_err();
        assert!(err.to_string().contains("linearly dependent"), "{err}");
    }

    #[test]
    fn strain_and_scaling_preserve_periodicity_flags() {
        let slab = Lattice::from_vectors(
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 6.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap();
        // Isotropic scaling touches only the periodic axes: the vacuum gap of a slab is not a
        // degree of freedom the cell owns.
        let s = slab.scaled(1.1).unwrap();
        assert!((s.cell.col[0].x - 8.8).abs() < 1.0e-12);
        assert!((s.cell.col[2].z - 40.0).abs() < 1.0e-12);
        assert_eq!(s.periodic, slab.periodic);

        let eps = [[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let d = slab.strained(&eps).unwrap();
        assert!((d.cell.col[0].x - 8.08).abs() < 1.0e-12);
        assert_eq!(d.periodic, slab.periodic);
    }
}

#[cfg(test)]
mod projector_tests {
    use super::*;

    fn cell(periodic: [bool; 3]) -> Lattice {
        // Deliberately non-orthogonal, so a projector that only works for a cubic cell fails.
        Lattice::from_vectors(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(1.5, 5.0, 0.0),
            Vec3::new(0.4, -0.9, 7.0),
            periodic,
        )
        .unwrap()
    }

    /// A projector is symmetric, idempotent, and has trace equal to the dimension it projects on.
    ///
    /// All three are checked because each catches a different mistake: a non-symmetric one comes
    /// from forgetting to orthonormalize, a non-idempotent one from summing un-normalized cell
    /// vectors, and a wrong trace from including a non-periodic axis. The strain derivative of the
    /// cell measure is this projector, so any of the three would show up as a wrong stress in
    /// reduced dimensionality — where there are the fewest tests to catch it.
    #[test]
    fn the_periodic_projector_is_a_projector_of_the_right_rank() {
        for (label, periodic, rank) in [
            ("chain", [true, false, false], 1usize),
            ("slab", [true, true, false], 2),
            ("crystal", [true, true, true], 3),
            ("molecule", [false, false, false], 0),
        ] {
            let p = cell(periodic).periodic_projector();
            let mut trace = 0.0;
            let mut worst_sym = 0.0_f64;
            let mut worst_idem = 0.0_f64;
            for i in 0..3 {
                trace += p[i][i];
                for j in 0..3 {
                    worst_sym = worst_sym.max((p[i][j] - p[j][i]).abs());
                    let pp: f64 = (0..3).map(|k| p[i][k] * p[k][j]).sum();
                    worst_idem = worst_idem.max((pp - p[i][j]).abs());
                }
            }
            assert!(
                worst_sym < 1.0e-14,
                "{label}: P is not symmetric ({worst_sym:.3e})"
            );
            assert!(worst_idem < 1.0e-14, "{label}: P² ≠ P ({worst_idem:.3e})");
            assert!(
                (trace - rank as f64).abs() < 1.0e-12,
                "{label}: trace P = {trace}, expected {rank}"
            );
        }
    }

    /// The projector must annihilate exactly the directions a field is allowed to point along, and
    /// leave the lattice vectors alone.
    ///
    /// This is the property `PbcOptions::electric_field` is checked against: `F·R` repeats with the
    /// lattice exactly when `P F = 0`. Testing it on the vectors themselves rather than on a
    /// convenient axis is what makes it a statement about the span and not about the basis.
    #[test]
    fn the_projector_keeps_lattice_vectors_and_kills_the_normal() {
        let slab = cell([true, true, false]);
        for axis in [0usize, 1] {
            let v = slab.cell.col[axis];
            let kept = slab.periodic_component(v);
            assert!(
                (kept - v).norm() < 1.0e-12 * v.norm(),
                "the projector moved lattice vector {axis}"
            );
        }
        // The normal to the two in-plane vectors must be annihilated.
        let n = slab.cell.col[0].cross(slab.cell.col[1]);
        let killed = slab.periodic_component(n);
        assert!(
            killed.norm() < 1.0e-12 * n.norm(),
            "the projector kept {:.3e} of the slab normal",
            killed.norm() / n.norm()
        );
    }
}
