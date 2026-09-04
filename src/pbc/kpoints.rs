// SPDX-License-Identifier: GPL-3.0-or-later

//! Brillouin-zone sampling.
//!
//! A k-point is stored by its **fractional** coordinates, because that is all the Bloch phase
//! needs: with `b_i · a_j = 2π δ_ij`, the phase of an image at integer offset `n` is
//!
//! ```text
//! e^{i k·T} = e^{i 2π (f₁n₁ + f₂n₂ + f₃n₃)}
//! ```
//!
//! so the Cartesian reciprocal vectors never enter. Non-periodic axes collapse to a single
//! `f = 0`, which is what makes one mesh type serve chains, slabs and crystals.

use crate::error::{Am1Error, Result};
use crate::lattice::{ImageOffset, Lattice};

/// One sampling point of the Brillouin zone.
#[derive(Clone, Copy, Debug)]
pub struct KPoint {
    /// Fractional coordinates, folded into `(-1/2, 1/2]`.
    pub fractional: [f64; 3],
    /// Weight; the weights of a mesh sum to 1.
    pub weight: f64,
}

impl KPoint {
    pub const fn gamma() -> Self {
        Self {
            fractional: [0.0, 0.0, 0.0],
            weight: 1.0,
        }
    }

    /// `(cos, sin)` of `e^{i k·T}` for the image at integer offset `n`.
    #[inline]
    pub fn phase(&self, offset: ImageOffset) -> (f64, f64) {
        let theta = std::f64::consts::TAU
            * (self.fractional[0] * offset.n[0] as f64
                + self.fractional[1] * offset.n[1] as f64
                + self.fractional[2] * offset.n[2] as f64);
        (theta.cos(), theta.sin())
    }

    /// Whether this point is its own time-reversal image, i.e. `k ≡ −k` modulo a reciprocal
    /// lattice vector. At such a point every Bloch phase is ±1 and the Hamiltonian is real.
    pub fn is_real(&self) -> bool {
        self.fractional.iter().all(|f| {
            let d = 2.0 * f;
            (d - d.round()).abs() < 1.0e-9
        })
    }
}

/// A Brillouin-zone mesh specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KMesh {
    /// Γ only.
    #[default]
    Gamma,
    /// Monkhorst–Pack grid of the given size, Γ-centred.
    MonkhorstPack([usize; 3]),
    /// Monkhorst–Pack grid with the standard `(2r − n + 1)/(2n)` offset, which excludes Γ for
    /// even `n`.
    MonkhorstPackShifted([usize; 3]),
}

impl KMesh {
    /// Expand into explicit k-points for `lattice`.
    ///
    /// Non-periodic axes are collapsed to one point regardless of the requested size — a slab
    /// has no dispersion normal to its surface, and sampling it would be sampling nothing.
    /// When `fold_time_reversal` is set, `k` and `−k` are merged and their weights added,
    /// which is exact for a real Hamiltonian and roughly halves the number of
    /// diagonalizations.
    pub fn resolve(&self, lattice: &Lattice, fold_time_reversal: bool) -> Result<Vec<KPoint>> {
        let (size, shifted) = match self {
            Self::Gamma => ([1, 1, 1], false),
            Self::MonkhorstPack(n) => (*n, false),
            Self::MonkhorstPackShifted(n) => (*n, true),
        };
        for (axis, n) in size.iter().enumerate() {
            if *n == 0 {
                return Err(Am1Error::InvalidInput(format!(
                    "k-mesh size along axis {axis} must be at least 1"
                )));
            }
        }

        let m = [
            if lattice.periodic[0] { size[0] } else { 1 },
            if lattice.periodic[1] { size[1] } else { 1 },
            if lattice.periodic[2] { size[2] } else { 1 },
        ];
        let total = m[0] * m[1] * m[2];
        let weight = 1.0 / total as f64;

        let mut points = Vec::with_capacity(total);
        for a in 0..m[0] {
            for b in 0..m[1] {
                for c in 0..m[2] {
                    points.push(KPoint {
                        fractional: [
                            mp_coordinate(a, m[0], shifted),
                            mp_coordinate(b, m[1], shifted),
                            mp_coordinate(c, m[2], shifted),
                        ],
                        weight,
                    });
                }
            }
        }

        Ok(if fold_time_reversal {
            fold(&points)
        } else {
            points
        })
    }

    pub fn sizes(&self) -> [usize; 3] {
        match self {
            Self::Gamma => [1, 1, 1],
            Self::MonkhorstPack(n) | Self::MonkhorstPackShifted(n) => *n,
        }
    }

    pub fn is_gamma_only(&self) -> bool {
        self.sizes() == [1, 1, 1]
    }
}

/// Fractional coordinate of grid index `r` out of `n`, folded into `(-1/2, 1/2]`.
fn mp_coordinate(r: usize, n: usize, shifted: bool) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let raw = if shifted {
        (2 * r as i64 - n as i64 + 1) as f64 / (2.0 * n as f64)
    } else {
        r as f64 / n as f64
    };
    let wrapped = raw - (raw + 0.5).floor();
    if wrapped <= -0.5 {
        wrapped + 1.0
    } else {
        wrapped
    }
}

/// Merge `k` with `−k`, accumulating weights.
fn fold(points: &[KPoint]) -> Vec<KPoint> {
    let close = |a: f64, b: f64| {
        let d = a - b;
        (d - d.round()).abs() < 1.0e-9
    };
    let mut reduced: Vec<KPoint> = Vec::new();
    'outer: for kp in points {
        for existing in &mut reduced {
            let same = (0..3).all(|i| close(existing.fractional[i], kp.fractional[i]));
            let negated = (0..3).all(|i| close(existing.fractional[i], -kp.fractional[i]));
            if same || negated {
                existing.weight += kp.weight;
                continue 'outer;
            }
        }
        reduced.push(*kp);
    }
    reduced
}

/// Whether every point of a mesh is its own time-reversal image, so the whole calculation can
/// stay in real arithmetic.
pub fn all_real(points: &[KPoint]) -> bool {
    points.iter().all(|k| k.is_real())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    fn bulk() -> Lattice {
        Lattice::cubic(8.0).unwrap()
    }

    fn slab() -> Lattice {
        Lattice::from_vectors(
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 8.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap()
    }

    #[test]
    fn gamma_is_a_single_point_of_unit_weight() {
        let pts = KMesh::Gamma.resolve(&bulk(), false).unwrap();
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].fractional, [0.0, 0.0, 0.0]);
        assert!((pts[0].weight - 1.0).abs() < 1.0e-15);
        assert!(pts[0].is_real());
    }

    #[test]
    fn weights_always_sum_to_one() {
        for mesh in [
            KMesh::Gamma,
            KMesh::MonkhorstPack([4, 4, 4]),
            KMesh::MonkhorstPackShifted([3, 2, 5]),
        ] {
            for fold in [false, true] {
                let pts = mesh.resolve(&bulk(), fold).unwrap();
                let w: f64 = pts.iter().map(|p| p.weight).sum();
                assert!(
                    (w - 1.0).abs() < 1.0e-12,
                    "{mesh:?} fold={fold} weights sum to {w}"
                );
            }
        }
    }

    #[test]
    fn non_periodic_axes_collapse() {
        // A slab has no dispersion normal to its surface; asking for 4 points there must not
        // produce four copies of the same calculation with a quarter weight each.
        let pts = KMesh::MonkhorstPack([4, 4, 4])
            .resolve(&slab(), false)
            .unwrap();
        assert_eq!(pts.len(), 16, "expected a 4x4x1 mesh, got {}", pts.len());
        assert!(pts.iter().all(|p| p.fractional[2] == 0.0));
    }

    #[test]
    fn time_reversal_folding_preserves_weight_and_shrinks_the_mesh() {
        let full = KMesh::MonkhorstPack([4, 4, 4])
            .resolve(&bulk(), false)
            .unwrap();
        let folded = KMesh::MonkhorstPack([4, 4, 4])
            .resolve(&bulk(), true)
            .unwrap();
        assert!(folded.len() < full.len());
        let w: f64 = folded.iter().map(|p| p.weight).sum();
        assert!((w - 1.0).abs() < 1.0e-12);
        eprintln!(
            "    4x4x4: {} points -> {} after folding",
            full.len(),
            folded.len()
        );
    }

    #[test]
    fn the_gamma_phase_is_unity_and_a_zone_boundary_phase_is_real() {
        let g = KPoint::gamma();
        let (c, s) = g.phase(ImageOffset { n: [3, -2, 1] });
        assert!((c - 1.0).abs() < 1.0e-12 && s.abs() < 1.0e-12);

        // k = (1/2, 0, 0) is its own time-reversal image: phases are +-1.
        let x = KPoint {
            fractional: [0.5, 0.0, 0.0],
            weight: 1.0,
        };
        assert!(x.is_real());
        let (c, s) = x.phase(ImageOffset { n: [1, 0, 0] });
        assert!((c + 1.0).abs() < 1.0e-12 && s.abs() < 1.0e-12);
        let (c, s) = x.phase(ImageOffset { n: [2, 0, 0] });
        assert!((c - 1.0).abs() < 1.0e-12 && s.abs() < 1.0e-12);

        // A general point is not real.
        let g = KPoint {
            fractional: [0.25, 0.0, 0.0],
            weight: 1.0,
        };
        assert!(!g.is_real());
    }

    #[test]
    fn a_gamma_centred_mesh_contains_gamma_and_a_shifted_even_mesh_does_not() {
        let centred = KMesh::MonkhorstPack([2, 2, 2])
            .resolve(&bulk(), false)
            .unwrap();
        assert!(centred.iter().any(|p| p.fractional == [0.0, 0.0, 0.0]));

        let shifted = KMesh::MonkhorstPackShifted([2, 2, 2])
            .resolve(&bulk(), false)
            .unwrap();
        assert!(!shifted.iter().any(|p| p.fractional == [0.0, 0.0, 0.0]));
    }

    #[test]
    fn a_zero_sized_mesh_is_rejected() {
        let err = KMesh::MonkhorstPack([2, 0, 2])
            .resolve(&bulk(), false)
            .unwrap_err();
        assert!(err.to_string().contains("at least 1"), "{err}");
    }

    #[test]
    fn all_real_detects_the_meshes_that_stay_in_real_arithmetic() {
        assert!(all_real(&KMesh::Gamma.resolve(&bulk(), false).unwrap()));
        // A 2x2x2 Gamma-centred mesh has only 0 and 1/2 components, all self-conjugate.
        assert!(all_real(
            &KMesh::MonkhorstPack([2, 2, 2])
                .resolve(&bulk(), false)
                .unwrap()
        ));
        // A 3x3x3 mesh has 1/3 components, which are not.
        assert!(!all_real(
            &KMesh::MonkhorstPack([3, 3, 3])
                .resolve(&bulk(), false)
                .unwrap()
        ));
    }
}
