// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO two-center two-electron integrals (Dewar–Sabelli–Klopman multipole model) and the
//! electron–core attraction integrals.
//!
//! The kernels are written generically over [`crate::dual::Scalar`]: instantiating them at
//! `f64` gives the (validated) energy path; instantiating at [`crate::dual::Dual`] gives the
//! exact derivatives with respect to an interatomic displacement, used by the fully analytic
//! gradient. Distances are in Bohr, energies in eV (`AM1_EV`); orbital order is `s,px,py,pz`.

use crate::constants::AM1_EV;
use crate::dual::{Dual, Scalar};
use crate::math::Vec3;
use crate::params::Am1Element;

/// Lower-triangle pack indices for the sixteen orbital pairs of a 4-orbital atom block.
///
/// A table rather than the arithmetic, because [`pack`] sits in the innermost loop of the
/// two-centre Fock contraction — three nests of it per pair, tens of millions of pairs over a
/// Hessian — where a branch and a multiply are a measurable share of an operation that is
/// otherwise one multiply-add. Bit-identical to the closed form by construction, which
/// `the_pack_table_is_the_closed_form` asserts over the whole domain.
pub const PACK: [[usize; 4]; 4] = [[0, 1, 3, 6], [1, 2, 4, 7], [3, 4, 5, 8], [6, 7, 8, 9]];

/// Lower-triangle pack index of an orbital pair `(a, b)` within a 4-orbital atom block.
///
/// Prefer [`PACK`] on a hot path; this is the definition the table is checked against and the
/// entry point for anything outside `0..4`.
#[inline]
pub fn pack(a: usize, b: usize) -> usize {
    let (h, l) = if a >= b { (a, b) } else { (b, a) };
    h * (h + 1) / 2 + l
}

#[cfg(test)]
mod pack_tests {
    use super::{pack, PACK};

    /// The table has to be the closed form on every index a 4-orbital block can produce.
    ///
    /// A transcribed lookup table is exactly the kind of optimization that is correct when written
    /// and wrong after an unrelated edit, and a wrong pack index does not crash — it contracts the
    /// density against the wrong integral and shifts an energy. Sixteen comparisons settle it.
    #[test]
    fn the_pack_table_is_the_closed_form() {
        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(PACK[a][b], pack(a, b), "PACK[{a}][{b}]");
            }
        }
        // And it is symmetric, which is what makes one triangle enough.
        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(PACK[a][b], PACK[b][a]);
            }
        }
    }
}

/// Rotation matrix (rows) that rotates the unit vector `v` onto +x (`R·v = (1,0,0)`),
/// generic over the scalar type. Port of PySEQM `rotate_with_quaternion`.
///
/// **Not used by the integral kernels.** The shortest-arc construction below is singular at
/// `v = -x̂`, where it falls back to a constant quaternion — correct as a *value*, but it
/// discards the derivative information carried by a `Dual`/`Dual2` argument. The kernels
/// therefore work with [`transverse_projector`] instead, which needs no frame at all.
/// Retained because it is part of the public API and is a convenient orthonormal frame.
pub fn rotation_to_x_g<S: Scalar>(vx: S, vy: S, vz: S) -> [[S; 3]; 3] {
    let mut qx = S::cst(0.0);
    let mut qy = vz;
    let mut qz = -vy;
    let mut qw = vx + 1.0;
    if qw.val().abs() < 1.0e-7 {
        qx = S::cst(0.0);
        qy = S::cst(0.0);
        qz = S::cst(1.0);
        qw = S::cst(0.0);
    }
    let norm = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
    let inv = norm.recip();
    qx = qx * inv;
    qy = qy * inv;
    qz = qz * inv;
    qw = qw * inv;
    let _ = qx;
    [
        [
            (qy * qy + qz * qz) * (-2.0) + 1.0,
            qz * qw * (-2.0),
            qy * qw * 2.0,
        ],
        [qz * qw * 2.0, qz * qz * (-2.0) + 1.0, qy * qz * 2.0],
        [qy * qw * (-2.0), qy * qz * 2.0, qy * qy * (-2.0) + 1.0],
    ]
}

/// Rotation matrix for an `f64` unit vector (used by the overlap f64 path).
pub fn rotation_to_x(v: Vec3) -> [[f64; 3]; 3] {
    rotation_to_x_g(v.x, v.y, v.z)
}

/// Transverse projector `p = I − n ⊗ n` of a unit diatomic axis `n`.
///
/// The two-centre kernels never use the individual transverse directions `r1`, `r2` — they
/// only ever use the combination `r1_a r1_b + r2_a r2_b`, which **is** this projector for
/// any orthonormal completion `{n, r1, r2}` of the axis. Carrying `p` instead of an explicit
/// transverse frame removes the frame from the problem entirely.
///
/// That matters for derivatives. No globally smooth choice of `r1`, `r2` exists over the
/// sphere, so every explicit construction is singular somewhere — the quaternion in
/// [`rotation_to_x_g`] is singular at `n = -x̂`, precisely where an axis-aligned lattice puts
/// a large fraction of its neighbour images. `p`, by contrast, is a quadratic polynomial in
/// `n`: it differentiates exactly, to all orders, at every orientation. It is also cheaper
/// (no quaternion norm, no reciprocal square root).
#[inline]
pub fn transverse_projector<S: Scalar>(n: [S; 3]) -> [[S; 3]; 3] {
    let mut p = [[S::cst(0.0); 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let nn = -(n[a] * n[b]);
            p[a][b] = if a == b { nn + 1.0 } else { nn };
        }
    }
    p
}

/// Rotated two-electron integrals + electron–core attractions for one ordered atom pair,
/// generic over the scalar type.
pub struct PairTwoElecG<S: Scalar> {
    pub norb_i: usize,
    pub norb_j: usize,
    /// `(a_i b_i | c_j d_j)` in eV, row-major over the packed pair indices, so entry
    /// `(pack_i(a,b), pack_j(c,d))` lives at `pack_i * w_cols + pack_j`.
    ///
    /// Flat, not `Vec<Vec<_>>`. The nested form cost eleven heap allocations per atom pair
    /// and, for the heavy–heavy case, an explicit copy out of the flat `[S; 100]` the
    /// rotation had just produced. Every atom pair in the molecule is materialized, so at 801
    /// atoms that was 320k pairs times eleven allocations.
    w: Vec<S>,
    w_cols: usize,
    pub e1b: [[S; 4]; 4],
    pub e2a: [[S; 4]; 4],
}

impl<S: Scalar> PairTwoElecG<S> {
    #[inline]
    pub fn two_e(&self, a: usize, b: usize, c: usize, d: usize) -> S {
        self.w[pack(a, b) * self.w_cols + pack(c, d)]
    }

    /// Entry by packed indices, for tests that walk the block directly.
    #[inline]
    pub fn w_at(&self, i: usize, j: usize) -> S {
        self.w[i * self.w_cols + j]
    }

    /// The whole row of integrals for a fixed bra pair `(a, b)`, indexable by `pack(c, d)`.
    ///
    /// Hoisting this out of an inner loop matters more than it looks. The Fock build's innermost
    /// loop runs `pack(a,b) * w_cols + pack(c,d)` per element, so for a fixed bra it recomputes
    /// the same row offset sixteen times and pays a bounds check on a `Vec` each time — on a
    /// 150-atom cluster the CPHF solver executes that inner statement around 7 × 10⁹ times.
    #[inline]
    pub fn two_e_row(&self, a: usize, b: usize) -> &[S] {
        let start = pack(a, b) * self.w_cols;
        &self.w[start..start + self.w_cols]
    }
}

/// f64 alias used by the SCF/Fock.
pub type PairTwoElec = PairTwoElecG<f64>;

/// Compute the rotated two-electron integrals for the ordered pair (i, j) with `xij` the unit
/// vector from i to j and `r` the distance in Bohr (f64 energy path).
pub fn pair_two_electron(ei: &Am1Element, ej: &Am1Element, xij: Vec3, r: f64) -> PairTwoElec {
    pair_two_electron_g(ei, ej, [xij.x * r, xij.y * r, xij.z * r])
}

/// Generic driver: `dvec` is the interatomic displacement `R_j − R_i`. Seeding `dvec` with
/// [`Dual`] variables yields the integral derivatives.
pub fn pair_two_electron_g<S: Scalar>(
    ei: &Am1Element,
    ej: &Am1Element,
    dvec: [S; 3],
) -> PairTwoElecG<S> {
    let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    let inv = r.recip();
    let xij = [dvec[0] * inv, dvec[1] * inv, dvec[2] * inv];

    let heavy_i = ei.has_p();
    let heavy_j = ej.has_p();
    let mut e1b = [[S::cst(0.0); 4]; 4];
    let mut e2a = [[S::cst(0.0); 4]; 4];

    // Diatomic frame, held as the axis plus the transverse projector rather than as an
    // explicit orthonormal triad. The two-electron frame axis is v = -xij (PySEQM convention).
    let axis = [-xij[0], -xij[1], -xij[2]];
    let proj = transverse_projector(axis);

    if !heavy_i && !heavy_j {
        let aee = (ei.rho0 + ej.rho0).powi(2);
        let ee = (r * r + aee).sqrt().recip() * AM1_EV;
        e1b[0][0] = ee * (-ej.core_charge);
        e2a[0][0] = ee * (-ei.core_charge);
        return PairTwoElecG {
            norb_i: 1,
            norb_j: 1,
            w: vec![ee],
            w_cols: 1,
            e1b,
            e2a,
        };
    }

    if heavy_i && !heavy_j {
        let ri = local_xh_g(ei, ej, r);
        let mut wxh = [S::cst(0.0); 10];
        build_wxh_g(&ri, &axis, &proj, &mut wxh);
        for a in 0..4 {
            for b in 0..4 {
                e1b[a][b] = wxh[pack(a, b)] * (-ej.core_charge);
            }
        }
        e2a[0][0] = wxh[0] * (-ei.core_charge);
        return PairTwoElecG {
            norb_i: 4,
            norb_j: 1,
            w: wxh.to_vec(),
            w_cols: 1,
            e1b,
            e2a,
        };
    }

    let ri = local_xx_g(ei, ej, r);
    let w100 = rotate_xx_g(&ri, &axis, &proj);
    for a in 0..4 {
        for b in 0..4 {
            e1b[a][b] = w100[pack(a, b) * 10] * (-ej.core_charge);
            e2a[a][b] = w100[pack(a, b)] * (-ei.core_charge);
        }
    }
    PairTwoElecG {
        norb_i: 4,
        norb_j: 4,
        w: w100.to_vec(),
        w_cols: 10,
        e1b,
        e2a,
    }
}

/// `1/√A − 1/√B` without the cancellation, given the difference `B − A` computed exactly.
///
/// # Why this exists
///
/// The Dewar–Sabelli–Klopman multipole integrals are built as differences of Klopman–Ohno
/// kernels evaluated at displaced points, e.g. the charge–dipole term
///
/// ```text
/// ri[1] ∝ 1/√((r+d)² + ρ²) − 1/√((r−d)² + ρ²)
/// ```
///
/// Each term is `O(1/r)` while the difference is `O(d/r²)`, so writing it literally throws away
/// `log₁₀(r/d)` digits — around two at the 40 Bohr end of a periodic image list. The identity
///
/// ```text
/// 1/√A − 1/√B = (B − A) / (√A √B (√A + √B))
/// ```
///
/// moves the subtraction into `B − A`, which for these arguments is a **polynomial** in `r` and
/// the charge separation — `(r−d)² − (r+d)² = −4rd` exactly — so it is computed with no
/// cancellation at all. The caller passes that closed form in rather than letting this function
/// subtract `A` and `B`, which would reintroduce exactly the loss being removed.
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
fn inv_sqrt_difference<S: Scalar>(a: S, b: S, b_minus_a: S) -> S {
    let sa = a.sqrt();
    let sb = b.sqrt();
    b_minus_a / (sa * sb * (sa + sb))
}

fn local_xh_g<S: Scalar>(ei: &Am1Element, ej: &Am1Element, r: S) -> [S; 4] {
    let ev1 = AM1_EV / 2.0;
    let ev2 = AM1_EV / 4.0;
    let da = S::cst(ei.dd);
    let qa = S::cst(ei.qq * 2.0);
    let aee = (ei.rho0 + ej.rho0).powi(2);
    let ade = (ei.rho1 + ej.rho0).powi(2);
    let aqe = (ei.rho2 + ej.rho0).powi(2);
    let ee = (r * r + aee).sqrt().recip() * AM1_EV;
    let ev1dsqr6 = (r * r + aqe).sqrt().recip() * ev1;
    let mut ri = [S::cst(0.0); 4];
    ri[0] = ee;
    ri[1] = ((r + da) * (r + da) + ade).sqrt().recip() * ev1
        - ((r - da) * (r - da) + ade).sqrt().recip() * ev1;
    ri[2] = ee
        + ((r + qa) * (r + qa) + aqe).sqrt().recip() * ev2
        + ((r - qa) * (r - qa) + aqe).sqrt().recip() * ev2
        - ev1dsqr6;
    ri[3] = ee + (r * r + qa * qa + aqe).sqrt().recip() * ev1 - ev1dsqr6;
    ri
}

fn local_xx_g<S: Scalar>(ei: &Am1Element, ej: &Am1Element, r: S) -> [S; 22] {
    let ev1 = AM1_EV / 2.0;
    let ev2 = AM1_EV / 4.0;
    let ev3 = AM1_EV / 8.0;
    let ev4 = AM1_EV / 16.0;
    let da = S::cst(ei.dd);
    let db = S::cst(ej.dd);
    let qa = S::cst(ei.qq * 2.0);
    let qb = S::cst(ej.qq * 2.0);
    let qa1 = S::cst(ei.qq);
    let qb1 = S::cst(ej.qq);
    let aee = (ei.rho0 + ej.rho0).powi(2);
    let ade = (ei.rho1 + ej.rho0).powi(2);
    let aqe = (ei.rho2 + ej.rho0).powi(2);
    let aed = (ei.rho0 + ej.rho1).powi(2);
    let aeq = (ei.rho0 + ej.rho2).powi(2);
    let axx = (ei.rho1 + ej.rho1).powi(2);
    let adq = (ei.rho1 + ej.rho2).powi(2);
    let aqd = (ei.rho2 + ej.rho1).powi(2);
    let aqq = (ei.rho2 + ej.rho2).powi(2);

    // 1/sqrt(x + c) helper.
    let g = |x: S, c: f64| (x + c).sqrt().recip();
    let sq2 = |a: S, c: f64| (a * a + c).sqrt().recip(); // 1/sqrt(a^2 + c)

    let ee = (r * r + aee).sqrt().recip() * AM1_EV;
    let dze = sq2(r - da, ade) * ev1 - sq2(r + da, ade) * ev1;
    let ev1dsqr6 = (r * r + aqe).sqrt().recip() * ev1;
    let qzze = sq2(r - qa, aqe) * ev2 + sq2(r + qa, aqe) * ev2 - ev1dsqr6;
    let qxxe = g(r * r + qa * qa, aqe) * ev1 - ev1dsqr6;
    let edz = sq2(r + db, aed) * ev1 - sq2(r - db, aed) * ev1;
    let ev1dsqr12 = (r * r + aeq).sqrt().recip() * ev1;
    let eqzz = sq2(r - qb, aeq) * ev2 + sq2(r + qb, aeq) * ev2 - ev1dsqr12;
    let eqxx = g(r * r + qb * qb, aeq) * ev1 - ev1dsqr12;
    let ev2dsqr20 = sq2(r + da, adq) * ev2;
    let ev2dsqr22 = sq2(r - da, adq) * ev2;
    let ev2dsqr24 = sq2(r - db, aqd) * ev2;
    let ev2dsqr26 = sq2(r + db, aqd) * ev2;
    let ev2dsqr36 = (r * r + aqq).sqrt().recip() * ev2;
    let ev2dsqr39 = g(r * r + qa * qa, aqq) * ev2;
    let ev2dsqr40 = g(r * r + qb * qb, aqq) * ev2;
    let ev3dsqr42 = sq2(r - qb, aqq) * ev3;
    let ev3dsqr44 = sq2(r + qb, aqq) * ev3;
    let ev3dsqr46 = sq2(r + qa, aqq) * ev3;
    let ev3dsqr48 = sq2(r - qa, aqq) * ev3;

    let mut ri = [S::cst(0.0); 22];
    ri[0] = ee;
    ri[1] = -dze;
    ri[2] = ee + qzze;
    ri[3] = ee + qxxe;
    ri[4] = -edz;
    ri[5] = sq2(r + da - db, axx) * ev2 + sq2(r - da + db, axx) * ev2
        - sq2(r - da - db, axx) * ev2
        - sq2(r + da + db, axx) * ev2;
    ri[6] =
        g(r * r + (da - db) * (da - db), axx) * ev1 - g(r * r + (da + db) * (da + db), axx) * ev1;
    ri[7] = -edz + sq2(r + qa - db, aqd) * ev3 - sq2(r + qa + db, aqd) * ev3
        + sq2(r - qa - db, aqd) * ev3
        - sq2(r - qa + db, aqd) * ev3
        - ev2dsqr24
        + ev2dsqr26;
    ri[8] = -edz - ev2dsqr24 + g((r - db) * (r - db) + qa * qa, aqd) * ev2 + ev2dsqr26
        - g((r + db) * (r + db) + qa * qa, aqd) * ev2;
    ri[9] = g((qa1 - db) * (qa1 - db) + (r + qa1) * (r + qa1), aqd) * ev2
        - g((qa1 - db) * (qa1 - db) + (r - qa1) * (r - qa1), aqd) * ev2
        - g((qa1 + db) * (qa1 + db) + (r + qa1) * (r + qa1), aqd) * ev2
        + g((qa1 + db) * (qa1 + db) + (r - qa1) * (r - qa1), aqd) * ev2;
    ri[10] = ee + eqzz;
    ri[11] = ee + eqxx;
    ri[12] = -dze + sq2(r + da - qb, adq) * ev3 - sq2(r - da - qb, adq) * ev3
        + sq2(r + da + qb, adq) * ev3
        - sq2(r - da + qb, adq) * ev3
        + ev2dsqr22
        - ev2dsqr20;
    ri[13] = -dze - ev2dsqr20 + g((r + da) * (r + da) + qb * qb, adq) * ev2 + ev2dsqr22
        - g((r - da) * (r - da) + qb * qb, adq) * ev2;
    ri[14] = g((da - qb1) * (da - qb1) + (r - qb1) * (r - qb1), adq) * ev2
        - g((da - qb1) * (da - qb1) + (r + qb1) * (r + qb1), adq) * ev2
        - g((da + qb1) * (da + qb1) + (r - qb1) * (r - qb1), adq) * ev2
        + g((da + qb1) * (da + qb1) + (r + qb1) * (r + qb1), adq) * ev2;
    ri[15] = ee
        + eqzz
        + qzze
        + sq2(r + qa - qb, aqq) * ev4
        + sq2(r + qa + qb, aqq) * ev4
        + sq2(r - qa - qb, aqq) * ev4
        + sq2(r - qa + qb, aqq) * ev4
        - ev3dsqr48
        - ev3dsqr46
        - ev3dsqr42
        - ev3dsqr44
        + ev2dsqr36;
    ri[16] = ee
        + eqzz
        + qxxe
        + g((r - qb) * (r - qb) + qa * qa, aqq) * ev3
        + g((r + qb) * (r + qb) + qa * qa, aqq) * ev3
        - ev3dsqr42
        - ev3dsqr44
        - ev2dsqr39
        + ev2dsqr36;
    ri[17] = ee
        + eqxx
        + qzze
        + g((r + qa) * (r + qa) + qb * qb, aqq) * ev3
        + g((r - qa) * (r - qa) + qb * qb, aqq) * ev3
        - ev3dsqr46
        - ev3dsqr48
        - ev2dsqr40
        + ev2dsqr36;
    let qxxqxx = g(r * r + (qa - qb) * (qa - qb), aqq) * ev3
        + g(r * r + (qa + qb) * (qa + qb), aqq) * ev3
        - ev2dsqr39
        - ev2dsqr40
        + ev2dsqr36;
    ri[18] = ee + eqxx + qxxe + qxxqxx;
    ri[19] = g(
        (r + qa1 - qb1) * (r + qa1 - qb1) + (qa1 - qb1) * (qa1 - qb1),
        aqq,
    ) * ev3
        - g(
            (r + qa1 + qb1) * (r + qa1 + qb1) + (qa1 - qb1) * (qa1 - qb1),
            aqq,
        ) * ev3
        - g(
            (r - qa1 - qb1) * (r - qa1 - qb1) + (qa1 - qb1) * (qa1 - qb1),
            aqq,
        ) * ev3
        + g(
            (r - qa1 + qb1) * (r - qa1 + qb1) + (qa1 - qb1) * (qa1 - qb1),
            aqq,
        ) * ev3
        - g(
            (r + qa1 - qb1) * (r + qa1 - qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3
        + g(
            (r + qa1 + qb1) * (r + qa1 + qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3
        + g(
            (r - qa1 - qb1) * (r - qa1 - qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3
        - g(
            (r - qa1 + qb1) * (r - qa1 + qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3;
    let qxxqyy = g(r * r + qa * qa + qb * qb, aqq) * ev2 - ev2dsqr39 - ev2dsqr40 + ev2dsqr36;
    ri[20] = ee + eqxx + qxxe + qxxqyy;
    ri[21] = (qxxqxx - qxxqyy) * 0.5;
    ri
}

/// Heavy–hydrogen block, in terms of the diatomic axis `n` and transverse projector `p`.
fn build_wxh_g<S: Scalar>(ri: &[S; 4], n: &[S; 3], p: &[[S; 3]; 3], wxh: &mut [S; 10]) {
    wxh[pack(0, 0)] = ri[0];
    for k in 0..3 {
        wxh[pack(k + 1, 0)] = ri[1] * n[k];
    }
    for k in 0..3 {
        for l in 0..=k {
            wxh[pack(k + 1, l + 1)] = ri[2] * (n[k] * n[l]) + ri[3] * p[k][l];
        }
    }
}

/// Heavy–heavy block, in terms of the diatomic axis `n` and transverse projector `p`.
///
/// Every transverse dependence in the MNDO rotation reduces to `p`. The only place that is
/// not immediate is the rank-4 term, where the three quartic transverse tensors
///
/// ```text
/// T1 = r1r1r1r1 + r2r2r2r2
/// T2 = r1r1r2r2 + r2r2r1r1
/// T3 = (r1r2 + r2r1) ⊗ (r1r2 + r2r1)
/// ```
///
/// satisfy `T1 + T2 = p_kl p_mq` and `2 T1 + T3 = p_km p_lq + p_kq p_lm`. Substituting leaves
/// `T1` with the coefficient `ri[18] − ri[20] − 2 ri[21]`, and MNDO's in-plane isotropy makes
/// that identically zero: from the definitions of `ri[18]`, `ri[20]` and `ri[21]` in
/// [`local_xx_g`], `ri[18] = ri[20] + 2 ri[21]`. So the frame cancels exactly — not to
/// leading order, exactly — and only products of projectors remain.
fn rotate_xx_g<S: Scalar>(ri: &[S; 22], n: &[S; 3], p: &[[S; 3]; 3]) -> [S; 100] {
    let mut w = [S::cst(0.0); 100];
    let mut idx = 0usize;
    for kk in 0..4usize {
        for ll in 0..=kk {
            for mm in 0..4usize {
                for nn in 0..=mm {
                    let (k, l, m, q) = (
                        kk.wrapping_sub(1),
                        ll.wrapping_sub(1),
                        mm.wrapping_sub(1),
                        nn.wrapping_sub(1),
                    );
                    let val = if kk == 0 {
                        if mm == 0 {
                            ri[0]
                        } else if nn == 0 {
                            ri[4] * n[m]
                        } else {
                            ri[10] * (n[m] * n[q]) + ri[11] * p[m][q]
                        }
                    } else if ll == 0 {
                        if mm == 0 {
                            ri[1] * n[k]
                        } else if nn == 0 {
                            ri[5] * (n[k] * n[m]) + ri[6] * p[k][m]
                        } else {
                            ri[12] * (n[k] * n[m] * n[q])
                                + ri[13] * (p[m][q] * n[k])
                                + ri[14] * (p[k][q] * n[m] + p[k][m] * n[q])
                        }
                    } else if mm == 0 {
                        ri[2] * (n[k] * n[l]) + ri[3] * p[k][l]
                    } else if nn == 0 {
                        ri[7] * (n[k] * n[l] * n[m])
                            + ri[8] * (p[k][l] * n[m])
                            + ri[9] * (p[l][m] * n[k] + p[k][m] * n[l])
                    } else {
                        ri[15] * (n[k] * n[l] * n[m] * n[q])
                            + ri[16] * (p[k][l] * n[m] * n[q])
                            + ri[17] * (p[m][q] * n[k] * n[l])
                            + ri[19]
                                * (n[k] * (p[l][q] * n[m] + p[l][m] * n[q])
                                    + n[l] * (p[k][q] * n[m] + p[k][m] * n[q]))
                            + ri[20] * (p[k][l] * p[m][q])
                            + ri[21] * (p[k][m] * p[l][q] + p[k][q] * p[l][m])
                    };
                    w[idx] = val;
                    idx += 1;
                }
            }
        }
    }
    w
}

/// Dual-valued two-electron integrals for a pair, seeded on the displacement `R_j − R_i`.
pub fn pair_two_electron_dual(ei: &Am1Element, ej: &Am1Element, dvec: Vec3) -> PairTwoElecG<Dual> {
    pair_two_electron_g(
        ei,
        ej,
        [
            Dual::var(dvec.x, 0),
            Dual::var(dvec.y, 1),
            Dual::var(dvec.z, 2),
        ],
    )
}

#[cfg(test)]
mod legacy_frame {
    //! The pre-0.2.0 quaternion-frame assembly, kept only so the frame-free rewrite can be
    //! proved value-preserving against it. It is *not* correct for derivatives — the branch
    //! in [`rotation_to_x_g`] discards them near `n = -x̂` — but its values are the reference
    //! the rewrite must reproduce exactly.
    use super::{pack, rotation_to_x_g};
    use crate::dual::Scalar;

    pub fn build_wxh<S: Scalar>(ri: &[S; 4], n: &[S; 3], wxh: &mut [S; 10]) {
        let rot = rotation_to_x_g(n[0], n[1], n[2]);
        let (r0, r1, r2) = (rot[0], rot[1], rot[2]);
        wxh[pack(0, 0)] = ri[0];
        for k in 0..3 {
            wxh[pack(k + 1, 0)] = ri[1] * r0[k];
        }
        for k in 0..3 {
            for l in 0..=k {
                let t0 = r0[k] * r0[l];
                let t1 = r1[k] * r1[l] + r2[k] * r2[l];
                wxh[pack(k + 1, l + 1)] = ri[2] * t0 + ri[3] * t1;
            }
        }
    }

    pub fn rotate_xx<S: Scalar>(ri: &[S; 22], axis: &[S; 3]) -> [S; 100] {
        let rot = rotation_to_x_g(axis[0], axis[1], axis[2]);
        let (r0, r1, r2) = (rot[0], rot[1], rot[2]);
        let mut w = [S::cst(0.0); 100];
        let mut idx = 0usize;
        for kk in 0..4usize {
            for ll in 0..=kk {
                for mm in 0..4usize {
                    for nn in 0..=mm {
                        let (k, l, m, n) = (
                            kk.wrapping_sub(1),
                            ll.wrapping_sub(1),
                            mm.wrapping_sub(1),
                            nn.wrapping_sub(1),
                        );
                        let val = if kk == 0 {
                            if mm == 0 {
                                ri[0]
                            } else if nn == 0 {
                                ri[4] * r0[m]
                            } else {
                                ri[10] * (r0[m] * r0[n]) + ri[11] * (r1[m] * r1[n] + r2[m] * r2[n])
                            }
                        } else if ll == 0 {
                            if mm == 0 {
                                ri[1] * r0[k]
                            } else if nn == 0 {
                                ri[5] * (r0[k] * r0[m]) + ri[6] * (r1[k] * r1[m] + r2[k] * r2[m])
                            } else {
                                let t0 = r0[k] * r0[m] * r0[n];
                                let t1 = (r1[m] * r1[n] + r2[m] * r2[n]) * r0[k];
                                let mix = r1[k] * (r1[n] * r0[m] + r1[m] * r0[n])
                                    + r2[k] * (r2[m] * r0[n] + r2[n] * r0[m]);
                                ri[12] * t0 + ri[13] * t1 + ri[14] * mix
                            }
                        } else if mm == 0 {
                            let t0 = r0[k] * r0[l];
                            let t1 = r1[k] * r1[l] + r2[k] * r2[l];
                            ri[2] * t0 + ri[3] * t1
                        } else if nn == 0 {
                            let t0 = r0[k] * r0[l] * r0[m];
                            let t1 = (r1[k] * r1[l] + r2[k] * r2[l]) * r0[m];
                            let t2 = r1[l] * r1[m] + r2[l] * r2[m];
                            ri[7] * t0
                                + ri[8] * t1
                                + ri[9] * (r0[k] * t2 + r0[l] * (r1[k] * r1[m] + r2[k] * r2[m]))
                        } else {
                            let t0 = r0[k] * r0[l] * r0[m] * r0[n];
                            let mut v = ri[15] * t0;
                            v = v + ri[16] * ((r1[k] * r1[l] + r2[k] * r2[l]) * r0[m] * r0[n]);
                            v = v + ri[17] * ((r1[m] * r1[n] + r2[m] * r2[n]) * (r0[k] * r0[l]));
                            v = v + ri[18]
                                * (r1[k] * r1[l] * r1[m] * r1[n] + r2[k] * r2[l] * r2[m] * r2[n]);
                            let mix1 = r0[m] * (r1[l] * r1[n] + r2[l] * r2[n]);
                            let mix2 = r0[n] * (r1[l] * r1[m] + r2[l] * r2[m]);
                            let val5 = r0[k] * (mix1 + mix2)
                                + r0[l]
                                    * (r0[m] * (r1[k] * r1[n] + r2[k] * r2[n])
                                        + r0[n] * (r1[k] * r1[m] + r2[k] * r2[m]));
                            v = v + ri[19] * val5;
                            v = v + ri[20]
                                * (r1[k] * r1[l] * r2[m] * r2[n] + r2[k] * r2[l] * r1[m] * r1[n]);
                            let cross =
                                (r1[k] * r2[l] + r2[k] * r1[l]) * (r1[m] * r2[n] + r2[m] * r1[n]);
                            v = v + ri[21] * cross;
                            v
                        };
                        w[idx] = val;
                        idx += 1;
                    }
                }
            }
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Am1Parameters;

    #[test]
    fn the_multipole_differences_keep_their_digits_at_long_range() {
        // The DSK multipole integrals are differences of Klopman–Ohno kernels at displaced
        // points, and the difference is smaller than the terms by a factor `(d/r)ⁿ`. This
        // measures how many digits that costs across the range a periodic image list actually
        // uses — the real-space cutoff runs to 40 Bohr and beyond.
        //
        // `inv_sqrt_difference` is the reference because it provably cannot cancel: the
        // subtraction happens in `B − A`, which is a polynomial evaluated in closed form.
        let params = Am1Parameters::standard().unwrap();
        let o = params.element(8).unwrap();
        let h = params.element(1).unwrap();
        let ev1 = AM1_EV / 2.0;
        let da = o.dd;
        let ade = (o.rho1 + h.rho0).powi(2);

        eprintln!("      r (Bohr)     naive              stable             rel. difference");
        let mut worst = 0.0_f64;
        for &r in &[2.0_f64, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0] {
            let naive = ((r + da) * (r + da) + ade).sqrt().recip() * ev1
                - ((r - da) * (r - da) + ade).sqrt().recip() * ev1;
            let a = (r + da) * (r + da) + ade;
            let b = (r - da) * (r - da) + ade;
            // (r−d)² − (r+d)² = −4rd, exactly.
            let stable = inv_sqrt_difference::<f64>(a, b, -4.0 * r * da) * ev1;
            let rel = (naive - stable).abs() / stable.abs();
            eprintln!("      {r:8.1}     {naive:.12e}   {stable:.12e}   {rel:.2e}");
            worst = worst.max(rel);
        }
        eprintln!("      worst relative disagreement: {worst:.3e}");

        // This is a *measurement*, and the bound is what the measurement supports. If the naive
        // form were losing enough digits to matter it would show here as a disagreement far
        // above the rounding floor.
        assert!(
            worst < 1.0e-12,
            "the charge–dipole difference has lost {worst:.3e} of its precision at long range, \
             which is enough to warrant rewriting the multipole kernels in the stable form"
        );
    }

    /// Deterministic low-discrepancy directions on the sphere, so the A/B comparison below
    /// covers the whole sphere reproducibly rather than wherever an RNG happened to land.
    fn sphere_directions(count: usize) -> Vec<[f64; 3]> {
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..count)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
                let rho = (1.0 - z * z).max(0.0).sqrt();
                let phi = golden * i as f64;
                [rho * phi.cos(), rho * phi.sin(), z]
            })
            .collect()
    }

    #[test]
    fn frame_free_assembly_reproduces_the_quaternion_values() {
        // The rewrite removed the diatomic frame from the two-centre kernels. It must be a
        // pure refactor of the *values*: only the derivatives may differ (the quaternion
        // branch zeroes them near n = -x). Compare over the whole sphere, skipping a small
        // cone around the singular direction where the old code is simply wrong.
        let p = Am1Parameters::standard().unwrap();
        let pairs = [(6u8, 8u8), (6, 6), (7, 8), (16, 17), (6, 1), (8, 1)];
        let mut worst = 0.0_f64;
        for (za, zb) in pairs {
            let (ea, eb) = (p.element(za).unwrap(), p.element(zb).unwrap());
            for r in [1.5_f64, 2.6, 4.0, 8.0] {
                for n in sphere_directions(997) {
                    if 1.0 + n[0] < 1.0e-6 {
                        continue; // the quaternion pole: old values are fine, but skip anyway
                    }
                    let proj = transverse_projector(n);
                    if eb.has_p() && ea.has_p() {
                        let ri = local_xx_g::<f64>(ea, eb, r);
                        let new = rotate_xx_g(&ri, &n, &proj);
                        let old = legacy_frame::rotate_xx(&ri, &n);
                        for i in 0..100 {
                            worst = worst.max((new[i] - old[i]).abs());
                        }
                    } else if ea.has_p() {
                        let ri = local_xh_g::<f64>(ea, eb, r);
                        let (mut new, mut old) = ([0.0; 10], [0.0; 10]);
                        build_wxh_g(&ri, &n, &proj, &mut new);
                        legacy_frame::build_wxh(&ri, &n, &mut old);
                        for i in 0..10 {
                            worst = worst.max((new[i] - old[i]).abs());
                        }
                    }
                }
            }
        }
        assert!(
            worst < 1.0e-12,
            "frame-free assembly changed a value by {worst:.3e} eV; it must only change derivatives"
        );
    }

    #[test]
    fn two_electron_derivatives_survive_axis_alignment() {
        // The quaternion frame was singular for pairs on the x axis, silently returning a
        // zero derivative there. Check every Cartesian alignment against a finite difference.
        let p = Am1Parameters::standard().unwrap();
        let (c, o) = (p.element(6).unwrap(), p.element(8).unwrap());
        let h = 1.0e-6;
        for d in [
            Vec3::new(2.4, 0.0, 0.0),
            Vec3::new(-2.4, 0.0, 0.0),
            Vec3::new(0.0, 2.4, 0.0),
            Vec3::new(0.0, 0.0, -2.4),
            Vec3::new(2.4, 1.0e-9, 0.0),
        ] {
            let dual = pair_two_electron_dual(c, o, d);
            let mut worst = 0.0_f64;
            for axis in 0..3 {
                let (mut dp, mut dm) = (d, d);
                match axis {
                    0 => {
                        dp.x += h;
                        dm.x -= h;
                    }
                    1 => {
                        dp.y += h;
                        dm.y -= h;
                    }
                    _ => {
                        dp.z += h;
                        dm.z -= h;
                    }
                }
                let wp = pair_two_electron_g::<f64>(c, o, [dp.x, dp.y, dp.z]);
                let wm = pair_two_electron_g::<f64>(c, o, [dm.x, dm.y, dm.z]);
                for a in 0..10 {
                    for b in 0..10 {
                        let fd = (wp.w_at(a, b) - wm.w_at(a, b)) / (2.0 * h);
                        worst = worst.max((dual.w_at(a, b).d[axis] - fd).abs());
                    }
                }
            }
            assert!(
                worst < 1.0e-6,
                "two-electron derivative off by {worst:.3e} for displacement {d:?}"
            );
        }
    }

    #[test]
    fn rotation_is_orthonormal() {
        let v = Vec3::new(0.3, -0.5, 0.8).normalized();
        let r = rotation_to_x(v);
        for i in 0..3 {
            let ni: f64 = (0..3).map(|k| r[i][k] * r[i][k]).sum();
            assert!((ni - 1.0).abs() < 1e-10);
        }
        let rv0: f64 = (0..3).map(|k| r[0][k] * v.get(k)).sum();
        assert!((rv0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ssss_is_rotation_invariant() {
        let p = Am1Parameters::standard().unwrap();
        let c = p.element(6).unwrap();
        let r = 2.6;
        let a = pair_two_electron(c, c, Vec3::new(1.0, 0.0, 0.0), r);
        let b = pair_two_electron(c, c, Vec3::new(0.3, -0.5, 0.8).normalized(), r);
        assert!((a.w_at(0, 0) - b.w_at(0, 0)).abs() < 1e-9);
        let expect = AM1_EV / (r * r + (2.0 * c.rho0).powi(2)).sqrt();
        assert!((a.w_at(0, 0) - expect).abs() < 1e-9);
    }

    #[test]
    fn dual_two_electron_matches_fd() {
        // The dual derivative of a two-electron integral must match a finite difference.
        let p = Am1Parameters::standard().unwrap();
        let (c, o) = (p.element(6).unwrap(), p.element(8).unwrap());
        let d = Vec3::new(1.4, -0.9, 0.7);
        let dual = pair_two_electron_dual(c, o, d);
        let h = 1e-6;
        let mut max_delta = 0.0_f64;
        for axis in 0..3 {
            let mut dp = d;
            let mut dm = d;
            match axis {
                0 => {
                    dp.x += h;
                    dm.x -= h;
                }
                1 => {
                    dp.y += h;
                    dm.y -= h;
                }
                _ => {
                    dp.z += h;
                    dm.z -= h;
                }
            }
            let wp = pair_two_electron_g::<f64>(c, o, [dp.x, dp.y, dp.z]);
            let wm = pair_two_electron_g::<f64>(c, o, [dm.x, dm.y, dm.z]);
            for a in 0..10 {
                for b in 0..10 {
                    let fd = (wp.w_at(a, b) - wm.w_at(a, b)) / (2.0 * h);
                    max_delta = max_delta.max((dual.w_at(a, b).d[axis] - fd).abs());
                }
            }
        }
        eprintln!("dual 2e derivative max delta = {max_delta:.2e}");
        assert!(
            max_delta < 1e-6,
            "dual 2e derivative mismatch {max_delta:.3e}"
        );
    }
}
