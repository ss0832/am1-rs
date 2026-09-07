// SPDX-License-Identifier: GPL-3.0-or-later

//! Kerker preconditioning for the periodic SCF's charge-transfer modes.
//!
//! # The problem it solves
//!
//! Pulay mixing extrapolates from a history of residuals, which works when the map from input to
//! output density is a contraction in every direction. In a metal, a small-gap system or a long
//! cell it is not: a small change in the potential at one end of the cell moves charge to the
//! other end, the response comes back overshot, and the iteration oscillates with growing
//! amplitude. This is **charge sloshing**, and it is a property of the *long-wavelength* part of
//! the density alone — the short-wavelength part is well conditioned and must not be touched.
//!
//! # The plane-wave form, and why it cannot be used directly
//!
//! Plane-wave codes damp the long-wavelength modes with the Kerker factor
//!
//! ```text
//! A(q) = q² / (q² + q₀²)
//! ```
//!
//! which is `≈ 1` for large `q` and `→ 0` as `q → 0`. There is no plane-wave grid here: the
//! density is a set of real-space blocks over an atom-centred basis, and `q` is not one of its
//! indices. Transforming the residual over the lattice-translation index instead does not work
//! either — the k-mesh has far fewer points than the translation list, so the round trip is a
//! projection that would throw most of the residual away.
//!
//! # The form that *is* available
//!
//! Rewrite the factor against the Coulomb kernel `γ(q) = 4π/q²`:
//!
//! ```text
//! A(q) = q²/(q² + q₀²) = 1 / (1 + q₀²/q²) = 1 / (1 + κ γ(q)),   κ = q₀²/4π
//! ```
//!
//! In that form nothing is specific to plane waves: it is `(1 + κγ)⁻¹` with `γ` *the Coulomb
//! interaction between charge fluctuations*, and this crate has that matrix explicitly. NDDO's
//! monopole kernel `γ_ab` — the two-centre `(s_a s_a | s_b s_b)` integrals summed over the
//! translations the pair list carries, plus the Ewald remainder and the on-site `G_ss` — is the
//! atom-resolved Coulomb interaction, in eV per electron². Long-wavelength charge transfer is
//! exactly the direction in which `γ` has its large eigenvalues, so `(1 + κγ)⁻¹` damps it and
//! leaves the short-wavelength directions, where `γ` is small, essentially alone. In the
//! plane-wave limit the two expressions are the same function.
//!
//! # What it is applied to
//!
//! Only the **net charge per atom**. The residual `P_out − P_in` also carries intra-atomic
//! rehybridization and off-diagonal bond-order changes; those are short-ranged, are not what
//! sloshes, and damping them would slow every system down to fix a few. So the preconditioner
//! folds the residual to `Δq_a`, damps that, and puts the difference back on the atom's diagonal
//! spread evenly over its orbitals — leaving the *shape* of the on-atom residual untouched and
//! rescaling only the net transfer.

use crate::basis::Basis;
use crate::error::Result;
use crate::hamiltonian::PairIntegral;
use crate::linalg::Matrix;
use crate::params::Am1Parameters;
use crate::system::Molecule;

/// `(1 + κγ)⁻¹`, precomputed for a fixed geometry and `κ`.
#[derive(Clone, Debug)]
pub(crate) struct Kerker {
    /// `nat × nat`, dimensionless.
    damping: Matrix,
}

impl Kerker {
    /// Build the preconditioner from the monopole Coulomb kernel.
    ///
    /// `pairs` supplies the explicitly summed two-centre part and `delta` — the Ewald remainder
    /// from [`crate::pbc::ewald::LongRangeMonopole`] — the rest of the lattice sum. Passing both
    /// is what makes `γ` the *whole* Coulomb interaction rather than the truncated one: a
    /// preconditioner built on the truncated kernel would under-damp exactly the long-range modes
    /// it exists to damp.
    pub(crate) fn new(
        molecule: &Molecule,
        params: &Am1Parameters,
        pairs: &[PairIntegral],
        delta: Option<&Matrix>,
        kappa: f64,
    ) -> Result<Self> {
        let nat = molecule.atoms.len();
        let mut gamma = Matrix::zeros(nat, nat);
        // On-site: the one-centre Coulomb repulsion of the s shell.
        for (a, atom) in molecule.atoms.iter().enumerate() {
            gamma[(a, a)] += params.element(atom.z)?.g_ss;
        }
        // Two-centre, over every translation the pair list carries. `(s s | s s)` is the monopole
        // channel of the pair block: index `(0, 0, 0, 0)` of the rotated integrals.
        for pair in pairs {
            let g = pair.te.two_e(0, 0, 0, 0);
            gamma[(pair.a, pair.b)] += g;
            if pair.a != pair.b || !pair.offset.is_origin() {
                gamma[(pair.b, pair.a)] += g;
            }
        }
        if let Some(d) = delta {
            for a in 0..nat {
                for b in 0..nat {
                    gamma[(a, b)] += d[(a, b)];
                }
            }
        }
        // `(I + κγ)⁻¹`, by solving against the identity. `γ` is symmetric positive semidefinite
        // (it is a Coulomb kernel) and `κ > 0`, so `I + κγ` is positive definite and the solve
        // cannot be singular.
        let mut m = Matrix::zeros(nat, nat);
        for a in 0..nat {
            for b in 0..nat {
                m[(a, b)] = kappa * gamma[(a, b)];
            }
            m[(a, a)] += 1.0;
        }
        Ok(Self {
            damping: invert_spd(&m)?,
        })
    }

    /// Damp the charge-transfer content of a residual, in place.
    ///
    /// `residual` is `P_out − P_in` for one spin channel, flattened the way the SCF's mixer holds
    /// it: `blocks[t]` contiguous, `nao × nao` row-major. Only the origin block's diagonal is
    /// touched, because that is where the atomic charge lives.
    pub(crate) fn apply(&self, origin_block: &mut Matrix, basis: &Basis) {
        let nat = self.damping.rows;
        let mut dq = vec![0.0; nat];
        for (a, q) in dq.iter_mut().enumerate() {
            let off = basis.atom_offset[a];
            for k in 0..basis.atom_norb[a] {
                *q += origin_block[(off + k, off + k)];
            }
        }
        let mut damped = vec![0.0; nat];
        for (a, out) in damped.iter_mut().enumerate() {
            for (b, q) in dq.iter().enumerate() {
                *out += self.damping[(a, b)] * q;
            }
        }
        for a in 0..nat {
            let n = basis.atom_norb[a];
            if n == 0 {
                continue;
            }
            // Spread the correction evenly rather than scaling by `damped/dq`: the ratio is
            // unbounded as `dq → 0`, and near convergence `dq` goes to zero on every atom.
            let shift = (damped[a] - dq[a]) / n as f64;
            let off = basis.atom_offset[a];
            for k in 0..n {
                origin_block[(off + k, off + k)] += shift;
            }
        }
    }
}

/// Inverse of a symmetric positive-definite matrix, through its eigendecomposition.
///
/// Eigen rather than Cholesky because the crate already has a checked symmetric eigensolver and
/// this is `nat × nat` — tens, not thousands — so the constant does not matter and the
/// decomposition also reports a non-positive eigenvalue as what it is.
fn invert_spd(m: &Matrix) -> Result<Matrix> {
    let n = m.rows;
    let (values, vectors) = crate::linalg::symmetric_eigen(m)?;
    let mut out = Matrix::zeros(n, n);
    for (k, &lambda) in values.iter().enumerate() {
        // `I + κγ` has every eigenvalue at least 1 in exact arithmetic; a value at or below zero
        // means `γ` was not the kernel it is supposed to be, and inverting it anyway would hand
        // the mixer an amplifier. Dropping the direction leaves it unpreconditioned, which is
        // what the code did before this existed.
        if lambda <= 1.0e-12 {
            continue;
        }
        let inv = 1.0 / lambda;
        for i in 0..n {
            let vi = vectors[(i, k)];
            if vi == 0.0 {
                continue;
            }
            for j in 0..n {
                out[(i, j)] += inv * vi * vectors[(j, k)];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The identity kernel: with `γ = 0` the preconditioner must be the identity, so a residual
    /// passes through untouched. This is the `κ → 0` limit and the guarantee that turning the
    /// feature off costs nothing.
    #[test]
    fn zero_coupling_is_the_identity() {
        let m = Matrix::identity(3);
        let inv = invert_spd(&m).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((inv[(i, j)] - want).abs() < 1.0e-14);
            }
        }
    }

    /// `invert_spd` really inverts: `M M⁻¹ = I` on a matrix with a spread of eigenvalues.
    #[test]
    fn the_inverse_is_an_inverse() {
        let mut m = Matrix::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                m[(i, j)] = if i == j {
                    2.0 + i as f64
                } else {
                    0.3 / (1.0 + (i as f64 - j as f64).abs())
                };
            }
        }
        let inv = invert_spd(&m).unwrap();
        let product = m.matmul(&inv);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[(i, j)] - want).abs() < 1.0e-12,
                    "({i},{j}) = {}",
                    product[(i, j)]
                );
            }
        }
    }
}
