// SPDX-License-Identifier: GPL-3.0-or-later

//! Complex Hermitian matrices and their eigendecomposition, for the k-resolved Fock matrix.
//!
//! NDDO declares the AO basis orthonormal, so `S(k) = I` and the k-point problem is the
//! **standard** Hermitian eigenproblem `H(k) C(k) = C(k) ε(k)` — no overlap metric, no Löwdin
//! orthogonalization, no generalized solve. That is a real simplification over a
//! basis-set method and it is worth stating: the only thing needed here is Hermitian
//! eigenvalues and eigenvectors.
//!
//! Rather than pull in a complex eigensolver, a Hermitian `H = A + iB` (with `A` symmetric and
//! `B` antisymmetric) is solved through the real symmetric embedding
//!
//! ```text
//!         [ A  -B ]
//!   M  =  [ B   A ]        (2n x 2n, symmetric)
//! ```
//!
//! whose spectrum is that of `H` with every eigenvalue doubled: if `H(x + iy) = λ(x + iy)`
//! then `M` has eigenvectors `(x, y)` and `(−y, x)` both at `λ`. So the crate's existing,
//! well-tested real symmetric solver does the work, and the only care needed is picking one
//! member of each degenerate pair back out.

use crate::error::{Am1Error, Result};
use crate::linalg::{symmetric_eigen, Matrix};

/// Dense complex matrix, stored as separate real and imaginary parts.
#[derive(Clone, Debug)]
pub struct CMatrix {
    pub n: usize,
    pub re: Matrix,
    pub im: Matrix,
}

impl CMatrix {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            re: Matrix::zeros(n, n),
            im: Matrix::zeros(n, n),
        }
    }

    #[inline]
    pub fn add(&mut self, i: usize, j: usize, re: f64, im: f64) {
        self.re[(i, j)] += re;
        self.im[(i, j)] += im;
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> (f64, f64) {
        (self.re[(i, j)], self.im[(i, j)])
    }

    /// Force exact Hermiticity, `H = (H + H†)/2`.
    ///
    /// Accumulating a Bloch sum leaves the two triangles agreeing only to rounding, and an
    /// eigensolver fed a matrix that is Hermitian to 1e-16 rather than exactly can return
    /// eigenvalues with a small imaginary drift. Cheap insurance.
    pub fn hermitianize(&mut self) {
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let r = 0.5 * (self.re[(i, j)] + self.re[(j, i)]);
                let m = 0.5 * (self.im[(i, j)] - self.im[(j, i)]);
                self.re[(i, j)] = r;
                self.re[(j, i)] = r;
                self.im[(i, j)] = m;
                self.im[(j, i)] = -m;
            }
            self.im[(i, i)] = 0.0;
        }
    }

    /// Largest deviation from Hermiticity, for diagnostics.
    pub fn hermiticity_error(&self) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..self.n {
            for j in 0..self.n {
                worst = worst.max((self.re[(i, j)] - self.re[(j, i)]).abs());
                worst = worst.max((self.im[(i, j)] + self.im[(j, i)]).abs());
            }
            worst = worst.max(self.im[(i, i)].abs());
        }
        worst
    }

    /// `Re Tr(A† B)`, the real inner product used for energies.
    pub fn real_trace_product(&self, other: &CMatrix) -> f64 {
        let mut acc = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                acc += self.re[(i, j)] * other.re[(i, j)] + self.im[(i, j)] * other.im[(i, j)];
            }
        }
        acc
    }
}

/// Eigenvalues and eigenvectors of a Hermitian matrix.
#[derive(Clone, Debug)]
pub struct CEigen {
    /// Ascending eigenvalues.
    pub values: Vec<f64>,
    /// Eigenvectors as columns: `vectors_re[(mu, i)] + i vectors_im[(mu, i)]`.
    pub vectors_re: Matrix,
    pub vectors_im: Matrix,
}

/// Solve `H c = λ c` for Hermitian `H`.
pub fn hermitian_eigen(h: &CMatrix) -> Result<CEigen> {
    let n = h.n;
    if n == 0 {
        return Ok(CEigen {
            values: Vec::new(),
            vectors_re: Matrix::zeros(0, 0),
            vectors_im: Matrix::zeros(0, 0),
        });
    }

    // Real symmetric embedding: [[A, -B], [B, A]].
    let mut m = Matrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (h.re[(i, j)], h.im[(i, j)]);
            m[(i, j)] = a;
            m[(n + i, n + j)] = a;
            m[(i, n + j)] = -b;
            m[(n + i, j)] = b;
        }
    }

    let (values2, vectors2) = symmetric_eigen(&m)?;

    // Every eigenvalue appears twice: if `(x, y)` is an eigenvector of the embedding then so is
    // `(−y, x)`, which is the *same* complex vector times `i`. Walk the ascending spectrum and
    // keep one vector per pair, rejecting a candidate that the ones already taken span.
    //
    // # Why this is done twice, and why the threshold is not small
    //
    // Through 0.2.2 this projected once and accepted anything whose residual exceeded `1e-8`.
    // Both halves of that are wrong in the same place — a **degenerate** level, which is where
    // the duplicate is genuinely in the taken span:
    //
    // - one pass of classical Gram–Schmidt loses orthogonality by `κ ε`, and here the candidate
    //   *is* the thing being subtracted, so the surviving vector is cancellation noise. Repeating
    //   the projection restores orthogonality to `ε` — the standard "twice is enough" result, and
    //   the only cost is a second pass over an already-short list;
    // - normalising that noise to unit length and then comparing against `1e-8` accepted it as a
    //   physical eigenvector often enough to matter. The right cut is not small: candidates start
    //   as unit vectors, a duplicate's residual is `O(ε)`, and a genuinely new direction inside a
    //   `k`-fold block has residual² of at least `1 − 1/k ≥ 1/2` for some remaining column, since
    //   the `2k` columns share `2(k−1)` real dimensions of complement. `0.1` sits between those
    //   by seven orders of magnitude in one direction and one in the other.
    //
    // What it cost: a two-dimensional lattice of methane — a closed-shell insulator with a
    // 9 eV gap and a threefold degenerate HOMO — could not converge past `dP ≈ 3e-8` however
    // many iterations it was given, because the density rebuilt from these vectors carried that
    // much noise. `3e-8` is `√ε`, which is the signature of exactly this loss.
    const INDEPENDENCE_CUT: f64 = 0.1;
    let mut values = Vec::with_capacity(n);
    let mut vectors_re = Matrix::zeros(n, n);
    let mut vectors_im = Matrix::zeros(n, n);
    let mut taken: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(n);

    let mut col = 0usize;
    while col < 2 * n && taken.len() < n {
        // Candidate column of the embedding: (x, y) means the complex vector x + i y.
        let (mut x, mut y): (Vec<f64>, Vec<f64>) = (
            (0..n).map(|r| vectors2[(r, col)]).collect(),
            (0..n).map(|r| vectors2[(n + r, col)]).collect(),
        );

        // Project out everything already accepted (complex Gram-Schmidt), twice.
        for _pass in 0..2 {
            for (px, py) in &taken {
                // <p|v> = sum conj(p) v
                let mut dr = 0.0;
                let mut di = 0.0;
                for k in 0..n {
                    dr += px[k] * x[k] + py[k] * y[k];
                    di += px[k] * y[k] - py[k] * x[k];
                }
                for k in 0..n {
                    x[k] -= dr * px[k] - di * py[k];
                    y[k] -= dr * py[k] + di * px[k];
                }
            }
        }

        let norm = (0..n)
            .map(|k| x[k] * x[k] + y[k] * y[k])
            .sum::<f64>()
            .sqrt();
        if norm > INDEPENDENCE_CUT {
            let inv = 1.0 / norm;
            for v in x.iter_mut() {
                *v *= inv;
            }
            for v in y.iter_mut() {
                *v *= inv;
            }
            let i = taken.len();
            for k in 0..n {
                vectors_re[(k, i)] = x[k];
                vectors_im[(k, i)] = y[k];
            }
            values.push(values2[col]);
            taken.push((x, y));
        }
        col += 1;
    }

    if taken.len() != n {
        return Err(Am1Error::LinearAlgebra(format!(
            "the real embedding yielded {} independent eigenvectors, expected {n}",
            taken.len()
        )));
    }

    Ok(CEigen {
        values,
        vectors_re,
        vectors_im,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Hermitian matrix from an arbitrary complex one: `H = (M + M†)/2`.
    fn hermitian_from(n: usize, seed: u64) -> CMatrix {
        // A small deterministic generator; no rand dependency and reproducible failures.
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let mut m = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                m.add(i, j, next(), next());
            }
        }
        m.hermitianize();
        m
    }

    /// `‖H c − λ c‖∞` over all eigenpairs.
    fn residual(h: &CMatrix, e: &CEigen) -> f64 {
        let n = h.n;
        let mut worst = 0.0_f64;
        for i in 0..n {
            for r in 0..n {
                let (mut ar, mut ai) = (0.0, 0.0);
                for c in 0..n {
                    let (hr, hi) = h.get(r, c);
                    let (vr, vi) = (e.vectors_re[(c, i)], e.vectors_im[(c, i)]);
                    ar += hr * vr - hi * vi;
                    ai += hr * vi + hi * vr;
                }
                let lam = e.values[i];
                worst = worst.max((ar - lam * e.vectors_re[(r, i)]).abs());
                worst = worst.max((ai - lam * e.vectors_im[(r, i)]).abs());
            }
        }
        worst
    }

    #[test]
    fn eigenpairs_satisfy_the_eigenvalue_equation() {
        for n in [1usize, 2, 3, 6, 12] {
            let h = hermitian_from(n, 12345 + n as u64);
            let e = hermitian_eigen(&h).unwrap();
            assert_eq!(e.values.len(), n);
            let r = residual(&h, &e);
            eprintln!("    n={n:3}  max |Hc - λc| = {r:.3e}");
            assert!(r < 1.0e-10, "n={n}: residual {r:.3e}");
        }
    }

    /// **A degenerate level must give an exact projector, not a `√ε` one.**
    ///
    /// The real embedding doubles every eigenvalue, so a `k`-fold degenerate complex level arrives
    /// as `2k` real columns and the routine has to pick `k` complex vectors out of them. Through
    /// 0.2.2 it projected once and accepted any residual above `1e-8`; for a degenerate level that
    /// is cancellation noise renormalized to unit length, and the occupied projector `P = CC†`
    /// built from it carried about `3e-8` of error.
    ///
    /// The projector is what the SCF actually uses, and unlike the individual vectors it is
    /// unique — any orthonormal basis of the degenerate subspace gives the same `P`. So it is what
    /// is checked, against the one built directly from the exact eigenvectors.
    #[test]
    fn a_degenerate_level_gives_an_exact_projector() {
        // `H = λ Π + μ (I − Π)` for a rank-3 projector `Π` in six dimensions: eigenvalue `λ` is
        // exactly threefold and `μ` exactly threefold, with no arithmetic to blur them.
        let n = 6;
        let (lam, mu) = (-2.5, 4.0);
        // An orthonormal complex basis for the `λ` subspace, built by Gram–Schmidt on a fixed set.
        let raw: [[(f64, f64); 6]; 3] = [
            [
                (1.0, 0.0),
                (0.0, 1.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ],
            [
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, -1.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ],
            [
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ],
        ];
        let basis: Vec<Vec<(f64, f64)>> = raw
            .iter()
            .map(|v| {
                let s = v.iter().map(|(a, b)| a * a + b * b).sum::<f64>().sqrt();
                v.iter().map(|(a, b)| (a / s, b / s)).collect()
            })
            .collect();

        // Π_{rc} = Σ_v v_r conj(v_c)
        let mut pi = CMatrix::zeros(n);
        for v in &basis {
            for r in 0..n {
                for c in 0..n {
                    let (ar, ai) = v[r];
                    let (br, bi) = v[c];
                    pi.add(r, c, ar * br + ai * bi, ai * br - ar * bi);
                }
            }
        }
        let mut h = CMatrix::zeros(n);
        for r in 0..n {
            for c in 0..n {
                let (pr, pinm) = pi.get(r, c);
                let ident = if r == c { 1.0 } else { 0.0 };
                h.add(r, c, lam * pr + mu * (ident - pr), lam * pinm - mu * pinm);
            }
        }

        let e = hermitian_eigen(&h).unwrap();
        assert_eq!(e.values.len(), n);
        for i in 0..3 {
            assert!(
                (e.values[i] - lam).abs() < 1.0e-12,
                "value {i} = {}",
                e.values[i]
            );
        }

        // The projector onto the three lowest states, against `Π` itself.
        let mut worst = 0.0_f64;
        for r in 0..n {
            for c in 0..n {
                let (mut gr, mut gi) = (0.0, 0.0);
                for i in 0..3 {
                    let (ar, ai) = (e.vectors_re[(r, i)], e.vectors_im[(r, i)]);
                    let (br, bi) = (e.vectors_re[(c, i)], e.vectors_im[(c, i)]);
                    gr += ar * br + ai * bi;
                    gi += ai * br - ar * bi;
                }
                let (pr, pim) = pi.get(r, c);
                worst = worst.max((gr - pr).abs()).max((gi - pim).abs());
            }
        }
        eprintln!("    max |CC† - Π| over a threefold level = {worst:.3e}");
        assert!(
            worst < 1.0e-13,
            "the degenerate projector is off by {worst:.3e}; a single Gram-Schmidt pass gives \
             about 3e-8 here, which is what stalled the periodic SCF"
        );

        // And the vectors that build it are orthonormal to the same order.
        for i in 0..n {
            for j in 0..n {
                let (mut dr, mut di) = (0.0, 0.0);
                for k in 0..n {
                    let (xr, xi) = (e.vectors_re[(k, i)], e.vectors_im[(k, i)]);
                    let (yr, yi) = (e.vectors_re[(k, j)], e.vectors_im[(k, j)]);
                    dr += xr * yr + xi * yi;
                    di += xr * yi - xi * yr;
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dr - want).abs() < 1.0e-13 && di.abs() < 1.0e-13,
                    "<{i}|{j}> = {dr} + {di}i"
                );
            }
        }
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let n = 10;
        let h = hermitian_from(n, 999);
        let e = hermitian_eigen(&h).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let (mut dr, mut di) = (0.0, 0.0);
                for k in 0..n {
                    let (xr, xi) = (e.vectors_re[(k, i)], e.vectors_im[(k, i)]);
                    let (yr, yi) = (e.vectors_re[(k, j)], e.vectors_im[(k, j)]);
                    dr += xr * yr + xi * yi;
                    di += xr * yi - xi * yr;
                }
                let want = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((dr - want).abs()).max(di.abs());
            }
        }
        eprintln!("    max |<c_i|c_j> - delta_ij| = {worst:.3e}");
        assert!(worst < 1.0e-10, "orthonormality off by {worst:.3e}");
    }

    #[test]
    fn a_real_matrix_reproduces_the_real_solver() {
        // With no imaginary part the answer must be the ordinary symmetric eigenproblem.
        let n = 8;
        let mut h = hermitian_from(n, 4242);
        for i in 0..n {
            for j in 0..n {
                h.im[(i, j)] = 0.0;
            }
        }
        let complex = hermitian_eigen(&h).unwrap();
        let (real_values, _) = symmetric_eigen(&h.re).unwrap();
        let worst = complex
            .values
            .iter()
            .zip(&real_values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!("    max eigenvalue difference vs the real solver = {worst:.3e}");
        assert!(worst < 1.0e-12);
    }

    #[test]
    fn a_degenerate_spectrum_still_yields_independent_vectors() {
        // Two exactly degenerate levels: the embedding doubles *every* eigenvalue, so a real
        // twofold degeneracy appears four times and the extraction has to keep two
        // independent states rather than the same one twice.
        let n = 4;
        let mut h = CMatrix::zeros(n);
        for i in 0..n {
            h.re[(i, i)] = if i < 2 { -1.0 } else { 3.0 };
        }
        // A little off-diagonal coupling inside each degenerate block, still degenerate
        // overall after the shift below.
        h.add(0, 1, 0.0, 0.5);
        h.add(1, 0, 0.0, -0.5);
        h.hermitianize();

        let e = hermitian_eigen(&h).unwrap();
        assert_eq!(e.values.len(), n);
        assert!(residual(&h, &e) < 1.0e-10);

        // All four vectors must be mutually orthogonal, which fails if a degenerate pair was
        // extracted twice.
        for i in 0..n {
            for j in (i + 1)..n {
                let (mut dr, mut di) = (0.0, 0.0);
                for k in 0..n {
                    let (xr, xi) = (e.vectors_re[(k, i)], e.vectors_im[(k, i)]);
                    let (yr, yi) = (e.vectors_re[(k, j)], e.vectors_im[(k, j)]);
                    dr += xr * yr + xi * yi;
                    di += xr * yi - xi * yr;
                }
                assert!(
                    dr.abs() < 1.0e-9 && di.abs() < 1.0e-9,
                    "vectors {i} and {j} are not orthogonal: ({dr:.3e}, {di:.3e})"
                );
            }
        }
    }

    #[test]
    fn hermitianize_removes_rounding_asymmetry() {
        let mut h = hermitian_from(5, 77);
        h.re[(1, 3)] += 1.0e-14;
        h.im[(2, 4)] += 1.0e-14;
        assert!(h.hermiticity_error() > 0.0);
        h.hermitianize();
        assert!(h.hermiticity_error() < 1.0e-18);
    }
}
