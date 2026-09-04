// SPDX-License-Identifier: GPL-3.0-or-later

//! Dense linear algebra: a row-major `Matrix` wrapper plus the solvers the SCF needs.
//!
//! Parallels `gfn1-rs`'s `linalg.rs`. The heavy O(n³) work — the symmetric eigendecomposition
//! and the LU solve — is delegated to **faer** (pure Rust; no LAPACK/BLAS). Only a tiny
//! pivot-guarded Gaussian elimination for the DIIS coefficient system lives in `scf.rs`,
//! where faer's non-erroring behaviour on the near-singular DIIS matrix is undesirable.

use crate::error::{Am1Error, Result};

/// Hand faer the rayon thread pool, once per process.
///
/// faer is pulled in with `default-features = false`, and in that configuration its global
/// parallelism defaults to `Seq` -- so the eigendecomposition and every matmul routed through
/// it run on one core no matter how many the machine has. The eigensolve is the SCF's only
/// intrinsic parallelism, so leaving this unset costs most of the machine.
pub fn enable_parallelism() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // `rayon(0)` means "as many threads as the global rayon pool has".
        faer::set_global_parallelism(faer::Par::rayon(0));
    });
}

/// Row-major dense matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "matrix data length mismatch");
        Self { rows, cols, data }
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// Borrow as a faer row-major view (no copy).
    #[inline]
    fn as_faer(&self) -> faer::MatRef<'_, f64> {
        faer::MatRef::from_row_major_slice(&self.data, self.rows, self.cols)
    }

    /// Borrow mutably as a faer row-major view (no copy).
    #[inline]
    fn as_faer_mut(&mut self) -> faer::MatMut<'_, f64> {
        faer::MatMut::from_row_major_slice_mut(&mut self.data, self.rows, self.cols)
    }

    /// Standard `self · other`, through faer's blocked/SIMD kernel.
    ///
    /// This used to be a naive `i-k-j` triple loop with a zero-skip: no blocking, no SIMD, no
    /// threads. It is on the SCF's critical path twice per iteration (the `FP − PF` DIIS
    /// commutator) and several times per perturbation inside the CPHF, so it dominated both.
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        self.matmul_with(other, faer::get_global_parallelism())
    }

    /// `self · other`, forced single-threaded.
    ///
    /// Use this inside a rayon parallel region. faer's own parallelism nests badly with an
    /// outer rayon loop -- the worker threads fight over the same pool -- and the outer loop
    /// already supplies the parallelism. The CPHF perturbation loop is exactly that case.
    pub fn matmul_seq(&self, other: &Matrix) -> Matrix {
        self.matmul_with(other, faer::Par::Seq)
    }

    /// `selfᵀ · other`, **without materializing the transpose**.
    ///
    /// faer's `MatRef::transpose` is a view, so the transpose costs nothing here. Writing this
    /// as `self.transpose().matmul(other)` instead allocates and fills a whole extra matrix per
    /// call — on the CPHF path, once per perturbation per iteration.
    pub fn transpose_matmul(&self, other: &Matrix) -> Matrix {
        self.transpose_matmul_with(other, faer::get_global_parallelism())
    }

    /// `selfᵀ · other`, forced single-threaded. Use inside a rayon region; see
    /// [`Self::matmul_seq`].
    pub fn transpose_matmul_seq(&self, other: &Matrix) -> Matrix {
        self.transpose_matmul_with(other, faer::Par::Seq)
    }

    fn transpose_matmul_with(&self, other: &Matrix, par: faer::Par) -> Matrix {
        assert_eq!(self.rows, other.rows, "matmul dimension mismatch");
        let mut out = Matrix::zeros(self.cols, other.cols);
        if self.cols == 0 || other.cols == 0 || self.rows == 0 {
            return out;
        }
        let (lhs, rhs) = (self.as_faer(), other.as_faer());
        let acc = faer::Accum::Replace;
        faer::linalg::matmul::matmul(out.as_faer_mut(), acc, lhs.transpose(), rhs, 1.0, par);
        out
    }

    /// `self · otherᵀ`, without materializing the transpose. See [`Self::transpose_matmul`].
    pub fn matmul_transpose(&self, other: &Matrix) -> Matrix {
        self.matmul_transpose_with(other, faer::get_global_parallelism())
    }

    /// `self · otherᵀ`, forced single-threaded. Use inside a rayon region.
    pub fn matmul_transpose_seq(&self, other: &Matrix) -> Matrix {
        self.matmul_transpose_with(other, faer::Par::Seq)
    }

    /// `dst += alpha · (self · other)`, single-threaded.
    ///
    /// The accumulating forms exist for complex arithmetic built out of real blocks, where
    /// `(A + iB)(C + iD)` is four products combined with two adds and two subtracts. Allocating
    /// a matrix per product and then walking it again to combine costs more than the products
    /// themselves at small `n` — enough that the first version of the periodic `project_ov`
    /// rewrite was 1.4x *slower* than the `O(n⁴)` loop nest it replaced on the cells the test
    /// suite uses. Accumulating in place removes both the temporaries and the extra passes.
    pub fn matmul_acc_seq(&self, other: &Matrix, dst: &mut Matrix, alpha: f64) {
        assert_eq!(self.cols, other.rows, "matmul dimension mismatch");
        assert_eq!((dst.rows, dst.cols), (self.rows, other.cols), "acc shape");
        if self.rows == 0 || other.cols == 0 || self.cols == 0 {
            return;
        }
        let (lhs, rhs) = (self.as_faer(), other.as_faer());
        let acc = faer::Accum::Add;
        faer::linalg::matmul::matmul(dst.as_faer_mut(), acc, lhs, rhs, alpha, faer::Par::Seq);
    }

    /// `dst += alpha · (selfᵀ · other)`, single-threaded. See [`Self::matmul_acc_seq`].
    pub fn transpose_matmul_acc_seq(&self, other: &Matrix, dst: &mut Matrix, alpha: f64) {
        assert_eq!(self.rows, other.rows, "matmul dimension mismatch");
        assert_eq!((dst.rows, dst.cols), (self.cols, other.cols), "acc shape");
        if self.cols == 0 || other.cols == 0 || self.rows == 0 {
            return;
        }
        let (lhs, rhs) = (self.as_faer(), other.as_faer());
        let acc = faer::Accum::Add;
        let t = lhs.transpose();
        faer::linalg::matmul::matmul(dst.as_faer_mut(), acc, t, rhs, alpha, faer::Par::Seq);
    }

    /// `dst += alpha · (self · otherᵀ)`, single-threaded. See [`Self::matmul_acc_seq`].
    pub fn matmul_transpose_acc_seq(&self, other: &Matrix, dst: &mut Matrix, alpha: f64) {
        self.matmul_transpose_acc_with(other, dst, alpha, faer::Par::Seq);
    }

    /// `dst += alpha · (self · otherᵀ)`. Use outside a rayon region; see
    /// [`Self::matmul_transpose_acc_seq`] for the one to use inside.
    pub fn matmul_transpose_acc(&self, other: &Matrix, dst: &mut Matrix, alpha: f64) {
        self.matmul_transpose_acc_with(other, dst, alpha, faer::get_global_parallelism());
    }

    fn matmul_transpose_acc_with(
        &self,
        other: &Matrix,
        dst: &mut Matrix,
        alpha: f64,
        par: faer::Par,
    ) {
        assert_eq!(self.cols, other.cols, "matmul dimension mismatch");
        assert_eq!((dst.rows, dst.cols), (self.rows, other.rows), "acc shape");
        if self.rows == 0 || other.rows == 0 || self.cols == 0 {
            return;
        }
        let (lhs, rhs) = (self.as_faer(), other.as_faer());
        let acc = faer::Accum::Add;
        let t = rhs.transpose();
        faer::linalg::matmul::matmul(dst.as_faer_mut(), acc, lhs, t, alpha, par);
    }

    fn matmul_transpose_with(&self, other: &Matrix, par: faer::Par) -> Matrix {
        assert_eq!(self.cols, other.cols, "matmul dimension mismatch");
        let mut out = Matrix::zeros(self.rows, other.rows);
        if self.rows == 0 || other.rows == 0 || self.cols == 0 {
            return out;
        }
        let (lhs, rhs) = (self.as_faer(), other.as_faer());
        let acc = faer::Accum::Replace;
        faer::linalg::matmul::matmul(out.as_faer_mut(), acc, lhs, rhs.transpose(), 1.0, par);
        out
    }

    fn matmul_with(&self, other: &Matrix, par: faer::Par) -> Matrix {
        assert_eq!(self.cols, other.rows, "matmul dimension mismatch");
        let mut out = Matrix::zeros(self.rows, other.cols);
        if self.rows == 0 || other.cols == 0 || self.cols == 0 {
            return out;
        }
        let (lhs, rhs) = (self.as_faer(), other.as_faer());
        faer::linalg::matmul::matmul(out.as_faer_mut(), faer::Accum::Replace, lhs, rhs, 1.0, par);
        out
    }

    /// Frobenius inner product `Σ_ij A_ij B_ij` (used for energies `½Σ P(H+F)`).
    ///
    /// Accumulated in eight independent partial sums rather than one. Floating-point addition is
    /// not associative, so a single running total is a dependency chain the compiler is not
    /// allowed to reorder or vectorize — the loop runs at one add per latency regardless of how
    /// wide the machine is. Eight lanes give it eight independent chains to interleave.
    ///
    /// This **changes the summation order**, so the last bits move. That is the same class of
    /// change as packing the DIIS history in 0.2.1, and acceptable for the same reason: no
    /// identity here is asserted at a tighter tolerance than the reordering can move, and the
    /// quantity is a sum of `nao²` terms whose ordering was never a defined part of the result.
    pub fn frobenius_dot(&self, other: &Matrix) -> f64 {
        debug_assert_eq!(self.data.len(), other.data.len());
        const LANES: usize = 8;
        let (a, b) = (self.data.as_slice(), other.data.as_slice());
        let n = a.len().min(b.len());
        let mut acc = [0.0f64; LANES];
        let chunks = n / LANES;
        for c in 0..chunks {
            let base = c * LANES;
            for (l, s) in acc.iter_mut().enumerate() {
                *s += a[base + l] * b[base + l];
            }
        }
        let mut total = acc.iter().sum::<f64>();
        for k in (chunks * LANES)..n {
            total += a[k] * b[k];
        }
        total
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}
impl std::ops::IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

/// Symmetric eigendecomposition (faer, pure-Rust — no LAPACK/BLAS).
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvalues in **ascending** order and
/// the eigenvectors as the **columns** of the returned matrix, so `A = V diag(λ) Vᵀ`.
pub fn symmetric_eigen(a: &Matrix) -> Result<(Vec<f64>, Matrix)> {
    let n = a.rows;
    if a.cols != n {
        return Err(Am1Error::LinearAlgebra(
            "symmetric_eigen requires a square matrix".to_string(),
        ));
    }
    if n == 0 {
        return Ok((Vec::new(), Matrix::zeros(0, 0)));
    }
    let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a[(i, j)]);
    let eigen = fa
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| Am1Error::LinearAlgebra(format!("faer eigendecomposition failed: {e:?}")))?;
    let s = eigen.S();
    let u = eigen.U();
    // Sort eigenpairs into ascending order (the SCF aufbau occupies the lowest orbitals;
    // faer's ordering is not guaranteed ascending, so enforce it here).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| s[i].partial_cmp(&s[j]).unwrap_or(std::cmp::Ordering::Equal));
    let values: Vec<f64> = order.iter().map(|&k| s[k]).collect();
    let mut vectors = Matrix::zeros(n, n);
    for (new_col, &old_col) in order.iter().enumerate() {
        for i in 0..n {
            vectors[(i, new_col)] = u[(i, old_col)];
        }
    }
    Ok((values, vectors))
}

/// Solve `A x = b` via faer's partial-pivot LU (pure-Rust — no LAPACK/BLAS).
pub fn solve_linear(a: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    let n = a.rows;
    if a.cols != n || b.len() != n {
        return Err(Am1Error::LinearAlgebra(
            "solve_linear dimension mismatch".to_string(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    use faer::linalg::solvers::Solve;
    let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a[(i, j)]);
    let rhs = faer::Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
    let lu = fa.partial_piv_lu();
    let x = lu.solve(&rhs);
    let sol: Vec<f64> = (0..n).map(|i| x[(i, 0)]).collect();
    // faer's LU does not error on a singular/near-singular system (it returns a huge,
    // low-residual solution). The old Gaussian-elimination path returned `Err` on a tiny
    // pivot so DIIS callers using `.ok()` fell back. Reproduce that: reject non-finite or
    // unreasonably large solutions (for a well-posed DIIS system the coefficients are O(1)).
    let bmax = b.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
    if sol.iter().any(|v| !v.is_finite()) || sol.iter().any(|v| v.abs() > 1.0e8 * bmax) {
        return Err(Am1Error::LinearAlgebra(
            "singular or ill-conditioned linear solve".to_string(),
        ));
    }
    Ok(sol)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(rows: usize, cols: usize, seed: u64) -> Matrix {
        let mut m = Matrix::zeros(rows, cols);
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for v in m.as_mut_slice() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        }
        m
    }

    fn max_abs_diff(a: &Matrix, b: &Matrix) -> f64 {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .fold(0.0f64, |acc, (x, y)| acc.max((x - y).abs()))
    }

    /// The transpose-free products must equal the materialize-then-multiply forms they replace.
    ///
    /// They feed a transposed operand to faer as a *view*, which is a different memory layout
    /// for the kernel than a materialized copy — so this checks the answer, and the timing test
    /// below checks that the layout did not cost more than the copy saved.
    #[test]
    fn the_transpose_free_products_agree_with_materializing_the_transpose() {
        for &(m, k, n) in &[(7usize, 5usize, 9usize), (64, 32, 48), (33, 65, 17)] {
            let a = filled(m, k, 3);
            let b = filled(n, k, 5);
            let c = filled(k, n, 11);

            let want = a.matmul(&b.transpose());
            for got in [a.matmul_transpose(&b), a.matmul_transpose_seq(&b)] {
                assert!(
                    max_abs_diff(&want, &got) < 1.0e-14,
                    "matmul_transpose {m}x{k}x{n}"
                );
            }

            let d = filled(k, m, 13);
            let want2 = d.transpose().matmul(&c);
            for got in [d.transpose_matmul(&c), d.transpose_matmul_seq(&c)] {
                assert!(
                    max_abs_diff(&want2, &got) < 1.0e-14,
                    "transpose_matmul {m}x{k}x{n}"
                );
            }

            // The accumulating forms are the same products with `dst += alpha·(…)`.
            let mut acc = want.clone();
            a.matmul_transpose_acc_seq(&b, &mut acc, -1.0);
            assert!(
                acc.as_slice().iter().all(|v| v.abs() < 1.0e-14),
                "acc cancel"
            );
        }
    }

    /// `P = C_occ C_occᵀ` is built once per SCF iteration, and materializing `C_occᵀ` is an
    /// extra `nao × n_occ` allocation and copy each time. Feeding faer a transposed view instead
    /// changes the kernel's access pattern, which can as easily cost as save — so this checks
    /// that it does not cost *catastrophically*.
    ///
    /// # Why the bound is loose and the statistic is a minimum
    ///
    /// This is a wall-clock measurement in a unit test, and a test machine may be running
    /// anything. Interference only ever makes a sample *slower*, so the **minimum** over several
    /// repetitions is the least-contended estimate and far steadier than a mean — a mean of three
    /// runs here reported 1.24× on an idle machine and 1.65× on a busy one, which is the load
    /// talking, not the code.
    ///
    /// The bound is `3×` because the failure this guards against is a kernel falling off its fast
    /// path entirely, which costs an order of magnitude, not a few percent. Anything tighter
    /// would be asserting that the machine is idle. The *ratio is printed* so a real regression
    /// is visible to a reader even when the assertion does not fire.
    #[test]
    fn the_transpose_free_density_product_is_not_pathologically_slower() {
        let (nao, nocc) = (600usize, 400usize);
        let c = filled(nao, nocc, 17);

        let best = |mut f: Box<dyn FnMut()>| -> f64 {
            f(); // warm up: first touch pays the page faults
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let with_copy = best(Box::new(|| {
            std::hint::black_box(c.matmul(&c.transpose()));
        }));
        let transpose_free = best(Box::new(|| {
            std::hint::black_box(c.matmul_transpose(&c));
        }));

        eprintln!(
            "    nao={nao} n_occ={nocc}: materialized {:.2} ms, transpose-free {:.2} ms ({:.2}x)",
            with_copy * 1e3,
            transpose_free * 1e3,
            with_copy / transpose_free
        );
        assert!(
            transpose_free < 3.0 * with_copy,
            "the transposed view costs {:.2}x the materialized copy, which is a kernel falling \
             off its fast path rather than ordinary machine noise",
            transpose_free / with_copy
        );
    }

    #[test]
    fn jacobi_diagonalizes_known_matrix() {
        // [[2,1],[1,2]] -> eigenvalues 1, 3.
        let a = Matrix::from_row_major(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (vals, vecs) = symmetric_eigen(&a).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
        // Reconstruct A = V Λ Vᵀ.
        let mut lam = Matrix::zeros(2, 2);
        lam[(0, 0)] = vals[0];
        lam[(1, 1)] = vals[1];
        let recon = vecs.matmul(&lam).matmul(&vecs.transpose());
        for i in 0..2 {
            for j in 0..2 {
                assert!((recon[(i, j)] - a[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn eigen_reconstructs_asymmetric_eigenvectors() {
        // Distinct eigenvalues -> eigenvector matrix is NOT symmetric; catches transpose bugs.
        let a = Matrix::from_row_major(3, 3, vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 1.0]);
        let (vals, vecs) = symmetric_eigen(&a).unwrap();
        // ascending
        assert!(vals[0] <= vals[1] && vals[1] <= vals[2]);
        let mut lam = Matrix::zeros(3, 3);
        for i in 0..3 {
            lam[(i, i)] = vals[i];
        }
        let recon = vecs.matmul(&lam).matmul(&vecs.transpose());
        for i in 0..3 {
            for j in 0..3 {
                assert!((recon[(i, j)] - a[(i, j)]).abs() < 1e-9, "recon[{i}][{j}]");
            }
        }
        // Columns orthonormal.
        for j in 0..3 {
            let nj: f64 = (0..3).map(|i| vecs[(i, j)] * vecs[(i, j)]).sum();
            assert!((nj - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn solve_linear_basic() {
        let a = Matrix::from_row_major(2, 2, vec![3.0, 2.0, 1.0, 2.0]);
        let x = solve_linear(&a, &[7.0, 5.0]).unwrap();
        // 3x+2y=7, x+2y=5 -> x=1, y=2.
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn solve_diis_saddle_point() {
        // DIIS-like bordered system: B = [[1,0.5,-1],[0.5,1,-1],[-1,-1,0]], rhs=[0,0,-1].
        // Known solution: c0 = c1 = 0.5, lambda = 0.75.
        let a = Matrix::from_row_major(3, 3, vec![1.0, 0.5, -1.0, 0.5, 1.0, -1.0, -1.0, -1.0, 0.0]);
        let x = solve_linear(&a, &[0.0, 0.0, -1.0]).unwrap();
        eprintln!("DIIS solve: {x:?}");
        assert!((x[0] - 0.5).abs() < 1e-9, "c0={}", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-9, "c1={}", x[1]);
    }
}
