// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO SCF driver — restricted (RHF) closed shell and unrestricted (UHF) open shell.
//!
//! Because NDDO assumes an orthonormal AO basis, the working equations are the plain
//! eigenproblem `F C = C ε` (no `S`); overlap enters only the resonance term of `H_core`.
//! The initial density is a **superposition of atomic densities** ([`sad_density`]) — the
//! exact free-atom density in a minimal valence basis, far better than the bare-core guess —
//! and charge convergence is accelerated with the A-DIIS→CDIIS hybrid on the `[F,P]` commutator.

use crate::basis::Basis;
use crate::constants::{AU_DIPOLE_TO_DEBYE, EV_TO_KCAL};
use crate::error::{Am1Error, Result};
use crate::fock::{build_fock, build_fock_spin};
use crate::hamiltonian::{build_core, CoreHamiltonian};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::params::Am1Parameters;
use crate::repulsion::core_core_energy;
use crate::system::Molecule;

/// Choice of SCF reference (spin treatment) — restricted vs unrestricted.
///
/// This is orthogonal to [`Am1Options::multiplicity`]: the multiplicity fixes the α/β electron
/// counts, while the reference chooses how the orbitals are solved. A closed-shell singlet can
/// therefore be run **either** restricted (RHF, the usual choice) **or** unrestricted (UHF, e.g.
/// as the starting point for a broken-symmetry singlet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScfReference {
    /// RHF for a closed shell (equal α/β counts), UHF otherwise. The default; preserves the
    /// historical behavior.
    Auto,
    /// Restricted Hartree–Fock: doubly-occupied spatial orbitals. Requires a closed shell
    /// (equal α/β counts); an open-shell request is rejected (no ROHF).
    Restricted,
    /// Unrestricted Hartree–Fock: independent α/β orbitals, used even for a singlet.
    Unrestricted,
}

/// SCF charge-convergence accelerator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScfAccelerator {
    /// No extrapolation (plain iteration).
    None,
    /// Pulay CDIIS on the `[F,P]` commutator throughout.
    Cdiis,
    /// **A-DIIS** (Hu & Yang, *J. Chem. Phys.* **132**, 054109 (2010)) while far from
    /// convergence, switching to CDIIS once the commutator error drops below a threshold —
    /// the robust hybrid recommended for hard cases (radicals, small gaps, poor guesses).
    AdiisCdiis,
}

#[derive(Clone, Debug)]
pub struct Am1Options {
    pub charge: f64,
    pub multiplicity: usize,
    /// Restricted vs unrestricted reference (see [`ScfReference`]). Default [`ScfReference::Auto`].
    pub reference: ScfReference,
    pub max_scf: usize,
    pub e_tol: f64,
    pub p_tol: f64,
    /// Legacy flag: `false` forces [`ScfAccelerator::None`] regardless of `accelerator`.
    pub use_diis: bool,
    pub accelerator: ScfAccelerator,
    /// Commutator-error norm below which the ADIIS→CDIIS hybrid switches to CDIIS.
    pub adiis_switch: f64,
}

impl Default for Am1Options {
    fn default() -> Self {
        Self {
            charge: 0.0,
            multiplicity: 1,
            reference: ScfReference::Auto,
            max_scf: 200,
            e_tol: 1.0e-8,
            p_tol: 1.0e-7,
            use_diis: true,
            accelerator: ScfAccelerator::AdiisCdiis,
            adiis_switch: 0.1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Am1Result {
    pub density: Matrix,
    /// Spin density `P_α − P_β` (open-shell UHF only; `None` for RHF).
    pub spin_density: Option<Matrix>,
    pub mo_energies: Vec<f64>,
    pub mo_coeff: Matrix,
    pub n_occ: usize,
    pub electronic_ev: f64,
    pub core_ev: f64,
    pub total_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub charges: Vec<f64>,
    pub dipole_debye: Vec3,
    pub dipole_magnitude: f64,
    pub homo_ev: Option<f64>,
    pub lumo_ev: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// True when the UHF (open-shell) path was used.
    pub unrestricted: bool,
}

pub struct Am1Calculator {
    pub params: Am1Parameters,
    pub options: Am1Options,
}

impl Am1Calculator {
    pub fn new(params: Am1Parameters) -> Self {
        Self {
            params,
            options: Am1Options::default(),
        }
    }
    pub fn with_options(params: Am1Parameters, options: Am1Options) -> Self {
        Self { params, options }
    }
    pub fn calculate(&self, molecule: &Molecule) -> Result<Am1Result> {
        run_am1(molecule, &self.params, &self.options)
    }
}

struct ScfState {
    density: Matrix,
    spin_density: Option<Matrix>,
    mo_energies: Vec<f64>,
    mo_coeff: Matrix,
    n_occ: usize,
    electronic_ev: f64,
    converged: bool,
    iterations: usize,
    unrestricted: bool,
}

pub fn run_am1(molecule: &Molecule, params: &Am1Parameters, options: &Am1Options) -> Result<Am1Result> {
    if options.multiplicity < 1 {
        return Err(Am1Error::InvalidInput("multiplicity must be >= 1".to_string()));
    }
    let basis = Basis::build(molecule, params)?;
    let core = build_core(molecule, &basis, params)?;

    let mut n_elec = 0.0;
    for atom in &molecule.atoms {
        n_elec += params.element(atom.z)?.core_charge;
    }
    n_elec -= options.charge;
    let n_elec_int = n_elec.round() as i64;
    if (n_elec - n_elec_int as f64).abs() > 1.0e-6 || n_elec_int < 0 {
        return Err(Am1Error::InvalidInput(format!(
            "invalid electron count {n_elec}"
        )));
    }
    let n_unpaired = (options.multiplicity - 1) as i64;
    if (n_elec_int - n_unpaired) < 0 || (n_elec_int - n_unpaired) % 2 != 0 {
        return Err(Am1Error::InvalidInput(format!(
            "electron count {n_elec_int} is incompatible with multiplicity {} (need same parity)",
            options.multiplicity
        )));
    }
    let n_alpha = ((n_elec_int + n_unpaired) / 2) as usize;
    let n_beta = ((n_elec_int - n_unpaired) / 2) as usize;
    let closed_shell = n_alpha == n_beta;

    // The reference (restricted/unrestricted) is chosen independently of the multiplicity:
    // `Auto` keeps the historical rule (RHF closed shell, UHF open shell); `Restricted` demands
    // a closed shell (no ROHF here); `Unrestricted` runs UHF even for a singlet.
    let use_uhf = match options.reference {
        ScfReference::Auto => !closed_shell,
        ScfReference::Unrestricted => true,
        ScfReference::Restricted => {
            if !closed_shell {
                return Err(Am1Error::InvalidInput(format!(
                    "restricted (RHF) reference requires a closed shell, but multiplicity {} \
                     gives {n_alpha} α and {n_beta} β electrons; use the unrestricted (UHF) \
                     reference for open-shell systems",
                    options.multiplicity
                )));
            }
            false
        }
    };

    let state = if !use_uhf {
        rhf_loop(molecule, &basis, params, &core, n_alpha, options)?
    } else {
        uhf_loop(molecule, &basis, params, &core, n_alpha, n_beta, options)?
    };

    let core_ev = core_core_energy(molecule, params)?;
    let total_ev = state.electronic_ev + core_ev;

    let mut e_isol_sum = 0.0;
    let mut eheat_sum = 0.0;
    for atom in &molecule.atoms {
        let e = params.element(atom.z)?;
        e_isol_sum += e.e_isol;
        eheat_sum += e.eheat_ev;
    }
    let heat_of_formation_kcal = (total_ev - e_isol_sum + eheat_sum) * EV_TO_KCAL;

    if !state.converged {
        return Err(Am1Error::ScfNotConverged {
            iterations: state.iterations,
            error: f64::NAN,
        });
    }

    // Mulliken net charges from the total density.
    let mut charges = vec![0.0; molecule.atoms.len()];
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let mut pop = 0.0;
        for mu in 0..n {
            pop += state.density[(off + mu, off + mu)];
        }
        charges[ia] = params.element(atom.z)?.core_charge - pop;
    }

    // Dipole: point-charge term + s–p hybrid polarization (both in e·Bohr).
    let mut dip = Vec3::zero();
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        dip += atom.position * charges[ia];
        let elem = params.element(atom.z)?;
        if elem.has_p() {
            let off = basis.atom_offset[ia];
            let hyb = -2.0 * elem.dd;
            dip += Vec3::new(
                hyb * state.density[(off, off + 1)],
                hyb * state.density[(off, off + 2)],
                hyb * state.density[(off, off + 3)],
            );
        }
    }
    let dipole_debye = dip * AU_DIPOLE_TO_DEBYE;
    let dipole_magnitude = dipole_debye.norm();

    let nao = basis.nao;
    let homo_ev = (state.n_occ >= 1).then(|| state.mo_energies[state.n_occ - 1]);
    let lumo_ev = (state.n_occ < nao).then(|| state.mo_energies[state.n_occ]);

    Ok(Am1Result {
        density: state.density,
        spin_density: state.spin_density,
        mo_energies: state.mo_energies,
        mo_coeff: state.mo_coeff,
        n_occ: state.n_occ,
        electronic_ev: state.electronic_ev,
        core_ev,
        total_ev,
        heat_of_formation_kcal,
        charges,
        dipole_debye,
        dipole_magnitude,
        homo_ev,
        lumo_ev,
        iterations: state.iterations,
        converged: state.converged,
        unrestricted: state.unrestricted,
    })
}

/// **Superposition of Atomic Densities (SAD)** initial guess: a block-diagonal density built
/// from each atom's spherically-averaged neutral valence configuration `s^{min(2,Zv)} p^{…}`.
/// In a minimal valence NDDO basis a free atom's density *is* diagonal (spherical symmetry), so
/// this superposition is the exact isolated-atom density and a far better SCF start than the
/// bare core (zero-density) guess — fewer iterations, more robust for larger systems.
fn sad_density(molecule: &Molecule, basis: &Basis, params: &Am1Parameters) -> Result<Matrix> {
    let nao = basis.nao;
    let mut p = Matrix::zeros(nao, nao);
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let zv = elem.core_charge;
        let n_s = zv.min(2.0); // s shell fills first (up to 2)
        let n_p = (zv - n_s).max(0.0); // remaining valence electrons in the p shell
        p[(off, off)] = n_s;
        if n == 4 {
            let per_p = n_p / 3.0; // spherically averaged over px, py, pz
            for k in 1..4 {
                p[(off + k, off + k)] = per_p;
            }
        }
    }
    Ok(p)
}

/// Build a density `P = w Σ_{k<n_occ} c_k c_kᵀ` from MO coefficients (`w` = 2 for RHF, 1 for UHF).
fn density_from_coeff(c: &Matrix, n_occ: usize, weight: f64) -> Matrix {
    let nao = c.rows;
    let mut p = Matrix::zeros(nao, nao);
    for mu in 0..nao {
        for nu in 0..nao {
            let mut acc = 0.0;
            for k in 0..n_occ {
                acc += c[(mu, k)] * c[(nu, k)];
            }
            p[(mu, nu)] = weight * acc;
        }
    }
    p
}

fn commutator(f: &Matrix, p: &Matrix) -> Matrix {
    let fp = f.matmul(p);
    let pf = p.matmul(f);
    let mut e = fp;
    for (ev, pv) in e.as_mut_slice().iter_mut().zip(pf.as_slice()) {
        *ev -= *pv;
    }
    e
}

fn rms_diff(a: &Matrix, b: &Matrix) -> f64 {
    let n = a.as_slice().len().max(1);
    (a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / n as f64)
        .sqrt()
}

fn rhf_loop(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    n_occ: usize,
    options: &Am1Options,
) -> Result<ScfState> {
    let nao = basis.nao;
    let mut density = sad_density(molecule, basis, params)?; // SAD initial guess
    let mut e_old = 0.0;
    let mut mo_energies = vec![0.0; nao];
    let mut mo_coeff = Matrix::zeros(nao, nao);
    let mut converged = false;
    let mut iterations = 0;
    let mut diis_f: Vec<Matrix> = Vec::new();
    let mut diis_e: Vec<Matrix> = Vec::new();
    let mut diis_d: Vec<Matrix> = Vec::new();
    let max_diis = 8;
    let accel = if options.use_diis {
        options.accelerator
    } else {
        ScfAccelerator::None
    };

    for iter in 0..options.max_scf {
        iterations = iter + 1;
        let f = build_fock(molecule, basis, params, core, &density)?;
        let e_elec = 0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f));
        let err = commutator(&f, &density);
        let err_norm = err.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();

        // History (Fock, commutator, density) for CDIIS / A-DIIS.
        diis_f.push(f.clone());
        diis_e.push(err);
        diis_d.push(density.clone());
        if diis_f.len() > max_diis {
            diis_f.remove(0);
            diis_e.remove(0);
            diis_d.remove(0);
        }

        let f_use = match accel {
            ScfAccelerator::None => f,
            ScfAccelerator::Cdiis => diis_extrapolate(&diis_f, &diis_e).unwrap_or_else(|| f.clone()),
            ScfAccelerator::AdiisCdiis => {
                if err_norm > options.adiis_switch {
                    adiis_extrapolate(&diis_d, &diis_f).unwrap_or_else(|| f.clone())
                } else {
                    diis_extrapolate(&diis_f, &diis_e).unwrap_or_else(|| f.clone())
                }
            }
        };
        let (eps, c) = symmetric_eigen(&f_use)?;
        let p_new = density_from_coeff(&c, n_occ, 2.0);
        let dp = rms_diff(&p_new, &density);
        let de = (e_elec - e_old).abs();

        mo_energies = eps;
        mo_coeff = c;
        density = p_new;
        e_old = e_elec;
        if iter > 0 && de < options.e_tol && (dp < options.p_tol || err_norm < 1.0e-7) {
            converged = true;
            break;
        }
    }
    let f_final = build_fock(molecule, basis, params, core, &density)?;
    let electronic_ev = 0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f_final));

    Ok(ScfState {
        density,
        spin_density: None,
        mo_energies,
        mo_coeff,
        n_occ,
        electronic_ev,
        converged,
        iterations,
        unrestricted: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn uhf_loop(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    n_alpha: usize,
    n_beta: usize,
    options: &Am1Options,
) -> Result<ScfState> {
    let nao = basis.nao;
    // SAD guess split by spin population; the different α/β aufbau counts break spin symmetry.
    let sad = sad_density(molecule, basis, params)?;
    let n_tot = (n_alpha + n_beta).max(1) as f64;
    let (fa, fb) = (n_alpha as f64 / n_tot, n_beta as f64 / n_tot);
    let mut pa = sad.clone();
    for v in pa.as_mut_slice() {
        *v *= fa;
    }
    let mut pb = sad;
    for v in pb.as_mut_slice() {
        *v *= fb;
    }
    let mut e_old = 0.0;
    let mut eps_a = vec![0.0; nao];
    let mut c_a = Matrix::zeros(nao, nao);
    let mut converged = false;
    let mut iterations = 0;

    let mut hist_fa: Vec<Matrix> = Vec::new();
    let mut hist_fb: Vec<Matrix> = Vec::new();
    let mut hist_err: Vec<Matrix> = Vec::new();
    let max_diis = 8;

    for iter in 0..options.max_scf {
        iterations = iter + 1;
        let mut p_tot = pa.clone();
        for (t, b) in p_tot.as_mut_slice().iter_mut().zip(pb.as_slice()) {
            *t += *b;
        }
        let fa = build_fock_spin(molecule, basis, params, core, &p_tot, &pa)?;
        let fb = build_fock_spin(molecule, basis, params, core, &p_tot, &pb)?;

        let e_elec = 0.5
            * (p_tot.frobenius_dot(&core.h_core)
                + pa.frobenius_dot(&fa)
                + pb.frobenius_dot(&fb));

        // Combined DIIS error = [F_a,P_a] ⊕ [F_b,P_b].
        let ea = commutator(&fa, &pa);
        let eb = commutator(&fb, &pb);
        let mut err = Matrix::zeros(2 * nao, nao);
        for i in 0..nao {
            for j in 0..nao {
                err[(i, j)] = ea[(i, j)];
                err[(nao + i, j)] = eb[(i, j)];
            }
        }
        let err_norm = err.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();

        let (fa_use, fb_use) = if options.use_diis {
            hist_fa.push(fa.clone());
            hist_fb.push(fb.clone());
            hist_err.push(err);
            if hist_fa.len() > max_diis {
                hist_fa.remove(0);
                hist_fb.remove(0);
                hist_err.remove(0);
            }
            match diis_coeffs(&hist_err) {
                Some(coeffs) => (
                    combine(&hist_fa, &coeffs),
                    combine(&hist_fb, &coeffs),
                ),
                None => (fa, fb),
            }
        } else {
            (fa, fb)
        };

        let (ea_eps, ca) = symmetric_eigen(&fa_use)?;
        let (_eb_eps, cb) = symmetric_eigen(&fb_use)?;
        let pa_new = density_from_coeff(&ca, n_alpha, 1.0);
        let pb_new = density_from_coeff(&cb, n_beta, 1.0);

        let dp = rms_diff(&pa_new, &pa) + rms_diff(&pb_new, &pb);
        let de = (e_elec - e_old).abs();
        eps_a = ea_eps;
        c_a = ca;
        pa = pa_new;
        pb = pb_new;
        e_old = e_elec;
        if iter > 0 && de < options.e_tol && (dp < options.p_tol || err_norm < 1.0e-7) {
            converged = true;
            break;
        }
    }

    let mut density = pa.clone();
    for (t, b) in density.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *t += *b;
    }
    let mut spin = pa.clone();
    for (s, b) in spin.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *s -= *b;
    }
    // Final energy.
    let fa = build_fock_spin(molecule, basis, params, core, &density, &pa)?;
    let fb = build_fock_spin(molecule, basis, params, core, &density, &pb)?;
    let electronic_ev =
        0.5 * (density.frobenius_dot(&core.h_core) + pa.frobenius_dot(&fa) + pb.frobenius_dot(&fb));

    Ok(ScfState {
        density,
        spin_density: Some(spin),
        mo_energies: eps_a,
        mo_coeff: c_a,
        n_occ: n_alpha,
        electronic_ev,
        converged,
        iterations,
        unrestricted: true,
    })
}

fn combine(fs: &[Matrix], coeffs: &[f64]) -> Matrix {
    let (r, c) = (fs[0].rows, fs[0].cols);
    let mut out = Matrix::zeros(r, c);
    for (i, f) in fs.iter().enumerate() {
        let ci = coeffs[i];
        for (o, v) in out.as_mut_slice().iter_mut().zip(f.as_slice()) {
            *o += ci * v;
        }
    }
    out
}

/// Solve the Pulay DIIS coefficient system from a stack of error matrices.
fn diis_coeffs(es: &[Matrix]) -> Option<Vec<f64>> {
    let n = es.len();
    if n < 2 {
        return None;
    }
    let dim = n + 1;
    let mut b = Matrix::zeros(dim, dim);
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] = es[i].frobenius_dot(&es[j]);
        }
        b[(i, n)] = -1.0;
        b[(n, i)] = -1.0;
    }
    let mut rhs = vec![0.0; dim];
    rhs[n] = -1.0;
    // The DIIS matrix (a small bordered saddle-point system) becomes singular near
    // convergence when the error vectors turn linearly dependent. A pivot-guarded
    // Gaussian elimination returns `None` there so the caller falls back to the plain
    // Fock — faer's LU instead returns a degenerate solution that derails DIIS. The heavy
    // O(n^3) eigendecomposition still uses faer; only this tiny solve is bespoke.
    solve_bordered_small(&b, &rhs)
}

/// Gaussian elimination with partial pivoting for the small DIIS system; returns `None`
/// if the matrix is (near-)singular.
fn solve_bordered_small(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.rows;
    let mut m = a.clone();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[(col, col)].abs();
        for row in (col + 1)..n {
            let v = m[(row, col)].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1.0e-12 {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                let t = m[(col, j)];
                m[(col, j)] = m[(pivot, j)];
                m[(pivot, j)] = t;
            }
            rhs.swap(col, pivot);
        }
        for row in (col + 1)..n {
            let factor = m[(row, col)] / m[(col, col)];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                let v = m[(col, j)];
                m[(row, j)] -= factor * v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..n {
            sum -= m[(col, j)] * x[j];
        }
        x[col] = sum / m[(col, col)];
    }
    Some(x)
}

fn diis_extrapolate(fs: &[Matrix], es: &[Matrix]) -> Option<Matrix> {
    let coeffs = diis_coeffs(es)?;
    Some(combine(fs, &coeffs))
}

fn mat_sub(a: &Matrix, b: &Matrix) -> Matrix {
    let mut o = a.clone();
    for (ov, bv) in o.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *ov -= *bv;
    }
    o
}

/// A-DIIS (Hu & Yang 2010): minimize `f(c) = 2 Σ c_i ⟨D_i−D_n|F_n⟩ + Σ c_i c_j ⟨D_i−D_n|F_j−F_n⟩`
/// over the simplex `{c ≥ 0, Σc = 1}`, then return the extrapolated Fock `Σ c_i F_i`.
/// Robust far from convergence (nonnegative weights prevent the runaway extrapolation that
/// plain DIIS can produce with a poor initial guess).
fn adiis_extrapolate(densities: &[Matrix], focks: &[Matrix]) -> Option<Matrix> {
    let n = densities.len();
    if n < 2 {
        return None;
    }
    let dn = &densities[n - 1];
    let fnl = &focks[n - 1];
    let dd: Vec<Matrix> = densities.iter().map(|d| mat_sub(d, dn)).collect();
    let ff: Vec<Matrix> = focks.iter().map(|f| mat_sub(f, fnl)).collect();
    let d: Vec<f64> = dd.iter().map(|ddi| ddi.frobenius_dot(fnl)).collect();
    let mut s = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            s[i][j] = dd[i].frobenius_dot(&ff[j]);
        }
    }
    let c = solve_adiis_simplex(&d, &s);
    Some(combine(focks, &c))
}

/// Projected-gradient minimization of the A-DIIS quadratic on the probability simplex.
fn solve_adiis_simplex(d: &[f64], s: &[Vec<f64>]) -> Vec<f64> {
    let n = d.len();
    // Lipschitz estimate for the step size from (S + Sᵀ).
    let mut l: f64 = 1.0e-12;
    for i in 0..n {
        let mut row = 0.0;
        for j in 0..n {
            row += (s[i][j] + s[j][i]).abs();
        }
        l = l.max(row);
    }
    let lr = 1.0 / l;
    // Start from the latest point (all weight on the newest Fock/density).
    let mut c = vec![0.0; n];
    c[n - 1] = 1.0;
    for _ in 0..400 {
        // grad_k = 2 d_k + Σ_j (s_kj + s_jk) c_j
        let mut g = vec![0.0; n];
        for k in 0..n {
            let mut acc = 2.0 * d[k];
            for j in 0..n {
                acc += (s[k][j] + s[j][k]) * c[j];
            }
            g[k] = acc;
        }
        let trial: Vec<f64> = (0..n).map(|i| c[i] - lr * g[i]).collect();
        let proj = simplex_project(&trial);
        let mut delta = 0.0;
        for i in 0..n {
            delta += (proj[i] - c[i]).abs();
        }
        c = proj;
        if delta < 1.0e-12 {
            break;
        }
    }
    c
}

/// Euclidean projection of `v` onto the probability simplex `{c ≥ 0, Σc = 1}`.
fn simplex_project(v: &[f64]) -> Vec<f64> {
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut css = 0.0;
    let mut rho = 0;
    let mut theta = 0.0;
    for (j, &uj) in u.iter().enumerate() {
        css += uj;
        let t = (css - 1.0) / (j as f64 + 1.0);
        if uj - t > 0.0 {
            rho = j + 1;
            theta = t;
        }
    }
    let _ = rho;
    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(xyz: &str, charge: f64) -> Am1Result {
        let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
        let params = Am1Parameters::standard().unwrap();
        run_am1(&mol, &params, &Am1Options::default()).unwrap()
    }

    fn run_mult(xyz: &str, charge: f64, mult: usize) -> Am1Result {
        let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options {
            charge,
            multiplicity: mult,
            ..Am1Options::default()
        };
        run_am1(&mol, &params, &opts).unwrap()
    }

    #[test]
    fn water_heat_of_formation() {
        let xyz = "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let r = run(xyz, 0.0);
        eprintln!(
            "H2O: dHf={:.3} kcal/mol  elec={:.4} eV core={:.4} eV  dipole={:.3} D  charges={:?}  iters={}",
            r.heat_of_formation_kcal, r.electronic_ev, r.core_ev, r.dipole_magnitude, r.charges, r.iterations
        );
        assert!(r.converged);
        assert!((r.heat_of_formation_kcal - (-59.24)).abs() < 0.5);
        assert!((r.dipole_magnitude - 1.86).abs() < 0.15);
        let qsum: f64 = r.charges.iter().sum();
        assert!(qsum.abs() < 1e-6);
        assert!(r.charges[0] < 0.0 && r.charges[1] > 0.0);
    }

    #[test]
    fn water_no_diis_debug() {
        let xyz = "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options {
            use_diis: false,
            max_scf: 500,
            ..Am1Options::default()
        };
        let r = run_am1(&mol, &params, &opts).unwrap();
        eprintln!(
            "H2O(no-diis): dHf={:.3} conv={} iters={}",
            r.heat_of_formation_kcal, r.converged, r.iterations
        );
    }

    #[test]
    fn methane_heat_of_formation() {
        let xyz = "5\nmethane\nC 0.0000 0.0000 0.0000\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";
        let r = run(xyz, 0.0);
        eprintln!("CH4: dHf={:.3} kcal/mol dipole={:.3} D iters={}", r.heat_of_formation_kcal, r.dipole_magnitude, r.iterations);
        assert!(r.converged);
    }

    #[test]
    fn accelerators_agree_on_energy() {
        // A-DIIS→CDIIS, plain CDIIS, and no acceleration must reach the same converged
        // energy (same SCF fixed point); the hybrid should not need more iterations.
        let xyz = "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.0 0.0 1.21\nH 0.94 0.0 -0.54\nH -0.94 0.0 -0.54\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let run = |acc: ScfAccelerator| {
            let opts = Am1Options {
                accelerator: acc,
                ..Am1Options::default()
            };
            run_am1(&mol, &params, &opts).unwrap()
        };
        let hybrid = run(ScfAccelerator::AdiisCdiis);
        let cdiis = run(ScfAccelerator::Cdiis);
        let none = run(ScfAccelerator::None);
        eprintln!(
            "iters: hybrid={} cdiis={} none={}  E(hybrid)={:.6}",
            hybrid.iterations, cdiis.iterations, none.iterations, hybrid.total_ev
        );
        assert!((hybrid.total_ev - cdiis.total_ev).abs() < 1e-6);
        assert!((hybrid.total_ev - none.total_ev).abs() < 1e-6);
        // Both accelerated paths should be far faster than plain iteration.
        assert!(hybrid.iterations < none.iterations);
    }

    #[test]
    fn methyl_radical_uhf() {
        // Planar CH3 radical (doublet): UHF must converge, be open-shell, and carry net spin.
        let xyz = "4\nmethyl\nC 0.0 0.0 0.0\nH 1.079 0.0 0.0\nH -0.5395 0.9344 0.0\nH -0.5395 -0.9344 0.0\n";
        let r = run_mult(xyz, 0.0, 2);
        eprintln!(
            "CH3.: dHf={:.3} kcal/mol unrestricted={} iters={}",
            r.heat_of_formation_kcal, r.unrestricted, r.iterations
        );
        assert!(r.converged);
        assert!(r.unrestricted);
        // Total spin population (∫ P_α − P_β) should be ≈ 1 unpaired electron.
        let spin = r.spin_density.as_ref().unwrap();
        let n_spin: f64 = (0..spin.rows).map(|i| spin[(i, i)]).sum();
        assert!((n_spin - 1.0).abs() < 1e-6, "net spin {n_spin}");
        // Published AM1 ΔHf(CH3·) ≈ +32 kcal/mol.
        assert!((r.heat_of_formation_kcal - 32.0).abs() < 6.0);
    }

    #[test]
    fn forced_uhf_singlet_matches_rhf() {
        // A closed-shell singlet run as UHF must take the unrestricted path yet converge to the
        // RHF energy (a symmetric guess yields zero spin density).
        let xyz = "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let rhf = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let uhf = run_am1(
            &mol,
            &params,
            &Am1Options { reference: ScfReference::Unrestricted, ..Am1Options::default() },
        )
        .unwrap();
        assert!(!rhf.unrestricted);
        assert!(uhf.unrestricted, "forced reference did not select UHF");
        assert!((rhf.total_ev - uhf.total_ev).abs() < 1e-6, "UHF singlet != RHF energy");
        // Net spin of a (symmetric) singlet must be ≈ 0.
        let spin = uhf.spin_density.as_ref().unwrap();
        let n_spin: f64 = (0..spin.rows).map(|i| spin[(i, i)]).sum();
        assert!(n_spin.abs() < 1e-6, "unexpected net spin {n_spin}");
    }

    #[test]
    fn forced_rhf_singlet_matches_auto() {
        // Explicitly restricting a closed shell is identical to Auto.
        let xyz = "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let auto = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let rhf = run_am1(
            &mol,
            &params,
            &Am1Options { reference: ScfReference::Restricted, ..Am1Options::default() },
        )
        .unwrap();
        assert!(!rhf.unrestricted);
        assert!((auto.total_ev - rhf.total_ev).abs() < 1e-12);
    }

    #[test]
    fn restricted_rejects_open_shell() {
        // RHF cannot represent an open shell (no ROHF): a doublet request must be an error.
        let xyz = "4\nmethyl\nC 0.0 0.0 0.0\nH 1.079 0.0 0.0\nH -0.5395 0.9344 0.0\nH -0.5395 -0.9344 0.0\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let res = run_am1(
            &mol,
            &params,
            &Am1Options {
                multiplicity: 2,
                reference: ScfReference::Restricted,
                ..Am1Options::default()
            },
        );
        assert!(res.is_err(), "restricted open-shell request should be rejected");
    }
}
