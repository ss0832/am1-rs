// SPDX-License-Identifier: GPL-3.0-or-later

//! A finite electric field **along** a periodic direction, by the Berry-phase electric enthalpy.
//!
//! # Why `F·R` cannot be used here
//!
//! Under a cell, `F·R` shifts by `F·T` under translation by `T`, so it is a lattice-periodic
//! perturbation exactly when `F·T = 0` for every lattice vector. A field orthogonal to every one
//! of them — normal to a slab, transverse to a chain — is an ordinary calculation and goes through
//! [`crate::pbc::PbcOptions::electric_field`]. Along a periodic direction the potential is
//! unbounded, the spectrum has no lower bound, and no amount of care in the assembly fixes it: the
//! ground state of `H − F·R` on a periodic lattice does not exist.
//!
//! # What replaces it
//!
//! The **electric enthalpy** of Nunes and Gonze, minimized instead of the energy:
//!
//! ```text
//! F[ψ, 𝓔] = E[ψ] − Ω 𝓔·P[ψ]
//! ```
//!
//! with `P` the Berry-phase polarization ([`crate::pbc::berry`]) rather than `⟨r⟩`. `P` is a
//! property of the occupied *manifold* across the Brillouin zone, well defined modulo `e a/Ω`, and
//! its derivative with respect to the orbitals is what enters the Hamiltonian.
//!
//! Because `P` is built from overlaps between **neighbouring k points**, that derivative couples
//! them: the field term at `k` reads the coefficients at `k ± b`. So the k points can no longer be
//! solved one at a time, which is the structural reason this is not a small change to the SCF.
//!
//! # The coupling, derived rather than quoted
//!
//! Sign and factor conventions for the Berry phase differ between sources, and a wrong factor here
//! does not fail — it returns a plausible polarizability. So it is derived from *this crate's own*
//! polarization convention, which is fixed and documented in [`crate::pbc::berry`]:
//!
//! ```text
//! P_el = (2/Ω) Σ_α a_α φ̃_α,     φ̃_α = (1/2π) Im ln Π_j det S_j
//! ```
//!
//! with the factor 2 for the spin pair and `S_j = C_j† Λ C_{j+1}`, `Λ = diag(e^{−ib·τ_μ})`. Then
//!
//! ```text
//! Ω 𝓔·P_el = 2 Σ_α (𝓔·a_α) φ̃_α
//! ```
//!
//! Differentiating `Im ln Z` with respect to `C*(k_j)`, treating `C` and `C*` as independent:
//!
//! ```text
//! ∂(Im ln Z)/∂C*(k_j) = (1/2i) [ Λ C_{j+1} S_j⁻¹  −  Λ* C_{j−1} (S_{j−1}⁻¹)† ]
//! ```
//!
//! The energy's own gradient is `w_k f (H C)`, with `w_k = 1/(J N_⊥)` on a regular mesh and
//! `f = 2`; dividing through by it turns the enthalpy's gradient into an operator:
//!
//! ```text
//! ΔH C(k_j) = i λ_α [ Λ C_{j+1} S_j⁻¹ − Λ* C_{j−1} (S_{j−1}⁻¹)† ],   λ_α = (𝓔·a_α) J/(4π)
//! ```
//!
//! and `ΔH = ½(M + M†)` with `M = i λ (W₊ − W₋) C(k_j)†`, Hermitized because the two halves agree
//! only at self-consistency.
//!
//! # What says the factor is right
//!
//! Not the derivation. `tests/pbc_finite_field.rs` computes `α = Ω ∂P/∂𝓔` by finite differences of
//! this and compares it against the **CPHF** polarizability from
//! [`crate::pbc::dielectric_tensor`] — two formalisms sharing only the SCF. A factor of two, a
//! missing `J`, or a sign shows up there immediately and nowhere else.

use crate::error::{Am1Error, Result};
use crate::lattice::Lattice;
use crate::math::Vec3;
use crate::params::Am1Parameters;
use crate::pbc::complex::{hermitian_eigen, CMatrix};
use crate::pbc::kpoints::KPoint;
use crate::pbc::scf::{PbcOptions, PbcResult};
use crate::system::Molecule;

/// A converged finite-field calculation.
#[derive(Clone, Debug)]
pub struct FiniteFieldResult {
    /// The self-consistent state in the field.
    pub scf: PbcResult,
    /// The applied field, eV per (e·Bohr).
    pub field: Vec3,
    /// Berry phases in turns, one per lattice direction, averaged over transverse strings.
    pub phase: [f64; 3],
    /// Electronic polarization, `e/Bohr²` in 3D.
    pub electronic_polarization: Vec3,
    /// Ionic polarization, same units.
    pub ionic_polarization: Vec3,
    /// `electronic + ionic`, modulo the quantum `e a_α/Ω`.
    pub polarization: Vec3,
    /// `E − Ω 𝓔·P`, eV: the quantity actually minimized.
    pub enthalpy_ev: f64,
    /// Outer (field-operator) iterations taken.
    pub iterations: usize,
    pub converged: bool,
}

/// Outer-loop settings for the field operator.
#[derive(Clone, Copy, Debug)]
pub struct FiniteFieldOptions {
    /// Convergence threshold on the largest change in `ΔH` between outer iterations, eV.
    pub tol: f64,
    pub max_iter: usize,
    /// Linear mixing on `ΔH`. The field operator is a strongly non-local function of the
    /// coefficients and a full step oscillates on anything but the smallest fields.
    pub mixing: f64,
}

impl Default for FiniteFieldOptions {
    fn default() -> Self {
        Self {
            tol: 1.0e-8,
            max_iter: 60,
            mixing: 0.5,
        }
    }
}

/// Solve the periodic SCF in a finite field with a component along a periodic direction.
///
/// `options.kmesh` must be a Γ-centred Monkhorst–Pack grid with **at least three points** along
/// every direction the field has a component in: the Berry phase is a product of nearest-neighbour
/// overlaps around the zone, and two points cannot resolve a winding.
pub fn run_finite_field(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    field: Vec3,
    ff: &FiniteFieldOptions,
) -> Result<FiniteFieldResult> {
    let cell = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a finite-field calculation needs a cell".into()))?;
    if !cell.is_fully_periodic() {
        return Err(Am1Error::InvalidInput(
            "the Berry-phase finite field is implemented for a three-dimensional cell, matching \
             `pbc::berry`. A slab or a chain has a polarization along its periodic directions \
             only, which that module does not yet separate out — and a field along a \
             *non-periodic* direction needs none of this machinery: set \
             `PbcOptions::electric_field`."
                .into(),
        ));
    }
    if options.smearing_ev > 0.0 {
        return Err(Am1Error::InvalidInput(
            "a Berry phase needs a gapped, integer-filled manifold, so the finite field cannot be \
             combined with Fermi smearing. Set `smearing_ev: 0.0`."
                .into(),
        ));
    }
    if options.unrestricted || options.multiplicity > 1 {
        return Err(Am1Error::InvalidInput(
            "the Berry-phase finite field is restricted-only, matching `pbc::berry`: an \
             open-shell cell would need the phase of each spin manifold separately"
                .into(),
        ));
    }
    if options.electric_field.is_some() {
        return Err(Am1Error::InvalidInput(
            "`PbcOptions::electric_field` and the Berry-phase finite field are two treatments of \
             the same perturbation; pass the field to `run_finite_field` only. The `F·R` form is \
             for a field orthogonal to every lattice vector, this one for a field along a \
             periodic direction."
                .into(),
        ));
    }

    let mesh = grid_from(options, &cell, field)?;
    let kpoints = mesh.kpoints();
    // The field operator is **odd** under `k → −k`, so time-reversal folding — which merges the
    // two and is exact for the ground state — would average it against its own negative. The
    // explicit list also pins the ordering the `k_terms` are indexed by.
    let scf_options = PbcOptions {
        kpoints: Some(kpoints.clone()),
        fold_time_reversal: false,
        ..options.clone()
    };

    let basis = crate::basis::Basis::build(molecule, params)?;
    let nao = basis.nao;
    let n_occ = occupied_bands(molecule, params, options)?;

    // `e^{−ib·τ_μ}` needs each orbital's atom position; `b` differs per direction, so the
    let mut k_terms: Vec<CMatrix> = kpoints.iter().map(|_| CMatrix::zeros(nao)).collect();
    let mut scf = crate::pbc::scf::run_pbc_scf_with_k_terms(molecule, params, &scf_options, None)?;
    let mut phase = [0.0_f64; 3];
    let mut converged = false;
    let mut iterations = 0usize;

    for iter in 0..ff.max_iter {
        iterations = iter + 1;
        scf = crate::pbc::scf::run_pbc_scf_with_k_terms(
            molecule,
            params,
            &scf_options,
            Some(&k_terms),
        )?;
        if !scf.converged {
            return Err(Am1Error::InvalidInput(format!(
                "the periodic SCF did not converge inside the finite field at outer iteration \
                 {iterations}; a smaller field or more `max_scf` may help"
            )));
        }

        // The converged coefficients, from the same Hamiltonian the SCF diagonalized: its own
        // Fock plus the field term that was held fixed through it.
        let coefficients = coefficients_in_field(
            molecule,
            params,
            &scf_options,
            &basis,
            &scf,
            &kpoints,
            &k_terms,
        )?;

        let (next, new_phase) = field_operator(
            &mesh,
            molecule,
            &basis,
            params,
            &cell,
            &coefficients,
            n_occ,
            field,
        )?;
        phase = new_phase;

        let mut change = 0.0_f64;
        for (old, new) in k_terms.iter().zip(&next) {
            for i in 0..nao {
                for j in 0..nao {
                    let (ar, ai) = old.get(i, j);
                    let (br, bi) = new.get(i, j);
                    change = change.max((ar - br).abs()).max((ai - bi).abs());
                }
            }
        }
        for (old, new) in k_terms.iter_mut().zip(&next) {
            for i in 0..nao {
                for j in 0..nao {
                    let (ar, ai) = old.get(i, j);
                    let (br, bi) = new.get(i, j);
                    old.re[(i, j)] = ar + ff.mixing * (br - ar);
                    old.im[(i, j)] = ai + ff.mixing * (bi - ai);
                }
            }
        }
        if change < ff.tol {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(Am1Error::InvalidInput(format!(
            "the finite-field operator did not converge in {} outer iterations",
            ff.max_iter
        )));
    }

    let volume = cell.volume();
    let mut electronic = Vec3::zero();
    for alpha in 0..3 {
        electronic += cell.cell.col[alpha] * (2.0 * phase[alpha] / volume);
    }
    let mut ionic = Vec3::zero();
    for atom in &molecule.atoms {
        ionic += atom.position * (params.element(atom.z)?.core_charge / volume);
    }
    let polarization = electronic + ionic;

    Ok(FiniteFieldResult {
        // `E − Ω 𝓔·P`: the enthalpy is the variational quantity, and `scf.total_ev` is the plain
        // energy of the polarized state. Reporting both is what lets a caller check that the
        // field lowered the enthalpy while the energy itself rose.
        enthalpy_ev: scf.total_ev - volume * field.dot(polarization),
        scf,
        field,
        phase,
        electronic_polarization: electronic,
        ionic_polarization: ionic,
        polarization,
        iterations,
        converged,
    })
}

/// A Γ-centred Monkhorst–Pack grid, with the index arithmetic the strings need.
struct Grid {
    n: [usize; 3],
}

impl Grid {
    fn kpoints(&self) -> Vec<KPoint> {
        let total = (self.n[0] * self.n[1] * self.n[2]) as f64;
        let mut out = Vec::with_capacity(total as usize);
        for i in 0..self.n[0] {
            for j in 0..self.n[1] {
                for l in 0..self.n[2] {
                    out.push(KPoint {
                        fractional: [
                            i as f64 / self.n[0] as f64,
                            j as f64 / self.n[1] as f64,
                            l as f64 / self.n[2] as f64,
                        ],
                        weight: 1.0 / total,
                    });
                }
            }
        }
        out
    }

    #[inline]
    fn index(&self, i: usize, j: usize, l: usize) -> usize {
        (i * self.n[1] + j) * self.n[2] + l
    }

    /// The index of the point one step along `alpha` from `(i, j, l)`, wrapping the zone.
    #[inline]
    fn step(&self, idx: [usize; 3], alpha: usize, forward: bool) -> usize {
        let mut next = idx;
        let n = self.n[alpha];
        next[alpha] = if forward {
            (idx[alpha] + 1) % n
        } else {
            (idx[alpha] + n - 1) % n
        };
        self.index(next[0], next[1], next[2])
    }
}

/// The mesh, checked against what the field needs of it.
fn grid_from(options: &PbcOptions, cell: &Lattice, field: Vec3) -> Result<Grid> {
    if options.kpoints.is_some() {
        return Err(Am1Error::InvalidInput(
            "the finite field needs a regular Γ-centred grid so that `k ± b` is in the sampled \
             set; give it `kmesh`, not an explicit `kpoints` list"
                .into(),
        ));
    }
    let n = options.kmesh.sizes();
    let recip = cell.reciprocal_vectors_2pi();
    for (alpha, size) in n.iter().enumerate() {
        // Only the directions the field actually has a component along need a string. `a_α·𝓔` is
        // the coupling, so a field perpendicular to `a_α` puts no requirement on that axis.
        if field.dot(cell.cell.col[alpha]).abs() < 1.0e-12 * field.norm().max(1.0) {
            continue;
        }
        let _ = recip;
        if *size < 3 {
            return Err(Am1Error::InvalidInput(format!(
                "the field has a component along lattice vector {alpha}, which needs at least 3 \
                 k points along that direction to resolve a Berry phase; the mesh has {size}"
            )));
        }
    }
    Ok(Grid {
        n: [n[0].max(1), n[1].max(1), n[2].max(1)],
    })
}

/// Doubly occupied bands per cell, or an error when the manifold is not filled.
fn occupied_bands(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<usize> {
    let mut n_elec = -options.charge;
    for atom in &molecule.atoms {
        n_elec += params.element(atom.z)?.core_charge;
    }
    let n_occ = (n_elec / 2.0).round() as usize;
    if (n_elec / 2.0 - n_occ as f64).abs() > 1.0e-9 {
        return Err(Am1Error::InvalidInput(format!(
            "a Berry phase needs a filled manifold, and {n_elec} electrons per cell do not fill \
             an integer number of doubly occupied bands"
        )));
    }
    if n_occ == 0 {
        return Err(Am1Error::InvalidInput(
            "no occupied bands: there is no polarization to couple a field to".into(),
        ));
    }
    Ok(n_occ)
}

/// Diagonalize `H(k) + ΔH(k)` at every k of the mesh and return the coefficients.
///
/// The Fock is rebuilt from the converged density rather than carried out of the SCF, exactly as
/// [`crate::pbc::berry`] does it — `PbcResult` keeps the real-space density and not the mesh's
/// coefficients, and one extra diagonalization per k per outer iteration is cheaper than changing
/// that.
fn coefficients_in_field(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    basis: &crate::basis::Basis,
    scf: &PbcResult,
    kpoints: &[KPoint],
    k_terms: &[CMatrix],
) -> Result<Vec<CMatrix>> {
    let cell = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a finite-field calculation needs a cell".into()))?;
    let neighbors = crate::neighbors::NeighborList::build(molecule, options.realspace_cutoff);
    let translations = cell.image_offsets(options.realspace_cutoff);
    let (core, pairs) = crate::pbc::scf::build_realspace_core(
        molecule,
        basis,
        params,
        &neighbors,
        &translations,
        options.exchange_cutoff,
        options.electric_field,
    )?;
    let fock = crate::pbc::scf::build_realspace_fock(
        &core,
        &pairs,
        scf.density.origin()?,
        &scf.density,
        0.5,
        basis,
        molecule,
        params,
        crate::pbc::hessian::long_range_delta(molecule, params, &neighbors, options)?.as_ref(),
    )?;

    let nao = basis.nao;
    let mut out = Vec::with_capacity(kpoints.len());
    for (idx, kp) in kpoints.iter().enumerate() {
        let mut hk = fock.bloch_sum(kp);
        let add = &k_terms[idx];
        for i in 0..nao {
            for j in 0..nao {
                let (r, m) = add.get(i, j);
                hk.re[(i, j)] += r;
                hk.im[(i, j)] += m;
            }
        }
        let eig = hermitian_eigen(&hk)?;
        out.push(CMatrix {
            n: nao,
            re: eig.vectors_re,
            im: eig.vectors_im,
        });
    }
    Ok(out)
}

/// Build the field operator at every k, and the Berry phases it was built from.
#[allow(clippy::too_many_arguments)]
fn field_operator(
    grid: &Grid,
    molecule: &Molecule,
    basis: &crate::basis::Basis,
    params: &Am1Parameters,
    cell: &Lattice,
    coefficients: &[CMatrix],
    n_occ: usize,
    field: Vec3,
) -> Result<(Vec<CMatrix>, [f64; 3])> {
    let nao = coefficients[0].n;
    let mut terms: Vec<CMatrix> = coefficients.iter().map(|_| CMatrix::zeros(nao)).collect();
    let mut phase = [0.0_f64; 3];

    for alpha in 0..3 {
        let a_alpha = cell.cell.col[alpha];
        let coupling = field.dot(a_alpha);
        let j_len = grid.n[alpha];
        // `b` is the step between neighbouring points of the string, so `Λ = diag(e^{−ib·τ_μ})`.
        let b = cell.reciprocal_vectors_2pi()[alpha] / j_len as f64;
        // The same operator `pbc::berry` uses — block-diagonal by atom, carrying the on-site
        // `s`–`p` moment as well as the atom's phase. Built once per direction: it depends on `b`
        // and the geometry, not on the string or the k point.
        let link = crate::pbc::berry::LinkOperator::new(molecule, basis, params, b)?;

        // Every string along `alpha`: the two transverse indices label them.
        let (beta, gamma) = ((alpha + 1) % 3, (alpha + 2) % 3);
        let mut total_phase = 0.0;
        let mut n_strings = 0usize;
        for ib in 0..grid.n[beta] {
            for ig in 0..grid.n[gamma] {
                let mut idx = [0usize; 3];
                idx[beta] = ib;
                idx[gamma] = ig;

                // The string's points, and the link overlaps `S_j = C_j† Λ C_{j+1}`.
                let mut points = Vec::with_capacity(j_len);
                for j in 0..j_len {
                    let mut here = idx;
                    here[alpha] = j;
                    points.push(grid.index(here[0], here[1], here[2]));
                }
                let mut links: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(j_len);
                let mut product = [1.0_f64, 0.0];
                for j in 0..j_len {
                    let next = (j + 1) % j_len;
                    let s =
                        link.sandwich(&coefficients[points[j]], &coefficients[points[next]], n_occ);
                    let mut re = s.0.clone();
                    let mut im = s.1.clone();
                    let det = crate::pbc::berry::complex_determinant(&mut re, &mut im, n_occ);
                    product = [
                        product[0] * det[0] - product[1] * det[1],
                        product[0] * det[1] + product[1] * det[0],
                    ];
                    let m = (product[0] * product[0] + product[1] * product[1]).sqrt();
                    if m > 1.0e-300 {
                        product[0] /= m;
                        product[1] /= m;
                    }
                    links.push(s);
                }
                total_phase += product[1].atan2(product[0]) / std::f64::consts::TAU;
                n_strings += 1;

                if coupling.abs() < 1.0e-14 {
                    continue;
                }
                // `λ = (𝓔·a_α) J/(4π)`; see this module's header for where it comes from.
                let lambda = coupling * j_len as f64 / (4.0 * std::f64::consts::PI);
                for j in 0..j_len {
                    let here = points[j];
                    let fwd = grid.step(
                        {
                            let mut t = idx;
                            t[alpha] = j;
                            t
                        },
                        alpha,
                        true,
                    );
                    let back = grid.step(
                        {
                            let mut t = idx;
                            t[alpha] = j;
                            t
                        },
                        alpha,
                        false,
                    );
                    let prev = (j + j_len - 1) % j_len;

                    // `W₊ = Λ C_{j+1} S_j⁻¹` and `W₋ = Λ* C_{j−1} (S_{j−1}⁻¹)†`.
                    let w_plus = weight_block(&link, &coefficients[fwd], &links[j], n_occ, false)?;
                    let w_minus =
                        weight_block(&link, &coefficients[back], &links[prev], n_occ, true)?;

                    accumulate_field_term(
                        &mut terms[here],
                        &coefficients[here],
                        &w_plus,
                        &w_minus,
                        n_occ,
                        lambda,
                    );
                }
            }
        }
        phase[alpha] = total_phase / n_strings.max(1) as f64;
    }
    Ok((terms, phase))
}

/// `S_{mn} = Σ_μ conj(c_{μm}(k)) e^{−ib·τ_μ} c_{μn}(k')`, row-major `n_occ × n_occ`.
/// `Λ^{(†)} C X`, with `X` either `S⁻¹` or `(S⁻¹)†`.
///
/// `backward` selects the `W₋` case, which takes both the adjoint of `Λ` and the adjoint of the
/// inverse — the pair comes from differentiating `conj(ln Z)`, so they travel together and are one
/// flag rather than two.
fn weight_block(
    link: &crate::pbc::berry::LinkOperator,
    c: &CMatrix,
    s: &(Vec<f64>, Vec<f64>),
    n_occ: usize,
    backward: bool,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let nao = c.n;
    let (mut inv_re, mut inv_im) = complex_inverse(&s.0, &s.1, n_occ)?;
    if backward {
        let (mut ar, mut ai) = (vec![0.0; n_occ * n_occ], vec![0.0; n_occ * n_occ]);
        for i in 0..n_occ {
            for j in 0..n_occ {
                ar[i * n_occ + j] = inv_re[j * n_occ + i];
                ai[i * n_occ + j] = -inv_im[j * n_occ + i];
            }
        }
        inv_re = ar;
        inv_im = ai;
    }
    // `C X` first, `nao × n_occ`, then `Λ` (or `Λ†`) on the left.
    let mut cx = CMatrix::zeros(nao);
    for mu in 0..nao {
        for m in 0..n_occ {
            let (mut ar, mut ai) = (0.0, 0.0);
            for n in 0..n_occ {
                let (cr, ci) = (c.re[(mu, n)], c.im[(mu, n)]);
                let (xr, xi) = (inv_re[n * n_occ + m], inv_im[n * n_occ + m]);
                ar += cr * xr - ci * xi;
                ai += cr * xi + ci * xr;
            }
            cx.re[(mu, m)] = ar;
            cx.im[(mu, m)] = ai;
        }
    }
    Ok(link.apply_columns(&cx, n_occ, backward))
}

fn accumulate_field_term(
    out: &mut CMatrix,
    c: &CMatrix,
    w_plus: &(Vec<f64>, Vec<f64>),
    w_minus: &(Vec<f64>, Vec<f64>),
    n_occ: usize,
    lambda: f64,
) {
    let nao = c.n;
    // `D = i λ (W₊ − W₋)`, `nao × n_occ`.
    let mut d_re = vec![0.0; nao * n_occ];
    let mut d_im = vec![0.0; nao * n_occ];
    for mu in 0..nao {
        for m in 0..n_occ {
            let dr = w_plus.0[mu * n_occ + m] - w_minus.0[mu * n_occ + m];
            let di = w_plus.1[mu * n_occ + m] - w_minus.1[mu * n_occ + m];
            // `i λ (dr + i di) = λ(−di + i dr)`
            d_re[mu * n_occ + m] = -lambda * di;
            d_im[mu * n_occ + m] = lambda * dr;
        }
    }

    // `X = C† D`, then `Y = X + X†` — Hermitian, `n_occ × n_occ`.
    let mut y_re = vec![0.0; n_occ * n_occ];
    let mut y_im = vec![0.0; n_occ * n_occ];
    {
        let mut x_re = vec![0.0; n_occ * n_occ];
        let mut x_im = vec![0.0; n_occ * n_occ];
        for m in 0..n_occ {
            for n in 0..n_occ {
                let (mut ar, mut ai) = (0.0, 0.0);
                for mu in 0..nao {
                    let (cr, ci) = (c.re[(mu, m)], -c.im[(mu, m)]);
                    let (dr, di) = (d_re[mu * n_occ + n], d_im[mu * n_occ + n]);
                    ar += cr * dr - ci * di;
                    ai += cr * di + ci * dr;
                }
                x_re[m * n_occ + n] = ar;
                x_im[m * n_occ + n] = ai;
            }
        }
        for m in 0..n_occ {
            for n in 0..n_occ {
                y_re[m * n_occ + n] = x_re[m * n_occ + n] + x_re[n * n_occ + m];
                y_im[m * n_occ + n] = x_im[m * n_occ + n] - x_im[n * n_occ + m];
            }
        }
    }

    // `C Y` once, `nao × n_occ`, so the last product costs `nao² n_occ` and not `nao² n_occ²`.
    let mut cy_re = vec![0.0; nao * n_occ];
    let mut cy_im = vec![0.0; nao * n_occ];
    for mu in 0..nao {
        for n in 0..n_occ {
            let (mut ar, mut ai) = (0.0, 0.0);
            for m in 0..n_occ {
                let (cr, ci) = (c.re[(mu, m)], c.im[(mu, m)]);
                let (yr, yi) = (y_re[m * n_occ + n], y_im[m * n_occ + n]);
                ar += cr * yr - ci * yi;
                ai += cr * yi + ci * yr;
            }
            cy_re[mu * n_occ + n] = ar;
            cy_im[mu * n_occ + n] = ai;
        }
    }

    for mu in 0..nao {
        for nu in 0..nao {
            let (mut ar, mut ai) = (0.0, 0.0);
            for m in 0..n_occ {
                // `D C†`
                let (dr, di) = (d_re[mu * n_occ + m], d_im[mu * n_occ + m]);
                let (cr, ci) = (c.re[(nu, m)], -c.im[(nu, m)]);
                ar += dr * cr - di * ci;
                ai += dr * ci + di * cr;

                // `C D†`
                let (er, ei) = (c.re[(mu, m)], c.im[(mu, m)]);
                let (fr, fi) = (d_re[nu * n_occ + m], -d_im[nu * n_occ + m]);
                ar += er * fr - ei * fi;
                ai += er * fi + ei * fr;

                // `− (C Y) C† / 2`
                let (gr, gi) = (cy_re[mu * n_occ + m], cy_im[mu * n_occ + m]);
                ar -= 0.5 * (gr * cr - gi * ci);
                ai -= 0.5 * (gr * ci + gi * cr);
            }
            out.re[(mu, nu)] += ar;
            out.im[(mu, nu)] += ai;
        }
    }
}

/// Inverse of a small complex matrix, by Gauss–Jordan with partial pivoting.
///
/// `n_occ` is the number of filled bands, so this is small and an elimination is the right tool.
fn complex_inverse(re: &[f64], im: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut a_re = re.to_vec();
    let mut a_im = im.to_vec();
    let mut b_re = vec![0.0; n * n];
    let mut b_im = vec![0.0; n * n];
    for i in 0..n {
        b_re[i * n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = a_re[col * n + col].hypot(a_im[col * n + col]);
        for row in (col + 1)..n {
            let m = a_re[row * n + col].hypot(a_im[row * n + col]);
            if m > best {
                best = m;
                pivot = row;
            }
        }
        if best < 1.0e-300 {
            return Err(Am1Error::InvalidInput(
                "a Berry-phase link overlap is singular: the occupied manifolds at neighbouring k \
                 points have become orthogonal, which means the band ordering changed across the \
                 link. A denser mesh or a smaller field is the fix."
                    .into(),
            ));
        }
        if pivot != col {
            for j in 0..n {
                a_re.swap(col * n + j, pivot * n + j);
                a_im.swap(col * n + j, pivot * n + j);
                b_re.swap(col * n + j, pivot * n + j);
                b_im.swap(col * n + j, pivot * n + j);
            }
        }
        let (pr, pi) = (a_re[col * n + col], a_im[col * n + col]);
        let d = pr * pr + pi * pi;
        for j in 0..n {
            let (xr, xi) = (a_re[col * n + j], a_im[col * n + j]);
            a_re[col * n + j] = (xr * pr + xi * pi) / d;
            a_im[col * n + j] = (xi * pr - xr * pi) / d;
            let (yr, yi) = (b_re[col * n + j], b_im[col * n + j]);
            b_re[col * n + j] = (yr * pr + yi * pi) / d;
            b_im[col * n + j] = (yi * pr - yr * pi) / d;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let (fr, fi) = (a_re[row * n + col], a_im[row * n + col]);
            if fr == 0.0 && fi == 0.0 {
                continue;
            }
            for j in 0..n {
                let (xr, xi) = (a_re[col * n + j], a_im[col * n + j]);
                a_re[row * n + j] -= fr * xr - fi * xi;
                a_im[row * n + j] -= fr * xi + fi * xr;
                let (yr, yi) = (b_re[col * n + j], b_im[col * n + j]);
                b_re[row * n + j] -= fr * yr - fi * yi;
                b_im[row * n + j] -= fr * yi + fi * yr;
            }
        }
    }
    Ok((b_re, b_im))
}

#[cfg(test)]
mod field_component_tests {
    use super::*;

    /// `A A⁻¹ = I` for a complex matrix with no particular structure.
    ///
    /// The field operator is `Λ C S⁻¹`, so a wrong inverse is a wrong gradient and the SCF simply
    /// converges to the wrong state — there is no assertion inside the loop that would notice.
    /// Both orders are checked, because a Gauss–Jordan that pivots correctly on one side and not
    /// the other gives a left inverse that is not a right one.
    #[test]
    fn the_complex_inverse_is_a_two_sided_inverse() {
        let n = 4;
        let mut re = vec![0.0; n * n];
        let mut im = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                re[i * n + j] =
                    ((i * 5 + j * 3) as f64 * 0.7).sin() + if i == j { 2.0 } else { 0.0 };
                im[i * n + j] = ((i * 2 + j * 9) as f64 * 0.4).cos();
            }
        }
        let (inv_re, inv_im) = complex_inverse(&re, &im, n).unwrap();

        for (label, left, right) in [
            ("A A⁻¹", (&re, &im), (&inv_re, &inv_im)),
            ("A⁻¹ A", (&inv_re, &inv_im), (&re, &im)),
        ] {
            let mut worst = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let (mut pr, mut pi) = (0.0, 0.0);
                    for k in 0..n {
                        let (ar, ai) = (left.0[i * n + k], left.1[i * n + k]);
                        let (br, bi) = (right.0[k * n + j], right.1[k * n + j]);
                        pr += ar * br - ai * bi;
                        pi += ar * bi + ai * br;
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    worst = worst.max((pr - want).abs()).max(pi.abs());
                }
            }
            assert!(worst < 1.0e-12, "{label} − I is {worst:.3e}");
        }
    }

    /// A singular link overlap is an error naming what went wrong, not a division by zero that
    /// propagates `NaN` into the Hamiltonian.
    ///
    /// It happens for a real reason — the occupied manifolds at neighbouring k points becoming
    /// orthogonal, which means the band ordering changed across the link — and a caller who sees
    /// `NaN` frequencies has no way to guess that.
    #[test]
    fn a_singular_overlap_is_refused_by_name() {
        let n = 2;
        // Rank one: the second row is twice the first.
        let re = vec![1.0, 2.0, 2.0, 4.0];
        let im = vec![0.0, 0.0, 0.0, 0.0];
        let err = complex_inverse(&re, &im, n).unwrap_err().to_string();
        assert!(
            err.contains("singular"),
            "expected a message naming the singularity, got: {err}"
        );
    }

    /// The grid's neighbour arithmetic wraps the zone, and `k + b` really is one step along.
    ///
    /// The Berry string is a closed loop through the whole zone; an off-by-one in the wrap makes it
    /// a loop through all but one link, which still produces a plausible phase.
    #[test]
    fn the_grid_steps_wrap_the_zone() {
        let grid = Grid { n: [4, 3, 2] };
        let points = grid.kpoints();
        assert_eq!(points.len(), 4 * 3 * 2);
        let total: f64 = points.iter().map(|k| k.weight).sum();
        assert!(
            (total - 1.0).abs() < 1.0e-15,
            "the k weights sum to {total}"
        );

        for i in 0..4 {
            for j in 0..3 {
                for l in 0..2 {
                    let here = grid.index(i, j, l);
                    for (alpha, n) in [(0usize, 4usize), (1, 3), (2, 2)] {
                        let fwd = grid.step([i, j, l], alpha, true);
                        let back = grid.step([i, j, l], alpha, false);
                        // Stepping forward then back returns, and `n` steps forward is a full loop.
                        assert_eq!(grid.step(index_of(&grid, fwd), alpha, false), here);
                        let mut walk = [i, j, l];
                        for _ in 0..n {
                            walk = index_of(&grid, grid.step(walk, alpha, true));
                        }
                        assert_eq!(grid.index(walk[0], walk[1], walk[2]), here);
                        // And forward is genuinely one step in the fractional coordinate.
                        let d = points[fwd].fractional[alpha] - points[here].fractional[alpha];
                        let step = 1.0 / n as f64;
                        assert!(
                            (d - step).abs() < 1.0e-12 || (d + 1.0 - step).abs() < 1.0e-12,
                            "a forward step along {alpha} moved {d}, not {step}"
                        );
                        let _ = back;
                    }
                }
            }
        }
    }

    /// Invert `Grid::index` for the walk above.
    fn index_of(grid: &Grid, flat: usize) -> [usize; 3] {
        let l = flat % grid.n[2];
        let j = (flat / grid.n[2]) % grid.n[1];
        let i = flat / (grid.n[1] * grid.n[2]);
        [i, j, l]
    }
}
