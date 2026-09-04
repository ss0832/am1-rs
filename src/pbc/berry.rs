// SPDX-License-Identifier: GPL-3.0-or-later

//! **Berry-phase electronic polarization** — the modern theory of polarization, King-Smith and
//! Vanderbilt.
//!
//! # Why polarization is not `Σ q r`
//!
//! Under periodic boundary conditions the dipole of a cell is not a property of the crystal. Move
//! the cell boundary and `Σ_a Q_a R_a` changes; there is no "the" charge distribution to take a
//! first moment of, because the electrons are delocalized across cells. [`crate::dipole`] says as
//! much and offers its operator for molecules only.
//!
//! What *is* well defined is the **change** in polarization along an adiabatic path, and the
//! quantity whose changes those are is a Berry phase of the occupied Bloch states:
//!
//! ```text
//! P_el,α = −(e/Ω) (1/2π) · a_α · (1/N_⊥) Σ_{k_⊥} Im ln Π_j det S(k_j, k_{j+1})
//! ```
//!
//! where the product runs along a **string** of k-points spanning the Brillouin zone in direction
//! `α`, and `S` is the overlap of the occupied manifolds at neighbouring points. The result is
//! defined only **modulo the quantum** `e a_α / Ω`: a different branch of the logarithm is a
//! different — equally valid — choice of which unit cell the electrons are assigned to. That
//! ambiguity is the physics, not a defect, and [`BerryPolarization::quantum`] reports it so a
//! caller can reduce two values to the same branch before subtracting them.
//!
//! # The overlap in an NDDO basis
//!
//! `S_mn(k, k+b) = ⟨u_mk | u_n,k+b⟩` needs the cell-periodic parts. This crate's k-point
//! Hamiltonian is built as `H(k) = Σ_T e^{ik·T} H(0,T)` — the phase on the lattice translation
//! only — so the coefficients are in the "cell" gauge and
//!
//! ```text
//! S_mn(k, k+b) = Σ_μ c*_{μm}(k) e^{−i b·τ_μ} c_{μn}(k+b)
//! ```
//!
//! with `τ_μ` the position of the atom carrying orbital `μ`. The `e^{−ib·τ_μ}` factor is where the
//! atomic positions enter, and it is **the same approximation the dipole operator already makes**:
//! NDDO assumes an orthonormal AO basis and treats each orbital as sitting at its atom, which is
//! why [`crate::dipole::dipole_operator`] puts `R_a` on the diagonal of atom `a`'s block. Using
//! anything else here would make the Berry phase and the dipole disagree about where an orbital is.
//!
//! What this drops is the intra-atomic `s`–`p` hybridization moment `dd_a`, which the dipole
//! operator does carry. It is a genuine difference and it is measured rather than argued away:
//! `tests/pbc_berry.rs` compares `Ω ∂P/∂τ` against the Born charges the CPHF route produces, which
//! *do* include it.
//!
//! # The last link of the string
//!
//! `k_J = k_0 + b` is the same state as `k_0` in a different gauge: `|u_{k+b}⟩ = e^{−ib·r}|u_k⟩`.
//! Rather than special-case it, the string is closed by applying that phase to the `k_0`
//! coefficients — which in this basis is exactly the `e^{−ib·τ_μ}` already in `S`. So the final
//! overlap is `S(k_{J−1}, k_0)` with one extra factor of the same form, and the product is
//! automatically gauge invariant: any phase a diagonalizer puts on an eigenvector at an interior
//! point appears once as `c*` and once as `c` and cancels.

use crate::basis::Basis;
use crate::error::{Am1Error, Result};
use crate::lattice::Lattice;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::neighbors::NeighborList;
use crate::params::Am1Parameters;
use crate::pbc::complex::{hermitian_eigen, CMatrix};
use crate::pbc::kpoints::KPoint;
use crate::pbc::scf::{run_pbc_scf, PbcOptions};
use crate::system::Molecule;

/// The polarization of a periodic cell, in `e·Bohr / Bohr³` (atomic units).
#[derive(Clone, Debug)]
pub struct BerryPolarization {
    /// Electronic contribution, from the Berry phase of the occupied manifold.
    pub electronic: Vec3,
    /// Ionic contribution, `(1/Ω) Σ_A Z_A τ_A` over the core charges.
    pub ionic: Vec3,
    /// Their sum. Defined **modulo** [`Self::quantum`].
    pub total: Vec3,
    /// The Berry phase per direction, in units of `2π`. This is the raw, branch-dependent number.
    pub phase: [f64; 3],
    /// The polarization quantum along each lattice vector, `e a_α / Ω`. Two polarizations are
    /// physically the same if they differ by an integer combination of these.
    pub quantum: [Vec3; 3],
    /// How many k points each string used.
    pub string_length: usize,
}

impl BerryPolarization {
    /// `other − self`, reduced to the branch nearest zero along each lattice direction.
    ///
    /// The only physically meaningful thing to do with two polarizations. Subtracting the `total`
    /// fields directly is wrong whenever the two happened to land on different branches, which for
    /// a finite displacement is common and gives an answer off by exactly one quantum.
    pub fn difference(&self, other: &Self) -> Vec3 {
        let mut delta = other.total - self.total;
        // Reduce along each lattice vector in turn. The quanta are the lattice vectors scaled by
        // `e/Ω`, so this is a lattice reduction of the difference.
        for q in &self.quantum {
            let n2 = q.norm2();
            if n2 < 1.0e-30 {
                continue;
            }
            let n = (delta.dot(*q) / n2).round();
            delta -= *q * n;
        }
        delta
    }
}

/// Compute the Berry-phase polarization of a periodic cell.
///
/// `strings` is the number of k points along each Brillouin-zone string — the convergence
/// parameter, and the one the answer must become independent of. The transverse sampling is taken
/// from `options.kmesh`.
///
/// Restricted (closed-shell) only: the phase is defined on a filled manifold, and a partially
/// occupied one has no gap to make it adiabatic.
pub fn berry_polarization(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    strings: usize,
) -> Result<BerryPolarization> {
    let cell = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a Berry phase needs a periodic cell".into()))?;
    if !cell.is_fully_periodic() {
        return Err(Am1Error::InvalidInput(
            "the Berry-phase polarization is defined for a three-dimensional cell: the quantum is \
             `e a/Ω` and Ω has to be a volume. A slab or a chain has a polarization along its \
             periodic directions only, which this does not yet separate out."
                .into(),
        ));
    }
    if strings < 3 {
        return Err(Am1Error::InvalidInput(
            "a Berry-phase string needs at least 3 k points; the discretized phase is a product of \
             nearest-neighbour overlaps and two points cannot resolve a winding"
                .into(),
        ));
    }

    let scf = run_pbc_scf(molecule, params, options)?;
    if !scf.converged {
        return Err(Am1Error::InvalidInput(
            "the periodic SCF did not converge; a Berry phase built on it would be meaningless"
                .into(),
        ));
    }
    if scf.unrestricted {
        return Err(Am1Error::InvalidInput(
            "the Berry-phase polarization is restricted-only; an open-shell cell would need the \
             phase of each spin manifold separately"
                .into(),
        ));
    }

    let basis = Basis::build(molecule, params)?;
    let neighbors = NeighborList::build(molecule, options.realspace_cutoff);
    let translations = cell.image_offsets(options.realspace_cutoff);
    let (core, pairs) = crate::pbc::scf::build_realspace_core(
        molecule,
        &basis,
        params,
        &neighbors,
        &translations,
        options.exchange_cutoff,
        options.electric_field,
    )?;
    let fock = crate::pbc::scf::build_realspace_fock(
        &core,
        &pairs,
        scf.density
            .get(crate::lattice::ImageOffset::origin())
            .ok_or_else(|| Am1Error::InvalidInput("density is missing the origin block".into()))?,
        &scf.density,
        0.5,
        &basis,
        molecule,
        params,
        crate::pbc::hessian::long_range_delta(molecule, params, &neighbors, options)?.as_ref(),
    )?;

    // Electrons per cell, from the same expression the SCF uses.
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
            "no occupied bands: there is no electronic polarization to compute".into(),
        ));
    }

    // Transverse sampling: the requested mesh, used for the two directions orthogonal to the
    // string. A string's own direction is resampled at `strings` points.
    let mesh = options.kmesh.sizes();
    let mut phase = [0.0_f64; 3];
    for alpha in 0..3 {
        let (beta, gamma) = ((alpha + 1) % 3, (alpha + 2) % 3);
        let (nb, ng) = (mesh[beta].max(1), mesh[gamma].max(1));
        let mut total = 0.0;
        for ib in 0..nb {
            for ig in 0..ng {
                let mut base = [0.0_f64; 3];
                base[beta] = ib as f64 / nb as f64;
                base[gamma] = ig as f64 / ng as f64;
                total += string_phase(
                    &fock, molecule, &basis, params, &cell, alpha, base, strings, n_occ,
                )?;
            }
        }
        phase[alpha] = total / (nb * ng) as f64;
    }

    // `P_el = +(e/Ω) Σ_α φ_α a_α`, with `φ` in turns and the factor 2 for the spin pair.
    //
    // # Where the sign comes from
    //
    // Fixed by the one case that can be worked out by hand rather than looked up, since sign
    // conventions for the Berry phase differ between sources. Take a single electron whose only
    // orbital sits at `τ`: then `c = 1` at every `k`, every link contributes `S = e^{−ib·τ}`, and
    // the product over the `J` links of the string is `e^{−iB·τ}` with `B = Jb` the full
    // reciprocal vector. So `φ = −B·τ/2π = −τ_α/a_α` in turns.
    //
    // That electron is a charge `−1` at `τ`, so its polarization is `−τ/Ω`. Matching:
    // `C·a_α·(−τ_α/a_α) = −τ_α/Ω` gives `C = +1/Ω`.
    //
    // The first draft had this negative, and the acoustic sum rule found it immediately: the Born
    // charges summed to `+2 n_elec` instead of zero, the electronic term carrying the same sign as
    // the ionic one instead of cancelling it.
    let volume = cell.volume();
    let mut electronic = Vec3::zero();
    for alpha in 0..3 {
        electronic += cell.cell.col[alpha] * (2.0 * phase[alpha] / volume);
    }

    let mut ionic = Vec3::zero();
    for atom in &molecule.atoms {
        ionic += atom.position * (params.element(atom.z)?.core_charge / volume);
    }

    let quantum = [
        cell.cell.col[0] / volume,
        cell.cell.col[1] / volume,
        cell.cell.col[2] / volume,
    ];
    Ok(BerryPolarization {
        electronic,
        ionic,
        total: electronic + ionic,
        phase,
        quantum,
        string_length: strings,
    })
}

/// The Bloch link operator `Λ_{μν} = ⟨χ_μ| e^{−i b·r} |χ_ν⟩`, in the NDDO AO basis.
///
/// # Why this is not simply a phase
///
/// Through 0.2.2 this was the diagonal `e^{−i b·τ_μ}`: each orbital treated as a point at its own
/// atom. That is the leading term, and it is what made the Berry phase track the charge
/// **centres** and nothing else — so it carried no on-site `s`–`p` moment, and disagreed with the
/// CPHF by 0.207 e on HF's Born charges and by 12 % on a water crystal's polarizability, while
/// agreeing to 7.5e-13 e on a hydrogen-only cell where the moment cannot exist.
///
/// The exact matrix element, with `r = τ_a + (r − τ_a)`, is
///
/// ```text
/// Λ_{μν} = e^{−i b·τ_a} ⟨χ_μ| e^{−i b·(r−τ_a)} |χ_ν⟩
/// ```
///
/// and NDDO's differential-overlap neglect leaves only the same-atom block. Expanding the second
/// factor to first order in the *dipole* gives `I − i b·D^a`, with `D^a_{μν} = ⟨χ_μ|(r−τ_a)|χ_ν⟩`
/// — which in a minimal `sp` basis is exactly `dd` on the `(s, p_α)` pairs and zero elsewhere.
/// That is the same `dd` [`crate::dipole::dipole_operator`] puts on those elements, taken from the
/// same parameter, so the two operators cannot drift apart.
///
/// # Exponentiated rather than truncated
///
/// `b·D^a = dd |b| (|s⟩⟨u| + |u⟩⟨s|)` with `|u⟩` the unit `p` combination along `b`, so its
/// exponential is a rotation in that two-dimensional subspace and is available in closed form:
///
/// ```text
/// Λ^a = e^{−i b·τ_a} [ cos θ on (s,s) and (u,u);  −i sin θ on (s,u) and (u,s);  1 elsewhere ]
/// θ = dd_a |b|
/// ```
///
/// Truncating at `I − i b·D` instead would leave `|det Λ| ≠ 1`, and the string's product of
/// determinants would drift in modulus rather than only in phase. `θ` is small — `|b| = 2π/(Ja)`
/// — so the two agree to `O(θ³)`, but the exponential costs nothing and keeps the operator
/// unitary, which is the property the phase is extracted under.
pub(crate) struct LinkOperator {
    re: Matrix,
    im: Matrix,
}

impl LinkOperator {
    /// Build `Λ` for the step `b` between neighbouring points of a string.
    pub(crate) fn new(
        molecule: &Molecule,
        basis: &Basis,
        params: &Am1Parameters,
        b: Vec3,
    ) -> Result<Self> {
        let nao = basis.nao;
        let mut re = Matrix::zeros(nao, nao);
        let mut im = Matrix::zeros(nao, nao);
        let b_norm = b.norm();
        // `b̂`, or anything unit when `b` is zero — `θ` is then zero and the direction is unused.
        let u = if b_norm > 1.0e-300 {
            b / b_norm
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };

        for (ia, atom) in molecule.atoms.iter().enumerate() {
            let elem = params.element(atom.z)?;
            let off = basis.atom_offset[ia];
            let norb = basis.atom_norb[ia];
            let (sin_phase, cos_phase) = (-b.dot(atom.position)).sin_cos();

            // The on-site rotation, in the AO basis of this atom. `norb == 1` (hydrogen) has no
            // p shell and therefore no on-site moment: the block is the identity.
            let theta = if norb == 4 { elem.dd * b_norm } else { 0.0 };
            let (sin_t, cos_t) = theta.sin_cos();
            let ub = [u.x, u.y, u.z];

            for i in 0..norb {
                for j in 0..norb {
                    // `[exp(−iθ(|s⟩⟨u|+|u⟩⟨s|))]_{ij}` written in the Cartesian p basis: the
                    // `(s,s)` element and the `u`-projection of the `p` block carry `cos θ`, the
                    // rest of the `p` block is untouched, and the `s`–`p` elements carry
                    // `−i sin θ` weighted by the direction cosine.
                    let (mut br, mut bi) = (0.0, 0.0);
                    match (i, j) {
                        (0, 0) => br = cos_t,
                        (0, _) => bi = -sin_t * ub[j - 1],
                        (_, 0) => bi = -sin_t * ub[i - 1],
                        _ => {
                            let (pi_, pj) = (ub[i - 1], ub[j - 1]);
                            // `δ_ij + (cos θ − 1) û_i û_j`: the identity outside the `u`
                            // direction, `cos θ` along it.
                            br = if i == j { 1.0 } else { 0.0 } + (cos_t - 1.0) * pi_ * pj;
                        }
                    }
                    // Times the atom's own phase.
                    re[(off + i, off + j)] = cos_phase * br - sin_phase * bi;
                    im[(off + i, off + j)] = cos_phase * bi + sin_phase * br;
                }
            }
        }
        Ok(Self { re, im })
    }

    /// `S = A† Λ B`, over the first `n` columns of each — row-major `n × n`.
    pub(crate) fn sandwich(&self, a: &CMatrix, b: &CMatrix, n: usize) -> (Vec<f64>, Vec<f64>) {
        let (t_re, t_im) = self.apply_columns(b, n, false);
        let nao = a.n;
        let mut re = vec![0.0; n * n];
        let mut im = vec![0.0; n * n];
        for m in 0..n {
            for p in 0..n {
                let (mut ar, mut ai) = (0.0, 0.0);
                for mu in 0..nao {
                    // conj(a_{μm})
                    let (cr, ci) = (a.re[(mu, m)], -a.im[(mu, m)]);
                    let (tr, ti) = (t_re[mu * n + p], t_im[mu * n + p]);
                    ar += cr * tr - ci * ti;
                    ai += cr * ti + ci * tr;
                }
                re[m * n + p] = ar;
                im[m * n + p] = ai;
            }
        }
        (re, im)
    }

    /// `Λ X` (or `Λ† X` when `adjoint`), over the first `n` columns of `X` — row-major `nao × n`.
    pub(crate) fn apply_columns(
        &self,
        x: &CMatrix,
        n: usize,
        adjoint: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let nao = x.n;
        let mut re = vec![0.0; nao * n];
        let mut im = vec![0.0; nao * n];
        for mu in 0..nao {
            for col in 0..n {
                let (mut ar, mut ai) = (0.0, 0.0);
                for nu in 0..nao {
                    // `Λ†_{μν} = conj(Λ_{νμ})`.
                    let (lr, li) = if adjoint {
                        (self.re[(nu, mu)], -self.im[(nu, mu)])
                    } else {
                        (self.re[(mu, nu)], self.im[(mu, nu)])
                    };
                    if lr == 0.0 && li == 0.0 {
                        continue;
                    }
                    let (xr, xi) = (x.re[(nu, col)], x.im[(nu, col)]);
                    ar += lr * xr - li * xi;
                    ai += lr * xi + li * xr;
                }
                re[mu * n + col] = ar;
                im[mu * n + col] = ai;
            }
        }
        (re, im)
    }
}

/// The discretized Berry phase of one string, in units of `2π`.
///
/// `Im ln Π_j det S(k_j, k_{j+1})`, accumulated as a running complex product so the branch of the
/// logarithm is taken once at the end rather than per link — taking it per link is the classic way
/// to lose a `2π` when a determinant crosses the negative real axis.
#[allow(clippy::too_many_arguments)]
fn string_phase(
    fock: &crate::pbc::scf::RealSpaceBlocks,
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    cell: &Lattice,
    alpha: usize,
    base: [f64; 3],
    strings: usize,
    n_occ: usize,
) -> Result<f64> {
    let nao = basis.nao;
    // Occupied coefficients at each point of the string.
    let mut occupied: Vec<CMatrix> = Vec::with_capacity(strings);
    for j in 0..strings {
        let mut frac = base;
        frac[alpha] = j as f64 / strings as f64;
        let k = KPoint {
            fractional: frac,
            weight: 1.0,
        };
        let eig = hermitian_eigen(&fock.bloch_sum(&k))?;
        // `hermitian_eigen` returns ascending eigenvalues, so the lowest `n_occ` are the filled
        // ones — which is the aufbau filling a gapped, closed-shell cell has by construction.
        occupied.push(CMatrix {
            n: nao,
            re: eig.vectors_re,
            im: eig.vectors_im,
        });
    }

    // `b` is the step between neighbouring points of the string.
    let b = cell.reciprocal_vectors_2pi()[alpha] / strings as f64;
    let link = LinkOperator::new(molecule, basis, params, b)?;

    let mut product = [1.0_f64, 0.0];
    for j in 0..strings {
        let next = (j + 1) % strings;
        // On the wrap-around link the two points are the same state in different gauges, and `Λ`
        // is exactly the gauge transformation that relates them — so the link needs no special
        // case, only the full reciprocal vector's worth of phase.
        let (mut s_re, mut s_im) = link.sandwich(&occupied[j], &occupied[next], n_occ);
        let det = complex_determinant(&mut s_re, &mut s_im, n_occ);
        product = [
            product[0] * det[0] - product[1] * det[1],
            product[0] * det[1] + product[1] * det[0],
        ];
        // Renormalize: `n_occ` determinants of modulus < 1 underflow for a long string, and only
        // the argument matters.
        let m = (product[0] * product[0] + product[1] * product[1]).sqrt();
        if m > 1.0e-300 {
            product[0] /= m;
            product[1] /= m;
        }
    }
    // `atan2` in units of 2π.
    Ok(product[1].atan2(product[0]) / std::f64::consts::TAU)
}
/// Determinant of a complex matrix by Gaussian elimination with partial pivoting.
///
/// Operates in place on the row-major `(re, im)` pair. Small — `n_occ` is the number of filled
/// bands — so an elimination is the right tool and there is no reason to reach for a factorization.
pub(crate) fn complex_determinant(re: &mut [f64], im: &mut [f64], n: usize) -> [f64; 2] {
    let mut det = [1.0_f64, 0.0];
    for col in 0..n {
        // Partial pivot on modulus.
        let mut pivot = col;
        let mut best = re[col * n + col].hypot(im[col * n + col]);
        for row in (col + 1)..n {
            let m = re[row * n + col].hypot(im[row * n + col]);
            if m > best {
                best = m;
                pivot = row;
            }
        }
        if best < 1.0e-300 {
            return [0.0, 0.0];
        }
        if pivot != col {
            for c in 0..n {
                re.swap(col * n + c, pivot * n + c);
                im.swap(col * n + c, pivot * n + c);
            }
            det = [-det[0], -det[1]];
        }
        let (pr, pi) = (re[col * n + col], im[col * n + col]);
        det = [det[0] * pr - det[1] * pi, det[0] * pi + det[1] * pr];
        let inv = 1.0 / (pr * pr + pi * pi);
        for row in (col + 1)..n {
            let (ar, ai) = (re[row * n + col], im[row * n + col]);
            // factor = a / p
            let fr = (ar * pr + ai * pi) * inv;
            let fi = (ai * pr - ar * pi) * inv;
            for c in col..n {
                let (br, bi) = (re[col * n + c], im[col * n + c]);
                re[row * n + c] -= fr * br - fi * bi;
                im[row * n + c] -= fr * bi + fi * br;
            }
        }
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::complex::CMatrix;

    /// Water: an atom with a `p` shell (so a nonzero on-site moment) and two without.
    fn water() -> Molecule {
        let ang = 1.0 / crate::constants::AM1_A0;
        Molecule::new(vec![
            crate::system::Atom {
                z: 8,
                position: Vec3::zero(),
            },
            crate::system::Atom {
                z: 1,
                position: Vec3::new(0.9614, 0.0, 0.0) * ang,
            },
            crate::system::Atom {
                z: 1,
                position: Vec3::new(-0.2246, 0.9348, 0.0) * ang,
            },
        ])
    }

    fn link_for(molecule: &Molecule, b: Vec3) -> (LinkOperator, usize) {
        let params = Am1Parameters::standard().unwrap();
        let basis = Basis::build(molecule, &params).unwrap();
        let nao = basis.nao;
        (
            LinkOperator::new(molecule, &basis, &params, b).unwrap(),
            nao,
        )
    }

    /// `Λ† Λ = I`, to machine precision.
    ///
    /// This is the property the *exponential* form buys and the truncation `I − i b·D` does not.
    /// It matters because the string's phase is read off a product of determinants: a `Λ` with
    /// `|det Λ| ≠ 1` makes that product drift in modulus as well as in argument, and the
    /// renormalization that keeps it from underflowing would then be hiding a real error rather
    /// than a representational one.
    #[test]
    fn the_link_operator_is_unitary() {
        let molecule = water();
        // A step big enough that `θ = dd|b|` is not numerically zero — a real string's `b` is
        // smaller, so this is the harder case.
        let (link, nao) = link_for(&molecule, Vec3::new(0.3, -0.15, 0.05));
        let mut worst = 0.0_f64;
        for i in 0..nao {
            for j in 0..nao {
                // `(Λ†Λ)_{ij} = Σ_μ conj(Λ_{μi}) Λ_{μj}`
                let (mut re, mut im) = (0.0, 0.0);
                for mu in 0..nao {
                    let (ar, ai) = (link.re[(mu, i)], -link.im[(mu, i)]);
                    let (br, bi) = (link.re[(mu, j)], link.im[(mu, j)]);
                    re += ar * br - ai * bi;
                    im += ar * bi + ai * br;
                }
                let want = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((re - want).abs()).max(im.abs());
            }
        }
        assert!(worst < 1.0e-14, "Λ†Λ − I is {worst:.3e}");
    }

    /// With no `p` orbital there is no on-site moment, and `Λ` collapses to the diagonal phase
    /// the pre-0.2.2 code used.
    ///
    /// The two implementations have to agree exactly where the extra term cannot exist — that is
    /// what says the generalization did not change the case it was already right about, and it is
    /// the same argument `tests/pbc_berry.rs` makes on a hydrogen-only cell's Born charges.
    #[test]
    fn without_p_orbitals_the_link_operator_is_the_bare_phase() {
        let ang = 1.0 / crate::constants::AM1_A0;
        let h2 = Molecule::new(vec![
            crate::system::Atom {
                z: 1,
                position: Vec3::zero(),
            },
            crate::system::Atom {
                z: 1,
                position: Vec3::new(0.74, 0.1, -0.2) * ang,
            },
        ]);
        let b = Vec3::new(0.2, 0.1, -0.05);
        let (link, nao) = link_for(&h2, b);
        for mu in 0..nao {
            for nu in 0..nao {
                let (want_re, want_im) = if mu == nu {
                    let (s, c) = (-b.dot(h2.atoms[mu].position)).sin_cos();
                    (c, s)
                } else {
                    (0.0, 0.0)
                };
                assert!(
                    (link.re[(mu, nu)] - want_re).abs() < 1.0e-15
                        && (link.im[(mu, nu)] - want_im).abs() < 1.0e-15,
                    "Λ[{mu},{nu}] = {} + {}i, expected {want_re} + {want_im}i",
                    link.re[(mu, nu)],
                    link.im[(mu, nu)]
                );
            }
        }
    }

    /// `Λ → I` as `b → 0`, and its first-order departure is `−i b·D` with `D` the **same** on-site
    /// dipole `crate::dipole::dipole_operator` carries.
    ///
    /// This is the check that ties the Berry phase's position operator to the CPHF's. They are two
    /// expressions of one physical quantity, they are built in different files from the same `dd`,
    /// and if they ever drift apart the symptom is a polarizability that disagrees with the
    /// finite-field one by a few percent — which is exactly what the diagonal-`Λ` version did.
    #[test]
    fn the_link_operator_differentiates_into_the_dipole_operator() {
        let molecule = water();
        let params = Am1Parameters::standard().unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let nao = basis.nao;
        let m = crate::dipole::dipole_operator(&molecule, &basis, &params).unwrap();

        // `Λ(b) = I − i b·M + O(b²)` with `M` the **full** dipole operator: the atom's position on
        // the diagonal is the `e^{−ib·τ}` phase to first order, and the on-site `dd` is the block.
        let h = 1.0e-6;
        for axis in 0..3 {
            let mut b = Vec3::zero();
            match axis {
                0 => b.x = h,
                1 => b.y = h,
                _ => b.z = h,
            }
            let (link, _) = link_for(&molecule, b);
            let mut worst = 0.0_f64;
            for mu in 0..nao {
                for nu in 0..nao {
                    let want_re = if mu == nu { 1.0 } else { 0.0 };
                    let want_im = -h * m[axis][(mu, nu)];
                    worst = worst
                        .max((link.re[(mu, nu)] - want_re).abs())
                        .max((link.im[(mu, nu)] - want_im).abs());
                }
            }
            // Second order in `b·M`; `M` has entries of order a few Bohr, so `h² |M|²` is ~1e-11.
            assert!(
                worst < 1.0e-10,
                "axis {axis}: Λ(b) − (I − i b·M) is {worst:.3e}, so the Berry phase's position \
                 operator is not the dipole operator"
            );
        }
    }

    /// `sandwich` is `A† Λ B`, and `apply_columns` with `adjoint` is `Λ† X`. Checked against the
    /// definitions written out by hand, because both are hand-rolled complex products where a
    /// conjugation is one sign away from being wrong and nothing downstream would say so.
    #[test]
    fn the_link_products_are_what_they_say() {
        let molecule = water();
        let (link, nao) = link_for(&molecule, Vec3::new(0.11, -0.07, 0.03));
        // Two arbitrary but reproducible complex coefficient blocks.
        let fill = |seed: f64| {
            let mut c = CMatrix::zeros(nao);
            for i in 0..nao {
                for j in 0..nao {
                    c.re[(i, j)] = ((i * 7 + j * 3) as f64 * seed).sin();
                    c.im[(i, j)] = ((i * 5 + j * 11) as f64 * seed).cos();
                }
            }
            c
        };
        let (a, b) = (fill(0.31), fill(0.17));
        let n = 3.min(nao);

        let (s_re, s_im) = link.sandwich(&a, &b, n);
        for m in 0..n {
            for p in 0..n {
                let (mut re, mut im) = (0.0, 0.0);
                for mu in 0..nao {
                    for nu in 0..nao {
                        // conj(a_{μm}) Λ_{μν} b_{νp}
                        let (ar, ai) = (a.re[(mu, m)], -a.im[(mu, m)]);
                        let (lr, li) = (link.re[(mu, nu)], link.im[(mu, nu)]);
                        let (br, bi) = (b.re[(nu, p)], b.im[(nu, p)]);
                        let (xr, xi) = (ar * lr - ai * li, ar * li + ai * lr);
                        re += xr * br - xi * bi;
                        im += xr * bi + xi * br;
                    }
                }
                assert!(
                    (s_re[m * n + p] - re).abs() < 1.0e-12
                        && (s_im[m * n + p] - im).abs() < 1.0e-12,
                    "sandwich[{m},{p}] disagrees with A†ΛB"
                );
            }
        }

        let (t_re, t_im) = link.apply_columns(&b, n, true);
        for mu in 0..nao {
            for col in 0..n {
                let (mut re, mut im) = (0.0, 0.0);
                for nu in 0..nao {
                    // `Λ†_{μν} = conj(Λ_{νμ})`
                    let (lr, li) = (link.re[(nu, mu)], -link.im[(nu, mu)]);
                    let (xr, xi) = (b.re[(nu, col)], b.im[(nu, col)]);
                    re += lr * xr - li * xi;
                    im += lr * xi + li * xr;
                }
                assert!(
                    (t_re[mu * n + col] - re).abs() < 1.0e-12
                        && (t_im[mu * n + col] - im).abs() < 1.0e-12,
                    "apply_columns(adjoint)[{mu},{col}] disagrees with Λ†X"
                );
            }
        }
    }

    /// A determinant computed two ways on a small matrix: elimination against cofactor expansion.
    ///
    /// The string's phase is the argument of a product of these, so a determinant that is right up
    /// to a sign would put the polarization off by half a quantum and nothing else would notice.
    #[test]
    fn the_complex_determinant_matches_a_cofactor_expansion() {
        let n = 3;
        let re = vec![1.0, 2.0, -1.0, 0.5, -3.0, 2.5, 4.0, 1.0, 0.25];
        let im = vec![0.0, 1.0, 0.5, -2.0, 0.25, 1.5, 0.75, -1.0, 2.0];
        let (mut r, mut i) = (re.clone(), im.clone());
        let got = complex_determinant(&mut r, &mut i, n);

        // `det = Σ_σ sgn(σ) Π a_{k σ(k)}` over the six permutations of three indices.
        let at = |row: usize, col: usize| (re[row * n + col], im[row * n + col]);
        let mul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
        let perms: [([usize; 3], f64); 6] = [
            ([0, 1, 2], 1.0),
            ([1, 2, 0], 1.0),
            ([2, 0, 1], 1.0),
            ([0, 2, 1], -1.0),
            ([2, 1, 0], -1.0),
            ([1, 0, 2], -1.0),
        ];
        let (mut wr, mut wi) = (0.0, 0.0);
        for (sigma, sign) in perms {
            let mut term = (1.0, 0.0);
            for (row, &col) in sigma.iter().enumerate() {
                term = mul(term, at(row, col));
            }
            wr += sign * term.0;
            wi += sign * term.1;
        }
        assert!(
            (got[0] - wr).abs() < 1.0e-12 && (got[1] - wi).abs() < 1.0e-12,
            "elimination gave {got:?}, the cofactor expansion gives ({wr}, {wi})"
        );
    }
}
