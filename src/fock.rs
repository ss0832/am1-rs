// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO Fock-matrix build, spin-resolved: `F^σ = H_core + J(P_tot) − K(P^σ)`.
//!
//! The Coulomb part `J` is built from the **total** density (both spins); the exchange
//! part `K` from the **same-spin** density. The RHF (closed-shell) Fock is the special case
//! `P^σ = ½ P_tot`, i.e. `F = H_core + J(P) − K(½P)`. The one-center block uses the exact
//! one-center two-electron integrals ([`oc_two_electron`]); the two-center block uses the
//! rotated integrals from [`crate::integrals`].

use crate::basis::Basis;
use crate::error::Result;
use crate::hamiltonian::CoreHamiltonian;
use crate::linalg::Matrix;
use crate::params::Am1Parameters;
use crate::system::Molecule;

/// One-center two-electron integral `(a b | c d)` (all orbitals on the same atom), from the
/// AM1 one-center parameters. Orbital indices: 0 = s, 1..3 = p. Uses the NDDO index
/// symmetries `(ab|cd) = (ba|cd) = (ab|dc) = (cd|ab)`.
#[inline]
pub fn oc_two_electron(
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    gss: f64,
    gsp: f64,
    gpp: f64,
    gp2: f64,
    hsp: f64,
) -> f64 {
    // Diagonal-pair cases: bra = (x,x), ket = (y,y).
    if a == b && c == d {
        return match (a == 0, c == 0) {
            (true, true) => gss,           // (ss|ss)
            (true, false) => gsp,          // (ss|pp)
            (false, true) => gsp,          // (pp|ss)
            (false, false) => {
                if a == c {
                    gpp // (pp|pp)
                } else {
                    gp2 // (pp|p'p')
                }
            }
        };
    }
    // Off-diagonal-pair cases: sort bra/ket index pairs.
    let (ba, bb) = (a.min(b), a.max(b));
    let (kc, kd) = (c.min(d), c.max(d));
    // (s p_i | s p_i) = H_sp
    if ba == 0 && bb != 0 && kc == 0 && kd != 0 && bb == kd {
        return hsp;
    }
    // (p_i p_j | p_i p_j) = ½(G_pp − G_p2),  i ≠ j
    if ba != 0 && bb != 0 && ba != bb && ba == kc && bb == kd {
        return 0.5 * (gpp - gp2);
    }
    0.0
}

/// Build the spin-σ Fock matrix `F = H_core + J(p_tot) − K(p_spin)`.
pub fn build_fock_spin(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    p_tot: &Matrix,
    p_spin: &Matrix,
) -> Result<Matrix> {
    let mut f = core.h_core.clone();

    // One-center (intra-atomic) contributions.
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let (gss, gsp, gpp, gp2, hsp) =
            (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        // Coulomb (μν|λσ) from total density.
                        acc += p_tot[(off + la, off + si)]
                            * oc_two_electron(mu, nu, la, si, gss, gsp, gpp, gp2, hsp);
                        // Exchange (μλ|νσ) from same-spin density.
                        acc -= p_spin[(off + la, off + si)]
                            * oc_two_electron(mu, la, nu, si, gss, gsp, gpp, gp2, hsp);
                    }
                }
                f[(off + mu, off + nu)] += acc;
            }
        }
    }

    // Two-center (inter-atomic) contributions.
    for pair in &core.pairs {
        let (a, b) = (pair.a, pair.b);
        let te = &pair.te;
        let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
        let (na, nb) = (te.norb_i, te.norb_j);

        // Coulomb J from total density.
        for mu in 0..na {
            for nu in 0..na {
                let mut acc = 0.0;
                for la in 0..nb {
                    for si in 0..nb {
                        acc += p_tot[(ob + la, ob + si)] * te.two_e(mu, nu, la, si);
                    }
                }
                f[(oa + mu, oa + nu)] += acc;
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let mut acc = 0.0;
                for mu in 0..na {
                    for nu in 0..na {
                        acc += p_tot[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si);
                    }
                }
                f[(ob + la, ob + si)] += acc;
            }
        }

        // Exchange K from same-spin density: F(μ_a, λ_b) −= Σ P^σ(ν_a, σ_b) (μν|λσ).
        for mu in 0..na {
            for la in 0..nb {
                let mut acc = 0.0;
                for nu in 0..na {
                    for si in 0..nb {
                        acc += p_spin[(oa + nu, ob + si)] * te.two_e(mu, nu, la, si);
                    }
                }
                let v = -acc;
                f[(oa + mu, ob + la)] += v;
                f[(ob + la, oa + mu)] = f[(oa + mu, ob + la)];
            }
        }
    }

    Ok(f)
}

/// RHF (closed-shell) Fock: `F = H_core + J(P) − K(½P)`.
pub fn build_fock(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    density: &Matrix,
) -> Result<Matrix> {
    let mut half = density.clone();
    for v in half.as_mut_slice() {
        *v *= 0.5;
    }
    build_fock_spin(molecule, basis, params, core, density, &half)
}
