// SPDX-License-Identifier: GPL-3.0-or-later

//! Core (one-electron) Hamiltonian assembly.
//!
//! `H_core` holds the diagonal atomic energies `U_ss/U_pp`, the electron–core attraction
//! to every other atom (from the NDDO integrals), and the inter-atomic resonance
//! `H_μν = ½(β_μ + β_ν) S_μν`. The per-pair two-electron integrals are returned alongside
//! for reuse in the Fock build.

use crate::basis::Basis;
use crate::error::Result;
use crate::integrals::{pair_two_electron, PairTwoElec};
use crate::linalg::Matrix;
use crate::overlap::diatom_overlap;
use crate::params::Am1Parameters;
use crate::system::Molecule;

/// Rotated two-electron integrals for one atom pair, tagged with the ordered atom indices
/// (`a` is the heavy atom when the other is H).
pub struct PairIntegral {
    pub a: usize,
    pub b: usize,
    pub te: PairTwoElec,
}

pub struct CoreHamiltonian {
    pub h_core: Matrix,
    pub pairs: Vec<PairIntegral>,
}

/// Return the resonance β for orbital index `orb` (0 = s, else p).
#[inline]
fn beta_of(elem: &crate::params::Am1Element, orb: u8) -> f64 {
    if orb == 0 {
        elem.beta_s
    } else {
        elem.beta_p
    }
}

pub fn build_core(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
) -> Result<CoreHamiltonian> {
    let nao = basis.nao;
    let mut h = Matrix::zeros(nao, nao);

    // Diagonal U_ss / U_pp.
    for (mu, ao) in basis.aos.iter().enumerate() {
        let elem = params.element(ao.z)?;
        h[(mu, mu)] = if ao.orb == 0 { elem.u_ss } else { elem.u_pp };
    }

    use rayon::prelude::*;

    let nat = molecule.atoms.len();

    // Enumerate atom pairs, then compute their (independent) integrals in parallel.
    let pair_indices: Vec<(usize, usize)> =
        (0..nat).flat_map(|u| ((u + 1)..nat).map(move |v| (u, v))).collect();
    let computed: Vec<(usize, usize, PairTwoElec, [[f64; 4]; 4])> = pair_indices
        .par_iter()
        .map(|&(u, v)| -> Result<(usize, usize, PairTwoElec, [[f64; 4]; 4])> {
            let eu = params.element(molecule.atoms[u].z)?;
            let ev = params.element(molecule.atoms[v].z)?;
            // Ordered pair: heavy atom first when the other is H.
            let (a, b) = if eu.has_p() || !ev.has_p() { (u, v) } else { (v, u) };
            let (ea, eb) = (
                params.element(molecule.atoms[a].z)?,
                params.element(molecule.atoms[b].z)?,
            );
            let pos_a = molecule.atoms[a].position;
            let pos_b = molecule.atoms[b].position;
            let d = pos_b - pos_a;
            let r = d.norm();
            let xij = d / r;
            let te = pair_two_electron(ea, eb, xij, r);
            let s_block = diatom_overlap(ea, pos_a, eb, pos_b)?;
            Ok((a, b, te, s_block))
        })
        .collect::<Result<Vec<_>>>()?;

    // Assemble H_core serially from the precomputed per-pair integrals.
    let mut pairs = Vec::with_capacity(computed.len());
    for (a, b, te, s_block) in computed {
        {
            let (ea, eb) = (
                params.element(molecule.atoms[a].z)?,
                params.element(molecule.atoms[b].z)?,
            );
            let off_a = basis.atom_offset[a];
            let off_b = basis.atom_offset[b];
            let na = basis.atom_norb[a];
            let nb = basis.atom_norb[b];

            // Electron–core attraction: e1b onto atom a's block, e2a onto atom b's block.
            for i in 0..na {
                for j in 0..na {
                    h[(off_a + i, off_a + j)] += te.e1b[i][j];
                }
            }
            for i in 0..nb {
                for j in 0..nb {
                    h[(off_b + i, off_b + j)] += te.e2a[i][j];
                }
            }

            // Resonance β·S (inter-atomic, symmetric).
            for i in 0..na {
                let bi = beta_of(ea, basis.aos[off_a + i].orb);
                for j in 0..nb {
                    let bj = beta_of(eb, basis.aos[off_b + j].orb);
                    let value = 0.5 * (bi + bj) * s_block[i][j];
                    h[(off_a + i, off_b + j)] = value;
                    h[(off_b + j, off_a + i)] = value;
                }
            }

            pairs.push(PairIntegral { a, b, te });
        }
    }

    Ok(CoreHamiltonian { h_core: h, pairs })
}
