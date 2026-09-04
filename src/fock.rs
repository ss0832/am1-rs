// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO Fock-matrix build, spin-resolved: `F^σ = H_core + J(P_tot) − K(P^σ)`.
//!
//! The Coulomb part `J` is built from the **total** density (both spins); the exchange
//! part `K` from the **same-spin** density. The RHF (closed-shell) Fock is the special case
//! `P^σ = ½ P_tot`, i.e. `F = H_core + J(P) − K(½P)`. The one-center block uses the exact
//! one-center two-electron integrals ([`oc_two_electron`]); the two-center block uses the
//! rotated integrals from [`crate::integrals`].

use rayon::prelude::*;

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
// Four orbital indices and five one-centre parameters. Bundling them into a struct would put an
// indirection in the innermost loop of the Fock build to satisfy a lint.
#[allow(clippy::too_many_arguments)]
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
            (true, true) => gss,  // (ss|ss)
            (true, false) => gsp, // (ss|pp)
            (false, true) => gsp, // (pp|ss)
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

/// How the two-centre pair loop is driven.
///
/// The distinction exists because this function is called from two very different places.
///
/// The SCF calls it once per iteration with the whole machine idle, and wants
/// [`PairLoop::Parallel`]. The CPHF solver calls it once per perturbation per iteration, with
/// `3N` perturbations already running under rayon — and there, an inner rayon pool contends with
/// the outer one for the same worker threads. Measured on a 150-atom cluster: 21 ms per call
/// from the SCF against **38 ms** per call from inside the CPHF, for the same work on the same
/// matrices. [`PairLoop::Sequential`] is for that case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairLoop {
    Parallel,
    /// Run the pair loop on the calling thread. Use when the caller is already parallel.
    Sequential,
}

/// What to put in the matrix before the two-electron terms are added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FockBase {
    /// Start from `H_core`, giving the Fock matrix `F = H_core + J − K`.
    CoreHamiltonian,
    /// Start from zero, giving only the two-electron part `G = J − K`.
    ///
    /// This is what the CPHF response needs. It used to build the full Fock matrix and subtract
    /// `H_core` again afterwards, which is two extra `nao²` passes over memory per call — at
    /// 6296 calls on a 150-atom system, about 9 GB of pointless traffic.
    Zero,
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
    build_fock_spin_with(
        molecule,
        basis,
        params,
        core,
        p_tot,
        p_spin,
        1.0,
        FockBase::CoreHamiltonian,
        PairLoop::Parallel,
    )
}

/// Long-range monopole potential `V_a = Σ_b Δ_ab Q_b` (eV per unit charge), one entry per atom.
///
/// `Q_b = Z_b − p_b` is the **net** Mulliken charge and `Δ` is the Ewald-minus-truncated-sum
/// correction described on [`CoreHamiltonian::long_range`]. Empty when there is no correction.
///
/// # Derivation
///
/// The energy this reproduces is `E_corr = ½ Σ_ab Q_a Q_b Δ_ab = ½ Σ_a Q_a V_a`. Since
/// `∂Q_a/∂P_μμ = −1` for `μ` on `a`,
///
/// ```text
/// ∂E_corr/∂P_μμ = −Σ_b Δ_ab Q_b = −V_a
/// ```
///
/// so the Fock shift is `−V_a` — which is what [`build_fock_spin_with`] applies.
///
/// That shift then also lands in the ordinary energy trace, which has to be undone. Writing
/// `F = F₀ + S` with `S_μμ = −V_a`,
///
/// ```text
/// ½(Tr[P H₀] + Tr[P F]) = E_elec⁰ + ½ Σ_a p_a (−V_a)
/// ```
///
/// so the corrected electronic energy is
///
/// ```text
/// E = ½(Tr[P H₀] + Tr[P F]) + ½ Σ_a p_a V_a + ½ Σ_a Q_a V_a
///   = ½(Tr[P H₀] + Tr[P F]) + ½ Σ_a Z_a V_a
/// ```
///
/// using `p_a + Q_a = Z_a`. That last expression is [`long_range_energy_term`], and it is why
/// the correction needs no term in `H_core` and none in the core–core repulsion.
pub fn long_range_potential(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    p_tot: &Matrix,
) -> Result<Vec<f64>> {
    // Two sources, the same shape. The Ewald correction replaces a truncated lattice sum; the
    // far field replaces the multipole structure of distant pairs. Both are `Σ_b (kernel) Q_b`
    // over the **net** charges, so both go through this one potential and the single matching
    // energy term rather than each growing its own copy of the derivation.
    let mut v = match &core.long_range {
        Some(delta) => long_range_potential_from_delta(molecule, basis, params, delta, p_tot)?,
        None => Vec::new(),
    };
    if let Some(far) = &core.far_field {
        let charges = crate::pbc::ewald::net_charges(molecule, basis, params, p_tot)?;
        let contribution = far.potential(&charges);
        if v.is_empty() {
            v = contribution;
        } else {
            for (a, c) in v.iter_mut().zip(&contribution) {
                *a += c;
            }
        }
    }
    Ok(v)
}

/// [`long_range_potential`] against an explicit `Δ` matrix.
///
/// The k-point path assembles its own real-space blocks rather than a [`CoreHamiltonian`], and
/// the correction is **k-independent** — it is a shift on each atom's own diagonal, so it lands
/// entirely in the `T = 0` block — so both paths share this one definition.
pub fn long_range_potential_from_delta(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    delta: &Matrix,
    p_tot: &Matrix,
) -> Result<Vec<f64>> {
    let nat = molecule.atoms.len();
    let mut charges = Vec::with_capacity(nat);
    for (a, atom) in molecule.atoms.iter().enumerate() {
        let mut population = 0.0;
        let off = basis.atom_offset[a];
        for k in 0..basis.atom_norb[a] {
            population += p_tot[(off + k, off + k)];
        }
        charges.push(params.element(atom.z)?.core_charge - population);
    }
    let mut v = vec![0.0; nat];
    for (a, va) in v.iter_mut().enumerate() {
        for (b, q) in charges.iter().enumerate() {
            *va += delta[(a, b)] * q;
        }
    }
    Ok(v)
}

/// The `+½ Σ_a Z_a V_a` (eV) that the energy expression must carry alongside the Fock shift.
///
/// Zero when there is no long-range correction. See [`long_range_potential`] for the derivation;
/// **every** energy expression that uses a Fock built by [`build_fock_spin_with`] has to add
/// this, or the reported energy will not be the one whose stationary point the SCF found.
pub fn long_range_energy_term(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    p_tot: &Matrix,
) -> Result<f64> {
    let v = long_range_potential(molecule, basis, params, core, p_tot)?;
    long_range_energy_from_potential(molecule, params, &v)
}

/// The `+½ Σ_a Z_a V_a` (eV) for a potential already in hand. See [`long_range_energy_term`].
pub fn long_range_energy_from_potential(
    molecule: &Molecule,
    params: &Am1Parameters,
    v: &[f64],
) -> Result<f64> {
    let mut e = 0.0;
    for (a, va) in v.iter().enumerate() {
        e += params.element(molecule.atoms[a].z)?.core_charge * va;
    }
    Ok(0.5 * e)
}

/// [`build_fock_spin`] with explicit control over the starting matrix and the pair loop.
///
/// `spin_scale` multiplies `p_spin` wherever the exchange reads it, so a closed-shell caller can
/// pass its **total** density with `0.5` instead of materializing a halved copy. The exchange is
/// linear in `p_spin`, so this is exact, and it removes an `nao²` allocation, copy and scale from
/// every call — which matters because [`build_g_matrix`] is called `3N` times per CPHF iteration
/// and at 1602 AOs each of those copies is 20 MB. `crate::pbc::scf::build_realspace_fock` already
/// took a scale for the same reason.
#[allow(clippy::too_many_arguments)]
pub fn build_fock_spin_with(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    p_tot: &Matrix,
    p_spin: &Matrix,
    spin_scale: f64,
    base: FockBase,
    pair_loop: PairLoop,
) -> Result<Matrix> {
    let mut f = match base {
        FockBase::CoreHamiltonian => core.h_core.clone(),
        FockBase::Zero => Matrix::zeros(basis.nao, basis.nao),
    };

    // Long-range monopole correction.
    //
    // Which form depends on what is being built, and the distinction is not cosmetic.
    // `−V_a = −Σ_b Δ_ab Z_b + Σ_b Δ_ab p_b` splits into a part independent of `P` and a part
    // linear in it:
    //
    // * A **full Fock** needs both, and takes them together as `−V_a` through the net charges,
    //   because the two halves are individually enormous and cancel — see
    //   [`crate::hamiltonian::CoreHamiltonian::long_range`].
    // * A **G matrix** ([`FockBase::Zero`], used for the CPHF response) needs only the part
    //   linear in `P`, `+Σ_b Δ_ab p_b`. Including the `Z` term there would add a constant to a
    //   quantity that must be linear in its argument, and the CPHF would be solving a different
    //   equation than the one whose solution it reports. Measured: the CPHF stopped converging.
    if core.long_range.is_some() || core.far_field.is_some() {
        let nat = molecule.atoms.len();
        let mut shift = vec![0.0; nat];
        match base {
            FockBase::CoreHamiltonian => {
                let v = long_range_potential(molecule, basis, params, core, p_tot)?;
                for (s, va) in shift.iter_mut().zip(&v) {
                    *s = -va;
                }
            }
            FockBase::Zero => {
                let mut population = vec![0.0; nat];
                for (a, pop) in population.iter_mut().enumerate() {
                    let off = basis.atom_offset[a];
                    for k in 0..basis.atom_norb[a] {
                        *pop += p_tot[(off + k, off + k)];
                    }
                }
                if let Some(delta) = &core.long_range {
                    for (a, s) in shift.iter_mut().enumerate() {
                        for (b, pop) in population.iter().enumerate() {
                            *s += delta[(a, b)] * pop;
                        }
                    }
                }
                if let Some(far) = &core.far_field {
                    // The population form, for the same reason as above: only the part linear
                    // in `P` belongs in an operator that must be linear in its argument.
                    let contribution = far.potential(&population);
                    for (s, c) in shift.iter_mut().zip(&contribution) {
                        *s += c;
                    }
                }
            }
        }
        for (a, s) in shift.iter().enumerate() {
            let off = basis.atom_offset[a];
            for k in 0..basis.atom_norb[a] {
                f[(off + k, off + k)] += s;
            }
        }
    }

    // One-center (intra-atomic) contributions.
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let (gss, gsp, gpp, gp2, hsp) = (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        // Coulomb (μν|λσ) from total density.
                        acc += p_tot[(off + la, off + si)]
                            * oc_two_electron(mu, nu, la, si, gss, gsp, gpp, gp2, hsp);
                        // Exchange (μλ|νσ) from same-spin density.
                        acc -= spin_scale
                            * p_spin[(off + la, off + si)]
                            * oc_two_electron(mu, la, nu, si, gss, gsp, gpp, gp2, hsp);
                    }
                }
                f[(off + mu, off + nu)] += acc;
            }
        }
    }

    // Two-center (inter-atomic) contributions.
    //
    // The per-pair work is independent, but the Coulomb terms accumulate into the *atom*
    // diagonal blocks, which neighbouring pairs share. So compute in parallel and scatter
    // serially. The scatter touches 48 numbers per pair against ~100 multiply-adds of
    // integral contraction, so it is not what costs.
    //
    // Chunked rather than collected whole: one buffer entry is 48 f64, and at 801 atoms
    // there are 320k pairs, so collecting the lot would add ~123 MB of transient allocation
    // on top of the pair integrals themselves.
    const CHUNK: usize = 8192;
    let pair_contribution = |pair: &crate::hamiltonian::PairIntegral| -> PairFock {
        let te = &pair.te;
        let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
        let (na, nb) = (te.norb_i, te.norb_j);
        let mut out = PairFock {
            oa,
            ob,
            na,
            nb,
            ja: [0.0; 16],
            jb: [0.0; 16],
            k: [0.0; 16],
        };
        // Copy the three density blocks this pair needs into flat stack arrays first. Each is
        // read sixteen times by the loops below, and going through `Matrix`'s 2D index every time
        // costs a multiply and a bounds check to fetch data that fits in two cache lines.
        //
        // Honest note on what this bought: nothing measurable. The Fock build costs about 26x
        // what its floating-point work alone accounts for, and this was the hypothesis for where
        // that goes — it was wrong, and hoisting the blocks left the timing inside run-to-run
        // noise. It is kept because it is clearer and cannot be slower, not because it is a
        // proven optimization. What actually cut the Hessian cost was reducing the *number* of
        // Fock builds (conjugate gradient in `hessian.rs`) and not building `H_core` only to
        // subtract it again (`FockBase::Zero`).
        //
        // # The batching question is settled, and the answer is no
        //
        // This comment used to end by naming the next thing to try: batch the CPHF perturbations
        // so the pair integrals are read once per iteration instead of once per perturbation.
        // That experiment has since been run — in a sibling NDDO crate with this same loop — and
        // it is **slower**. On a 102-atom Hessian, three variants of the same idea:
        //
        //   | attempt                          | result        |
        //   |----------------------------------|---------------|
        //   | batch the response Fock over DOFs | 5.2 -> 8.8 s  |
        //   | gather the density sub-blocks     | 5.2 -> 9.8 s  |
        //   | pack the Coulomb contraction      | 4.17 -> 4.39 s |
        //
        // The batching did what it was meant to structurally — 70 Fock passes instead of 3961 —
        // and was still slower. The premise was that this loop is memory bound; at that size the
        // whole integral set is about 4 MB, so it sits in L3 across calls and there was no traffic
        // to save, while batching costs the per-DOF parallelism the CPHF's `par_iter` gets for
        // free. Packing is the sharpest case: strictly less arithmetic, still slower.
        //
        // The common thread is that at NDDO block sizes (4x4) this loop is neither memory bound
        // nor arithmetic bound — it is bound by **per-pair overhead**. Anything that adds a fixed
        // per-pair cost loses even when it removes work from the inner loop. Note that the second
        // row is what the block below *is*, measured as a 2x loss elsewhere and as nothing here;
        // it stays only because it is clearer, and it is on the list if this ever needs revisiting.
        //
        // Recorded here so the afternoon is not spent again.
        let mut p_bb = [0.0f64; 16];
        let mut p_aa = [0.0f64; 16];
        let mut p_ab = [0.0f64; 16];
        for i in 0..nb {
            for j in 0..nb {
                p_bb[i * 4 + j] = p_tot[(ob + i, ob + j)];
            }
        }
        for i in 0..na {
            for j in 0..na {
                p_aa[i * 4 + j] = p_tot[(oa + i, oa + j)];
            }
        }
        for i in 0..na {
            for j in 0..nb {
                p_ab[i * 4 + j] = spin_scale * p_spin[(oa + i, ob + j)];
            }
        }

        // Coulomb J from the total density, both directions.
        for mu in 0..na {
            for nu in 0..na {
                // One row lookup per bra pair instead of one per element.
                let row = te.two_e_row(mu, nu);
                let mut acc = 0.0;
                for la in 0..nb {
                    for si in 0..nb {
                        acc += p_bb[la * 4 + si] * row[crate::integrals::PACK[la][si]];
                    }
                }
                out.ja[mu * 4 + nu] = acc;
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let ket = crate::integrals::PACK[la][si];
                let mut acc = 0.0;
                for mu in 0..na {
                    for nu in 0..na {
                        acc += p_aa[mu * 4 + nu] * te.two_e_row(mu, nu)[ket];
                    }
                }
                out.jb[la * 4 + si] = acc;
            }
        }
        // Exchange K from the same-spin density:
        // F(mu_a, lambda_b) -= sum P^sigma(nu_a, sigma_b) (mu nu | lambda sigma).
        for mu in 0..na {
            for la in 0..nb {
                let mut acc = 0.0;
                for nu in 0..na {
                    let row = te.two_e_row(mu, nu);
                    for si in 0..nb {
                        acc += p_ab[nu * 4 + si] * row[crate::integrals::PACK[la][si]];
                    }
                }
                out.k[mu * 4 + la] = -acc * pair.exchange_scale;
            }
        }
        out
    };

    let mut buf: Vec<PairFock> = Vec::with_capacity(CHUNK.min(core.pairs.len()));
    for chunk in core.pairs.chunks(CHUNK) {
        buf.clear();
        match pair_loop {
            PairLoop::Parallel => {
                chunk
                    .par_iter()
                    .map(&pair_contribution)
                    .collect_into_vec(&mut buf);
            }
            PairLoop::Sequential => buf.extend(chunk.iter().map(&pair_contribution)),
        }

        for pf in &buf {
            for mu in 0..pf.na {
                for nu in 0..pf.na {
                    f[(pf.oa + mu, pf.oa + nu)] += pf.ja[mu * 4 + nu];
                }
            }
            for la in 0..pf.nb {
                for si in 0..pf.nb {
                    f[(pf.ob + la, pf.ob + si)] += pf.jb[la * 4 + si];
                }
            }
            // Accumulate into both triangles rather than mirroring one into the other. The
            // pair list holds one representative per physical pair, so this entry also stands
            // for its mirror, whose exchange contribution to the transposed block is equal.
            // Mirroring by assignment would be wrong the moment two lattice translations
            // connect the same pair of home-cell atoms, since the second would overwrite the
            // first. For a molecule each block is visited once and the two forms agree.
            for mu in 0..pf.na {
                for la in 0..pf.nb {
                    let v = pf.k[mu * 4 + la];
                    f[(pf.oa + mu, pf.ob + la)] += v;
                    f[(pf.ob + la, pf.oa + mu)] += v;
                }
            }
        }
    }

    Ok(f)
}

/// One pair's contribution to the Fock matrix, staged so the parallel contraction and the
/// serial scatter can be separated.
struct PairFock {
    oa: usize,
    ob: usize,
    na: usize,
    nb: usize,
    ja: [f64; 16],
    jb: [f64; 16],
    k: [f64; 16],
}

/// RHF (closed-shell) Fock: `F = H_core + J(P) − K(½P)`.
pub fn build_fock(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    density: &Matrix,
) -> Result<Matrix> {
    // The total density with a spin scale of ½, not a halved copy: `G(P) = J(P) − K(½P)`, and
    // the exchange is linear in its argument.
    build_fock_spin_with(
        molecule,
        basis,
        params,
        core,
        density,
        density,
        0.5,
        FockBase::CoreHamiltonian,
        PairLoop::Parallel,
    )
}

/// Closed-shell **two-electron** matrix alone: `G(P) = J(P) − K(½P)`, with no `H_core`.
///
/// The CPHF response kernel, and the only thing that kernel wants. Building the full Fock matrix
/// and subtracting `H_core` afterwards costs two extra `nao²` passes per call for a result that
/// is thrown away, and the CPHF makes thousands of calls.
///
/// `pair_loop` should be [`PairLoop::Sequential`] whenever the caller is itself running under
/// rayon — see [`PairLoop`].
pub fn build_g_matrix(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    density: &Matrix,
    pair_loop: PairLoop,
) -> Result<Matrix> {
    build_fock_spin_with(
        molecule,
        basis,
        params,
        core,
        density,
        density,
        0.5,
        FockBase::Zero,
        pair_loop,
    )
}
