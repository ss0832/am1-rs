// SPDX-License-Identifier: GPL-3.0-or-later

//! The AM1 dipole operator, and the coupling to a uniform external electric field.
//!
//! # Why this module exists
//!
//! Three places in the crate need the same object and used to build it twice: the SCF reports a
//! dipole ([`crate::scf::run_am1`]), the periodic field response differentiates one
//! ([`crate::pbc::hessian::dielectric_tensor`]), and the Born charges and the infrared atomic
//! polar tensor ([`crate::ir`]) differentiate it with respect to nuclear position. A sign or a
//! factor of two that disagrees between them does not fail loudly — it produces a plausible
//! number with the wrong sign somewhere downstream. So the operator is defined **once**, here,
//! and the sign convention is written down rather than left to be re-derived at each use.
//!
//! # The convention, in full
//!
//! ```text
//! Q_a  = Z_a − p_a                     net charge on atom a; p_a = Σ_{μ ∈ a} P_μμ
//!
//! M_α  : +R_{a,α}  on every diagonal element of atom a's block
//!        +dd_a     on both (s, p_α) and (p_α, s) of atom a, for atoms with a p shell
//!
//! μ_α  = Σ_a Z_a R_{a,α} − Tr[P M_α]                                    (e·Bohr)
//!
//! E(F) = E₀ − μ·F   ⇒   h^F = +Σ_α F_α M_α  added to H_core,
//!                       and  −F·Σ_a Z_a R_a  added to the core energy
//!
//! ∂E/∂R_{a,β} += −F_β Q_a            ⇒   the force on atom a gains  +Q_a F
//! ```
//!
//! `dd_a` is the NDDO `s`–`p` charge separation ([`crate::params::Am1Element::dd`]); the `−2 dd`
//! that appears in the SCF's dipole assembly is this `+dd` on two symmetric matrix elements,
//! carried through the minus sign in `μ = ΣZR − Tr[PM]`. That factor of two is exactly the kind
//! of thing this module exists to stop from being rediscovered.
//!
//! # What the operator is not
//!
//! `R_a` is a *position*, so `M_α` is not a lattice-periodic operator. Under periodic boundary
//! conditions the absolute dipole it produces is not physically meaningful — the polarization of
//! a solid is defined only modulo a quantum, and this is not a Berry phase. Its **derivatives**
//! are well defined, because charge is conserved (`Σ_b ∂Q_b = 0`) and the origin dependence
//! cancels term by term; that is why [`crate::pbc::born_charges`] is computable and an absolute
//! periodic dipole is not. See [`crate::pbc::hessian::dielectric_tensor`] for the measurement of
//! that cancellation.
//!
//! For the same reason a uniform external field is offered for **molecules only**: `F·R` is
//! unbounded below and is not a periodic perturbation.

use crate::basis::Basis;
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Am1Parameters;
use crate::system::Molecule;

/// The three Cartesian components of the AM1 dipole operator `M_α`, in the AO basis.
///
/// See the module documentation for the sign convention. Index `[0]`, `[1]`, `[2]` are `x`, `y`,
/// `z`; each is `nao × nao` and symmetric.
pub fn dipole_operator(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
) -> Result<[Matrix; 3]> {
    let nao = basis.nao;
    let mut m = [
        Matrix::zeros(nao, nao),
        Matrix::zeros(nao, nao),
        Matrix::zeros(nao, nao),
    ];
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let norb = basis.atom_norb[ia];
        let r = atom.position;
        for (alpha, block) in m.iter_mut().enumerate() {
            // The atom's position, on every diagonal element of its block: the charge on this
            // atom sits at this position whichever orbital carries it.
            let ra = r.get(alpha);
            for k in 0..norb {
                block[(off + k, off + k)] += ra;
            }
            // The on-site s-p hybridization moment. Only atoms with a p shell have one.
            if norb == 4 {
                block[(off, off + 1 + alpha)] += elem.dd;
                block[(off + 1 + alpha, off)] += elem.dd;
            }
        }
    }
    Ok(m)
}

/// `Σ_a Z_a R_a` — the core (nuclear) contribution to the dipole, in e·Bohr.
pub fn nuclear_dipole(molecule: &Molecule, params: &Am1Parameters) -> Result<Vec3> {
    let mut acc = Vec3::zero();
    for atom in &molecule.atoms {
        acc += atom.position * params.element(atom.z)?.core_charge;
    }
    Ok(acc)
}

/// Total dipole `μ = Σ_a Z_a R_a − Tr[P M]` for a density `p`, in e·Bohr.
///
/// This is the same quantity [`crate::scf::Am1Result::dipole_debye`] reports, before the Debye
/// conversion. `tests/` asserts the two agree, which is what keeps this module's convention and
/// the SCF's assembly from drifting apart.
pub fn dipole_from_density(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    p: &Matrix,
) -> Result<Vec3> {
    let m = dipole_operator(molecule, basis, params)?;
    let nuclear = nuclear_dipole(molecule, params)?;
    Ok(Vec3::new(
        nuclear.x - p.frobenius_dot(&m[0]),
        nuclear.y - p.frobenius_dot(&m[1]),
        nuclear.z - p.frobenius_dot(&m[2]),
    ))
}

/// The one-electron field term `h^F = +Σ_α F_α M_α` added to `H_core`, in eV.
///
/// `field` is in eV per (e·Bohr), which is what the crate's eV/Bohr interior wants; the Python
/// and ASE boundaries convert.
pub fn field_hamiltonian(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    field: Vec3,
) -> Result<Matrix> {
    let m = dipole_operator(molecule, basis, params)?;
    let mut h = Matrix::zeros(basis.nao, basis.nao);
    for (alpha, block) in m.iter().enumerate() {
        let f = field.get(alpha);
        if f == 0.0 {
            continue;
        }
        for (hv, bv) in h.as_mut_slice().iter_mut().zip(block.as_slice()) {
            *hv += f * bv;
        }
    }
    Ok(h)
}

/// The nuclear half of `−μ·F`, i.e. `−F · Σ_a Z_a R_a`, in eV.
///
/// It belongs with the core–core energy rather than with the electronic term because it does not
/// involve the density; keeping it there is what makes `∂E/∂R_a = −F Q_a` come out as the sum of
/// this term's `−F Z_a` and the electronic term's `+F p_a`.
pub fn field_core_energy(molecule: &Molecule, params: &Am1Parameters, field: Vec3) -> Result<f64> {
    Ok(-field.dot(nuclear_dipole(molecule, params)?))
}

/// The field's contribution to `∂E/∂R_a`, in eV/Bohr, given the net charges.
///
/// `M_α` is *linear* in `R`, so this is the whole nuclear derivative at fixed density and there
/// is no second derivative — which is why a field changes the analytic Hessian only through the
/// CPHF response. `charges` is `Q_a = Z_a − p_a`, as [`crate::pbc::ewald::net_charges`] returns.
pub fn field_gradient(field: Vec3, charges: &[f64]) -> Vec<Vec3> {
    charges.iter().map(|q| field * -*q).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf::{run_am1, Am1Options};

    /// The operator's dipole must be the SCF's dipole. This is the assertion that keeps the
    /// convention in the module documentation honest: if either side is edited, this fails.
    #[test]
    fn the_operator_reproduces_the_scf_dipole() {
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let scf = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let basis = Basis::build(&mol, &params).unwrap();

        let mine = dipole_from_density(&mol, &basis, &params, &scf.density).unwrap();
        // `Am1Result` reports Debye; convert back to e·Bohr for the comparison.
        let theirs = scf.dipole_debye / crate::constants::AU_DIPOLE_TO_DEBYE;
        eprintln!(
            "    operator ({:.8}, {:.8}, {:.8})  scf ({:.8}, {:.8}, {:.8}) e·Bohr",
            mine.x, mine.y, mine.z, theirs.x, theirs.y, theirs.z
        );
        assert!((mine - theirs).norm() < 1.0e-12, "{mine:?} != {theirs:?}");
    }

    /// `M_α` is symmetric, and its trace against a density is real by construction. A broken
    /// hybridization scatter (one of the two off-diagonal entries missing) shows up here.
    #[test]
    fn the_operator_is_symmetric() {
        let xyz = "5\nmethane\nC 0.0 0.0 0.0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let basis = Basis::build(&mol, &params).unwrap();
        let m = dipole_operator(&mol, &basis, &params).unwrap();
        for block in &m {
            for i in 0..block.rows {
                for j in 0..block.cols {
                    assert!(
                        (block[(i, j)] - block[(j, i)]).abs() < 1.0e-15,
                        "M is not symmetric at ({i}, {j})"
                    );
                }
            }
        }
    }

    /// Shifting the origin shifts `μ` by `q · shift` and by nothing else — zero for a neutral
    /// molecule. This is the property that makes the *derivatives* of `μ` well defined, and it
    /// is a statement about the operator rather than about any particular system.
    #[test]
    fn the_origin_shifts_the_dipole_only_through_the_net_charge() {
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let params = Am1Parameters::standard().unwrap();
        let shift = Vec3::new(1.7, -0.3, 0.9);

        // Water has 8 valence electrons, so the +1 cation is a doublet: a singlet request would
        // be rejected on parity, which is the SCF being right rather than the test being unlucky.
        for (charge, multiplicity) in [(0.0, 1), (1.0, 2)] {
            let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
            let opts = Am1Options {
                charge,
                multiplicity,
                ..Am1Options::default()
            };
            let scf = run_am1(&mol, &params, &opts).unwrap();
            let basis = Basis::build(&mol, &params).unwrap();
            let base = dipole_from_density(&mol, &basis, &params, &scf.density).unwrap();

            let mut moved = mol.clone();
            for atom in &mut moved.atoms {
                atom.position += shift;
            }
            // The same density, deliberately: this is a property of the operator, not of a
            // re-converged SCF at a translated geometry.
            let shifted = dipole_from_density(&moved, &basis, &params, &scf.density).unwrap();
            let expected = base + shift * charge;
            eprintln!(
                "    charge {charge:+.1}: |μ(shifted) − μ − qΔ| = {:.3e}",
                (shifted - expected).norm()
            );
            assert!((shifted - expected).norm() < 1.0e-10);
        }
    }
}
