// SPDX-License-Identifier: GPL-3.0-or-later

//! **Geometry optimization under periodic boundary conditions.**
//!
//! [`crate::optimizer::optimize`] relaxes a molecule against the molecular gradient. It cannot
//! relax a periodic structure: the gradient it drives on is the molecular one, and it has no way
//! to move the lattice at all. This module drives on [`crate::pbc::pbc_gradient`] — forces *and*
//! stress from the k-point SCF — so both the atoms and the cell can relax.
//!
//! # Two sets of variables, and why they are not both Cartesian
//!
//! With the cell fixed, the variables are the Cartesian positions and this is the molecular
//! optimizer with a periodic gradient underneath.
//!
//! With the cell free they cannot be. An atom at a Cartesian position does not stay at the same
//! place in the *structure* when the lattice deforms — relaxing the cell would drag every atom out
//! of its site and the two motions would fight each other. The standard variables are therefore
//!
//! * **scaled (fractional) coordinates** `f`, with `r = h f` and `h` the lattice-vector matrix, so
//!   an atom on a symmetry site stays on it under any deformation; and
//! * a **strain** `ε` measured from the cell the run started with, `h(ε) = (I + ε) h_ref`.
//!
//! Their gradients are `∂E/∂f_a = hᵀ (∂E/∂r_a)` and `∂E/∂ε = Ω σ`, the second being the definition
//! of the stress this crate computes (see [`crate::pbc::gradient`]). At a target pressure `P` the
//! objective is the enthalpy `H = E + PΩ`, whose strain gradient is `Ω(σ + P I)` because
//! `∂Ω/∂ε_αβ = Ω δ_αβ`.
//!
//! # The two gradients are not the same size, so they are scaled
//!
//! `∂E/∂f` is an energy per fractional unit — a whole lattice vector — while `∂E/∂ε` is an energy
//! per unit strain of the *whole cell*, which grows with the number of atoms in it. Feeding both
//! to one L-BFGS unscaled makes the step length appropriate for one of them and wrong for the
//! other. [`PbcOptOptions::cell_factor`] divides the strain variable's gradient (and multiplies
//! the variable), defaulting to the atom count, which is the usual choice and makes the two
//! blocks comparable for a cell of any size.
//!
//! # What is frozen
//!
//! A strain component `ε_αβ` is a variable only when axes `α` and `β` are **both** periodic. A
//! slab's vacuum direction has no cell to relax, and straining it would change nothing but the
//! reported measure; a chain has one free component out of six. This is read off
//! [`crate::lattice::Lattice::periodic`] rather than being an option, because it is a property of
//! the structure and not a choice.
//!
//! # Convergence
//!
//! On the physical quantities, not on the step: the largest Cartesian force component against
//! [`PbcOptOptions::gtol`], and — when the cell is free — the largest free component of
//! `σ + P I` against [`PbcOptOptions::stress_tol`]. A run that stops because the line search
//! could not make progress reports `converged = false` rather than pretending.

use crate::error::{Am1Error, Result};
use crate::lattice::Lattice;
use crate::math::{Mat3, Vec3};
use crate::optimizer::Lbfgs;
use crate::params::Am1Parameters;
use crate::pbc::gradient::{pbc_gradient, PbcGradient};
use crate::pbc::scf::{run_pbc_scf, PbcOptions, PbcResult};
use crate::system::Molecule;

/// Settings for [`optimize_periodic`].
#[derive(Clone, Debug)]
pub struct PbcOptOptions {
    pub max_iter: usize,
    /// Convergence on the largest Cartesian force component (eV/Bohr).
    pub gtol: f64,
    /// L-BFGS history length.
    pub history: usize,
    /// Relax the lattice against the stress as well as the atoms against the forces.
    pub relax_cell: bool,
    /// Target pressure, eV per Bohr^d with `d` the number of periodic directions.
    ///
    /// Positive compresses. Ignored unless [`Self::relax_cell`] is set.
    pub pressure: f64,
    /// Convergence on the largest free component of `σ + P I` (same units as the stress).
    pub stress_tol: f64,
    /// Balances the position and strain blocks; see the module documentation. `None` uses the
    /// atom count.
    pub cell_factor: Option<f64>,
}

impl Default for PbcOptOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            gtol: 1.0e-3,
            history: 8,
            relax_cell: false,
            pressure: 0.0,
            stress_tol: 1.0e-5,
            cell_factor: None,
        }
    }
}

/// One accepted step of a periodic relaxation.
#[derive(Clone, Debug)]
pub struct PbcOptStep {
    pub energy_ev: f64,
    /// Largest Cartesian force component, eV/Bohr.
    pub max_force: f64,
    /// Largest free component of `σ + P I`.
    pub max_stress: f64,
    pub positions: Vec<Vec3>,
    pub cell: Lattice,
}

/// The outcome of a periodic relaxation.
#[derive(Clone, Debug)]
pub struct PbcOptResult {
    /// The relaxed structure: positions **and** cell.
    pub molecule: Molecule,
    pub scf: PbcResult,
    pub gradient: PbcGradient,
    pub converged: bool,
    pub iterations: usize,
    pub trajectory: Vec<PbcOptStep>,
}

/// Relax a periodic structure with L-BFGS on the analytic forces and stress.
pub fn optimize_periodic(
    molecule: &Molecule,
    params: &Am1Parameters,
    scf_options: &PbcOptions,
    opt: &PbcOptOptions,
) -> Result<PbcOptResult> {
    let reference = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a periodic optimization needs a cell".into()))?;
    let nat = molecule.atoms.len();
    if nat == 0 {
        return Err(Am1Error::InvalidInput(
            "a periodic optimization needs at least one atom".into(),
        ));
    }
    let free = free_strain_components(&reference);
    if opt.relax_cell && free.is_empty() {
        return Err(Am1Error::InvalidInput(
            "relax_cell was requested for a cell with no periodic directions".into(),
        ));
    }
    let cell_factor = opt.cell_factor.unwrap_or(nat as f64).max(1.0);

    // Fractional coordinates are fixed once, against the reference cell; the strain carries every
    // subsequent change of shape. `h_inv` is the row-inverse used by `Lattice::fractional`.
    let fractional: Vec<Vec3> = molecule
        .atoms
        .iter()
        .map(|a| reference.frac_of(a.position))
        .collect();

    let nvar = 3 * nat + if opt.relax_cell { free.len() } else { 0 };
    let mut x = vec![0.0; nvar];
    for (a, f) in fractional.iter().enumerate() {
        x[3 * a] = f.x;
        x[3 * a + 1] = f.y;
        x[3 * a + 2] = f.z;
    }

    let build = |x: &[f64]| -> Result<Molecule> {
        let deformed = if opt.relax_cell {
            reference.strained(&strain_from(&x[3 * nat..], &free, cell_factor))?
        } else {
            reference
        };
        let mut m = molecule.clone();
        for a in 0..nat {
            let f = Vec3::new(x[3 * a], x[3 * a + 1], x[3 * a + 2]);
            m.atoms[a].position = deformed.cart_of(f);
        }
        m.cell = Some(deformed);
        Ok(m)
    };

    let evaluate = |x: &[f64]| -> Result<(Molecule, PbcResult, PbcGradient)> {
        let m = build(x)?;
        let scf = run_pbc_scf(&m, params, scf_options)?;
        let grad = pbc_gradient(&m, params, scf_options, &scf)?;
        Ok((m, scf, grad))
    };

    let (mut mol, mut scf, mut grad) = evaluate(&x)?;
    let mut energy = objective(&scf, &mol, opt);
    let mut g = pack_gradient(&mol, &grad, &free, opt, cell_factor, nat);
    let mut max_force = grad.max_gradient;
    let mut max_stress = stress_residual(&grad, &free, opt);

    let mut memory = Lbfgs::new(opt.history);
    let mut trajectory = vec![step_of(&mol, &scf, max_force, max_stress)];
    let mut converged = max_force < opt.gtol && (!opt.relax_cell || max_stress < opt.stress_tol);
    let mut iterations = 0;

    for iter in 0..opt.max_iter {
        iterations = iter + 1;
        if converged {
            break;
        }
        let scale = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let d = memory.direction(&g, scale);
        let g_dot_d: f64 = g.iter().zip(&d).map(|(a, b)| a * b).sum();

        // Backtracking Armijo line search, on the enthalpy when a pressure is set.
        let mut step = 1.0;
        let mut accepted: Option<(Vec<f64>, Molecule, PbcResult, PbcGradient, f64)> = None;
        loop {
            let trial: Vec<f64> = x.iter().zip(&d).map(|(a, b)| a + step * b).collect();
            if let Ok((m, s, gr)) = evaluate(&trial) {
                let e = objective(&s, &m, opt);
                if e <= energy + 1.0e-4 * step * g_dot_d {
                    accepted = Some((trial, m, s, gr, e));
                    break;
                }
            }
            step *= 0.5;
            if step < 1.0e-8 {
                break;
            }
        }
        let Some((x_new, m_new, s_new, gr_new, e_new)) = accepted else {
            // No downhill step exists at this resolution; stop where we are rather than report a
            // structure the line search rejected.
            break;
        };

        let g_new = pack_gradient(&m_new, &gr_new, &free, opt, cell_factor, nat);
        memory.push(
            (0..nvar).map(|i| x_new[i] - x[i]).collect(),
            (0..nvar).map(|i| g_new[i] - g[i]).collect(),
        );

        x = x_new;
        mol = m_new;
        scf = s_new;
        grad = gr_new;
        energy = e_new;
        g = g_new;
        max_force = grad.max_gradient;
        max_stress = stress_residual(&grad, &free, opt);
        converged = max_force < opt.gtol && (!opt.relax_cell || max_stress < opt.stress_tol);
        trajectory.push(step_of(&mol, &scf, max_force, max_stress));
    }

    Ok(PbcOptResult {
        molecule: mol,
        scf,
        gradient: grad,
        converged,
        iterations,
        trajectory,
    })
}

/// The quantity the line search decreases: the energy, or the enthalpy when a pressure is set.
fn objective(scf: &PbcResult, molecule: &Molecule, opt: &PbcOptOptions) -> f64 {
    let mut e = scf.total_ev;
    if opt.relax_cell && opt.pressure != 0.0 {
        if let Some(cell) = molecule.cell {
            e += opt.pressure * cell.measure();
        }
    }
    e
}

fn step_of(molecule: &Molecule, scf: &PbcResult, max_force: f64, max_stress: f64) -> PbcOptStep {
    PbcOptStep {
        energy_ev: scf.total_ev,
        max_force,
        max_stress,
        positions: molecule.atoms.iter().map(|a| a.position).collect(),
        cell: molecule.cell.expect("a periodic step always has a cell"),
    }
}

/// The `(α, β)` strain components that are variables: both axes periodic, upper triangle only.
///
/// Upper triangle because `ε` is symmetric — a strain is a deformation, and its antisymmetric part
/// is a rigid rotation of the whole structure, which changes no energy and would make the search
/// space degenerate.
fn free_strain_components(cell: &Lattice) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for a in 0..3 {
        for b in a..3 {
            if cell.periodic[a] && cell.periodic[b] {
                out.push((a, b));
            }
        }
    }
    out
}

/// Rebuild the symmetric strain tensor from the packed free components.
fn strain_from(packed: &[f64], free: &[(usize, usize)], cell_factor: f64) -> [[f64; 3]; 3] {
    let mut e = [[0.0_f64; 3]; 3];
    for (k, &(a, b)) in free.iter().enumerate() {
        let v = packed[k] / cell_factor;
        e[a][b] = v;
        e[b][a] = v;
    }
    e
}

/// Pack `(∂E/∂f, ∂H/∂ε)` into the flat variable order, with the strain block scaled.
fn pack_gradient(
    molecule: &Molecule,
    grad: &PbcGradient,
    free: &[(usize, usize)],
    opt: &PbcOptOptions,
    cell_factor: f64,
    nat: usize,
) -> Vec<f64> {
    let cell = molecule
        .cell
        .expect("a periodic gradient always has a cell");
    let mut out = vec![0.0; 3 * nat + if opt.relax_cell { free.len() } else { 0 }];
    // ∂E/∂f_a = hᵀ (∂E/∂r_a): row `i` of hᵀ is lattice vector `i`.
    for a in 0..nat {
        let gc = grad.gradient[a];
        for i in 0..3 {
            out[3 * a + i] = cell.cell.col[i].dot(gc);
        }
    }
    if opt.relax_cell {
        let omega = cell.measure();
        for (k, &(alpha, beta)) in free.iter().enumerate() {
            let sigma = component(&grad.stress, alpha, beta);
            let target = if alpha == beta { opt.pressure } else { 0.0 };
            // Off-diagonal components appear twice in the symmetric strain, so the derivative
            // with respect to the single packed variable carries both.
            let multiplicity = if alpha == beta { 1.0 } else { 2.0 };
            out[3 * nat + k] = multiplicity * omega * (sigma + target) / cell_factor;
        }
    }
    out
}

/// The largest free component of `σ + P I` — what `relax_cell` converges on.
fn stress_residual(grad: &PbcGradient, free: &[(usize, usize)], opt: &PbcOptOptions) -> f64 {
    if !opt.relax_cell {
        return 0.0;
    }
    free.iter().fold(0.0_f64, |m, &(a, b)| {
        let target = if a == b { opt.pressure } else { 0.0 };
        m.max((component(&grad.stress, a, b) + target).abs())
    })
}

fn component(m: &Mat3, row: usize, col: usize) -> f64 {
    m.col[col].get(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::kpoints::KMesh;
    use crate::system::Atom;

    const ANG: f64 = 1.0 / 0.529167;

    fn options() -> PbcOptions {
        PbcOptions {
            kmesh: KMesh::MonkhorstPack([1, 1, 4]),
            realspace_cutoff: 60.0,
            exchange_cutoff: Some(20.0),
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            max_scf: 500,
            ..PbcOptions::default()
        }
    }

    /// An H₂ chain with the bond stretched away from AM1's own minimum: the relaxation has to
    /// shorten it and lower the energy, driving the force below the tolerance.
    fn stretched_chain() -> Molecule {
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.85 * ANG),
            },
        ])
        .with_cell(
            Lattice::from_vectors(
                Vec3::new(60.0, 0.0, 0.0),
                Vec3::new(0.0, 60.0, 0.0),
                Vec3::new(0.0, 0.0, 6.0),
                [false, false, true],
            )
            .unwrap(),
        )
    }

    #[test]
    fn positions_relax_under_a_fixed_cell() {
        let params = Am1Parameters::standard().unwrap();
        let start = stretched_chain();
        let result = optimize_periodic(
            &start,
            &params,
            &options(),
            &PbcOptOptions {
                max_iter: 40,
                ..PbcOptOptions::default()
            },
        )
        .unwrap();
        let first = result.trajectory.first().unwrap();
        let last = result.trajectory.last().unwrap();
        let bond = (result.molecule.atoms[1].position - result.molecule.atoms[0].position).norm()
            * 0.529167;
        eprintln!(
            "    {} steps, converged={}, E {:.6} -> {:.6} eV, max|F| {:.2e} -> {:.2e}, r(HH) {:.4} A",
            result.iterations, result.converged, first.energy_ev, last.energy_ev, first.max_force,
            last.max_force, bond
        );
        assert!(result.converged, "the relaxation did not converge");
        assert!(last.energy_ev < first.energy_ev, "the energy went up");
        assert!(last.max_force < 1.0e-3);
        // The cell was fixed, so it must come back byte-identical.
        let cell = result.molecule.cell.unwrap();
        assert_eq!(cell.cell.col[2].z, start.cell.unwrap().cell.col[2].z);
        // AM1's H₂ bond is 0.6766 Å; a chain at 6 Å spacing barely perturbs it.
        assert!((bond - 0.6766).abs() < 0.02, "r(HH) relaxed to {bond} A");
    }

    /// With the cell free, the periodic axis must relax too, and the stress along it must come
    /// down to the tolerance. The two non-periodic axes are frozen by construction — checked,
    /// because straining a vacuum direction changes the reported measure and nothing else.
    #[test]
    fn the_periodic_axis_relaxes_and_the_others_do_not() {
        let params = Am1Parameters::standard().unwrap();
        let start = stretched_chain();
        let result = optimize_periodic(
            &start,
            &params,
            &options(),
            &PbcOptOptions {
                relax_cell: true,
                max_iter: 40,
                stress_tol: 1.0e-5,
                ..PbcOptOptions::default()
            },
        )
        .unwrap();
        let before = start.cell.unwrap();
        let after = result.molecule.cell.unwrap();
        eprintln!(
            "    {} steps, converged={}, c {:.4} -> {:.4} Bohr, max|sigma| {:.2e}",
            result.iterations,
            result.converged,
            before.cell.col[2].z,
            after.cell.col[2].z,
            result.trajectory.last().unwrap().max_stress
        );
        assert_eq!(
            after.cell.col[0].x, before.cell.col[0].x,
            "a non-periodic axis was strained"
        );
        assert_eq!(after.cell.col[1].y, before.cell.col[1].y);
        assert!(
            (after.cell.col[2].z - before.cell.col[2].z).abs() > 1.0e-6,
            "the periodic axis did not move at all"
        );
        assert!(result.trajectory.last().unwrap().max_stress < 1.0e-4);
    }

    /// A molecule without a cell is an error, not a silently molecular optimization.
    #[test]
    fn a_cell_is_required() {
        let params = Am1Parameters::standard().unwrap();
        let mol = Molecule::new(vec![Atom {
            z: 1,
            position: Vec3::zero(),
        }]);
        assert!(optimize_periodic(&mol, &params, &options(), &PbcOptOptions::default()).is_err());
    }
}
