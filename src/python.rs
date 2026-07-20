// SPDX-License-Identifier: GPL-3.0-or-later

//! Python bindings (pyo3), built into the `am1-rs-python` distribution as `am1_rs._native`.
//!
//! Every function here returns **atomic units (Hartree, Bohr)** — the raw native surface.
//! Input coordinates are Ångström (the common Python convention). The eV/Å ASE boundary is
//! applied in the pure-Python `am1_rs.ase` layer.

use crate::gradient::closed_form_gradient;
use crate::optimizer::{optimize as opt_geom, OptOptions};
use crate::params::Am1Parameters;
use crate::scf::{run_am1, Am1Options};
use crate::system::{Atom, Molecule};
use crate::constants::{ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, EV_TO_HARTREE};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py_err(e: crate::error::Am1Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Parse the `reference` string into an [`ScfReference`]. Accepts (case-insensitive) `"auto"`,
/// `"rhf"`/`"r"`/`"restricted"`, and `"uhf"`/`"u"`/`"unrestricted"`.
fn parse_reference(s: &str) -> PyResult<crate::scf::ScfReference> {
    use crate::scf::ScfReference;
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" | "" => Ok(ScfReference::Auto),
        "rhf" | "r" | "restricted" => Ok(ScfReference::Restricted),
        "uhf" | "u" | "unrestricted" => Ok(ScfReference::Unrestricted),
        other => Err(PyValueError::new_err(format!(
            "invalid reference '{other}': expected 'auto', 'rhf', or 'uhf'"
        ))),
    }
}

fn build_molecule(numbers: &[u8], positions: &[Vec<f64>], charge: f64, mult: usize) -> PyResult<Molecule> {
    if numbers.len() != positions.len() {
        return Err(PyValueError::new_err("numbers and positions length mismatch"));
    }
    let mut atoms = Vec::with_capacity(numbers.len());
    for (z, p) in numbers.iter().zip(positions) {
        if p.len() != 3 {
            return Err(PyValueError::new_err("each position must have 3 components"));
        }
        atoms.push(Atom {
            z: *z,
            position: crate::math::Vec3::new(p[0], p[1], p[2]) * ANGSTROM_TO_BOHR,
        });
    }
    Ok(Molecule {
        atoms,
        charge,
        multiplicity: mult.max(1),
    })
}

/// Single-point AM1. Returns a dict in atomic units (energy in Hartree, plus ΔHf in kcal/mol).
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto"))]
fn single_point(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Am1Parameters::standard().map_err(to_py_err)?;
    let opts = Am1Options {
        charge,
        multiplicity,
        reference: parse_reference(reference)?,
        ..Am1Options::default()
    };
    let r = run_am1(&mol, &params, &opts).map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("energy_hartree", r.total_ev * EV_TO_HARTREE)?;
    d.set_item("energy_ev", r.total_ev)?;
    d.set_item("heat_of_formation_kcal", r.heat_of_formation_kcal)?;
    d.set_item("electronic_ev", r.electronic_ev)?;
    d.set_item("core_ev", r.core_ev)?;
    d.set_item("charges", r.charges)?;
    d.set_item("dipole_debye", [r.dipole_debye.x, r.dipole_debye.y, r.dipole_debye.z])?;
    d.set_item("homo_ev", r.homo_ev)?;
    d.set_item("lumo_ev", r.lumo_ev)?;
    d.set_item("converged", r.converged)?;
    Ok(d.into())
}

/// Energy + gradient. Gradient is returned in Hartree/Bohr (atomic units).
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto"))]
fn gradient(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Am1Parameters::standard().map_err(to_py_err)?;
    let opts = Am1Options {
        charge,
        multiplicity,
        reference: parse_reference(reference)?,
        ..Am1Options::default()
    };
    let g = closed_form_gradient(&mol, &params, &opts).map_err(to_py_err)?;
    let grad_au: Vec<[f64; 3]> = g
        .gradient
        .iter()
        .map(|v| [v.x * EV_TO_HARTREE, v.y * EV_TO_HARTREE, v.z * EV_TO_HARTREE])
        .collect();
    // eV/Å = (eV/Bohr) · (Bohr per Å); AM1's own a0 keeps the ASE boundary self-consistent.
    let grad_ev_ang: Vec<[f64; 3]> = g
        .gradient
        .iter()
        .map(|v| {
            [
                v.x * ANGSTROM_TO_BOHR,
                v.y * ANGSTROM_TO_BOHR,
                v.z * ANGSTROM_TO_BOHR,
            ]
        })
        .collect();
    let d = PyDict::new(py);
    d.set_item("energy_hartree", g.energy_ev * EV_TO_HARTREE)?;
    d.set_item("energy_ev", g.energy_ev)?;
    d.set_item("heat_of_formation_kcal", g.scf.heat_of_formation_kcal)?;
    d.set_item("gradient_hartree_per_bohr", grad_au)?;
    d.set_item("gradient_ev_per_angstrom", grad_ev_ang)?;
    Ok(d.into())
}

/// L-BFGS geometry optimization. Returns optimized positions in Ångström.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto"))]
fn optimize(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Am1Parameters::standard().map_err(to_py_err)?;
    let opts = Am1Options {
        charge,
        multiplicity,
        reference: parse_reference(reference)?,
        ..Am1Options::default()
    };
    let res = opt_geom(&mol, &params, &opts, &OptOptions::default()).map_err(to_py_err)?;
    let coords: Vec<[f64; 3]> = res
        .molecule
        .atoms
        .iter()
        .map(|a| {
            let p = a.position * BOHR_TO_ANGSTROM;
            [p.x, p.y, p.z]
        })
        .collect();
    let d = PyDict::new(py);
    d.set_item("positions_angstrom", coords)?;
    d.set_item("energy_hartree", res.scf.total_ev * EV_TO_HARTREE)?;
    d.set_item("heat_of_formation_kcal", res.scf.heat_of_formation_kcal)?;
    d.set_item("converged", res.converged)?;
    d.set_item("iterations", res.iterations)?;
    Ok(d.into())
}

/// AM1-BCC partial charges for AMBER.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0))]
fn am1_bcc(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, 1)?;
    let params = Am1Parameters::standard().map_err(to_py_err)?;
    let opts = Am1Options {
        charge,
        ..Am1Options::default()
    };
    let bcc = crate::bcc::am1_bcc_charges(&mol, &params, &opts).map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("charges", bcc.charges)?;
    d.set_item("mulliken", bcc.mulliken)?;
    d.set_item("atom_types", bcc.atom_types)?;
    Ok(d.into())
}

/// Harmonic vibrational frequencies (cm⁻¹) at the given geometry.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto"))]
fn frequencies(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Am1Parameters::standard().map_err(to_py_err)?;
    let opts = Am1Options {
        charge,
        multiplicity,
        reference: parse_reference(reference)?,
        ..Am1Options::default()
    };
    let vib = crate::hessian::vibrational_analysis(&mol, &params, &opts, 1.0e-3).map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("frequencies_cm", vib.frequencies_cm)?;
    d.set_item("eigenvalues", vib.eigenvalues)?;
    Ok(d.into())
}

/// Analytic (CPHF) Cartesian Hessian at the given geometry.
///
/// Returns the full `3N × 3N` second-derivative matrix in **atomic units (Hartree/Bohr²)** —
/// the native surface's convention — and, for convenience, the same matrix in eV/Å². Row/column
/// index `3·i + k` is atom `i`, Cartesian axis `k` (x, y, z), matching the input atom order.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto"))]
fn hessian(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Am1Parameters::standard().map_err(to_py_err)?;
    let opts = Am1Options {
        charge,
        multiplicity,
        reference: parse_reference(reference)?,
        ..Am1Options::default()
    };
    // Fully analytic (CPHF) Hessian, returned by the core in eV/Bohr².
    let h = crate::hessian::analytic_hessian(&mol, &params, &opts, 1.0e-3).map_err(to_py_err)?;
    let ndof = h.rows;
    // eV/Bohr² → eV/Å²: scale each second derivative by (Bohr per Å)².
    let ang2 = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;
    let mut h_au: Vec<Vec<f64>> = Vec::with_capacity(ndof);
    let mut h_ev_ang: Vec<Vec<f64>> = Vec::with_capacity(ndof);
    for i in 0..ndof {
        let mut row_au = Vec::with_capacity(ndof);
        let mut row_ev = Vec::with_capacity(ndof);
        for j in 0..ndof {
            let v = h[(i, j)]; // eV/Bohr²
            row_au.push(v * EV_TO_HARTREE);
            row_ev.push(v * ang2);
        }
        h_au.push(row_au);
        h_ev_ang.push(row_ev);
    }
    let d = PyDict::new(py);
    d.set_item("hessian_hartree_per_bohr2", h_au)?;
    d.set_item("hessian_ev_per_angstrom2", h_ev_ang)?;
    d.set_item("ndof", ndof)?;
    Ok(d.into())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(single_point, m)?)?;
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(hessian, m)?)?;
    m.add_function(wrap_pyfunction!(am1_bcc, m)?)?;
    Ok(())
}
