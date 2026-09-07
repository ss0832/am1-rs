// SPDX-License-Identifier: GPL-3.0-or-later

//! Python bindings (pyo3), built into the `am1-rs-python` distribution as `am1_rs._native`.
//!
//! Every function here returns **atomic units (Hartree, Bohr)** — the raw native surface.
//! Input coordinates are Ångström (the common Python convention). The eV/Å ASE boundary is
//! applied in the pure-Python `am1_rs.ase` layer.

// A `#[pyfunction]`'s parameter list **is** the Python signature: callers pass these by keyword,
// and the defaults in `#[pyo3(signature = ...)]` are part of the published API. Gathering them
// into a struct — the usual remedy for this lint, and the one applied to the Rust-side
// `CoreBuildOptions` — would replace a documented keyword surface with an opaque object that
// Python callers would then have to construct. Allowed once here, with the reason, rather than
// repeated on a dozen functions.
#![allow(clippy::too_many_arguments)]

use crate::constants::{ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, EV_TO_HARTREE};
use crate::gradient::closed_form_gradient;
use crate::optimizer::{optimize as opt_geom, OptOptions};
use crate::params::Am1Parameters;
use crate::scf::{run_am1, Am1Options};
use crate::system::{Atom, Molecule};
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

/// Resolve the parameter set named by `method` (`"am1"`, `"rm1"`, …).
///
/// The method travels *with* the parameters rather than in [`Am1Options`], so every downstream
/// branch (core–core corrections, element coverage, the reported method name) follows from one
/// argument and there is no way to pair RM1 parameters with an AM1 code path.
fn params_for(method: &str) -> PyResult<Am1Parameters> {
    let m = crate::method::NddoMethod::parse(method).map_err(to_py_err)?;
    Am1Parameters::for_method(m).map_err(to_py_err)
}

/// Parse an optional external field given in **atomic units** (Hartree per e·Bohr) into the
/// crate's internal eV per e·Bohr.
///
/// Atomic units because that is this module's convention throughout; the ASE layer takes V/Å and
/// converts at its own boundary. Doing the conversion here, once, is what stops the two surfaces
/// from disagreeing about what "field = 0.01" means.
fn field_from(field: Option<Vec<f64>>) -> PyResult<Option<crate::math::Vec3>> {
    let Some(f) = field else { return Ok(None) };
    if f.len() != 3 {
        return Err(PyValueError::new_err(
            "electric_field must have three components",
        ));
    }
    Ok(Some(
        crate::math::Vec3::new(f[0], f[1], f[2]) * crate::constants::HARTREE_TO_EV,
    ))
}

/// The molecular options every entry point builds the same way.
fn molecular_options(
    charge: f64,
    multiplicity: usize,
    reference: &str,
    field: Option<Vec<f64>>,
) -> PyResult<Am1Options> {
    molecular_options_with(charge, multiplicity, reference, field, None, None)
}

/// [`molecular_options`] with the response solver's limits, for the entry points that run one.
///
/// `None` keeps [`Am1Options`]'s defaults. Exposed because through 0.2.2 these were private
/// constants: a molecule whose CPHF needed more than a hundred passes could not be given them from
/// either API.
fn molecular_options_with(
    charge: f64,
    multiplicity: usize,
    reference: &str,
    field: Option<Vec<f64>>,
    cphf_max_iter: Option<usize>,
    cphf_tol: Option<f64>,
) -> PyResult<Am1Options> {
    let base = Am1Options::default();
    Ok(Am1Options {
        charge,
        multiplicity,
        reference: parse_reference(reference)?,
        electric_field: field_from(field)?,
        cphf_max_iter: cphf_max_iter.unwrap_or(base.cphf_max_iter),
        cphf_tol: cphf_tol.unwrap_or(base.cphf_tol),
        ..base
    })
}

/// Put a `Matrix` into a Python list of rows.
fn matrix_rows(m: &crate::linalg::Matrix) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|i| (0..m.cols).map(|j| m[(i, j)]).collect())
        .collect()
}

fn build_molecule(
    numbers: &[u8],
    positions: &[Vec<f64>],
    charge: f64,
    mult: usize,
) -> PyResult<Molecule> {
    if numbers.len() != positions.len() {
        return Err(PyValueError::new_err(
            "numbers and positions length mismatch",
        ));
    }
    let mut atoms = Vec::with_capacity(numbers.len());
    for (z, p) in numbers.iter().zip(positions) {
        if p.len() != 3 {
            return Err(PyValueError::new_err(
                "each position must have 3 components",
            ));
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
        cell: None,
    })
}

/// Single-point AM1. Returns a dict in atomic units (energy in Hartree, plus ΔHf in kcal/mol).
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1", electric_field=None))]
fn single_point(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options(charge, multiplicity, reference, electric_field)?;
    let r = py
        .allow_threads(|| run_am1(&mol, &params, &opts))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("method", params.method.name())?;
    d.set_item("energy_hartree", r.total_ev * EV_TO_HARTREE)?;
    d.set_item("energy_ev", r.total_ev)?;
    d.set_item("heat_of_formation_kcal", r.heat_of_formation_kcal)?;
    d.set_item("electronic_ev", r.electronic_ev)?;
    d.set_item("core_ev", r.core_ev)?;
    d.set_item("charges", r.charges)?;
    d.set_item(
        "dipole_debye",
        [r.dipole_debye.x, r.dipole_debye.y, r.dipole_debye.z],
    )?;
    // The magnitude as well as the vector. `Am1Result` carries both, and every consumer that
    // wants a single dipole number would otherwise recompute the norm.
    d.set_item("dipole_magnitude", r.dipole_magnitude)?;
    d.set_item("homo_ev", r.homo_ev)?;
    d.set_item("lumo_ev", r.lumo_ev)?;
    d.set_item("homo_beta_ev", r.homo_beta_ev)?;
    d.set_item("lumo_beta_ev", r.lumo_beta_ev)?;
    // The nuclear half of `−μ·F`; the electronic half is already inside `electronic_ev`. Zero
    // without a field.
    d.set_item("field_nuclear_ev", r.field_nuclear_ev)?;
    d.set_item("converged", r.converged)?;
    // `iterations` and `unrestricted` were on the periodic results but not the molecular ones,
    // so a caller could report how an SCF went for a crystal and not for a molecule.
    d.set_item("iterations", r.iterations)?;
    d.set_item("unrestricted", r.unrestricted)?;
    Ok(d.into())
}

/// Energy + gradient. Gradient is returned in Hartree/Bohr (atomic units).
///
/// The SCF properties (charges, dipole) are returned alongside because the gradient already
/// converged an SCF to obtain them. A caller that wants energy, forces *and* charges — the ASE
/// calculator does, on every step of a molecular dynamics run — would otherwise have to call
/// `single_point` as well and pay for a second SCF at the same geometry.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1", electric_field=None))]
fn gradient(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options(charge, multiplicity, reference, electric_field)?;
    let g = py
        .allow_threads(|| closed_form_gradient(&mol, &params, &opts))
        .map_err(to_py_err)?;
    let grad_au: Vec<[f64; 3]> = g
        .gradient
        .iter()
        .map(|v| {
            [
                v.x * EV_TO_HARTREE,
                v.y * EV_TO_HARTREE,
                v.z * EV_TO_HARTREE,
            ]
        })
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
    // The same breakdown `single_point` reports. A gradient run has already done the SCF, so
    // withholding these only forces a caller who wants them to run the SCF a second time.
    d.set_item("electronic_ev", g.scf.electronic_ev)?;
    d.set_item("core_ev", g.scf.core_ev)?;
    d.set_item("homo_ev", g.scf.homo_ev)?;
    d.set_item("lumo_ev", g.scf.lumo_ev)?;
    d.set_item("gradient_hartree_per_bohr", grad_au)?;
    d.set_item("gradient_ev_per_angstrom", grad_ev_ang)?;
    d.set_item("charges", g.scf.charges.clone())?;
    d.set_item(
        "dipole_debye",
        [
            g.scf.dipole_debye.x,
            g.scf.dipole_debye.y,
            g.scf.dipole_debye.z,
        ],
    )?;
    d.set_item("dipole_magnitude", g.scf.dipole_magnitude)?;
    // In the same units as `gradient_hartree_per_bohr`, so the two can be compared directly;
    // `GradientResult::max_gradient` is in eV/Bohr like the rest of the Rust interior.
    d.set_item(
        "max_gradient_hartree_per_bohr",
        g.max_gradient * EV_TO_HARTREE,
    )?;
    d.set_item("converged", g.scf.converged)?;
    d.set_item("iterations", g.scf.iterations)?;
    d.set_item("unrestricted", g.scf.unrestricted)?;
    d.set_item("method", params.method.name())?;
    Ok(d.into())
}

/// Periodic single point with forces and stress.
///
/// `cell` is three lattice vectors in Ångström; `pbc` is three booleans, so one entry point
/// covers a chain, a slab and a crystal. `kpts` is the Monkhorst–Pack mesh.
///
/// Returns eV / Å units directly, unlike the molecular functions above: a periodic result is
/// consumed by ASE, and converting a stress tensor twice is an easy way to be wrong.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc,
    kpts=(1, 1, 1), charge=0.0, multiplicity=1, unrestricted=false,
    smearing_ev=0.0, realspace_cutoff=40.0, exchange_cutoff=20.0, method="am1",
    e_tol=1.0e-8, p_tol=1.0e-7, max_scf=300, mixing=0.3, electric_field=None
))]
fn pbc_point(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    unrestricted: bool,
    smearing_ev: f64,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    method: &str,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    mixing: f64,
    electric_field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    use crate::pbc::{pbc_energy_and_gradient, KMesh, PbcOptions};

    if cell.len() != 3 || cell.iter().any(|v| v.len() != 3) {
        return Err(PyValueError::new_err("cell must be three 3-vectors"));
    }
    if pbc.len() != 3 {
        return Err(PyValueError::new_err("pbc must have three entries"));
    }
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let v = |r: &Vec<f64>| crate::math::Vec3::new(r[0], r[1], r[2]) * ANGSTROM_TO_BOHR;
    let lattice = crate::lattice::Lattice::from_vectors(
        v(&cell[0]),
        v(&cell[1]),
        v(&cell[2]),
        [pbc[0], pbc[1], pbc[2]],
    )
    .map_err(to_py_err)?;
    let n_periodic = lattice.n_periodic();
    mol.cell = Some(lattice);

    let params = params_for(method)?;
    let opts = PbcOptions {
        kmesh: KMesh::MonkhorstPack([kpts.0.max(1), kpts.1.max(1), kpts.2.max(1)]),
        realspace_cutoff,
        exchange_cutoff: Some(exchange_cutoff),
        smearing_ev,
        charge,
        multiplicity: multiplicity.max(1),
        unrestricted,
        // Must be orthogonal to every lattice vector; a component along a periodic direction is
        // rejected by the SCF with a message naming it. See `PbcOptions::electric_field`.
        electric_field: field_from(electric_field)?,
        e_tol,
        p_tol,
        max_scf,
        mixing,
        ..PbcOptions::default()
    };

    let (scf, grad) = py
        .allow_threads(|| pbc_energy_and_gradient(&mol, &params, &opts))
        .map_err(to_py_err)?;
    if !scf.converged {
        return Err(PyValueError::new_err(format!(
            "periodic SCF did not converge in {} iterations",
            scf.iterations
        )));
    }

    // Forces in eV/Å. `gradient` is eV/Bohr, so multiply by Bohr per Å.
    let forces: Vec<[f64; 3]> = grad
        .forces
        .iter()
        .map(|v| {
            [
                v.x * ANGSTROM_TO_BOHR,
                v.y * ANGSTROM_TO_BOHR,
                v.z * ANGSTROM_TO_BOHR,
            ]
        })
        .collect();

    // Stress in eV/Å^d, from eV/Bohr^d. The numerator is a strain derivative of an energy —
    // strain is dimensionless, so it is just eV — and the denominator is the periodic measure,
    // a volume, an area or a length in Bohr^d. So the whole quantity is eV/Bohr^d and the
    // conversion is (Bohr per Å)^d, with no stray factor from the pair separation: that length
    // is already inside the virial's own eV.
    let stress_scale = ANGSTROM_TO_BOHR.powi(n_periodic as i32);
    let s = &grad.stress;
    let voigt: Vec<f64> = grad
        .stress_voigt()
        .iter()
        .map(|v| v * stress_scale)
        .collect();
    let stress_matrix: Vec<[f64; 3]> = (0..3)
        .map(|row| {
            [
                component_of(&s.col[0], row) * stress_scale,
                component_of(&s.col[1], row) * stress_scale,
                component_of(&s.col[2], row) * stress_scale,
            ]
        })
        .collect();

    let d = PyDict::new(py);
    d.set_item("energy_ev", scf.total_ev)?;
    d.set_item("energy_hartree", scf.total_ev * EV_TO_HARTREE)?;
    // The two halves of the total. The molecular `single_point` has reported them since 0.1;
    // this did not, so anything wanting the same breakdown for a cell — the CLI's periodic
    // `energy` mode, for one — had no way to get it and raised `KeyError` reaching for it.
    d.set_item("electronic_ev", scf.electronic_ev)?;
    d.set_item("core_ev", scf.core_ev)?;
    d.set_item("band_energy_ev", scf.band_energy_ev)?;
    d.set_item("converged", scf.converged)?;
    d.set_item("free_energy_ev", scf.free_energy_ev())?;
    d.set_item("forces_ev_per_angstrom", forces)?;
    // The same number the Rust CLI prints, rather than one recomputed from the eV/Å forces: the
    // two differ in the last bit after the unit round trip, which is invisible everywhere except
    // in a `{:.6e}` field, where it made the two front ends disagree about `max |grad|`.
    d.set_item("max_gradient_ev_per_bohr", grad.max_gradient)?;
    d.set_item("stress_voigt", voigt)?;
    d.set_item("stress_matrix", stress_matrix)?;
    // The same stress in the crate's own eV/Bohr^d, unconverted. The `stress_voigt` above is in
    // eV/Å^d because that is what ASE's `results["stress"]` means; a caller working in atomic
    // units — the CLI, for one — would otherwise have to redo the `(Bohr per Å)^d` conversion,
    // and `d` is the number of *periodic* directions rather than 3. Doing that arithmetic twice
    // in two languages is how the two front ends came to print stresses differing by exactly
    // 1.8898 for a chain.
    d.set_item("stress_voigt_ev_per_bohr", grad.stress_voigt())?;
    d.set_item("charges", scf.charges.clone())?;
    d.set_item("fermi_energy_ev", scf.fermi_energy_ev)?;
    d.set_item("entropy_ev", scf.entropy_ev)?;
    d.set_item("k_points", scf.k_points)?;
    d.set_item("iterations", scf.iterations)?;
    d.set_item("n_periodic", n_periodic)?;
    d.set_item("max_image_overlap", scf.max_image_overlap)?;
    d.set_item("unrestricted", scf.unrestricted)?;
    d.set_item("charged_cell_warning", scf.charged_cell_warning.clone())?;
    Ok(d.into())
}

#[inline]
fn component_of(v: &crate::math::Vec3, index: usize) -> f64 {
    match index {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Divide-and-conquer single point with forces.
///
/// Returns eV / Å, like `pbc_point` and unlike the molecular functions: this is a large-system
/// entry point and its consumer is the ASE calculator.
///
/// The returned dict carries the scaling counters as well as the energy, because the whole
/// point of the method is a claim about cost and that claim should be inspectable from the
/// result rather than taken on trust.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions,
    charge=0.0, multiplicity=1, reference="auto", method="am1",
    // Kept in step with DcOptions::default() by tests/dc_optimize.rs, which asserts that the
    // two agree -- three front ends resolving to three different partitions is what this file
    // used to do.
    core_size=12, buffer_radius=11.0, smearing_ev=0.05,
    e_tol=1.0e-7, p_tol=1.0e-6, max_scf=300, mixing=0.4, gap_warn_ev=0.5,
    forces=true, multipole_cutoff=None, electric_field=None,
    cell=None, pbc=None, realspace_cutoff=40.0, exchange_cutoff=20.0,
    optimize=false, opt_max_iter=200, opt_gtol=1.0e-3
))]
fn divide_conquer(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    core_size: usize,
    buffer_radius: f64,
    smearing_ev: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    mixing: f64,
    gap_warn_ev: f64,
    forces: bool,
    multipole_cutoff: Option<f64>,
    electric_field: Option<Vec<f64>>,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    optimize: bool,
    opt_max_iter: usize,
    opt_gtol: f64,
) -> PyResult<PyObject> {
    use crate::divide_conquer::{
        divide_conquer_gradient, optimize_divide_conquer, run_divide_conquer, DcOptions,
    };
    use crate::fermi::Filling;

    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let mut opts = Am1Options {
        multipole_cutoff,
        ..molecular_options(charge, multiplicity, reference, electric_field)?
    };

    // Periodic divide-and-conquer. The subsystem buffers are built from the image-aware pair
    // list, so a cell needs the same real-space and exchange cutoffs the periodic SCF uses --
    // at Γ the two-centre exchange decays only as `1/R` and the image sum needs the taper.
    if cell.is_some() != pbc.is_some() {
        return Err(PyValueError::new_err(
            "cell and pbc must be given together: a lattice without a periodicity flag, or the \
             reverse, is ambiguous",
        ));
    }
    if let (Some(cell), Some(pbc)) = (cell, pbc) {
        if cell.len() != 3 || cell.iter().any(|v| v.len() != 3) {
            return Err(PyValueError::new_err("cell must be three 3-vectors"));
        }
        if pbc.len() != 3 {
            return Err(PyValueError::new_err("pbc must have three entries"));
        }
        let v = |r: &Vec<f64>| crate::math::Vec3::new(r[0], r[1], r[2]) * ANGSTROM_TO_BOHR;
        mol.cell = Some(
            crate::lattice::Lattice::from_vectors(
                v(&cell[0]),
                v(&cell[1]),
                v(&cell[2]),
                [pbc[0], pbc[1], pbc[2]],
            )
            .map_err(to_py_err)?,
        );
        opts.realspace_cutoff = realspace_cutoff;
        opts.exchange_cutoff = Some(exchange_cutoff);
    }
    let dc_opts = DcOptions {
        core_size,
        buffer_radius,
        filling: if smearing_ev > 0.0 {
            Filling::Fermi { kt: smearing_ev }
        } else {
            Filling::Aufbau
        },
        max_scf,
        e_tol,
        p_tol,
        mixing,
        gap_warn_ev,
    };

    // `optimize` is a flag here rather than a second `dc_optimize` function, unlike the periodic
    // pair. A periodic relaxation takes cell and pressure arguments a single point has no use
    // for and returns a stress a single point does not have; a divide-and-conquer relaxation
    // takes two extra numbers and returns the same dictionary plus the geometry it ended at. One
    // entry point keeps every divide-and-conquer control in one signature.
    let (mol, r, gradient, opt_state) = py
        .allow_threads(|| -> crate::error::Result<_> {
            let (mol, opt_state) = if optimize {
                let opt = crate::optimizer::OptOptions {
                    max_iter: opt_max_iter,
                    gtol: opt_gtol,
                    ..Default::default()
                };
                let res = optimize_divide_conquer(&mol, &params, &opts, &dc_opts, &opt)?;
                let max_force = res.trajectory.last().map(|s| s.max_gradient).unwrap_or(0.0);
                (
                    res.molecule,
                    Some((res.converged, res.iterations, max_force)),
                )
            } else {
                (mol, None)
            };
            let r = run_divide_conquer(&mol, &params, &opts, &dc_opts)?;
            let g = if forces {
                Some(divide_conquer_gradient(&mol, &params, &opts, &r)?)
            } else {
                None
            };
            Ok((mol, r, g, opt_state))
        })
        .map_err(to_py_err)?;

    if !r.converged {
        return Err(PyValueError::new_err(format!(
            "divide-and-conquer SCF did not converge in {} iterations",
            r.iterations
        )));
    }

    let d = PyDict::new(py);
    d.set_item("energy_ev", r.total_ev)?;
    d.set_item("energy_hartree", r.total_ev * EV_TO_HARTREE)?;
    d.set_item("free_energy_ev", r.free_energy_ev())?;
    d.set_item("heat_of_formation_kcal", r.heat_of_formation_kcal)?;
    // The same split the molecular and periodic single points report. Absent through 0.2.2,
    // which is the shape of gap that made the periodic CLI die on a missing `electronic_ev`.
    d.set_item("electronic_ev", r.electronic_ev)?;
    d.set_item("core_ev", r.core_ev)?;
    d.set_item("converged", r.converged)?;
    d.set_item(
        "positions_angstrom",
        mol.atoms
            .iter()
            .map(|a| {
                [
                    a.position.x * BOHR_TO_ANGSTROM,
                    a.position.y * BOHR_TO_ANGSTROM,
                    a.position.z * BOHR_TO_ANGSTROM,
                ]
            })
            .collect::<Vec<_>>(),
    )?;
    if let Some((converged, steps, max_force)) = opt_state {
        d.set_item("opt_converged", converged)?;
        d.set_item("opt_iterations", steps)?;
        d.set_item("max_force_ev_per_bohr", max_force)?;
    }
    if let Some(g) = gradient {
        // eV/Bohr -> eV/Å; forces are the negative gradient.
        let f: Vec<[f64; 3]> = g
            .iter()
            .map(|v| {
                [
                    -v.x * ANGSTROM_TO_BOHR,
                    -v.y * ANGSTROM_TO_BOHR,
                    -v.z * ANGSTROM_TO_BOHR,
                ]
            })
            .collect();
        d.set_item("forces_ev_per_angstrom", f)?;
    }
    d.set_item("charges", r.charges.clone())?;
    d.set_item("fermi_energy_ev", r.fermi_energy_ev)?;
    d.set_item("fermi_energies_ev", r.fermi_energies_ev.clone())?;
    d.set_item("entropy_ev", r.entropy_ev)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("unrestricted", r.unrestricted)?;
    d.set_item("homo_lumo_gap_ev", r.homo_lumo_gap_ev)?;
    d.set_item("small_gap_warning", r.small_gap_warning.clone())?;
    d.set_item("subsystems", r.subsystems)?;
    d.set_item("largest_subsystem_aos", r.largest_subsystem_aos)?;
    d.set_item("diagonalization_work", r.diagonalization_work)?;
    d.set_item("coulomb_work", r.coulomb_work)?;
    d.set_item("exchange_work", r.exchange_work)?;
    d.set_item("retained_density_blocks", r.retained_density_blocks)?;
    // The DIIS history's memory, and what it would have been dense. The first is linear in the
    // atom count and the second quadratic; returning both is what makes that checkable.
    d.set_item("diis_pattern_elements", r.diis_pattern_elements)?;
    d.set_item("dense_triangle_elements", r.dense_triangle_elements)?;
    d.set_item("method", params.method.name())?;
    Ok(d.into())
}

/// Phonons of a periodic system: force constants, and frequencies along a path of `q` points.
///
/// `supercell` sets how far the real-space force constants `Φ(T)` are resolved before
/// truncation, and it is the convergence knob: a larger supercell that gives the same answer is
/// the only evidence the smaller one was enough. It also sets the k-sampling of the electronic
/// structure, because Γ on an `n`-fold supercell is the primitive cell at `n` k-points.
///
/// `q_points` are **fractional** coordinates of the primitive reciprocal lattice. Omit them to
/// get Γ only.
///
/// **No LO–TO splitting here.** A truncated real-space `Φ(T)` cannot carry the dipole–dipole
/// tail, so longitudinal and transverse optical branches come out degenerate as `q → 0`; in a
/// polar crystal they are not. Use [`lo_to_frequencies`], which adds the non-analytic term from
/// `born_charges` and `dielectric` — both of which this version *does* compute. See
/// `docs/pbc.md`.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, supercell,
    q_points=None, charge=0.0, multiplicity=1, method="am1",
    realspace_cutoff=40.0, exchange_cutoff=20.0,
    e_tol=1.0e-10, p_tol=1.0e-9, max_scf=500, enforce_acoustic_sum_rule=true,
    eigenvectors=false
))]
fn phonons(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    supercell: (usize, usize, usize),
    q_points: Option<Vec<Vec<f64>>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    enforce_acoustic_sum_rule: bool,
    eigenvectors: bool,
) -> PyResult<PyObject> {
    use crate::pbc::kpoints::KPoint;
    use crate::pbc::phonon::ForceConstants;

    if cell.len() != 3 || cell.iter().any(|v| v.len() != 3) {
        return Err(PyValueError::new_err("cell must be three 3-vectors"));
    }
    if pbc.len() != 3 {
        return Err(PyValueError::new_err("pbc must have three entries"));
    }
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let v = |r: &Vec<f64>| crate::math::Vec3::new(r[0], r[1], r[2]) * ANGSTROM_TO_BOHR;
    mol.cell = Some(
        crate::lattice::Lattice::from_vectors(
            v(&cell[0]),
            v(&cell[1]),
            v(&cell[2]),
            [pbc[0], pbc[1], pbc[2]],
        )
        .map_err(to_py_err)?,
    );

    let params = params_for(method)?;
    let opts = Am1Options {
        charge,
        multiplicity,
        realspace_cutoff,
        exchange_cutoff: Some(exchange_cutoff),
        e_tol,
        p_tol,
        max_scf,
        ..Am1Options::default()
    };
    let repeats = [supercell.0.max(1), supercell.1.max(1), supercell.2.max(1)];

    let (fc, asr_before, transpose_before) = py
        .allow_threads(|| -> crate::error::Result<_> {
            let mut fc = ForceConstants::from_supercell(&mol, &params, &opts, repeats)?;
            let before = fc.acoustic_sum_rule_error();
            // The *other* sum rule. Reported because the two are independent and only the pair
            // of them puts three modes at exactly zero; see `enforce_acoustic_sum_rule`.
            let transpose = fc.transpose_asymmetry();
            if enforce_acoustic_sum_rule {
                fc.enforce_acoustic_sum_rule();
            }
            Ok((fc, before, transpose))
        })
        .map_err(to_py_err)?;

    let requested: Vec<KPoint> = match q_points {
        Some(qs) => qs
            .iter()
            .map(|q| {
                if q.len() != 3 {
                    return Err(PyValueError::new_err("each q point needs three components"));
                }
                Ok(KPoint {
                    fractional: [q[0], q[1], q[2]],
                    weight: 1.0,
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        None => vec![KPoint {
            fractional: [0.0, 0.0, 0.0],
            weight: 1.0,
        }],
    };

    // One diagonalization per `q` either way; `eigenvectors` decides only what is kept from it.
    let solved: Vec<crate::pbc::phonon::PhononModes> = requested
        .iter()
        .map(|q| fc.modes(*q))
        .collect::<crate::error::Result<Vec<_>>>()
        .map_err(to_py_err)?;
    let bands: Vec<Vec<f64>> = solved.iter().map(|m| m.frequencies_cm.clone()).collect();

    let d = PyDict::new(py);
    d.set_item(
        "q_points",
        requested.iter().map(|q| q.fractional).collect::<Vec<_>>(),
    )?;
    d.set_item("frequencies_cm", bands)?;
    d.set_item("supercell", repeats)?;
    d.set_item(
        "commensurate_q",
        fc.commensurate_q()
            .iter()
            .map(|q| q.fractional)
            .collect::<Vec<_>>(),
    )?;
    d.set_item("acoustic_sum_rule_error_before", asr_before)?;
    d.set_item("acoustic_sum_rule_error", fc.acoustic_sum_rule_error())?;
    d.set_item("transpose_asymmetry_before", transpose_before)?;
    d.set_item("transpose_asymmetry", fc.transpose_asymmetry())?;
    if eigenvectors {
        // Real and imaginary parts as separate arrays rather than interleaved pairs: this is what
        // `numpy.asarray(re) + 1j*numpy.asarray(im)` wants, and the consumer of a phonon
        // eigenvector is almost always numpy.
        //
        // `O(n_q · (3N)²)` complex numbers, which is why it is opt-in: a band structure over a
        // few hundred `q` is a large array, and most callers want the frequencies.
        let pack = |pick: fn(&crate::pbc::phonon::PhononModes) -> &crate::pbc::complex::CMatrix,
                    part: bool| {
            solved
                .iter()
                .map(|m| {
                    let c = pick(m);
                    matrix_rows(if part { &c.im } else { &c.re })
                })
                .collect::<Vec<_>>()
        };
        d.set_item("polarization_re", pack(|m| &m.polarization, false))?;
        d.set_item("polarization_im", pack(|m| &m.polarization, true))?;
        d.set_item("displacements_re", pack(|m| &m.displacements, false))?;
        d.set_item("displacements_im", pack(|m| &m.displacements, true))?;
    }
    d.set_item("method", params.method.name())?;
    Ok(d.into())
}

/// Build a periodic molecule and the matching options, shared by every periodic response entry
/// point below so they cannot drift apart in how they read a cell.
/// Phonon frequencies with the **LO–TO splitting** restored, from `Z*` and `ε_∞`.
///
/// **Three-dimensional cells only**, because the non-analytic term `4π(q·Z*)²/(Ω q·ε_∞·q)` is
/// the 3D one and `Ω` must be a volume.
///
/// This is the *supercell* route plus its missing long-range piece. A truncated real-space
/// `Φ(T)` structurally cannot carry the dipole–dipole tail, so `phonons` alone gives the
/// transverse branches at Γ and misses the longitudinal shift; this adds it analytically. It
/// therefore needs `direction`, the unit vector along which `q → 0` is taken, because the limit
/// is direction dependent — that *is* the physics of LO–TO.
///
/// Do **not** compose this with [`dfpt`]: `dfpt` already carries the long-range monopole channel
/// inside `D(q)`, so applying both counts it twice. Use `dfpt` or this, not both.
///
/// Returns `frequencies_cm` (with splitting), `frequencies_cm_no_lo_to` (without, i.e. what
/// `phonons` gives), `born_charges` and `dielectric`, so the size of the shift is visible.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, supercell=(2, 2, 2), direction=(1.0, 0.0, 0.0),
    q_points=None, kpts=(2, 2, 2), charge=0.0, multiplicity=1, method="am1",
    realspace_cutoff=40.0, exchange_cutoff=20.0, e_tol=1.0e-11, p_tol=1.0e-10, max_scf=500,
    enforce_acoustic_sum_rule=true, allow_fractional_occupations=false
))]
fn lo_to_frequencies(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    supercell: (usize, usize, usize),
    direction: (f64, f64, f64),
    q_points: Option<Vec<Vec<f64>>>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    enforce_acoustic_sum_rule: bool,
    allow_fractional_occupations: bool,
) -> PyResult<PyObject> {
    use crate::pbc::kpoints::KPoint;
    use crate::pbc::phonon::ForceConstants;

    let (mol, params, pbc_opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        multiplicity,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        allow_fractional_occupations,
    )?;
    let dir = crate::math::Vec3::new(direction.0, direction.1, direction.2);
    if dir.norm() < 1.0e-12 {
        return Err(PyValueError::new_err(
            "direction must be a non-zero vector: the q -> 0 limit of the non-analytic term is \
             direction dependent, which is what LO-TO splitting is",
        ));
    }
    let lattice = mol
        .cell
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("a lattice is required"))?;
    let measure = lattice.measure();

    let molecular = Am1Options {
        charge,
        multiplicity,
        realspace_cutoff,
        exchange_cutoff: Some(exchange_cutoff),
        e_tol,
        p_tol,
        max_scf,
        ..Am1Options::default()
    };
    let repeats = [supercell.0.max(1), supercell.1.max(1), supercell.2.max(1)];

    let requested: Vec<KPoint> = match q_points {
        Some(qs) => qs
            .iter()
            .map(|q| {
                if q.len() != 3 {
                    return Err(PyValueError::new_err("each q point needs three components"));
                }
                Ok(KPoint {
                    fractional: [q[0], q[1], q[2]],
                    weight: 1.0,
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        None => vec![KPoint {
            fractional: [0.0, 0.0, 0.0],
            weight: 1.0,
        }],
    };

    type LoTo = (
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<[[f64; 3]; 3]>,
        [[f64; 3]; 3],
    );
    let (with, without, z, eps) = py
        .allow_threads(|| -> crate::error::Result<LoTo> {
            let mut fc = ForceConstants::from_supercell(&mol, &params, &molecular, repeats)?;
            if enforce_acoustic_sum_rule {
                fc.enforce_acoustic_sum_rule();
            }
            let z = crate::pbc::born_charges(&mol, &params, &pbc_opts)?;
            // `(alpha, epsilon)`; the non-analytic term needs the dielectric tensor.
            let eps = crate::pbc::dielectric_tensor(&mol, &params, &pbc_opts)?.1;
            let mut with = Vec::with_capacity(requested.len());
            let mut without = Vec::with_capacity(requested.len());
            for q in &requested {
                with.push(fc.frequencies_with_lo_to(*q, dir, &z, &eps, measure)?);
                without.push(fc.frequencies(*q)?);
            }
            Ok((with, without, z, eps))
        })
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item(
        "q_points",
        requested.iter().map(|q| q.fractional).collect::<Vec<_>>(),
    )?;
    d.set_item("frequencies_cm", with)?;
    d.set_item("frequencies_cm_no_lo_to", without)?;
    d.set_item("born_charges", z)?;
    d.set_item("dielectric", eps)?;
    d.set_item("direction", [dir.x, dir.y, dir.z])?;
    d.set_item("supercell", repeats)?;
    d.set_item("method", params.method.name())?;
    Ok(d.into())
}

/// Build the periodic molecule and options every response entry point shares.
///
/// `smearing_ev` is a parameter rather than a hardcoded zero because the ground state a response
/// is built on has to converge before the response means anything, and a small-gap or coarsely
/// sampled system does not converge under sharp aufbau filling. Measured on zincblende AlP: a
/// 2×2×2 mesh does not converge at all with `smearing_ev = 0` and takes 248 iterations at 0.2 eV.
/// Every entry point here passed 0 until 0.2.3, so DFPT simply could not be run on such a system
/// — which is most inorganic solids.
///
/// **What smearing does and does not fix.** It converges the *ground state*. The coupled-perturbed
/// equations solved on top of it assume integer occupations, so a response computed at a genuinely
/// fractional occupancy is outside what this code derives; use it to reach a converged insulating
/// state on a coarse mesh, not to describe a metal.
#[allow(clippy::too_many_arguments)]
fn periodic_setup(
    numbers: &[u8],
    positions: &[Vec<f64>],
    cell: &[Vec<f64>],
    pbc: &[bool],
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    smearing_ev: f64,
    allow_fractional_occupations: bool,
) -> PyResult<(Molecule, Am1Parameters, crate::pbc::PbcOptions)> {
    use crate::pbc::{KMesh, PbcOptions};
    if cell.len() != 3 || cell.iter().any(|v| v.len() != 3) {
        return Err(PyValueError::new_err("cell must be three 3-vectors"));
    }
    if pbc.len() != 3 {
        return Err(PyValueError::new_err("pbc must have three entries"));
    }
    let mut mol = build_molecule(numbers, positions, charge, multiplicity)?;
    let v = |r: &Vec<f64>| crate::math::Vec3::new(r[0], r[1], r[2]) * ANGSTROM_TO_BOHR;
    mol.cell = Some(
        crate::lattice::Lattice::from_vectors(
            v(&cell[0]),
            v(&cell[1]),
            v(&cell[2]),
            [pbc[0], pbc[1], pbc[2]],
        )
        .map_err(to_py_err)?,
    );
    let params = params_for(method)?;
    let opts = PbcOptions {
        kmesh: KMesh::MonkhorstPack([kpts.0.max(1), kpts.1.max(1), kpts.2.max(1)]),
        // A response must not fold `k` with `−k`: exact for the ground state, wrong for a
        // `q`-point response, and the two share this options struct.
        fold_time_reversal: false,
        realspace_cutoff,
        exchange_cutoff: Some(exchange_cutoff),
        smearing_ev,
        charge,
        multiplicity: multiplicity.max(1),
        e_tol,
        p_tol,
        max_scf,
        require_integer_occupations: !allow_fractional_occupations,
        ..PbcOptions::default()
    };
    Ok((mol, params, opts))
}

/// Analytic force constants at `q = 0` **with k-point sampling** — the periodic Hessian.
///
/// Unlike the Γ-only path this does not need the exchange taper to stand in for a decaying
/// density matrix: sampling `k` makes `P(0,T)` decay on its own. See `docs/pbc.md`.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, kpts=(2, 2, 2), charge=0.0, multiplicity=1, method="am1",
    realspace_cutoff=40.0, exchange_cutoff=20.0, e_tol=1.0e-11, p_tol=1.0e-10, max_scf=500, allow_fractional_occupations=false
))]
fn pbc_hessian(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    allow_fractional_occupations: bool,
) -> PyResult<PyObject> {
    let (mol, params, opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        multiplicity,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        allow_fractional_occupations,
    )?;
    let h = py
        .allow_threads(|| crate::pbc::pbc_hessian(&mol, &params, &opts))
        .map_err(to_py_err)?;
    let ang2 = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;
    let d = PyDict::new(py);
    d.set_item(
        "hessian_hartree_per_bohr2",
        (0..h.rows)
            .map(|i| (0..h.cols).map(|j| h[(i, j)] * EV_TO_HARTREE).collect())
            .collect::<Vec<Vec<f64>>>(),
    )?;
    d.set_item(
        "hessian_ev_per_angstrom2",
        (0..h.rows)
            .map(|i| (0..h.cols).map(|j| h[(i, j)] * ang2).collect())
            .collect::<Vec<Vec<f64>>>(),
    )?;
    d.set_item("ndof", h.rows)?;
    Ok(d.into())
}

/// Born effective charges `Z*_{a,αβ}`, in units of `e`.
///
/// `Σ_a Z*_a = 0` by charge conservation; the returned tensors satisfy it to machine precision
/// and `tests/pbc_born_charges.rs` asserts it.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, kpts=(2, 2, 2), charge=0.0, multiplicity=1, method="am1",
    realspace_cutoff=40.0, exchange_cutoff=20.0, e_tol=1.0e-11, p_tol=1.0e-10, max_scf=500, allow_fractional_occupations=false
))]
fn born_charges(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    allow_fractional_occupations: bool,
) -> PyResult<PyObject> {
    let (mol, params, opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        multiplicity,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        allow_fractional_occupations,
    )?;
    let z = py
        .allow_threads(|| crate::pbc::born_charges(&mol, &params, &opts))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("born_charges", z.clone())?;
    // The sum rule, reported rather than left to the caller to recompute.
    let mut sum = [[0.0_f64; 3]; 3];
    for t in &z {
        for a in 0..3 {
            for b in 0..3 {
                sum[a][b] += t[a][b];
            }
        }
    }
    d.set_item("acoustic_sum_rule_error", sum)?;
    Ok(d.into())
}

/// Clamped-ion polarizability `α` (Bohr³), in **any** periodic dimensionality.
///
/// The same `α` `dielectric()` returns, without the `ε_∞` conversion — which is why this one works
/// for a chain and a slab and that one does not: `ε_∞ = 1 + 4πα/Ω` needs `Ω` to be a volume.
/// Divide by the cell measure yourself and mind what the result is: dimensionless in 3D, a
/// **length** for a slab, an **area** for a chain.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, kpts=(2, 2, 2), charge=0.0, multiplicity=1, method="am1",
    realspace_cutoff=40.0, exchange_cutoff=20.0, e_tol=1.0e-11, p_tol=1.0e-10, max_scf=500, allow_fractional_occupations=false
))]
#[allow(clippy::too_many_arguments)]
fn polarizability(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    allow_fractional_occupations: bool,
) -> PyResult<PyObject> {
    let (mol, params, opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        multiplicity,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        allow_fractional_occupations,
    )?;
    let alpha = py
        .allow_threads(|| crate::pbc::polarizability(&mol, &params, &opts))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("polarizability_bohr3", alpha)?;
    d.set_item("measure", mol.cell.map(|c| c.measure()))?;
    d.set_item("n_periodic", mol.cell.map(|c| c.n_periodic()))?;
    Ok(d.into())
}
/// The macroscopic longitudinal dielectric function `ε(q)` along `q`, in any dimensionality.
///
/// `q` is a Cartesian wavevector in **inverse Bohr**, and must lie in the periodic subspace.
///
/// Three dimensions gives a constant — the familiar `ε_∞`, and `dielectric` returns it directly. A
/// slab and a chain do **not**: `ε(q) → 1` at long wavelength, because a sheet or a wire does not
/// screen a field whose wavelength exceeds its own extent. That is the same fact as a slab having
/// no LO–TO splitting at Γ.
///
/// `chain_radius` (Bohr) is required for a chain and ignored otherwise: the one-dimensional
/// Coulomb kernel is a logarithm and has no value without a reference length.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, q, chain_radius=None, kpts=(2, 2, 2), charge=0.0,
    multiplicity=1, method="am1", realspace_cutoff=40.0, exchange_cutoff=20.0,
    e_tol=1.0e-11, p_tol=1.0e-10, max_scf=500, allow_fractional_occupations=false
))]
#[allow(clippy::too_many_arguments)]
fn dielectric_function(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    q: Vec<f64>,
    chain_radius: Option<f64>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    allow_fractional_occupations: bool,
) -> PyResult<f64> {
    if q.len() != 3 {
        return Err(PyValueError::new_err("q must have three components"));
    }
    let qv = crate::math::Vec3::new(q[0], q[1], q[2]);
    let (mol, params, opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        multiplicity,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        allow_fractional_occupations,
    )?;
    py.allow_threads(|| crate::pbc::dielectric_function(&mol, &params, &opts, qv, chain_radius))
        .map_err(to_py_err)
}

/// Berry-phase polarization: `P = P_electronic + P_ionic`, modulo the quantum `e a_α/Ω`.
///
/// **Three-dimensional, restricted cells only.** `strings` is the number of k points per
/// Berry-phase string, resampled independently of `kpts` — the transverse directions use `kpts`,
/// the string's own direction uses this.
///
/// The phase in this atom-centred minimal basis tracks the charge **centres** and carries no
/// on-site `s`–`p` moment; `docs/pbc.md` records how large that gap is and where it shows.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, kpts=(2, 2, 2), strings=8, charge=0.0, method="am1",
    realspace_cutoff=40.0, exchange_cutoff=20.0, e_tol=1.0e-11, p_tol=1.0e-10, max_scf=500
))]
#[allow(clippy::too_many_arguments)]
fn polarization(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    kpts: (usize, usize, usize),
    strings: usize,
    charge: f64,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
) -> PyResult<PyObject> {
    let (mol, params, opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        1,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        false,
    )?;
    let r = py
        .allow_threads(|| crate::pbc::berry::berry_polarization(&mol, &params, &opts, strings))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    let v = |x: crate::math::Vec3| vec![x.x, x.y, x.z];
    d.set_item("polarization", v(r.total))?;
    d.set_item("electronic", v(r.electronic))?;
    d.set_item("ionic", v(r.ionic))?;
    d.set_item("phase_turns", r.phase.to_vec())?;
    d.set_item(
        "quantum",
        r.quantum.iter().map(|q| v(*q)).collect::<Vec<_>>(),
    )?;
    d.set_item("string_length", r.string_length)?;
    Ok(d.into())
}

/// A finite electric field **along** a periodic direction, by the Berry-phase electric enthalpy.
///
/// `field` is in **atomic units** (Hartree per e·Bohr), like every other field on this surface.
/// Three-dimensional, restricted, no smearing, and at least three k points along any direction the
/// field has a component in.
///
/// For a field *orthogonal* to every lattice vector — normal to a slab, transverse to a chain —
/// use `pbc_point(..., electric_field=...)`: that case is an ordinary `F·R` calculation and needs
/// none of this machinery.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, field, kpts=(4, 4, 4), charge=0.0, method="am1",
    realspace_cutoff=40.0, exchange_cutoff=20.0, e_tol=1.0e-11, p_tol=1.0e-10, max_scf=500,
    max_outer=60, outer_tol=1.0e-8, outer_mixing=0.5
))]
#[allow(clippy::too_many_arguments)]
fn finite_field(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    field: Vec<f64>,
    kpts: (usize, usize, usize),
    charge: f64,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    max_outer: usize,
    outer_tol: f64,
    outer_mixing: f64,
) -> PyResult<PyObject> {
    let f = field_from(Some(field))?
        .ok_or_else(|| PyValueError::new_err("finite_field needs a three-component field"))?;
    let (mol, params, mut opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        1,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        false,
    )?;
    // The Berry phase needs a gapped, integer-filled manifold. `periodic_setup` leaves whatever
    // smearing the shared default carries, and a silent smear would make the phase meaningless —
    // `run_finite_field` would reject it, but rejecting a default the caller never chose is worse
    // than setting it.
    opts.smearing_ev = 0.0;
    let ff = crate::pbc::FiniteFieldOptions {
        tol: outer_tol,
        max_iter: max_outer,
        mixing: outer_mixing,
    };
    let r = py
        .allow_threads(|| crate::pbc::run_finite_field(&mol, &params, &opts, f, &ff))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    let v = |x: crate::math::Vec3| vec![x.x, x.y, x.z];
    d.set_item("energy_ev", r.scf.total_ev)?;
    d.set_item("enthalpy_ev", r.enthalpy_ev)?;
    d.set_item("polarization", v(r.polarization))?;
    d.set_item("electronic", v(r.electronic_polarization))?;
    d.set_item("ionic", v(r.ionic_polarization))?;
    d.set_item("phase_turns", r.phase.to_vec())?;
    d.set_item("charges", r.scf.charges.clone())?;
    d.set_item("outer_iterations", r.iterations)?;
    d.set_item("scf_iterations", r.scf.iterations)?;
    Ok(d.into())
}

/// Clamped-ion polarizability `α` (Bohr³) and the electronic dielectric tensor `ε_∞`.
///
/// **Three-dimensional cells only.** `ε_∞ = 1 + 4πα/Ω` needs `Ω` to be a volume; a chain or a
/// slab is an error rather than a number in the wrong units. Since 0.2.2 the companion
/// `dielectric_with_extent()` handles those, once you say how thick the material is.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, kpts=(2, 2, 2), charge=0.0, multiplicity=1, method="am1",
    realspace_cutoff=40.0, exchange_cutoff=20.0, e_tol=1.0e-11, p_tol=1.0e-10, max_scf=500, allow_fractional_occupations=false
))]
fn dielectric(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    allow_fractional_occupations: bool,
) -> PyResult<PyObject> {
    let (mol, params, opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        multiplicity,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        allow_fractional_occupations,
    )?;
    let (alpha, epsilon) = py
        .allow_threads(|| crate::pbc::dielectric_tensor(&mol, &params, &opts))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("polarizability_bohr3", alpha)?;
    d.set_item("epsilon_infinity", epsilon)?;
    Ok(d.into())
}

/// `ε_∞` for a **slab or a chain**, given the extent you assign to the material.
///
/// Exactly one of `slab_thickness` (Bohr, for a 2D-periodic cell) or `wire_cross_section` (Bohr²,
/// for a 1D-periodic cell) is required. There is no default: a supercell says where the atoms are,
/// not where the material stops, and every choice here changes `ε`.
///
/// The conversion is not a division. `α` is the response to the **external** field — the
/// depolarizing field the induced charges make is already inside it — so it is
/// `ε = 1 + 4πχ/(1 − 4πNχ)` with `χ = α/(measure · extent)` and `N` the depolarization factor of
/// the assumed body: `0` in a slab's plane and along a wire, `1` along a slab normal, `1/2`
/// transverse to a wire's circular section.
///
/// Returned alongside `epsilon_infinity` are the two combinations that do **not** depend on the
/// choice, and which are what a low-dimensional calculation can report without one:
///
/// ```text
/// sheet_susceptibility     = 4π α_∥ / measure = (ε_∥ − 1) · extent
/// inverse_sheet_response   = 4π α_⊥ / measure = (1 − 1/ε_⊥) · extent
/// ```
///
/// Both are **scalars**, and `α_∥` in them is the *mean* over the two-dimensional half — the plane
/// for a slab, the transverse pair for a wire. The identities above hold per direction against the
/// returned tensor, and coincide with these scalars only when the response is isotropic there;
/// water on a square lattice is already 7 % anisotropic in plane. Take the tensor when the
/// direction matters and these when quoting one number.
///
/// Half the first is the Rytova–Keldysh screening length of the layer. `axis_mixing` reports how
/// much of `α` couples the distinguished axis to its complement, which the split drops; it is zero
/// whenever that axis is a principal axis of the response, and any symmetry at all makes it so.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, slab_thickness=None, wire_cross_section=None, kpts=(2, 2, 2),
    charge=0.0, multiplicity=1, method="am1", realspace_cutoff=40.0, exchange_cutoff=20.0,
    e_tol=1.0e-11, p_tol=1.0e-10, max_scf=500, allow_fractional_occupations=false
))]
#[allow(clippy::too_many_arguments)]
fn dielectric_with_extent(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    slab_thickness: Option<f64>,
    wire_cross_section: Option<f64>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    allow_fractional_occupations: bool,
) -> PyResult<PyObject> {
    use crate::pbc::ExtentConvention;
    let extent =
        match (slab_thickness, wire_cross_section) {
            (Some(d), None) => ExtentConvention::SlabThickness(d),
            (None, Some(s)) => ExtentConvention::WireCrossSection(s),
            (None, None) => return Err(PyValueError::new_err(
                "pass `slab_thickness` (Bohr) for a slab or `wire_cross_section` (Bohr^2) for a \
                 chain. There is no default: the cell does not say where the material stops, and \
                 the number returned is only as meaningful as that choice.",
            )),
            (Some(_), Some(_)) => return Err(PyValueError::new_err(
                "pass one of `slab_thickness` or `wire_cross_section`, not both: they describe \
                 different dimensionalities and carry different units",
            )),
        };
    let (mol, params, opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        multiplicity,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        allow_fractional_occupations,
    )?;
    let (alpha, epsilon) = py
        .allow_threads(|| crate::pbc::dielectric_tensor_with_extent(&mol, &params, &opts, extent))
        .map_err(to_py_err)?;

    let cell_ref = mol
        .cell
        .ok_or_else(|| PyValueError::new_err("a dielectric tensor needs a cell"))?;
    let measure = cell_ref.measure();
    let ax = cell_ref.periodic_axes();
    let axis = match extent {
        ExtentConvention::SlabThickness(_) => {
            cell_ref.cell.col[ax[0]].cross(cell_ref.cell.col[ax[1]])
        }
        ExtentConvention::WireCrossSection(_) => cell_ref.cell.col[ax[0]],
    };
    let unit = axis / axis.norm();
    let along = |u: crate::math::Vec3| -> f64 {
        let c = [u.x, u.y, u.z];
        let mut acc = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                acc += c[i] * alpha[i][j] * c[j];
            }
        }
        acc
    };
    // The `N = 0` channel is the plane for a slab and the axis for a wire; the depolarizing one is
    // the other. `α` is symmetric, so the mean over the two-dimensional half is the trace less the
    // distinguished component, halved.
    let trace: f64 = (0..3).map(|i| alpha[i][i]).sum();
    let a_axis = along(unit);
    let a_plane = (trace - a_axis) / 2.0;
    let (a_par, a_perp) = match extent {
        ExtentConvention::SlabThickness(_) => (a_plane, a_axis),
        ExtentConvention::WireCrossSection(_) => (a_axis, a_plane),
    };

    let d = PyDict::new(py);
    d.set_item("polarizability_bohr3", alpha)?;
    d.set_item("epsilon_infinity", epsilon)?;
    d.set_item("extent", extent.value())?;
    d.set_item("measure", measure)?;
    d.set_item("n_periodic", cell_ref.n_periodic())?;
    d.set_item("axis", vec![unit.x, unit.y, unit.z])?;
    d.set_item(
        "sheet_susceptibility",
        4.0 * std::f64::consts::PI * a_par / measure,
    )?;
    d.set_item(
        "inverse_sheet_response",
        4.0 * std::f64::consts::PI * a_perp / measure,
    )?;
    // `2π χ₂D` is the Rytova–Keldysh screening length, and it is a *sheet* quantity — a wire's
    // axial susceptibility is not one, so the key is only present for a slab rather than being
    // filled with a number that would be read as one.
    if matches!(extent, ExtentConvention::SlabThickness(_)) {
        d.set_item(
            "rytova_keldysh_length",
            2.0 * std::f64::consts::PI * a_par / measure,
        )?;
    }
    d.set_item("axis_mixing", crate::pbc::extent_axis_mixing(&alpha, unit))?;
    Ok(d.into())
}

/// Phonons at arbitrary `q` by density-functional perturbation theory.
///
/// No supercell: the response is solved directly on the primitive cell, coupling `k` to `k + q`.
/// `q_points` are fractional coordinates of the primitive reciprocal lattice.
///
/// **This is the analytic part of `D(q)`.** The direction-dependent non-analytic term of a polar
/// 3D crystal is not included here — it comes from `born_charges` and `dielectric`, and adding
/// both would double count. See `docs/pbc.md`.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, q_points, kpts=(2, 2, 2), charge=0.0, multiplicity=1,
    method="am1", realspace_cutoff=40.0, exchange_cutoff=20.0, e_tol=1.0e-11, p_tol=1.0e-10,
    max_scf=500, long_range="auto",
    cpscf_tol=1.0e-8, cpscf_max_iter=200, cpscf_mixing=0.7, smearing_ev=0.0, allow_fractional_occupations=false
))]
#[allow(clippy::too_many_arguments)]
fn dfpt(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    q_points: Vec<Vec<f64>>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    method: &str,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    long_range: &str,
    cpscf_tol: f64,
    cpscf_max_iter: usize,
    cpscf_mixing: f64,
    smearing_ev: f64,
    allow_fractional_occupations: bool,
) -> PyResult<PyObject> {
    use crate::pbc::kpoints::KPoint;
    use crate::pbc::{DfptOptions, LongRange};
    let (mol, params, opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        multiplicity,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        smearing_ev,
        allow_fractional_occupations,
    )?;
    let long_range = match long_range.trim().to_ascii_lowercase().as_str() {
        "auto" | "" => LongRange::Auto,
        "require" => LongRange::Require,
        "off" => LongRange::Off,
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid long_range '{other}': expected 'auto', 'require' or 'off'"
            )))
        }
    };
    let dfpt_opts = DfptOptions {
        long_range,
        cpscf_tol,
        cpscf_max_iter,
        cpscf_mixing,
        ..DfptOptions::default()
    };
    let qs: Vec<KPoint> = q_points
        .iter()
        .map(|q| {
            if q.len() != 3 {
                return Err(PyValueError::new_err("each q point needs three components"));
            }
            Ok(KPoint {
                fractional: [q[0], q[1], q[2]],
                weight: 1.0,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let bands = py
        .allow_threads(|| -> crate::error::Result<Vec<Vec<f64>>> {
            qs.iter()
                .map(|q| crate::pbc::frequencies_dfpt_with(&mol, &params, &opts, &dfpt_opts, *q))
                .collect()
        })
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item(
        "q_points",
        qs.iter().map(|q| q.fractional).collect::<Vec<_>>(),
    )?;
    d.set_item("frequencies_cm", bands)?;
    d.set_item("method", params.method.name())?;
    Ok(d.into())
}

/// L-BFGS geometry optimization. Returns optimized positions in Ångström.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1", electric_field=None))]
fn optimize(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options(charge, multiplicity, reference, electric_field)?;
    let res = py
        .allow_threads(|| opt_geom(&mol, &params, &opts, &OptOptions::default()))
        .map_err(to_py_err)?;
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
    d.set_item("energy_ev", res.scf.total_ev)?;
    d.set_item("heat_of_formation_kcal", res.scf.heat_of_formation_kcal)?;
    d.set_item("converged", res.converged)?;
    d.set_item("iterations", res.iterations)?;
    // The converged SCF at the final geometry, so that reporting it does not mean running the
    // whole calculation a second time on a structure that has already been solved.
    d.set_item("electronic_ev", res.scf.electronic_ev)?;
    d.set_item("core_ev", res.scf.core_ev)?;
    d.set_item("homo_ev", res.scf.homo_ev)?;
    d.set_item("lumo_ev", res.scf.lumo_ev)?;
    d.set_item("charges", res.scf.charges.clone())?;
    d.set_item(
        "dipole_debye",
        [
            res.scf.dipole_debye.x,
            res.scf.dipole_debye.y,
            res.scf.dipole_debye.z,
        ],
    )?;
    d.set_item("dipole_magnitude", res.scf.dipole_magnitude)?;
    d.set_item("scf_iterations", res.scf.iterations)?;
    d.set_item("unrestricted", res.scf.unrestricted)?;
    Ok(d.into())
}

/// AM1-BCC partial charges for AMBER.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1))]
fn am1_bcc(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
) -> PyResult<PyObject> {
    // AM1-BCC is defined on the AM1 Mulliken charges, so the method is fixed here by the
    // parameterization itself -- an RM1 or SAM1 density would need its own BCC parameter set.
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Am1Parameters::standard().map_err(to_py_err)?;
    let opts = Am1Options {
        charge,
        multiplicity: multiplicity.max(1),
        ..Am1Options::default()
    };
    let bcc = py
        .allow_threads(|| crate::bcc::am1_bcc_charges(&mol, &params, &opts))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    // The mol2 text, rendered by the crate rather than by the caller: the Python front end
    // writes a `--mol2-output` file and it has to be byte-identical to the Rust one's. Built
    // before the fields are moved out below.
    d.set_item("mol2", crate::bcc::to_mol2(&mol, &bcc))?;
    d.set_item("charges", bcc.charges)?;
    d.set_item("mulliken", bcc.mulliken)?;
    d.set_item("atom_types", bcc.atom_types)?;
    d.set_item("warnings", bcc.warnings)?;
    Ok(d.into())
}

/// Harmonic vibrational frequencies (cm⁻¹) at the given geometry.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1", electric_field=None, cphf_max_iter=None, cphf_tol=None))]
fn frequencies(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
    cphf_max_iter: Option<usize>,
    cphf_tol: Option<f64>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options_with(
        charge,
        multiplicity,
        reference,
        electric_field,
        cphf_max_iter,
        cphf_tol,
    )?;
    let vib = py
        .allow_threads(|| crate::hessian::vibrational_analysis(&mol, &params, &opts, 1.0e-3))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    set_mode_items(&d, &vib)?;
    Ok(d.into())
}

/// Analytic (CPHF) Cartesian Hessian at the given geometry.
///
/// Returns the full `3N × 3N` second-derivative matrix in **atomic units (Hartree/Bohr²)** —
/// the native surface's convention — and, for convenience, the same matrix in eV/Å². Row/column
/// index `3·i + k` is atom `i`, Cartesian axis `k` (x, y, z), matching the input atom order.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1", electric_field=None, cphf_max_iter=None, cphf_tol=None))]
fn hessian(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
    cphf_max_iter: Option<usize>,
    cphf_tol: Option<f64>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options_with(
        charge,
        multiplicity,
        reference,
        electric_field,
        cphf_max_iter,
        cphf_tol,
    )?;
    // Fully analytic (CPHF) Hessian, returned by the core in eV/Bohr².
    let h = py
        .allow_threads(|| crate::hessian::analytic_hessian(&mol, &params, &opts, 1.0e-3))
        .map_err(to_py_err)?;
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

/// Orbital energies, coefficients and occupations — the wavefunction as numbers.
///
/// Both spin channels for an unrestricted run. Energies are in **Hartree** (the native surface's
/// convention) and additionally in eV, since orbital energies are conventionally quoted there.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1", electric_field=None))]
fn orbitals(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options(charge, multiplicity, reference, electric_field)?;
    let r = py
        .allow_threads(|| run_am1(&mol, &params, &opts))
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item(
        "energies_hartree",
        r.mo_energies
            .iter()
            .map(|e| e * EV_TO_HARTREE)
            .collect::<Vec<_>>(),
    )?;
    d.set_item("energies_ev", r.mo_energies.clone())?;
    // Columns are orbitals, rows are atomic orbitals — the same layout the Rust `Matrix` uses.
    d.set_item("coefficients", matrix_rows(&r.mo_coeff))?;
    d.set_item("n_occupied", r.n_occ)?;
    d.set_item("homo_ev", r.homo_ev)?;
    d.set_item("lumo_ev", r.lumo_ev)?;
    d.set_item("unrestricted", r.unrestricted)?;
    if let Some(b) = &r.beta {
        d.set_item(
            "beta_energies_hartree",
            b.energies
                .iter()
                .map(|e| e * EV_TO_HARTREE)
                .collect::<Vec<_>>(),
        )?;
        d.set_item("beta_energies_ev", b.energies.clone())?;
        d.set_item("beta_coefficients", matrix_rows(&b.coeff))?;
        d.set_item("beta_n_occupied", b.n_occ)?;
        d.set_item("homo_beta_ev", r.homo_beta_ev)?;
        d.set_item("lumo_beta_ev", r.lumo_beta_ev)?;
    }
    // The AO labels, so a caller can say which coefficient is which without rebuilding the basis.
    let basis = crate::basis::Basis::build(&mol, &params).map_err(to_py_err)?;
    let labels: Vec<(usize, String)> = basis
        .aos
        .iter()
        .map(|ao| {
            let orb = match ao.orb {
                0 => "s",
                1 => "px",
                2 => "py",
                _ => "pz",
            };
            (ao.atom, orb.to_string())
        })
        .collect();
    d.set_item("ao_labels", labels)?;
    d.set_item("method", params.method.name())?;
    Ok(d.into())
}

/// The wavefunction as a **Molden**-format string.
///
/// Write it to a file and open it in a viewer. `basis` selects the section: `"gto"` (the default)
/// writes a fitted Gaussian expansion of the Slater functions, which is what every viewer reads;
/// `"sto"` writes the Slater functions themselves, which is exact and which most viewers skip.
/// `primitives` is the expansion length for `"gto"`.
///
/// `deorthogonalize` (default `True`) writes `S^{-1/2} C`, so the coefficients belong to the
/// non-orthogonal functions the file declares rather than to NDDO's implicitly orthogonalized
/// basis. Pass `False` for the raw NDDO coefficients, which is what 0.2.2 and most other NDDO
/// codes emit.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1",
    electric_field=None, basis="gto", primitives=crate::gto::DEFAULT_NGAUSS, deorthogonalize=true
))]
#[allow(clippy::too_many_arguments)]
fn molden(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
    basis: &str,
    primitives: usize,
    deorthogonalize: bool,
) -> PyResult<String> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options(charge, multiplicity, reference, electric_field)?;
    let section = match basis.trim().to_ascii_lowercase().as_str() {
        "gto" | "gaussian" => crate::molden::MoldenBasis::Gaussian(primitives),
        "sto" | "slater" => crate::molden::MoldenBasis::Slater,
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid molden basis '{other}' (expected 'gto' or 'sto')"
            )))
        }
    };
    let molden_opts = crate::molden::MoldenOptions {
        basis: section,
        deorthogonalize,
    };
    py.allow_threads(|| {
        let scf = run_am1(&mol, &params, &opts)?;
        crate::molden::to_molden_with(&mol, &params, &scf, &molden_opts)
    })
    .map_err(to_py_err)
}

/// **L-BFGS geometry optimization under periodic boundary conditions.**
///
/// The periodic counterpart of [`optimize`]: relaxes the atoms against the k-point forces and,
/// with `relax_cell`, the lattice against the analytic stress. Returns the relaxed positions in
/// Ångström **and the relaxed cell**, because with `relax_cell` the positions alone do not
/// describe the structure.
///
/// `pressure` is a target in eV per Bohr^d (`d` = number of periodic directions) and is the
/// pressure the enthalpy `E + PΩ` is minimized at; it does nothing without `relax_cell`.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, cell, pbc, kpts=(1, 1, 1), charge=0.0, multiplicity=1,
    unrestricted=false, method="am1", smearing_ev=0.0, realspace_cutoff=40.0,
    exchange_cutoff=20.0, e_tol=1.0e-8, p_tol=1.0e-7, max_scf=300,
    relax_cell=false, pressure=0.0, max_iter=200, gtol=1.0e-3, stress_tol=1.0e-5,
    electric_field=None
))]
#[allow(clippy::too_many_arguments)]
fn pbc_optimize(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Vec<bool>,
    kpts: (usize, usize, usize),
    charge: f64,
    multiplicity: usize,
    unrestricted: bool,
    method: &str,
    smearing_ev: f64,
    realspace_cutoff: f64,
    exchange_cutoff: f64,
    e_tol: f64,
    p_tol: f64,
    max_scf: usize,
    relax_cell: bool,
    pressure: f64,
    max_iter: usize,
    gtol: f64,
    stress_tol: f64,
    // A relaxation under a field is a relaxation of a different Hamiltonian, so it belongs here
    // alongside every other term of it. Absent through 0.2.2, which meant the Python CLI's
    // periodic `optimize` silently ignored `--field` while the Rust one applied it.
    electric_field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    use crate::pbc::optimizer::{optimize_periodic, PbcOptOptions};
    use crate::pbc::PbcOptions;

    let (mol, params, opts) = periodic_setup(
        &numbers,
        &positions,
        &cell,
        &pbc,
        kpts,
        charge,
        multiplicity,
        method,
        realspace_cutoff,
        exchange_cutoff,
        e_tol,
        p_tol,
        max_scf,
        0.0,
        false,
    )?;
    // ield_from, not a second conversion: this module's one place where atomic units become
    // the crate's eV per e*Bohr.
    let field = field_from(electric_field)?;
    let opts = PbcOptions {
        unrestricted: unrestricted || multiplicity > 1,
        smearing_ev,
        electric_field: field,
        // `periodic_setup` turns time-reversal folding **off**, because the response entry points
        // it also serves must sample `k` and `−k` separately. A ground-state relaxation has no
        // such constraint, folding is exact for a real Hamiltonian, and leaving it off here would
        // make this entry point sample a different mesh from `run_pbc_scf`'s default — which is
        // what the two CLI front ends disagreed about: 4 k-points against 3.
        fold_time_reversal: true,
        ..opts
    };
    let settings = PbcOptOptions {
        max_iter,
        gtol,
        relax_cell,
        pressure,
        stress_tol,
        ..PbcOptOptions::default()
    };
    let res = py
        .allow_threads(|| optimize_periodic(&mol, &params, &opts, &settings))
        .map_err(to_py_err)?;

    let coords: Vec<[f64; 3]> = res
        .molecule
        .atoms
        .iter()
        .map(|a| {
            let p = a.position * BOHR_TO_ANGSTROM;
            [p.x, p.y, p.z]
        })
        .collect();
    let relaxed = res
        .molecule
        .cell
        .ok_or_else(|| PyValueError::new_err("a periodic optimization must return a cell"))?;
    let cell_rows: Vec<[f64; 3]> = relaxed
        .cell
        .col
        .iter()
        .map(|v| {
            let a = *v * BOHR_TO_ANGSTROM;
            [a.x, a.y, a.z]
        })
        .collect();
    let forces: Vec<[f64; 3]> = res
        .gradient
        .forces
        .iter()
        .map(|f| {
            let v = *f * BOHR_TO_ANGSTROM.recip();
            [v.x, v.y, v.z]
        })
        .collect();

    let d = PyDict::new(py);
    d.set_item("positions_angstrom", coords)?;
    d.set_item("cell_angstrom", cell_rows)?;
    d.set_item("pbc", relaxed.periodic.to_vec())?;
    d.set_item("energy_ev", res.scf.total_ev)?;
    d.set_item("energy_hartree", res.scf.total_ev * EV_TO_HARTREE)?;
    d.set_item("electronic_ev", res.scf.electronic_ev)?;
    d.set_item("core_ev", res.scf.core_ev)?;
    d.set_item("free_energy_ev", res.scf.free_energy_ev())?;
    d.set_item("forces_ev_per_angstrom", forces)?;
    // eV/Å^d, matching `pbc_point` and ASE — see the note there. The crate's native eV/Bohr^d is
    // beside it rather than instead of it, because the CLI needs one and ASE needs the other and
    // converting in either front end is what put the two out of step.
    let stress_scale = ANGSTROM_TO_BOHR.powi(relaxed.n_periodic() as i32);
    d.set_item(
        "stress_voigt",
        res.gradient
            .stress_voigt()
            .iter()
            .map(|v| v * stress_scale)
            .collect::<Vec<f64>>(),
    )?;
    d.set_item("stress_voigt_ev_per_bohr", res.gradient.stress_voigt())?;
    d.set_item("pressure", res.gradient.pressure(relaxed.n_periodic()))?;
    d.set_item("charges", res.scf.charges.clone())?;
    d.set_item("fermi_energy_ev", res.scf.fermi_energy_ev)?;
    d.set_item("entropy_ev", res.scf.entropy_ev)?;
    d.set_item("k_points", res.scf.k_points)?;
    d.set_item("converged", res.converged)?;
    d.set_item("iterations", res.iterations)?;
    d.set_item("scf_iterations", res.scf.iterations)?;
    d.set_item("unrestricted", res.scf.unrestricted)?;
    d.set_item("max_image_overlap", res.scf.max_image_overlap)?;
    d.set_item("charged_cell_warning", res.scf.charged_cell_warning.clone())?;
    d.set_item("max_force_ev_per_bohr", res.gradient.max_gradient)?;
    d.set_item("method", params.method.name())?;
    Ok(d.into())
}

/// Infrared spectrum: the atomic polar tensor and the mode-resolved intensities.
///
/// **Expensive** — it solves the CPHF equations, i.e. it costs an analytic Hessian. Called
/// explicitly rather than folded into a single point for exactly that reason.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1", electric_field=None, cphf_max_iter=None, cphf_tol=None))]
fn ir_spectrum(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
    cphf_max_iter: Option<usize>,
    cphf_tol: Option<f64>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options_with(
        charge,
        multiplicity,
        reference,
        electric_field,
        cphf_max_iter,
        cphf_tol,
    )?;
    let s = py
        .allow_threads(|| crate::ir::ir_spectrum(&mol, &params, &opts))
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    // The raw tensor: 3 rows (dipole component) by 3N columns (atom, axis), in units of e.
    d.set_item("dipole_derivatives", matrix_rows(&s.dipole_derivatives))?;
    d.set_item("frequencies_cm", s.frequencies_cm.clone())?;
    d.set_item("intensities_km_per_mol", s.intensities_km_per_mol.clone())?;
    // The dense per-mode tensor, which keeps the transition dipole's *direction* — the intensity
    // throws it away, and a polarized measurement sees it.
    d.set_item(
        "mode_dipole_derivatives",
        matrix_rows(&s.mode_dipole_derivatives),
    )?;
    d.set_item(
        "translation_rotation_overlap",
        s.modes.translation_rotation_overlap.clone(),
    )?;
    d.set_item("rigid_body_count", s.modes.rigid_body_count)?;
    d.set_item(
        "rigid_body_frequencies_cm",
        s.modes.rigid_body_frequencies_cm.clone(),
    )?;
    d.set_item("modes", matrix_rows(&s.modes.modes))?;
    let bands = s.vibrational_bands(0.5);
    d.set_item(
        "vibrational_modes",
        bands.iter().map(|(k, _, _)| *k).collect::<Vec<_>>(),
    )?;
    d.set_item("method", params.method.name())?;
    Ok(d.into())
}

/// The atomic polar tensor `∂μ_α/∂R_{a,β}` on its own, in units of `e`.
///
/// The molecular counterpart of the Born effective charges. Same cost as an infrared spectrum —
/// it is where most of that cost goes.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1", electric_field=None))]
fn dipole_derivatives(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options(charge, multiplicity, reference, electric_field)?;
    let apt = py
        .allow_threads(|| crate::ir::dipole_derivatives(&mol, &params, &opts))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("dipole_derivatives", matrix_rows(&apt))?;
    d.set_item("ndof", apt.cols)?;
    Ok(d.into())
}

/// First-order orbital response `U^j_{ai}` — the CPHF coefficients, one block per Cartesian
/// degree of freedom.
///
/// **Expensive**, and a by-product of the analytic Hessian: this returns the `U` that Hessian
/// already solves for rather than recomputing it, but it does have to run the Hessian.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1", electric_field=None, response_density=false))]
fn orbital_response(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
    response_density: bool,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options(charge, multiplicity, reference, electric_field)?;
    let r = py
        .allow_threads(|| {
            crate::hessian::analytic_hessian_with_response(&mol, &params, &opts, 1.0e-3)
        })
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    let pack = |c: &crate::hessian::ResponseChannel| -> Vec<Vec<Vec<f64>>> {
        c.u_ov.iter().map(matrix_rows).collect()
    };
    d.set_item("u_ov", pack(&r.alpha))?;
    d.set_item(
        "g_ov",
        r.alpha.g_ov.iter().map(matrix_rows).collect::<Vec<_>>(),
    )?;
    d.set_item("n_occupied", r.alpha.occupied.cols)?;
    d.set_item("n_virtual", r.alpha.virtuals.cols)?;
    d.set_item("ndof", r.ndof())?;
    if let Some(b) = &r.beta {
        d.set_item("beta_u_ov", pack(b))?;
        d.set_item("beta_n_occupied", b.occupied.cols)?;
        d.set_item("beta_n_virtual", b.virtuals.cols)?;
    }
    // Off by default: `3N` AO-basis matrices is the largest array in the calculation.
    if response_density {
        let dp: Vec<Vec<Vec<f64>>> = (0..r.ndof())
            .map(|j| matrix_rows(&r.response_density(j)))
            .collect();
        d.set_item("response_density", dp)?;
    }
    d.set_item(
        "cphf_iterations",
        r.cphf.iter().map(|o| o.iterations).collect::<Vec<_>>(),
    )?;
    d.set_item("hessian_ev_per_bohr2", matrix_rows(&r.hessian))?;
    Ok(d.into())
}

/// The whole vibrational group from **one** SCF and one CPHF solve.
///
/// `hessian`, `frequencies`, `ir_spectrum`, `dipole_derivatives` and `orbital_response` are five
/// entry points that each run [`crate::hessian::analytic_hessian_with_response`] in full and then
/// keep a different contraction of it. A caller wanting several — an infrared spectrum *and* the
/// Hessian it came from, which is the ordinary case — paid for the CPHF once per question. They
/// are all contractions of the same response, so this returns them together.
///
/// Every section is opt-in, because the two expensive ones are not always wanted: `U^j_{ai}` is
/// `O(ndof · n_occ · n_vir)` and the response densities are `O(ndof · nao²)`, the largest array in
/// the calculation.
///
/// The five existing functions are unchanged and still work; this is the one the ASE layer uses.
#[pyfunction]
#[pyo3(signature = (
    numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="am1",
    electric_field=None, hessian=true, frequencies=true, ir=true,
    orbital_response=false, response_density=false, cphf_max_iter=None, cphf_tol=None
))]
#[allow(clippy::too_many_arguments)]
fn vibrations(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    electric_field: Option<Vec<f64>>,
    hessian: bool,
    frequencies: bool,
    ir: bool,
    orbital_response: bool,
    response_density: bool,
    cphf_max_iter: Option<usize>,
    cphf_tol: Option<f64>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = params_for(method)?;
    let opts = molecular_options_with(
        charge,
        multiplicity,
        reference,
        electric_field,
        cphf_max_iter,
        cphf_tol,
    )?;

    // The one solve.
    let r = py
        .allow_threads(|| {
            crate::hessian::analytic_hessian_with_response(&mol, &params, &opts, 1.0e-3)
        })
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item("ndof", r.ndof())?;
    d.set_item("method", params.method.name())?;
    d.set_item(
        "cphf_iterations",
        r.cphf.iter().map(|o| o.iterations).collect::<Vec<_>>(),
    )?;

    if hessian {
        let ndof = r.hessian.rows;
        let ang2 = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;
        let mut h_au: Vec<Vec<f64>> = Vec::with_capacity(ndof);
        let mut h_ev_ang: Vec<Vec<f64>> = Vec::with_capacity(ndof);
        for i in 0..ndof {
            let mut row_au = Vec::with_capacity(ndof);
            let mut row_ev = Vec::with_capacity(ndof);
            for j in 0..ndof {
                let v = r.hessian[(i, j)]; // eV/Bohr²
                row_au.push(v * EV_TO_HARTREE);
                row_ev.push(v * ang2);
            }
            h_au.push(row_au);
            h_ev_ang.push(row_ev);
        }
        d.set_item("hessian_hartree_per_bohr2", h_au)?;
        d.set_item("hessian_ev_per_angstrom2", h_ev_ang)?;
        d.set_item("hessian_ev_per_bohr2", matrix_rows(&r.hessian))?;
    }

    // The infrared spectrum carries the normal modes, so asking for both costs one analysis.
    if ir {
        let s = py
            .allow_threads(|| crate::ir::ir_spectrum_from_response(&mol, &params, &r))
            .map_err(to_py_err)?;
        d.set_item("dipole_derivatives", matrix_rows(&s.dipole_derivatives))?;
        d.set_item("intensities_km_per_mol", s.intensities_km_per_mol.clone())?;
        d.set_item(
            "mode_dipole_derivatives",
            matrix_rows(&s.mode_dipole_derivatives),
        )?;
        let bands = s.vibrational_bands(0.5);
        d.set_item(
            "vibrational_modes",
            bands.iter().map(|(k, _, _)| *k).collect::<Vec<_>>(),
        )?;
        set_mode_items(&d, &s.modes)?;
    } else if frequencies {
        let vib = py
            .allow_threads(|| {
                crate::hessian::vibrational_analysis_from_hessian(&mol, r.hessian.clone())
            })
            .map_err(to_py_err)?;
        set_mode_items(&d, &vib)?;
    }

    if orbital_response {
        let pack = |c: &crate::hessian::ResponseChannel| -> Vec<Vec<Vec<f64>>> {
            c.u_ov.iter().map(matrix_rows).collect()
        };
        d.set_item("u_ov", pack(&r.alpha))?;
        d.set_item(
            "g_ov",
            r.alpha.g_ov.iter().map(matrix_rows).collect::<Vec<_>>(),
        )?;
        d.set_item("n_occupied", r.alpha.occupied.cols)?;
        d.set_item("n_virtual", r.alpha.virtuals.cols)?;
        if let Some(b) = &r.beta {
            d.set_item("beta_u_ov", pack(b))?;
            d.set_item("beta_n_occupied", b.occupied.cols)?;
            d.set_item("beta_n_virtual", b.virtuals.cols)?;
        }
    }
    if response_density {
        let dp: Vec<Vec<Vec<f64>>> = (0..r.ndof())
            .map(|j| matrix_rows(&r.response_density(j)))
            .collect();
        d.set_item("response_density", dp)?;
    }
    Ok(d.into())
}

/// The `VibrationalModes` fields, under the same keys `frequencies` and `ir_spectrum` use.
fn set_mode_items(d: &Bound<'_, PyDict>, vib: &crate::hessian::VibrationalModes) -> PyResult<()> {
    d.set_item("frequencies_cm", vib.frequencies_cm.clone())?;
    d.set_item("eigenvalues", vib.eigenvalues.clone())?;
    d.set_item("modes", matrix_rows(&vib.modes))?;
    d.set_item(
        "cartesian_displacements",
        matrix_rows(&vib.cartesian_displacements),
    )?;
    d.set_item(
        "translation_rotation_overlap",
        vib.translation_rotation_overlap.clone(),
    )?;
    // Since 0.2.3 the rigid-body directions are projected out before the diagonalization, so
    // `frequencies_cm` is `3N − rigid_body_count` long and holds vibrations only. These two say
    // what was removed: the count (6, 5 for a linear molecule, 3 for an atom) and the curvature
    // left in each removed direction, which is zero only at a stationary point.
    d.set_item("rigid_body_count", vib.rigid_body_count)?;
    d.set_item(
        "rigid_body_frequencies_cm",
        vib.rigid_body_frequencies_cm.clone(),
    )?;
    // The unprojected `3N` spectrum — what 0.2.2 returned — so the projection can be checked
    // rather than taken on trust.
    d.set_item("all_frequencies_cm", vib.all_frequencies_cm.clone())?;
    d.set_item("all_eigenvalues", vib.all_eigenvalues.clone())?;
    Ok(())
}

/// The model's unit conversions.
///
/// These are exported rather than left for a caller to write down because they are deliberately
/// **not** CODATA: the crate uses MOPAC7's `ev = 27.21` and `a0 = 0.529167` throughout, and
/// mixing a CODATA value into a conversion at the boundary shifts a heat of formation by a few
/// hundredths of a kcal/mol without anything failing. A Python-side copy of these numbers is a
/// copy that can drift; this cannot.
#[pyfunction]
fn constants(py: Python<'_>) -> PyResult<PyObject> {
    use crate::constants as c;
    let d = PyDict::new(py);
    d.set_item("hartree_to_ev", c::HARTREE_TO_EV)?;
    d.set_item("ev_to_hartree", c::EV_TO_HARTREE)?;
    d.set_item("angstrom_to_bohr", c::ANGSTROM_TO_BOHR)?;
    d.set_item("bohr_to_angstrom", c::BOHR_TO_ANGSTROM)?;
    d.set_item("ev_to_kcal", c::EV_TO_KCAL)?;
    d.set_item("kcal_to_ev", c::KCAL_TO_EV)?;
    d.set_item("au_dipole_to_debye", c::AU_DIPOLE_TO_DEBYE)?;
    // The infrared chain, for the same reason. `42.2561` converts |dmu/dQ|^2 in
    // D^2 / (A^2 . amu) to km/mol, and `e_to_debye_per_angstrom` is the step that gets an
    // atomic polar tensor in `e` into those units. A caller computing intensities from
    // `dipole_derivatives` needs both, and hardcoding them Python-side is exactly the drift
    // this function exists to prevent -- the second one is built from *this crate's* Bohr.
    d.set_item(
        "ir_intensity_km_per_mol",
        crate::ir::IR_INTENSITY_KM_PER_MOL,
    )?;
    d.set_item(
        "e_to_debye_per_angstrom",
        crate::ir::E_TO_DEBYE_PER_ANGSTROM,
    )?;
    Ok(d.into())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(constants, m)?)?;
    m.add_function(wrap_pyfunction!(single_point, m)?)?;
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(pbc_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(hessian, m)?)?;
    m.add_function(wrap_pyfunction!(am1_bcc, m)?)?;
    m.add_function(wrap_pyfunction!(pbc_point, m)?)?;
    m.add_function(wrap_pyfunction!(divide_conquer, m)?)?;
    m.add_function(wrap_pyfunction!(phonons, m)?)?;
    m.add_function(wrap_pyfunction!(orbitals, m)?)?;
    m.add_function(wrap_pyfunction!(molden, m)?)?;
    m.add_function(wrap_pyfunction!(ir_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(dipole_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(orbital_response, m)?)?;
    m.add_function(wrap_pyfunction!(vibrations, m)?)?;
    m.add_function(wrap_pyfunction!(pbc_hessian, m)?)?;
    m.add_function(wrap_pyfunction!(born_charges, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_with_extent, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_function, m)?)?;
    m.add_function(wrap_pyfunction!(polarizability, m)?)?;
    m.add_function(wrap_pyfunction!(polarization, m)?)?;
    m.add_function(wrap_pyfunction!(finite_field, m)?)?;
    m.add_function(wrap_pyfunction!(dfpt, m)?)?;
    m.add_function(wrap_pyfunction!(lo_to_frequencies, m)?)?;
    Ok(())
}
