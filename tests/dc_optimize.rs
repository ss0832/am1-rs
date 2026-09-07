// SPDX-License-Identifier: GPL-3.0-or-later

//! Divide-and-conquer geometry optimization.
//!
//! Three things have to hold, and they fail independently:
//!
//! * the force is the derivative of the energy the line search descends — otherwise the search
//!   accepts steps that raise the quantity the gradient is reducing, and the relaxation stalls
//!   above the minimum looking like a tolerance problem;
//! * the driver descends;
//! * with the partition switched off — one subsystem holding every atom — it reproduces the
//!   ordinary relaxation, which is the degenerate case where there is no approximation left for a
//!   difference to hide in.
//!
//! None of these is a comparison against the full-SCF *minimum*. Divide-and-conquer relaxes to
//! its own stationary point, a buffer-sized distance from the exact one, and asserting otherwise
//! would be asserting that the approximation is not one.

use am1_rs::divide_conquer::{
    divide_conquer_gradient, optimize_divide_conquer, run_divide_conquer, DcOptions,
};
use am1_rs::fermi::Filling;
use am1_rs::optimizer::{optimize, OptOptions};
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule, Vec3};

const ANG: f64 = 1.0 / 0.529167;

/// Three waters, far enough apart that a small core size gives a genuine partition.
///
/// Deliberately **not** a system whose relaxation converges: three loosely bound waters have
/// intermolecular modes so flat that the ordinary optimizer does not reach `gtol` on them either
/// (measured: 200 steps, still moving). That makes it a poor convergence test and a good descent
/// test, and the assertions below are written accordingly.
fn water_trimer() -> Molecule {
    let unit = [
        (8, 0.0, 0.0, 0.0),
        (1, 0.9584, 0.0, 0.0),
        (1, -0.24, 0.9278, 0.0),
    ];
    let shifts = [(0.0, 0.0, 0.0), (0.0, 0.0, 4.0), (4.0, 0.0, 0.0)];
    let mut atoms = Vec::new();
    for (sx, sy, sz) in shifts {
        for (z, x, y, zc) in unit {
            atoms.push(Atom {
                z,
                position: Vec3::new((x + sx) * ANG, (y + sy) * ANG, (zc + sz) * ANG),
            });
        }
    }
    Molecule::new(atoms)
}

fn dc_options() -> DcOptions {
    DcOptions {
        core_size: 3,
        buffer_radius: 12.0,
        ..DcOptions::default()
    }
}

/// The analytic force is the derivative of the divide-and-conquer energy.
///
/// # What this does and does not settle
///
/// At a finite electronic temperature the variational quantity is the free energy `E − TS`, not
/// `E`, and a relaxation has to line-search on whichever one the force differentiates. This test
/// computes the finite difference of **both** and reports them. On this system they come out
/// equal to eight digits — the HOMO–LUMO gap is around 16 eV, so at `kt = 0.1 eV` every level is
/// occupied to within `e^{−160}` and the entropy is identically zero — so the measurement pins
/// the force's accuracy and cannot distinguish the two functionals. It is recorded rather than
/// asserted away: the distinction becomes real only for a partially occupied system, and this
/// test would not catch a driver that got it wrong there.
#[test]
fn the_divide_and_conquer_force_is_the_derivative_of_its_energy() {
    let molecule = water_trimer();
    let params = Am1Parameters::standard().unwrap();
    let options = Am1Options::default();
    let dc_opts = dc_options();

    let reference = run_divide_conquer(&molecule, &params, &options, &dc_opts).unwrap();
    assert!(reference.converged);
    let analytic = divide_conquer_gradient(&molecule, &params, &options, &reference).unwrap();

    // One atom, one axis: enough, and each evaluation is a full divide-and-conquer SCF.
    let (atom, h) = (1usize, 1.0e-3);
    let displaced = |delta: f64| {
        let mut m = molecule.clone();
        m.atoms[atom].position.x += delta;
        run_divide_conquer(&m, &params, &options, &dc_opts).unwrap()
    };
    let (plus, minus) = (displaced(h), displaced(-h));
    let d_energy = (plus.total_ev - minus.total_ev) / (2.0 * h);
    let d_free = (plus.free_energy_ev() - minus.free_energy_ev()) / (2.0 * h);
    let g = analytic[atom].x;
    eprintln!(
        "analytic {g:.8}  dE/dx {d_energy:.8}  dF/dx {d_free:.8}  \
         (entropy {:.3e} eV)",
        reference.entropy_ev
    );
    assert!(
        (g - d_energy).abs() < 1.0e-4,
        "analytic {g:.8} against finite difference {d_energy:.8}"
    );
}

/// Three stretched waters, far enough apart to partition and stiff enough to relax.
///
/// The intramolecular O–H stretch is the stiffest mode a water has and the intermolecular ones
/// are the flattest, so starting from stretched bonds at a wide separation makes the force
/// overwhelmingly intramolecular: the relaxation has a large force to remove and a well
/// conditioned direction to remove it along. The plain trimer above does not, which is why the
/// ordinary optimizer does not converge on it either.
fn stretched_trimer() -> Molecule {
    let unit = [
        (8, 0.0, 0.0, 0.0),
        (1, 1.25, 0.0, 0.0),
        (1, -0.31, 1.21, 0.0),
    ];
    let shifts = [(0.0, 0.0, 0.0), (0.0, 0.0, 8.0), (8.0, 0.0, 0.0)];
    let mut atoms = Vec::new();
    for (sx, sy, sz) in shifts {
        for (z, x, y, zc) in unit {
            atoms.push(Atom {
                z,
                position: Vec3::new((x + sx) * ANG, (y + sy) * ANG, (zc + sz) * ANG),
            });
        }
    }
    Molecule::new(atoms)
}

/// The relaxation descends, monotonically, and removes most of the force.
///
/// Monotonicity is what the Armijo condition guarantees and the first thing to break if the line
/// search and the gradient disagree about which function they are working on. The force reduction
/// is the second: a driver that lowers the energy without reducing the force is walking along a
/// valley, which is what a mis-scaled quasi-Newton step looks like from the outside.
#[test]
fn a_divide_and_conquer_relaxation_descends_and_removes_the_force() {
    let molecule = stretched_trimer();
    let params = Am1Parameters::standard().unwrap();
    let options = Am1Options::default();
    let dc_opts = dc_options();
    let opt = OptOptions {
        max_iter: 80,
        ..OptOptions::default()
    };

    let res = optimize_divide_conquer(&molecule, &params, &options, &dc_opts, &opt).unwrap();
    assert!(res.dc.subsystems > 1, "the partition did nothing");
    let first = res.trajectory.first().unwrap();
    let last = res.trajectory.last().unwrap();
    eprintln!(
        "{} subsystems, {} steps, dE = {:.6} eV, max|g| {:.3e} -> {:.3e} eV/Bohr, converged {}",
        res.dc.subsystems,
        res.iterations,
        last.energy_ev - first.energy_ev,
        first.max_gradient,
        last.max_gradient,
        res.converged
    );
    assert!(
        last.energy_ev < first.energy_ev,
        "the relaxation went uphill"
    );
    assert!(
        last.max_gradient < 0.02 * first.max_gradient,
        "the force barely moved: {:.3e} -> {:.3e}",
        first.max_gradient,
        last.max_gradient
    );
    for pair in res.trajectory.windows(2) {
        assert!(
            pair[1].energy_ev <= pair[0].energy_ev + 1.0e-6,
            "the trajectory went uphill: {} -> {}",
            pair[0].energy_ev,
            pair[1].energy_ev
        );
    }
    assert_eq!(res.molecule.atoms.len(), molecule.atoms.len());
}

/// One subsystem, aufbau filling: the same *calculation* the ordinary path performs.
///
/// With the partition switched off — every atom in one core, a buffer nothing is outside of —
/// divide-and-conquer is a full SCF, so its energy and its force must be the full SCF's. That is
/// the claim, and it is made here at a fixed geometry rather than by comparing two relaxations
/// step for step: on a water trimer the intermolecular surface is flat enough that a `1e-9`
/// difference in converged density sends the two paths 0.18 Bohr apart in twenty-five steps
/// (measured), which says nothing about either driver.
#[test]
fn one_subsystem_reproduces_the_full_scf_energy_and_force() {
    let molecule = stretched_trimer();
    let params = Am1Parameters::standard().unwrap();
    let options = Am1Options::default();
    let whole = DcOptions {
        core_size: 1000,
        buffer_radius: 1.0e4,
        // Aufbau, to match what the molecular SCF does; a finite `kt` would be a genuine and
        // correct difference and would mask the thing under test.
        filling: Filling::Aufbau,
        e_tol: 1.0e-10,
        p_tol: 1.0e-9,
        ..DcOptions::default()
    };
    let dc = run_divide_conquer(&molecule, &params, &options, &whole).unwrap();
    assert_eq!(dc.subsystems, 1);
    let dc_g = divide_conquer_gradient(&molecule, &params, &options, &dc).unwrap();
    let full = am1_rs::closed_form_gradient(&molecule, &params, &options).unwrap();

    let de = (dc.total_ev - full.scf.total_ev).abs();
    let dg = dc_g
        .iter()
        .zip(&full.gradient)
        .flat_map(|(a, b)| [(a.x - b.x).abs(), (a.y - b.y).abs(), (a.z - b.z).abs()])
        .fold(0.0_f64, f64::max);
    eprintln!("one subsystem: dE = {de:.3e} eV, dg = {dg:.3e} eV/Bohr");
    assert!(de < 1.0e-6, "energies differ by {de:.3e} eV");
    assert!(dg < 1.0e-6, "gradients differ by {dg:.3e} eV/Bohr");

    // And one step of each driver lands in the same place, which is the loop-level claim.
    let opt = OptOptions {
        max_iter: 1,
        ..OptOptions::default()
    };
    let a = optimize_divide_conquer(&molecule, &params, &options, &whole, &opt).unwrap();
    let b = optimize(&molecule, &params, &options, &opt).unwrap();
    let dx = a
        .molecule
        .atoms
        .iter()
        .zip(&b.molecule.atoms)
        .map(|(p, q)| (p.position - q.position).norm())
        .fold(0.0_f64, f64::max);
    eprintln!("after one step: dx = {dx:.3e} Bohr");
    // Not zero, and the bound is read off the quantity above rather than tuned: the two SCFs
    // stop at different tolerances, their gradients differ by 6e-7 eV/Bohr, and the first
    // L-BFGS step scales as `0.1/max|g|`, so a difference of order 1e-8 Bohr is what that
    // gradient difference has to produce. Two orders of headroom on it.
    assert!(dx < 1.0e-6, "one step apart by {dx:.3e} Bohr");
}
/// Every front end resolves `--dc` to the same partition.
///
/// [`DcOptions::default`] is the Rust library's, `native.divide_conquer`'s signature is the
/// Python surface's, and `AM1(divide_conquer=True)` is the ASE calculator's. They are three
/// hand-written copies of the same four numbers, and they had drifted: the library said
/// `core_size 8, buffer 12.0 Bohr, kt 0.1, max_scf 200` while both Python layers said
/// `12, 11.0, 0.05, 300`. Nothing failed, because every test passes its own values — what it
/// produced was two command lines running different calculations and reporting Fermi energies
/// half a Hartree apart.
///
/// This test reads the other two out of the source and compares. A comparison through the
/// bindings would need the Python interpreter; the defaults are literals in a signature and in a
/// dict, and reading them is what makes the check possible from the Rust suite at all.
#[test]
fn the_front_ends_agree_on_the_default_partition() {
    let d = DcOptions::default();
    let binding = include_str!("../src/python.rs");
    let needle = format!(
        "core_size={}, buffer_radius={:.1}, smearing_ev={}",
        d.core_size,
        d.buffer_radius,
        match d.filling {
            Filling::Fermi { kt } => format!("{kt}"),
            Filling::Aufbau => "0.0".to_string(),
        }
    );
    assert!(
        binding.contains(&needle),
        "`native.divide_conquer`'s signature does not carry `{needle}`; it and \
         `DcOptions::default()` have drifted apart"
    );
    assert!(
        binding.contains(&format!("max_scf={}", d.max_scf)),
        "the binding's max_scf default is not {}",
        d.max_scf
    );

    // `python/am1_rs/native.py` is a hand-written wrapper in front of the compiled binding, with
    // its own copy of every default and a *positional* forwarding call. It is the layer that
    // silently swallowed `optimize=` while the binding underneath accepted it — a keyword the
    // wrapper does not list is a `TypeError` from the wrapper, however current the extension is.
    let wrapper = include_str!("../python/am1_rs/native.py");
    for (name, value) in [
        ("core_size: int", format!("{}", d.core_size)),
        ("buffer_radius: float", format!("{:.1}", d.buffer_radius)),
        ("max_scf: int", format!("{}", d.max_scf)),
    ] {
        let want = format!("{name} = {value}");
        assert!(
            wrapper.contains(&want),
            "python/am1_rs/native.py does not carry `{want}`"
        );
    }
    assert!(
        wrapper.contains("optimize: bool = False") && wrapper.contains("bool(optimize)"),
        "the wrapper must both accept `optimize` and forward it"
    );

    let ase = include_str!("../python/am1_rs/ase.py");
    assert!(
        ase.contains(&format!("\"core_size\": {}", d.core_size)),
        "the ASE calculator's core_size default is not {}",
        d.core_size
    );
    assert!(
        ase.contains(&format!("\"buffer_radius\": {:.1}", d.buffer_radius)),
        "the ASE calculator's buffer_radius default is not {:.1}",
        d.buffer_radius
    );
}
