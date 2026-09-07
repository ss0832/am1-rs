// SPDX-License-Identifier: GPL-3.0-or-later
//! `am1_rs_cli` — command-line front end for am1-rs.
//!
//! Modes: `energy`, `gradient`, `optimize`, `frequencies`, `charges`, `orbitals`, `ir`,
//! `molden`, `phonons`. Native output is in atomic units (Hartree, Bohr) throughout, including
//! forces; heats of formation are additionally reported in kcal/mol. eV/Å is reserved for the ASE
//! boundary.
//!
//! **Every printed string is ASCII**, and must stay that way. This front end writes UTF-8 bytes
//! whatever the locale, so it cannot fail on a non-ASCII character — but `python/am1_rs/__main__.py`
//! mirrors it and encodes with the *locale's* codec, where a `cp932` or `C` locale raises
//! `UnicodeEncodeError` part-way through the output. Units are spelled `e*a0`, `cm^-1`, `eV/A` so
//! that both front ends produce the same bytes on every platform. Comments are exempt; they are
//! not printed.
//!
//! # Periodic boundary conditions
//!
//! A cell comes from the extended-XYZ comment line (`Lattice="…"`, `pbc="T T F"`) or from
//! `--cell`, and which axes are periodic from `--pbc`/`--pbc-x`/`--pbc-y`/`--pbc-z`. When one is
//! present, `energy`, `gradient`, `optimize` and `frequencies` take the periodic path — k-point
//! SCF, analytic stress, and phonons rather than molecular vibrations. `charges` (AM1-BCC),
//! `ir` and `molden` are molecular-only and say so rather than silently ignoring the cell.

use am1_rs::bcc::{am1_bcc_charges, write_mol2};
use am1_rs::constants::{
    ANGSTROM_TO_BOHR, AU_DIPOLE_TO_DEBYE, BOHR_TO_ANGSTROM, EV_TO_HARTREE, HARTREE_TO_EV,
    KCAL_TO_EV,
};
use am1_rs::divide_conquer::{
    divide_conquer_gradient, optimize_divide_conquer, run_divide_conquer, DcOptions,
};
use am1_rs::gradient::closed_form_gradient;
use am1_rs::gto::{DEFAULT_NGAUSS, MAX_NGAUSS};
use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::method::NddoMethod;
use am1_rs::molden::{MoldenBasis, MoldenOptions};
use am1_rs::optimizer::{optimize, OptOptions};
use am1_rs::params::Am1Parameters;
use am1_rs::pbc::optimizer::{optimize_periodic, PbcOptOptions};
use am1_rs::pbc::phonon::ForceConstants;
use am1_rs::pbc::{pbc_energy_and_gradient, run_pbc_scf, KMesh, KPoint, PbcOptions, PbcResult};
use am1_rs::scf::{run_am1, Am1Options, ScfReference};
use am1_rs::system::{z_to_symbol, Molecule};
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
        exit(1);
    }
    let mode = args[1].clone();
    let path = args[2].clone();

    let mut cli = Cli {
        mode: &mode,
        path: &path,
        method: NddoMethod::default(),
        charge: 0.0,
        multiplicity: 1,
        reference: ScfReference::Auto,
        opt_output: None,
        mol2_output: None,
        molden_output: None,
        molden_basis: MoldenBasis::Gaussian(DEFAULT_NGAUSS),
        molden_deorthogonalize: true,
        orbital_coefficients: false,
        field: None,
        use_bcc: true,
        cell: None,
        pbc: None,
        kpts: [1, 1, 1],
        supercell: [2, 2, 2],
        supercell_explicit: false,
        qpoints: None,
        qpath: None,
        qpath_points: 20,
        smearing: 0.0,
        max_scf: None,
        relax_cell: false,
        pressure: 0.0,
        dc: false,
        dc_core: DcOptions::default().core_size,
        dc_buffer: None,
    };
    let mut primitives = DEFAULT_NGAUSS;
    let mut molden_slater = false;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--field" => {
                // Three values, in atomic units (Hartree per e·Bohr), matching the native
                // Python surface rather than ASE's V/Å.
                let x: f64 = parse_next(&args, &mut i, "--field");
                let y: f64 = parse_next(&args, &mut i, "--field");
                let z: f64 = parse_next(&args, &mut i, "--field");
                cli.field = Some(Vec3::new(x, y, z) * HARTREE_TO_EV);
            }
            "--molden-output" => cli.molden_output = Some(next(&args, &mut i, "--molden-output")),
            "--molden-basis" => {
                let s = next(&args, &mut i, "--molden-basis");
                molden_slater = match s.trim().to_ascii_lowercase().as_str() {
                    "gto" | "gaussian" => false,
                    "sto" | "slater" => true,
                    other => {
                        eprintln!("invalid --molden-basis value: {other} (expected gto|sto)");
                        exit(1);
                    }
                };
            }
            "--molden-primitives" => {
                primitives = parse_next::<f64>(&args, &mut i, "--molden-primitives") as usize;
                if !(1..=MAX_NGAUSS).contains(&primitives) {
                    eprintln!("--molden-primitives must be between 1 and {MAX_NGAUSS}");
                    exit(1);
                }
            }
            "--molden-orthogonal" => cli.molden_deorthogonalize = false,
            "--orbital-coefficients" => cli.orbital_coefficients = true,
            "--method" => {
                let s = next(&args, &mut i, "--method");
                cli.method = match NddoMethod::parse(&s) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("{e}");
                        exit(1);
                    }
                };
            }
            "--charge" => cli.charge = parse_next(&args, &mut i, "--charge"),
            "--multiplicity" | "--spin-multiplicity" => {
                cli.multiplicity = parse_next::<f64>(&args, &mut i, "--multiplicity") as usize;
            }
            "--reference" | "--ref" => {
                let s = next(&args, &mut i, "--reference");
                cli.reference = match s.trim().to_ascii_lowercase().as_str() {
                    "auto" => ScfReference::Auto,
                    "rhf" | "r" | "restricted" => ScfReference::Restricted,
                    "uhf" | "u" | "unrestricted" => ScfReference::Unrestricted,
                    other => {
                        eprintln!("invalid --reference value: {other} (expected auto|rhf|uhf)");
                        exit(1);
                    }
                };
            }
            "--rhf" => cli.reference = ScfReference::Restricted,
            "--uhf" => cli.reference = ScfReference::Unrestricted,
            "--opt-output" => cli.opt_output = Some(next(&args, &mut i, "--opt-output")),
            "--mol2-output" => cli.mol2_output = Some(next(&args, &mut i, "--mol2-output")),
            "--mulliken" => cli.use_bcc = false,
            "--cell" => {
                let mut values = Vec::new();
                while i + 1 < args.len() && args[i + 1].parse::<f64>().is_ok() {
                    i += 1;
                    values.push(args[i].parse::<f64>().unwrap());
                }
                cli.cell = Some(match cell_from(&values) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("{e}");
                        exit(1);
                    }
                });
            }
            "--pbc" => {
                let s = next(&args, &mut i, "--pbc");
                cli.pbc = Some(merge_pbc(
                    cli.pbc,
                    match parse_pbc(&s) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("{e}");
                            exit(1);
                        }
                    },
                ));
            }
            // Per-axis switches, so a slab or a chain can be asked for without spelling a
            // combined token. They accumulate, and they compose with `--pbc`.
            "--pbc-x" => cli.pbc = Some(merge_pbc(cli.pbc, [true, false, false])),
            "--pbc-y" => cli.pbc = Some(merge_pbc(cli.pbc, [false, true, false])),
            "--pbc-z" => cli.pbc = Some(merge_pbc(cli.pbc, [false, false, true])),
            "--no-pbc" => cli.pbc = Some([false, false, false]),
            "--kpts" => {
                for k in 0..3 {
                    cli.kpts[k] = parse_next::<f64>(&args, &mut i, "--kpts") as usize;
                }
            }
            "--supercell" => {
                for k in 0..3 {
                    cli.supercell[k] = parse_next::<f64>(&args, &mut i, "--supercell") as usize;
                }
                cli.supercell_explicit = true;
            }
            // Explicit q points and a straight-line q path. Both take a flat run of numbers, in
            // multiples of three, in **fractional** coordinates of the reciprocal lattice — the
            // same convention `KPoint::fractional` and every q in this crate uses, so a band
            // structure asked for here is the one `ForceConstants::frequencies` computes.
            "--qpoints" => cli.qpoints = Some(collect_triples(&args, &mut i, "--qpoints")),
            "--qpath" => cli.qpath = Some(collect_triples(&args, &mut i, "--qpath")),
            "--qpath-points" => {
                cli.qpath_points = parse_next::<f64>(&args, &mut i, "--qpath-points") as usize;
            }
            "--smearing" => cli.smearing = parse_next(&args, &mut i, "--smearing"),
            "--max-scf" => {
                cli.max_scf = Some(parse_next::<f64>(&args, &mut i, "--max-scf") as usize);
            }
            "--dc" => cli.dc = true,
            "--dc-core" => {
                cli.dc = true;
                cli.dc_core = parse_next::<f64>(&args, &mut i, "--dc-core") as usize;
            }
            "--dc-buffer" => {
                cli.dc = true;
                cli.dc_buffer = Some(parse_next::<f64>(&args, &mut i, "--dc-buffer"));
            }
            "--relax-cell" => cli.relax_cell = true,
            "--pressure" => cli.pressure = parse_next(&args, &mut i, "--pressure"),
            "-h" | "--help" => {
                usage();
                exit(0);
            }
            other => {
                eprintln!("unknown option: {other}");
                usage();
                exit(1);
            }
        }
        i += 1;
    }
    cli.molden_basis = if molden_slater {
        MoldenBasis::Slater
    } else {
        MoldenBasis::Gaussian(primitives)
    };

    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        exit(1);
    }
}

/// Everything the command line resolved to, in one place.
///
/// A struct rather than twenty positional arguments: the list had already reached the point where
/// a call site said nothing about which `Option<String>` was which output file.
struct Cli<'a> {
    mode: &'a str,
    path: &'a str,
    method: NddoMethod,
    charge: f64,
    multiplicity: usize,
    reference: ScfReference,
    opt_output: Option<String>,
    mol2_output: Option<String>,
    molden_output: Option<String>,
    molden_basis: MoldenBasis,
    molden_deorthogonalize: bool,
    orbital_coefficients: bool,
    field: Option<Vec3>,
    use_bcc: bool,
    cell: Option<Lattice>,
    pbc: Option<[bool; 3]>,
    kpts: [usize; 3],
    supercell: [usize; 3],
    /// Whether --supercell was given, as opposed to defaulted. A repeat on a non-periodic axis
    /// is an error when asked for and a silent 1 when it is only the default.
    supercell_explicit: bool,
    /// Explicit q points, fractional. `None` uses the supercell's commensurate set.
    qpoints: Option<Vec<[f64; 3]>>,
    /// Corners of a straight-line q path, fractional.
    qpath: Option<Vec<[f64; 3]>>,
    /// Points per segment of `qpath`.
    qpath_points: usize,
    smearing: f64,
    max_scf: Option<usize>,
    relax_cell: bool,
    pressure: f64,
    /// Run the divide-and-conquer SCF instead of the full one.
    dc: bool,
    /// Target atoms per core region.
    dc_core: usize,
    /// Buffer radius in Angstrom, converted to Bohr when the options are built. None keeps
    /// the library default.
    dc_buffer: Option<f64>,
}

/// `--cell` accepts one, three, six or nine numbers, in Ångström and degrees.
///
/// One is a cube, three an orthorhombic cell, six the crystallographic `a b c alpha beta gamma`,
/// and nine the three lattice vectors written out. Anything else is rejected rather than
/// interpreted: a four-number cell is a typo, and guessing which four would be worse than saying
/// so.
fn cell_from(values: &[f64]) -> Result<Lattice, String> {
    let all = [true, true, true];
    let a = ANGSTROM_TO_BOHR;
    match values.len() {
        1 => Lattice::cubic(values[0] * a),
        3 => Lattice::from_vectors(
            Vec3::new(values[0] * a, 0.0, 0.0),
            Vec3::new(0.0, values[1] * a, 0.0),
            Vec3::new(0.0, 0.0, values[2] * a),
            all,
        ),
        6 => Lattice::from_lengths_angles(
            values[0] * a,
            values[1] * a,
            values[2] * a,
            values[3],
            values[4],
            values[5],
            all,
        ),
        9 => Lattice::from_vectors(
            Vec3::new(values[0] * a, values[1] * a, values[2] * a),
            Vec3::new(values[3] * a, values[4] * a, values[5] * a),
            Vec3::new(values[6] * a, values[7] * a, values[8] * a),
            all,
        ),
        n => {
            return Err(format!(
                "--cell takes 1, 3, 6 or 9 numbers (cube; a b c; a b c alpha beta gamma; or three \
                 lattice vectors), got {n}"
            ))
        }
    }
    .map_err(|e| format!("{e}"))
}

/// `--pbc x`, `--pbc xy`, `--pbc x,z`, `--pbc xyz`, `--pbc none`.
fn parse_pbc(text: &str) -> Result<[bool; 3], String> {
    let cleaned = text.trim().to_ascii_lowercase();
    if cleaned == "none" || cleaned == "false" || cleaned == "f" {
        return Ok([false, false, false]);
    }
    if cleaned == "all" || cleaned == "true" || cleaned == "t" {
        return Ok([true, true, true]);
    }
    let mut out = [false; 3];
    for ch in cleaned.chars() {
        match ch {
            'x' => out[0] = true,
            'y' => out[1] = true,
            'z' => out[2] = true,
            ',' | ' ' | '+' => {}
            other => {
                return Err(format!(
                    "invalid --pbc value: '{other}' in '{text}' (expected x, y, z, none, or a \
                     combination such as xy)"
                ))
            }
        }
    }
    Ok(out)
}

/// Per-axis switches accumulate rather than replace, so `--pbc-x --pbc-z` is a two-dimensional
/// slab in the xz plane and not just `z`.
fn merge_pbc(current: Option<[bool; 3]>, add: [bool; 3]) -> [bool; 3] {
    let base = current.unwrap_or([false, false, false]);
    [base[0] || add[0], base[1] || add[1], base[2] || add[2]]
}

/// A value that rounds to zero at `decimals` places, printed without a sign.
///
/// `-0.0` and `0.0` are the same number at any printed precision, but they are different *text*,
/// and the CLI's text is compared against the Python front end's in `tests/test_cli.py`. The two
/// take different routes to the same eigenvalue, so a quantity that is zero — a residual
/// rigid-body frequency, an atomic polar tensor element forbidden by symmetry — can land on either
/// side of it and make the two disagree about a number they both agree is zero.
fn unsigned_zero(v: f64, decimals: i32) -> f64 {
    if v.abs() < 0.5 * 10f64.powi(-decimals) {
        0.0
    } else {
        v
    }
}

fn run(cli: Cli<'_>) -> am1_rs::Result<()> {
    let mut molecule = Molecule::from_xyz_file(cli.path, cli.charge)?;
    molecule.multiplicity = cli.multiplicity;

    // A cell on the command line replaces one from the file; `--pbc` replaces the axes of
    // whichever cell is in play. Neither silently invents the other.
    if let Some(cell) = cli.cell {
        molecule.cell = Some(cell);
    }
    if let Some(pbc) = cli.pbc {
        match molecule.cell {
            Some(existing) => {
                molecule.cell = Some(Lattice::from_vectors(
                    existing.cell.col[0],
                    existing.cell.col[1],
                    existing.cell.col[2],
                    pbc,
                )?);
            }
            None if pbc.iter().any(|p| *p) => {
                eprintln!(
                    "error: --pbc asks for a periodic direction but there is no cell; give one \
                     with --cell or a Lattice=\"...\" comment line in the XYZ file"
                );
                exit(1);
            }
            None => {}
        }
    }
    if molecule.cell.map(|c| c.n_periodic()) == Some(0) {
        molecule.cell = None;
    }
    let periodic = molecule.cell.is_some();

    let params = Am1Parameters::for_method(cli.method)?;
    let scf_opts = Am1Options {
        charge: cli.charge,
        multiplicity: cli.multiplicity,
        reference: cli.reference,
        electric_field: cli.field,
        ..Am1Options::default()
    };
    let dc_opts = dc_options_from(&cli);
    // Refused, not ignored. `--dc` on `frequencies` would return full-SCF frequencies, and the
    // only sign that the flag did nothing would be the run being no faster — which is exactly
    // the observation a user reaches for `--dc` to avoid having to make. There is no
    // divide-and-conquer Hessian, no divide-and-conquer polar tensor and no wavefunction file
    // for a partitioned density, so those modes say so.
    if cli.dc && !matches!(cli.mode, "energy" | "gradient" | "optimize") {
        return Err(am1_rs::error::Am1Error::InvalidInput(format!(
            "--dc has no effect in `{}` mode; it applies to energy, gradient and optimize. \
             (There is no divide-and-conquer second derivative: the Hessian needs the coupled \
              response of the whole system, which is what the partition does not have.)",
            cli.mode
        )));
    }
    let pbc_opts = PbcOptions {
        kmesh: KMesh::MonkhorstPack(cli.kpts),
        charge: cli.charge,
        multiplicity: cli.multiplicity,
        unrestricted: matches!(cli.reference, ScfReference::Unrestricted),
        smearing_ev: cli.smearing,
        electric_field: cli.field,
        max_scf: cli.max_scf.unwrap_or(PbcOptions::default().max_scf),
        ..PbcOptions::default()
    };

    match cli.mode {
        "energy" => {
            if cli.dc {
                let r = run_divide_conquer(&molecule, &params, &scf_opts, &dc_opts)?;
                print_dc_energy(&molecule, &r);
            } else if periodic {
                let scf = run_pbc_scf(&molecule, &params, &pbc_opts)?;
                require_converged(&scf)?;
                print_pbc_energy(&molecule, &scf);
            } else {
                let r = run_am1(&molecule, &params, &scf_opts)?;
                print_energy(&molecule, &r);
            }
        }
        "gradient" => {
            if cli.dc {
                let r = run_divide_conquer(&molecule, &params, &scf_opts, &dc_opts)?;
                let grad = divide_conquer_gradient(&molecule, &params, &scf_opts, &r)?;
                print_dc_energy(&molecule, &r);
                println!("\nforces (Hartree/Bohr):");
                let mut max = 0.0_f64;
                for (a, g) in molecule.atoms.iter().zip(&grad) {
                    max = max.max(g.x.abs()).max(g.y.abs()).max(g.z.abs());
                    println!(
                        "  {:<2}  {:14.8} {:14.8} {:14.8}",
                        z_to_symbol(a.z).unwrap_or("?"),
                        -g.x * EV_TO_HARTREE,
                        -g.y * EV_TO_HARTREE,
                        -g.z * EV_TO_HARTREE
                    );
                }
                println!("max |grad| = {:.6e} Hartree/Bohr", max * EV_TO_HARTREE);
            } else if periodic {
                let (scf, g) = pbc_energy_and_gradient(&molecule, &params, &pbc_opts)?;
                require_converged(&scf)?;
                print_pbc_energy(&molecule, &scf);
                println!("\nforces (Hartree/Bohr):");
                for (a, f) in molecule.atoms.iter().zip(&g.forces) {
                    println!(
                        "  {:<2}  {:14.8} {:14.8} {:14.8}",
                        z_to_symbol(a.z).unwrap_or("?"),
                        f.x * EV_TO_HARTREE,
                        f.y * EV_TO_HARTREE,
                        f.z * EV_TO_HARTREE
                    );
                }
                println!(
                    "max |grad| = {:.6e} Hartree/Bohr",
                    g.max_gradient * EV_TO_HARTREE
                );
                print_stress(
                    &g.stress_voigt(),
                    g.pressure(molecule.cell.unwrap().n_periodic()),
                );
            } else {
                let g = closed_form_gradient(&molecule, &params, &scf_opts)?;
                print_energy(&molecule, &g.scf);
                // Forces, not the gradient: the force is minus the energy derivative.
                println!("\nforces (Hartree/Bohr):");
                for (a, f) in molecule.atoms.iter().zip(&g.forces) {
                    println!(
                        "  {:<2}  {:14.8} {:14.8} {:14.8}",
                        z_to_symbol(a.z).unwrap_or("?"),
                        f.x * EV_TO_HARTREE,
                        f.y * EV_TO_HARTREE,
                        f.z * EV_TO_HARTREE
                    );
                }
                println!(
                    "max |grad| = {:.6e} Hartree/Bohr",
                    g.max_gradient * EV_TO_HARTREE
                );
            }
        }
        "optimize" => {
            if cli.dc {
                let res = optimize_divide_conquer(
                    &molecule,
                    &params,
                    &scf_opts,
                    &dc_opts,
                    &OptOptions::default(),
                )?;
                println!(
                    "optimization {} in {} steps",
                    if res.converged {
                        "converged"
                    } else {
                        "did NOT converge"
                    },
                    res.iterations
                );
                print_dc_energy(&res.molecule, &res.dc);
                println!(
                    "max |force| = {:.6e} Hartree/Bohr",
                    res.trajectory.last().map(|s| s.max_gradient).unwrap_or(0.0) * EV_TO_HARTREE
                );
                let xyz = to_xyz(&res.molecule, "am1-rs optimized (divide-and-conquer)");
                write_or_print_geometry(cli.opt_output, &xyz)?;
            } else if periodic {
                let opt = PbcOptOptions {
                    relax_cell: cli.relax_cell,
                    pressure: cli.pressure,
                    ..PbcOptOptions::default()
                };
                let res = optimize_periodic(&molecule, &params, &pbc_opts, &opt)?;
                println!(
                    "optimization {} in {} steps",
                    if res.converged {
                        "converged"
                    } else {
                        "did NOT converge"
                    },
                    res.iterations
                );
                require_converged(&res.scf)?;
                print_pbc_energy(&res.molecule, &res.scf);
                println!(
                    "max |force| = {:.6e} Hartree/Bohr",
                    res.gradient.max_gradient * EV_TO_HARTREE
                );
                let cell = res.molecule.cell.unwrap();
                print_stress(
                    &res.gradient.stress_voigt(),
                    res.gradient.pressure(cell.n_periodic()),
                );
                print_cell(&cell);
                let xyz = to_xyz(&res.molecule, "am1-rs optimized");
                write_or_print_geometry(cli.opt_output, &xyz)?;
            } else {
                let res = optimize(&molecule, &params, &scf_opts, &OptOptions::default())?;
                println!(
                    "optimization {} in {} steps",
                    if res.converged {
                        "converged"
                    } else {
                        "did NOT converge"
                    },
                    res.iterations
                );
                print_energy(&res.molecule, &res.scf);
                let xyz = to_xyz(&res.molecule, "am1-rs optimized");
                write_or_print_geometry(cli.opt_output, &xyz)?;
            }
        }
        "frequencies" => {
            if periodic {
                // Gamma phonons of the cell itself: one primitive cell, one q point. The acoustic
                // sum rule is imposed, and its violation reported first, because that number is
                // the honest measure of what imposing it moved.
                let mut fc = ForceConstants::from_supercell(
                    &molecule,
                    &params,
                    &pbc_options_to_scf(&pbc_opts, &scf_opts),
                    [1, 1, 1],
                )?;
                let before = fc.acoustic_sum_rule_error();
                fc.enforce_acoustic_sum_rule();
                let after = fc.acoustic_sum_rule_error();
                let freqs = fc.frequencies(gamma())?;
                println!(
                    "phonon frequencies at q = 0 (cm^-1), {} modes:",
                    freqs.len()
                );
                for (i, f) in freqs.iter().enumerate() {
                    println!("  {:>3}  {:>10.1}", i + 1, unsigned_zero(*f, 1));
                }
                print_asr(before, after);
            } else {
                let vib =
                    am1_rs::hessian::vibrational_analysis(&molecule, &params, &scf_opts, 1.0e-3)?;
                println!(
                    "harmonic vibrational frequencies (cm^-1), {} modes \
                     ({} rigid-body removed):",
                    vib.frequencies_cm.len(),
                    vib.rigid_body_count
                );
                for (i, f) in vib.frequencies_cm.iter().enumerate() {
                    // Anything that rounds to zero at this precision prints as `0.0`, never
                    // `-0.0`; see `unsigned_zero`.
                    println!("  {:>3}  {:>10.1}", i + 1, unsigned_zero(*f, 1));
                }
                // What the projection removed, so that a geometry which is not a stationary point
                // announces itself instead of hiding behind a "translation/rotation" label.
                let residual = vib
                    .rigid_body_frequencies_cm
                    .iter()
                    .fold(0.0_f64, |m, f| m.max(f.abs()));
                println!(
                    "\nrigid-body residual: {:>10.1} cm^-1 (0 at a stationary point)",
                    unsigned_zero(residual, 1)
                );
                println!("(compute at an optimized geometry for meaningful frequencies)");
            }
        }
        "phonons" => {
            let Some(_) = molecule.cell else {
                eprintln!(
                    "error: phonons need a periodic cell; give one with --cell or a \
                     Lattice=\"...\" comment line, or use `frequencies` for a molecule"
                );
                exit(1);
            };
            // A repeat along a direction that is not periodic is meaningless, and the default
            // `2 2 2` would therefore refuse every slab and every chain — which are the systems
            // this crate handles best. Clamping is what makes the default usable; an *explicit*
            // repeat on a non-periodic axis is still an error, because that is a mistake worth
            // naming rather than silently ignoring.
            // A repeat along a direction that is not periodic is meaningless, and the default
            // `2 2 2` would therefore refuse every slab and every chain — which are the systems
            // this crate handles best. Clamping is what makes the default usable; an *explicit*
            // repeat on a non-periodic axis is still an error, because that is a mistake worth
            // naming rather than silently ignoring.
            let cell = molecule.cell.expect("checked above");
            let mut repeats = cli.supercell;
            for (axis, repeat) in repeats.iter_mut().enumerate() {
                if !cell.periodic[axis] {
                    if cli.supercell_explicit && *repeat > 1 {
                        eprintln!(
                            "error: --supercell asks for {} repeats along axis {axis}, which is \
                             not periodic",
                            *repeat
                        );
                        exit(1);
                    }
                    *repeat = 1;
                }
            }
            let mut fc = ForceConstants::from_supercell(
                &molecule,
                &params,
                &pbc_options_to_scf(&pbc_opts, &scf_opts),
                repeats,
            )?;
            let before = fc.acoustic_sum_rule_error();
            fc.enforce_acoustic_sum_rule();
            let after = fc.acoustic_sum_rule_error();
            let (requested, note) = resolve_q(&cli, &fc);
            println!(
                "phonon frequencies (cm^-1) from a {}x{}x{} supercell:",
                cli.supercell[0], cli.supercell[1], cli.supercell[2]
            );
            for q in requested {
                let freqs = fc.frequencies(q)?;
                let row: Vec<String> = freqs
                    .iter()
                    .map(|f| format!("{:>10.1}", unsigned_zero(*f, 1)))
                    .collect();
                println!(
                    "  q = ({:6.3} {:6.3} {:6.3}) {}",
                    q.fractional[0],
                    q.fractional[1],
                    q.fractional[2],
                    row.join(" ")
                );
            }
            print_asr(before, after);
            println!("({note})");
        }
        "orbitals" => {
            let r = run_am1(&molecule, &params, &scf_opts)?;
            println!("orbital energies (Hartree), {} occupied:", r.n_occ);
            print_orbitals(
                &r.mo_energies,
                r.n_occ,
                if r.unrestricted { "alpha" } else { "" },
            );
            if let Some(b) = &r.beta {
                println!("\nbeta channel, {} occupied:", b.n_occ);
                print_orbitals(&b.energies, b.n_occ, "beta");
            }
            print_frontier(&r);
            if cli.orbital_coefficients {
                let labels = ao_labels(&molecule, &params)?;
                println!("\nMO coefficients (rows are AOs, columns are orbitals) [alpha]:");
                print_coefficients(&labels, &r.mo_coeff);
                if let Some(b) = &r.beta {
                    println!("\nMO coefficients [beta]:");
                    print_coefficients(&labels, &b.coeff);
                }
            }
        }
        "ir" => {
            reject_cell(periodic, "ir")?;
            let s = am1_rs::ir::ir_spectrum(&molecule, &params, &scf_opts)?;
            println!("atomic polar tensor d(mu_a)/d(R_b) (e), rows x/y/z, columns 3*atom+axis:");
            for a in 0..3 {
                let row: Vec<String> = (0..s.dipole_derivatives.cols)
                    .map(|j| format!("{:9.5}", unsigned_zero(s.dipole_derivatives[(a, j)], 5)))
                    .collect();
                println!("  {}", row.join(" "));
            }
            println!("\ninfrared spectrum:");
            println!("  mode   freq (cm^-1)   intensity (km/mol)   rigid-body");
            for (k, f) in s.frequencies_cm.iter().enumerate() {
                println!(
                    "  {:>4}  {:>13.2}  {:>19.4}  {:>10.3}",
                    k + 1,
                    unsigned_zero(*f, 2),
                    s.intensities_km_per_mol[k],
                    s.modes.translation_rotation_overlap[k]
                );
            }
            println!(
                "\n({} rigid-body modes were projected out; the column above measures what is \
                 left of them and must be 0)",
                s.modes.rigid_body_count
            );
        }
        "molden" => {
            reject_cell(periodic, "molden")?;
            let r = run_am1(&molecule, &params, &scf_opts)?;
            let text = am1_rs::molden::to_molden_with(
                &molecule,
                &params,
                &r,
                &MoldenOptions {
                    basis: cli.molden_basis,
                    deorthogonalize: cli.molden_deorthogonalize,
                },
            )?;
            match cli.molden_output {
                Some(out) => {
                    std::fs::write(&out, text)?;
                    println!("molden wavefunction written to {out}");
                }
                None => print!("{text}"),
            }
        }
        "charges" => {
            reject_cell(periodic, "charges")?;
            if cli.use_bcc {
                match am1_bcc_charges(&molecule, &params, &scf_opts) {
                    Ok(bcc) => {
                        println!("AM1-BCC charges (e):");
                        for (a, (q, t)) in molecule
                            .atoms
                            .iter()
                            .zip(bcc.charges.iter().zip(&bcc.atom_types))
                        {
                            println!(
                                "  {:<2}  {:+.5}   [type {}]",
                                z_to_symbol(a.z).unwrap_or("?"),
                                q,
                                t
                            );
                        }
                        println!("sum = {:+.5} e", bcc.charges.iter().sum::<f64>());
                        if let Some(out) = cli.mol2_output {
                            write_mol2(&out, &molecule, &bcc)?;
                            println!("\nmol2 written to {out}");
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "AM1-BCC unavailable ({e}); falling back to AM1 Mulliken charges"
                        );
                        let r = run_am1(&molecule, &params, &scf_opts)?;
                        print_charges(&molecule, &r.charges);
                    }
                }
            } else {
                let r = run_am1(&molecule, &params, &scf_opts)?;
                print_charges(&molecule, &r.charges);
            }
        }
        other => {
            eprintln!("unknown mode: {other}");
            usage();
            exit(1);
        }
    }

    // The phase breakdown, once, at the end — after every phase of whatever was asked for has
    // run. No library function reports, because reporting clears the accumulator and would cut
    // the measurement off at its own boundary. See `am1_rs::timing`.
    am1_rs::timing::report(&format!(
        "{}, {} atoms, {}",
        cli.mode,
        molecule.atoms.len(),
        params.method.display_name()
    ));
    Ok(())
}

/// A mode that has no periodic implementation says so rather than quietly dropping the cell.
fn reject_cell(periodic: bool, mode: &str) -> am1_rs::Result<()> {
    if periodic {
        eprintln!(
            "error: `{mode}` is molecular only and the structure has a periodic cell; remove the \
             cell (--no-pbc) to run it on the contents of one cell"
        );
        exit(1);
    }
    Ok(())
}

/// The molecular SCF options the supercell Hessian needs, carrying the periodic cutoffs.
///
/// The Γ Hessian is the *molecular* code walking an image-aware pair list (see
/// `crate::pbc`), so it takes `Am1Options`; the periodic cutoffs and the field have to be carried
/// across by hand or the phonons would be computed with different screening from the SCF.
fn pbc_options_to_scf(pbc: &PbcOptions, base: &Am1Options) -> Am1Options {
    Am1Options {
        realspace_cutoff: pbc.realspace_cutoff,
        exchange_cutoff: pbc.exchange_cutoff,
        ewald: pbc.ewald,
        klopman_ohno_tail: pbc.klopman_ohno_tail,
        max_scf: pbc.max_scf,
        ..base.clone()
    }
}

fn gamma() -> KPoint {
    KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    }
}

/// Consume a run of numbers from the command line as `(x, y, z)` triples.
///
/// Variadic like `--cell`: it stops at the first argument that is not a number, so a flag can
/// follow without a separator. A count that is not a multiple of three is a typo and is refused
/// rather than truncated.
fn collect_triples(args: &[String], i: &mut usize, flag: &str) -> Vec<[f64; 3]> {
    let mut values = Vec::new();
    while *i + 1 < args.len() {
        match args[*i + 1].parse::<f64>() {
            Ok(v) => {
                *i += 1;
                values.push(v);
            }
            Err(_) => break,
        }
    }
    if values.is_empty() || values.len() % 3 != 0 {
        eprintln!(
            "{flag} takes a multiple of three numbers (x y z per point, fractional), got {}",
            values.len()
        );
        exit(1);
    }
    values.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Which q points the `phonons` mode reports, and the one-line note that says why those.
///
/// `--qpoints` wins over `--qpath`, and both over the default. The default is the supercell's
/// **commensurate** set, because those are the only points it represents exactly — everything
/// else is an interpolation of a truncated `Φ(T)`, and it is worth being able to tell the two
/// apart from the output alone.
fn resolve_q(cli: &Cli<'_>, fc: &ForceConstants) -> (Vec<KPoint>, String) {
    let as_points = |v: &[[f64; 3]]| -> Vec<KPoint> {
        v.iter()
            .map(|q| KPoint {
                fractional: *q,
                weight: 1.0,
            })
            .collect()
    };
    if let Some(q) = &cli.qpoints {
        return (
            as_points(q),
            "these are the q points you asked for".to_string(),
        );
    }
    if let Some(corners) = &cli.qpath {
        let path = am1_rs::pbc::q_path(corners, cli.qpath_points.max(1));
        return (
            path,
            format!(
                "a straight-line path through {} corners, {} points per segment; only the q \
                 commensurate with the supercell are exact",
                corners.len(),
                cli.qpath_points.max(1)
            ),
        );
    }
    (
        fc.commensurate_q(),
        "these are the q the supercell represents exactly".to_string(),
    )
}

fn write_or_print_geometry(out: Option<String>, xyz: &str) -> am1_rs::Result<()> {
    match out {
        Some(path) => {
            std::fs::write(&path, xyz)?;
            println!("\noptimized geometry written to {path}");
        }
        None => println!("\noptimized geometry (Angstrom):\n{xyz}"),
    }
    Ok(())
}

fn print_asr(before: f64, after: f64) {
    // Reported both ways round on purpose: imposing the rule guarantees three modes at exactly
    // zero, and the *before* number is the only statement of how much error that moved into the
    // on-site block rather than removed.
    println!("\nacoustic sum rule violation: {before:.3e} -> {after:.3e} eV/Bohr^2 (imposed)");
}

fn print_stress(voigt: &[f64; 6], pressure: f64) {
    println!(
        "stress (Hartree/Bohr^d, Voigt xx yy zz yz xz xy):\n  {}",
        voigt
            .iter()
            .map(|s| format!("{:14.8}", s * EV_TO_HARTREE))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "pressure          : {:16.8} Hartree/Bohr^d",
        pressure * EV_TO_HARTREE
    );
}

fn print_cell(cell: &Lattice) {
    println!("cell (Angstrom), periodic {}:", pbc_text(cell.periodic));
    for v in &cell.cell.col {
        println!(
            "  {:14.8} {:14.8} {:14.8}",
            v.x * BOHR_TO_ANGSTROM,
            v.y * BOHR_TO_ANGSTROM,
            v.z * BOHR_TO_ANGSTROM
        );
    }
}

fn pbc_text(periodic: [bool; 3]) -> String {
    let mut s = String::new();
    for (axis, name) in ["x", "y", "z"].iter().enumerate() {
        if periodic[axis] {
            s.push_str(name);
        }
    }
    if s.is_empty() {
        "none".to_string()
    } else {
        s
    }
}

/// Refuse to report an unconverged periodic result.
///
/// `run_pbc_scf` returns `converged: false` rather than an error, and this front end used to
/// print **"SCF converged in N iterations"** regardless — so an unconverged crystal came back
/// looking like an answer. `native.pbc_point` has always raised here, so the two front ends
/// disagreed about whether the same input was a result or a failure. This is the Rust side
/// agreeing with the Python one.
/// The wording is `native.pbc_point`'s, word for word, so the two front ends fail identically.
/// `PbcResult` carries no residual, so this does not invent one — the previous release printed a
/// hardcoded `NaN` in exactly this position and it cost an afternoon.
fn require_converged(r: &PbcResult) -> am1_rs::Result<()> {
    if !r.converged {
        return Err(am1_rs::Am1Error::InvalidInput(format!(
            "periodic SCF did not converge in {} iterations",
            r.iterations
        )));
    }
    Ok(())
}

fn print_pbc_energy(molecule: &Molecule, r: &PbcResult) {
    if let Some(w) = &r.charged_cell_warning {
        eprintln!("warning: {w}");
    }
    println!(
        "SCF converged in {} iterations ({} k-points{})",
        r.iterations,
        r.k_points,
        if r.unrestricted { ", UHF" } else { "" }
    );
    println!(
        "total energy      : {:16.8} Hartree",
        r.total_ev * EV_TO_HARTREE
    );
    println!(
        "  electronic      : {:16.8} Hartree",
        r.electronic_ev * EV_TO_HARTREE
    );
    println!(
        "  core repulsion  : {:16.8} Hartree",
        r.core_ev * EV_TO_HARTREE
    );
    println!(
        "Fermi energy      : {:16.8} Hartree",
        r.fermi_energy_ev * EV_TO_HARTREE
    );
    println!(
        "max image overlap : {:16.8}   (NDDO assumes 0)",
        r.max_image_overlap
    );
    print_charges(molecule, &r.charges);
}

/// The divide-and-conquer options `--dc`, `--dc-core` and `--dc-buffer` resolve to.
///
/// `--dc-buffer` is taken in Ångström, like every other length on this command line, and
/// converted here; [`DcOptions::buffer_radius`] is in Bohr. Getting that wrong is not an error a
/// user would see — a 12 Å buffer instead of 12 Bohr just makes every subsystem the whole
/// molecule and the run slow but correct — which is exactly why it is written down.
fn dc_options_from(cli: &Cli<'_>) -> DcOptions {
    let mut opts = DcOptions {
        core_size: cli.dc_core.max(1),
        ..DcOptions::default()
    };
    if let Some(r) = cli.dc_buffer {
        opts.buffer_radius = r * ANGSTROM_TO_BOHR;
    }
    opts
}

/// The divide-and-conquer counterpart of [`print_energy`].
///
/// It reports the subsystem count and the largest subsystem's basis alongside the energy, because
/// those are what say whether the partition did anything: one subsystem holding every atom is a
/// full SCF with extra steps, and a user who asked for `--dc` on a small molecule should be able
/// to see that is what they got rather than infer it from the timing.
fn print_dc_energy(molecule: &Molecule, r: &am1_rs::divide_conquer::DcResult) {
    let hf_hartree = r.heat_of_formation_kcal * KCAL_TO_EV * EV_TO_HARTREE;
    println!(
        "divide-and-conquer SCF {} in {} iterations{}",
        if r.converged {
            "converged"
        } else {
            "did NOT converge"
        },
        r.iterations,
        if r.unrestricted { " (UHF)" } else { "" }
    );
    println!(
        "subsystems        : {} (largest {} AOs)",
        r.subsystems, r.largest_subsystem_aos
    );
    println!(
        "total energy      : {:16.8} Hartree",
        r.total_ev * EV_TO_HARTREE
    );
    println!(
        "  electronic      : {:16.8} Hartree",
        r.electronic_ev * EV_TO_HARTREE
    );
    println!(
        "  core repulsion  : {:16.8} Hartree",
        r.core_ev * EV_TO_HARTREE
    );
    println!(
        "heat of formation : {:16.8} Hartree   ({:.6} kcal/mol)",
        hf_hartree, r.heat_of_formation_kcal
    );
    println!(
        "Fermi energy      : {:16.8} Hartree",
        r.fermi_energy_ev * EV_TO_HARTREE
    );
    println!(
        "HOMO-LUMO gap     : {:16.8} Hartree",
        r.homo_lumo_gap_ev * EV_TO_HARTREE
    );
    println!("Mulliken charges (e):");
    for (a, q) in molecule.atoms.iter().zip(&r.charges) {
        println!("  {:<2}  {:+.5}", z_to_symbol(a.z).unwrap_or("?"), q);
    }
    if let Some(w) = &r.small_gap_warning {
        println!("\nwarning: {w}");
    }
}

fn print_energy(molecule: &Molecule, r: &am1_rs::scf::Am1Result) {
    // Native CLI output is in atomic units (Hartree, e·a0). eV/Å is reserved for the ASE API.
    let hf_hartree = r.heat_of_formation_kcal * KCAL_TO_EV * EV_TO_HARTREE;
    let dip_au = |d: f64| d / AU_DIPOLE_TO_DEBYE; // Debye -> e·a0
    println!(
        "SCF converged in {} iterations{}",
        r.iterations,
        if r.unrestricted { " (UHF)" } else { "" }
    );
    println!(
        "total energy      : {:16.8} Hartree",
        r.total_ev * EV_TO_HARTREE
    );
    println!(
        "  electronic      : {:16.8} Hartree",
        r.electronic_ev * EV_TO_HARTREE
    );
    println!(
        "  core repulsion  : {:16.8} Hartree",
        r.core_ev * EV_TO_HARTREE
    );
    println!(
        "heat of formation : {:16.8} Hartree   ({:.6} kcal/mol)",
        hf_hartree, r.heat_of_formation_kcal
    );
    if let (Some(h), Some(l)) = (r.homo_ev, r.lumo_ev) {
        println!(
            "HOMO / LUMO       : {:.6} / {:.6} Hartree  (gap {:.6})",
            h * EV_TO_HARTREE,
            l * EV_TO_HARTREE,
            (l - h) * EV_TO_HARTREE
        );
    }
    println!(
        "dipole            : {:.6} e*a0  ({:.6}, {:.6}, {:.6})",
        dip_au(r.dipole_magnitude),
        dip_au(r.dipole_debye.x),
        dip_au(r.dipole_debye.y),
        dip_au(r.dipole_debye.z)
    );
    print_charges(molecule, &r.charges);
}

/// The frontier orbitals in both eV and Hartree, with the gap.
///
/// Orbital energies are conventionally quoted in eV — Koopmans' theorem makes the HOMO an
/// ionization potential — while the rest of this front end is in Hartree, so both are printed
/// rather than making the reader convert.
fn print_frontier(r: &am1_rs::scf::Am1Result) {
    let line = |name: &str, ev: Option<f64>| {
        if let Some(v) = ev {
            println!(
                "  {name:<12} {:14.8} Hartree  {:12.6} eV",
                v * EV_TO_HARTREE,
                v
            );
        }
    };
    println!("\nfrontier orbitals:");
    line("HOMO", r.homo_ev);
    line("LUMO", r.lumo_ev);
    if let (Some(h), Some(l)) = (r.homo_ev, r.lumo_ev) {
        line("gap", Some(l - h));
    }
    if r.unrestricted {
        line("HOMO [beta]", r.homo_beta_ev);
        line("LUMO [beta]", r.lumo_beta_ev);
    }
}

/// `s`, `px`, `py`, `pz` per atom, in the order [`am1_rs::basis::Basis`] lists them.
fn ao_labels(molecule: &Molecule, params: &Am1Parameters) -> am1_rs::Result<Vec<String>> {
    let basis = am1_rs::basis::Basis::build(molecule, params)?;
    Ok(basis
        .aos
        .iter()
        .map(|ao| {
            let orbital = match ao.orb {
                0 => "s",
                1 => "px",
                2 => "py",
                _ => "pz",
            };
            format!(
                "{}{}:{}",
                z_to_symbol(ao.z).unwrap_or("?"),
                ao.atom + 1,
                orbital
            )
        })
        .collect())
}

fn print_coefficients(labels: &[String], c: &am1_rs::linalg::Matrix) {
    let header: Vec<String> = (0..c.cols).map(|k| format!("{:>10}", k + 1)).collect();
    println!("  {:<10} {}", "AO", header.join(" "));
    for (mu, label) in labels.iter().enumerate() {
        let row: Vec<String> = (0..c.cols)
            .map(|k| format!("{:>10.5}", unsigned_zero(c[(mu, k)], 5)))
            .collect();
        println!("  {label:<10} {}", row.join(" "));
    }
}

/// One spin channel's orbital energies, with the frontier marked.
fn print_orbitals(energies: &[f64], n_occ: usize, spin: &str) {
    let tag = if spin.is_empty() {
        String::new()
    } else {
        format!(" [{spin}]")
    };
    for (i, e) in energies.iter().enumerate() {
        let marker = if i + 1 == n_occ {
            "  <- HOMO"
        } else if i == n_occ {
            "  <- LUMO"
        } else {
            ""
        };
        println!(
            "  {:>4}  {:>14.8}  occ {:.1}{marker}{tag}",
            i + 1,
            e * EV_TO_HARTREE,
            if i < n_occ { 2.0 } else { 0.0 }
        );
    }
}

fn print_charges(molecule: &Molecule, charges: &[f64]) {
    println!("Mulliken charges (e):");
    for (a, q) in molecule.atoms.iter().zip(charges) {
        println!("  {:<2}  {:+.5}", z_to_symbol(a.z).unwrap_or("?"), q);
    }
}

fn to_xyz(molecule: &Molecule, comment: &str) -> String {
    // Extended XYZ when there is a cell, so an optimized periodic structure round-trips straight
    // back into this CLI rather than losing its lattice on the way out.
    let comment = match molecule.cell {
        Some(cell) => {
            let v = |k: usize| {
                let c = cell.cell.col[k] * BOHR_TO_ANGSTROM;
                format!("{:.8} {:.8} {:.8}", c.x, c.y, c.z)
            };
            let flag = |b: bool| if b { "T" } else { "F" };
            format!(
                "{comment} Lattice=\"{} {} {}\" pbc=\"{} {} {}\"",
                v(0),
                v(1),
                v(2),
                flag(cell.periodic[0]),
                flag(cell.periodic[1]),
                flag(cell.periodic[2])
            )
        }
        None => comment.to_string(),
    };
    let mut s = format!("{}\n{}\n", molecule.atoms.len(), comment);
    for a in &molecule.atoms {
        let p = a.position * BOHR_TO_ANGSTROM;
        s.push_str(&format!(
            "{:<2} {:14.8} {:14.8} {:14.8}\n",
            z_to_symbol(a.z).unwrap_or("?"),
            p.x,
            p.y,
            p.z
        ));
    }
    s
}

fn next(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    if *i >= args.len() {
        eprintln!("{flag} needs an argument");
        exit(1);
    }
    args[*i].clone()
}

fn parse_next<T: std::str::FromStr>(args: &[String], i: &mut usize, flag: &str) -> T {
    let s = next(args, i, flag);
    s.parse::<T>().unwrap_or_else(|_| {
        eprintln!("invalid value for {flag}: {s}");
        exit(1);
    })
}

fn usage() {
    eprintln!(
        "am1_rs_cli - AM1/RM1 semiempirical calculations\n\
         \n\
         USAGE:\n  \
         am1_rs_cli <mode> <file.xyz> [options]\n\
         \n\
         MODES:\n  \
         energy      single point: heat of formation, charges, dipole, HOMO/LUMO\n  \
         gradient    energy + forces (Hartree/Bohr), and stress under a cell\n  \
         optimize    L-BFGS geometry optimization (periodic when a cell is given)\n  \
         frequencies harmonic vibrational frequencies (cm^-1); phonons at q=0 under a cell\n  \
         phonons     phonon frequencies from a supercell (needs a cell)\n  \
         charges     AM1-BCC partial charges for AMBER (--mulliken for raw AM1)\n  \
         orbitals    orbital energies, occupations and optionally MO coefficients\n  \
         ir          infrared spectrum: atomic polar tensor and km/mol intensities\n  \
         molden      wavefunction in Molden format (stdout, or --molden-output)\n\
         \n\
         OPTIONS:\n  \
         --method M            NDDO parameterization: am1|rm1 (default am1)\n  \
         --charge Q            total molecular charge (default 0)\n  \
         --multiplicity M      spin multiplicity 2S+1 (default 1; M>1 requires UHF)\n  \
         --reference REF       SCF reference: auto|rhf|uhf (default auto)\n  \
         --rhf | --uhf         shortcuts for --reference rhf / uhf (force restricted/unrestricted)\n  \
         --field FX FY FZ      uniform electric field, atomic units (Hartree per e*Bohr)\n  \
         --opt-output FILE     write optimized geometry (extended XYZ, with the cell)\n  \
         --mol2-output FILE    write AM1-BCC charges as a mol2 file\n  \
         --molden-output FILE  write the Molden wavefunction to a file instead of stdout\n  \
         --molden-basis B      Molden basis section: gto|sto (default gto)\n  \
         --molden-primitives N Gaussians per shell for gto (default 6, max 10)\n  \
         --molden-orthogonal   write raw NDDO coefficients instead of S^-1/2 C\n  \
         --orbital-coefficients  orbitals mode: also print the MO coefficient matrix\n  \
         --mulliken            charges mode: use raw AM1 Mulliken charges\n\
         \n\
         PERIODIC OPTIONS (a cell also comes from a Lattice=\"...\" XYZ comment line):\n  \
         --cell ...            1, 3, 6 or 9 numbers: a | a b c | a b c alpha beta gamma |\n                        \
         ax ay az bx by bz cx cy cz  (Angstrom, degrees)\n  \
         --pbc AXES            periodic axes: x, y, z, xy, xyz, none (combinations allowed)\n  \
         --pbc-x --pbc-y --pbc-z   make one axis periodic; they accumulate\n  \
         --no-pbc              drop the cell and run the contents of it as a molecule\n  \
         --kpts NX NY NZ       Monkhorst-Pack k-mesh for the SCF (default 1 1 1)\n  \
         --smearing EV         Fermi-Dirac electronic temperature (default 0)\n  \
         --max-scf N           SCF iteration limit\n  \
         --supercell NX NY NZ  phonons mode: supercell for the force constants (default 2 2 2)\n  \
         --qpoints QX QY QZ .. phonons mode: explicit q points, fractional, three numbers each\n  \
         --qpath QX QY QZ ..   phonons mode: corners of a straight-line q path, fractional\n  \
         --qpath-points N      points per segment of --qpath (default 20)\n  \
         --relax-cell          optimize mode: relax the lattice against the stress too\n  \
         --pressure P          target pressure with --relax-cell, Hartree/Bohr^d (default 0)\n\
         DIVIDE-AND-CONQUER (energy, gradient and optimize modes):\n  \
         --dc                  linear-scaling divide-and-conquer SCF instead of the full one\n  \
         --dc-core N           target atoms per core region (default 12; implies --dc)\n  \
         --dc-buffer R         buffer radius in Angstrom (default 5.82; implies --dc)\n"
    );
}
