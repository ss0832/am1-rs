// SPDX-License-Identifier: GPL-3.0-or-later
//! `am1_rs_cli` — command-line front end for am1-rs.
//!
//! Modes: `energy`, `gradient`, `optimize`, `frequencies`, `charges`, `orbitals`, `ir`,
//! `molden`. Native output is in atomic units (Hartree, Bohr) throughout, including forces;
//! heats of formation are additionally reported in kcal/mol. eV/Å is reserved for the ASE
//! boundary.
//!
//! **Every printed string is ASCII**, and must stay that way. This front end writes UTF-8 bytes
//! whatever the locale, so it cannot fail on a non-ASCII character — but `python/am1_rs/__main__.py`
//! mirrors it and encodes with the *locale's* codec, where a `cp932` or `C` locale raises
//! `UnicodeEncodeError` part-way through the output. Units are spelled `e*a0`, `cm^-1`, `eV/A` so
//! that both front ends produce the same bytes on every platform. Comments are exempt; they are
//! not printed.

use am1_rs::bcc::{am1_bcc_charges, write_mol2};
use am1_rs::constants::{
    AU_DIPOLE_TO_DEBYE, BOHR_TO_ANGSTROM, EV_TO_HARTREE, HARTREE_TO_EV, KCAL_TO_EV,
};
use am1_rs::gradient::closed_form_gradient;
use am1_rs::math::Vec3;
use am1_rs::method::NddoMethod;
use am1_rs::optimizer::{optimize, OptOptions};
use am1_rs::params::Am1Parameters;
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

    let mut charge = 0.0_f64;
    let mut multiplicity = 1usize;
    let mut reference = ScfReference::Auto;
    let mut opt_output: Option<String> = None;
    let mut mol2_output: Option<String> = None;
    let mut use_bcc = true;
    let mut method = NddoMethod::default();
    let mut molden_output: Option<String> = None;
    let mut field: Option<Vec3> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--field" => {
                // Three values, in atomic units (Hartree per e·Bohr), matching the native
                // Python surface rather than ASE's V/Å.
                let x: f64 = parse_next(&args, &mut i, "--field");
                let y: f64 = parse_next(&args, &mut i, "--field");
                let z: f64 = parse_next(&args, &mut i, "--field");
                field = Some(Vec3::new(x, y, z) * HARTREE_TO_EV);
            }
            "--molden-output" => {
                molden_output = Some(next(&args, &mut i, "--molden-output"));
            }
            "--method" => {
                let s = next(&args, &mut i, "--method");
                method = match NddoMethod::parse(&s) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("{e}");
                        exit(1);
                    }
                };
            }
            "--charge" => {
                charge = parse_next(&args, &mut i, "--charge");
            }
            "--multiplicity" | "--spin-multiplicity" => {
                multiplicity = parse_next::<f64>(&args, &mut i, "--multiplicity") as usize;
            }
            "--reference" | "--ref" => {
                let s = next(&args, &mut i, "--reference");
                reference = match s.trim().to_ascii_lowercase().as_str() {
                    "auto" => ScfReference::Auto,
                    "rhf" | "r" | "restricted" => ScfReference::Restricted,
                    "uhf" | "u" | "unrestricted" => ScfReference::Unrestricted,
                    other => {
                        eprintln!("invalid --reference value: {other} (expected auto|rhf|uhf)");
                        exit(1);
                    }
                };
            }
            "--rhf" => {
                reference = ScfReference::Restricted;
            }
            "--uhf" => {
                reference = ScfReference::Unrestricted;
            }
            "--opt-output" => {
                opt_output = Some(next(&args, &mut i, "--opt-output"));
            }
            "--mol2-output" => {
                mol2_output = Some(next(&args, &mut i, "--mol2-output"));
            }
            "--mulliken" => {
                use_bcc = false;
            }
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

    if let Err(e) = run(Cli {
        mode: &mode,
        path: &path,
        method,
        charge,
        multiplicity,
        reference,
        opt_output,
        mol2_output,
        molden_output,
        field,
        use_bcc,
    }) {
        eprintln!("error: {e}");
        exit(1);
    }
}

/// Everything the command line resolved to, in one place.
///
/// A struct rather than eleven positional arguments: the list had already reached the point where
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
    field: Option<Vec3>,
    use_bcc: bool,
}

/// A value that rounds to zero at `decimals` places, printed without a sign.
///
/// `-0.0` and `0.0` are the same number at any printed precision, but they are different *text*,
/// and the CLI's text is compared against the Python front end's in `tests/test_cli.py`. The two
/// take different routes to the same eigenvalue, so a quantity that is zero — a rigid-body
/// frequency, an atomic polar tensor element forbidden by symmetry — can land on either side of it
/// and make the two disagree about a number they both agree is zero.
fn unsigned_zero(v: f64, decimals: i32) -> f64 {
    if v.abs() < 0.5 * 10f64.powi(-decimals) {
        0.0
    } else {
        v
    }
}
fn run(cli: Cli<'_>) -> am1_rs::Result<()> {
    let Cli {
        mode,
        path,
        method,
        charge,
        multiplicity,
        reference,
        opt_output,
        mol2_output,
        molden_output,
        field,
        use_bcc,
    } = cli;
    let mut molecule = Molecule::from_xyz_file(path, charge)?;
    molecule.multiplicity = multiplicity;
    let params = Am1Parameters::for_method(method)?;
    let scf_opts = Am1Options {
        charge,
        multiplicity,
        reference,
        electric_field: field,
        ..Am1Options::default()
    };

    match mode {
        "energy" => {
            let r = run_am1(&molecule, &params, &scf_opts)?;
            print_energy(&molecule, &r);
        }
        "gradient" => {
            let g = closed_form_gradient(&molecule, &params, &scf_opts)?;
            print_energy(&molecule, &g.scf);
            // Forces in atomic units (Hartree/Bohr); g.forces are eV/Bohr.
            println!("\nforces (Hartree/Bohr):");
            for (a, f) in molecule.atoms.iter().zip(&g.forces) {
                let s = z_to_symbol(a.z).unwrap_or("?");
                println!(
                    "  {:<2}  {:14.8} {:14.8} {:14.8}",
                    s,
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
        "optimize" => {
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
            if let Some(out) = opt_output {
                std::fs::write(&out, xyz)?;
                println!("\noptimized geometry written to {out}");
            } else {
                println!("\noptimized geometry (Angstrom):\n{xyz}");
            }
        }
        "frequencies" => {
            let vib = am1_rs::hessian::vibrational_analysis(&molecule, &params, &scf_opts, 1.0e-3)?;
            println!("harmonic vibrational frequencies (cm^-1):");
            for (i, f) in vib.frequencies_cm.iter().enumerate() {
                let tag = if f.abs() < 50.0 {
                    "  (translation/rotation)"
                } else {
                    ""
                };
                // Anything that rounds to zero at this precision prints as `0.0`, never `-0.0`.
                // A rigid-body mode's frequency is zero plus arithmetic noise, and its sign is
                // not information — but it *is* printed, so it made `tests/test_cli.py` compare
                // the two front ends' last bits. They took different routes to the same
                // eigenvalue, landed on opposite sides of zero, and the CLI text disagreed on a
                // number both agreed was zero.
                let shown = unsigned_zero(*f, 1);
                println!("  {:>3}  {:>10.1}{}", i + 1, shown, tag);
            }
            println!("\n(compute at an optimized geometry for meaningful frequencies)");
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
        }
        "ir" => {
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
            println!("\n(rigid-body near 1 is a translation or rotation, not a vibration)");
        }
        "molden" => {
            let r = run_am1(&molecule, &params, &scf_opts)?;
            let text = am1_rs::molden::to_molden(&molecule, &params, &r)?;
            match molden_output {
                Some(out) => {
                    std::fs::write(&out, text)?;
                    println!("molden wavefunction written to {out}");
                }
                None => print!("{text}"),
            }
        }
        "charges" => {
            if use_bcc {
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
                        if let Some(out) = mol2_output {
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
        "{mode}, {} atoms, {}",
        molecule.atoms.len(),
        params.method.display_name()
    ));
    Ok(())
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
        "am1_rs_cli - AM1 semiempirical calculations\n\
         \n\
         USAGE:\n  \
         am1_rs_cli <mode> <file.xyz> [options]\n\
         \n\
         MODES:\n  \
         energy      single point: heat of formation, charges, dipole, HOMO/LUMO\n  \
         gradient    energy + forces (Hartree/Bohr)\n  \
         optimize    L-BFGS geometry optimization\n  \
         frequencies harmonic vibrational frequencies (cm^-1)\n  \
         charges     AM1-BCC partial charges for AMBER (--mulliken for raw AM1)\n  \
         orbitals    orbital energies and occupations (both spins for UHF)\n  \
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
         --opt-output FILE     write optimized geometry (XYZ)\n  \
         --mol2-output FILE    write AM1-BCC charges as a mol2 file\n  \
         --molden-output FILE  write the Molden wavefunction to a file instead of stdout\n  \
         --mulliken            charges mode: use raw AM1 Mulliken charges\n"
    );
}
