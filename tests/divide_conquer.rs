// SPDX-License-Identifier: GPL-3.0-or-later

//! Divide-and-conquer: the identities it rests on, and what actually became linear.
//!
//! The tests are ordered from "this is an identity that must hold exactly" to "this is a
//! measured approximation", because an energy agreeing with a full SCF is weak evidence on its
//! own — errors cancel — while the partition sum rule either holds to machine precision or the
//! method is not the method.

use am1_rs::divide_conquer::{
    build_subsystems, divide_conquer_gradient, partition_atoms, partition_weight_sum,
    run_divide_conquer, DcOptions,
};
use am1_rs::fermi::Filling;
use am1_rs::{
    closed_form_gradient, run_am1, Am1Options, Am1Parameters, Atom, Molecule, ScfReference, Vec3,
};

const ANG: f64 = 1.0 / 0.529167;

/// A chain of water molecules along x: a gapped, genuinely extended system with real
/// intermolecular interactions, so the buffer has something to do.
fn water_chain(n: usize) -> Molecule {
    let mut atoms = Vec::with_capacity(3 * n);
    for i in 0..n {
        // Alternating tilt so the chain is not a symmetry-degenerate line.
        let base = Vec3::new(i as f64 * 2.9, if i % 2 == 0 { 0.0 } else { 0.35 }, 0.0) * ANG;
        atoms.push(Atom {
            z: 8,
            position: base,
        });
        atoms.push(Atom {
            z: 1,
            position: base + Vec3::new(0.9584, 0.0, 0.0) * ANG,
        });
        atoms.push(Atom {
            z: 1,
            position: base + Vec3::new(-0.2400, 0.9279, 0.0) * ANG,
        });
    }
    Molecule::new(atoms)
}

/// An alkane chain: covalently bonded end to end, so the density matrix genuinely has to be
/// truncated across chemical bonds rather than across a van der Waals gap.
fn alkane(n_carbon: usize) -> Molecule {
    let mut atoms = Vec::new();
    let cc = 1.526 * ANG;
    let ch = 1.09 * ANG;
    for i in 0..n_carbon {
        // A planar zig-zag backbone.
        let x = i as f64 * cc * 0.8165;
        let y = if i % 2 == 0 { 0.0 } else { cc * 0.5774 };
        let c = Vec3::new(x, y, 0.0);
        atoms.push(Atom { z: 6, position: c });
        let s = if i % 2 == 0 { 1.0 } else { -1.0 };
        atoms.push(Atom {
            z: 1,
            position: c + Vec3::new(0.0, s * ch * 0.5, ch * 0.8),
        });
        atoms.push(Atom {
            z: 1,
            position: c + Vec3::new(0.0, s * ch * 0.5, -ch * 0.8),
        });
    }
    // Cap the ends so there are no dangling valences.
    let first = atoms[0].position;
    let last = atoms[3 * (n_carbon - 1)].position;
    atoms.push(Atom {
        z: 1,
        position: first + Vec3::new(-ch, 0.0, 0.0),
    });
    atoms.push(Atom {
        z: 1,
        position: last + Vec3::new(ch, 0.0, 0.0),
    });
    Molecule::new(atoms)
}

fn dc_options(buffer: f64) -> DcOptions {
    DcOptions {
        core_size: 6,
        buffer_radius: buffer,
        filling: Filling::Fermi { kt: 0.05 },
        max_scf: 400,
        e_tol: 1.0e-9,
        p_tol: 1.0e-8,
        mixing: 0.4,
        ..DcOptions::default()
    }
}

// ------------------------------------------------------------------------------ the sum rule
#[test]
fn the_partition_weights_sum_to_exactly_one() {
    // The identity the whole method rests on: every density-matrix block that is kept must
    // receive total weight 1 across all subsystems, and every block that is dropped must
    // receive exactly 0. Checked directly rather than inferred from the energy, because an
    // energy can come out right through cancelling errors and this cannot.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(9);
    let basis = am1_rs::basis::Basis::build(&molecule, &params).unwrap();

    for buffer in [6.0_f64, 9.0, 14.0, 40.0] {
        let cores = partition_atoms(&molecule, 6);
        let subs = build_subsystems(&molecule, &basis, &cores, buffer);
        let total = partition_weight_sum(&molecule, &subs);

        let mut worst_kept = 0.0_f64;
        let mut worst_dropped = 0.0_f64;
        let mut kept = 0;
        for a in 0..molecule.atoms.len() {
            for b in 0..molecule.atoms.len() {
                let d = (molecule.atoms[a].position - molecule.atoms[b].position).norm();
                if d <= buffer {
                    worst_kept = worst_kept.max((total[(a, b)] - 1.0).abs());
                    kept += 1;
                } else {
                    worst_dropped = worst_dropped.max(total[(a, b)].abs());
                }
            }
        }
        eprintln!(
            "    buffer {buffer:5.1} Bohr: {} subsystems, {kept} kept blocks, \
             max |Σw − 1| = {worst_kept:.3e}, max |Σw| outside = {worst_dropped:.3e}",
            subs.len()
        );
        assert!(
            worst_kept < 1.0e-12,
            "kept blocks must receive weight exactly 1, worst deviation {worst_kept:.3e}"
        );
        assert!(
            worst_dropped == 0.0,
            "dropped blocks must receive weight exactly 0, got {worst_dropped:.3e}"
        );
    }
}

#[test]
fn the_partition_covers_every_atom_exactly_once() {
    let molecule = alkane(11);
    let n = molecule.atoms.len();
    for core_size in [1usize, 3, 7, 16] {
        let cores = partition_atoms(&molecule, core_size);
        let mut seen = vec![0usize; n];
        for core in &cores {
            assert!(core.len() <= core_size);
            for &a in core {
                seen[a] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "core_size {core_size}: atoms covered {seen:?}"
        );
    }
}

// ----------------------------------------------------------------- convergence to the full SCF
#[test]
fn a_buffer_covering_the_system_reproduces_the_full_scf() {
    // The limiting case, and the sharpest correctness test there is: when the buffer swallows
    // the whole molecule every subsystem is the whole molecule, the Yang weights still sum to
    // one, and the assembled density must be the full SCF density to roundoff. Any error in
    // the projection, the common Fermi level or the assembly shows up here as a real
    // disagreement rather than a small one.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(4);
    let options = Am1Options {
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        ..Am1Options::default()
    };

    let full = run_am1(&molecule, &params, &options).unwrap();
    // kt -> 0 as well: aufbau is what the full SCF does, so a finite smearing would show up as
    // a genuine (and correct) difference and mask the thing being tested.
    let dc = run_divide_conquer(
        &molecule,
        &params,
        &options,
        &DcOptions {
            core_size: 3,
            buffer_radius: 500.0,
            filling: Filling::Aufbau,
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            max_scf: 400,
            ..DcOptions::default()
        },
    )
    .unwrap();

    assert!(dc.converged, "the divide-and-conquer SCF did not converge");
    let de = (dc.total_ev - full.total_ev).abs();
    let dq = dc
        .charges
        .iter()
        .zip(&full.charges)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "    full SCF {:.10} eV, DC {:.10} eV, Δ = {de:.3e} eV ({:.3e} eV/atom); \
         max Δq = {dq:.3e} e",
        full.total_ev,
        dc.total_ev,
        de / molecule.atoms.len() as f64
    );
    assert!(
        de < 1.0e-7,
        "全系バッファで full SCF と一致しない: {de:.3e} eV"
    );
    assert!(dq < 1.0e-7, "charges disagree by {dq:.3e} e");
}

#[test]
fn the_error_falls_monotonically_as_the_buffer_grows() {
    // The approximation has to be controlled: a larger buffer must mean a smaller error, all
    // the way down. A method that merely happened to be accurate at one buffer radius, or that
    // wandered, would not be usable.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(6);
    let options = Am1Options {
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        ..Am1Options::default()
    };
    let full = run_am1(&molecule, &params, &options).unwrap();

    eprintln!("    buffer(Bohr)   E_DC (eV)          ΔE (eV)      ΔE/atom (eV)   subsystems");
    let mut previous = f64::INFINITY;
    for buffer in [6.0_f64, 9.0, 12.0, 16.0, 22.0, 30.0] {
        let dc = run_divide_conquer(&molecule, &params, &options, &dc_options(buffer)).unwrap();
        assert!(dc.converged, "buffer {buffer}: DC did not converge");
        let err = (dc.total_ev - full.total_ev).abs();
        eprintln!(
            "    {buffer:9.1}   {:15.9}   {err:10.3e}   {:10.3e}   {}",
            dc.total_ev,
            err / molecule.atoms.len() as f64,
            dc.subsystems
        );
        assert!(
            err <= previous * 1.5 + 1.0e-9,
            "the error grew when the buffer grew: {err:.3e} after {previous:.3e}"
        );
        previous = err;
    }
    assert!(
        previous < 1.0e-4,
        "at a 30 Bohr buffer the error should be tiny, got {previous:.3e} eV"
    );
}

// ------------------------------------------------------------------------------- open shell
#[test]
fn forced_unrestricted_reproduces_the_restricted_answer_on_a_closed_shell() {
    // UHF parity with RHF, the standard sanity check, run through divide-and-conquer. If the
    // spin bookkeeping were wrong -- a capacity of 2 where 1 belongs, one Fermi level where two
    // are needed, or the Coulomb term seeing one channel instead of the total -- this is where
    // it shows, and it shows as a large error rather than a subtle one.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(4);
    let dc = dc_options(14.0);

    let restricted = run_divide_conquer(
        &molecule,
        &params,
        &Am1Options {
            reference: ScfReference::Restricted,
            ..Am1Options::default()
        },
        &dc,
    )
    .unwrap();
    let unrestricted = run_divide_conquer(
        &molecule,
        &params,
        &Am1Options {
            reference: ScfReference::Unrestricted,
            ..Am1Options::default()
        },
        &dc,
    )
    .unwrap();

    assert!(restricted.converged && unrestricted.converged);
    assert!(unrestricted.unrestricted && !restricted.unrestricted);
    let de = (unrestricted.total_ev - restricted.total_ev).abs();
    eprintln!(
        "    DC restricted {:.9} eV, DC forced-unrestricted {:.9} eV, Δ = {de:.3e} eV",
        restricted.total_ev, unrestricted.total_ev
    );
    assert!(de < 1.0e-6, "forced UHF disagrees with RHF by {de:.3e} eV");

    // With no symmetry breaking the spin density must vanish.
    let spin = unrestricted.spin_density.as_ref().unwrap();
    let worst = spin.as_slice().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("    largest |P_α − P_β| element = {worst:.3e}");
    assert!(
        worst < 1.0e-5,
        "a closed shell acquired a spin density of {worst:.3e}"
    );
}

#[test]
fn an_open_shell_matches_the_full_unrestricted_scf() {
    // The real open-shell case: a doublet cation, where the two spin channels hold different
    // electron counts and therefore need their own chemical potentials.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(3);
    let options = Am1Options {
        charge: 1.0,
        multiplicity: 2,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        ..Am1Options::default()
    };

    let full = run_am1(&molecule, &params, &options).unwrap();
    let dc = run_divide_conquer(
        &molecule,
        &params,
        &options,
        &DcOptions {
            core_size: 3,
            buffer_radius: 500.0,
            filling: Filling::Aufbau,
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            max_scf: 600,
            ..DcOptions::default()
        },
    )
    .unwrap();

    assert!(dc.converged && dc.unrestricted);
    assert_eq!(
        dc.fermi_energies_ev.len(),
        2,
        "UHF needs one Fermi level per spin channel"
    );
    let de = (dc.total_ev - full.total_ev).abs();
    eprintln!(
        "    open-shell cation: full UHF {:.9} eV, DC {:.9} eV, Δ = {de:.3e} eV; \
         μ_α = {:.4} eV, μ_β = {:.4} eV",
        full.total_ev, dc.total_ev, dc.fermi_energies_ev[0], dc.fermi_energies_ev[1]
    );
    assert!(
        de < 1.0e-6,
        "open-shell DC disagrees with full UHF by {de:.3e} eV"
    );
}

// ------------------------------------------------------------------------------ charged cells
#[test]
fn a_net_charge_is_conserved_and_shared_between_subsystems() {
    // Non-neutral systems are the case the common chemical potential exists for: the charge is
    // not assigned to any subsystem, it is distributed by the filling. Two things must hold —
    // the Mulliken charges sum to the formal charge, and the extra charge is not parked
    // entirely on whichever subsystem happened to sort first.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(6);

    for (charge, multiplicity) in [(-1.0, 2), (0.0, 1), (1.0, 2)] {
        let options = Am1Options {
            charge,
            multiplicity,
            ..Am1Options::default()
        };
        let dc = run_divide_conquer(&molecule, &params, &options, &dc_options(14.0)).unwrap();
        assert!(dc.converged, "charge {charge}: DC did not converge");
        let total: f64 = dc.charges.iter().sum();
        eprintln!(
            "    formal charge {charge:+.0}: Σq = {total:+.9} e, E = {:.6} eV, μ = {:.4} eV",
            dc.total_ev, dc.fermi_energy_ev
        );
        assert!(
            (total - charge).abs() < 1.0e-8,
            "Mulliken charges sum to {total}, not the formal charge {charge}"
        );
    }
}

// --------------------------------------------------------------------------- scaling counters
#[test]
fn the_diagonalization_work_is_linear_while_the_coulomb_work_is_quadratic() {
    // The claim this project is willing to make, and the one it is not, asserted separately.
    //
    // Counters rather than a stopwatch: wall-clock in a test is a flaky test, and the point
    // here is a statement about the algorithm, not about this machine. The quadratic assertion
    // is as important as the linear one -- it pins down the part that did *not* become linear,
    // so a later change cannot quietly leave the impression that everything did.
    let params = Am1Parameters::standard().unwrap();
    let options = Am1Options::default();
    let dc = dc_options(12.0);

    let mut rows = Vec::new();
    eprintln!(
        "    n_water  atoms   AOs   subs   Σn³/atom     coulomb/atom   exchange/atom  kept/atom  \
         DIIS elems   dense elems"
    );
    for n in [4usize, 8, 16, 32] {
        let molecule = water_chain(n);
        let nat = molecule.atoms.len() as f64;
        let r = run_divide_conquer(&molecule, &params, &options, &dc).unwrap();
        assert!(r.converged, "n = {n} did not converge");
        eprintln!(
            "    {n:7}  {:5}  {:4}   {:4}   {:10.1}   {:12.1}   {:12.2}   {:8.2}  {:10}  {:12}",
            molecule.atoms.len(),
            r.largest_subsystem_aos,
            r.subsystems,
            r.diagonalization_work / nat,
            r.coulomb_work as f64 / nat,
            r.exchange_work as f64 / nat,
            r.retained_density_blocks as f64 / nat,
            r.diis_pattern_elements,
            r.dense_triangle_elements,
        );
        assert!(
            r.diis_pattern_elements <= r.dense_triangle_elements,
            "the sparse pattern cannot be larger than the dense triangle it replaces"
        );
        rows.push((
            nat,
            r.diagonalization_work,
            r.coulomb_work as f64,
            r.exchange_work as f64,
            r.retained_density_blocks as f64,
            r.diis_pattern_elements as f64,
            r.dense_triangle_elements as f64,
        ));
    }

    /// One system size's counters: `(atoms, diagonalization, coulomb, exchange, retained blocks,
    /// DIIS pattern, dense triangle)`. Named so the closure below has something to be generic
    /// over without spelling a seven-tuple out twice.
    type Row = (f64, f64, f64, f64, f64, f64, f64);
    let exponent = |pick: fn(&Row) -> f64| -> f64 {
        // Slope of log(work) against log(atoms) over the top two sizes, which is where the
        // asymptotic behaviour has actually set in.
        let a = &rows[rows.len() - 2];
        let b = &rows[rows.len() - 1];
        (pick(b).ln() - pick(a).ln()) / (b.0.ln() - a.0.ln())
    };

    let diag = exponent(|r| r.1);
    let coulomb = exponent(|r| r.2);
    let exchange = exponent(|r| r.3);
    let kept = exponent(|r| r.4);
    let diis_memory = exponent(|r| r.5);
    let dense_memory = exponent(|r| r.6);
    eprintln!(
        "\n    scaling exponents: diagonalization {diag:.3}, exchange {exchange:.3}, \
         retained blocks {kept:.3}, Coulomb {coulomb:.3}"
    );
    eprintln!(
        "    DIIS history per entry: {diis_memory:.3} (dense triangle would be {dense_memory:.3})"
    );

    // The dominant memory term of a large run. The history used to hold a dense triangle, which is
    // quadratic; the divide-and-conquer density is identically zero outside the buffer radius, so
    // all of that beyond the pattern was zeros. Both exponents are asserted, because the point is
    // the *difference* between them — a linear number on its own could be an accident of size.
    assert!(
        diis_memory < 1.15,
        "the DIIS history should now be linear in the atom count, exponent {diis_memory:.3}"
    );
    assert!(
        dense_memory > 1.7,
        "the dense triangle it replaces should still be measured as quadratic, exponent \
         {dense_memory:.3}; if it is not, this comparison is not showing what it claims"
    );

    assert!(
        diag < 1.15,
        "the diagonalization work should be linear, exponent {diag:.3}"
    );
    assert!(
        exchange < 1.15,
        "the exchange work should be linear once the density is truncated, exponent {exchange:.3}"
    );
    assert!(
        kept < 1.15,
        "the retained density blocks should be linear, exponent {kept:.3}"
    );
    // Not a wish: a statement of what has *not* been achieved, so that it cannot silently be
    // claimed later. The NDDO Coulomb sum is over all pairs because the two-centre integrals
    // decay as 1/R; making it linear needs the multipole/Ewald split, which is not implemented.
    assert!(
        coulomb > 1.7,
        "the Coulomb work is still quadratic by construction; exponent {coulomb:.3} suggests \
         something changed and the documentation needs revisiting"
    );
}

#[test]
fn the_subsystem_size_stops_growing_with_the_molecule() {
    // The mechanism behind the linear diagonalization: with a fixed buffer radius the largest
    // subsystem stops growing once the molecule is longer than the buffer is wide. If this
    // failed, `Σ n³` could still look linear over a short range while being cubic underneath.
    let params = Am1Parameters::standard().unwrap();
    let basis_sizes: Vec<usize> = [8usize, 16, 32, 64]
        .iter()
        .map(|&n| {
            let molecule = water_chain(n);
            let basis = am1_rs::basis::Basis::build(&molecule, &params).unwrap();
            let cores = partition_atoms(&molecule, 6);
            let subs = build_subsystems(&molecule, &basis, &cores, 12.0);
            subs.iter().map(|s| s.nao()).max().unwrap()
        })
        .collect();
    eprintln!("    largest subsystem (AOs) for 8/16/32/64 waters: {basis_sizes:?}");
    assert_eq!(
        basis_sizes[2], basis_sizes[3],
        "the largest subsystem must saturate; got {basis_sizes:?}"
    );
}

// -------------------------------------------------------------------------------- gradient
#[test]
fn the_gradient_approaches_the_full_scf_gradient_as_the_buffer_grows() {
    // The divide-and-conquer density is not variational, so the fixed-density (Hellmann-Feynman)
    // gradient is not the exact derivative of the divide-and-conquer energy: the term that
    // vanishes for a stationary density does not vanish here. Rather than assert a tolerance
    // that would only describe this one molecule, measure the residual and check that it is
    // controlled by the same parameter that controls the energy.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(4);
    let options = Am1Options {
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        ..Am1Options::default()
    };
    let reference = closed_form_gradient(&molecule, &params, &options).unwrap();

    eprintln!("    buffer(Bohr)   max |g_DC − g_full| (eV/Bohr)");
    let mut previous = f64::INFINITY;
    for buffer in [6.0_f64, 9.0, 12.0, 18.0, 30.0, 500.0] {
        let dc = run_divide_conquer(
            &molecule,
            &params,
            &options,
            &DcOptions {
                filling: Filling::Aufbau,
                e_tol: 1.0e-11,
                p_tol: 1.0e-10,
                max_scf: 600,
                ..dc_options(buffer)
            },
        )
        .unwrap();
        assert!(dc.converged, "buffer {buffer}: DC did not converge");
        let g = divide_conquer_gradient(&molecule, &params, &options, &dc).unwrap();
        let worst = g
            .iter()
            .zip(&reference.gradient)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        eprintln!("    {buffer:9.1}      {worst:.4e}");
        assert!(
            worst <= previous * 2.0 + 1.0e-10,
            "the gradient error grew when the buffer grew: {worst:.3e} after {previous:.3e}"
        );
        previous = worst;
    }
    assert!(
        previous < 1.0e-8,
        "with the buffer covering the molecule the DC gradient must equal the full one, \
         got {previous:.3e} eV/Bohr"
    );
}

#[test]
fn the_forces_sum_to_zero() {
    // Translational invariance: independent of any comparison, and it catches an image or
    // aggregation error in the assembled density that a magnitude comparison could miss.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(5);
    let dc = run_divide_conquer(
        &molecule,
        &params,
        &Am1Options::default(),
        &dc_options(12.0),
    )
    .unwrap();
    let g = divide_conquer_gradient(&molecule, &params, &Am1Options::default(), &dc).unwrap();
    let mut sum = Vec3::zero();
    for v in &g {
        sum += *v;
    }
    let worst = sum.x.abs().max(sum.y.abs()).max(sum.z.abs());
    eprintln!("    |Σ g| = {worst:.3e} eV/Bohr");
    assert!(worst < 1.0e-9, "the gradient sums to {worst:.3e}");
}

// ----------------------------------------------------------------------------- honest warnings
#[test]
fn a_small_gap_is_reported_rather_than_hidden() {
    // Divide-and-conquer assumes the density matrix decays with distance, which is a property of
    // gapped systems. Where it does not hold, the result has to say so rather than return a
    // plausible number in silence.
    //
    // The warning is tested by moving the *threshold* across a measured gap, in both directions,
    // rather than by trying to build a metal. That is deliberate: AM1 has no parameterization in
    // which a small molecule is reliably metallic — an evenly spaced hydrogen chain, the usual
    // textbook example, Peierls-distorts into H₂ units and opens a 3.8 eV gap — so a test that
    // claimed to produce one would be asserting something false about the physics in order to
    // reach the branch. What is actually being checked here is that the gap is measured and that
    // the threshold decides, which is the whole of the mechanism.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(4);

    let quiet = run_divide_conquer(
        &molecule,
        &params,
        &Am1Options::default(),
        &DcOptions {
            gap_warn_ev: 0.5,
            ..dc_options(12.0)
        },
    )
    .unwrap();
    let gap = quiet.homo_lumo_gap_ev;
    eprintln!("    measured smallest subsystem gap: {gap:.4} eV");
    assert!(
        gap > 1.0,
        "a water chain should be comfortably gapped, got {gap:.3} eV"
    );
    assert!(
        quiet.small_gap_warning.is_none(),
        "a {gap:.3} eV gap must not trip a 0.5 eV threshold"
    );

    let loud = run_divide_conquer(
        &molecule,
        &params,
        &Am1Options::default(),
        &DcOptions {
            gap_warn_ev: gap + 1.0,
            ..dc_options(12.0)
        },
    )
    .unwrap();
    assert!(
        loud.small_gap_warning.is_some(),
        "a {gap:.3} eV gap must trip a {:.3} eV threshold",
        gap + 1.0
    );
    let message = loud.small_gap_warning.unwrap();
    eprintln!("    warning text: {message}");
    assert!(
        message.contains("buffer_radius") && message.contains("full SCF"),
        "the warning should say what to do about it, got: {message}"
    );
}
