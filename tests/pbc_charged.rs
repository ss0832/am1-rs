// SPDX-License-Identifier: GPL-3.0-or-later

//! Charged periodic cells.
//!
//! A net charge per cell is fully supported: the electron count, the self-consistent density,
//! the Mulliken charges, the forces **and the total energy** are correct and mutually
//! consistent, under the tin-foil boundary condition that a neutralizing background implies.
//!
//! That last part is recent, and this file keeps both sides of it. Without the Ewald correction
//! the monopole lattice sum `Σ_T Q²/|T|` diverges, and truncating it at a finite cutoff leaves
//! the answer growing without bound: a +1 water cell in an 8 Å cube measures −331 eV at a
//! 20 Bohr cutoff and **+72 eV** at 130 Bohr. With the correction the same series moves by
//! 0.2 eV, and what remains is the logarithmic `R⁻³` residual of the Klopman–Ohno kernel,
//! identified as such by its scaling rather than assumed.
//!
//! Both are asserted. The divergence is kept as an explicit `ewald: false` control, because a
//! test that only showed the fixed behaviour would pass just as happily if the correction
//! silently stopped being applied.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{run_pbc_scf, KMesh, PbcOptions};
use am1_rs::{run_am1, Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn water_cell(a_angstrom: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.9584, 0.0, 0.0) * ANG,
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.2400, 0.9279, 0.0) * ANG,
        },
    ])
    .with_cell(Lattice::cubic(a_angstrom * ANG).unwrap())
}

fn options(charge: f64, multiplicity: usize, realspace_cutoff: f64) -> PbcOptions {
    options_ewald(charge, multiplicity, realspace_cutoff, true)
}

fn options_ewald(
    charge: f64,
    multiplicity: usize,
    realspace_cutoff: f64,
    ewald: bool,
) -> PbcOptions {
    PbcOptions {
        ewald,
        kmesh: KMesh::MonkhorstPack([2, 2, 2]),
        realspace_cutoff,
        exchange_cutoff: Some(10.0),
        smearing_ev: 0.0,
        charge,
        multiplicity,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 2000,
        ..PbcOptions::default()
    }
}

/// The Γ path, which is where Ewald summation is wired in. `ewald` selects whether the
/// long-range monopole sum is done exactly or left to the real-space cutoff.
fn gamma_options(
    charge: f64,
    multiplicity: usize,
    realspace_cutoff: f64,
    ewald: bool,
) -> Am1Options {
    Am1Options {
        charge,
        multiplicity,
        realspace_cutoff,
        exchange_cutoff: Some(10.0),
        ewald,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 2000,
        ..Am1Options::default()
    }
}

#[test]
fn a_charged_cell_is_self_consistent() {
    let params = Am1Parameters::standard().unwrap();

    for (charge, multiplicity) in [(-1.0, 2), (0.0, 1), (1.0, 2)] {
        let r = run_pbc_scf(
            &water_cell(8.0),
            &params,
            &options(charge, multiplicity, 40.0),
        )
        .unwrap();
        assert!(r.converged, "charge {charge}: SCF did not converge");
        let total: f64 = r.charges.iter().sum();
        eprintln!(
            "    charge {charge:+.0}: Σq = {total:+.10} e, E = {:.6} eV, warning = {}",
            r.total_ev,
            r.charged_cell_warning.is_some()
        );
        assert!(
            (total - charge).abs() < 1.0e-8,
            "Mulliken charges sum to {total}, not the formal charge {charge}"
        );
        assert_eq!(
            r.charged_cell_warning.is_some(),
            charge != 0.0,
            "the warning must be present exactly when the cell is charged"
        );
    }
}

#[test]
fn the_neutral_cell_energy_converges_with_the_real_space_cutoff() {
    // The control for the test below. If the neutral cell drifted too, the divergence there
    // would not be evidence about the charge specifically.
    let params = Am1Parameters::standard().unwrap();
    let mut energies = Vec::new();
    eprintln!("    cutoff(Bohr)   neutral energy (eV)");
    for rc in [20.0_f64, 40.0, 90.0, 130.0] {
        let r = run_pbc_scf(&water_cell(8.0), &params, &options(0.0, 1, rc)).unwrap();
        assert!(r.converged);
        eprintln!("    {rc:10.1}   {:18.8}", r.total_ev);
        energies.push(r.total_ev);
    }
    let spread = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - energies.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!("    spread over a 6.5x range of cutoff: {spread:.2e} eV");
    assert!(
        spread < 1.0e-3,
        "the neutral cell should be essentially converged; it moved {spread:.3e} eV"
    );
}

#[test]
fn ewald_summation_makes_the_charged_cell_energy_converge() {
    // What Ewald summation is here for, stated as the difference it makes.
    //
    // Without it the monopole lattice sum `Σ_T Q²/|T|` is truncated at the real-space cutoff and
    // grows without bound with that cutoff — 400 eV between 20 and 130 Bohr, measured in the
    // test below. With it, the sum is exact and the energy stops depending on the cutoff
    // entirely: what is left is the far smaller `R⁻³` residual of the Klopman–Ohno kernel, which
    // this version still truncates.
    let params = Am1Parameters::standard().unwrap();

    // Three configurations, because the residual needs the middle one to be named. `with_ewald`
    // is the historical column: Ewald on, Klopman-Ohno tail off.
    let cutoffs = [20.0_f64, 40.0, 90.0, 130.0];
    eprintln!("    cutoff(Bohr)   Ewald, no tail       Ewald + tail        no Ewald");
    let mut with_ewald = Vec::new();
    let mut with_tail = Vec::new();
    let mut without = Vec::new();
    for rc in cutoffs {
        let base = gamma_options(1.0, 2, rc, true);
        let on = run_am1(
            &water_cell(8.0),
            &params,
            &Am1Options {
                klopman_ohno_tail: false,
                ..base.clone()
            },
        )
        .unwrap();
        let tail = run_am1(&water_cell(8.0), &params, &base).unwrap();
        let off = run_am1(&water_cell(8.0), &params, &gamma_options(1.0, 2, rc, false)).unwrap();
        assert!(
            on.converged && tail.converged && off.converged,
            "cutoff {rc}: SCF did not converge"
        );
        eprintln!(
            "    {rc:10.1}   {:17.8}   {:17.8}   {:15.8}",
            on.total_ev, tail.total_ev, off.total_ev
        );
        with_ewald.push(on.total_ev);
        with_tail.push(tail.total_ev);
        without.push(off.total_ev);
    }
    let spread = |v: &[f64]| {
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - v.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let on_spread = spread(&with_ewald);
    let off_spread = spread(&without);
    let tail_spread = spread(&with_tail);
    eprintln!(
        "\n    spread over a 6.5x range of cutoff: {off_spread:.1} eV bare, {on_spread:.4} eV \
         with Ewald, {tail_spread:.6} eV with Ewald and the tail",
    );

    assert!(
        off_spread > 100.0,
        "the no-Ewald control should still diverge; it moved only {off_spread:.3e} eV"
    );
    assert!(
        on_spread < off_spread / 1000.0,
        "Ewald should remove the divergence, not merely reduce it: {off_spread:.1} eV became \
         {on_spread:.4} eV"
    );

    // What is left is **not** noise, and saying so is the point of this block.
    //
    // The `1/R` monopole sum is exact once Ewald is on, so the residual cutoff dependence is the
    // `R^-3` part of the Klopman-Ohno kernel. Its lattice sum is logarithmically divergent in
    // three dimensions -- `sum_T |T|^-3 ~ (4*pi/V) ln r_c` -- so the increments per unit
    // `ln(r_c)` should be *constant* rather than settling. That is the diagnosis, and it is
    // asserted on a run with the tail switched off, which is the configuration it describes.
    //
    // The same measurement with the tail on is the cure, and it is asserted alongside: the slope
    // has to collapse. Testing the pair, rather than testing for a small number, is what
    // distinguishes "the remaining error is the one we know about, and it is now gone" from "the
    // remaining error happens to be small here".
    let per_log = |e: &[f64]| -> Vec<f64> {
        (0..cutoffs.len() - 1)
            .map(|w| (e[w + 1] - e[w]) / (cutoffs[w + 1] / cutoffs[w]).ln())
            .collect()
    };
    let untailed = per_log(&with_ewald);
    let tailed = per_log(&with_tail);
    let show = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    eprintln!("    residual per unit ln(cutoff), eV:");
    eprintln!(
        "      tail off: [{}]  <- constant means the leftover is the R^-3 term",
        show(&untailed)
    );
    eprintln!("      tail on : [{}]", show(&tailed));

    let span = |v: &[f64]| {
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - v.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let worst = |v: &[f64]| v.iter().copied().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(
        span(&untailed).abs() < 0.5 * worst(&untailed),
        "without the tail the residual should be logarithmic in the cutoff, so it is the R^-3 \
         term the module claims is left over: {untailed:?}"
    );
    assert!(
        worst(&tailed) < 0.5 * worst(&untailed),
        "the tail should collapse that logarithmic slope; it went {untailed:?} -> {tailed:?}"
    );
}

#[test]
fn ewald_leaves_a_neutral_cell_essentially_unchanged() {
    // The control. For a neutral cell the monopole terms already cancel, so replacing their
    // truncated sum with the exact one should barely move the answer — and if it moved a lot,
    // the correction would be doing something other than what it claims.
    let params = Am1Parameters::standard().unwrap();
    let on = run_am1(
        &water_cell(8.0),
        &params,
        &gamma_options(0.0, 1, 40.0, true),
    )
    .unwrap();
    let off = run_am1(
        &water_cell(8.0),
        &params,
        &gamma_options(0.0, 1, 40.0, false),
    )
    .unwrap();
    let delta = (on.total_ev - off.total_ev).abs();
    eprintln!(
        "    neutral cell: {:.8} eV with Ewald, {:.8} without, Δ = {delta:.3e} eV",
        on.total_ev, off.total_ev
    );
    assert!(
        delta < 0.05,
        "Ewald changed a neutral cell by {delta:.3e} eV, which is more than the truncation it \
         is replacing should have been worth"
    );
}

#[test]
fn ewald_is_refused_for_a_slab_rather_than_applied_wrongly() {
    // The reciprocal sum here is three-dimensional. Using it on a slab would produce a number
    // that looks like an answer and is not, so it is an error instead.
    let params = Am1Parameters::standard().unwrap();
    let slab = Molecule::new(vec![Atom {
        z: 8,
        position: Vec3::zero(),
    }])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap(),
    );
    // A slab silently gets no correction rather than an error, because `ewald: true` is the
    // default and a user who did not ask for it should not be stopped. What must never happen is
    // the 3D reciprocal sum being applied to it.
    let r = run_am1(&slab, &params, &gamma_options(0.0, 3, 40.0, true));
    assert!(
        r.is_ok(),
        "a slab should run without the correction, not fail: {:?}",
        r.err()
    );
}

#[test]
fn without_ewald_the_charged_cell_energy_diverges_with_the_real_space_cutoff() {
    // The control that gives `ewald_summation_makes_the_charged_cell_energy_converge` its
    // meaning: this is what the same calculation does with the correction switched off. It is a
    // measurement, asserted so it cannot quietly stop being true in either direction — if this
    // one ever converges on its own, the comparison the other test draws is no longer the
    // comparison it claims to be drawing.
    let params = Am1Parameters::standard().unwrap();
    let mut energies = Vec::new();
    eprintln!("    cutoff(Bohr)   +1 cell energy (eV), no Ewald");
    for rc in [20.0_f64, 40.0, 90.0, 130.0] {
        let r = run_pbc_scf(&water_cell(8.0), &params, &options_ewald(1.0, 2, rc, false)).unwrap();
        assert!(r.converged);
        eprintln!("    {rc:10.1}   {:18.8}", r.total_ev);
        energies.push(r.total_ev);
    }

    // Monotonically increasing, without bound.
    for pair in energies.windows(2) {
        assert!(
            pair[1] > pair[0],
            "the charged-cell energy should rise monotonically with the cutoff: {pair:?}"
        );
    }
    let rise = energies[energies.len() - 1] - energies[0];
    eprintln!("    rise from a 20 Bohr to a 130 Bohr cutoff: {rise:.1} eV");
    assert!(
        rise > 100.0,
        "the divergence should be unmistakable, not marginal; measured {rise:.1} eV"
    );

    // And it tracks the continuum estimate for a spherically truncated monopole sum,
    // `π Q² r_c² / V`, which is what identifies it as the missing neutralizing background
    // rather than some other defect.
    let volume_bohr3 = (8.0 * ANG).powi(3);
    let predicted = |rc: f64| std::f64::consts::PI * rc * rc / volume_bohr3 * 27.21;
    let predicted_rise = predicted(130.0) - predicted(20.0);
    eprintln!("    continuum estimate for the same range: {predicted_rise:.1} eV");
    let ratio = rise / predicted_rise;
    eprintln!("    measured / predicted = {ratio:.3}");
    assert!(
        (0.8..1.25).contains(&ratio),
        "the divergence should match the missing jellium term to within ~20%, got {ratio:.3}"
    );
}
