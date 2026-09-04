// SPDX-License-Identifier: GPL-3.0-or-later

//! What the `q = 0` periodic response holds in memory, measured rather than reasoned about.
//!
//! # The claim
//!
//! The coupled-perturbed solve needs one perturbation's response density at a time. Until 0.2.2 it
//! built all `3N` of them before the loop that consumed them one by one, and kept a second array
//! of the same size for the spin-summed total — so the resident set was
//! `(1 + n_channels) · ndof · n_T · nao²` doubles where `(1 + n_channels) · n_T · nao²` will do.
//! The arithmetic is identical either way; only the order of the loop nest changed.
//!
//! # Why an allocator and not a counter
//!
//! A counter incremented by hand records what the author believed was allocated. This records what
//! *was*. The test installs a global allocator that tracks the high-water mark, runs the response,
//! and compares the peak against the size of the array the old shape would have needed — which is
//! computed here from the system's own dimensions, not from a remembered number.
//!
//! The bound is deliberately loose (a third of the old array). The point is the **order**: the
//! response density array no longer scales with the number of perturbations, and a factor-of-three
//! margin distinguishes that from a constant-factor tweak without pinning the test to the
//! incidental size of everything else the solve allocates.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{pbc_hessian, KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// Live bytes, and the high-water mark.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Tracking;

unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            // A compare-and-swap loop rather than `fetch_max`, which is not on the stable
            // `AtomicUsize` surface this crate's MSRV (1.75) can rely on.
            let mut peak = PEAK.load(Ordering::Relaxed);
            while now > peak {
                match PEAK.compare_exchange_weak(peak, now, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(seen) => peak = seen,
                }
            }
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(p, layout)
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let q = System.realloc(p, layout, new_size);
        if !q.is_null() {
            if new_size >= layout.size() {
                let grew = new_size - layout.size();
                let now = LIVE.fetch_add(grew, Ordering::Relaxed) + grew;
                let mut peak = PEAK.load(Ordering::Relaxed);
                while now > peak {
                    match PEAK.compare_exchange_weak(
                        peak,
                        now,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(seen) => peak = seen,
                    }
                }
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Tracking = Tracking;

/// A cubic cell of waters on a simple grid: three-dimensional so the translation set is large,
/// which is what makes `n_T · nao²` — the size of one response density — big enough for the
/// comparison below to be about the response and not about everything else the solve allocates.
fn water_cube(side: usize, spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    let mut atoms = Vec::new();
    for i in 0..side {
        for j in 0..side {
            let shift = Vec3::new(step * i as f64, step * j as f64, 0.0);
            for (z, r) in [
                (8u8, [0.0, 0.0, 0.0]),
                (1, [0.9614, 0.0, 0.0]),
                (1, [-0.2246, 0.9348, 0.0]),
            ] {
                atoms.push(Atom {
                    z,
                    position: Vec3::new(r[0], r[1], r[2]) * ANG + shift,
                });
            }
        }
    }
    let a = step * side as f64;
    Molecule::new(atoms).with_cell(Lattice::cubic(a).unwrap())
}

#[test]
fn the_response_does_not_hold_a_density_per_perturbation() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_cube(3, 3.6);
    let cutoff = 26.0;
    let options = PbcOptions {
        kmesh: KMesh::MonkhorstPack([2, 1, 1]),
        fold_time_reversal: false,
        realspace_cutoff: cutoff,
        exchange_cutoff: Some(12.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-9,
        p_tol: 1.0e-8,
        max_scf: 800,
        ..PbcOptions::default()
    };

    // The three arrays of this shape the response used to hold, taken from the system's own
    // dimensions rather than from a remembered number:
    //
    // 1. the per-channel response densities, one per perturbation;
    // 2. the spin-summed total, one per perturbation;
    // 3. the bare `∂F/∂R`, one per perturbation.
    //
    // The first two were removed in 0.2.2 by streaming the CPHF over perturbations; the third is
    // still held, and `CHANGELOG.md` records what removing it would take. So the peak should sit
    // near a **third** of what the old shape needed, and the bound below is half — loose enough
    // not to pin the test to the incidental size of the pair integrals and the SCF, tight enough
    // that leaving either of the two arrays in would fail it.
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let nao: usize = molecule
        .atoms
        .iter()
        .map(|a| if a.z == 1 { 1 } else { 4 })
        .sum();
    let n_t = molecule.cell.unwrap().image_offsets(cutoff).len();
    let one_density = n_t * nao * nao * std::mem::size_of::<f64>();
    let old_shape = 3 * ndof * one_density;

    // Two runs, and the **difference** is the claim. The ground-state SCF on its own already
    // allocates the pair integrals, the density blocks and the thread pools, and on a system this
    // size that baseline is larger than everything the response adds — so a bound on the absolute
    // peak would be a bound on the pair list. Subtracting the SCF's own peak leaves what the
    // response costs, which is what changed.
    let scf_peak = {
        PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
        let r = am1_rs::pbc::run_pbc_scf(&molecule, &params, &options).unwrap();
        assert!(r.converged, "the ground state did not converge");
        PEAK.load(Ordering::Relaxed) - LIVE.load(Ordering::Relaxed)
    };
    let (hessian, full_peak) = {
        PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
        let base = LIVE.load(Ordering::Relaxed);
        let h = pbc_hessian(&molecule, &params, &options).unwrap();
        (h, PEAK.load(Ordering::Relaxed) - base)
    };
    let response_peak = full_peak.saturating_sub(scf_peak);

    eprintln!(
        "    {nat} atoms, nao = {nao}, {n_t} translations, ndof = {ndof}\n    \
         one response density = {:.2} MB; the three per-perturbation arrays would have held \
         {:.1} MB\n    peak over the SCF's own: {:.1} MB (SCF {:.1} MB, total {:.1} MB)",
        one_density as f64 / 1.0e6,
        old_shape as f64 / 1.0e6,
        response_peak as f64 / 1.0e6,
        scf_peak as f64 / 1.0e6,
        full_peak as f64 / 1.0e6
    );

    // The Hessian still has to be right; a solver that allocates nothing and returns nonsense
    // would pass the bound below.
    let scale = hessian
        .as_slice()
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        scale > 1.0,
        "the force constants are {scale:.3e}, so nothing was solved"
    );
    assert!(
        old_shape > 8 * one_density,
        "this system is too small for the comparison to mean anything: ndof = {ndof}"
    );
    assert!(
        response_peak < old_shape / 2,
        "the response added {:.1} MB over the SCF, against the {:.1} MB the old shape needed, so \
         at least one of the two per-perturbation density arrays is still being held",
        response_peak as f64 / 1.0e6,
        old_shape as f64 / 1.0e6
    );
}
