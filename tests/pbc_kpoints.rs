// SPDX-License-Identifier: GPL-3.0-or-later

//! k-point sampling.
//!
//! Two independent checks pin the machinery down, and between them they cover essentially
//! everything that can go wrong:
//!
//! 1. **Γ agrees with the Γ-only path.** The k-point code keeps real-space blocks `H(0,T)`
//!    and Bloch-sums them; the Γ-only path sums them into one matrix up front. At `k = 0`
//!    every phase is 1, so the two must agree exactly. This validates the block bookkeeping
//!    against code that is already tested.
//!
//! 2. **Band folding.** An `n`-point mesh along an axis of the primitive cell must give the
//!    same energy per cell as Γ on an `n`-fold supercell, because the two sample the same
//!    Hilbert space. This is the sharp test of the *phases*: a sign error, a transpose, or a
//!    `+ik·T` where `−ik·T` belongs all survive test 1 and fail this one.
//!
//! Band folding also demonstrates the thing k-points are here for. At Γ the real-space
//! density matrix `P(0,T)` is the same for every translation — it does not decay — which is
//! why NDDO's `1/R` exchange has to be tapered away by hand. With a mesh, `P(0,T)` decays,
//! and the last test measures that directly.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{run_pbc_scf, KMesh, PbcOptions};
use am1_rs::{run_am1, Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn hydrogen_chain(cells: usize, a: f64) -> Molecule {
    // A chain of H2 units along x: a clean, gapped, genuinely one-dimensional test case with
    // real dispersion, so a k-mesh has something to sample.
    let mut atoms = Vec::new();
    for c in 0..cells {
        let base = c as f64 * a;
        for d in [0.0, 1.4 * ANG] {
            atoms.push(Atom {
                z: 1,
                position: Vec3::new(base + d, 0.0, 0.0),
            });
        }
    }
    Molecule::new(atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(cells as f64 * a, 0.0, 0.0),
            Vec3::new(0.0, 60.0, 0.0),
            Vec3::new(0.0, 0.0, 60.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn pbc_options(kmesh: KMesh) -> PbcOptions {
    PbcOptions {
        kmesh,
        fold_time_reversal: false,
        realspace_cutoff: 60.0,
        exchange_cutoff: Some(25.0),
        smearing_ev: 0.0,
        max_scf: 400,
        mixing: 0.25,
        ..PbcOptions::default()
    }
}

#[test]
fn gamma_through_the_k_point_path_matches_the_gamma_only_path() {
    let params = Am1Parameters::standard().unwrap();
    let a = 6.0_f64;
    let system = hydrogen_chain(1, a);

    let via_kpoints = run_pbc_scf(&system, &params, &pbc_options(KMesh::Gamma)).unwrap();
    let via_gamma = run_am1(
        &system,
        &params,
        &Am1Options {
            realspace_cutoff: 60.0,
            exchange_cutoff: Some(25.0),
            ..Am1Options::default()
        },
    )
    .unwrap();

    assert!(via_kpoints.converged, "k-point path did not converge");
    assert!(via_gamma.converged, "Gamma-only path did not converge");
    let d = via_kpoints.total_ev - via_gamma.total_ev;
    eprintln!(
        "    k-point path at Gamma {:.9} eV, Gamma-only path {:.9} eV, delta {d:.3e}",
        via_kpoints.total_ev, via_gamma.total_ev
    );
    assert!(
        d.abs() < 1.0e-6,
        "the two Gamma routes disagree by {d:.3e} eV; the real-space block bookkeeping is wrong"
    );
}

#[test]
fn a_k_mesh_folds_onto_the_equivalent_supercell() {
    // The phase test. An n-point mesh on the primitive cell and Gamma on the n-fold supercell
    // describe the same infinite chain, so the energy *per primitive cell* must agree.
    let params = Am1Parameters::standard().unwrap();
    let a = 6.0_f64;

    eprintln!("    n    k-mesh on primitive     Gamma on n-supercell/n      delta");
    let mut worst = 0.0_f64;
    for n in [2usize, 3, 4] {
        let primitive = hydrogen_chain(1, a);
        let supercell = hydrogen_chain(n, a);

        let mesh = run_pbc_scf(
            &primitive,
            &params,
            &pbc_options(KMesh::MonkhorstPack([n, 1, 1])),
        )
        .unwrap();
        let folded = run_pbc_scf(&supercell, &params, &pbc_options(KMesh::Gamma)).unwrap();

        assert!(mesh.converged, "n={n}: k-mesh run did not converge");
        assert!(folded.converged, "n={n}: supercell run did not converge");

        let per_cell = folded.total_ev / n as f64;
        let d = mesh.total_ev - per_cell;
        eprintln!(
            "    {n}    {:18.9}     {per_cell:18.9}   {d:+.3e}",
            mesh.total_ev
        );
        worst = worst.max(d.abs());
    }
    assert!(
        worst < 1.0e-5,
        "band folding is broken: worst disagreement {worst:.3e} eV"
    );
}

#[test]
fn a_forced_unrestricted_closed_shell_reproduces_the_restricted_answer() {
    // The standard UHF sanity check, at k-points: for a closed shell with no symmetry
    // breaking, the alpha and beta channels must converge to the same solution and reproduce
    // the restricted energy. If the spin bookkeeping is wrong -- a capacity of 2 where 1
    // belongs, or the Coulomb term seeing one channel instead of the total -- this is where it
    // shows.
    let params = Am1Parameters::standard().unwrap();
    let chain = hydrogen_chain(1, 6.0);
    let mesh = KMesh::MonkhorstPack([8, 1, 1]);

    let rhf = run_pbc_scf(&chain, &params, &pbc_options(mesh)).unwrap();
    let uhf = run_pbc_scf(
        &chain,
        &params,
        &PbcOptions {
            unrestricted: true,
            ..pbc_options(mesh)
        },
    )
    .unwrap();

    assert!(rhf.converged && uhf.converged);
    assert!(uhf.unrestricted && !rhf.unrestricted);
    let d = uhf.total_ev - rhf.total_ev;
    eprintln!(
        "    restricted {:.9} eV, forced unrestricted {:.9} eV, delta {d:.3e}",
        rhf.total_ev, uhf.total_ev
    );
    assert!(
        d.abs() < 1.0e-7,
        "forced UHF disagrees with RHF on a closed shell by {d:.3e} eV"
    );

    // With no symmetry breaking the spin density must vanish everywhere.
    let spin = uhf.spin_density.expect("UHF should report a spin density");
    let worst = spin
        .translations
        .iter()
        .map(|t| spin.block_norm(*t))
        .fold(0.0_f64, f64::max);
    eprintln!("    largest |P_alpha - P_beta| element = {worst:.3e}");
    assert!(
        worst < 1.0e-6,
        "a closed shell acquired a spin density of {worst:.3e}"
    );
}

/// An equally spaced hydrogen chain, one atom per cell: one electron and one orbital, so the
/// band is exactly half filled.
fn atomic_hydrogen_chain(a: f64) -> Molecule {
    Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.0, 0.0, 0.0),
    }])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, 60.0, 0.0),
            Vec3::new(0.0, 0.0, 60.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn metallic_options() -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack([12, 1, 1]),
        fold_time_reversal: false,
        realspace_cutoff: 48.0,
        exchange_cutoff: Some(20.0),
        smearing_ev: 0.2,
        mixing: 0.2,
        max_scf: 600,
        ..PbcOptions::default()
    }
}

#[test]
fn a_half_filled_band_carries_entropy_and_a_fermi_level() {
    // One electron per cell in a band of capacity two: a metal. This is the case sharp aufbau
    // cannot describe, because the occupation of the frontier states is genuinely fractional
    // and varies across the zone. It is also the case a molecular SCF never meets.
    let params = Am1Parameters::standard().unwrap();
    let result = run_pbc_scf(&atomic_hydrogen_chain(4.0), &params, &metallic_options()).unwrap();

    eprintln!(
        "    half-filled H chain: E = {:.6} eV, E_F = {:.4} eV, T*S = {:.3e} eV, {} k-points",
        result.total_ev, result.fermi_energy_ev, result.entropy_ev, result.k_points
    );
    assert!(result.converged, "the metallic chain did not converge");
    assert!(
        result.entropy_ev > 1.0e-4,
        "a half-filled band at 0.2 eV smearing should carry entropy, got {:.3e}",
        result.entropy_ev
    );
    assert!(result.total_ev.is_finite() && result.total_ev < 0.0);
    // The free energy and the T -> 0 extrapolation must bracket the band energy sensibly.
    assert!(result.free_energy_ev() < result.total_ev);
    assert!(result.extrapolated_energy_ev() < result.total_ev);
}

#[test]
fn a_spin_polarized_chain_actually_polarizes() {
    // Same chain forced to maximum spin: alpha holds the electron, beta is empty, so the spin
    // density must be large rather than zero -- the complement of the closed-shell test above.
    let params = Am1Parameters::standard().unwrap();
    let result = run_pbc_scf(
        &atomic_hydrogen_chain(4.0),
        &params,
        &PbcOptions {
            multiplicity: 2,
            ..metallic_options()
        },
    )
    .unwrap();

    assert!(result.converged, "the polarized chain did not converge");
    assert!(result.unrestricted);
    let spin = result
        .spin_density
        .as_ref()
        .expect("UHF reports a spin density");
    let on_site = spin.block_norm(am1_rs::lattice::ImageOffset::origin());
    eprintln!(
        "    fully polarized H chain: E = {:.6} eV, on-site |P_alpha - P_beta| = {on_site:.4}",
        result.total_ev
    );
    assert!(
        on_site > 0.5,
        "one unpaired electron per cell should give an on-site spin density near 1, got {on_site:.4}"
    );
}

#[test]
fn the_density_matrix_decays_with_a_mesh_but_not_at_gamma() {
    // The reason k-points matter for NDDO. At Gamma, P(0,T) = P(Gamma) for every translation:
    // it is constant, so the 1/R exchange summed over images diverges. With a mesh the phases
    // interfere and P(0,T) falls off, which is what makes the exchange converge on its own.
    use am1_rs::lattice::ImageOffset;

    let params = Am1Parameters::standard().unwrap();
    let a = 6.0_f64;
    let chain = hydrogen_chain(1, a);

    let gamma = run_pbc_scf(&chain, &params, &pbc_options(KMesh::Gamma)).unwrap();
    let mesh = run_pbc_scf(
        &chain,
        &params,
        &pbc_options(KMesh::MonkhorstPack([16, 1, 1])),
    )
    .unwrap();

    eprintln!("     T      |P(0,T)| at Gamma    |P(0,T)| with 16 k-points");
    let mut gamma_far = 0.0_f64;
    let mut mesh_far = 0.0_f64;
    for n in [0i32, 1, 2, 3, 5, 8] {
        let t = ImageOffset { n: [n, 0, 0] };
        let g = gamma.density.block_norm(t);
        let m = mesh.density.block_norm(t);
        eprintln!("    {n:3}      {g:16.9}      {m:16.9}");
        if n >= 5 {
            gamma_far = gamma_far.max(g);
            mesh_far = mesh_far.max(m);
        }
    }

    let near = mesh.density.block_norm(ImageOffset { n: [0, 0, 0] });
    eprintln!(
        "\n    far/near ratio: Gamma {:.3e}, mesh {:.3e}",
        gamma_far / near,
        mesh_far / near
    );
    assert!(
        mesh_far < 0.2 * gamma_far,
        "the sampled density matrix should decay much faster than the Gamma one: \
         {mesh_far:.3e} vs {gamma_far:.3e}"
    );
}
