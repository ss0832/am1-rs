// SPDX-License-Identifier: GPL-3.0-or-later

//! The batched Bloch sum: that it is the same arithmetic, and that it is worth doing.
//!
//! `RealSpaceBlocks::bloch_sum_all` replaced a loop of `bloch_sum` on the periodic SCF's critical
//! path. Two things have to hold for that to be a good trade, and they fail independently:
//!
//! * it must produce **exactly** what the loop produced — the batched form reassociates the sum
//!   into a matrix product, and a transposed phase matrix or a mispacked panel would still give a
//!   Hermitian matrix with plausible eigenvalues;
//! * it must actually be faster, measured **in one process against the loop it replaced** rather
//!   than across runs. Cross-run wall clock on a loaded machine moved every phase of this
//!   calculation by 2.5x in both directions while the arithmetic was unchanged, which is enough
//!   to manufacture or hide a speedup of the size being claimed.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::params::Am1Parameters;
use am1_rs::pbc::kpoints::KMesh;
use am1_rs::pbc::scf::{run_pbc_scf, PbcOptions};
use am1_rs::system::{Atom, Molecule};

const ANG: f64 = am1_rs::constants::ANGSTROM_TO_BOHR;

/// Rutile GeO₂ — six atoms, 24 AOs, and a cell small enough that the 40 Bohr real-space cutoff
/// admits several hundred translations, which is the regime the batching is for.
fn rutile() -> Molecule {
    let (a, c, u) = (4.3966 * ANG, 2.8624 * ANG, 0.3059);
    let cell = Lattice::from_vectors(
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(0.0, a, 0.0),
        Vec3::new(0.0, 0.0, c),
        [true; 3],
    )
    .unwrap();
    let atoms = vec![
        (32, 0.0, 0.0, 0.0),
        (32, 0.5, 0.5, 0.5),
        (8, u, u, 0.0),
        (8, 1.0 - u, 1.0 - u, 0.0),
        (8, 0.5 + u, 0.5 - u, 0.5),
        (8, 0.5 - u, 0.5 + u, 0.5),
    ]
    .into_iter()
    .map(|(z, x, y, zc)| Atom {
        z,
        position: Vec3::new(x * a, y * a, zc * c),
    })
    .collect();
    Molecule::new(atoms).with_cell(cell)
}

fn converged() -> (
    am1_rs::pbc::scf::PbcResult,
    Vec<am1_rs::pbc::kpoints::KPoint>,
) {
    let molecule = rutile();
    let params = Am1Parameters::standard().unwrap();
    let options = PbcOptions {
        kmesh: KMesh::MonkhorstPack([3, 3, 4]),
        max_scf: 300,
        ..Default::default()
    };
    let cell = molecule.cell.unwrap();
    let kpoints = options.resolve_kpoints(&cell).unwrap();
    let scf = run_pbc_scf(&molecule, &params, &options).unwrap();
    (scf, kpoints)
}

#[test]
fn the_batched_bloch_sum_is_the_loop_it_replaced() {
    let (scf, kpoints) = converged();
    let blocks = &scf.density;
    let batched = blocks.bloch_sum_all(&kpoints);
    assert_eq!(batched.len(), kpoints.len());
    let mut worst = 0.0_f64;
    for (ki, kp) in kpoints.iter().enumerate() {
        let one = blocks.bloch_sum(kp);
        let n = one.re.rows;
        for i in 0..n {
            for j in 0..n {
                let (ar, ai) = one.get(i, j);
                let (br, bi) = batched[ki].get(i, j);
                worst = worst.max((ar - br).abs()).max((ai - bi).abs());
            }
        }
    }
    // Not `==`: the batched form sums the translations in the blocked kernel's order rather than
    // the loop's, so the two differ by floating-point reassociation and nothing else. The blocks
    // themselves reach ~10 in magnitude, so this is a few ulp.
    assert!(
        worst < 1.0e-12,
        "batched and looped Bloch sums differ by {worst:e}"
    );
}

/// Interleaved A/B, in one process, on the same data.
///
/// Ignored by default because it is a measurement rather than an assertion — but it asserts the
/// direction, so a change that made the batch *slower* would fail rather than print.
#[test]
#[ignore]
fn the_batch_beats_the_loop() {
    use std::time::Instant;
    let (scf, kpoints) = converged();
    let blocks = &scf.density;
    let (mut looped, mut batched) = (f64::INFINITY, f64::INFINITY);
    // Alternating, best-of-N: a background load that arrives partway through hits both arms.
    for _ in 0..7 {
        let t = Instant::now();
        let mut sink = 0.0;
        for kp in &kpoints {
            sink += blocks.bloch_sum(kp).re[(0, 0)];
        }
        looped = looped.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        for m in blocks.bloch_sum_all(&kpoints) {
            sink += m.re[(0, 0)];
        }
        batched = batched.min(t.elapsed().as_secs_f64());
        assert!(sink.is_finite());
    }
    eprintln!(
        "{} translations x {} AOs at {} k-points: loop {:.1} ms, batch {:.1} ms ({:.1}x)",
        blocks.translations.len(),
        blocks.blocks[0].rows,
        kpoints.len(),
        looped * 1e3,
        batched * 1e3,
        looped / batched
    );
    assert!(
        batched < looped,
        "the batch was meant to be the faster one: {batched:.4} s against {looped:.4} s"
    );
}
