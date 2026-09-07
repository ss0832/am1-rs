// SPDX-License-Identifier: GPL-3.0-or-later

//! The two ways of contracting `C(q)`, and that they are the same sum.
//!
//! `C(q)_{j,j'} = Σ_σ Σ_k w_k Tr[h⁽¹⁾ʲσ(k)† ΔPʲ'σ(k)]` can be evaluated two ways:
//!
//! * **sparse** — keep `h⁽¹⁾ʲ` in real space as its nonzero entries and fold the Bloch phase in
//!   per entry, which costs `ndof² · n_k · nnz`;
//! * **dense** — Bloch-sum `h⁽¹⁾ʲ(k)` once per `(j, k)` and take a full `nao²` trace, which costs
//!   `ndof · n_k · nnz + ndof² · n_k · nao²`.
//!
//! Neither is right for every cell. The module assumed sparse always won, and on the systems it
//! was validated against — an H₂ chain with `nao = 4` — it does. But `nnz` counts entries across
//! every translation inside the real-space cutoff, and a small 3D cell admits hundreds of them:
//! rutile GeO₂ measures `nnz = 110944` against `nao² = 576`, where the sparse route does 190 times
//! the work. `select_contraction` now picks by cost.
//!
//! That makes this file's job the correctness half. The two orders of summation must agree to
//! roundoff, and a mistransposed `h†` or a dropped weight would still give a Hermitian `C(q)` with
//! plausible frequencies — so the comparison has to be against the other route, on the same
//! system, rather than against a symmetry property either would satisfy.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::params::Am1Parameters;
use am1_rs::pbc::dfpt::{force_constants_at_q_with, DfptOptions};
use am1_rs::pbc::kpoints::{KMesh, KPoint};
use am1_rs::pbc::PbcOptions;
use am1_rs::system::{Atom, Molecule};

const ANG: f64 = am1_rs::constants::ANGSTROM_TO_BOHR;

/// A short H₂ chain: `nao = 4`, few translations, so the *sparse* route is the cheap one.
fn chain() -> Molecule {
    let a = 3.0 * ANG;
    let cell = Lattice::from_vectors(
        Vec3::new(60.0 * ANG, 0.0, 0.0),
        Vec3::new(0.0, 60.0 * ANG, 0.0),
        Vec3::new(0.0, 0.0, a),
        [false, false, true],
    )
    .unwrap();
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.6766 * ANG),
        },
    ])
    .with_cell(cell)
}

/// The same chain repeated `n` times along its periodic axis.
///
/// `nao` grows with the supercell while `nnz` does not — the nonzeros are set by how far the
/// perturbation reaches, which is a distance and not a cell count — so this walks the crossover
/// between the two contraction routes without changing the physics.
fn chain_supercell(repeats: usize) -> Molecule {
    let a = 3.0 * ANG;
    let cell = Lattice::from_vectors(
        Vec3::new(60.0 * ANG, 0.0, 0.0),
        Vec3::new(0.0, 60.0 * ANG, 0.0),
        Vec3::new(0.0, 0.0, a * repeats as f64),
        [false, false, true],
    )
    .unwrap();
    let mut atoms = Vec::new();
    for cell_index in 0..repeats {
        let z0 = a * cell_index as f64;
        atoms.push(Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, z0),
        });
        atoms.push(Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, z0 + 0.6766 * ANG),
        });
    }
    Molecule::new(atoms).with_cell(cell)
}
fn options(mesh: [usize; 3]) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        fold_time_reversal: false,
        max_scf: 400,
        ..Default::default()
    }
}

/// Run both routes and return the worst elementwise disagreement, with the scale of `C(q)`.
fn both_routes(molecule: &Molecule, mesh: [usize; 3], q: KPoint) -> (f64, f64, usize, usize) {
    let params = Am1Parameters::standard().unwrap();
    let options = options(mesh);
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    let mut counts = (0usize, 0usize);
    let sparse = force_constants_at_q_with(
        molecule,
        &params,
        &options,
        &DfptOptions {
            dense_contraction: Some(false),
            ..Default::default()
        },
        q,
    )
    .unwrap();
    let dense = force_constants_at_q_with(
        molecule,
        &params,
        &options,
        &DfptOptions {
            dense_contraction: Some(true),
            ..Default::default()
        },
        q,
    )
    .unwrap();
    counts.0 = sparse.bare_nonzeros;
    counts.1 = sparse.bare_dense_elements;
    let n = sparse.force_constants.re.rows;
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = sparse.force_constants.get(i, j);
            let (br, bi) = dense.force_constants.get(i, j);
            worst = worst.max((ar - br).abs()).max((ai - bi).abs());
            scale = scale.max(ar.abs()).max(ai.abs());
        }
    }
    (worst, scale, counts.0, counts.1)
}

/// A short chain, where the counts put it in the **dense** regime.
///
/// Worth stating because it is not what the module assumed. Even two hydrogens in a 3 A cell have
/// `nnz = 38` against `nao^2 = 4`: the nonzeros are counted over every translation inside the
/// 40 Bohr real-space cutoff, and a 3 A cell admits a lot of them. Sparse wins only once `nao^2`
/// overtakes that, which needs a cell with many atoms rather than many images.
#[test]
fn the_two_contractions_agree_on_a_short_chain() {
    let (worst, scale, nnz, dense) = both_routes(&chain(), [1, 1, 4], KPoint::gamma());
    eprintln!("chain: nnz {nnz} against nao^2 {dense}; |C| ~ {scale:.3e}, differ by {worst:.3e}");
    assert!(
        nnz > dense,
        "expected the dense regime here: nnz {nnz}, nao^2 {dense}"
    );
    assert!(
        worst < 1.0e-10 * scale.max(1.0),
        "the two contractions differ by {worst:e} on a matrix of scale {scale:e}"
    );
}

/// A supercell large enough that `nao^2` overtakes `nnz` — the regime the sparse route is for.
///
/// Both regimes have to be exercised, because the two routes share no code: a mistake in either
/// is invisible from the other side of the crossover.
#[test]
fn the_two_contractions_agree_where_sparse_wins() {
    let (worst, scale, nnz, dense) = both_routes(&chain_supercell(6), [1, 1, 2], KPoint::gamma());
    eprintln!(
        "chain x6: nnz {nnz} against nao^2 {dense}; |C| ~ {scale:.3e}, differ by {worst:.3e}"
    );
    assert!(
        nnz < dense,
        "expected the sparse regime here: nnz {nnz}, nao^2 {dense}"
    );
    assert!(
        worst < 1.0e-10 * scale.max(1.0),
        "the two contractions differ by {worst:e} on a matrix of scale {scale:e}"
    );
}

/// And away from `q = 0`, where the phase is not 1 and a mishandled conjugation shows.
///
/// At Γ every Bloch factor is real, so the sparse route's `hr = c·v0 − s·v1` reduces to `v0` and
/// the imaginary half of the contraction never runs. A sign error in it would pass both tests
/// above and fail here.
#[test]
fn the_two_contractions_agree_away_from_gamma() {
    let q = KPoint {
        fractional: [0.0, 0.0, 0.5],
        weight: 1.0,
    };
    let (worst, scale, ..) = both_routes(&chain(), [1, 1, 4], q);
    eprintln!("chain at q = (0,0,1/2): |C| ~ {scale:.3e}, differ by {worst:.3e}");
    assert!(
        worst < 1.0e-10 * scale.max(1.0),
        "the two contractions differ by {worst:e} on a matrix of scale {scale:e}"
    );
}

/// The incrementally carried Gram matrix is the one that would have been rebuilt.
///
/// `PulayGram` keeps `<r_i, r_j>` between CPSCF iterations and computes only the newest row,
/// because rebuilding all of it measured as 17 % of a DFPT phonon run — more than the two-electron
/// kernel build beside it. That is only worth having if it is the *same* matrix: a stale row would
/// give plausible Pulay coefficients pointing somewhere slightly wrong, and the solve would still
/// converge, to the same fixed point, a little slower. Nothing downstream would notice.
///
/// So this drives the incremental path through the same push-and-trim pattern the solver uses and
/// compares its coefficients against `pulay_coefficients`, which rebuilds from scratch.
#[test]
fn the_carried_gram_matches_a_rebuilt_one() {
    use am1_rs::pbc::scf::{pulay_coefficients, PulayGram};

    // Residuals with a deliberately wide dynamic range and a near-dependent pair, which is where
    // the scaling and the ridge in the solve actually do something.
    let mut history: Vec<Vec<f64>> = Vec::new();
    let mut gram = PulayGram::new();
    let depth = 6;
    let mut worst = 0.0_f64;
    for step in 0..20i32 {
        let scale = 10.0_f64.powi(-step / 3);
        let r: Vec<f64> = (0..97)
            .map(|i| scale * ((i as f64 * 0.37 + f64::from(step)).sin() + 0.001 * f64::from(step)))
            .collect();
        if history.len() == depth {
            history.remove(0);
            gram.drop_oldest();
        }
        history.push(r);
        let refs: Vec<&[f64]> = history.iter().map(|h| h.as_slice()).collect();
        let carried: Option<Vec<f64>> = gram.push_and_solve(&refs);
        let rebuilt: Option<Vec<f64>> = pulay_coefficients(&refs);
        match (carried, rebuilt) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(
                    a.len(),
                    b.len(),
                    "step {step}: different coefficient counts"
                );
                for (x, y) in a.iter().zip(&b) {
                    worst = worst.max((x - y).abs());
                }
            }
            (a, b) => panic!(
                "step {step}: the two paths disagree about whether a step exists: \
                 carried {} rebuilt {}",
                a.is_some(),
                b.is_some()
            ),
        }
    }
    eprintln!("carried against rebuilt Pulay coefficients: worst {worst:.3e}");
    assert!(
        worst < 1.0e-12,
        "the carried Gram gives different coefficients, worst {worst:e}"
    );
}
