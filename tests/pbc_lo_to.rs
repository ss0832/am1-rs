// SPDX-License-Identifier: GPL-3.0-or-later

//! LO–TO splitting: the non-analytic term in `D(q)`.
//!
//! In a polar material the dipole–dipole force constants decay as `R⁻³`, so `Φ(T)` is not
//! short-ranged and no finite supercell captures it. Fourier-transforming a truncated `Φ(T)`
//! therefore gets the `q → 0` limit wrong however large the supercell is — the limit is
//! **direction dependent**, and a truncated sum has no way to be.
//!
//! The missing piece is supplied analytically from the Born charges and the electronic
//! dielectric tensor:
//!
//! ```text
//! D_NA(q)_{aα,bβ} = (4π/Ω) (q·Z*_a)_α (q·Z*_b)_β / (q·ε_∞·q) / √(m_a m_b)
//! ```
//!
//! # Why every system here is three-dimensional
//!
//! That expression is the **3D** one: `4π/(Ω q·ε·q)` is the Fourier transform of the
//! dipole–dipole interaction in three dimensions, and `Ω` is a volume. In two dimensions the
//! kernel is `2π/(A q)`; in one, the non-analytic part vanishes as `q² ln q`, so a genuinely
//! 1D-periodic chain has **no** LO–TO splitting as `q → 0`.
//!
//! Before 0.2.1 these tests ran the 3D formula on 1D chains, with `Ω` taken from
//! `Lattice::measure` — which returns a *length* for a chain. The splitting they measured was an
//! artifact of a dimensionally inconsistent denominator, not physics. The systems below are
//! fully periodic, and the chain now appears only in the test that asserts it is **refused**.
//!
//! Three things have to be true for the term to be right rather than merely present:
//!
//! 1. it must **vanish for a non-polar system**, where `Z* = 0`;
//! 2. it must **raise** frequencies, never lower them — it is a positive-semidefinite rank-one
//!    addition, so no branch can soften; and
//! 3. it must make the limit **direction dependent**, which is the whole point.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::phonon::ForceConstants;
use am1_rs::pbc::{born_charges, dielectric_tensor, KMesh, PbcOptions};
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn cubic(a_ang: f64) -> Lattice {
    Lattice::cubic(a_ang * ANG).unwrap()
}

/// One water molecule per cubic cell: a polar molecular crystal, fully periodic.
fn water_crystal(a_ang: f64) -> Molecule {
    let atoms: Vec<Atom> = [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9614, 0.0, 0.0]),
        (1, [-0.2246, 0.9348, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG,
    })
    .collect();
    Molecule::new(atoms).with_cell(cubic(a_ang))
}

/// One H₂ per cubic cell. The crystal has an inversion centre at the bond midpoint that swaps the
/// two atoms, so `Z*_1 = Z*_2`; with the acoustic sum rule `Σ_a Z*_a = 0` that forces both to
/// vanish, and the non-analytic term with them.
fn h2_crystal(a_ang: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.7 * ANG, 0.0, 0.0),
        },
    ])
    .with_cell(cubic(a_ang))
}

/// A 1D chain — used only to assert that it is refused.
fn water_chain(spacing_ang: f64) -> Molecule {
    let l = spacing_ang * ANG;
    let atoms: Vec<Atom> = [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9614, 0.0, 0.0]),
        (1, [-0.2246, 0.9348, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG,
    })
    .collect();
    Molecule::new(atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(l, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn scf_options() -> Am1Options {
    Am1Options {
        realspace_cutoff: 25.0,
        exchange_cutoff: Some(9.0),
        e_tol: 1.0e-10,
        p_tol: 1.0e-9,
        max_scf: 600,
        ..Am1Options::default()
    }
}

fn pbc_options() -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack([2, 2, 2]),
        realspace_cutoff: 25.0,
        exchange_cutoff: Some(9.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        ..PbcOptions::default()
    }
}

/// Frequencies at Γ with and without the non-analytic term, for one approach direction.
fn compare(molecule: &Molecule, direction: Vec3) -> (Vec<f64>, Vec<f64>) {
    let params = Am1Parameters::standard().unwrap();
    let fc = ForceConstants::from_supercell(molecule, &params, &scf_options(), [1, 1, 1]).unwrap();
    let z = born_charges(molecule, &params, &pbc_options()).unwrap();
    let (_, eps) = dielectric_tensor(molecule, &params, &pbc_options()).unwrap();
    let measure = molecule.cell.unwrap().measure();
    let gamma = am1_rs::pbc::kpoints::KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    };
    let plain = fc.frequencies(gamma).unwrap();
    let split = fc
        .frequencies_with_lo_to(gamma, direction, &z, &eps, measure)
        .unwrap();
    (plain, split)
}

#[test]
fn a_non_polar_crystal_is_untouched() {
    // `Z* = 0` by symmetry, so the non-analytic term must vanish. This separates "the term is
    // right" from "the term is large": a wrong prefactor would still leave this case alone, but a
    // term that ignored `Z*` would not.
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_crystal(4.0);
    let z = born_charges(&molecule, &params, &pbc_options()).unwrap();
    let worst_z = z
        .iter()
        .flatten()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("    symmetric H2 crystal: max |Z*| = {worst_z:.3e} e");
    assert!(
        worst_z < 1.0e-8,
        "inversion symmetry plus the acoustic sum rule force Z* = 0 here; got {worst_z:.3e} e"
    );

    let (plain, split) = compare(&molecule, Vec3::new(1.0, 0.0, 0.0));
    let mut worst = 0.0_f64;
    for (a, b) in plain.iter().zip(&split) {
        worst = worst.max((a - b).abs());
    }
    // The residual is `Z*²` fed through a `4π/Ω` prefactor and then a square root, so it is far
    // larger than `Z*` itself. Recorded at the size it has rather than bounded by argument.
    eprintln!("    max |Δν| from LO-TO = {worst:.3e} cm^-1");
    assert!(
        worst < 1.0e-3,
        "a non-polar system must have no LO-TO splitting, got {worst:.3e} cm^-1"
    );
}

/// The sharp test: the term added is **exactly** its closed form.
///
/// Comparing `D_with − D_without` against `(4π/Ω)(q̂·Z*_a)_α(q̂·Z*_b)_β / (q̂·ε∞·q̂) / √(m_a m_b)`,
/// assembled here from the same `Z*` and `ε∞` but by separately written arithmetic, checks the
/// prefactor, the tensor contraction order and the mass weighting all at once — none of which a
/// "the splitting is bigger than X cm⁻¹" threshold can see. It also does not care whether the
/// splitting happens to be large for the system chosen, which for a molecular crystal in a roomy
/// cell it is not.
#[test]
fn the_non_analytic_term_matches_its_closed_form() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_crystal(4.5);
    let direction = Vec3::new(1.0, 0.0, 0.0);

    let fc = ForceConstants::from_supercell(&molecule, &params, &scf_options(), [1, 1, 1]).unwrap();
    let z = born_charges(&molecule, &params, &pbc_options()).unwrap();
    let (_, eps) = dielectric_tensor(&molecule, &params, &pbc_options()).unwrap();
    let measure = molecule.cell.unwrap().measure();
    let gamma = am1_rs::pbc::kpoints::KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    };

    let plain = fc.dynamical_matrix(gamma);
    let split = fc
        .dynamical_matrix_with_lo_to(gamma, direction, &z, &eps, measure)
        .unwrap();

    // The closed form, written out independently.
    let n = direction.norm();
    let qhat = [direction.x / n, direction.y / n, direction.z / n];
    let mut denom = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            denom += qhat[a] * eps[a][b] * qhat[b];
        }
    }
    let qz: Vec<[f64; 3]> = z
        .iter()
        .map(|zt| {
            let mut v = [0.0_f64; 3];
            for (alpha, vv) in v.iter_mut().enumerate() {
                for (gamma_i, qg) in qhat.iter().enumerate() {
                    *vv += qg * zt[gamma_i][alpha];
                }
            }
            v
        })
        .collect();
    let a0_sq = (1.0 / 0.529167_f64).powi(2);
    let prefactor = 4.0 * std::f64::consts::PI / (measure * denom);

    let nat = molecule.atoms.len();
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for a in 0..nat {
        for b in 0..nat {
            let inv_mass = 1.0 / (fc.masses[a] * fc.masses[b]).sqrt();
            for i in 0..3 {
                for j in 0..3 {
                    let expected = prefactor * qz[a][i] * qz[b][j] * a0_sq * inv_mass;
                    let (sr, _) = split.get(3 * a + i, 3 * b + j);
                    let (pr, _) = plain.get(3 * a + i, 3 * b + j);
                    worst = worst.max((sr - pr - expected).abs());
                    scale = scale.max(expected.abs());
                }
            }
        }
    }
    eprintln!(
        "    max |(D_with − D_without) − closed form| = {worst:.3e} of a term whose largest \
         element is {scale:.3e} eV/(Å²·amu)"
    );
    assert!(
        scale > 1.0e-6,
        "the non-analytic term is numerically zero here ({scale:.3e}), so this test proves \
         nothing; pick a more polar system"
    );
    assert!(
        worst < 1.0e-10 * scale.max(1.0),
        "the added term is not its own closed form: off by {worst:.3e}"
    );
}

/// The term is positive semidefinite, so it can only raise **eigenvalues**.
///
/// Asserted on `λ`, not on `ν`: the theorem (Weyl interlacing for a rank-one positive
/// semidefinite update) is a statement about eigenvalues, and `ν = −√|λ|` for an imaginary mode
/// makes the frequency a non-monotonic re-encoding of it near zero.
#[test]
fn the_non_analytic_term_can_only_raise_eigenvalues() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_crystal(4.5);
    let fc = ForceConstants::from_supercell(&molecule, &params, &scf_options(), [1, 1, 1]).unwrap();
    let z = born_charges(&molecule, &params, &pbc_options()).unwrap();
    let (_, eps) = dielectric_tensor(&molecule, &params, &pbc_options()).unwrap();
    let measure = molecule.cell.unwrap().measure();
    let gamma = am1_rs::pbc::kpoints::KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    };

    let plain = am1_rs::pbc::hermitian_eigen(&fc.dynamical_matrix(gamma)).unwrap();
    let split = am1_rs::pbc::hermitian_eigen(
        &fc.dynamical_matrix_with_lo_to(gamma, Vec3::new(1.0, 0.0, 0.0), &z, &eps, measure)
            .unwrap(),
    )
    .unwrap();

    let mut worst_drop = 0.0_f64;
    let mut max_rise = 0.0_f64;
    for (i, (a, b)) in plain.values.iter().zip(&split.values).enumerate() {
        let shift = b - a;
        eprintln!("      lambda {i:3}: {a:14.6} -> {b:14.6}  ({shift:+.3e})");
        worst_drop = worst_drop.min(shift);
        max_rise = max_rise.max(shift);
    }
    eprintln!("    largest rise {max_rise:.3e}, largest fall {worst_drop:.3e} eV/(Å²·amu)");
    assert!(
        worst_drop > -1.0e-9,
        "an eigenvalue fell by {worst_drop:.3e}; a positive-semidefinite rank-one addition cannot \
         lower one"
    );
    assert!(
        max_rise > 1.0e-6,
        "the addition did nothing measurable ({max_rise:.3e}); this test would pass on a no-op"
    );
}

#[test]
fn the_limit_depends_on_the_direction_of_approach() {
    // The defining property, and the one a Fourier-interpolated `Φ(T)` structurally cannot have.
    let molecule = water_crystal(4.5);
    let params = Am1Parameters::standard().unwrap();
    let (alpha, eps) = dielectric_tensor(&molecule, &params, &pbc_options()).unwrap();
    eprintln!(
        "    alpha_xx = {:.4} Bohr^3, epsilon_inf diagonal = ({:.4}, {:.4}, {:.4})",
        alpha[0][0], eps[0][0], eps[1][1], eps[2][2]
    );
    // `ε_∞` sits in the denominator of the non-analytic term. Before 0.2.1's unit fix the
    // polarizability was 27x too small, so `ε_∞` was ~1.003 rather than ~1.09 and the term was
    // correspondingly too large. Asserting it is meaningfully above 1 is what makes the
    // splitting below a test of the whole chain rather than of `Z*` alone.
    assert!(
        eps[0][0] > 1.02,
        "epsilon_infinity came out as {:.6}, indistinguishable from vacuum -- suspect the \
         polarizability's units",
        eps[0][0]
    );

    let along = compare(&molecule, Vec3::new(1.0, 0.0, 0.0)).1;
    let across = compare(&molecule, Vec3::new(0.0, 0.0, 1.0)).1;
    let mut worst = 0.0_f64;
    for (a, b) in along.iter().zip(&across) {
        worst = worst.max((a - b).abs());
    }
    eprintln!("    largest difference between two approach directions: {worst:.4} cm^-1");
    assert!(
        worst > 1.0,
        "the q -> 0 limit should depend on direction once the non-analytic term is present; the \
         two directions differ by only {worst:.4} cm^-1"
    );
}

#[test]
fn exactly_zero_q_is_refused_rather_than_guessed() {
    // The term is genuinely undefined at `q = 0`: it depends on the direction of approach and
    // there is no direction there. Returning some value would be inventing one.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_crystal(4.5);
    let fc = ForceConstants::from_supercell(&molecule, &params, &scf_options(), [1, 1, 1]).unwrap();
    let z = born_charges(&molecule, &params, &pbc_options()).unwrap();
    let (_, eps) = dielectric_tensor(&molecule, &params, &pbc_options()).unwrap();
    let gamma = am1_rs::pbc::kpoints::KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    };
    let err = fc
        .frequencies_with_lo_to(
            gamma,
            Vec3::zero(),
            &z,
            &eps,
            molecule.cell.unwrap().measure(),
        )
        .unwrap_err();
    assert!(err.to_string().contains("direction"), "{err}");
}

/// A chain and a slab are **refused**, not silently given the three-dimensional answer.
///
/// This is the 0.2.1 correction. The formula's `4π/(Ω q·ε∞·q)` is the 3D dipole–dipole kernel and
/// `Ω` must be a volume; `Lattice::measure` hands back a length for a chain, so the pre-0.2.1
/// result was dimensionally not a dielectric response at all. A genuinely 1D-periodic chain has
/// no LO–TO splitting as `q → 0`.
#[test]
fn a_low_dimensional_cell_is_refused_rather_than_given_the_three_dimensional_answer() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(3.4);

    // The dielectric tensor is refused first, because `ε∞ = 1 + 4πα/Ω` needs a volume.
    let err = dielectric_tensor(&molecule, &params, &pbc_options()).unwrap_err();
    eprintln!("    dielectric: {err}");
    assert!(err.to_string().contains("three-dimensional"), "{err}");

    // And so is the non-analytic term itself, reached with hand-made inputs so that the refusal
    // is the function's own and not inherited from the tensor above.
    let fc = ForceConstants::from_supercell(&molecule, &params, &scf_options(), [2, 1, 1]).unwrap();
    let z = vec![[[0.0_f64; 3]; 3]; molecule.atoms.len()];
    let eps = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let gamma = am1_rs::pbc::kpoints::KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    };
    let err = fc
        .frequencies_with_lo_to(gamma, Vec3::new(1.0, 0.0, 0.0), &z, &eps, 1.0)
        .unwrap_err();
    eprintln!("    LO-TO: {err}");
    assert!(err.to_string().contains("three-dimensional"), "{err}");
}
