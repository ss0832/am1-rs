// SPDX-License-Identifier: GPL-3.0-or-later

// The loops below index by *orbital*, and which index is which is the whole content of the
// property being checked -- `(μν|λσ)` against `(νμ|λσ)`, `S_{μν}` against its parity, a rotation
// mixing `p_x, p_y, p_z`. Rewriting them as `iter().enumerate()` would hide exactly the thing
// under test, so the lint is declined here rather than obeyed, as it is in `pbc_dielectric.rs`.
#![allow(clippy::needless_range_loop)]

//! The **pieces** of the formulas, checked against what theory says each one must be.
//!
//! Everywhere else in this suite the test is an end-to-end identity: a frozen phonon against a
//! DFPT response, an analytic gradient against a finite difference, a heat of formation against
//! MOPAC's. Those are strong, and they are also *blunt* — they say a whole chain is right or
//! wrong without saying which link failed, and a compensating pair of errors passes them.
//!
//! The NDDO expressions are hard enough that this matters. A two-centre two-electron integral is
//! a sum of twenty-two Klopman–Ohno kernels evaluated at displaced points, rotated out of a local
//! diatomic frame; the electrostatics is an Ewald sum split between real and reciprocal space; a
//! converged SCF is a fixed point with its own algebra. Each of those has properties that follow
//! from the mathematics alone — an asymptotic limit, a permutation symmetry, an idempotency —
//! and a term that is wrong will usually break one of them.
//!
//! So this file tests the parts, not the whole. Each test names the property it checks and why
//! that property must hold, rather than pinning a number that came out of the code.
//!
//! What is deliberately *not* here: anything already covered at the part level elsewhere.
//! `src/integrals.rs` checks the rotation matrix's orthonormality and the frame-free rewrite;
//! `src/pbc/ewald*.rs` checks the splitting-parameter independence and reproduces the 1D, 2D and
//! 3D Madelung constants in closed form; `src/dual*.rs` checks the derivative arithmetic;
//! `src/fermi.rs` checks the filling. Those are the same idea and they stay where they are.

use am1_rs::constants::AM1_EV;
use am1_rs::integrals::{pack, pair_two_electron};
use am1_rs::linalg::Matrix;
use am1_rs::math::Vec3;
use am1_rs::overlap::diatom_overlap;
use am1_rs::scf::{run_am1, Am1Options};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// Orbital indices inside an atom's block: `s` and the three `p`.
const S: usize = 0;
const PX: usize = 1;
const PY: usize = 2;
const PZ: usize = 3;

fn params() -> Am1Parameters {
    Am1Parameters::standard().unwrap()
}

// ---------------------------------------------------------------- two-electron integrals

/// `(ss|ss) → e²/R` as the atoms separate.
///
/// The whole NDDO electrostatic model rests on this: at long range two atoms interact as point
/// charges, and every multipole refinement is a correction to that. The Klopman–Ohno kernel
/// `1/√(R² + ρ²)` reduces to `1/R` when `R ≫ ρ`, so the leading integral must approach the
/// Coulomb law — in **eV**, which is where the `AM1_EV` conversion enters.
///
/// This is what a wrong `ρ` or a dropped conversion factor breaks, and it breaks it by a
/// constant factor that no symmetry test would notice.
///
/// The test goes further than the limit, because the *approach* to it is also predicted. With
/// `(ss|ss) = e²/√(R² + (ρ⁰_i + ρ⁰_j)²)` the relative deviation from `e²/R` is
///
/// ```text
///   R/√(R² + ρ²) − 1  ≈  −ρ²/(2R²),      ρ = ρ⁰_i + ρ⁰_j
/// ```
///
/// so `deviation × R²` is a constant, and that constant *identifies* `ρ`. Recovering the
/// parameter table's own `rho0` from the integral's long-range behaviour is a far sharper check
/// than a magnitude bound: it pins the functional form, the two elements' parameters, and the
/// unit conversion at once.
#[test]
fn the_monopole_integral_approaches_the_coulomb_law_as_klopman_ohno_predicts() {
    let p = params();
    let (o, c) = (p.element(8).unwrap(), p.element(6).unwrap());
    let dir = Vec3::new(1.0, 0.0, 0.0);
    let rho = o.rho0 + c.rho0;

    eprintln!("      R (Bohr)   (ss|ss) eV      e^2/R eV      relative     dev*R^2");
    let mut implied = Vec::new();
    for r in [20.0_f64, 50.0, 100.0, 200.0] {
        let got = pair_two_electron(o, c, dir, r).two_e(S, S, S, S);
        let coulomb = AM1_EV / r;
        let dev = (got - coulomb) / coulomb;
        eprintln!(
            "      {r:8.1}  {got:12.8}  {coulomb:12.8}  {dev:+12.3e}  {:+9.4}",
            dev * r * r
        );
        assert!(
            dev < 0.0,
            "the Klopman-Ohno kernel is softer than 1/R, so the integral must fall below it"
        );
        implied.push((-2.0 * dev * r * r).sqrt());
    }

    // Every radius must imply the same ρ, and it must be the one in the parameter table.
    let worst = implied
        .iter()
        .map(|v| (v - rho).abs() / rho)
        .fold(0.0_f64, f64::max);
    eprintln!(
        "    implied rho = {:?} Bohr; rho0(O) + rho0(C) = {rho:.6}",
        implied
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
    );
    assert!(
        worst < 0.01,
        "the long-range deviation implies rho = {implied:?}, but the parameters say {rho:.6}"
    );
}

/// Each multipole channel falls off at the power its expansion order demands.
///
/// The Dewar–Sabelli–Klopman construction represents an orbital pair's charge distribution by a
/// point multipole: `(s,s)` is a monopole, `(s,p)` a dipole, `(p,p)` a monopole plus a
/// quadrupole. Their interactions must therefore decay as
///
/// ```text
///   monopole–monopole   1/R
///   monopole–dipole     1/R²
///   dipole–dipole       1/R³
/// ```
///
/// and this measures the exponent rather than asserting it. Getting a channel's *displacement*
/// wrong — the `dd`/`qq` additive terms that place the point charges — leaves the integral finite
/// and smooth and changes this exponent, which is exactly the kind of error the end-to-end tests
/// cannot localize.
#[test]
fn each_multipole_channel_decays_at_its_own_order() {
    let p = params();
    let (o, c) = (p.element(8).unwrap(), p.element(6).unwrap());
    let dir = Vec3::new(1.0, 0.0, 0.0);

    // Fit `log|integral|` against `log R` over a range where the multipole expansion is valid
    // and the values have not yet lost their digits to cancellation.
    let radii = [30.0_f64, 45.0, 60.0, 80.0];
    let exponent = |f: &dyn Fn(&am1_rs::integrals::PairTwoElec) -> f64| -> f64 {
        let (lx, ly): (Vec<f64>, Vec<f64>) = radii
            .iter()
            .map(|&r| {
                let te = pair_two_electron(o, c, dir, r);
                (r.ln(), f(&te).abs().ln())
            })
            .unzip();
        let n = lx.len() as f64;
        let (mx, my) = (lx.iter().sum::<f64>() / n, ly.iter().sum::<f64>() / n);
        let num: f64 = lx.iter().zip(&ly).map(|(a, b)| (a - mx) * (b - my)).sum();
        let den: f64 = lx.iter().map(|a| (a - mx) * (a - mx)).sum();
        num / den
    };

    /// One channel: its name, the exponent theory predicts, and how to read it out of the block.
    type Channel<'a> = (
        &'a str,
        f64,
        Box<dyn Fn(&am1_rs::integrals::PairTwoElec) -> f64>,
    );

    // The pair axis is x, so `p_x` is the longitudinal (sigma) direction.
    let cases: [Channel<'_>; 3] = [
        (
            "(ss|ss)  monopole-monopole",
            -1.0,
            Box::new(|t| t.two_e(S, S, S, S)),
        ),
        (
            "(ss|s px) monopole-dipole",
            -2.0,
            Box::new(|t| t.two_e(S, S, S, PX)),
        ),
        (
            "(s px|s px) dipole-dipole",
            -3.0,
            Box::new(|t| t.two_e(S, PX, S, PX)),
        ),
    ];
    for (name, want, f) in cases {
        let got = exponent(&*f);
        eprintln!("    {name:28} R^{got:+.3}  (theory R^{want:+.1})");
        assert!(
            (got - want).abs() < 0.05,
            "{name} decays as R^{got:.3}, not R^{want:.1}"
        );
    }
}

/// `(μν|λσ)` is symmetric under the three permutations its definition allows.
///
/// The integral is `∫∫ φ_μ(1)φ_ν(1) r₁₂⁻¹ φ_λ(2)φ_σ(2)`, so swapping `μ↔ν`, swapping `λ↔σ`, or
/// swapping the two electrons all leave it unchanged. The last one is the interesting case here:
/// it exchanges the two *atoms*, and the code reaches the two sides through different branches
/// (one centre supplies the bra multipoles, the other the ket), so it is a genuine check of the
/// assembly rather than of the packing.
#[test]
fn the_two_electron_integrals_have_their_permutation_symmetries() {
    let p = params();
    let (o, c) = (p.element(8).unwrap(), p.element(6).unwrap());
    let dir = Vec3::new(0.37, -0.55, 0.75).normalized();
    let r = 4.1;
    let te = pair_two_electron(o, c, dir, r);
    // The reversed pair: same separation, opposite direction, roles exchanged.
    let rev = pair_two_electron(c, o, dir * -1.0, r);

    let mut worst_bra = 0.0_f64;
    let mut worst_exchange = 0.0_f64;
    let mut scale = 0.0_f64;
    for a in 0..4 {
        for b in 0..4 {
            for cc in 0..4 {
                for d in 0..4 {
                    let v = te.two_e(a, b, cc, d);
                    scale = scale.max(v.abs());
                    // (μν|λσ) = (νμ|λσ) = (μν|σλ)
                    worst_bra = worst_bra.max((v - te.two_e(b, a, cc, d)).abs());
                    worst_bra = worst_bra.max((v - te.two_e(a, b, d, cc)).abs());
                    // Electron exchange, which here is atom exchange: (μν|λσ)_ij = (λσ|μν)_ji.
                    worst_exchange = worst_exchange.max((v - rev.two_e(cc, d, a, b)).abs());
                }
            }
        }
    }
    eprintln!("    bra/ket index swaps: {worst_bra:.3e};  atom exchange: {worst_exchange:.3e} of {scale:.3e}");
    assert!(
        worst_bra < 1.0e-13 * scale,
        "index swap broke by {worst_bra:.3e}"
    );
    assert!(
        worst_exchange < 1.0e-10 * scale,
        "exchanging the two centres changed the integrals by {worst_exchange:.3e}"
    );
}

/// The whole `10 × 10` block transforms as a tensor when the pair is rotated.
///
/// `src/integrals.rs` already checks that `(ss|ss)` — a scalar — is rotation invariant. That is
/// the weakest case: it cannot see the `p` functions at all. Under a rotation `R` the `p`
/// orbitals mix as vectors, so an integral carrying `p` indices must pick up the corresponding
/// factors of `R`, and the local-frame construction is exactly the code responsible for that.
///
/// The check is done on the **fully contracted** quantity `Σ P¹_{μν} P²_{λσ} (μν|λσ)` with the
/// densities rotated alongside: a genuine tensor identity that holds only if every index
/// transforms correctly, and one that does not require reimplementing the rotation to state.
#[test]
fn the_pair_block_is_rotation_covariant() {
    let p = params();
    let (o, c) = (p.element(8).unwrap(), p.element(6).unwrap());
    let r = 3.7;
    let dir = Vec3::new(1.0, 0.0, 0.0);

    // A rotation about z by 40 degrees, then about y by 25.
    let (a, b) = (40.0_f64.to_radians(), 25.0_f64.to_radians());
    let rot = |v: Vec3| -> Vec3 {
        let (x, y, z) = (v.x, v.y, v.z);
        let (x1, y1, z1) = (a.cos() * x - a.sin() * y, a.sin() * x + a.cos() * y, z);
        Vec3::new(
            b.cos() * x1 + b.sin() * z1,
            y1,
            -b.sin() * x1 + b.cos() * z1,
        )
    };
    // The same rotation acting on the p block of a density matrix.
    let rotate_density = |d: &[[f64; 4]; 4]| -> [[f64; 4]; 4] {
        let mut m = [[0.0_f64; 4]; 4];
        let e = [
            rot(Vec3::new(1.0, 0.0, 0.0)),
            rot(Vec3::new(0.0, 1.0, 0.0)),
            rot(Vec3::new(0.0, 0.0, 1.0)),
        ];
        let rm = |i: usize, j: usize| [e[j].x, e[j].y, e[j].z][i];
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = 0.0;
                for k in 0..4 {
                    for l in 0..4 {
                        let ri = if i == 0 {
                            if k == 0 {
                                1.0
                            } else {
                                0.0
                            }
                        } else if k == 0 {
                            0.0
                        } else {
                            rm(i - 1, k - 1)
                        };
                        let rj = if j == 0 {
                            if l == 0 {
                                1.0
                            } else {
                                0.0
                            }
                        } else if l == 0 {
                            0.0
                        } else {
                            rm(j - 1, l - 1)
                        };
                        acc += ri * rj * d[k][l];
                    }
                }
                m[i][j] = acc;
            }
        }
        m
    };

    // Two arbitrary symmetric "densities", one per centre.
    let mut d1 = [[0.0_f64; 4]; 4];
    let mut d2 = [[0.0_f64; 4]; 4];
    let mut seed = 1.0;
    for i in 0..4 {
        for j in 0..=i {
            seed = (seed * 7.13 + 0.29) % 1.0;
            d1[i][j] = seed - 0.5;
            d1[j][i] = d1[i][j];
            seed = (seed * 5.71 + 0.37) % 1.0;
            d2[i][j] = seed - 0.5;
            d2[j][i] = d2[i][j];
        }
    }
    let contract =
        |te: &am1_rs::integrals::PairTwoElec, a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]| -> f64 {
            let mut acc = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    for k in 0..4 {
                        for l in 0..4 {
                            acc += a[i][j] * b[k][l] * te.two_e(i, j, k, l);
                        }
                    }
                }
            }
            acc
        };

    let plain = contract(&pair_two_electron(o, c, dir, r), &d1, &d2);
    let turned = contract(
        &pair_two_electron(o, c, rot(dir), r),
        &rotate_density(&d1),
        &rotate_density(&d2),
    );
    let rel = (plain - turned).abs() / plain.abs().max(1.0e-12);
    eprintln!("    contracted energy: {plain:.10} vs rotated {turned:.10}  ({rel:.3e} relative)");
    assert!(
        rel < 1.0e-12,
        "the pair block is not rotation covariant: {rel:.3e} relative"
    );
}

// ---------------------------------------------------------------------------- overlap

/// The overlap has the inversion parity its orbitals imply, and is unchanged by relabelling.
///
/// Two distinct statements, and the first draft of this test confused them.
///
/// **Parity.** With `φ_μ` at the origin and `φ_ν` at `d`, substituting `r → −r` in
/// `S_{μν}(d) = ∫ φ_μ(r) φ_ν(r − d)` gives
///
/// ```text
///   S_{μν}(−d) = (−1)^{l_μ + l_ν} S_{μν}(d)
/// ```
///
/// so inverting the *displacement* flips the sign of every element carrying an odd number of `p`
/// indices. That sign lives in the local-frame construction, and a closed-shell energy cancels
/// it, so nothing else here would notice it.
///
/// **Relabelling.** Exchanging the two *arguments* flips nothing: the geometry is unchanged and
/// only the index order swaps, so `S(ej, ei)ᵀ = S(ei, ej)` exactly. Applying a parity factor
/// there — as the first draft did — fails by exactly `2×` the value, which is how the confusion
/// announced itself.
#[test]
fn the_overlap_has_the_parity_its_orbitals_imply() {
    let p = params();
    let (o, c) = (p.element(8).unwrap(), p.element(6).unwrap());
    let origin = Vec3::zero();
    let d = Vec3::new(1.3, -0.9, 0.7) * ANG;

    let plus = diatom_overlap(o, origin, c, d).unwrap();
    let minus = diatom_overlap(o, origin, c, d * -1.0).unwrap();
    let swapped = diatom_overlap(c, d, o, origin).unwrap();

    let is_p = |k: usize| k != S;
    let (mut worst_parity, mut worst_swap, mut scale) = (0.0_f64, 0.0_f64, 0.0_f64);
    for a in 0..4 {
        for b in 0..4 {
            let parity = if is_p(a) ^ is_p(b) { -1.0 } else { 1.0 };
            scale = scale.max(plus[a][b].abs());
            worst_parity = worst_parity.max((minus[a][b] - parity * plus[a][b]).abs());
            worst_swap = worst_swap.max((swapped[b][a] - plus[a][b]).abs());
        }
    }
    eprintln!(
        "    max |S(-d) - parity*S(d)| = {worst_parity:.3e};  relabelling {worst_swap:.3e};  \
         of {scale:.3e}"
    );
    assert!(
        scale > 1.0e-3,
        "the overlap is ~zero; the test proves nothing"
    );
    assert!(
        worst_parity < 1.0e-12,
        "inversion parity broken by {worst_parity:.3e}"
    );
    assert!(
        worst_swap < 1.0e-12,
        "exchanging the two atoms changed the overlap by {worst_swap:.3e}"
    );
}

/// Overlap of an orbital with itself at zero separation is 1, and it decays monotonically.
///
/// Slater orbitals are normalized, so `S(0) = I`; and two exponentially decaying functions
/// overlap less the further apart they are. A normalization slip shows up here as a value that
/// is not 1, and nowhere else — NDDO never forms `S` for the energy, it only uses it in the
/// resonance term, where a constant factor is absorbed by `β`.
#[test]
fn the_overlap_is_normalized_and_decays() {
    let p = params();
    let c = p.element(6).unwrap();
    let at_zero = diatom_overlap(c, Vec3::zero(), c, Vec3::new(1.0e-10, 0.0, 0.0)).unwrap();
    for k in 0..4 {
        assert!(
            (at_zero[k][k] - 1.0).abs() < 1.0e-9,
            "orbital {k} is not normalized: S = {}",
            at_zero[k][k]
        );
    }

    let mut previous = f64::INFINITY;
    eprintln!("      R (Bohr)   |S(s,s)|");
    for r in [1.0_f64, 2.0, 3.0, 4.0, 6.0, 8.0] {
        let s = diatom_overlap(c, Vec3::zero(), c, Vec3::new(r, 0.0, 0.0)).unwrap()[S][S].abs();
        eprintln!("      {r:8.1}  {s:12.3e}");
        assert!(
            s < previous,
            "the s-s overlap grew from {previous:.3e} to {s:.3e} between R and R'"
        );
        previous = s;
    }
}

// -------------------------------------------------------------- converged-wavefunction algebra

/// The converged density is idempotent, and its trace is the electron count.
///
/// For a closed shell in an orthonormal basis `P = 2 C_occ C_occᵀ`, so `P² = 2P` and
/// `Tr P = 2 n_occ`. These are properties of a *single-determinant* wavefunction, not of this
/// model — they hold for any converged RHF — and they are what a mis-set occupation, a dropped
/// factor of two, or a non-orthonormal eigenvector set would break.
///
/// NDDO's assumed-orthonormal AO basis is what makes them this simple: with a real `S` the
/// conditions would be `PSP = 2P` and `Tr PS = N`.
#[test]
fn the_converged_density_is_an_idempotent_projector() {
    let p = params();
    for (name, mol, charge) in [
        ("water", water(), 0.0),
        ("methane", methane(), 0.0),
        ("formaldehyde", formaldehyde(), 0.0),
    ] {
        let opts = Am1Options {
            charge,
            e_tol: 1.0e-12,
            p_tol: 1.0e-11,
            ..Am1Options::default()
        };
        let scf = run_am1(&mol, &p, &opts).unwrap();
        assert!(scf.converged, "{name} did not converge");
        let d = &scf.density;
        let n = d.rows;

        // P² = 2P.
        let pp = d.matmul(d);
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                scale = scale.max(d[(i, j)].abs());
                worst = worst.max((pp[(i, j)] - 2.0 * d[(i, j)]).abs());
            }
        }
        // Tr P = number of electrons.
        let trace: f64 = (0..n).map(|i| d[(i, i)]).sum();
        let electrons = 2.0 * scf.n_occ as f64;
        eprintln!(
            "    {name:14} max |P^2 - 2P| = {worst:.3e} of {scale:.3e};  Tr P = {trace:.10} \
             (expected {electrons:.1})"
        );
        assert!(worst < 1.0e-9, "{name}: P is not idempotent ({worst:.3e})");
        assert!(
            (trace - electrons).abs() < 1.0e-9,
            "{name}: Tr P = {trace}, expected {electrons}"
        );
    }
}

/// At the SCF solution the Fock matrix and the density commute.
///
/// `[F, P] = 0` **is** the self-consistency condition — it says the occupied space is invariant
/// under `F`, which is what makes the energy stationary. The SCF drives this residual to zero, so
/// it is not an independent check of convergence; what it *is* independent of is the energy
/// expression. A Fock matrix assembled with a term the density does not see, or a density built
/// from the wrong eigenvectors, leaves this residual finite while the energy still looks settled.
#[test]
fn the_fock_matrix_commutes_with_the_converged_density() {
    let p = params();
    let mol = formaldehyde();
    let opts = Am1Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        ..Am1Options::default()
    };
    let scf = run_am1(&mol, &p, &opts).unwrap();
    assert!(scf.converged);

    let basis = am1_rs::basis::Basis::build(&mol, &p).unwrap();
    let neighbors = am1_rs::neighbors::NeighborList::build(&mol, opts.realspace_cutoff);
    let core = am1_rs::hamiltonian::build_core_with_neighbors(
        &mol,
        &basis,
        &p,
        &neighbors,
        opts.core_build(),
    )
    .unwrap();
    let f = am1_rs::fock::build_fock(&mol, &basis, &p, &core, &scf.density).unwrap();

    let n = f.rows;
    let fp = f.matmul(&scf.density);
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    let mut asym = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            scale = scale.max(fp[(i, j)].abs());
            // [F,P]_ij = (FP)_ij − (PF)_ij, and PF = (FP)ᵀ because F and P are symmetric.
            worst = worst.max((fp[(i, j)] - fp[(j, i)]).abs());
            asym = asym.max((f[(i, j)] - f[(j, i)]).abs());
        }
    }
    eprintln!("    max |[F,P]| = {worst:.3e} of {scale:.3e};  Fock asymmetry {asym:.3e}");
    assert!(
        asym < 1.0e-12,
        "the Fock matrix is not symmetric: {asym:.3e}"
    );
    assert!(
        worst < 1.0e-6,
        "[F,P] does not vanish at the SCF solution: {worst:.3e}"
    );
}

/// The reported electronic energy is the one the Fock and density actually imply.
///
/// `E_elec = ½ Tr[P(H_core + F)]` is the Hartree–Fock energy expression, and recomputing it from
/// the pieces catches a term that was added to the Fock but not to the energy, or vice versa —
/// the failure mode that makes an SCF converge to a stationary point of something other than the
/// energy it reports. `long_range_energy_term` exists precisely because one such term needs an
/// explicit correction, and this is what would catch its omission.
#[test]
fn the_electronic_energy_is_the_trace_it_claims_to_be() {
    let p = params();
    for (name, mol) in [("water", water()), ("formaldehyde", formaldehyde())] {
        let opts = Am1Options {
            e_tol: 1.0e-12,
            p_tol: 1.0e-11,
            ..Am1Options::default()
        };
        let scf = run_am1(&mol, &p, &opts).unwrap();
        let basis = am1_rs::basis::Basis::build(&mol, &p).unwrap();
        let neighbors = am1_rs::neighbors::NeighborList::build(&mol, opts.realspace_cutoff);
        let core = am1_rs::hamiltonian::build_core_with_neighbors(
            &mol,
            &basis,
            &p,
            &neighbors,
            opts.core_build(),
        )
        .unwrap();
        let f = am1_rs::fock::build_fock(&mol, &basis, &p, &core, &scf.density).unwrap();
        let recomputed =
            0.5 * (scf.density.frobenius_dot(&core.h_core) + scf.density.frobenius_dot(&f));
        let rel = (recomputed - scf.electronic_ev).abs() / scf.electronic_ev.abs();
        eprintln!(
            "    {name:14} reported {:.9} eV, recomputed {recomputed:.9} eV  ({rel:.3e})",
            scf.electronic_ev
        );
        assert!(
            rel < 1.0e-11,
            "{name}: the reported electronic energy is not ½Tr[P(H+F)] ({rel:.3e})"
        );
    }
}

/// Koopmans: the reported ionization potential is the negative of the highest occupied level.
///
/// A one-line identity, and the point is *which* level: off-by-one in the occupied count would
/// report the HOMO−1 or the LUMO, both of which are plausible numbers. `n_occ` also has to agree
/// with the electron count, which is where a charge or multiplicity mishandling would show.
#[test]
fn koopmans_uses_the_level_it_should() {
    let p = params();
    let mol = water();
    let scf = run_am1(&mol, &p, &Am1Options::default()).unwrap();
    let eps = &scf.mo_energies;
    let homo = eps[scf.n_occ - 1];
    let lumo = eps[scf.n_occ];
    assert!(
        (scf.homo_ev.unwrap() - homo).abs() < 1.0e-12,
        "homo_ev is not eigenvalue {}",
        scf.n_occ - 1
    );
    assert!((scf.lumo_ev.unwrap() - lumo).abs() < 1.0e-12);
    assert!(homo < lumo, "the HOMO is above the LUMO");
    // Water: 8 valence electrons, so four filled levels.
    assert_eq!(scf.n_occ, 4, "water should have four occupied levels");
    eprintln!(
        "    HOMO {homo:.6} eV, LUMO {lumo:.6} eV, gap {:.6} eV",
        lumo - homo
    );
}

/// The energy does not depend on where the molecule sits or how it is turned.
///
/// Translational and rotational invariance of a total energy is not a modelling choice — it
/// follows from the Hamiltonian depending only on interatomic vectors. It is worth checking at
/// the *energy* level even though `tests/axis_alignment.rs` checks the Hessian, because the two
/// fail differently: the Hessian's frame problem was a vanishing derivative, while an energy that
/// moved under rotation would mean a term evaluated in the wrong frame.
#[test]
fn the_energy_is_invariant_under_rigid_motion() {
    let p = params();
    let mol = formaldehyde();
    let opts = Am1Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        ..Am1Options::default()
    };
    let reference = run_am1(&mol, &p, &opts).unwrap().total_ev;

    let shifted = Molecule::new(
        mol.atoms
            .iter()
            .map(|a| Atom {
                z: a.z,
                position: a.position + Vec3::new(7.3, -2.9, 4.1),
            })
            .collect(),
    );
    let e_shift = run_am1(&shifted, &p, &opts).unwrap().total_ev;

    // A rotation that is not about a symmetry axis, so nothing cancels by accident.
    let (ca, sa) = (0.6, 0.8);
    let turned = Molecule::new(
        mol.atoms
            .iter()
            .map(|a| {
                let v = a.position;
                Atom {
                    z: a.z,
                    position: Vec3::new(ca * v.x - sa * v.y, sa * v.x + ca * v.y, v.z),
                }
            })
            .collect(),
    );
    let e_rot = run_am1(&turned, &p, &opts).unwrap().total_ev;

    eprintln!(
        "    translation {:.3e} eV, rotation {:.3e} eV",
        (e_shift - reference).abs(),
        (e_rot - reference).abs()
    );
    assert!(
        (e_shift - reference).abs() < 1.0e-9,
        "the energy moved by {:.3e} eV under a translation",
        (e_shift - reference).abs()
    );
    assert!(
        (e_rot - reference).abs() < 1.0e-9,
        "the energy moved by {:.3e} eV under a rotation",
        (e_rot - reference).abs()
    );
}

/// `pack` is the packing the integral store assumes: symmetric, dense, and covering `0..10`.
///
/// A small function, and the one every two-electron lookup goes through. If it ever stopped being
/// a bijection onto `0..10` the integrals would silently alias onto each other — which is a
/// wrong answer, not a crash.
#[test]
fn the_orbital_pair_packing_is_a_bijection() {
    let mut seen = vec![0usize; 10];
    for a in 0..4 {
        for b in 0..4 {
            let k = pack(a, b);
            assert_eq!(k, pack(b, a), "pack is not symmetric at ({a},{b})");
            assert!(
                k < 10,
                "pack({a},{b}) = {k} is outside the 10-element block"
            );
            if a >= b {
                seen[k] += 1;
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "pack does not cover 0..10 exactly once: {seen:?}"
    );
}

// ------------------------------------------------------------------------------ geometries

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.9584 * ANG, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.2400 * ANG, 0.9278 * ANG, 0.0),
        },
    ])
}

fn methane() -> Molecule {
    let b = 1.087 * ANG / 3.0_f64.sqrt();
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(b, b, b),
        },
        Atom {
            z: 1,
            position: Vec3::new(b, -b, -b),
        },
        Atom {
            z: 1,
            position: Vec3::new(-b, b, -b),
        },
        Atom {
            z: 1,
            position: Vec3::new(-b, -b, b),
        },
    ])
}

/// Carries a heteroatom double bond, so the `p` channels and the hybridization moment are
/// genuinely exercised rather than nearly vanishing by symmetry.
fn formaldehyde() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 8,
            position: Vec3::new(1.208 * ANG, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.5650 * ANG, 0.9376 * ANG, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.5650 * ANG, -0.9376 * ANG, 0.0),
        },
    ])
}

/// Silence the unused-constant warnings for the orbital labels this file documents but does not
/// use in every test.
#[allow(dead_code)]
const _LABELS: [usize; 3] = [PY, PZ, S];

/// `Matrix` is used through `frobenius_dot` and `matmul` above; naming it keeps the import
/// meaningful to a reader scanning the header.
#[allow(dead_code)]
fn _matrix_is_used(_: &Matrix) {}
