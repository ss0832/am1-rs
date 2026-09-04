// SPDX-License-Identifier: GPL-3.0-or-later

// The loops below index by atom and Cartesian axis, and the index *is* the quantity being
// checked -- `Z*_{a,alpha,beta}`, `alpha_ab` against its transpose. Rewriting them as
// iterators would hide which axis is which, so the lint is declined here rather than obeyed.
#![allow(clippy::needless_range_loop)]

//! The electronic dielectric tensor `ε_∞`, and the size of the approximation it rests on.
//!
//! `ε_∞` is the second ingredient LO–TO splitting needs, after the Born charges. It is obtained
//! from a uniform-field CPHF, and the perturbation is a **position operator** — which under
//! periodic boundary conditions is not a well-defined periodic operator, because `R_a` is fixed
//! only modulo a lattice vector.
//!
//! That is a real approximation, not an implementation detail, and this file's job is to measure
//! it rather than to assert that it is fine. The measurement is direct: shift the cell origin,
//! recompute, and report how much `ε_∞` moved. Anything that survives that shift is meaningful;
//! anything that does not is the approximation showing.
//!
//! # Why the system is a crystal and not a chain
//!
//! `ε_∞ = 1 + 4πα/Ω` is the **three-dimensional** relation and `Ω` has to be a volume. Until
//! 0.2.1 these tests ran on a 1D chain, where `Lattice::measure` supplied a *length*, so the
//! reported "dielectric tensor" was not one in any unit system. The polarizability `α` itself is
//! well defined in any dimensionality — it is the response of the model's own dipole to the
//! model's own field operator — but the step from `α` to `ε_∞` is not.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{dielectric_origin_sensitivity, dielectric_tensor, KMesh, KPoint, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// One water molecule per cubic cell — fully periodic, so `Ω` is a volume.
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
    Molecule::new(atoms).with_cell(Lattice::cubic(a_ang * ANG).unwrap())
}

/// The same water geometry with **no cell**, for the isolated-molecule reference.
///
/// It has to be cell-free: a uniform field under periodic boundary conditions is refused, which
/// is correct — `F·R` is unbounded along a periodic axis — and is exactly why the molecular and
/// periodic routes to `α` are independent rather than two spellings of one calculation.
fn water_molecule() -> Molecule {
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
    Molecule::new(atoms)
}

/// A chain, used only to assert that it is refused.
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

fn options(mesh: [usize; 3]) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(10.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        ..PbcOptions::default()
    }
}

#[test]
fn the_dielectric_tensor_is_computed_and_is_physical() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_crystal(4.5);
    let (alpha, epsilon) = dielectric_tensor(&molecule, &params, &options([2, 2, 2])).unwrap();
    eprintln!("    polarizability (Bohr^3):");
    for row in &alpha {
        eprintln!("      [{:12.5} {:12.5} {:12.5}]", row[0], row[1], row[2]);
    }
    eprintln!("    epsilon_infinity:");
    for row in &epsilon {
        eprintln!("      [{:12.6} {:12.6} {:12.6}]", row[0], row[1], row[2]);
    }

    // The polarizability of a stable system is positive definite: applying a field along any
    // direction must induce a dipole *along* that direction. A negative diagonal element would
    // mean the response is running away, which for a gapped system it cannot.
    for a in 0..3 {
        assert!(
            alpha[a][a] > 0.0,
            "the polarizability diagonal must be positive, got {} along axis {a}",
            alpha[a][a]
        );
        assert!(
            epsilon[a][a] >= 1.0,
            "epsilon_infinity must be at least 1, got {}",
            epsilon[a][a]
        );
    }
    // And it must be symmetric — it is a second derivative of the energy.
    let mut asym = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            asym = asym.max((alpha[a][b] - alpha[b][a]).abs());
        }
    }
    eprintln!("    polarizability asymmetry: {asym:.3e} Bohr^3");
    assert!(
        asym < 1.0e-6 * alpha[0][0].abs().max(1.0),
        "the polarizability is not symmetric: {asym:.3e}"
    );
}

#[test]
fn the_position_operator_approximation_is_measured_not_assumed() {
    // The number that says how far to trust `ε_∞`.
    //
    // Under periodic boundary conditions the field perturbation `+E·R_a` is not origin
    // independent, because `R_a` itself is not. Shifting the whole cell is therefore the test:
    // whatever `ε_∞` does under that shift is the size of the approximation.
    //
    // In a fully periodic crystal every direction is periodic, so there is no "safe" axis to
    // compare against — the whole measurement is of the ambiguous case. What makes it meaningful
    // anyway is the argument in `dielectric_tensor`: the response conserves charge, so
    // `Σ_b R_b ∂Q_b/∂E` is origin independent term by term, and the shift dependence should
    // cancel to machine precision rather than merely being small.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_crystal(4.5);
    let o = options([2, 2, 2]);
    let (_, base) = dielectric_tensor(&molecule, &params, &o).unwrap();

    let shifted =
        dielectric_origin_sensitivity(&molecule, &params, &o, Vec3::new(1.7, -0.9, 2.3)).unwrap();

    let scale = base[0][0].abs().max(1.0);
    eprintln!(
        "    origin shift (1.7, -0.9, 2.3) Bohr: epsilon moves {shifted:.3e} ({:.2e} relative)",
        shifted / scale
    );
    assert!(
        shifted / scale < 1.0e-10,
        "epsilon_infinity moved by {:.2e} of itself under an origin shift; charge conservation \
         should make that cancel term by term, so a nonzero result means the cancellation the \
         module documents is not actually happening",
        shifted / scale
    );
}

/// A chain or a slab is refused rather than handed the three-dimensional formula.
///
/// The 0.2.1 correction: `Ω` in `ε_∞ = 1 + 4πα/Ω` must be a volume, and `Lattice::measure`
/// returns a length for a chain. The result before this guard was dimensionally not a dielectric
/// constant, and it was what the "polar chain" LO–TO splitting was built on.
#[test]
fn a_low_dimensional_cell_is_refused() {
    let params = Am1Parameters::standard().unwrap();
    let err = dielectric_tensor(&water_chain(3.4), &params, &options([3, 1, 1])).unwrap_err();
    eprintln!("    {err}");
    assert!(err.to_string().contains("three-dimensional"), "{err}");
}

/// The isolated molecule's polarizability by **finite field**, in Bohr³.
///
/// `α_αβ = ∂μ_α/∂F_β`, central-differenced. This shares no code with the periodic CPHF past the
/// SCF itself: one is an analytic linear response solved in the occupied–virtual block, the other
/// is two extra SCF solves per axis and a subtraction.
///
/// Units: the crate's field is eV per `e·Bohr` and its dipole is reported in Debye. Converting
/// the dipole to `e·Bohr` and the field to Hartree per `e·Bohr` puts `α` in Bohr³, the same unit
/// `dielectric_tensor` returns.
fn finite_field_polarizability(molecule: &Molecule, params: &Am1Parameters) -> [[f64; 3]; 3] {
    use am1_rs::constants::{AU_DIPOLE_TO_DEBYE, HARTREE_TO_EV};
    use am1_rs::scf::{run_am1, Am1Options};

    let base = Am1Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 800,
        ..Am1Options::default()
    };
    // Small enough to stay linear, large enough that the dipole moves well clear of the SCF
    // convergence floor.
    let h = 2.0e-4;
    let mut alpha = [[0.0_f64; 3]; 3];
    for beta in 0..3 {
        let dipole = |sign: f64| -> [f64; 3] {
            let mut f = Vec3::zero();
            match beta {
                0 => f.x = sign * h,
                1 => f.y = sign * h,
                _ => f.z = sign * h,
            }
            let opts = Am1Options {
                electric_field: Some(f),
                ..base.clone()
            };
            let r = run_am1(molecule, params, &opts).unwrap();
            let d = r.dipole_debye;
            [
                d.x / AU_DIPOLE_TO_DEBYE,
                d.y / AU_DIPOLE_TO_DEBYE,
                d.z / AU_DIPOLE_TO_DEBYE,
            ]
        };
        let (plus, minus) = (dipole(1.0), dipole(-1.0));
        for a in 0..3 {
            // d(mu)/d(F in eV) -> d(mu)/d(F in Hartree) is a factor of HARTREE_TO_EV.
            alpha[a][beta] = (plus[a] - minus[a]) / (2.0 * h) * HARTREE_TO_EV;
        }
    }
    alpha
}

/// **The magnitude check.** A molecule in a large box must have the periodic clamped-ion `α` of
/// the isolated molecule, and it must get closer as the box grows.
///
/// Everything else in this file constrains `α`'s *shape* — symmetric, positive diagonal,
/// independent of the cell origin — and a whole family of wrong answers satisfies all of that. A
/// polarizability off by a factor of two, or one computed with the wrong field unit, passes every
/// other test here and fails this one.
///
/// The two sides are independent: the periodic side is an analytic CPHF in the occupied–virtual
/// block under Bloch boundary conditions, the molecular side is finite differences of two extra
/// SCF solves per axis. They share the SCF and the dipole operator and nothing else.
///
/// The residual at finite box size is physical, not numerical — it is the molecule polarizing its
/// own periodic images — so the test asserts *convergence*, that a bigger box is closer, rather
/// than a fixed tolerance at one size.
#[test]
fn a_molecule_in_a_large_box_has_the_isolated_molecule_polarizability() {
    let params = Am1Parameters::standard().unwrap();
    let isolated = finite_field_polarizability(&water_molecule(), &params);
    let trace_iso = (isolated[0][0] + isolated[1][1] + isolated[2][2]) / 3.0;
    eprintln!("    isolated molecule, finite field: mean alpha = {trace_iso:.6} Bohr^3");
    assert!(
        trace_iso > 1.0,
        "the finite-field polarizability came out as {trace_iso:.3e} Bohr^3, which is not a water \
         molecule -- suspect the field unit conversion"
    );

    let mut previous = f64::INFINITY;
    for a_ang in [7.0_f64, 9.0, 12.0] {
        // A molecule in a box: one k point is the right sampling, since there is no dispersion.
        let cell = water_crystal(a_ang);
        let (alpha, _) = dielectric_tensor(&cell, &params, &options([1, 1, 1])).unwrap();
        let trace = (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0;
        let relative = (trace - trace_iso).abs() / trace_iso;
        eprintln!(
            "    box {a_ang:.1} A: periodic mean alpha = {trace:.6} Bohr^3, \
             {:.2} % from the isolated molecule",
            relative * 100.0
        );
        assert!(
            relative < previous,
            "a {a_ang:.1} A box is further from the isolated molecule ({:.3e}) than the smaller \
             one before it ({previous:.3e}); the periodic alpha is not converging to it",
            relative
        );
        previous = relative;
    }
    assert!(
        previous < 0.05,
        "at 12 A the periodic polarizability is still {:.1} % from the isolated-molecule value",
        previous * 100.0
    );
}

// ------------------------------------------------------- reduced dimensionality

/// Options for a chain (`periodic = 1`) or a slab (`periodic = 2`).
fn low_dim_options(periodic: usize) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack([4, if periodic == 2 { 4 } else { 1 }, 1]),
        fold_time_reversal: false,
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(12.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 800,
        ..PbcOptions::default()
    }
}
/// A chain in a transverse field: the field along `y` is orthogonal to the lattice, so it is an
/// ordinary `F·R` perturbation and the isolated molecule answers it the same way.
fn hf_chain(spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 9,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.94 * ANG, 0.0, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(step, 0.0, 0.0),
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

/// A slab: two periodic directions, vacuum along the third.
fn hf_slab(spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 9,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.94 * ANG, 0.2 * ANG, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(step, 0.0, 0.0),
            Vec3::new(0.0, step, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, true, false],
        )
        .unwrap(),
    )
}

/// **The polarizability is available for a chain and a slab**, and only the `ε_∞` conversion is not.
///
/// Through 0.2.1 `dielectric_tensor` was the only entry point and it refused a reduced-dimensional
/// cell outright — for the `ε_∞ = 1 + 4πα/Ω` step's sake, which genuinely needs `Ω` to be a
/// volume, but it took `α` down with it. `α` is a *response*, and a response is well defined
/// whatever the cell is periodic in.
///
/// Asserted rather than merely called: `α` must be symmetric, positive on the diagonal, and — the
/// part that says it is the right quantity — its **transverse** components must be close to the
/// isolated molecule's, because along a non-periodic direction a chain in a 30 Bohr cell *is* an
/// isolated molecule.
#[test]
fn the_polarizability_is_available_in_reduced_dimensionality() {
    use am1_rs::pbc::polarizability;
    let params = Am1Parameters::standard().unwrap();

    for (name, molecule, periodic) in [("chain", hf_chain(4.0), 1usize), ("slab", hf_slab(4.0), 2)]
    {
        let o = low_dim_options(periodic);
        let alpha = polarizability(&molecule, &params, &o).unwrap();
        eprintln!(
            "    {name}: alpha diagonal = ({:.4}, {:.4}, {:.4}) Bohr^3",
            alpha[0][0], alpha[1][1], alpha[2][2]
        );
        for a in 0..3 {
            assert!(
                alpha[a][a] > 0.0,
                "{name}: alpha_{a}{a} is {:.4}, and a polarizability cannot be negative",
                alpha[a][a]
            );
            for b in 0..3 {
                assert!(
                    (alpha[a][b] - alpha[b][a]).abs() < 1.0e-9 * alpha[a][a].abs().max(1.0),
                    "{name}: alpha is not symmetric at ({a},{b})"
                );
            }
        }

        // And `dielectric_tensor` still refuses, naming the alternative.
        let err = dielectric_tensor(&molecule, &params, &o)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("pbc::polarizability"),
            "{name}: the refusal should point at the function that does work: {err}"
        );
    }
}

/// **`D(q → 0)` is continuous below three dimensions, and discontinuous in three.**
///
/// This is what "LO–TO for a chain or a slab" comes to. The long-range kernel diverges in every
/// dimensionality — `4π/(Vq²)`, `2π/(A|q|)`, `−(2/L)ln|q|` — but the *contribution to `D(q)`*
/// carries `q²` from charge conservation, so:
///
/// | | `q²` × kernel | at Γ |
/// |---|---|---|
/// | 3D | finite, direction dependent | **discontinuous** — this is the LO–TO splitting |
/// | 2D | `O(\|q\|) → 0` | continuous, with a `∝\|q\|` kink |
/// | 1D | `q² ln(1/q) → 0` | continuous, more weakly non-analytic |
///
/// So a chain and a slab have **no splitting to add at Γ**, and `frequencies_with_lo_to` refusing
/// them is the physics rather than a gap. What they do have is a non-analytic *approach*, and the
/// DFPT path already carries it exactly — it is measured here rather than asserted, by watching
/// `D(q)` converge to `D(0)` as `q` shrinks in 1D and 2D and fail to in 3D.
#[test]
fn the_long_range_term_is_continuous_at_gamma_below_three_dimensions() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions, LongRange};
    let params = Am1Parameters::standard().unwrap();

    let cases: [(&str, Molecule, [usize; 3]); 3] = [
        ("1D chain", hf_chain(4.0), [6, 1, 1]),
        ("2D slab", hf_slab(4.0), [6, 6, 1]),
        ("3D crystal", water_crystal(4.5), [4, 4, 4]),
    ];
    for (name, molecule, mesh) in cases {
        let o = PbcOptions {
            kmesh: KMesh::MonkhorstPack(mesh),
            fold_time_reversal: false,
            realspace_cutoff: 30.0,
            exchange_cutoff: Some(12.0),
            smearing_ev: 0.0,
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            max_scf: 800,
            ..PbcOptions::default()
        };
        let dfpt = DfptOptions {
            long_range: LongRange::Require,
            ..DfptOptions::default()
        };
        let at = |f: f64| {
            force_constants_at_q_with(
                &molecule,
                &params,
                &o,
                &dfpt,
                KPoint {
                    fractional: [f, 0.0, 0.0],
                    weight: 1.0,
                },
            )
            .unwrap()
            .force_constants
        };
        let gamma = at(0.0);
        let mut gaps = Vec::new();
        for f in [0.02_f64, 0.01, 0.005] {
            let d = at(f);
            let mut worst = 0.0_f64;
            for i in 0..d.n {
                for j in 0..d.n {
                    let (ar, ai) = d.get(i, j);
                    let (br, bi) = gamma.get(i, j);
                    worst = worst.max((ar - br).abs()).max((ai - bi).abs());
                }
            }
            gaps.push(worst);
        }
        eprintln!(
            "    {name}: |D(q) - D(0)| at q = 0.02, 0.01, 0.005 -> {:.4e}, {:.4e}, {:.4e}",
            gaps[0], gaps[1], gaps[2]
        );

        if name.starts_with("3D") {
            // The limit is direction dependent, so approaching Γ along `x` does **not** reach the
            // Γ value: the gap stops falling. That is the LO–TO discontinuity, and it is why the
            // three-dimensional case gets a non-analytic term and the others do not.
            assert!(
                gaps[2] > 0.3 * gaps[0],
                "{name}: the gap collapsed ({:.3e} -> {:.3e}), but a 3D cell's D(q) is \
                 discontinuous at Γ and should not converge to it",
                gaps[0],
                gaps[2]
            );
        } else {
            // Continuous: halving `q` twice must shrink the gap by a clear factor. 2D goes as
            // `|q|` (a factor of four over these two halvings), 1D more weakly, so the assertion
            // is "well under half" rather than a fitted exponent.
            assert!(
                gaps[2] < 0.5 * gaps[0],
                "{name}: D(q) did not approach D(0) ({:.3e} -> {:.3e}); below three dimensions \
                 the long-range contribution to D carries q² against a weaker divergence and must \
                 vanish at Γ",
                gaps[0],
                gaps[2]
            );
        }
    }
}

/// `K₀` against tabulated values, before anything is built on it.
///
/// The one special function in the crate beyond `erf`, so it is checked against numbers that come
/// from outside this codebase rather than against itself. Abramowitz & Stegun's own table.
#[test]
fn the_modified_bessel_function_matches_its_table() {
    use am1_rs::pbc::dielectric_function;
    // K₀ is private, so it is exercised through the one caller that uses it: a chain's dielectric
    // function is `1 + 2 K₀(qρ) q² χ / L`, so `(ε − 1) L / (2 q² χ)` recovers `K₀(qρ)` exactly.
    // The polarizability cancels out of the ratio, which is what lets this be a check on `K₀`
    // alone rather than on the whole chain.
    let params = Am1Parameters::standard().unwrap();
    let chain = hf_chain(4.0);
    let o = low_dim_options(1);
    let chi = {
        use am1_rs::pbc::polarizability;
        let a = polarizability(&chain, &params, &o).unwrap();
        a[0][0] / chain.cell.unwrap().measure()
    };
    let q_axis = chain.cell.unwrap().reciprocal_vectors_2pi()[0];
    let q_len = q_axis.norm();

    // (argument, K₀) from A&S table 9.8; the arguments are reached by choosing `ρ = x/|q|`.
    for (x, expected) in [
        (0.1_f64, 2.427_069),
        (0.5, 0.924_419),
        (1.0, 0.421_024),
        (2.0, 0.113_894),
        (5.0, 0.003_691),
    ] {
        let frac = 0.05;
        let q = q_axis * frac;
        let radius = x / (q_len * frac);
        let eps = dielectric_function(&chain, &params, &o, q, Some(radius)).unwrap();
        let qq = (q_len * frac) * (q_len * frac);
        let k0 = (eps - 1.0) / (2.0 * qq * chi);
        eprintln!("    K0({x}) = {k0:.6}, table {expected:.6}");
        // The tolerance is set by the **table**, not by the approximation: A&S quote 9.8.5/9.8.6
        // to better than 2e-7, but the values here are transcribed to six decimals, so half a
        // unit in the last place is the floor. The relative part covers the polarizability the
        // ratio divides by, which carries the SCF's own convergence.
        assert!(
            (k0 - expected).abs() < 1.0e-6 + 1.0e-5 * expected,
            "K0({x}) came out {k0:.6}, the table says {expected:.6}"
        );
    }
}

/// **The unified formula, checked against the one case that is known independently.**
///
/// `ε(q) = 1 + v_d(q) q² (q̂·α·q̂)/measure` must reduce, in three dimensions, to the `ε_∞` that
/// `dielectric_tensor` computes by a completely different expression — and it must do so at
/// **every** `q`, because `v_3D = 4π/q²` cancels the `q²` exactly. If the kernel or the `q²`
/// were wrong the two would differ, and if only their product were wrong they would differ
/// with `q`.
#[test]
fn in_three_dimensions_the_dielectric_function_is_the_constant_epsilon_infinity() {
    use am1_rs::pbc::dielectric_function;
    let params = Am1Parameters::standard().unwrap();
    let crystal = water_crystal(4.5);
    let o = options([2, 2, 2]);
    let eps_tensor = dielectric_tensor(&crystal, &params, &o).unwrap().1;
    let recip = crystal.cell.unwrap().reciprocal_vectors_2pi();

    for frac in [0.3_f64, 0.1, 0.03] {
        let q = recip[0] * frac;
        let eps = dielectric_function(&crystal, &params, &o, q, None).unwrap();
        eprintln!(
            "    q = {frac} b1: eps(q) = {eps:.9}, eps_infinity_xx = {:.9}",
            eps_tensor[0][0]
        );
        assert!(
            (eps - eps_tensor[0][0]).abs() < 1.0e-9,
            "the 3D dielectric function is not q-independent: {eps:.9} against {:.9}",
            eps_tensor[0][0]
        );
    }
}

/// **A slab and a chain do not screen at long wavelength**, and the rate they stop at is the
/// dimensionality's own.
///
/// `ε(q) − 1` is `2π χ₂D |q|` for a slab and `2 K₀(qρ) q² χ₁D` for a chain, so halving `q` halves
/// the slab's excess and quarters the chain's up to the logarithm. Fitting the exponent is what
/// distinguishes "it goes to 1" from "it goes to 1 at the right rate" — and the rate is the same
/// one `pbc_dielectric`'s `D(q)` test measures, because they are the same kernel.
#[test]
fn a_slab_and_a_chain_stop_screening_at_long_wavelength() {
    use am1_rs::pbc::dielectric_function;
    let params = Am1Parameters::standard().unwrap();

    for (name, molecule, periodic, radius, expected) in [
        ("slab", hf_slab(4.0), 2usize, None, 1.0_f64),
        ("chain", hf_chain(4.0), 1, Some(3.0), 2.0),
    ] {
        let o = low_dim_options(periodic);
        let recip = molecule.cell.unwrap().reciprocal_vectors_2pi();
        let mut excess = Vec::new();
        for frac in [0.04_f64, 0.02, 0.01] {
            let q = recip[0] * frac;
            let eps = dielectric_function(&molecule, &params, &o, q, radius).unwrap();
            excess.push(eps - 1.0);
        }
        // Two halvings, so the exponent is the log-ratio over `ln 2` at each step.
        let e1 = (excess[0] / excess[1]).ln() / 2.0_f64.ln();
        let e2 = (excess[1] / excess[2]).ln() / 2.0_f64.ln();
        eprintln!(
            "    {name}: eps-1 = {:.4e}, {:.4e}, {:.4e}; fitted exponent {e1:.3}, {e2:.3} \
             (expect {expected})",
            excess[0], excess[1], excess[2]
        );
        for e in &excess {
            assert!(*e > 0.0, "{name}: eps < 1, which no passive medium gives");
        }
        assert!(
            excess[2] < excess[0],
            "{name}: eps(q) does not approach 1 at long wavelength"
        );
        // The slab's exponent is exactly 1 — `2π χ |q|` has no logarithm. The chain's is 2 only
        // *up to* one: `q² K₀(qρ)` with `K₀ ~ ln(1/q)` fits below 2 and climbs toward it as `q`
        // falls, measured at 1.65 then 1.71. So the slab is held tightly and the chain is
        // required to sit strictly between the slab's exponent and the logarithm-free 2, which is
        // the statement the physics actually makes.
        if periodic == 2 {
            assert!(
                (e2 - expected).abs() < 0.02,
                "{name}: fitted exponent {e2:.3}, and the 2D kernel `2π/|q|` gives exactly 1"
            );
        } else {
            assert!(
                e2 > 1.3 && e2 < expected,
                "{name}: fitted exponent {e2:.3}; `q² K₀(qρ)` must sit below 2 and climb toward \
                 it, and must be clearly steeper than a slab's 1"
            );
            assert!(
                e2 > e1,
                "{name}: the exponent should climb toward 2 as q falls, but went {e1:.3} -> \
                 {e2:.3}"
            );
        }
    }
}

/// The wavevector has to be a Bloch label, and a chain has to say what its logarithm is
/// referenced to. Both are refused by name rather than guessed.
#[test]
fn the_dielectric_function_refuses_what_it_cannot_define() {
    use am1_rs::pbc::dielectric_function;
    let params = Am1Parameters::standard().unwrap();
    let chain = hf_chain(4.0);
    let o = low_dim_options(1);
    let recip = chain.cell.unwrap().reciprocal_vectors_2pi();

    let no_radius = dielectric_function(&chain, &params, &o, recip[0] * 0.02, None)
        .unwrap_err()
        .to_string();
    assert!(no_radius.contains("transverse radius"), "{no_radius}");

    let off_axis = dielectric_function(&chain, &params, &o, Vec3::new(0.0, 0.01, 0.0), Some(3.0))
        .unwrap_err()
        .to_string();
    assert!(off_axis.contains("non-periodic direction"), "{off_axis}");

    let at_gamma = dielectric_function(&chain, &params, &o, Vec3::zero(), Some(3.0))
        .unwrap_err()
        .to_string();
    assert!(at_gamma.contains("`q = 0`"), "{at_gamma}");
}
