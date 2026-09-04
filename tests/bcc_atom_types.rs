// SPDX-License-Identifier: GPL-3.0-or-later

//! AM1-BCC atom typing against `ATOMTYPE_BCC.DEF`, rule by rule.
//!
//! `tests/topology_bcc.rs` checks the *perception* — rings, aromaticity, bond orders — and the
//! charges that follow. This file checks the layer between them: given a correctly perceived
//! molecule, does the typing return the code antechamber's definition file specifies?
//!
//! Every case here is one where it did not, and each is silent: a parameter exists for the wrong
//! type too, so the wrong charge comes back with no warning attached. The sizes are not roundoff
//! — the nitro nitrogen is off by 0.67 e.
//!
//! The rules being tested, quoted from `third_party/antechamber/ATOMTYPE_BCC.DEF`:
//!
//! ```text
//! ATD  33    *   8   1  *  *  *      (C3[RG](O2)) &     <- ring ester/lactone only
//! ATD  33    *   8   1  *  *  *      (C3[RG](N3)) &     <- ring amide/lactam only
//! ATD  32    *   8   1  *  *  *      (C3(O2))     &     <- acid/ester: the C bears an O2
//! ATD  31    *   8   1  &                                <- everything else, incl. ketone/amide
//! ATD  23    *   7   3  *  *  *      (O1,O1)      &     <- nitro
//! ATD  17    *   6   3  *  *  [AR1.AR2]  (N2[AR1.AR2]) & <- aromatic N *of two connections*
//! ATD  16    *   6   3  *  *  [AR1.AR2]  &
//! ```
//!
//! The `[RG]` on both `33` rules and the `2` in `N2` are the load-bearing details: the typing
//! reproduced neither, so it read "count the oxygens on the carbon" and "is any aromatic
//! neighbour an N or an O" instead.

use am1_rs::bcc::{assign_bcc_types, bcc_table};
use am1_rs::topology::Topology;
use am1_rs::{Atom, Molecule, Vec3};

const ANG: f64 = 1.0 / 0.529167;

fn mol(atoms: &[(u8, [f64; 3])]) -> Molecule {
    Molecule::new(
        atoms
            .iter()
            .map(|(z, p)| Atom {
                z: *z,
                position: Vec3::new(p[0], p[1], p[2]) * ANG,
            })
            .collect(),
    )
}

fn types_of(m: &Molecule) -> Vec<u32> {
    assign_bcc_types(m, &Topology::perceive(m))
}

/// H2C=O. The carbonyl carbon bears no `O2`, so the `32` rule cannot match and the oxygen falls
/// through to the generic `31`. Every ketone and aldehyde is this case.
#[test]
fn a_ketone_oxygen_is_31_not_32() {
    let formaldehyde = mol(&[
        (6, [0.0, 0.0, 0.0]),
        (8, [0.0, 1.21, 0.0]),
        (1, [0.94, -0.59, 0.0]),
        (1, [-0.94, -0.59, 0.0]),
    ]);
    let t = types_of(&formaldehyde);
    eprintln!("    formaldehyde: C {} O {}", t[0], t[1]);
    assert_eq!(t[0], 14, "carbonyl carbon");
    assert_eq!(
        t[1], 31,
        "a ketone/aldehyde oxygen has no O2 on its carbon, so `32 (C3(O2))` cannot match"
    );
}

/// HCOOH. Here the carbon *does* bear an `O2` (the hydroxyl), so `32` matches — but the molecule
/// is not a ring, so neither `33` rule applies. The typing counted oxygens instead and said 33.
#[test]
fn an_acid_or_ester_oxygen_is_32_not_33() {
    let formic_acid = mol(&[
        (6, [0.0, 0.0, 0.0]),
        (8, [0.0, 1.21, 0.0]),     // carbonyl, 1-connected
        (8, [1.163, -0.672, 0.0]), // hydroxyl, 2-connected
        (1, [-1.03, -0.42, 0.0]),  // on C
        (1, [1.10, -1.61, 0.0]),   // on the hydroxyl O
    ]);
    let topo = Topology::perceive(&formic_acid);
    assert_eq!(topo.neighbors[2].len(), 2, "the hydroxyl O must be O2");
    let t = assign_bcc_types(&formic_acid, &topo);
    eprintln!("    formic acid: C {} O(=) {} O(H) {}", t[0], t[1], t[2]);
    assert_eq!(
        t[1], 32,
        "the carbon bears an O2, and `33` needs the carbon to be in a ring"
    );
}

/// HC(=O)NH2. `33 (C3[RG](N3))` needs a ring; a chain amide has none, and its carbon bears no
/// `O2` either, so the oxygen is the generic `31`. This is every peptide carbonyl.
#[test]
fn an_amide_oxygen_is_31_not_33() {
    let formamide = mol(&[
        (6, [0.0, 0.0, 0.0]),
        (8, [0.0, 1.22, 0.0]),
        (7, [1.20, -0.72, 0.0]),
        (1, [-1.03, -0.43, 0.0]), // on C
        (1, [2.19, -0.55, 0.0]),  // on N
        (1, [1.15, -1.72, 0.0]),  // on N
    ]);
    let topo = Topology::perceive(&formamide);
    assert_eq!(topo.neighbors[2].len(), 3, "the amide N must be N3");
    let t = assign_bcc_types(&formamide, &topo);
    eprintln!("    formamide: C {} O {} N {}", t[0], t[1], t[2]);
    assert_eq!(t[2], 22, "amide nitrogen");
    assert_eq!(
        t[1], 31,
        "`33 (C3[RG](N3))` is the lactam rule and needs a ring; a chain amide is 31"
    );
}

/// CH3NO2. `23 * 7 3 * * * (O1,O1) &` is the nitro rule and it sits above the `21` fallback.
/// The typing had no such rule, so every nitro nitrogen came out 21.
#[test]
fn a_nitro_nitrogen_is_23_not_21() {
    let t = types_of(&nitromethane());
    eprintln!(
        "    nitromethane: N {} O {} O {} C {}",
        t[0], t[1], t[2], t[3]
    );
    assert_eq!(
        t[0], 23,
        "a three-connected N with two 1-connected oxygens is the nitro rule"
    );
}

/// The nitro mistyping in electrons, so the size is on the record and not just the code.
///
/// Every bond at the nitrogen changes, not only the two N–O bonds: the correction is looked up on
/// the *pair* of types, so the C–N single bond moves too. That is why counting one bond
/// understates it.
#[test]
fn the_nitro_mistyping_moves_the_nitrogen_by_two_thirds_of_an_electron() {
    let table = bcc_table();
    let get = |a, b, bt| *table.get(&(a, b, bt)).expect("parameter present");

    // Sign convention (`bcc::am1_bcc_charges`): a table entry `(ta, tb, bt) = d` puts `+d` on the
    // atom of type `ta` and `−d` on the atom of type `tb`.
    let n_o_right = get(23, 31, 6); // N(23) gains this, twice
    let n_o_wrong = get(21, 31, 6);
    let c_n_right = -get(11, 23, 1); // entry is (C, N), so N gains the negative
    let c_n_wrong = -get(11, 21, 1);

    let right = 2.0 * n_o_right + c_n_right;
    let wrong = 2.0 * n_o_wrong + c_n_wrong;
    eprintln!(
        "    nitro N correction: correct {right:+.4} e, mistyped {wrong:+.4} e, error {:.4} e",
        right - wrong
    );
    assert!(
        (right - wrong).abs() > 0.6,
        "the nitro mistyping should move the nitrogen by more than 0.6 e, got {:.4}",
        right - wrong
    );

    // And it really is what the pipeline applies: the shift from raw Mulliken to BCC on the
    // nitrogen is exactly the sum of its three bond corrections.
    let params = am1_rs::Am1Parameters::standard().unwrap();
    let r =
        am1_rs::am1_bcc_charges(&nitromethane(), &params, &am1_rs::Am1Options::default()).unwrap();
    let applied = r.charges[0] - r.mulliken[0];
    eprintln!("    applied to N: {applied:+.4} e (expected {right:+.4})");
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    assert!(
        (applied - right).abs() < 1.0e-9,
        "N shifted by {applied:+.6}, expected {right:+.6}"
    );
}

/// Pyrrole's α carbons. `17` requires an aromatic **`N2`** neighbour — two connections, as in
/// pyridine. Pyrrole's nitrogen is `N3`, so its neighbours fall to `16`. The typing checked only
/// that some aromatic neighbour was an N or an O.
#[test]
fn a_pyrrole_alpha_carbon_is_16_not_17() {
    let pyrrole = five_ring_with_hetero(7);
    let topo = Topology::perceive(&pyrrole);
    assert!(
        (0..5).all(|i| topo.aromatic[i]),
        "the fixture must be perceived as aromatic for this to test the typing"
    );
    assert_eq!(topo.neighbors[0].len(), 3, "pyrrole N is N3");
    let t = assign_bcc_types(&pyrrole, &topo);
    eprintln!("    pyrrole: N {} ring C {:?}", t[0], &t[1..5]);
    assert_eq!(t[0], 23, "aromatic N3");
    for (k, ty) in t[1..5].iter().enumerate() {
        assert_eq!(
            *ty,
            16,
            "ring carbon {} — 17 needs an aromatic N2, and pyrrole's N has three connections",
            k + 1
        );
    }
}

/// Pyridine, the case `17` is actually for: its nitrogen *is* `N2`, so both α carbons are 17.
///
/// Included so the fix is not "delete the 17 branch": it has to keep firing where the definition
/// file says it should.
#[test]
fn a_pyridine_alpha_carbon_is_still_17() {
    let bond = 1.39;
    let radius = bond / (2.0 * (std::f64::consts::PI / 6.0).sin());
    let mut atoms = Vec::new();
    for k in 0..6 {
        let t = 2.0 * std::f64::consts::PI * k as f64 / 6.0;
        atoms.push((
            if k == 0 { 7u8 } else { 6u8 },
            [radius * t.cos(), radius * t.sin(), 0.0],
        ));
    }
    for k in 1..6 {
        let t = 2.0 * std::f64::consts::PI * k as f64 / 6.0;
        let r = radius + 1.08;
        atoms.push((1u8, [r * t.cos(), r * t.sin(), 0.0]));
    }
    let pyridine = mol(&atoms);
    let topo = Topology::perceive(&pyridine);
    assert_eq!(topo.neighbors[0].len(), 2, "pyridine N is N2");
    let t = assign_bcc_types(&pyridine, &topo);
    eprintln!("    pyridine: N {} ring C {:?}", t[0], &t[1..6]);
    assert_eq!(t[1], 17, "alpha carbon");
    assert_eq!(t[5], 17, "the other alpha carbon");
    assert_eq!(t[2], 16, "beta carbon");
}

// --------------------------------------------------------------- the unreachable bond types
//
// `docs/scope.md` and `THIRD_PARTY_NOTICES.md` both claim that bond types 8, 10 and 11 are
// unreachable *and* inconsequential — 8 and 10 duplicate type 7, and 11 is identically zero. The
// claim was never asserted anywhere. These two tests put it against the parameter file itself,
// which is the only thing that can settle it.
//
// The claim is deliberately about the **parameter values**, not about charges: "no bit of any
// charge changes" would additionally need the order of the additions to be preserved, and adding
// `0.0` to a `−0.0` flips a sign bit. What the file supports is that the number applied would be
// the same number.

#[test]
fn type_8_and_10_are_byte_identical_to_type_7() {
    let table = bcc_table();
    for dup in [8u32, 10u32] {
        let mut checked = 0;
        for (&(i, j, bt), &value) in table.iter() {
            if bt != dup {
                continue;
            }
            let seven = table.get(&(i, j, 7)).copied().unwrap_or_else(|| {
                panic!("({i},{j},{dup}) has no type-7 counterpart, so the claim is false")
            });
            assert_eq!(
                value.to_bits(),
                seven.to_bits(),
                "({i},{j},{dup}) = {value} but ({i},{j},7) = {seven}"
            );
            checked += 1;
        }
        eprintln!("    bond type {dup}: {checked} entries, all byte-identical to type 7");
        assert!(checked > 0, "no type-{dup} entries found");
    }
}

#[test]
fn type_11_is_identically_zero() {
    let table = bcc_table();
    let mut checked = 0;
    for (&(i, j, bt), &value) in table.iter() {
        if bt != 11 {
            continue;
        }
        assert_eq!(value, 0.0, "({i},{j},11) = {value}");
        checked += 1;
    }
    eprintln!("    bond type 11: {checked} entries, all exactly zero");
    assert_eq!(checked, 26, "the file holds 26 type-11 entries");
}

// ------------------------------------------------------------------------------------ fixtures

fn nitromethane() -> Molecule {
    mol(&[
        (7, [0.0, 0.0, 0.0]),
        (8, [1.10, 0.53, 0.0]),
        (8, [-1.10, 0.53, 0.0]),
        (6, [0.0, -1.49, 0.0]),
        (1, [0.90, -1.90, 0.45]),
        (1, [-0.90, -1.90, 0.45]),
        (1, [0.0, -1.85, -1.03]),
    ])
}

/// A planar 5-ring with `hetero` at index 0 and carbons at 1..5, hydrogenated. Indices 1 and 4
/// are the α positions.
fn five_ring_with_hetero(hetero: u8) -> Molecule {
    let bond = if hetero == 16 { 1.71 } else { 1.37 };
    let radius = bond / (2.0 * (std::f64::consts::PI / 5.0).sin());
    let mut atoms = Vec::new();
    for k in 0..5 {
        let t = 2.0 * std::f64::consts::PI * k as f64 / 5.0 + std::f64::consts::FRAC_PI_2;
        let z = if k == 0 { hetero } else { 6u8 };
        atoms.push((z, [radius * t.cos(), radius * t.sin(), 0.0]));
    }
    for k in 0..5 {
        if k == 0 && hetero != 7 {
            continue;
        }
        let t = 2.0 * std::f64::consts::PI * k as f64 / 5.0 + std::f64::consts::FRAC_PI_2;
        let r = radius + 1.02;
        atoms.push((1u8, [r * t.cos(), r * t.sin(), 0.0]));
    }
    mol(&atoms)
}
