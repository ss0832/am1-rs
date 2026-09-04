// SPDX-License-Identifier: GPL-3.0-or-later

//! An interpreter for antechamber's `ATOMTYPE_BCC.DEF`, so the AM1-BCC atom types are **read from
//! the definition file** rather than transcribed into Rust `match` arms.
//!
//! # Why this exists
//!
//! The previous typing was a hand translation of the same file, and it had drifted from it in ten
//! places — silently, because a wrong type still finds a parameter. A chain amide's carbonyl oxygen
//! came out 33 instead of 31, an ester's 33 instead of 32, a ketone's 32 instead of 31, and a nitro
//! nitrogen 21 instead of 23, which moved that nitrogen by **0.67 e**. Each of those is one clause
//! of one rule: the two `33` rules both carry `[RG]` and apply to lactones and lactams only, the
//! `32` rule asks whether the *carbon* bears a two-connected oxygen rather than how many oxygens it
//! has, and there is a nitro rule (`(O1,O1)`) that the translation simply did not contain.
//!
//! Reading the file removes that whole class of defect: the rules, and their order — which is
//! load-bearing, since the first match wins — are the file's, not a reading of it.
//!
//! # The subset interpreted
//!
//! `ATOMTYPE_BCC.DEF` is written in a general language, but this particular file uses a small part
//! of it, and only that part is implemented:
//!
//! * fields `f4` (atomic number) and `f5` (number of attached atoms). `f3` (residue), `f6`
//!   (attached hydrogens) and `f7` (electron-withdrawing neighbours) are `*` on every line here.
//! * `f8` atom properties: `[AR1.AR2]`, `[db]`, `[2sb]`, `[sb,db]`, `[RG5]`. `.` is the file's
//!   documented "or"; `,` is "and"; a leading integer is a count.
//! * `f9` chemical environments: nested, comma-separated neighbour patterns such as
//!   `(C3[RG](O2))`, `(O1,O1)` and `(N2[RG5](N2[RG5](N2[RG5])))`, including the `'` suffix that
//!   pins a bond property to the bond back to the previous atom in the chain (`(N2[db'])`).
//! * `WILDATOM` expansion for `XX`, `XA`, `XB` and `XD`.
//!
//! Comma-separated environment items must match **distinct** neighbours — `(O1,O1)` is two
//! oxygens, not one matched twice — so the match is an injective assignment found by backtracking.
//!
//! # What is read but not enforced
//!
//! Two `23` rules carry chain labels and a trailing constraint, `a1:a2:any`. The labels are parsed
//! and the constraint is read as its name says — `any` relation between the two labelled atoms,
//! i.e. no restriction — so those rules are applied on their structural part alone. This is the one
//! place where the interpretation is a reading of the syntax rather than a transcription of it, and
//! it is recorded here rather than left silent. It can only move a nitrogen from the `21` fallback
//! to `23`, which is the direction the file's ordering intends.
//!
//! The uppercase strict bond kinds (`SB`, `DB`, `TB`, `AB`, `DL`) and `AR3`..`AR5`, `NR`, `RG3`,
//! `RG4`, `RG6`..`RG9` are implemented because the language has them, but this file uses none of
//! them; `AR1` and `AR2` are only ever asked for as the union `[AR1.AR2]`, which is why
//! [`crate::topology`] carries one aromatic flag rather than five classes.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::system::{symbol_to_z, Molecule};
use crate::topology::{BondOrder, Topology};

/// The antechamber definition file, verbatim (AmberTools, GPL-3). See `THIRD_PARTY_NOTICES.md`.
const ATOMTYPE_BCC_DEF: &str = include_str!("../../third_party/antechamber/ATOMTYPE_BCC.DEF");

/// One `(element, connection count)` alternative that a spec can match. `None` is "any".
#[derive(Clone, Copy, Debug)]
struct Alt {
    z: Option<u8>,
    connections: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BondKind {
    /// Lowercase `sb`: single, aromatic-single or delocalized.
    Sb,
    /// Lowercase `db`: double or aromatic-double.
    Db,
    Tb,
    StrictSingle,
    StrictDouble,
    StrictTriple,
    StrictAromatic,
    StrictDelocalized,
}

#[derive(Clone, Copy, Debug)]
enum Prop {
    /// `AR1`..`AR5`. Only ever asked for as the union in this file.
    Aromatic,
    /// `RG`, or `RG3`..`RG9` with a size.
    InRing(Option<usize>),
    /// `NR`.
    NonRing,
    Bond {
        kind: BondKind,
        count: usize,
        /// `Some(true)` for the `'` suffix (must form this bond with the predecessor),
        /// `Some(false)` for `''` (must not), `None` for any bond of the atom.
        predecessor: Option<bool>,
    },
}

/// An "and of ors": `[sb,db]` is two groups of one, `[AR1.AR2]` is one group of two.
#[derive(Clone, Debug, Default)]
struct PropExpr(Vec<Vec<Prop>>);

#[derive(Clone, Debug)]
struct Spec {
    /// Empty means `*` — any atom.
    alts: Vec<Alt>,
    props: PropExpr,
}

#[derive(Clone, Debug)]
struct EnvItem {
    spec: Spec,
    children: Vec<EnvItem>,
}

#[derive(Clone, Debug)]
struct Rule {
    /// The emitted type code. `0` is the file's `DU` catch-all.
    type_code: u32,
    z: Option<u8>,
    connections: Option<usize>,
    props: PropExpr,
    env: Vec<EnvItem>,
}

/// The parsed definition file, built once per process.
fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| parse_def(ATOMTYPE_BCC_DEF))
}

/// The number of `ATD` rules the definition file yielded. Exposed so a test can assert that the
/// parser consumed the whole file rather than silently skipping lines it did not understand.
pub fn rule_count() -> usize {
    rules().len()
}

/// Assign the antechamber BCC atom-type code (11–91) per atom by evaluating the definition file.
///
/// Rules are tried in file order and the first match wins — the file's own closing note says the
/// order is crucial, and it is: `21 * 7 4 &` precedes the aromatic `23` rule, so a four-coordinate
/// aromatic nitrogen is 21 and not 23.
///
/// `0` means no rule but the `DU` catch-all matched, i.e. the element has no BCC type;
/// [`crate::topology::Topology::warnings`] already reports those.
pub fn assign_bcc_types(molecule: &Molecule, topo: &Topology) -> Vec<u32> {
    (0..molecule.atoms.len())
        .map(|i| {
            rules()
                .iter()
                .find(|r| rule_matches(r, molecule, topo, i))
                .map(|r| r.type_code)
                .unwrap_or(0)
        })
        .collect()
}

// ------------------------------------------------------------------------------------ matching

fn rule_matches(rule: &Rule, molecule: &Molecule, topo: &Topology, atom: usize) -> bool {
    if let Some(z) = rule.z {
        if molecule.atoms[atom].z != z {
            return false;
        }
    }
    if let Some(n) = rule.connections {
        if topo.neighbors[atom].len() != n {
            return false;
        }
    }
    if !props_hold(&rule.props, topo, atom, None) {
        return false;
    }
    match_env(&rule.env, molecule, topo, atom, None)
}

fn props_hold(expr: &PropExpr, topo: &Topology, atom: usize, predecessor: Option<usize>) -> bool {
    expr.0.iter().all(|group| {
        group
            .iter()
            .any(|p| prop_holds(*p, topo, atom, predecessor))
    })
}

fn prop_holds(prop: Prop, topo: &Topology, atom: usize, predecessor: Option<usize>) -> bool {
    match prop {
        Prop::Aromatic => topo.aromatic[atom],
        Prop::NonRing => !topo.in_ring[atom],
        Prop::InRing(None) => topo.in_ring[atom],
        Prop::InRing(Some(n)) => topo
            .rings
            .iter()
            .any(|r| r.size() == n && r.atoms.contains(&atom)),
        Prop::Bond {
            kind,
            count,
            predecessor: to_prev,
        } => match to_prev {
            None => {
                let have = topo
                    .bonds
                    .iter()
                    .enumerate()
                    .filter(|(k, b)| (b.i == atom || b.j == atom) && bond_is(topo, *k, kind))
                    .count();
                have >= count
            }
            Some(want) => {
                let Some(prev) = predecessor else {
                    return !want;
                };
                let held = topo
                    .bond_between(atom, prev)
                    .is_some_and(|k| bond_is(topo, k, kind));
                held == want
            }
        },
    }
}

fn bond_is(topo: &Topology, k: usize, kind: BondKind) -> bool {
    let (sb, db, tb) = topo.bond_kinds(k);
    match kind {
        BondKind::Sb => sb,
        BondKind::Db => db,
        BondKind::Tb => tb,
        BondKind::StrictSingle => topo.bonds[k].order == BondOrder::Single,
        BondKind::StrictDouble => topo.bonds[k].order == BondOrder::Double,
        BondKind::StrictTriple => topo.bonds[k].order == BondOrder::Triple,
        BondKind::StrictAromatic => topo.bonds[k].order == BondOrder::Aromatic,
        BondKind::StrictDelocalized => topo.bonds[k].order == BondOrder::Delocalized,
    }
}

fn spec_matches(
    spec: &Spec,
    molecule: &Molecule,
    topo: &Topology,
    atom: usize,
    predecessor: Option<usize>,
) -> bool {
    let z = molecule.atoms[atom].z;
    let cn = topo.neighbors[atom].len();
    // `map_or(true, ..)` rather than `is_none_or`: the latter is Rust 1.82 and this crate's
    // declared MSRV is 1.75, which CI checks.
    #[allow(clippy::unnecessary_map_or)]
    let element_ok = spec.alts.is_empty()
        || spec.alts.iter().any(|a| {
            a.z.map_or(true, |want| want == z) && a.connections.map_or(true, |want| want == cn)
        });
    element_ok && props_hold(&spec.props, topo, atom, predecessor)
}

/// Match a comma-separated environment against the neighbours of `atom`, injectively.
///
/// The predecessor is excluded from the candidates so a pattern cannot satisfy itself by walking
/// back the way it came: `(C3(O1))` on an amide nitrogen has to find an oxygen on the carbon, not
/// rediscover the nitrogen.
fn match_env(
    items: &[EnvItem],
    molecule: &Molecule,
    topo: &Topology,
    atom: usize,
    predecessor: Option<usize>,
) -> bool {
    if items.is_empty() {
        return true;
    }
    let candidates: Vec<usize> = topo.neighbors[atom]
        .iter()
        .copied()
        .filter(|&n| Some(n) != predecessor)
        .collect();
    let mut used = vec![false; candidates.len()];
    assign_items(items, 0, &candidates, &mut used, molecule, topo, atom)
}

fn assign_items(
    items: &[EnvItem],
    at: usize,
    candidates: &[usize],
    used: &mut [bool],
    molecule: &Molecule,
    topo: &Topology,
    parent: usize,
) -> bool {
    let Some(item) = items.get(at) else {
        return true;
    };
    for (slot, &cand) in candidates.iter().enumerate() {
        if used[slot] {
            continue;
        }
        if !spec_matches(&item.spec, molecule, topo, cand, Some(parent)) {
            continue;
        }
        if !match_env(&item.children, molecule, topo, cand, Some(parent)) {
            continue;
        }
        used[slot] = true;
        if assign_items(items, at + 1, candidates, used, molecule, topo, parent) {
            return true;
        }
        used[slot] = false;
    }
    false
}

// ------------------------------------------------------------------------------------- parsing

fn parse_def(text: &str) -> Vec<Rule> {
    let mut wildatoms: HashMap<String, Vec<Alt>> = HashMap::new();
    let mut rules = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        let mut tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first() {
            Some(&"WILDATOM") if tokens.len() >= 3 => {
                let name = tokens[1].to_string();
                let alts = tokens[2..]
                    .iter()
                    .filter_map(|t| parse_alt(t, &wildatoms))
                    .flatten()
                    .collect();
                wildatoms.insert(name, alts);
            }
            Some(&"ATD") => {
                // The `&` terminator is not a field. The two rules that carry a chain constraint
                // have no `&` at all, and their trailing `a1:a2:any` is read but not enforced —
                // see the module documentation.
                if tokens.last() == Some(&"&") {
                    tokens.pop();
                }
                if let Some(rule) = parse_rule(&tokens, &wildatoms) {
                    rules.push(rule);
                }
            }
            _ => {}
        }
    }
    rules
}

fn parse_rule(tokens: &[&str], wildatoms: &HashMap<String, Vec<Alt>>) -> Option<Rule> {
    let name = tokens.get(1)?;
    // `DU` is the file's catch-all; everything else is a numeric BCC type.
    let type_code = if *name == "DU" {
        0
    } else {
        name.parse::<u32>().ok()?
    };
    let field =
        |i: usize| -> Option<&str> { tokens.get(i).copied().filter(|t| *t != "*" && *t != "&") };
    Some(Rule {
        type_code,
        z: field(3).and_then(|t| t.parse::<u8>().ok()),
        connections: field(4).and_then(|t| t.parse::<usize>().ok()),
        props: field(7).map(parse_props).unwrap_or_default(),
        env: field(8)
            .map(|t| parse_env(t, wildatoms))
            .unwrap_or_default(),
    })
}

/// `[AR1.AR2]`, `[sb,db]`, `[2sb]`, `[db']` — with or without the brackets.
fn parse_props(text: &str) -> PropExpr {
    let inner = text.trim_start_matches('[').trim_end_matches(']');
    if inner.is_empty() {
        return PropExpr::default();
    }
    PropExpr(
        inner
            .split(',')
            .map(|group| group.split('.').filter_map(parse_prop).collect::<Vec<_>>())
            .filter(|g: &Vec<Prop>| !g.is_empty())
            .collect(),
    )
}

fn parse_prop(text: &str) -> Option<Prop> {
    let text = text.trim();
    // An optional leading repeat count, as in `2sb`.
    let digits: String = text.chars().take_while(char::is_ascii_digit).collect();
    let rest = &text[digits.len()..];
    let count = digits.parse::<usize>().unwrap_or(1);

    // Trailing `'` (forms this bond with the predecessor) or `''` (does not).
    let primes = rest.chars().rev().take_while(|c| *c == '\'').count();
    let name = &rest[..rest.len() - primes];
    let predecessor = match primes {
        0 => None,
        1 => Some(true),
        _ => Some(false),
    };

    let bond = |kind| {
        Some(Prop::Bond {
            kind,
            count,
            predecessor,
        })
    };
    match name {
        "AR1" | "AR2" | "AR3" | "AR4" | "AR5" => Some(Prop::Aromatic),
        "NR" => Some(Prop::NonRing),
        "RG" => Some(Prop::InRing(None)),
        "sb" => bond(BondKind::Sb),
        "db" => bond(BondKind::Db),
        "tb" => bond(BondKind::Tb),
        "SB" => bond(BondKind::StrictSingle),
        "DB" => bond(BondKind::StrictDouble),
        "TB" => bond(BondKind::StrictTriple),
        "AB" => bond(BondKind::StrictAromatic),
        "DL" => bond(BondKind::StrictDelocalized),
        other => other
            .strip_prefix("RG")
            .and_then(|n| n.parse::<usize>().ok())
            .map(|n| Prop::InRing(Some(n))),
    }
}

/// `(C3[RG](O2))` → one item; `(O1,O1)` → two. The outer parentheses are the field's own.
fn parse_env(text: &str, wildatoms: &HashMap<String, Vec<Alt>>) -> Vec<EnvItem> {
    let inner = strip_outer_parens(text.trim());
    parse_env_list(inner, wildatoms)
}

fn strip_outer_parens(text: &str) -> &str {
    if text.starts_with('(') && text.ends_with(')') && balanced_to_end(text) {
        &text[1..text.len() - 1]
    } else {
        text
    }
}

/// Whether the opening parenthesis of `text` closes only at its final character, so stripping the
/// pair is safe. `(a)(b)` would not qualify.
fn balanced_to_end(text: &str) -> bool {
    let mut depth = 0usize;
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return i == text.len() - 1;
                }
            }
            _ => {}
        }
    }
    false
}

fn parse_env_list(text: &str, wildatoms: &HashMap<String, Vec<Alt>>) -> Vec<EnvItem> {
    split_top_level(text, ',')
        .into_iter()
        .filter_map(|piece| parse_env_item(piece.trim(), wildatoms))
        .collect()
}

/// Split on `sep` at parenthesis depth zero.
fn split_top_level(text: &str, sep: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if c == sep && depth == 0 => {
                out.push(&text[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

fn parse_env_item(text: &str, wildatoms: &HashMap<String, Vec<Alt>>) -> Option<EnvItem> {
    // The spec runs up to the first top-level `(`, which begins the child environment.
    let mut depth = 0usize;
    let mut split = text.len();
    for (i, c) in text.char_indices() {
        if c == '(' {
            if depth == 0 {
                split = i;
            }
            depth += 1;
        } else if c == ')' {
            depth = depth.saturating_sub(1);
        }
    }
    let (head, tail) = text.split_at(split);
    let spec = parse_spec(head.trim(), wildatoms)?;
    let children = if tail.is_empty() {
        Vec::new()
    } else {
        parse_env_list(strip_outer_parens(tail.trim()), wildatoms)
    };
    Some(EnvItem { spec, children })
}

/// `C3[RG]`, `XA1`, `N2[db']`, `XX[AR1.AR2]<a1>`, `*`.
fn parse_spec(text: &str, wildatoms: &HashMap<String, Vec<Alt>>) -> Option<Spec> {
    // The chain label is parsed and discarded; see the module documentation on `a1:a2:any`.
    let text = match text.find('<') {
        Some(i) => &text[..i],
        None => text,
    };
    let (head, props) = match text.find('[') {
        Some(i) => (&text[..i], parse_props(&text[i..])),
        None => (text, PropExpr::default()),
    };
    let head = head.trim();
    if head.is_empty() || head == "*" {
        return Some(Spec {
            alts: Vec::new(),
            props,
        });
    }
    Some(Spec {
        alts: parse_alt(head, wildatoms)?,
        props,
    })
}

/// Resolve `C3`, `XA1`, `XB` or `O` into the alternatives it stands for.
///
/// A count on the outer token applies to any wildatom member that does not carry one of its own:
/// `XA1` is `O1` or `S1`, while `XB` keeps the counts written into its own definition
/// (`C3 N2 N3 O2 S2 P2`).
fn parse_alt(text: &str, wildatoms: &HashMap<String, Vec<Alt>>) -> Option<Vec<Alt>> {
    let text = text.trim();
    if text.is_empty() || text == "*" {
        return Some(Vec::new());
    }
    let name: String = text
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    let digits = &text[name.len()..];
    let connections = digits.parse::<usize>().ok();

    if let Some(members) = wildatoms.get(&name) {
        return Some(
            members
                .iter()
                .map(|m| Alt {
                    z: m.z,
                    connections: m.connections.or(connections),
                })
                .collect(),
        );
    }
    let z = symbol_to_z(&name)?;
    Some(vec![Alt {
        z: Some(z),
        connections,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parser must consume every `ATD` line. A rule silently dropped would show up as a type
    /// falling through to a later, more general rule — exactly the failure this module exists to
    /// remove — so the count is asserted rather than assumed.
    #[test]
    fn every_rule_in_the_definition_file_is_parsed() {
        let expected = ATOMTYPE_BCC_DEF
            .lines()
            .filter(|l| l.trim_start().starts_with("ATD "))
            .count();
        eprintln!("    {} ATD lines, {} rules parsed", expected, rule_count());
        assert_eq!(rule_count(), expected);
    }

    /// The four `WILDATOM` lines have to reach the specs that use them: `XA1` in the type-14 rule
    /// is where a carbonyl is recognized, and if `XA` resolved to nothing that rule would match
    /// every three-connected carbon.
    #[test]
    fn wildatoms_expand_with_the_outer_count_applied() {
        let mut wild = HashMap::new();
        wild.insert(
            "XA".to_string(),
            vec![
                Alt {
                    z: Some(8),
                    connections: None,
                },
                Alt {
                    z: Some(16),
                    connections: None,
                },
            ],
        );
        let alts = parse_alt("XA1", &wild).unwrap();
        assert_eq!(alts.len(), 2);
        assert!(alts.iter().all(|a| a.connections == Some(1)));
        assert!(alts.iter().any(|a| a.z == Some(8)));
        assert!(alts.iter().any(|a| a.z == Some(16)));

        // A member that carries its own count keeps it.
        wild.insert(
            "XB".to_string(),
            vec![Alt {
                z: Some(6),
                connections: Some(3),
            }],
        );
        let alts = parse_alt("XB", &wild).unwrap();
        assert_eq!(alts[0].connections, Some(3));
    }

    #[test]
    fn nested_environments_parse_to_the_right_depth() {
        let wild = HashMap::new();
        let env = parse_env("(N2[RG5](N2[RG5](N2[RG5])))", &wild);
        assert_eq!(env.len(), 1);
        assert_eq!(env[0].children.len(), 1);
        assert_eq!(env[0].children[0].children.len(), 1);

        // Two items at the top level, the second with a child.
        let env = parse_env("(N2[RG5],N2[RG5](N2[RG5]))", &wild);
        assert_eq!(env.len(), 2);
        assert!(env[0].children.is_empty());
        assert_eq!(env[1].children.len(), 1);
    }

    #[test]
    fn a_primed_bond_property_is_read_as_a_constraint_on_the_predecessor() {
        let expr = parse_props("[db']");
        match expr.0[0][0] {
            Prop::Bond {
                kind,
                predecessor,
                count,
            } => {
                assert_eq!(kind, BondKind::Db);
                assert_eq!(predecessor, Some(true));
                assert_eq!(count, 1);
            }
            other => panic!("expected a bond property, got {other:?}"),
        }
    }

    #[test]
    fn a_count_prefix_is_read_as_a_count() {
        let expr = parse_props("[2sb]");
        match expr.0[0][0] {
            Prop::Bond { kind, count, .. } => {
                assert_eq!(kind, BondKind::Sb);
                assert_eq!(count, 2);
            }
            other => panic!("expected a bond property, got {other:?}"),
        }
        // `,` is "and", `.` is "or" — the file uses both and they are not interchangeable.
        assert_eq!(parse_props("[sb,db]").0.len(), 2);
        assert_eq!(parse_props("[AR1.AR2]").0.len(), 1);
        assert_eq!(parse_props("[AR1.AR2]").0[0].len(), 2);
    }
}
