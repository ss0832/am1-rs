// SPDX-License-Identifier: GPL-3.0-or-later

//! Licensing and attribution, as invariants rather than as a thing someone remembers.
//!
//! This crate bundles three third-party works — PySEQM (BSD-3-Clause), MOPAC (Apache-2.0) and
//! antechamber (GPL-3) — and compiles their data into every binary it ships. Each of those
//! licences attaches conditions to *binary* redistribution: BSD-3-Clause clause 2 wants the
//! copyright notice to accompany it, Apache-2.0 §4(a) wants the licence text and §4(b)–(c) want
//! the statement of changes and the retained attribution. A wheel and a `.crate` are both binary
//! redistributions.
//!
//! # Why this is a test and not a checklist
//!
//! The failure mode is silent and it has already happened twice here. Through 0.2.1 the
//! antechamber licence was in no file anywhere while its data was compiled into every wheel, and
//! the CI job meant to catch that asserted only that *some* licence was bundled — which two out
//! of three satisfies. Through 0.2.2 the per-work provenance notes reached the sdist and not the
//! wheel, while three separate documents claimed CI checked for them in both. Nobody noticed
//! either, because nothing fails when attribution goes missing; it just goes missing.
//!
//! So the invariants are asserted from the Rust suite, which runs on every change, rather than
//! only from the packaging job, which runs at release. What this file cannot see is the *built*
//! artifacts — that check lives in `.github/workflows/ci.yml`, against a real wheel. What it can
//! see is every input that determines them, which is where a regression starts.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every file under `dir` with one of `extensions`, recursively, skipping build output.
fn sources(dir: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "target" || name == "__pycache__" {
            continue;
        }
        if path.is_dir() {
            out.extend(sources(&path, extensions));
        } else if extensions
            .iter()
            .any(|e| path.extension().is_some_and(|x| x == *e))
        {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// The subdirectories of `third_party/`, which is the list everything else is checked against.
fn bundled_works() -> BTreeSet<String> {
    fs::read_dir(root().join("third_party"))
        .expect("third_party/ must exist")
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()
}

/// Every bundled work carries both its licence and its provenance note.
///
/// Both, not either. The licence says what the terms are; the README beside it says which files
/// they cover, where those came from and — for Apache-2.0 §4(b) — what was changed. A directory
/// with only the licence satisfies the clause that is easy to satisfy and none of the others.
#[test]
fn every_bundled_work_has_a_licence_and_a_provenance_note() {
    for work in bundled_works() {
        let dir = root().join("third_party").join(&work);
        for required in ["LICENSE", "README.md"] {
            let path = dir.join(required);
            assert!(
                path.is_file(),
                "third_party/{work}/ has no {required}; bundled material must carry both"
            );
            assert!(
                fs::read_to_string(&path).unwrap().trim().len() > 200,
                "third_party/{work}/{required} is too short to be the real thing"
            );
        }
    }
}

/// The cross-cutting notices file accounts for every bundled work.
///
/// A fourth work can otherwise arrive with its own directory, pass the check above, and never be
/// mentioned in the document a reader actually opens.
#[test]
fn the_notices_file_accounts_for_every_bundled_work() {
    let notices = fs::read_to_string(root().join("THIRD_PARTY_NOTICES.md")).unwrap();
    for work in bundled_works() {
        assert!(
            notices.contains(&format!("third_party/{work}/")),
            "THIRD_PARTY_NOTICES.md never mentions third_party/{work}/"
        );
    }
}

/// The wheel's notice set is declared, and covers licences *and* provenance notes.
///
/// PEP 639 `license-files` is what puts these under `dist-info/licenses/`. Through 0.2.2 it
/// listed `third_party/*/LICENSE` only, so the provenance notes were sdist-only — which is where
/// Apache-2.0 §4(b)'s statement of changes lives, so the clause that needed the wheel was the one
/// that did not reach it.
#[test]
fn the_wheel_declares_licences_and_provenance_notes() {
    let pyproject = fs::read_to_string(root().join("pyproject.toml")).unwrap();
    // `license-files = [`, not `license-files`: the key is discussed in a comment a few lines
    // above its own definition, and matching the prose reads the wrong block.
    let start = pyproject
        .find("license-files = [")
        .expect("pyproject.toml must declare license-files");
    let end = start
        + pyproject[start..]
            .find(']')
            .expect("license-files must be a list");
    let block = &pyproject[start..end];
    for required in [
        "\"LICENSE\"",
        "\"THIRD_PARTY_NOTICES.md\"",
        "\"THIRD_PARTY_LICENSES.md\"",
        "third_party/*/LICENSE",
        "third_party/*/README.md",
    ] {
        assert!(
            block.contains(required),
            "pyproject.toml `license-files` does not list {required}; it would not reach the wheel"
        );
    }
}

/// Nothing in the crates.io package excludes the notices.
///
/// `cargo package` ships everything git-tracked unless told otherwise, so today the `.crate`
/// carries all of it — *incidentally*. An `exclude` added later for build hygiene would drop the
/// notices from the Rust distribution channel as quietly as `license-files` dropped them from the
/// wheel, and there is no equivalent of PEP 639 here to make it explicit.
#[test]
fn the_crate_package_does_not_exclude_the_notices() {
    let manifest = fs::read_to_string(root().join("Cargo.toml")).unwrap();
    let package = manifest
        .split("\n[")
        .next()
        .expect("Cargo.toml must have a [package] section");
    for key in ["exclude", "include"] {
        if let Some(pos) = package.find(&format!("{key} =")) {
            let block = &package[pos..(pos + 400).min(package.len())];
            for guarded in ["LICENSE", "THIRD_PARTY_NOTICES", "third_party"] {
                assert!(
                    !block.contains(guarded),
                    "Cargo.toml `{key}` mentions {guarded}; the notices must reach the .crate. \
                     If this is deliberate, the notices need an explicit `include` entry and this \
                     test needs to check for it instead."
                );
            }
        }
    }
}

/// Every source file carries an SPDX identifier.
///
/// Not decoration: it is what makes the licence of a file survive being copied out of the
/// repository into somewhere else, which is how most people actually encounter a snippet.
#[test]
fn every_source_file_declares_its_licence() {
    let checks: [(&str, &[&str]); 4] = [
        ("src", &["rs"]),
        ("tests", &["rs", "py"]),
        ("python", &["py"]),
        ("tools", &["py"]),
    ];
    let mut missing = Vec::new();
    let mut count = 0;
    for (dir, extensions) in checks {
        for path in sources(&root().join(dir), extensions) {
            count += 1;
            let text = fs::read_to_string(&path).unwrap_or_default();
            let head: String = text.chars().take(400).collect();
            if !head.contains("SPDX-License-Identifier") {
                missing.push(path.display().to_string());
            }
        }
    }
    assert!(count > 100, "the file walk found only {count} files");
    assert!(
        missing.is_empty(),
        "{} source files carry no SPDX identifier:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

/// Every compiled-in data file either carries its own provenance header or sits beside its licence.
///
/// These are the files that actually end up inside a binary, and a licence elsewhere in the
/// repository does not travel with one that is extracted and copied on. The parameter CSVs solve
/// that with a header naming their source. `BCCPARM.DAT` and `ATOMTYPE_BCC.DEF` cannot — a bare
/// numeric table has no comment syntax, and both are retained *verbatim*, so a header would mean
/// editing a file this project calls unmodified. They live under `third_party/antechamber/`
/// instead, next to the licence and the provenance note. Either route is acceptable; neither is
/// not, and `bccparm.dat` sat in `src/data/` with neither until 0.2.3.
#[test]
fn the_embedded_data_files_carry_their_provenance() {
    // Route one: a provenance header inside the file.
    for (file, must_mention) in [
        ("src/data/am1_parameters.csv", "PySEQM"),
        ("src/data/rm1_parameters.csv", "MOPAC"),
    ] {
        let text = fs::read_to_string(root().join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        let head: String = text.chars().take(2000).collect();
        assert!(
            head.contains(must_mention) || head.contains(&must_mention.to_uppercase()),
            "{file} does not name {must_mention} in its header; the provenance would not survive \
             the file being read on its own"
        );
    }

    // Route two: verbatim files, which live beside their licence instead. The assertion is that
    // they are *there* — a file that drifts back into `src/data/` has neither route.
    for file in [
        "third_party/antechamber/BCCPARM.DAT",
        "third_party/antechamber/ATOMTYPE_BCC.DEF",
    ] {
        assert!(
            root().join(file).is_file(),
            "{file} is missing; a verbatim data file must sit beside the licence covering it"
        );
    }

    // And nothing embedded is left in `src/data/` without a header. This is the check that would
    // have caught `bccparm.dat`, which sat there for three releases with no provenance in it and
    // none next to it.
    for entry in fs::read_dir(root().join("src/data")).unwrap().flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = fs::read_to_string(&path).unwrap_or_default();
        let head: String = text.chars().take(2000).collect();
        assert!(
            head.contains("PROVENANCE") || head.contains("Provenance"),
            "src/data/{name} carries no provenance header. Either give it one, or move it under \
             `third_party/<work>/` beside the licence that covers it if it must stay verbatim."
        );
    }
}

/// Claims about what CI checks are claims, and stale ones have shipped here before.
///
/// Three separate documents said CI verified the provenance notes were in the wheel while it
/// verified only the licences. This pins the prose to the workflow: if the check is weakened, the
/// sentence describing it has to be weakened too, in the same commit.
#[test]
fn the_documented_ci_check_is_the_one_that_runs() {
    let ci = fs::read_to_string(root().join(".github/workflows/ci.yml")).unwrap();
    assert!(
        ci.contains("THIRD_PARTY_NOTICES.md") && ci.contains("LICENSE"),
        "the CI wheel check no longer looks for the notices"
    );
    assert!(
        ci.contains("README.md"),
        "the CI wheel check no longer looks for the provenance notes, which \
         THIRD_PARTY_NOTICES.md, README.md and the third_party READMEs all say it does"
    );
}

/// Every crate that links into a shipped binary is named in `THIRD_PARTY_LICENSES.md`.
///
/// # The layer above the bundled data
///
/// `THIRD_PARTY_NOTICES.md` accounts for the parameter tables this project copies. It says nothing
/// about the code that gets *linked*, and that is the larger set by two orders of magnitude: over a
/// hundred crates end up statically inside `_native.pyd` and `am1_rs_cli`. Most are MIT, and MIT is
/// explicit that the copyright and permission notice must be included in "all copies or substantial
/// portions of the Software" — a statically linked binary being a copy. Apache-2.0 §4(a) says the
/// same of its licence text. Through 0.2.2 none of it was recorded anywhere.
///
/// This reads the direct dependencies out of `Cargo.toml` and checks each appears in the generated
/// file. It deliberately does not re-resolve the whole graph — that needs `cargo metadata`, which
/// is what `tools/collect_dependency_licenses.py --check` does and what CI runs. What this catches
/// is the common case: a dependency added to the manifest without regenerating.
#[test]
fn every_direct_dependency_appears_in_the_dependency_licences() {
    let licenses = fs::read_to_string(root().join("THIRD_PARTY_LICENSES.md"))
        .expect("THIRD_PARTY_LICENSES.md must exist; run tools/collect_dependency_licenses.py");
    assert!(
        licenses.len() > 50_000,
        "THIRD_PARTY_LICENSES.md is {} bytes — too short to hold a hundred licence texts",
        licenses.len()
    );

    let manifest = fs::read_to_string(root().join("Cargo.toml")).unwrap();
    let start = manifest
        .find("\n[dependencies]")
        .expect("Cargo.toml must have [dependencies]");
    let block = &manifest[start + 1..];
    let end = block[1..].find("\n[").map_or(block.len(), |i| i + 1);
    let mut names = Vec::new();
    for line in block[..end].lines().skip(1) {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, _)) = line.split_once('=') {
            names.push(name.trim().to_string());
        }
    }
    assert!(
        names.len() >= 3,
        "parsed only {names:?} out of [dependencies]; the parser has drifted from the manifest"
    );
    for name in names {
        assert!(
            licenses.contains(&format!("| `{name}` |")),
            "`{name}` is a dependency but is not in THIRD_PARTY_LICENSES.md; rerun \
             tools/collect_dependency_licenses.py"
        );
    }
}

/// The generated file says it is generated, and by what.
///
/// A file this size invites hand-editing, and a hand-edit is silently lost on the next
/// regeneration. The header is what tells a reader not to.
#[test]
fn the_dependency_licences_say_how_to_regenerate_them() {
    let licenses = fs::read_to_string(root().join("THIRD_PARTY_LICENSES.md")).unwrap();
    let head: String = licenses.chars().take(600).collect();
    assert!(
        head.contains("tools/collect_dependency_licenses.py"),
        "THIRD_PARTY_LICENSES.md does not name the tool that generates it"
    );
    assert!(
        root()
            .join("tools/collect_dependency_licenses.py")
            .is_file(),
        "the generator named in THIRD_PARTY_LICENSES.md does not exist"
    );
}
