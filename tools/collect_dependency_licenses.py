#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate `THIRD_PARTY_LICENSES.md` from the Rust dependency graph.

Why this exists
---------------
`THIRD_PARTY_NOTICES.md` covers the *data* this project bundles — parameter tables copied from
PySEQM, MOPAC and antechamber. It does not cover the *code* that gets linked in, and that is the
larger set: 112 crates end up statically linked inside `_native.pyd` and the `am1_rs_cli`
executable. Most are MIT, and MIT is explicit that "the above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software". A statically
linked binary is a copy. Apache-2.0 §4(a) says the same thing about its own licence text.

So a wheel shipped without these notices is out of compliance with several dozen licences at once,
and no amount of care about the parameter tables fixes that.

Why it is generated rather than written
---------------------------------------
A hand-maintained list is wrong the first time a dependency is added, updated or removed, and
nothing fails when it is. This reads the actual resolved graph and the actual licence files out of
the cargo registry cache, so the output cannot drift from what is really linked. Run it after any
dependency change; `tests/attribution.rs` checks that the file is present and covers the graph.

What it includes, and what it deliberately does not
---------------------------------------------------
* The **normal** dependency closure of this package. `dev-dependencies` do not link into anything
  shipped, and `build-dependencies` run on the builder's machine rather than being linked, so
  neither is a redistribution.
* Each distinct licence text **once**, with the list of crates it covers. Deduplication is by
  exact text, so two crates sharing a licence share an entry only when their copyright lines are
  also identical — which is what makes this compact without dropping a single copyright notice.

Usage
-----
    python tools/collect_dependency_licenses.py            # writes THIRD_PARTY_LICENSES.md
    python tools/collect_dependency_licenses.py --check    # exits non-zero if it is out of date
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "THIRD_PARTY_LICENSES.md"

#: Filenames a crate might keep its licence in. Ordered so the most specific wins in reporting.
LICENSE_NAMES = (
    "LICENSE-APACHE",
    "LICENSE-MIT",
    "LICENSE-ZLIB",
    "LICENSE-BSD",
    "LICENSE-BSL",
    "LICENSE.md",
    "LICENSE.txt",
    "LICENSE",
    "COPYING",
    "UNLICENSE",
    "LICENSE-THIRD-PARTY",
)


#: Canonical text for licences whose crates ship no file of their own.
#:
#: Only the *permission notice* — the part that is identical across every instance of the licence
#: and that MIT requires to accompany a copy. The copyright line is deliberately absent: it varies
#: per crate, upstream does not publish one in these tarballs, and writing a plausible one would be
#: inventing an attribution. The declared authors are listed in the table instead.
CANONICAL = {
    "MIT": """Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""",
}


def metadata() -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--all-features"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return json.loads(out.stdout.decode("utf-8-sig"))


def linked_closure(meta: dict) -> list[dict]:
    """Packages reachable from the root through `normal` dependency edges."""
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        for dep in nodes[current]["deps"]:
            # `kind: null` is a normal dependency. "dev" and "build" are not linked into a
            # shipped artifact and are excluded.
            if not any(k["kind"] is None for k in dep["dep_kinds"]):
                continue
            if dep["pkg"] not in seen:
                seen.add(dep["pkg"])
                stack.append(dep["pkg"])
    return sorted((packages[i] for i in seen), key=lambda p: (p["name"], p["version"]))


def license_texts(package: dict) -> list[tuple[str, str]]:
    """`(filename, text)` for every licence file shipped with the crate."""
    directory = pathlib.Path(package["manifest_path"]).parent
    found = []
    for name in LICENSE_NAMES:
        path = directory / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                found.append((name, text))
    return found


def render(packages: list[dict]) -> str:
    # Group crates by the exact text of each licence file, so a copyright line that differs
    # produces its own entry and one that does not is not repeated 40 times.
    by_text: dict[str, dict] = {}
    missing: list[dict] = []
    for package in packages:
        texts = license_texts(package)
        if not texts:
            missing.append(package)
            continue
        for name, text in texts:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            entry = by_text.setdefault(key, {"text": text, "crates": [], "names": set()})
            entry["crates"].append(f"{package['name']} {package['version']}")
            entry["names"].add(name)

    lines: list[str] = []
    add = lines.append
    add("# Third-party licences — linked Rust dependencies")
    add("")
    add("**Generated** by `tools/collect_dependency_licenses.py`. Do not edit by hand; rerun it")
    add("after any dependency change.")
    add("")
    add(
        "Every crate below is statically linked into `am1_rs._native` and into the `am1_rs_cli`"
    )
    add(
        "executable, which makes each shipped binary a copy of it. MIT requires its copyright and"
    )
    add(
        "permission notice to travel with such a copy; Apache-2.0 §4(a) requires its licence text."
    )
    add(
        "This file is what satisfies those, and it ships in the wheel, the sdist and the `.crate`."
    )
    add("")
    add(
        "Scope is the **normal** dependency closure. `dev-dependencies` are not linked into"
    )
    add(
        "anything distributed, and `build-dependencies` run on the builder's machine rather than"
    )
    add("being linked, so neither is redistributed here.")
    add("")
    add(f"**{len(packages)} crates**, {len(by_text)} distinct licence texts.")
    add("")
    add("## Crates")
    add("")
    add("| crate | version | SPDX | repository |")
    add("|---|---|---|---|")
    for package in packages:
        repo = package.get("repository") or ""
        spdx = package.get("license") or "(see licence file)"
        add(f"| `{package['name']}` | {package['version']} | {spdx} | {repo} |")
    add("")

    if missing:
        add("## Crates that publish no licence file")
        add("")
        add(
            "These declare a licence in `Cargo.toml` but ship no licence file inside the published"
        )
        add(
            "crate, so there is no upstream copyright line to reproduce verbatim. MIT asks for"
        )
        add(
            '"the above copyright notice **and** this permission notice"; the permission notice is'
        )
        add(
            "reproduced below, and the copyright holder is given as the crate's declared authors —"
        )
        add(
            "which is upstream's own metadata, not an attribution invented here. Where a crate"
        )
        add(
            "declares no author either, that is recorded as such rather than filled in."
        )
        add("")
        add("| crate | version | SPDX | declared authors | repository |")
        add("|---|---|---|---|---|")
        for package in missing:
            authors = ", ".join(package.get("authors") or []) or "*(none declared)*"
            repo = package.get("repository") or ""
            add(
                f"| `{package['name']}` | {package['version']} | {package.get('license')} "
                f"| {authors} | {repo} |"
            )
        add("")
        for spdx in sorted({p.get("license") or "" for p in missing}):
            canonical = CANONICAL.get(spdx)
            if canonical is None:
                add(
                    f"No canonical text is embedded for `{spdx}`; see the repositories above."
                )
                add("")
                continue
            covered = ", ".join(
                f"`{p['name']} {p['version']}`" for p in missing if p.get("license") == spdx
            )
            add(f"### Canonical {spdx} text")
            add("")
            add(f"Applies to: {covered}")
            add("")
            add("```")
            add(canonical)
            add("```")
            add("")

    add("## Licence texts")
    add("")
    for index, entry in enumerate(
        sorted(by_text.values(), key=lambda e: sorted(e["crates"])[0].lower()), start=1
    ):
        crates = ", ".join(f"`{c}`" for c in sorted(set(entry["crates"])))
        add(f"### {index}. {' / '.join(sorted(entry['names']))}")
        add("")
        add(f"Applies to: {crates}")
        add("")
        add("```")
        add(entry["text"])
        add("```")
        add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the committed file is out of date",
    )
    args = parser.parse_args()

    packages = linked_closure(metadata())
    rendered = render(packages)
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != rendered:
            print(
                f"{OUTPUT.name} is out of date; rerun tools/collect_dependency_licenses.py",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT.name} is up to date ({len(packages)} crates)")
        return 0
    OUTPUT.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.name}: {len(packages)} crates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
