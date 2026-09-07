# antechamber (AmberTools) — retained material and provenance

Two files in this repository come from **antechamber**, the small-molecule parameterization tool
distributed with **AmberTools**:

| file in this repository | upstream name | what it is |
|---|---|---|
| `third_party/antechamber/BCCPARM.DAT` | `BCCPARM.DAT` | the 405 AM1-BCC bond charge corrections, verbatim |
| `third_party/antechamber/ATOMTYPE_BCC.DEF` | `ATOMTYPE_BCC.DEF` | the AM1-BCC atom-type definition rules, verbatim |

Both are **compiled into every binary and wheel** by `include_str!` — `BCCPARM.DAT` from
`src/bcc/mod.rs` and `ATOMTYPE_BCC.DEF` from `src/bcc/atomtype.rs`, which interprets its rules
rather than transcribing them. Neither is read from disk at run time, so the copies here are the
distributed source of both.

`BCCPARM.DAT` moved here from `src/data/` in 0.2.3. It had been the one embedded data file with
no provenance anywhere near it: the two parameter CSVs carry a header naming their source, and a
bare numeric table has no comment syntax to put one in. Adding one would have meant either editing
a file this document calls verbatim or teaching the parser to skip something upstream does not
have. Moving it beside the licence and this README achieves the same end and leaves the bytes
untouched — which is what `ATOMTYPE_BCC.DEF` was already doing.

## Where they came from

The files were obtained from the redistribution in the
[`choderalab/ambermini`](https://github.com/choderalab/ambermini) package, which packages the
AmberTools 14 vintage of antechamber.

- antechamber: J. Wang, W. Wang, P. A. Kollman & D. A. Case, "Automatic atom type and bond type
  perception in molecular mechanical calculations," *J. Mol. Graph. Model.* **25**, 247 (2006).
- AmberTools: D. A. Case *et al.*, *AMBER*, University of California, San Francisco.
- The AM1-BCC method the parameters belong to: A. Jakalian, B. L. Bush, D. B. Jack & C. I. Bayly,
  *J. Comput. Chem.* **21**, 132 (2000) and **23**, 1623 (2002).

## Licence

This material is under the **GNU General Public License, version 3**. The full text is retained
here as `LICENSE`.

That file is the canonical GPL-3.0 text — the same document this project's own top-level `LICENSE`
carries, and it is byte-identical to it. It is duplicated here rather than referenced so that the
requirement is met by anything that unpacks only `third_party/`, and so that
`pyproject.toml`'s `license-files` glob picks it up and ships it inside the wheel, together with
this README. CI checks that every `third_party/` subdirectory carries both and that both reach
the built wheel.

**On "or later".** The `LICENSE` accompanying the ambermini redistribution is the plain GPL-3 text
and contains no separate "or (at your option) any later version" grant, so this project does not
claim one on Amber's behalf. (conda-forge's `ambertools` feedstock declares modern AmberTools as
`GPL-3.0-or-later`; that is a packager's declaration about a later release, not a statement about
the files retained here.) Taking the conservative reading — GPL-3.0 without an "or later" grant —
the combined work is distributable under GPL-3.0, which `am1-rs`'s own `GPL-3.0-or-later` permits.
See §6 of `THIRD_PARTY_NOTICES.md`.
