# MOPAC — retained material and provenance

**MOPAC** (*Molecular Orbital PACkage*, J. J. P. Stewart) is the reference implementation of the
AM1 family. Material derived from it in this repository:

| file in this repository | upstream name | what it is |
|---|---|---|
| `src/data/rm1_parameters.csv` | `src/models/parameters_for_RM1_C.F90` | the RM1 per-element parameters, extracted from MOPAC's Fortran tabulation |
| `tools/extract_rm1_parameters.py` | — | the extractor itself: original to this project, retained so the extraction above is reproducible rather than a set of hand-copied numbers |
| `src/data_tables.rs` — `EHEAT_KCAL` | the same `eheat` quantity as `src/models/parameters_C.F90` | experimental atomic heats of formation. **Most values agree with MOPAC's; several do not** — see below |

### What was checked, and what it did and did not establish

`src/data_tables.rs` was added to this table in 0.2.3 after a mercury bug, on the following
reasoning: `EHEAT_KCAL` and `MASS` both skipped fourteen slots above caesium — mercury's mass sat
at index 66 and `MASS[80]` was `0.0` — and MOPAC omits exactly fourteen elements (La–Yb) for want
of NDDO parameters, so the tables looked like copies of a slot-indexed MOPAC array.

**That inference was wrong, and checking the upstream tree at the pinned commit is what showed
it.** MOPAC's `parameters_C.F90` is indexed by *atomic number*, not by parameter slot: `eheat(80)`
is mercury's 14.690 and `eheat(57)…eheat(71)` carry real lanthanide values. Nothing upstream has
the fourteen-slot gap. Comparing values as well as shape:

- **`MASS` is not MOPAC's table at all.** Twelve of fifteen sampled elements differ, and
  systematically: Ti 47.867 against MOPAC's 47.90, Tc **97.0** against 98.9062, Sn 118.71 against
  118.69, W 183.84 against 183.85, Os 190.23 against 190.20. Those are modern IUPAC standard
  atomic weights against MOPAC's older set. It has been removed from the table above.
- **`EHEAT_KCAL` mostly agrees but not everywhere.** Eighteen entries at `Z ≤ 86` differ from this
  commit, including sodium (25.850 against MOPAC's 25.650) and vanadium (122.300 against 122.900),
  besides the lanthanides. So it is not a copy of *this* release either. It is listed, at its
  original strength, as reproducing the same tabulated quantity; where the two disagree this
  crate's values are what its own reference tests are calibrated against.
- `N_S`, `N_P` and `QN` are the neutral-atom valence configuration and valence shell — periodic
  table facts with no MOPAC-specific content. They were listed here in error and are not.

The origin of the fourteen-slot gap is **not established**. It is not MOPAC's layout at this
commit; whether it came from an older MOPAC, from an intermediate source, or from a hand
transcription that skipped the lanthanide rows is unknown. It is recorded as unknown rather than
attributed to a source that has been checked and does not have it.

## Modifications (Apache-2.0 §4(b))

- **`src/data_tables.rs`, 0.2.3.** `EHEAT_KCAL` was **re-indexed by atomic number**: fourteen
  slots were reopened above caesium and entries 57–69 moved to the positions their elements
  actually occupy, which is the indexing MOPAC itself uses. The reopened lanthanide slots are left
  at zero; MOPAC has values there and this crate does not use them, since no parameter set covers
  those elements.
- **`src/data/rm1_parameters.csv`.** Reshaped from Fortran source into CSV by
  `tools/extract_rm1_parameters.py`; the values are unaltered. Only RM1's main-group set is
  extracted (see below).

No other material derived from MOPAC has been modified.

`rm1_parameters.csv` is **compiled into every binary and wheel** by `include_str!` from
`src/params.rs`; it is not read from disk at run time, so the copy in `src/data/` is the
distributed form of this material. Its header line carries the same provenance note as this file,
so the attribution travels with the data even when the file is read on its own.

MOPAC is additionally the **consolidating source** for the AM1 element-extension parameters, whose
values are published scientific facts from the papers cited below; MOPAC is where they are
collected in one place, and the specific machine-readable AM1 tabulation this crate ships came by
way of PySEQM rather than directly from MOPAC — see `third_party/pyseqm/README.md`.

**No MOPAC source code is ported**, and that is a statement about code, not about data. Where a
MOPAC routine is named in a comment (`diat.f`, `calpar.f`, `block.f`) it identifies the published
formula being implemented, not lines that were copied; the average-of-configuration coefficients
in `src/params.rs` are written as closed forms in `(n_s, n_p)` rather than as a transcribed table.
The **tables** in the third row above are a different matter: those are copies of MOPAC's arrays,
and they are listed as such.

## Where it came from

- Repository: <https://github.com/openmopac/mopac>, commit
  `052691223d19935a89f0fe18cd12301bd83e4201`.
- Project home: <https://openmopac.github.io/>.
- RM1, the method the extracted parameters belong to: G. B. Rocha, R. O. Freire, A. M. Simas &
  J. J. P. Stewart, "RM1: A Reparameterization of AM1 for H, C, N, O, P, S, F, Cl, Br and I,"
  *J. Comput. Chem.* **27**, 1101–1111 (2006).
- AM1, which RM1 reparameterizes: M. J. S. Dewar, E. G. Zoebisch, E. F. Healy & J. J. P. Stewart,
  *J. Am. Chem. Soc.* **107**, 3902–3909 (1985).

Only RM1's published main-group set (H, C, N, O, F, P, S, Cl, Br, I) is extracted. The lanthanide
parameters (Z = 57–71) need d/f orbitals and the sparkle model, neither of which this crate
implements; shipping them would produce a parameter block that loads and then computes nonsense.

## Licence

**Apache License 2.0**, Copyright 2021 Virginia Polytechnic Institute and State University. The
full text is retained here as `LICENSE`.

It is kept in this directory rather than referenced so that the requirement is met by anything that
unpacks only `third_party/`, and so that `pyproject.toml`'s
`license-files` glob ships it inside the wheel -- along with this README, since 0.2.3. CI checks
that **every** subdirectory of `third_party/` has both a `LICENSE` and a `README.md` and that
both reach the built wheel.

Apache-2.0 is compatible with GPL-3.0-or-later in the direction used here: Apache-2.0 material may
be incorporated into a GPL-3 work. See §6 of `THIRD_PARTY_NOTICES.md`.

The clause-by-clause position, since a licence that is merely "included" is not the same as one
that is complied with:

- **§4(a)** — a copy of the licence goes to every recipient: `LICENSE`, in this directory, shipped
  in the wheel and the sdist and checked by CI.
- **§4(b)** — modified files carry a prominent notice of the change: the **Modifications** section
  above, plus the doc comments on the tables themselves and the `CHANGELOG.md` entry.
- **§4(c)** — attribution notices are retained: the copyright statement above, and the provenance
  header in `src/data/rm1_parameters.csv`, which travels with the data because that file is what
  `include_str!` compiles in.
- **§4(d)** — **verified, and does not apply.** The clause binds only if the upstream Work includes
  a `NOTICE` text file. Checked against a working tree at commit
  `052691223d19935a89f0fe18cd12301bd83e4201`: `git ls-tree -r HEAD` lists no `NOTICE` at any path,
  and no such file exists in the checkout. There is therefore nothing for this clause to require.

  The upstream copyright notice is carried in per-file headers instead — every `.F90` opens with
  `Copyright 2021 Virginia Polytechnic Institute and State University` and the Apache boilerplate
  — which is what §4(c) is satisfied against. Our `LICENSE` is byte-identical to upstream's
  (SHA-256 `6D1D968F…`), so it is the licence as published, unannotated.

## A note on constants, because it changes numbers

This crate uses MOPAC7's historical physical constants (`a0 = 0.529167`, `1 au = 27.21 eV`,
`1 eV = 23.061 kcal/mol`), because the AM1 and RM1 parameters were fitted against them; modern
MOPAC defaults to CODATA values. Measured against MOPAC 22's own reference outputs, this leaves a
systematic **+0.03 kcal/mol** offset in the heat of formation — identically for AM1 and RM1 — while
optimized bond lengths agree to ~1e-4 Å and Koopmans ionization potentials to ~4e-4 eV.
`tests/mopac_reference.rs` measures this rather than asserting it away.
