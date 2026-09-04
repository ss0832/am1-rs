# Methods: AM1 and RM1

Both methods share the same NDDO functional form, so they share the same code path entirely —
only the parameters differ. Select with `method="am1"` (default) or `method="rm1"` on every
Python entry point, or `Am1Parameters::for_method(NddoMethod::Rm1)` in Rust.

```python
am1_rs.single_point(numbers, positions, method="rm1")
atoms.calc = AM1(method="rm1")
```

```rust
let params = Am1Parameters::for_method(NddoMethod::Rm1)?;
```

The method travels **with the parameter set**, not in the options struct. That is deliberate:
every downstream branch — the core–core correction terms, element coverage, the method name in
error messages — follows from one argument, and there is no way to pair RM1 parameters with an
AM1 code path.

---

## Element coverage

| | Elements |
|---|---|
| **AM1** | H, Be, B, C, N, O, F, Al, Si, P, S, Cl, Zn, Ge, As, Se, Br, Sb, Te, I, Hg (21) |
| **RM1** | H, C, N, O, P, S, F, Cl, Br, I (10) |

Asking for an element the chosen method does not parameterize is an error naming both the
element and the method, rather than a silent fallback.

Note that **Be and B are parameterized for AM1** and were previously undocumented. The
`zeta_d` column in the parameter CSVs is read but unused: this implementation has no `d`
orbitals, and the column is retained so the files stay comparable with published tables.

---

## The functional form

Both methods are NDDO with the Dewar–Sabelli–Klopman multipole expansion for the two-centre
two-electron integrals, connected by the Klopman–Ohno kernel

```
γ = 1 / √(R² + (ρ_a + ρ_b)²)
```

and both add the same Gaussian core–core correction on top of the MNDO core repulsion:

```
E_core(A,B) = Z_A Z_B γ_AB (1 + f_A + f_B)  +  (Z_A Z_B / R) Σ_i K_i exp(−L_i (R − M_i)²)
```

with the MNDO N–H and O–H special cases using `R·e^{−αR}` in place of `e^{−αR}` in `f`. AM1
gives carbon four Gaussians and nitrogen three; RM1 refits all of them.

Because the form is identical, everything built on it — analytic gradients, CPHF Hessians,
periodic boundary conditions with k-points, analytic stress, divide-and-conquer — works
unchanged for both.

---

## RM1

Rocha, Freire, Simas & Stewart, *J. Comput. Chem.* **27**, 1101 (2006).

RM1 ("Recife Model 1") is a reparameterization of AM1 against a larger and more carefully
curated reference set, targeting heats of formation, dipole moments, ionization potentials and
geometries for the ten main-group elements most common in organic and biological chemistry.

The bundled parameters are extracted from MOPAC's RM1 parameter block; see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) §3 for the provenance, the licence and
the exact commit, and `tools/extract_rm1_parameters.py` for the extraction.

RM1 and AM1 differ modestly per parameter but meaningfully in aggregate — carbon's `U_ss`, for
instance, differs by 0.30 eV — and give clearly different energies. `tests/rm1.rs` checks the
parameter set against published values and the resulting properties against MOPAC.

---

## SAM1

**Not available in 0.2.0.** SAM1 (Dewar, Jie & Yu, *Tetrahedron* **49**, 5003 (1993)) replaces
the multipole expansion with two-centre integrals computed from an STO-3G Gaussian basis and
then scaled, so it is a genuinely different engine rather than a reparameterization — it does
not fit the shared code path that AM1 and RM1 share. It is deferred to 0.3.0.

---

## Absolute accuracy, and the constants

This implementation deliberately uses **MOPAC7's non-CODATA constants** (`ev = 27.21`,
`a0 = 0.529167 Å`) so that results match MOPAC rather than being marginally more "correct" in
isolation. Mixing CODATA values into new code would quietly break that agreement.

Against MOPAC 22, on water: optimized geometry to 5 × 10⁻⁵ Å, Koopmans ionization potential to
4 × 10⁻⁴ eV. The heat of formation carries a constant **+0.03 kcal/mol** offset which comes
from the constants choice, not from the model — AM1 and RM1 show the identical offset, which is
what identifies it.
