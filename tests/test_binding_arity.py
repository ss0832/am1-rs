# SPDX-License-Identifier: GPL-3.0-or-later
"""Every wrapper in `am1_rs.native` forwards as many arguments as the binding declares.

# The bug this exists for

`native.dfpt` grew a `smearing_ev` parameter in 0.2.3, documented in the changelog and in the
docstring, and the forwarding call to `_native.dfpt` never passed it. Setting it did nothing, and
nothing said so — the wrapper accepted the keyword, the binding used its default, and the answer
was a calculation at zero smearing that the caller believed was smeared.

Nothing already in the suite could catch that. `test_result_contract.py` checks the *keys a result
carries*, and this result carried all of them. `test_cli_matrix.py` diffs the two front ends
against each other, and the CLI does not expose `smearing_ev` on this path. A dropped argument is
invisible to both: the call succeeds and returns a plausible number.

The check here is deliberately structural rather than behavioural. Asserting that some measurable
quantity moves when a parameter changes would need one test per parameter, each one a calculation,
and would still miss the parameter nobody wrote a test for. Counting what is forwarded catches the
whole class at once — including the next parameter someone adds to a signature and forgets to
pass on.
"""

import ast
import inspect
import io
import re
from pathlib import Path

import pytest

from am1_rs import native

ROOT = Path(__file__).resolve().parent.parent
PYTHON_RS = ROOT / "src" / "python.rs"

pytestmark = pytest.mark.skipif(
    not PYTHON_RS.is_file(), reason="binding source is not in an installed distribution"
)


def _parameters(signature_body: str) -> list[str]:
    """Parameter names in a `#[pyo3(signature = (...))]` body.

    Comments have to go first: `divide_conquer`'s signature carries a `//` note explaining why its
    defaults are what they are, and that prose contains commas. Splitting before stripping it
    counted the comment as an extra parameter, which is how this parser first reported a bug that
    was its own.
    """
    body = re.sub(r"//[^\n]*", "", signature_body)
    # Then defaults, so a tuple like `kpts=(2, 2, 2)` does not contribute commas either.
    body = re.sub(r"=\s*\([^)]*\)", "", body)
    body = re.sub(r'=\s*"[^"]*"', "", body)
    body = re.sub(r"=\s*[^,)]+", "", body)
    return [p.strip() for p in body.split(",") if p.strip()]


def _pyo3_arities() -> dict[str, int]:
    """Parameter count of each `#[pyo3(signature = (...))]`, keyed by the function it precedes."""
    src = io.open(PYTHON_RS, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"#\[pyo3\(signature = \((.*?)\)\)\]", src, re.S):
        tail = src[m.end():]
        name = re.search(r"\nfn ([a-z_0-9]+)\(", tail)
        if not name:
            continue
        params = _parameters(m.group(1))
        out[name.group(1)] = len(params)
    return out


def _forwarded_counts() -> dict[str, int]:
    """Positional arguments each wrapper passes to its `_native.<name>(...)` call."""
    tree = ast.parse(io.open(native.__file__, encoding="utf-8").read())
    out = {}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "_native"
                and node.func.attr == fn.name
            ):
                assert not node.keywords, (
                    f"{fn.name} forwards by keyword; this check counts positionals"
                )
                out[fn.name] = len(node.args)
    return out


def test_every_wrapper_forwards_the_whole_binding_signature():
    declared = _pyo3_arities()
    forwarded = _forwarded_counts()
    assert forwarded, "no `_native.*` calls found; the parser is looking at the wrong thing"

    short = {
        name: (n, declared[name])
        for name, n in forwarded.items()
        if name in declared and n != declared[name]
    }
    assert not short, (
        "these wrappers drop arguments the binding declares, so setting them does nothing: "
        + ", ".join(f"{k} forwards {v[0]} of {v[1]}" for k, v in sorted(short.items()))
    )


def test_the_wrapper_and_the_binding_agree_on_parameter_names():
    """A count alone would miss two parameters swapped, which is a worse bug than a dropped one."""
    src = io.open(PYTHON_RS, encoding="utf-8").read()
    mismatches = []
    for m in re.finditer(r"#\[pyo3\(signature = \((.*?)\)\)\]", src, re.S):
        tail = src[m.end():]
        found = re.search(r"\nfn ([a-z_0-9]+)\(", tail)
        if not found:
            continue
        name = found.group(1)
        wrapper = getattr(native, name, None)
        if wrapper is None or not callable(wrapper):
            continue
        rust = _parameters(m.group(1))
        py = list(inspect.signature(wrapper).parameters)
        # The wrapper is free to rename its *first* arguments (`numbers`/`positions` are reshaped
        # before forwarding) and to add its own; what must line up is the order of the rest.
        common = [p for p in rust if p in py]
        if [p for p in py if p in rust] != common:
            mismatches.append(name)
    assert not mismatches, f"parameter order differs between wrapper and binding: {mismatches}"
