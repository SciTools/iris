# Hashing Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the array-hashing machinery out of `lib/iris/_concatenate.py` into
a new private module `lib/iris/_combine_common.py`, so that `_merge.py` can reach
it without importing `_concatenate.py`.

**Architecture:** `_combine_common.py` is a flat private module, sibling to
`_merge.py`, `_concatenate.py`, `_combine.py` and `_lazy_data.py`. It sits
*below* both engines and imports nothing from Iris at runtime — the only Iris
names it needs are annotations on `array_id`, guarded by `TYPE_CHECKING`. That
import-freedom is the whole point of the module: it is what makes the merge-side
import in PR 4 risk-free. Everything here is a move; there is no behaviour
change whatsoever.

**Tech Stack:** Python 3.12+, NumPy, Dask, xxhash, pytest, Ruff, towncrier.

**Spec:** [`../specs/2026-09-10-merge-concatenate-design.md`](../specs/2026-09-10-merge-concatenate-design.md)
— this plan implements **row 1** of the roadmap in merge spec §6, per merge
spec §5.1.

## How this ships

**This plan ships as its own pull request, ahead of the implementation.** The
implementation pull request then carries only the code change under review.

The reason is reviewer attention, and it is worth stating plainly because it
governs every plan in this programme. A core developer opening the
implementation pull request should see a diff they can hold in their head: a
module moved, an import repointed, a test added. A four-hundred-line plan
describing how an agent was told to do it is not evidence about the change —
it is homework handed in alongside the work, and it makes a small, easily
approved diff look like a large one. Keeping the two apart means the
implementation pull request can be reviewed on its merits by someone who has
never read this file and does not need to.

Consequences for the executor:

- Programme bookkeeping — updating this plan, the roadmap row, the decision
  register — is **not** part of the implementation pull request, with one
  exception: Task 3's changelog fragment and roadmap row, which cannot be
  written before the implementation pull request number exists.
- If implementation turns up something that makes this plan wrong, fix the plan.
  A plan is frozen when its work merges, not while the work is being done. Put
  the fix on the plan branch, not the implementation branch, so the
  implementation diff stays clean.

## Global Constraints

- **The substrate imports nothing from Iris at runtime.** Annotations that need
  Iris names go behind `if TYPE_CHECKING:`, which requires
  `from __future__ import annotations`. (merge spec §5.1)
- **No behaviour change.** PR 1 is part of the proving tranche (PRs 1–4), which
  "end with a measured speedup against a committed baseline and no behaviour
  change whatsoever". Every existing test must pass unmodified except for the
  module path it imports from. (merge spec §6)
- **Names lose their underscore prefix** on the move. The module is already
  private; a leading underscore on every name inside it is noise, and
  `_lazy_data.py` is the precedent. (merge spec §6, row 1). This is **decision 6
  of the register in merge spec §6, and it is still open** — it makes PR 1 a move
  *and* a rename, which a reviewer may reasonably object to. The location is what
  matters, not the name, so the rename is cheap to concede. Say so in the pull
  request body rather than waiting to be asked.
- **Target branch is `upstream/greenfield`, never `main`.** Labels:
  `Feature: Merge/Concatenate`, `Type: Feature Branch`, `Agentic`. The pull
  request body must say plainly that the work is agentic and cite the spec row.
  (merge spec §6, cross-cutting rules)
- **Changelog fragments cite `` :user:`claude` ``**, never the human raising the
  pull request. (root `AGENTS.md`, "Contribution Workflow")
- **Ruff, 88-character lines.** Every new Python file starts with the four-line
  Iris copyright header.
- Line numbers in this plan are accurate as of `greenfield` at `7689b98f6`.

---

### Task 1: Create the substrate and move the hashing machinery

This task is atomic and cannot be split: creating the new module without
removing the original leaves duplicated definitions, and removing the original
first breaks `_concatenate.py`. The move is verbatim — the plan gives full text
for the genuinely new header, and an exact mechanical recipe for the block being
relocated, because retyping 237 lines of working code into a plan is pure
transcription risk.

**Files:**
- Create: `lib/iris/_combine_common.py`
- Create: `lib/iris/tests/unit/combine_common/__init__.py`
- Move: `lib/iris/tests/unit/concatenate/test_hashing.py` →
  `lib/iris/tests/unit/combine_common/test_hashing.py`
- Modify: `lib/iris/_concatenate.py` — delete lines 305–543; adjust imports at
  lines 7–25; update seven call sites at lines 604, 608, 623, 1141, 1229, 1230,
  1233

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `iris._combine_common` exporting

  ```python
  ArrayHash          # namedtuple("ArrayHash", ["value", "chunks"]) with __eq__
  array_id(coord: DimCoord | AuxCoord | AncillaryVariable | CellMeasure,
           bound: bool) -> str
  compute_hashes(arrays: Mapping[str, np.ndarray | da.Array])
                -> dict[str, ArrayHash]
  hash_array(a: da.Array | np.ndarray) -> np.int64
  ```

  plus module-internal `hash_ndarray`, `hash_chunk`, `hash_aggregate`.
  Task 2 relies on the module path `iris._combine_common`; PR 4 relies on all
  four exported names.

- [ ] **Step 1: Create the test package**

Create `lib/iris/tests/unit/combine_common/__init__.py`, matching the style of
the sibling `lib/iris/tests/unit/combine/__init__.py`:

```python
# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris._combine_common` module."""
```

- [ ] **Step 2: Move the test file and repoint it**

```bash
git mv lib/iris/tests/unit/concatenate/test_hashing.py \
       lib/iris/tests/unit/combine_common/test_hashing.py
```

Then make exactly these four edits to the moved file — nothing else changes, all
25 parametrised cases stay as they are:

| Line | From | To |
|---|---|---|
| 5 | `"""Test array hashing in :mod:`iris._concatenate`."""` | `"""Test array hashing in :mod:`iris._combine_common`."""` |
| 11 | `from iris import _concatenate` | `from iris import _combine_common` |
| 76, 90 | `_concatenate._compute_hashes(` | `_combine_common.compute_hashes(` |
| 95, 96, 103 | `_concatenate._ArrayHash(` | `_combine_common.ArrayHash(` |

- [ ] **Step 3: Run the moved tests to verify they fail**

Run: `pytest lib/iris/tests/unit/combine_common/test_hashing.py -x -q`
Expected: collection error — `ImportError: cannot import name '_combine_common'
from 'iris'`

- [ ] **Step 4: Create the substrate module**

Create `lib/iris/_combine_common.py` with exactly this header:

```python
# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Machinery shared by cube merge and cube concatenate.

This module is the substrate beneath :mod:`iris._merge` and
:mod:`iris._concatenate`.  It deliberately imports nothing from Iris at
runtime, so that either engine may import it without any risk of a circular
import.  The only Iris names it requires are annotations, which are guarded
by :data:`typing.TYPE_CHECKING`.

"""

from __future__ import annotations

from collections import namedtuple
import itertools
from typing import TYPE_CHECKING, Any

import dask
import dask.array as da
import numpy as np
from xxhash import xxh3_64

if TYPE_CHECKING:
    from collections.abc import Mapping

    from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord

# Restrict the names imported from this namespace.
__all__ = ["ArrayHash", "array_id", "compute_hashes", "hash_array"]
```

`Mapping` is annotation-only, and `from __future__ import annotations` is what
makes that fact visible to Ruff — so `TC003` demands it sit behind
`TYPE_CHECKING` alongside the Iris imports.  `_concatenate.py` has no `__future__`
import, so the rule never fires there and the original module scopes `Mapping`
normally; copying that import across unchanged fails lint.

Then append lines 305–541 of `lib/iris/_concatenate.py` **verbatim** — from
`def _hash_ndarray(a: np.ndarray) -> np.ndarray:` through
`return {k: _ArrayHash(*v) for k, v in hashes.items()}` — and apply exactly
these seven renames throughout the appended block, definitions and call sites
alike:

| From | To |
|---|---|
| `_hash_ndarray` | `hash_ndarray` |
| `_hash_chunk` | `hash_chunk` |
| `_hash_aggregate` | `hash_aggregate` |
| `_hash_array` | `hash_array` |
| `_ArrayHash` | `ArrayHash` |
| `_array_id` | `array_id` |
| `_compute_hashes` | `compute_hashes` |

`ArrayHash` keeps its `namedtuple("ArrayHash", ...)` typename, which was already
unprefixed. Do not otherwise touch the moved code — not the docstrings, not the
comments, not the `__eq__` that raises. Whether `ArrayHash.__eq__` should raise
at all is an open decision owned by PR 4 (merge spec §6, decision 9); this PR
changes nothing about it.

There is one exception, forced by the `_array_id` → `array_id` rename.  Three
lines inside `compute_hashes` already bind a *local* called `array_id`, which
coexisted with a function called `_array_id` but shadows one called `array_id`.
No scope in the module calls the function, so the shadowing is inert — but it is
the same trap Step 7 defuses in `_concatenate.py`, and leaving it in the module
that *defines* the name is worse than leaving it anywhere else.  Rename those
three, and only those three:

| Original | Becomes |
|---|---|
| `array_id, a = item` (in `group_key`) | `_, a = item` |
| `for array_id, rechunked in zip(array_ids, rechunked_arrays):` | `for key, rechunked in zip(array_ids, rechunked_arrays):` |
| `hashes[array_id] = (hash_array(rechunked), chunks)` | `hashes[key] = (hash_array(rechunked), chunks)` |

`array_ids` (plural) does not collide and stays as it is.  These three lines are
the whole of the difference between the relocated code and the original modulo
the seven renames, which is what Verification 1 checks.

- [ ] **Step 5: Run the moved tests to verify they pass**

Run: `pytest lib/iris/tests/unit/combine_common/test_hashing.py -q`
Expected: `59 passed` — the same count the file gives on `greenfield` before the
move (25 `test_compute_hashes` cases, 32 `test_compute_hashes_vs_array_equal`
cases drawn from `iris.tests.unit.util.test_array_equal.TEST_CASES`, and the two
`raises` tests). A different count means the parametrisation was disturbed.

- [ ] **Step 6: Delete the moved block from `_concatenate.py`**

Delete lines 305–543 inclusive — the block plus its two trailing blank lines,
which leaves the two blank lines at 303–304 as the separator before
`def concatenate(`.

- [ ] **Step 7: Repoint `_concatenate.py` at the substrate**

In the import block, delete these four now-dead imports:

```python
import itertools                    # line 9  - only used by _compute_hashes
from typing import Any              # line 10 - only used by _ArrayHash.__eq__
import dask                         # line 13 - only used by _compute_hashes
from xxhash import xxh3_64          # line 16 - only used by _hash_ndarray
```

Keep `import dask.array as da` (still used at lines 1571–1573) and keep
`from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord`
(still used at lines 605, 1228, 1232). Add, in Ruff's import order:

```python
from iris._combine_common import ArrayHash, array_id, compute_hashes
```

Then update the seven remaining references:

| Line | From | To |
|---|---|---|
| 604 | `array_id = _array_id(coord, bound=False)` | `points_id = array_id(coord, bound=False)` |
| 606 | `arrays[array_id] = coord.core_points()` | `arrays[points_id] = coord.core_points()` |
| 608 | `bound_array_id = _array_id(coord, bound=True)` | `bounds_id = array_id(coord, bound=True)` |
| 609 | `arrays[bound_array_id] = coord.core_bounds()` | `arrays[bounds_id] = coord.core_bounds()` |
| 611 | `arrays[array_id] = coord.core_data()` | `arrays[points_id] = coord.core_data()` |
| 623 | `hashes = _compute_hashes(arrays)` | `hashes = compute_hashes(arrays)` |
| 1141 | `hashes: Mapping[str, _ArrayHash],` | `hashes: Mapping[str, ArrayHash],` |
| 1229 | `) -> tuple[_ArrayHash, ...]:` | `) -> tuple[ArrayHash, ...]:` |
| 1230 | `array_id = _array_id(coord, bound=False)` | `points_id = array_id(coord, bound=False)` |
| 1231 | `result = [hashes[array_id]]` | `result = [hashes[points_id]]` |
| 1233 | `bound_array_id = _array_id(coord, bound=True)` | `bounds_id = array_id(coord, bound=True)` |
| 1234 | `result.append(hashes[bound_array_id])` | `result.append(hashes[bounds_id])` |

**Name collision — this is the one non-mechanical part of the move.** Lines 604
and 1230 currently assign to a *local variable* called `array_id`. After the
rename that name is also the imported function, and a local assignment makes it
local for the whole scope, so the call on the right-hand side raises
`UnboundLocalError` at run time. Ruff does not flag it and the type checker does
not flag it; only the tests do. Rename the locals to `points_id` and `bounds_id`
in both enclosing scopes, so that the two helpers read in full as:

At `add_coords` (`_concatenate.py:601-611` before the deletion, `364-374`
after), the body becomes:

```python
    def add_coords(cube_signature: _CubeSignature, coord_type: str) -> None:
        for coord_and_dims in getattr(cube_signature, coord_type):
            coord = coord_and_dims.coord
            points_id = array_id(coord, bound=False)
            if isinstance(coord, (DimCoord, AuxCoord)):
                arrays[points_id] = coord.core_points()
                if coord.has_bounds():
                    bounds_id = array_id(coord, bound=True)
                    arrays[bounds_id] = coord.core_bounds()
            else:
                arrays[points_id] = coord.core_data()
```

At the `_ProtoCube.register` helper (`_concatenate.py:1226-1235` before the
deletion), the body becomes:

```python
        def get_hashes(
            coord: DimCoord | AuxCoord | AncillaryVariable | CellMeasure,
        ) -> tuple[ArrayHash, ...]:
            points_id = array_id(coord, bound=False)
            result = [hashes[points_id]]
            if isinstance(coord, (DimCoord, AuxCoord)) and coord.has_bounds():
                bounds_id = array_id(coord, bound=True)
                result.append(hashes[bounds_id])
            return tuple(result)
```

Confirm the exact pre-edit text of that second helper before rewriting it:

Run: `sed -n '1226,1236p' lib/iris/_concatenate.py`

- [ ] **Step 8: Verify no stale references remain**

Run:

```bash
grep -rn '_hash_ndarray\|_hash_chunk\|_hash_aggregate\|_hash_array\|_ArrayHash\|_array_id\|_compute_hashes' \
  lib/ benchmarks/ docs/src --include='*.py' --include='*.rst' --include='*.md' \
  | grep -v docs/src/_build
```

Expected: no output.

- [ ] **Step 9: Run the concatenate suite**

Run: `pytest -n auto lib/iris/tests/unit/concatenate lib/iris/tests/unit/combine_common lib/iris/tests/unit/combine -q`
Expected: PASS, no failures, no errors.

- [ ] **Step 10: Run lint and format**

Run:

```bash
ruff check lib/iris && ruff format lib/iris && ruff format --check lib/iris
```

Expected: all pass. Re-run Step 9 if `ruff format` rewrote anything.

- [ ] **Step 11: Commit**

```bash
git add lib/iris/_combine_common.py lib/iris/_concatenate.py \
        lib/iris/tests/unit/combine_common/
git commit -m "$(cat <<'EOF'
Move array hashing into a shared _combine_common substrate

Lifts the array-hashing machinery out of _concatenate.py into a new private
module, lib/iris/_combine_common.py, so that _merge.py can reach it without
importing _concatenate.py.  The substrate imports nothing from Iris at
runtime; the coordinate annotations on array_id sit behind TYPE_CHECKING.

Names lose their underscore prefix on the move, the module itself already
being private.  The two local variables that would have shadowed the imported
array_id are renamed to points_id and bounds_id.

Pure relocation: no behaviour change.  Row 1 of the roadmap in
docs/src/developers_guide/specs/2026-09-10-merge-concatenate-design.md.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Lock the no-Iris-imports invariant with a test

The substrate's import-freedom is an assertion the spec makes (merge spec §5.1)
and the reason the module exists at all. It is also exactly the kind of property
that rots silently: any future contributor adding `from iris.cube import Cube`
for a type hint would break PR 4's premise without a single test failing. This
task makes that a test failure.

A runtime check is not available — `import iris._combine_common` executes
`iris/__init__.py` first, so `sys.modules` cannot distinguish the substrate's own
imports from the package's. The check is therefore static, over the module's AST.

**Files:**
- Create: `lib/iris/tests/unit/combine_common/test_imports.py`

**Interfaces:**
- Consumes: the module path `iris._combine_common` from Task 1.
- Produces: nothing later tasks rely on.

- [ ] **Step 1: Write the failing test**

Create `lib/iris/tests/unit/combine_common/test_imports.py`:

```python
# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the import-freedom of :mod:`iris._combine_common`."""

import ast
from pathlib import Path

from iris import _combine_common


class RuntimeImports(ast.NodeVisitor):
    """Collect modules imported at runtime, skipping ``TYPE_CHECKING`` blocks."""

    def __init__(self):
        self.modules: set[str] = set()

    def visit_If(self, node: ast.If) -> None:
        guard = node.test
        type_checking = (isinstance(guard, ast.Name) and guard.id == "TYPE_CHECKING") or (
            isinstance(guard, ast.Attribute) and guard.attr == "TYPE_CHECKING"
        )
        if type_checking:
            # Only the ``else`` branch runs at runtime.
            for statement in node.orelse:
                self.visit(statement)
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.modules.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            # A relative import is necessarily an Iris import.
            self.modules.add("." * node.level + (node.module or ""))
        elif node.module is not None:
            self.modules.add(node.module)


def test_no_runtime_iris_imports():
    """The substrate sits below merge and concatenate, so imports no Iris."""
    source = Path(_combine_common.__file__).read_text(encoding="utf-8")
    visitor = RuntimeImports()
    visitor.visit(ast.parse(source))
    offenders = {
        module
        for module in visitor.modules
        if module == "iris" or module.startswith(("iris.", "."))
    }
    assert offenders == set(), (
        "iris._combine_common must import nothing from Iris at runtime; "
        f"found {sorted(offenders)}. Put annotation-only imports behind "
        "`if TYPE_CHECKING:`."
    )


def test_type_checking_imports_are_detected():
    """Guard the guard: a runtime Iris import must be seen as one."""
    source = "\n".join(
        [
            "from typing import TYPE_CHECKING",
            "if TYPE_CHECKING:",
            "    from iris.coords import DimCoord",
            "from iris.cube import Cube",
        ]
    )
    visitor = RuntimeImports()
    visitor.visit(ast.parse(source))
    assert "iris.cube" in visitor.modules
    assert "iris.coords" not in visitor.modules
```

The second test is not ceremony. Without it, a `RuntimeImports` visitor that
silently collected nothing at all — the classic way an AST check dies — would
leave the first test passing forever and the invariant unguarded.

- [ ] **Step 2: Run the test to verify the guard test fails first**

Temporarily add `import iris.cube` at the top of `lib/iris/_combine_common.py`,
below the copyright header.

Run: `pytest lib/iris/tests/unit/combine_common/test_imports.py -q`
Expected: `test_no_runtime_iris_imports` FAILS with
`AssertionError: iris._combine_common must import nothing from Iris at runtime;
found ['iris.cube']`, and `test_type_checking_imports_are_detected` PASSES.

- [ ] **Step 3: Remove the temporary import and re-run**

Delete the `import iris.cube` line just added.

Run: `pytest lib/iris/tests/unit/combine_common/test_imports.py -q`
Expected: PASS, 2 passed.

- [ ] **Step 4: Lint and commit**

```bash
ruff check lib/iris && ruff format --check lib/iris
git add lib/iris/tests/unit/combine_common/test_imports.py
git commit -m "$(cat <<'EOF'
Test that the combine substrate imports nothing from Iris

The import-freedom of _combine_common is the reason the module exists, and
is the premise of the merge-side import that follows in a later pull request.
It is also the kind of property that rots without a test: a type hint added
carelessly would break it in silence.

Checks the module's AST rather than sys.modules, since importing a submodule
of iris necessarily executes iris/__init__.py first.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Open the pull request, then land the fragment and roadmap row

A towncrier fragment is named `<PR-number>.<type>.rst`, and the roadmap row cites
`{pull}`NNNN``. Neither number exists until the pull request is open, so this
task deliberately runs last and spans the push.

**Files:**
- Create: `changelog/<PR>.internal.rst`
- Modify: `docs/src/developers_guide/specs/2026-09-10-merge-concatenate-design.md`
  — roadmap row 1 (merge spec §6)

**Interfaces:**
- Consumes: the commits from Tasks 1–2.
- Produces: nothing.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin pr1-hashing-substrate
```

- [ ] **Step 2: Open the pull request against `greenfield`**

```bash
gh pr create --repo SciTools/iris --base greenfield \
  --head bjlittle:pr1-hashing-substrate \
  --title "[Agentic] Move array hashing into a shared _combine_common substrate" \
  --body-file -
```

Body must:

- State plainly that the work is agentic.
- Cite the spec and its row — "row 1 of the roadmap in the merge/concatenate
  design spec, per §5.1" — and reference `#7274` for the spec itself.
- Confirm the change is a pure relocation with no behaviour change, and point at
  the two verification checks below.
- Flag the `array_id` local-variable renames as the only non-mechanical part of
  the move — the five call sites in `_concatenate.py` and the three shadowed
  bindings inside `compute_hashes` — so a reviewer knows where to look. Nothing
  else in the diff needs reading closely, and saying so is what makes the rest
  of the review cheap.
- **Put decision 6 to the reviewer explicitly**: dropping the underscore prefix
  makes this a move *and* a rename. Offer to drop the rename and keep the
  underscores if they would rather review a pure `git mv`. Conceding costs
  nothing — the module's location is what the rest of the programme depends on.
  Note that keeping `_array_id` would also retire all eight local renames above,
  since every one of them exists only to get out of the unprefixed name's way.
Keep the body about the code. The plan, the register and the programme
bookkeeping ship separately — see "How this ships" above.

- [ ] **Step 3: Apply the labels**

```bash
PR=$(gh pr view --repo SciTools/iris --json number --jq .number \
     --head bjlittle:pr1-hashing-substrate)
gh api -X POST "repos/SciTools/iris/issues/${PR}/labels" \
  -f 'labels[]=Feature: Merge/Concatenate' \
  -f 'labels[]=Type: Feature Branch' \
  -f 'labels[]=Agentic'
```

Use `gh api` rather than `gh pr edit --add-label`: the repository's labeler
workflow races `gh pr edit` and the labels do not stick. Verify with
`gh pr view "$PR" --repo SciTools/iris --json labels`.

- [ ] **Step 4: Write the changelog fragment**

Create `changelog/<PR>.internal.rst` with the real number substituted:

```rst
:user:`claude` moved the array hashing machinery shared by
:meth:`~iris.cube.CubeList.merge` and :meth:`~iris.cube.CubeList.concatenate`
into a new private ``iris._combine_common`` module, in preparation for merge
adopting it. No behaviour change.
```

- [ ] **Step 5: Update roadmap row 1**

In `docs/src/developers_guide/specs/2026-09-10-merge-concatenate-design.md`,
change row 1 of the §6 table's final column from `not started` to
`✅ complete ({pull}`NNNN`)`, substituting the real number. Change nothing else
in the table — rows 2–12 are updated by their own pull requests.

Leave decisions 6 and 7 in the register alone. Both are owned by PR 1 but
resolved by the *reviewers' response* to it, not by opening it: 6 is whether the
rename survives review, 7 is whether the programme proceeds at all. Record their
outcomes in a follow-up once that response exists.

- [ ] **Step 6: Verify the fragment and the docs build**

```bash
towncrier build --draft --version 3.17.0 | grep -A3 '_combine_common'
cd docs && make html
```

Expected: the fragment renders under "💼 Internal"; the docs build succeeds with
no new warnings.

- [ ] **Step 7: Run the full unit suite before handing over for review**

Run: `pytest -n auto lib/iris/tests/unit -q`
Expected: PASS. Any failure here is a genuine regression — this pull request
changes no behaviour.

- [ ] **Step 8: Commit and push**

```bash
git add changelog/ docs/src/developers_guide/specs/
git commit -m "$(cat <<'EOF'
Add changelog fragment and mark roadmap row 1 complete

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Verification

The whole of PR 1 is a relocation, so the strongest evidence that it is correct
is that nothing changed. Two checks make that concrete, and both should be
quoted in the pull request body:

1. **The moved code is byte-identical apart from the renames.** From the branch:

   ```bash
   git show upstream/greenfield:lib/iris/_concatenate.py | sed -n '305,541p' \
     | sed -e 's/\b_hash_ndarray\b/hash_ndarray/g' -e 's/\b_hash_chunk\b/hash_chunk/g' \
           -e 's/\b_hash_aggregate\b/hash_aggregate/g' -e 's/\b_hash_array\b/hash_array/g' \
           -e 's/\b_ArrayHash\b/ArrayHash/g' -e 's/\b_array_id\b/array_id/g' \
           -e 's/\b_compute_hashes\b/compute_hashes/g' > /tmp/expected.py
   sed -n '/^def hash_ndarray/,$p' lib/iris/_combine_common.py > /tmp/actual.py
   diff -u /tmp/expected.py /tmp/actual.py
   ```

   Expected: the three de-shadowing lines from Step 4 and nothing else — two
   hunks, three changed lines. Anything further is unintended; if `ruff format`
   reflowed something, the diff shows exactly what and why, so inspect it rather
   than accepting it.

   Within lines 305–541 the `\b` word boundaries change nothing, because no one
   of the seven names is a substring of another. They are there for the moment
   someone widens the range to take in the call sites, where `bound_array_id`
   *does* contain `_array_id` and an unanchored substitution would quietly
   produce `boundarray_id`.

2. **The concatenate suite passes unmodified.** No test under
   `lib/iris/tests/unit/concatenate/` is edited by this pull request; only
   `test_hashing.py` moves out of it.

   ```bash
   git diff upstream/greenfield --stat -- lib/iris/tests/unit/concatenate/
   ```

   Expected: `test_hashing.py` shown as deleted, nothing else touched.
