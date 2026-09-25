# AGENTS.md

Agent instructions for `lib/iris/` — the Iris library source.

These rules govern **how code is written** under this tree. Tests have their
own rules in [`tests/AGENTS.md`](tests/AGENTS.md).

For **where code lives and how data flows**, read
[`ARCHITECTURE.md`](ARCHITECTURE.md) before exploring unfamiliar subsystems.
It is an orientation map only; algorithm detail belongs in module docstrings.


## Why This File Exists

Iris is maintained by humans and agents together. Agents read through a
narrow context window using text search and cannot run the code; humans read
with intuition but limited patience. Both want explicit names, local
reasoning and small units. This file records that shared style, plus the few
places the two diverge and which way to resolve them. Models are trained to
please diff reviewers rather than to produce code that is cheap to navigate
later, and drift towards long functions, redundant comments and defensive
wrapping; correct for that deliberately.


## Fast Rules

1. Write code that can be understood from the file in front of you.
2. Name things so they can be found by exact text search.
3. Prefer explicit and slightly verbose over implicit and clever.
4. Prefer duplication over an abstraction you are not confident in.
5. Match surrounding Iris idioms even where you would choose differently.
6. Every comment and docstring must still be true after your edit.
7. Explain subsystems in module docstrings, not inline or in side files.


## Locality — Readable In One Place

- Keep behaviour derivable from the function's own body, signature, type
  hints and docstring. Avoid designs that require opening three other files.
- Prefer composition and explicit delegation over deep inheritance and mixin
  stacks. Resolving an MRO across modules costs an agent many tool calls and
  is opaque to a newcomer. Where Iris already uses mixins (e.g.
  `iris.common.mixin`, metadata managers), follow the existing pattern rather
  than inventing a parallel one.
- Do not add indirection (registry, dispatch dict, plugin hook, base class)
  for a single caller. This governs implementation layers, not API surface;
  a package `__init__.py` that re-exports is not indirection, see below.
- Pass dependencies in as arguments rather than reaching for module-level
  mutable state.


## Greppability

Exact text search is the primary retrieval tool. Code that cannot be found
does not exist.

- Never construct identifiers at runtime: no `setattr(obj, f"{name}_bounds",
  ...)`, no `getattr(module, method_name)` dispatch where a dict or `if/elif`
  would do. A constructed name is unfindable for every reader. This bans
  *computed* names only — `getattr(var, "cf_role", "")` is a safe lookup with
  a greppable literal and is the preferred idiom for optional CF attributes.
- Give public names enough distinctiveness to search for. Local `data`,
  `cube`, `result` are fine; a public helper called `_process` is not.
- Declare `__all__` in modules with a public surface. No `import *`.
- **Re-export freely from a package `__init__.py`.** Surfacing objects at
  the level users import from is a real convenience, and it keeps the file
  layout free to change later. `iris.mesh` is the model: explicit
  `from .components import MeshXY`, gathered into `__all__`. Re-export
  verbatim — never rename on the way out, never wrap in `try`/conditional
  imports, never put logic in `__init__.py`. Each object stays defined in
  one module, so `class MeshXY` still finds it in a single grep.
  (`iris.common` uses `import *`; that is legacy, not a pattern to copy.)
- Make error and warning messages distinctive and mostly static, so a
  traceback maps to exactly one line. Put interpolated values at the end:
  `f"Cannot collapse a coordinate with bounds: {coord.name()}"`.
- Do not reuse one helper name across modules for differing behaviour.


## Explicitness

- Add type hints to new or modified public functions. Agents cannot execute
  code to discover types, so hints are the cheapest reliable signal. Do not
  retrofit hints to code you are not otherwise changing.
- Avoid `**kwargs` passthrough on public API — spell out the parameters. If
  passthrough is unavoidable, document the accepted keys.
- Use keyword arguments at call sites for anything not obviously positional.
  Booleans are always keyword.
- Replace repeated magic strings with module-level constants.
- Return one documented type. Do not write functions whose return type
  depends on an argument's value. Iris has some legacy examples; add no more.
- State laziness in the docstring: whether the result is lazy, and whether
  the call realises data.


## Conform to the Specification, Not to the Sample File

Iris implements published conventions — CF, UGRID, Zarr, netCDF. Sample files
are evidence that a code path gets exercised, not authority for what it should
do. Derive behaviour from the convention's text and cite the section; before
claiming a file taught you something general, check whether the convention
already says it. When a real file disagrees with the convention, the file is
wrong: warn (`iris.warnings`) naming the variable and the offending value,
and carry on. Do not reshape the reader around one publisher's output; raise
only where the data cannot be interpreted at all.


## Comments and Docstrings

A stale comment is worse than no comment — an agent treats it as evidence and
a human treats it as documentation.

- Comments explain **why**: the CF rule, the constraint, the bug worked
  around. Never restate what the code plainly does.
- Fix or delete any comment your change invalidates, even if you did not
  write it.
- A comment claiming a branch is unreachable is a testable claim. Pin it
  with a test or an `assert`, or restructure so it is true by construction.
  Left as prose it goes stale unnoticed: one reading "the saver always
  encodes" outlived its condition, and the branch it excused corrupted data.
- Do not add docstrings or comments to code you did not otherwise change —
  but see the module docstring exemption below.
- NumPy-style docstrings are mandatory and validated. State the contract:
  units, shapes, laziness, mutation, exceptions raised.
- Link non-obvious workarounds to their source:
  `# See https://github.com/SciTools/iris/issues/1234`.


## Module Docstrings Carry the Explanation

Prose explaining a subsystem belongs in the module docstring. Inline comments
scatter one argument across dozens of sites; a companion Markdown file is
validated by nothing and rots unseen. Only the docstring is read on every
visit, published by Sphinx, checked by numpydoc, and reviewed in the diff.

Be generous: sixty lines of orientation above a two-thousand-line module is
cheap, read once per visit rather than once per call site. Not a tutorial.

Cover whichever apply:

- what the module is for, and what it deliberately does not do;
- the vocabulary — name the concepts the code assumes you already know;
- invariants the code depends on but cannot assert;
- why the design is as it is, including approaches tried and rejected;
- known traps, and the modules this one is coupled to.

Leave out anything a reader or Sphinx can already derive: API listings,
signatures, call sequences, per-release history. Explanation scoped to a
single class belongs in that class's docstring instead.

A public module's docstring is also published as an API page, so it must open
with a `z_reference` item — a title, `:tags:` carrying at least one `topic_*`,
and a one-line summary — within its first 25 lines. Copy the form from a
neighbouring module; `pre-commit run check-docs-page-metadata` checks it.

**Exemption to "do not document code you did not change":** if you had to work
out how a subsystem behaves in order to edit it, write that understanding into
the module docstring — the comprehension is otherwise discarded when your
context ends. `_concatenate.py`, `_merge.py` and `common/resolve.py` are the
most commonly misread modules and carry barely a line each.


## Size and Shape

Agents pay for every line they read; humans lose the thread. Existing giants
such as `cube.py` (~5600 lines) are legacy: do not grow them without cause,
but do not opportunistically split them either — put genuinely separable new
code in a private sibling module instead.

- New modules: aim under ~1000 lines, one coherent concept each.
- Functions: aim to fit one screen (~50 lines).
- Use guard clauses and early returns instead of nested conditionals.
- Nesting beyond three levels is a signal to extract a named helper.


## Where Human and Agent Preferences Diverge

| Tension | Resolution |
|---|---|
| Abstraction vs duplication | Duplicate up to ~3 occurrences; abstract only once the shape is proven. |
| Clever idiom vs plain code | Plain. No nested comprehensions beyond two levels, no walrus inside complex expressions, no metaclass tricks. |
| Small files vs cohesion | Cohesion wins. Do not fragment into micro-modules purely to shrink files. |
| Type hints vs noise | Hint new and changed public code only. |
| Consistency vs local improvement | Consistency wins. A uniform mediocre idiom beats a mix of good ones. |


## Anti-Patterns — Do Not Introduce

- Dynamically generated attributes, methods or module members.
- `eval`, `exec`, or `getattr` string dispatch in library code.
- Metaclasses, or `__getattr__` used to invent behaviour. `__getattr__` has
  one legitimate use: presenting an open-ended set of *data* keys read from
  a file as attributes, as `CFVariable` does over CF-netCDF attributes. It
  must forward to a declared `Mapping`, that `Mapping` must be the path
  library code takes, and the docstring must say so. Note the cost: a class
  with `__getattr__` makes mypy stop checking *every* attribute on it and
  its subclasses, so confining it is what keeps the rest of the class typed.
- Silent `except Exception: pass`.
- Boolean flags that switch a function between two unrelated behaviours —
  write two functions.
- In-place mutation of input cubes or coordinates; return new objects.
- Refactors, renames or "improvements" outside the scope of the change.


## Pre-Finish Checklist

- `pre-commit run --files <changed paths>` passes, with any auto-fixes
  re-staged and the run repeated until clean.
- New and changed public functions have type hints and NumPy docstrings.
- No comment or docstring was left stale by the change.
- Laziness behaviour is preserved and documented.
- Tests updated per [`tests/AGENTS.md`](tests/AGENTS.md); changelog fragment
  added per [`../../changelog/AGENTS.md`](../../changelog/AGENTS.md).


## ⚠️ Meta-Instruction: Changing This File
- **Trigger**: If your work establishes a durable, reusable rule, you MUST
  propose it before your session ends.
- **Constraint 1**: Propose, never self-apply. Say it in your closing message,
  or raise it as its own pull request. NEVER edit an `AGENTS.md` silently, or
  as a side effect of unrelated work.
- **Constraint 2**: Keep this file under 300 lines — every agent loads it in
  full. If an addition would break that, tighten your wording; do NOT delete
  existing guidance to make room. Removing a rule is its own proposal.
- **Constraint 3**: Only global, reusable lessons. Do not propose temporary or
  component-specific fixes.
