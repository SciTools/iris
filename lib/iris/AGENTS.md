# AGENTS.md

Agent instructions for `lib/iris/` — the Iris library source.

These rules govern **how code is written** under this tree. Tests have their
own rules in [`tests/AGENTS.md`](tests/AGENTS.md).

For **where code lives and how data flows**, read
[`ARCHITECTURE.md`](ARCHITECTURE.md) before exploring unfamiliar subsystems.
It is an orientation map only; algorithm detail belongs in module docstrings.


## Why This File Exists

Iris will be maintained by humans and agents together for the foreseeable
future. Agents read code through a narrow context window using text search,
and cannot run the code to find out what it does. Humans read with intuition
and experience, but limited patience.

These two readers want mostly the same things: explicit names, local
reasoning, small units. This file records that shared style, plus the few
places where the two genuinely diverge and which way to resolve them.

Note that models are trained to imitate human code and to please human diff
reviewers — not to produce code that is cheap to navigate later. Default
output therefore drifts towards long functions, redundant comments and
defensive wrapping. Correct for that deliberately.


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
  for a single caller.
- Pass dependencies in as arguments rather than reaching for module-level
  mutable state.


## Greppability

Exact text search is the primary retrieval tool. Code that cannot be found
does not exist.

- Never construct identifiers at runtime: no `setattr(obj, f"{name}_bounds",
  ...)`, no `getattr(module, method_name)` dispatch where a dict or `if/elif`
  would do. A constructed name is unfindable for every reader.
- Give public names enough distinctiveness to search for. Local `data`,
  `cube`, `result` are fine; a public helper called `_process` is not.
- Declare `__all__` in modules with a public surface. No `import *`.
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


## Comments and Docstrings

A stale comment is worse than no comment — an agent treats it as evidence and
a human treats it as documentation.

- Comments explain **why**: the CF rule, the constraint, the bug worked
  around. Never restate what the code plainly does.
- Fix or delete any comment your change invalidates, even if you did not
  write it.
- Do not add docstrings or comments to code you did not otherwise change —
  but see the module docstring exemption below.
- NumPy-style docstrings are mandatory and validated. State the contract:
  units, shapes, laziness, mutation, exceptions raised.
- Link non-obvious workarounds to their source:
  `# See https://github.com/SciTools/iris/issues/1234`.


## Module Docstrings Carry the Explanation

Prose explaining a subsystem has to live somewhere, and the module docstring
is the best available home. Inline comments are the wrong place: they
interrupt the code for every reader and scatter a single argument across
dozens of sites. A companion Markdown file is also wrong: nothing validates
it, nothing forces anyone to open it, and it rots unseen. The module
docstring is the only location that is read automatically whenever the file
is opened, published by Sphinx, checked by numpydoc, and reviewed in the same
diff as the code it describes.

Be generous with it. Sixty lines of orientation above a two-thousand-line
module is cheap, because it is read once per visit rather than once per call
site. Stop short of a tutorial.

Cover whichever apply:

- what the module is for, and what it deliberately does not do;
- the vocabulary — name the concepts the code assumes you already know;
- invariants the code depends on but cannot assert;
- why the design is as it is, including approaches tried and rejected;
- known traps, and the modules this one is coupled to.

Leave out anything a reader or Sphinx can already derive: API listings,
signatures, call sequences, per-release history. Explanation scoped to a
single class belongs in that class's docstring instead.

**Exemption to "do not document code you did not change":** if you had to
work out how a subsystem behaves in order to edit it, writing that
understanding into the module docstring is in scope. The comprehension was
expensive and is otherwise discarded when your context ends; the next reader,
human or agent, should not have to repeat it. Several of the most commonly
misread modules — `_concatenate.py`, `_merge.py`, `common/resolve.py` —
currently carry only a line or two of module docstring apiece.


## Size and Shape

Agents pay for every line they read; humans lose the thread. Existing giants
such as `cube.py` (~5600 lines) are legacy — do not grow them without cause,
but do not opportunistically split them either.

- New modules: aim under ~1000 lines, one coherent concept each.
- Functions: aim to fit one screen (~50 lines).
- Use guard clauses and early returns instead of nested conditionals.
- Nesting beyond three levels is a signal to extract a named helper.
- When adding genuinely separable code to an oversized module, put it in a
  new private sibling module rather than growing the giant.


## Where Human and Agent Preferences Diverge

| Tension | Resolution |
|---|---|
| Verbosity vs concision | Favour explicit. But never narrate line by line. |
| Abstraction vs duplication | Duplicate up to ~3 occurrences; abstract only once the shape is proven. |
| Clever idiom vs plain code | Plain. No nested comprehensions beyond two levels, no walrus inside complex expressions, no metaclass tricks. |
| Small files vs cohesion | Cohesion wins. Do not fragment into micro-modules purely to shrink files. |
| Type hints vs noise | Hint new and changed public code only. |
| Consistency vs local improvement | Consistency wins. A uniform mediocre idiom beats a mix of good ones. |


## Anti-Patterns — Do Not Introduce

- Dynamically generated attributes, methods or module members.
- `eval`, `exec`, or `getattr` string dispatch in library code.
- Metaclasses or `__getattr__`, except in existing deprecation shims.
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


## ⚠️ Meta-Instruction: Auto-Update Rule
- **Trigger**: You MUST automatically update this `AGENTS.md` file if a new
  structural pattern, critical bug fix, or persistent repository rule is
  established during this session.
- **Constraint 1**: Keep the file under 200 lines. Delete obsolete rules to
  make room for new ones.
- **Constraint 2**: Do not ask for user permission before updating. Perform the
  edit silently as your final step.
- **Constraint 3**: Only log global, reusable lessons. Do not log temporary or
  component-specific fixes.
