# AGENTS.md

Agent instructions for `lib/iris/tests/`.
These rules apply to all test files under this directory tree.


## Purpose

This test suite validates Iris behaviour, metadata handling, and regressionc
overage.  Keep changes focused, deterministic, and compatible with the
existing test style.


## Fast Rules

1. Add or update tests for every production-code change.
2. Prefer the smallest test that reproduces behaviour.
3. Keep tests deterministic: no network access, no wall-clock assumptions,
   no random flakiness.
4. Use existing fixtures/helpers before introducing new ones.


## Test Layout

- `lib/iris/tests/unit/` for unit-level behaviour.
- `lib/iris/tests/integration/` for cross-component behaviour.
- `lib/iris/tests/graphics/` for plotting/image-comparison tests.


## Writing Tests

- Use `pytest` style and plain assertions.
- Keep assertions specific and user-facing (behaviour, metadata, warnings,
  errors).
- Prefer `pytest.raises(..., match=...)` for exception checks.
- Prefer warning assertions (`pytest.warns`) for deprecation or user warnings.
- For lazy-data behaviour, assert laziness explicitly where relevant.
- Avoid over-mocking; prefer realistic Cube/Coord setup via shared helpers.


## Running Tests

`-n auto` is **not** in `addopts`. Pass it yourself or the run is serial:
roughly half speed on the full suite, and no worse than break-even on
anything smaller (see below).

Work in tiers. The full suite is minutes; the area you are changing is
usually seconds, and is the right command for nearly all of an iteration.

| Tier | When | Command |
|---|---|---|
| The module you changed | every edit | `pytest lib/iris/tests/unit/<area>/` |
| Its immediate neighbours | before committing | `pytest -n auto lib/iris/tests/unit/<parent>/ lib/iris/tests/integration/<area>/` |
| Everything | before finishing, and to compare against a base branch | `pytest -n auto lib/iris/tests` |

Indicative on a 4-core machine: ~6s, ~1min, ~4min respectively. Running the
full suite on every edit is the single biggest avoidable delay in this
repository.

- **Do not add `-n auto` to small runs.** Spawning workers costs a few
  seconds, so it makes a sub-10-second selection slower, not faster.
- **Do not raise `-n` above the core count.** Measured on the full suite,
  `-n 8` on 4 cores bought 7% wall time for 60% more CPU.
- Prefer `-o cache_dir=<scratch>/pytest_cache` over `-p no:cacheprovider`
  when concurrent runs must not collide. Disabling the cache plugin also
  disables `--lf` and `--ff`, which are the cheapest speed-ups available
  while iterating on failures.


### Doctests

The pytest suite does not run them — nothing here covers the `>>>` examples
in library docstrings (29 modules) or the user guide (30 files). Only Sphinx
does:

| Command | Cost |
|---|---|
| `nox -s doctest` | what CI runs: `make clean html`, then `make doctest` |
| `cd docs/src && make doctest` | skips the gallery rebuild; faster, not identical to CI |

Both take minutes, so run one once before finishing, and only if you touched
a `>>>` example. Do not substitute `pytest --doctest-modules` or
`python -m doctest`: 150 `testsetup::` / `testcode::` directives supply
context only Sphinx applies, so plain doctest reports failures that are not
real.


## Style and Lint in Tests

- Follow Ruff formatting and linting used by the repository.
- Per-file ignores already relax some docstring rules in tests.
- Avoid unrelated refactors while touching tests.


## Test Data and Environment

- Some tests require external `iris-test-data`; Preferred env
  var: `OVERRIDE_TEST_DATA_REPOSITORY=/path/to/iris-test-data/test_data`
- **Set it before trusting any run**, and check it took effect. Missing data
  skips affected tests by design — silently, and without failing — so an
  unset variable quietly removes coverage from whatever you are changing. Ask
  for the path if you cannot find it; do not proceed and caveat the result.
- Keep new tests independent of local machine paths and optional system state.
- For optional dependencies, skip gracefully using existing patterns.


## Editing Rules

- Follow existing pytest style and fixture patterns in nearby tests.
- Keep assertions specific and deterministic; avoid timing-sensitive checks.
- Avoid broad snapshot-style updates without explaining intent in the change.
- Do not add network access in tests.
- Do not weaken existing checks just to make tests pass — but a test can pin
  a *bug*, and correcting one is strengthening. Show by archaeology that the
  expectation was wrong, invert rather than delete, and comment why.

## Graphics and Integration Notes

- For graphics tests, update reference artifacts only when behavior changes
  intentionally.
- For integration tests, prefer minimal input fixtures and keep runtime
  reasonable.


## Pre-Finish Checklist

- The changed tests pass locally.
- No unrelated tests were modified.
- Any required external data setup is documented in the change notes.
- **Read the skip count, not just the failures.** Comparing failures before
  and after a change — the standard way to show a refactor broke nothing —
  cannot detect a test that never ran. A skip is neither a pass nor a
  failure, so a `FAILED|ERROR` diff filters it out and the comparison comes
  back clean whether coverage is intact or absent. Treat any skip in the area
  you changed as a gap to close, usually missing `iris-test-data` or an
  optional dependency, rather than as background noise.
- **Never compare warning totals.** Under `-n auto` they are not
  reproducible — import-time deprecations are counted once per worker that
  imports the module, so the total tracks work distribution. One unchanging
  tree gave 6483, 6485, 6484, 6483, 6483. A warning delta is not evidence.
- **Compare like with like.** Collection order affects error counts:
  `integration/netcdf/test_coord_systems.py` reports ten errors in a tier
  run and none in the full suite. A tier baseline says nothing about a
  full-suite one.


## Scope Boundaries

- Do not modify lock files or environment specs for test-only fixes.
- Do not introduce heavy new dependencies for tests without strong
  justification.
- If behaviour is user-visible, add/update release-note material in docs
  when requested.


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
