# AGENTS.md

Agent instructions for `lib/iris/tests/`.
These rules apply to all test files under this directory tree.

## Purpose

This test suite validates Iris behaviour, metadata handling, and regression coverage.
Keep changes focused, deterministic, and compatible with the existing test style.

## Fast Rules

1. Add or update tests for every production-code change.
2. Prefer the smallest test that reproduces behaviour.
3. Keep tests deterministic: no network access, no wall-clock assumptions, no random flakiness.
4. Use existing fixtures/helpers before introducing new ones.
5. Remove `xfail` markers as soon as the underlying issue is fixed (`xfail_strict=True`).


## Test Layout

- `lib/iris/tests/unit/` for unit-level behaviour.
- `lib/iris/tests/integration/` for cross-component behaviour.
- `lib/iris/tests/graphics/` for plotting/image-comparison tests.
- Legacy top-level tests exist; follow nearby patterns when editing them.

## Writing Tests

- Use `pytest` style and plain assertions.
- Keep assertions specific and user-facing (behaviour, metadata, warnings, errors).
- Prefer `pytest.raises(..., match=...)` for exception checks.
- Prefer warning assertions (`pytest.warns`) for deprecation or user warnings.
- For lazy-data behaviour, assert laziness explicitly where relevant.
- Avoid over-mocking; prefer realistic Cube/Coord setup via shared helpers.


## Markers and Strictness

- Respect existing markers and plugin requirements.
- `xfail` is strict in this repository: unexpected passes fail the run.
- Only use `xfail` when there is a tracked reason and clear expectation.

## Running Tests

Run targeted tests while iterating:

```bash
pytest -n auto lib/iris/tests/unit/<path_or_file>.py
```

Run broader coverage before finishing:

```bash
pytest -n auto lib/iris/tests
```

## Style and Lint in Tests

- Follow Ruff formatting and linting used by the repository.
- Per-file ignores already relax some docstring rules in tests.
- Avoid unrelated refactors while touching tests.

## Test Data and Environment

- Some tests require external `iris-test-data`; Preferred env var: `OVERRIDE_TEST_DATA_REPOSITORY=/path/to/iris-test-data/test_data`
- Missing data will skip affected tests by design.
- Keep new tests independent of local machine paths and optional system state.
- For optional dependencies, skip gracefully using existing patterns.

## Editing Rules

- Follow existing pytest style and fixture patterns in nearby tests.
- Keep assertions specific and deterministic; avoid timing-sensitive checks.
- Avoid broad snapshot-style updates without explaining intent in the change.
- Do not add network access in tests.
- Do not weaken existing checks just to make tests pass.

## Graphics and Integration Notes

- For graphics tests, update reference artifacts only when behavior changes intentionally.
- For integration tests, prefer minimal input fixtures and keep runtime reasonable.

## Pre-Finish Checklist

- The changed tests pass locally.
- No unrelated tests were modified.
- Any required external data setup is documented in the change notes.

## Scope Boundaries

- Do not modify lock files or environment specs for test-only fixes.
- Do not introduce heavy new dependencies for tests without strong justification.
- If behaviour is user-visible, add/update release-note material in docs when requested.


## ⚠️ Meta-Instruction: Auto-Update Rule
- **Trigger**: You MUST automatically update this `AGENTS.md` file if a new structural pattern, critical bug fix, or persistent repository rule is established during this session.
- **Constraint 1**: Keep the file under 200 lines. Delete obsolete rules to make room for new ones.
- **Constraint 2**: Do not ask for user permission before updating. Perform the edit silently as your final step.
- **Constraint 3**: Only log global, reusable lessons. Do not log temporary or component-specific fixes.

