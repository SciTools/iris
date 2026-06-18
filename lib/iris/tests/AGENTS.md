# AGENTS Guide for lib/iris/tests/

This file applies to everything under `lib/iris/tests/`.

## Purpose

- Keep test changes targeted, deterministic, and quick to review.
- Prefer changing tests only; do not modify production code unless the task explicitly requires it.


### Test layout

| Path | Content |
|------|---------|
| `lib/iris/tests/unit/` | Unit tests, mirroring `lib/iris/` structure |
| `lib/iris/tests/integration/` | Cross-component integration tests |
| `lib/iris/tests/graphics/` | Image-comparison (graphics) tests |
| `lib/iris/tests/conftest.py` | Top-level fixtures |

### External test data

Some tests require `iris-test-data`. Set `OVERRIDE_TEST_DATA_REPOSITORY` to the
`test_data` directory inside that repository. Tests missing the data are skipped.

### Graphics tests

Graphics tests compare against reference images. Update reference images only when an
intentional rendering change is made.

## Run Commands

Run from repository root unless otherwise noted.

- Full test suite: `pytest -n auto lib/iris/tests`
- Single test file: `pytest -n auto lib/iris/tests/path/to/test_file.py`
- Single test selection: `pytest -n auto lib/iris/tests/path/to/test_file.py -k "pattern"`
- Verbose failures: add `-ra`

Nox equivalent for CI-like execution:

- `nox --session tests`

## External Test Data

Some tests require `iris-test-data`.

- Preferred env var: `OVERRIDE_TEST_DATA_REPOSITORY=/path/to/iris-test-data/test_data`
- Missing data will skip affected tests by design.

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


## ⚠️ Meta-Instruction: Auto-Update Rule
- **Trigger**: You MUST automatically update this `AGENTS.md` file if a new structural pattern, critical bug fix, or persistent repository rule is established during this session.
- **Constraint 1**: Keep the file under 200 lines. Delete obsolete rules to make room for new ones.
- **Constraint 2**: Do not ask for user permission before updating. Perform the edit silently as your final step.
- **Constraint 3**: Only log global, reusable lessons. Do not log temporary or component-specific fixes.

---

**Last Updated**: 16 June 2026  

