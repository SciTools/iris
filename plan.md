# Agentic Plan for Delivery

## Goal
Enable pytest to fail on problem warnings (particularly `DeprecationWarning`),
making it easy to detect newly-introduced issues. Achieve this by:
1. Classifying all warnings currently in the log
2. Fixing addressable warnings in Iris source/test code
3. Asserting expected warnings inside tests (not leaking)
4. Suppressing unavoidable external warnings
5. Promoting DeprecationWarning (and related) to errors in pyproject.toml

---

## Warning Inventory (from pytest_warnings.log, Jun 2026)

### A. IrisDeprecation — Iris-owned, still used in tests (MUST FIX)
- `GraphicsTestMixin`, `IrisTest`, `GraphicsTest`, `PPTest` classes deprecated (tests/__init__.py)
- `iris.experimental.regrid` / `regrid_conservative` deprecated since v3.2 (used in test_regrid_* tests)
- `iris.experimental.raster` deprecated since v3.2
- `iris.experimental.ugrid` deprecated → `iris.mesh`
- `iris.fileformats.abf` / `iris.fileformats.dot` deprecated
- `env_bin_path` deprecated
- `iris.fileformats.netcdf.saver` legacy attribute mode deprecated since v3.8
- `iris.analysis.maths.intersection_of_cubes` deprecated
- `iris.coord_systems.RotatedMercator` deprecated → `ObliqueMercator`
- Various `regrid_weighted_curvilinear_to_rectilinear`, `ProjectedUnstructuredNearest/Linear` deprecated

### B. DeprecationWarning — Iris test code (MUST FIX)
- `np.core.ndarray` → `np.ndarray` in test_DataManager.py (lines 335, 549, 550)
- Replace deprecated `np.matrix` coercion tests with a custom `np.ndarray` subclass instance (created via `.view(Subclass)`), preserving the original intent: verify ndarray-subclass inputs are coerced back to plain `np.ndarray`
- `Unit.is_long_time_interval()` path is deprecated in cf-units; align Iris with cf-units PR #279 by removing this pre-check and using `num2date` with try-except fallback in coords.py

### C. Pytest infrastructure warnings (MUST FIX)
- `PytestUnknownMarkWarning`: `pytest.mark.skipIf` (capital I) → `pytest.mark.skipif` in tests/graphics/__init__.py:288
- `PytestCollectionWarning`: `TestAuxFact` class with `__init__` in test_AuxCoordFactory.py:216
- `PytestMockWarning`: mocker.patch used as context manager in test_FF2PP.py (lines 75, 86, 89, 90, 91, 92, 220)

### D. IrisUserWarning / Iris domain warnings — intentionally triggered (ASSERT IN TESTS)
These are raised by Iris logic under test; tests should explicitly assert/capture them:
- `IrisUserWarning`: collapsing spatial coord without weighting, CRS mismatch, concatenate warnings
- `IrisLoadWarning`, `IrisSaveWarning`, `IrisCfMissingVarWarning`, `IrisCfNonSpanningVarWarning`
- `IrisCfLabelVarWarning`, `IrisGuessBoundsWarning`, `IrisVagueMetadataWarning`
- `IrisDefaultingWarning`, `IrisIgnoringBoundsWarning`, `IrisPpClimModifiedWarning`
- `IrisNimrodTranslationWarning`, `IrisGeometryExceedWarning`
- `_WarnComboDefaultingLoad`, `_WarnComboIgnoringCfLoad`, `_WarnComboLoadIgnoring`

### E. External/third-party warnings (SUPPRESS in pyproject.toml)
- `DeprecationWarning` from `distributed/client.py` (dask large graph)
- `DeprecationWarning` from `osgeo/gdal.py` (GDAL NumPy scalar conversion)
- `DeprecationWarning` from `numpy.ma.extras` / `numpy._core` (NumPy internal changes)
- `UserWarning` from `cartopy/crs.py` (Orthographic/NearsidePerspective elliptical globes)
- `UserWarning` from `distributed/client.py` (large dask graph size warning)
- `UserWarning` from `numpy.ma.core` (masked element to nan)
- `RuntimeWarning` from `numpy.ma.core`/`numpy._core` (invalid value in cast — in numpy internals)
- `RuntimeWarning` from `matplotlib/collections.py` (invalid value in sqrt)

### F. RuntimeWarning — Iris/test code (ASSESS)
- Divide by zero in test_divide.py/maths.py (intentional, test should assert)
- `invalid value encountered in cast` in analysis/_regrid.py:818 (may need investigation)

---

## PR Split
1. PR 1 covers Phases 1-3: pytest infrastructure fixes, direct deprecation fixes in Iris/tests, and migration off deprecated Iris test-framework utilities.
2. PR 2 covers Phases 4-5: assert expected domain warnings and then tighten warning policy/filtering in pytest configuration.
3. PR 2 depends on PR 1 merging first, to avoid large noise from unresolved deprecations and to keep warning-policy changes reviewable.
4. Merge gate between PRs: warning count from PR 1 branch must be stable and attributable, so PR 2 only changes behavior by policy, not by hidden code migrations.

## Handoff Protocol
1. Scope lock per run: before any agent changes, explicitly choose the active target (`PR 1` or `PR 2`) and confirm no cross-PR work is allowed in that run.
2. Plan re-read at session start: the agent must re-read this plan and state the active scope before editing.
3. Drift check before edits: verify branch state, current warning baseline, and plan changes since last run; if any drift exists, record a delta note first.
4. PR contract boundaries:
   - `PR 1` only: Phases 1-3 (infrastructure/deprecation fixes + migration off deprecated test-framework utilities).
   - `PR 2` only: Phases 4-5 (expected-warning assertions + pytest warning-policy tightening).
5. No scope bleed: out-of-scope findings are logged as deferred follow-ups, not implemented in the current PR.
6. Execution log continuity: keep a dated session log of completed items, deferred items, blockers, and decisions to bridge multi-day gaps.
7. Re-entry rule after gaps: first action is always plan reload + baseline warning check, then continue implementation.
8. Human review checklist mirrors plan phases: reviewer verifies each changed file matches the active PR scope and rejects unrelated "bonus" changes.

## Phased Implementation Plan

### Phase 1: Fix pytest infrastructure warnings (no functional changes, quick wins)
1. Fix `pytest.mark.skipIf` → `pytest.mark.skipif` in `lib/iris/tests/graphics/__init__.py:288`
2. Fix `TestAuxFact` class `__init__` in `lib/iris/tests/unit/aux_factory/test_AuxCoordFactory.py:216`
3. Fix mocker.patch context manager usage in `lib/iris/tests/unit/fileformats/ff/test_FF2PP.py` (lines 75, 86, 89-92, 220)

### Phase 2: Fix DeprecationWarnings in test code
4. Replace `np.core.ndarray` with `np.ndarray` in `lib/iris/tests/unit/data_manager/test_DataManager.py` (lines 335, 549, 550)
5. Replace deprecated `np.matrix` usages in `lib/iris/tests/unit/cube/test_Cube.py:72` and `lib/iris/tests/unit/data_manager/test_DataManager.py:546` with a purpose-built `np.ndarray` subclass input (for example `arr.view(Subclass)`), then assert output type is plain `np.ndarray` to preserve coercion semantics
6. Replace deprecated `is_long_time_interval()` guard in `lib/iris/coords.py:383` with a `num2date` attempt wrapped in `try/except` so invalid long-interval units fall back to numeric formatting (matching cf-units PR #279 guidance)

### Phase 3: Remove deprecated test-framework usage and selectively assert remaining deprecations
For each group of tests that exercise deprecated APIs:
1. Prefer code migration/removal of deprecated usage; only use `pytest.warns(IrisDeprecation)` when the deprecated API itself is the behavior under test
   - `test_regrid_area_weighted_rectilinear_src_and_grid.py` (regrid deprecated)
   - `test_regrid_conservative_via_esmpy.py` (regrid_conservative deprecated)
   - `test_raster.py` / `test_export_geotiff.py` (raster deprecated)
   - `test_regrid_ProjectedUnstructured.py` (ProjectedUnstructured deprecated)
   - `test_regrid_weighted_curvilinear_to_rectilinear.py`
   - `test_RotatedMercator.py`
   - Tests using IrisTest/GraphicsTestMixin/PPTest/GraphicsTest/env_bin_path
   - `test_intersect.py` (intersection_of_cubes)
   - tests using netcdf legacy save mode
   - tests using iris.experimental.ugrid

### Phase 4: Assert expected Iris domain warnings in tests (Category D)
8. Audit tests for each Iris warning class and add `pytest.warns()` assertions where warnings are expected but not currently being tested for
   - Highest priority: tests that use `IrisUserWarning`, `IrisLoadWarning`, `IrisSaveWarning`
   - Secondary: vague metadata, guess bounds, ignore bounds, etc.

### Phase 5: Configure pyproject.toml filterwarnings
9. Update `pyproject.toml` `[tool.pytest.ini_options]` to:
   - Promote `DeprecationWarning` (and `PendingDeprecationWarning`) to errors: `"error::DeprecationWarning"`, `"error::PendingDeprecationWarning"`
   - Add targeted ignores for unavoidable external warnings (Category E above)
   - Keep `"default"` as a fallback or replace with `"error"` for Iris-specific categories

---

## Key Files
- `pyproject.toml` — `[tool.pytest.ini_options].filterwarnings`
- `lib/iris/tests/graphics/__init__.py` — skipIf typo (line 288)
- `lib/iris/tests/unit/aux_factory/test_AuxCoordFactory.py` — TestAuxFact __init__ (line 216)
- `lib/iris/tests/unit/fileformats/ff/test_FF2PP.py` — PytestMockWarning (lines 75, 86, 89-92, 220)
- `lib/iris/tests/unit/data_manager/test_DataManager.py` — np.core (lines 335, 549, 550), ndarray-subclass coercion test (line 546)
- `lib/iris/tests/unit/cube/test_Cube.py` — ndarray-subclass coercion test (line 72)
- `lib/iris/coords.py` — cftime.is_long_time_interval (line 383)
- All experimental regrid/raster/ugrid test files

## Verification
1. PR 1 verification: run `pytest lib/iris --tb=no -q` and confirm infrastructure/deprecation-warning reductions from Phases 1-3 are real and expected.
2. PR 1 verification: confirm deprecated Iris test-framework utilities (IrisTest/GraphicsTestMixin/PPTest/GraphicsTest/env_bin_path) are migrated where in scope, not merely warning-wrapped.
3. PR 2 verification: run targeted warning-sensitive suites first, then full `pytest lib/iris --tb=no -q`, and confirm Phase 4 assertions pass without leaking expected warnings.
4. PR 2 verification: confirm warning policy in pytest config causes non-zero exit for newly introduced `DeprecationWarning`/`PendingDeprecationWarning`, with only explicitly approved external suppressions.
5. Final verification: no legitimate regression failures introduced by warning-policy tightening.

