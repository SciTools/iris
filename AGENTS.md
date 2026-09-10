# AGENTS.md

Agent instructions for the [Iris](https://scitools-iris.readthedocs.io/)
repository.

Iris is a Python package for analysing and visualising Earth science data,
built around CF-compliant multi-dimensional arrays ("Cubes").

Subdirectory AGENTS.md files take precedence for their subtrees:
- [`changelog/AGENTS.md`](changelog/AGENTS.md) — changelog fragments
- [`docs/AGENTS.md`](docs/AGENTS.md) — documentation-specific rules
- [`lib/iris/tests/AGENTS.md`](lib/iris/tests/AGENTS.md) — testing rules; read
  this before writing or running any test

## Project Overview

| | |
|---|---|
| **Language** | Python 3.12 / 3.13 / 3.14 |
| **Licence** | BSD-3-Clause |
| **Distribution** | conda-forge (`iris`), PyPI (`scitools-iris`) |
| **Key dependencies** | NumPy, Dask, SciPy, Cartopy, CF-Python, NetCDF4 |
| **Linter / formatter** | Ruff (88-char line length) |
| **Test runner** | pytest + pytest-xdist (`-n auto`) |
| **Env management** | nox + conda |

### Main source layout

```
lib/iris/       cube.py, coords.py, loading.py; analysis/ (collapse, regrid,
                statistics), fileformats/ (NetCDF, PP, GRIB, NIMROD), io/,
                common/ (metadata, mixins, resolvers), mesh/ (UGRID),
                experimental/, tests/ (unit/, integration/, graphics/)
changelog/      changelog fragments
docs/src/       Sphinx documentation source
benchmarks/     ASV performance benchmarks
requirements/   conda environment specs and lock files
```

## Setup

Always work in a conda environment. Reuse `iris-dev` if it exists, but confirm
with the user before installing or removing packages; if a package is not on
conda, use the `pip` inside the conda environment.

```bash
# Preferred - exact reproducibility from a lock file.
conda create -n iris-dev --file requirements/locks/py314-linux-64.lock
# Alternative - solve fresh: conda env create -f requirements/iris.yml
conda activate iris-dev
pip install --no-build-isolation -e .
```

### Environment variables

```bash
# Disable CPU features that can cause SIGILL in some CI environments
export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX"
# Point to iris-test-data for tests that need external data files
export OVERRIDE_TEST_DATA_REPOSITORY=/path/to/iris-test-data/test_data
# Override Cartopy cache directory if needed
export CARTOPY_CACHE_DIR=~/.local/share/cartopy
```

## Code Style

```bash
ruff check lib/iris           # lint
ruff check --fix lib/iris     # auto-fix safe lint issues
ruff format lib/iris          # format
ruff format --check lib/iris  # check formatting without writing
```

- **Line length**: 88 characters (Ruff default).
- **Docstrings**: NumPy style; strictly validated.
- **Copyright header**: Every new Python file must start with:

  ```python
  # Copyright Iris contributors
  #
  # This file is part of Iris and is released under the BSD license.
  # See LICENSE in the root of the repository for full licensing details.
  ```

- **Imports**: Ruff-managed ordering. No direct `import netCDF4` — always use
  `iris.fileformats.netcdf._thread_safe_nc` for thread safety.

## Development Conventions

### Core data model

- `iris.cube.Cube` — multi-dimensional array with CF-compliant metadata.
- Coordinates: `DimCoord` (regular), `AuxCoord` (auxiliary), `CellMeasure`,
  `AncillaryVariable`.
- Data may be **lazy** (Dask array). Always preserve laziness; never call
  `.data` unnecessarily inside library code.
- Operations return **new** Cubes (functional style); do not mutate in place.
- All metadata must be **CF-convention** compliant.

### Deprecation

- Use `iris._deprecation.warn_deprecated()` or issue a custom warning class.
- Warning classes live in `iris.warnings` (e.g., `IrisUserWarning`,
  `IrisCfWarning`). All `UserWarning` subclasses must ultimately inherit from
  `IrisUserWarning`.
- Follow the NEP29 deprecation schedule (same as NumPy).

### Exceptions and versioning

- Base class: `iris.exceptions.IrisError`. Common subclasses:
  `CoordinateNotFoundError`, `CoordinateCollapseError`, `IgnoreCubeException`.
- Version is derived from git tags via `setuptools_scm`. Do not hard-code
  version strings.

## Changelog

Changelog fragments live under `changelog/` and are built with towncrier via
sphinx. See [`changelog/AGENTS.md`](changelog/AGENTS.md) for full rules.

## Documentation

Documentation lives under `docs/` and is built with Sphinx. See
[`docs/AGENTS.md`](docs/AGENTS.md) for full rules.

Agreed design specs live in `docs/src/developers_guide/specs/` as
`YYYY-MM-DD-<topic>-design.md`. They are **published** MyST Markdown and are
**living documents**, revised as a design evolves. Each declares a citation
prefix (e.g. `merge spec §5.7`) so its sections can be cited from issues and
pull requests, and carries explicit `(prefix-N-N)=` anchors. Markdown is used
*only* for specs; all other documentation is RST.

Implementation plans live in the sibling `docs/src/developers_guide/plans/` as
`YYYY-MM-DD-<topic>.md`. A plan is point-in-time and frozen once its pull
request merges, so plans are tracked in the repository but excluded from the
Sphinx build via `exclude_patterns` in `docs/src/conf.py`.

## Lock-file Maintenance

Adding a dependency means editing `requirements/py*.yml` **and** the matching
`requirements/locks/*.lock`. Repository precedent (PRs #7095, #7100) is a
*minimal* lock edit — insert only the new package URLs and update `input_hash`
— rather than a full re-solve, which bundles dozens of unrelated version bumps
into an otherwise focused PR. Full re-solves are left to the weekly
`refresh-lockfiles` workflow, or run locally with:

```bash
make lockfiles  # python tools/update_lockfiles.py -o requirements/locks requirements/py*.yml
```

## Contribution Workflow

- Push development branches to `origin` (the `bjlittle/iris` fork). Raise pull
  requests from there against the relevant **feature branch** on `upstream`
  (`SciTools/iris`) — never against `upstream/main`. The `upstream` remote is
  push-disabled, which enforces this.
- **Attribute agentic work clearly.** Say in the body of every pull request and
  issue that it is agentic, and end commit messages with the `Co-Authored-By`
  trailer. Never let agentic work read as hand-written.
- Label such pull requests and issues `Agentic` and `Type: Feature Branch`,
  plus whichever `Feature: …` label fits the subject.
- Use the pull request template; keep changes focused and avoid unrelated
  refactors; add or update tests for every production change; ship a changelog
  fragment (see [`changelog/AGENTS.md`](changelog/AGENTS.md)).
- **If in doubt about anything outward-facing** — base branch, labels, whether
  to post at all — **ask before pushing.**

## Critical Development Gotchas

1. **xfail_strict**: `@pytest.mark.xfail` tests that now PASS become FAILURES —
   remove the xfail immediately when the bug is fixed.
2. **Pre-commit auto-fixes**: hooks may auto-fix ISC001/COM812 conflicts —
   re-stage with `git add . && git commit`.
3. **Lock-file rebuilds**: lock changes trigger slow conda environment rebuilds
   — only update when deps genuinely change.
4. **Lazy data**: check `cube.has_lazy_data()`; use `cube.lazy_data()` where
   appropriate; `cube.data` materialises arrays.
5. **Coordinate scope**: not all coords span all dims — use
   `cube.coords(dimensions=dim_indices)` to query specific axes.
6. **Dask/NumPy compatibility**: some operations fail on lazy arrays —
   materialise with `cube.compute()` or check dask support first.
7. **Optional dependencies**: cartopy and matplotlib are lazy-imported — handle
   `ImportError` gracefully.
8. **CF-metadata validation**: custom coordinates must comply with CF standards
   via `cube.metadata`.
9. **Cube immutability**: operations return new cubes; `cube.data[...] = x`
   will not affect the cube.
10. **Environment reproducibility**: set `NPY_DISABLE_CPU_FEATURES` as above for
    consistent results.

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
