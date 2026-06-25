# AGENTS.md

Agent instructions for the [Iris](https://scitools-iris.readthedocs.io/)
repository.

Iris is a Python package for analysing and visualising Earth science data,
built around CF-compliant multi-dimensional arrays ("Cubes").

Subdirectory AGENTS.md files take precedence for their subtrees:
- [`changelog/AGENTS.md`](changelog/AGENTS.md) — documentation on changelog
- [`docs/AGENTS.md`](docs/AGENTS.md) — documentation-specific rules
- [`lib/iris/tests/AGENTS.md`](lib/iris/tests/AGENTS.md) — test-specific rules


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
lib/iris/
    cube.py            # Core Cube / CubeList data structures
    coords.py          # DimCoord, AuxCoord, CellMeasure, AncillaryVariable
    loading.py         # File-loading entry points
    analysis/          # Collapse, regrid, statistics, calculus
    fileformats/       # NetCDF, PP, GRIB, NIMROD format handlers
    io/                # I/O registry and URI handling
    common/            # Shared metadata, mixins, resolvers
    mesh/              # Unstructured grid (UGRID) support
    experimental/      # Unstable / in-progress features
    tests/             # All tests (unit/, integration/, graphics/)
changelog/             # changelog fragments
docs/src/              # Sphinx documentation source
benchmarks/            # ASV performance benchmarks
requirements/          # Conda environment specs and lock files
```


## Setup

### Conda environment (recommended)

Always use a conda environment, reuse the iris-dev conda environment if it
already exists but confirm with the user before installing or removing packages.

If a package cannot be installed via conda then you can use pip that is in the
conda environment.

```bash
# Create and activate a development environment
conda env create -f requirements/iris.yml
conda activate iris-dev
pip install --no-build-isolation -e .
```

Alternatively, use lock files for exact reproducibility:

```bash
conda create -n iris-dev --file requirements/locks/py314-linux-64.lock
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


## Testing

- [`lib/iris/tests/AGENTS.md`](lib/iris/tests/AGENTS.md) — test-specific 


## Code Style

```bash
# Lint
ruff check lib/iris

# Auto-fix safe lint issues
ruff check --fix lib/iris

# Format
ruff format lib/iris

# Check formatting without writing
ruff format --check lib/iris
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
  `.data`
  unnecessarily inside library code.
- Operations return **new** Cubes (functional style); do not mutate in place.
- All metadata must be **CF-convention** compliant.


### Deprecation

- Use `iris._deprecation.warn_deprecated()` or issue a custom warning class.
- Warning classes live in `iris.warnings` (e.g., `IrisUserWarning`,
  `IrisCfWarning`).
- Follow the NEP29 deprecation schedule (same as NumPy).
- All `UserWarning` subclasses must ultimately inherit from `IrisUserWarning`.


### Exception hierarchy

Base class: `iris.exceptions.IrisError`. Common subclasses:
`CoordinateNotFoundError`, `CoordinateCollapseError`, `IgnoreCubeException`.


### Versioning

Version is derived from git tags via `setuptools_scm`. Do not hard-code version
strings.


## Changelog

Changelog fragments lives under `chngelog/` and is built with towncrier via
sphinx. See [`changelog/AGENTS.md`](changelog/AGENTS.md) for full rules.


## Documentation

Documentation lives under `docs/` and is built with Sphinx. See
[`docs/AGENTS.md`](docs/AGENTS.md) for full rules.


## Lock-file Maintenance

```bash
# Regenerate lock files for all supported Python versions
python tools/update_lockfiles.py -o requirements/locks requirements/py*.yml
# Shortcut via Makefile
make lockfiles
```


## Pull Request Guidelines

- When creating a pull request a template is provided to ensure all checks are
  considered.
- This project is configured to use pre-commit tht will ensure some checks are
  performed automatically.
- Keep changes focused; avoid unrelated refactors in the same PR.
- Add or update tests for every change to production code.
- Ensure a whatsnew fragment is added, see
[`changelog/AGENTS.md`](changelog/AGENTS.md)


## Critical Development Gotchas

1. **xfail_strict Behavior**: Tests marked `@pytest.mark.xfail` that now PASS
  become FAILURES -> **remove xfail immediately when bug is fixed**

2. **Pre-commit Auto-fixes**: Hooks may auto-fix ISC001/COM812 conflicts ->
  re-stage files: `git add . && git commit`

3. **Lockfile Rebuilds**: Updating `requirements/locks/*.lock` files triggers
  slow conda environment rebuilds - only update when deps genuinely change

4. **Lazy Data Pitfalls**: Check `cube.has_lazy_data()` before operations; use
  `cube.lazy_data()` when appropriate; `cube.data` materializes arrays

5. **Coordinate Scope**: Not all coordinates span all dimensions -> use
  `cube.coords(dimensions=dim_indices)` to query specific axes

6. **Dask/NumPy Compatibility**: Some operations fail on lazy (Dask) arrays
  -> materialize with `cube.compute()` or check dask support before use

7. **Optional Dependencies**: cartopy, matplotlib lazy-imported -> handle
  ImportError gracefully for optional visualization features

8. **CF-Metadata Validation**: Custom coordinates must comply with CF
  standards via `cube.metadata`

9. **Cube Immutability**: Operations return new cubes; don't modify in-place
  (`cube.data[...] = x` won't affect the cube)

10. **Environment Reproducibility**: Set CPU features for consistent results:
  `export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX"`


## Getting Help

- **Discussions**:
  [GitHub Discussions](https://github.com/SciTools/iris/discussions)
- **Contributing Guide**: [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)
- **Developer Guide**: [docs/src/developers_guide/](docs/src/developers_guide/)
- **Code of Conduct**: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)


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
