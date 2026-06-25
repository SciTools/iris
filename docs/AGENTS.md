# docs/AGENTS.md

Documentation-specific agent instructions for the [Iris](https://scitools-iris.readthedocs.io/) project.
This file takes precedence over the root [`AGENTS.md`](../AGENTS.md) for everything under `docs/`.


## Structure

```
docs/
    Makefile               # Top-level doc build entry point
    gallery_code/          # Gallery example scripts (general/, meteorology/, oceanography/)
    gallery_tests/         # Tests that execute gallery examples
    src/
        conf.py            # Sphinx configuration
        common_links.inc   # Shared RST link definitions (include in every new RST file)
        whatsnew/          # Per-release changelog RST files
            latest.rst     # Current unreleased changes
            index.rst      # Whatsnew index
        developers_guide/  # Contributor documentation
        userguide/         # User-facing guides
        user_manual/       # Detailed user reference
        sphinxext/         # Custom Sphinx extensions
```


## Building the Docs

```bash
# Full build (from docs/ directory)
make html

# Skip gallery (faster)
make html-noplot

# Skip API docs (faster). There will be build warnings.
make html-noapi

# Skip both gallery and API (fastest). There will be build warnings.
make html-quick

# Clean build artifacts
make clean

# Live rebuild on file changes (requires sphinx-autobuild)
cd docs/src && make livehtml
```

Build output goes to `docs/src/_build/html/`.


## Whatsnew Entries

- See [`AGENTS.md`](../changelog/AGENTS.md) in the `/changelog` directory. 


## RST / Sphinx Conventions

- All new RST files must include `.. include:: ../common_links.inc` (adjust relative path as needed) to access shared link definitions.
- Use NumPy-style docstrings for all Python API pages; Sphinx autodoc pulls these automatically.
- Warnings are treated as errors in the standard build (`-W --keep-going`). Fix all Sphinx warnings before merging.
- Do **not** commit build artifacts (`docs/src/_build/`).
- Cross-reference Iris symbols with ``:class:`iris.cube.Cube` ``, ``:func:`iris.load` ``, etc.
- Gallery scripts live under `docs/gallery_code/` and must be valid standalone Python files executable by `matplotlib` / `sphinx-gallery`.


## Gallery Examples

- Each gallery script must have a module-level docstring that becomes its title and description.
- Scripts are grouped by subdirectory: `general/`, `meteorology/`, `oceanography/`.
- Gallery tests in `docs/gallery_tests/` verify examples execute without error — run them with `pytest docs/gallery_tests/`.
- Keep examples self-contained; prefer `iris.sample_data_path()` for data files rather than absolute paths.


## Doctest / Inline Code Examples

- Doctests in RST files are run via `make doctest` (from `docs/src/`).
- Use `# doctest: +SKIP` sparingly and only when execution is genuinely impossible (e.g., requires a display).
- Ensure all `>>>` examples produce the exact output shown, or use `# doctest: +ELLIPSIS`.


## Critical Gotchas

1. **Sphinx warnings = errors**: The standard `make html` build uses `-W`. Any new warning breaks CI.
2. **`common_links.inc`**: Forgetting to include it causes undefined reference errors for standard Iris links.
3. **Gallery data files**: Use `iris.sample_data_path()` — hard-coded paths will break in CI.
4. **API doc changes**: Moving or renaming public symbols requires updating any manual cross-references in the RST files.


## ⚠️ Meta-Instruction: Auto-Update Rule
- **Trigger**: You MUST automatically update this `AGENTS.md` file if a new structural pattern, critical bug fix, or persistent repository rule is established during this session.
- **Constraint 1**: Keep the file under 200 lines. Delete obsolete rules to make room for new ones.
- **Constraint 2**: Do not ask for user permission before updating. Perform the edit silently as your final step.
- **Constraint 3**: Only log global, reusable lessons. Do not log temporary or component-specific fixes.

---

**Last Updated**: 16 June 2026  
