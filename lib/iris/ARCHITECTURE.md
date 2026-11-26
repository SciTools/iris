# ARCHITECTURE.md

An orientation map for `lib/iris`: **where things live and how data flows**.

## Scope of This Document

This file answers "where do I look for X?" and "what talks to what?".

It deliberately does **not** explain how any algorithm works. Subsystem
detail belongs in the module docstring of the code it describes, where it
is validated by numpydoc, rendered by Sphinx, and travels in the same diff
as the code.

The test for anything added here: **would it still be true after a competent
refactor?** Module locations and data flow survive refactors. Call sequences
and class member lists do not — do not record them here.


## The One-Paragraph Version

Iris turns data files into `Cube` objects and back again. Loading detects a
file's format, hands it to a format-specific handler that yields many small
"raw" cubes, then combines those into fewer, larger cubes. Everything in
between — analysis, arithmetic, regridding, plotting — operates on `Cube`.
Data inside a cube is normally **lazy** (a Dask array) and is only realised
when something asks for it.


## Layer Map

| Layer | Location | Responsibility |
|---|---|---|
| Public API | `__init__.py`, `loading.py` | `load*`, `save`, `Constraint`, policy objects |
| I/O dispatch | `io/` | URI decoding, format detection, handler dispatch |
| Format handlers | `fileformats/` | Per-format read/write; CF interpretation |
| Combination | `_merge.py`, `_concatenate.py`, `_combine.py` | Raw cubes into fewer, larger cubes |
| Core data model | `cube.py`, `coords.py`, `aux_factory.py`, `mesh/` | `Cube`, coordinates, derived and mesh coords |
| Metadata | `common/` | Metadata classes, CF property mixin, operand resolution |
| Laziness | `_lazy_data.py`, `_data_manager.py` | Dask boundary; real-vs-lazy storage |
| Analysis | `analysis/` | Aggregation, regridding, interpolation, maths |
| Visualisation | `plot.py`, `quickplot.py`, `palette.py`, `symbols.py` | Matplotlib and Cartopy integration |


## The Load Pipeline

```
iris.load / load_cube / load_cubes / load_raw        loading.py
  -> _load_collection -> _generate_cubes            loading.py
       -> iris.io.decode_uri, expand_filespecs      io/__init__.py
       -> FORMAT_AGENT.get_spec                     io/format_picker.py
            matches on magic number, leading line,
            file extension or URI protocol
       -> format handler yields raw cubes           fileformats/<format>.py
  -> constraint filtering (_CubeFilterCollection)   loading.py
  -> combination                                    _combine.py
       -> CubeList.merge      -> ProtoCube          _merge.py
       -> CubeList.concatenate                      _concatenate.py
```

Key points that are stable over time:

- **`load_raw` skips the combination step.** That is its entire purpose, and
  it is the fastest way to see what a format handler actually produced.
- **Format handlers are registered, not hard-coded.** `FORMAT_AGENT` in
  `fileformats/__init__.py` holds every `FormatSpecification`. To find out
  which formats are supported, read that registry — do not rely on any list,
  including this one, staying current.
- **Handlers are generators** yielding one `Cube` per field or variable.
  A single PP file commonly yields thousands.
- **Combination behaviour is policy-driven**, not fixed: `iris.LOAD_POLICY` /
  `iris.COMBINE_POLICY` (`CombineOptions` in `_combine.py`) decide whether
  merge, concatenate, both, or neither is applied, and in which order.
- **Failures are captured, not discarded.** `LOAD_PROBLEMS` in `loading.py`
  collects objects that could not be loaded, for inspection after the fact.

### Merge versus concatenate

Both combine a `CubeList` into fewer cubes, and they are not interchangeable:

- **merge** (`_merge.py`) builds a *new* dimension from cubes that differ only
  by scalar coordinate values.
- **concatenate** (`_concatenate.py`) extends an *existing* dimension by
  joining cubes along it.

### NetCDF specifics

`fileformats/netcdf/loader.py` drives CF interpretation through a rules engine
in `fileformats/_nc_load_rules/`, which translates CF metadata into Iris
coordinates. `fileformats/cf/` provides the CF-level view of a file: it
classifies variables (`_variables.py`), groups them (`_group.py`) and reads
them from a dataset (`_reader.py`, the only part coupled to netCDF).
Never `import netCDF4` directly — use
`fileformats/netcdf/_thread_safe_nc.py`.


## The Save Pipeline

`iris.save` in `io/__init__.py` dispatches on file extension to a
format-specific saver (`fileformats/netcdf/saver.py`, `pp_save_rules.py`,
and so on). Saving is *not* a registry lookup — it is extension-based, which
is a genuine asymmetry with loading and a common source of confusion.


## Core Data Model

- **`Cube`** (`cube.py`) — an n-dimensional data array plus CF metadata.
- **`DimCoord`** — describes exactly one dimension; points must be monotonic.
- **`AuxCoord`** — may span any number of dimensions, including none.
- **`CellMeasure`**, **`AncillaryVariable`** — further CF payloads on a cube.
- **`AuxCoordFactory`** (`aux_factory.py`) — manufactures a coordinate on
  demand from a formula over other coordinates, e.g. hybrid height. This is
  why a cube can report a coordinate that is not stored anywhere.
- **`MeshXY`**, **`MeshCoord`** (`mesh/`) — unstructured UGRID support.

Invariants that hold across the codebase:

- Operations return **new** cubes; they do not mutate their inputs.
- Metadata is carried by namedtuple-like classes in `common/metadata.py`;
  `CFVariableMixin` (`common/mixin.py`) gives every object the same CF
  property interface (`standard_name`, `units`, `attributes`, `rename()`).
- Comparison can be **lenient or strict** (`common/lenient.py`). Lenient is
  the default for most user-facing operations, which is why two objects may
  compare equal despite differing metadata.
- `Resolve` (`common/resolve.py`) decides the metadata and dimension mapping
  of the result when two cubes are combined; cube arithmetic goes through it.


## Laziness

Laziness is a cross-cutting concern, not a feature of one module.

- `DataManager` (`_data_manager.py`) holds *either* a real array or a lazy
  one, never both, for cubes and coordinates alike.
- `_lazy_data.py` is the Dask boundary: `as_lazy_data`, `as_concrete_data`,
  `co_realise_cubes`, and chunking policy.
- `cube.data` **realises** data. `cube.core_data()` and `cube.lazy_data()` do
  not. Library code should avoid `.data` unless realisation is intended.


## Structural Oddities Worth Knowing

These are real and will otherwise cost you a confused half-hour:

- **Several modules are very large** — `common/resolve.py` (~2600 lines),
  `fileformats/pp.py` (~2400), `cube.py` (~5600). They are legacy. Do not
  grow them unnecessarily, but do not opportunistically split them either.
- **GRIB support is an external package** (`iris_grib`), loaded through a
  registered handler; it is not vendored here.
- **`iris/etc/`** holds colour palettes and a config template — not the CF
  reference tables, which live in the repository-root `etc/`. CF standard
  names reach the code as the generated module `std_names.py`.
- **`experimental/`** is genuinely unstable API and exempt from the usual
  deprecation guarantees.


## Where Do I Look For...?

| Question | Start here |
|---|---|
| Why did my file not load? | `LOAD_PROBLEMS` in `loading.py` |
| Why was my format not recognised? | `FORMAT_AGENT` in `fileformats/__init__.py` |
| Why did my cubes not merge? | `_merge.py`; compare against `load_raw` output |
| Why is this cube's metadata wrong after arithmetic? | `common/resolve.py` |
| Why do these two objects compare equal? | `common/lenient.py` |
| Why did my data realise unexpectedly? | `_data_manager.py`, `_lazy_data.py` |
| Where does this coordinate come from? | `aux_factory.py` |
| How is a CF attribute interpreted? | `fileformats/_nc_load_rules/` |


## Maintaining This File

- Keep it under ~200 lines. It is an index, not a manual.
- Record structure and data flow only. Push explanation down into module
  docstrings.
- Do not enumerate things the code already enumerates (format lists, class
  members, function signatures). Point at the authoritative location instead.
- If a change here would also need a change to a module docstring, change the
  docstring and leave this file alone.
