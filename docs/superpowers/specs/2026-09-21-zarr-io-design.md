# Native Zarr I/O for Iris

**Status:** design proposed, awaiting review
**Date:** 2026-09-21
**Baseline:** `brownfield` at `b0ab90b98`, Iris `3.17.0.dev4`
**Verified against:** zarr-python 3.4.0, numcodecs 0.16.5, Python 3.13

Line references in this document are accurate as of the baseline commit.
Claims marked **[verified]** were checked by running code against the versions
above; everything else is read from source or specification.

Closes SciTools/iris#6977, #6979, #6980 (sub-issues of #6961).

---

## 1. Summary

Iris can already read and write Zarr, but only through the netCDF-c NCZarr
driver, which implements the **Zarr version 2** specification and writes scalar
variables as length-one arrays. This document specifies **native Zarr support**
built on zarr-python: reading Zarr version 2 and version 3 stores, and writing
version 3 stores.

The work is structured as seven pull requests against the `brownfield` branch.
Four of them are behaviour-preserving refactors that promote
`iris.fileformats.cf` to a package and extract the format-agnostic CF machinery
out of `iris.fileformats.netcdf`; three add user-visible Zarr capability on top.
No pull request both moves code and changes behaviour.

The central design move is a `CFDataset` abstraction: a narrow, explicit
interface describing what a CF-conforming array store must provide, with one
implementation per backend. The CF variable classes, the loader and the saver
are rewritten against that interface instead of against the netCDF4 Python API.

Every rule in §4 is derived from the Zarr and CF specifications. Real published
stores are cited as evidence that a code path will be exercised, and where such
a store is non-conforming it is named as a malformation and handled with a
warning — never by bending the reader to fit one publisher's files.

---

## 2. What exists today

### 2.1 The CF layer reaches straight into netCDF4

`lib/iris/fileformats/cf.py` (1721 lines) classifies the variables in a file
into `CFDataVariable`, `CFCoordinateVariable`, `CFGridMappingVariable` and so
on. It is nominally format-agnostic, but in practice it calls netCDF4 directly:

- `CFVariable.__init__` (cf.py:92-118) calls `data.ncattrs()` and
  `data.group().filepath()`.
- `CFVariable.__getattr__` (cf.py:198-206) forwards every unknown attribute to
  the underlying netCDF4 variable, conflating **properties** (`shape`, `dtype`,
  `dimensions`) with **CF attributes** (`standard_name`, `units`). It caches the
  result onto the instance, so the touch-tracking used by `cf_attrs_used()`
  registers only the first access.
- `CFReader.__init__` (cf.py:1336) branches on `self._dataset.file_format`, and
  (cf.py:1356) calls `set_auto_chartostring(False)`.

Across `cf.py`, `_nc_load_rules/` and `netcdf/`, there are 86 `getattr(cf_*`
or `hasattr(cf_*` call sites and 57 direct uses of netCDF-only APIs
(`ncattrs`, `getncattr`, `setncattr`, `chunking()`, `file_format`).

### 2.2 The loader and saver are generic code in a netCDF-shaped box

`netcdf/loader.py` (979 lines) and `netcdf/saver.py` (3190 lines) are mostly CF
logic. The genuinely netCDF-specific parts are small and identifiable:

| Module | netCDF-only content |
|---|---|
| `loader.py` | `_get_cf_var_data` chunking and proxy selection; VLEN handling; `_bytecoding_datasets` string machinery |
| `saver.py` | dataset open/close; `_setncattr`; `_create_cf_dimensions`; `_create_generic_cf_array_var`; write proxies; `_dask_locks` |

Everything else — `CFNameCoordMap`, `_get_dim_names`, `_add_aux_coords`,
`_add_cell_measures`, `_add_aux_factories`, `_create_cf_grid_mapping`, the CF
constant tables, `ChunkControl` — is format-agnostic and currently
unreachable from any other format.

### 2.3 The load and save chain assumes a single seekable file

A Zarr store is a **directory**, or a URL. Both break the existing chain:

- `iris.io.load_files` (io/__init__.py:241) does `open(fn, "rb")` on every path,
  which raises `IsADirectoryError` for a store.
- `iris.io.expand_filespecs` (io/__init__.py:196-199) calls
  `Path(fn.removeprefix("//")).expanduser().absolute()` on every spec, which
  mangles `s3://bucket/data.zarr` into `/<cwd>/s3:/bucket/data.zarr`.
- `iris.loading._generate_cubes` (loading.py:110) raises
  `ValueError("Iris cannot handle the URI scheme: %s")` for any scheme that is
  not `file`, `http`, `https`, `nczarr` or `data`.
- `iris.io.save` (io/__init__.py:530) sends a `CubeList` to a per-cube append
  loop unless `"iris.fileformats.netcdf" in saver.__module__`.
- `iris.mesh.load_meshes` carries a second copy of the same scheme dispatch in
  `netcdf/ugrid_load.py:156-183`.

### 2.4 Zarr facts that shape the design

All **[verified]** against zarr-python 3.4.0:

- Version 3 arrays carry `metadata.dimension_names` as a tuple. Version 2
  arrays have no such field; zarr-python raises `ValueError("Zarr format 2
  arrays do not support dimension names.")` if you try. Version 2 stores use
  the xarray convention of an `_ARRAY_DIMENSIONS` attribute holding a list.
- Scalar arrays have `shape == ()` and `dimension_names is None`.
- Attributes are JSON. NumPy scalars, NumPy arrays, `bytes` and `numpy.int64`
  all raise `TypeError: Object of type ... is not JSON serializable`. Python
  floats, ints, strings, lists and `numpy.str_` are accepted. A failed
  assignment leaves the in-memory attributes mapping dirty, so every later
  write to that mapping fails with the **first** error, not its own.
- `float("nan")` is accepted and round-trips, written as the bare token `NaN`,
  which is not valid strict JSON.
- Reads return a plain `numpy.ndarray`. Zarr applies **no** masking and **no**
  `scale_factor`/`add_offset` handling; netCDF4 does both automatically.
- `zarr.Array` is picklable, and `dask.array.from_zarr` produces an unmasked
  `numpy.ndarray` dask meta.
- `zarr.consolidate_metadata` warns that consolidated metadata is not part of
  the version 3 specification.

Two further facts come from the Zarr *specifications* rather than from
zarr-python, and they set the boundary between what Iris must support and what
Iris must merely defend against:

- An attribute value is **an arbitrary JSON literal**. Nested objects and
  arrays are legal Zarr. CF's attribute model is flat scalars and strings, so
  any conforming Zarr store may carry attributes that CF cannot express.
- An array's `fill_value` is a **required** version 3 metadata field. For a
  float the permitted encodings are a JSON number, `"NaN"`, `"Infinity"`,
  `"-Infinity"`, or the hex string form `"0xYYYYYYYY"`; base64 is not among
  them. CF's `_FillValue` is an *optional attribute*. These are two different
  things, and the design reconciles them rather than conflating them.

---

## 3. Constraints and decisions taken

These were agreed before design and are not revisited here.

| Decision | Choice |
|---|---|
| Dependency | `zarr` is **optional** and lazily imported. Absent, Zarr tests skip and Zarr use raises a clear `ImportError`. |
| Module layout | Extract the format-agnostic CF machinery out of `fileformats/netcdf/` into generic siblings. |
| Write support | **Version 3 only**, but the saver must be shaped so version 2 write is a later, small addition. |
| Read support | **Version 2 and version 3.** |
| Deprecation | Relocated public names keep working and warn on use. |
| Remote stores | URLs are passed through to zarr-python. Iris adds no dependency and handles no credentials. |
| Groups | Root group by default; a `group=` keyword selects another, on both load and save. |
| Real-world data | NOAA GFS from dynamical.org, and ESA EOPF Sentinel Zarr samples. |

Two further constraints come from the repository's own agent rules
(`lib/iris/AGENTS.md`): no dynamically generated attributes, no `getattr`
string dispatch, and `__getattr__` only in existing deprecation shims. The
design honours these; where it cannot, it says so.

---

## 4. Design

### 4.1 Module layout

`iris.fileformats.cf` becomes a **package**. It is where all format-agnostic
CF machinery ends up, and a flat module cannot hold a dataset abstraction, a
loader and a saver alongside 1721 lines of variable classification.

```
lib/iris/fileformats/
    cf/
        __init__.py      # public surface: re-exports + __all__, nothing else
        dataset.py       # NEW  CFDataset / CFDatasetVariable
        loader.py        # NEW  generic CF -> Cube  (from netcdf/loader.py)
        saver.py         # NEW  generic Cube -> CF  (from netcdf/saver.py)
        _variables.py    # CFVariable and the classic CF subclasses (was cf.py)
        _ugrid.py        # the three CFUGrid* variable classes      (was cf.py)
        _group.py        # CFGroup                                  (was cf.py)
        _reader.py       # CFReader                                 (was cf.py)
    netcdf/
        _dataset.py      # NEW  NetCDFDataset, NetCDFDatasetVariable
        loader.py        # netCDF load_cubes + deprecating aliases
        saver.py         # netCDF Saver, save, save_mesh
        ...              # _thread_safe_nc, _bytecoding_datasets, _dask_locks unchanged
    zarr/
        __init__.py      # NEW  public: load_cubes, save, Saver
        _dataset.py      # NEW  ZarrDataset, ZarrDatasetVariable
        _decode.py       # NEW  CF masking and unpacking that netCDF4 does for free
        loader.py        # NEW
        saver.py         # NEW
```

Every public name that `iris.fileformats.cf` exports today is re-exported from
`__init__.py`, so `from iris.fileformats.cf import CFReader` is unaffected and
no deprecation is needed for the move itself. This is the shape `iris.mesh`
already has: a 36-line `__init__.py` over `components.py` and `utils.py`.

The private split follows the four kinds of thing already in `cf.py` rather
than an arbitrary line count: classic CF variable classes (cf.py:78-895),
the UGRID variable classes (cf.py:896-1116), the `CFGroup` mapping
(cf.py:1117-1280) and the `CFReader` (cf.py:1281-1721). Each lands under the
~1000-line guidance in `lib/iris/AGENTS.md`, and the UGRID seam is the same
one `iris.mesh` draws.

The re-export layer in `__init__.py` is the one `lib/iris/AGENTS.md`
discourages, taken deliberately: the alternative is renaming the most
depended-upon public module in this subtree. The rule exists to stop
indirection being invented; here it preserves an existing import path.

`_variables.py`, `_ugrid.py`, `_group.py` and `_reader.py` are private because
they are a file layout, not an API. Third-party code that reaches past
`iris.fileformats.cf` into a submodule is reaching into the split itself, which
is exactly the thing that should stay free to move.

`iris.fileformats.zarr` shadows the third-party `zarr` distribution by name.
Absolute imports mean `import zarr` inside that package still resolves to the
third-party one; xarray has the same arrangement. The package docstring says so.

### 4.2 The `CFDataset` interface

`cf/dataset.py` defines two abstract classes. They are deliberately small:
only what the CF variable classes, `cf/loader.py` and `cf/saver.py` actually
need.

```python
class CFDatasetVariable(ABC):
    """One named array in a CF-conforming dataset."""
    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype
    size: int
    fill_value: Any | None
    chunking: tuple[int, ...] | None   # None when the store is unchunked
    attributes: MutableMapping[str, Any]

    def __getitem__(self, keys) -> np.ndarray: ...
    def __setitem__(self, keys, values) -> None: ...


class CFDataset(ABC):
    """A CF-conforming array store, open for reading or writing."""
    location: str                       # path or URL, for messages and proxies
    variables: Mapping[str, CFDatasetVariable]
    dimensions: Mapping[str, int]
    attributes: MutableMapping[str, Any]

    def create_dimension(self, name: str, size: int) -> None: ...
    def create_variable(self, name, dtype, dimensions, *,
                        fill_value=None, **encoding) -> CFDatasetVariable: ...
    def sync(self) -> None: ...
    def close(self) -> None: ...
```

The split that matters is `attributes` versus everything else. Today
`cf_var.units` might be a CF attribute or a netCDF property and the caller
cannot tell. After this change, CF attributes are only ever reached through
`.attributes`, and properties are named fields.

`create_variable` takes `**encoding` against the general rule about keyword
passthrough, because the accepted keys are backend-specific by nature
(`zlib`/`complevel`/`chunksizes` for netCDF, `compressors`/`chunks`/`shards`
for Zarr). Each implementation documents its own accepted keys, and
`cf/saver.py` never constructs them.

### 4.3 `CFVariable.__getattr__` and backwards compatibility

`CFVariable.__getattr__` stays, because it is public API. All 86 internal
`getattr`/`hasattr` call sites move to `.attributes`, so nothing inside Iris
depends on it.

For a netCDF-backed variable it behaves exactly as now. For a Zarr-backed
variable it raises `TypeError` with a message naming `.attributes` as the
replacement. `TypeError` rather than `AttributeError` is deliberate: an
`AttributeError` is swallowed by `hasattr`, which would turn a hard failure
into a silent wrong answer in third-party code.

### 4.4 Reading

**Dimension names.** `ZarrDatasetVariable.dimensions` reads
`metadata.dimension_names` on version 3 and the `_ARRAY_DIMENSIONS` attribute
on version 2. A scalar array yields `()`. An array with neither, and rank
greater than zero, raises `ValueError` naming the array: synthesising names
would silently produce wrong cubes. `ZarrDataset.dimensions` is then the union
of the per-array names with their sizes, and a name used at two different sizes
is an error.

NCZarr-written stores use a `_scalar_` pseudo-dimension for scalar variables.
`cf.py` already handles this in `CFVariable.spans` (cf.py:179-184) via
`_NCZARR_SCALAR_DIMENSION`. The three subclasses that override `spans`
(`CFBoundaryVariable` cf.py:415, `CFClimatologyVariable` cf.py:491,
`CFLabelVariable` cf.py:814) do **not**, which is a latent bug on the existing
NCZarr path. It is fixed in PR 2 with its own changelog entry.

**Fill values.** Zarr's array-level `fill_value` and CF's `_FillValue`
attribute are different things — see §2.4 — so Iris reads
`Array.fill_value`, which is required to exist and to be typed, and falls back
to the `_FillValue` attribute only when the array carries no usable one. That
ordering follows from the two specifications and holds for every store, not
just the ones that have been looked at.

A `_FillValue` attribute whose value is not a number of the array's type is a
CF violation on the producer's side: CF §2.5.1 requires "the scalar attribute
with the name `_FillValue` and of the same type as its variable". It is
nevertheless legal *Zarr*, because an attribute value may be any JSON literal.
Iris therefore ignores such a value and emits an `IrisCfLoadWarning` naming the
variable and the offending value. Warning rather than raising, because the
array-level `fill_value` already gives the correct answer and refusing to load
published data over a producer's metadata bug helps nobody.

The NOAA GFS store is a live instance: the `_FillValue` attribute of every
`float32` data variable is the string `'AAAAAAAA+H8='` **[verified]** — base64
of a `float64` NaN — while `Array.fill_value` is correctly `float32` NaN. This
is a **file malformation to defend against, not a property of Zarr.** It is
cited here as evidence that the warning path will be exercised in the wild, and
the cut-down fixture keeps the attribute verbatim so the test suite covers it.

**Attributes that are not scalars or strings.** This one *is* a property of
Zarr: the version 3 specification says an attribute value can be an arbitrary
JSON literal, and CF has no nested-attribute concept, so the gap exists for
every conforming store. It is also an established habit rather than one team's
slip — the GFS store gives `valid_time` a `statistics_approximate` attribute
whose value is `{'min': ..., 'max': ...}`, and the ESA EOPF product root
carries `stac_discovery`, `other_metadata` and `processing_history` as nested
objects **[verified, two independent producers]**.

Iris carries such values onto the cube unchanged, does not interpret them, and
does not write them back through the netCDF saver. This is the provisional
answer; the durable one needs the attribute-model discussion recorded in §10.

**Masking and unpacking.** netCDF4 applies `_FillValue`/`missing_value`
masking, `valid_min`/`valid_max`/`valid_range` masking and
`scale_factor`/`add_offset` unpacking for free. Zarr does none of it, so
`zarr/_decode.py` implements the same CF rules, applied inside the data proxy
so that laziness is preserved. It is Zarr-side rather than shared because the
netCDF path gets the behaviour from the C library and has nothing to reuse.

**Laziness.** `ZarrDataProxy` mirrors `NetCDFDataProxy`: it holds the store
location, the array path and the decode parameters, and opens the array on each
`__getitem__`. Going through a proxy rather than `dask.array.from_zarr`
preserves three things that matter:

1. `CHUNK_CONTROL` keeps working identically across formats.
2. The dask meta stays a **masked** array, as everywhere else in Iris.
3. `iris._lazy_data` (lines 302 and 334) special-cases `NetCDFDataProxy` to
   build a dask cache key from `repr(data)`. `ZarrDataProxy` is registered
   alongside it so Zarr arrays get the same graph-level caching.

`CFDatasetVariable.chunking` returns the store's own chunk shape, which
`cf/loader.py` uses exactly as it uses `chunking()` today, so
`ChunkControl.from_file()` works unchanged. For a sharded version 3 array,
`chunking` returns `Array.shards`, which is `None` on an unsharded array
**[verified]**, falling back to `Array.chunks`. The shard is the unit a reader
actually fetches, so it is the right dask chunk.

Zarr's own async concurrency is left at its default. Iris does not write to
`zarr.config`; the interaction between `zarr_async_concurrency` and the dask
worker count is documented instead.

### 4.5 Writing

Writing is version 3 only. The shape that keeps version 2 cheap to add later is
that **`cf/saver.py` never mentions a Zarr version.** Everything version-specific
lives in `ZarrDataset`:

- how dimension names are recorded (`dimension_names` field versus
  `_ARRAY_DIMENSIONS` attribute);
- which codecs are available;
- whether sharding is offered.

`ZarrDataset.__init__` takes `zarr_format: int = 3`. Today it raises
`NotImplementedError` for anything but 3, in one place. Adding version 2 write
means implementing those three points and deleting that guard.

**Dimensions.** Zarr has no dimension objects. `ZarrDataset.create_dimension`
records name and size in a plain dict, and `create_variable` writes the names
onto each array. The dict is what lets the saver check for the size conflicts
it checks for today.

**Attributes.** Values are converted to JSON-native types **before** being
handed to zarr, never after, because a rejected value poisons the attributes
mapping **[verified]**. The conversion is: NumPy scalar to its Python
equivalent, NumPy array to a list, `bytes` to `str` via ASCII, everything else
unchanged. Anything still unserialisable raises with the attribute name in the
message.

This loses the NumPy dtype of attributes. That is the same trade xarray makes,
and it is what makes the output readable by every other Zarr consumer. The
alternative offered in #6961 — a base64 envelope carrying `dtype` and `raw` —
is **rejected** for general use: it would make Iris output opaque to xarray and
to the GeoZarr tooling, which is the whole point of writing Zarr rather than
netCDF. Round-trip tests therefore compare attribute **values**, not dtypes.

This is the provisional answer to the wider question of how the CF attribute
model maps onto JSON in both directions — the same question the nested-object
paragraph in §4.4 runs into from the read side. §10 records it as parked for a
dedicated discussion; nothing else in the design depends on how it is settled.

Two attributes are exempt because their dtype is load-bearing: `_FillValue` and
`missing_value` are written as the Python scalar matching the array dtype, and
the array's own `fill_value` is set to the same value, so an unwritten chunk
reads back as missing.

`_bytes_if_ascii` and `_setncattr` (saver.py:271, 290) coerce attribute values
to ASCII `bytes` for netCDF. That coercion stays on the netCDF path only.

**Masked data.** Masked arrays are stored as `data.filled(fill_value)` with the
array's `fill_value` set to match, as recommended in #6961. Writing a separate
mask array is rejected as over-engineering for no CF benefit.

**Deferred writes.** The netCDF saver defers lazy writes by closing the file,
returning a `Delayed`, and reopening per chunk. Zarr needs none of that: the
store stays valid and concurrent writes to distinct chunks are safe. The Zarr
saver therefore builds one `da.store(..., compute=False, lock=False)` over all
lazy sources and returns it from `save(..., compute=False)`, matching the
netCDF signature without the reopen dance. This is also why the NCZarr special
case in `Saver.__exit__` (saver.py:485-498) is untouched: it is a different,
older path.

**Encoding.** `zlib`/`complevel` map to a Blosc or Gzip codec, `shuffle` to
Blosc shuffle, `chunksizes` to `chunks`, `fletcher32` to Crc32c. `contiguous`,
`endian` and `least_significant_digit` have no Zarr equivalent and raise
`TypeError` if passed. Zarr adds `shards=` and `compressors=`, and
`consolidate_metadata=True` (default `True`, since consolidation is what makes
remote stores usable, with the zarr-python warning suppressed and explained).

### 4.6 Groups

`load_cubes(..., group=None)` and `save(..., group=None)` both default to the
root group. A non-`None` `group` is a `/`-separated path within the store.

Iris loads **one** group per call, like xarray. Nested groups are not walked:
a Zarr store can legitimately hold unrelated datasets, and merging them into
one `CubeList` would be wrong. `iris.load("store.zarr")` therefore returns the
cubes in the root group, and a store whose root group holds no data variables
raises a message that names the groups that do.

### 4.7 Format detection and the load/save chain

A new `FormatSpecification` named `"Zarr"` is registered with a new
`ZarrStore` file element that, like `UriProtocol`, sets `requires_fh = False`
so it is consulted when no file handle is available. It matches when:

- the target is a directory containing `zarr.json` (version 3) or `.zgroup`
  (version 2); or
- the target's path ends in `.zarr`; or
- the URI scheme is one of `s3`, `gs`, `az`, `abfs` and the path ends in
  `.zarr`.

Reading the marker file is what distinguishes a real store from a directory
someone happened to name `.zarr`, so it is tried first.

Five changes make the chain reach that specification:

1. `iris.io.load_files` skips `open(fn, "rb")` when the path is a directory,
   and passes `None` as the buffer.
2. `iris.io.expand_filespecs` leaves a spec with a recognised remote scheme
   alone instead of absolutising it.
3. `iris.loading._generate_cubes` routes the remote schemes to a new
   `iris.io.load_stores`, instead of raising.
4. `iris.io.find_saver` matches `.zarr` with or without a trailing separator,
   since `iris.save(cube, "out.zarr/")` is a natural thing to type.
5. `iris.io.save` gets a module-level `_CUBELIST_SAVER_MODULES` frozenset
   naming the modules whose `save` accepts a `CubeList` directly, replacing the
   `"iris.fileformats.netcdf" not in saver.__module__` substring test at
   io/__init__.py:530. Explicit and greppable; one line per new format.

The duplicated dispatch in `netcdf/ugrid_load.py:156-183` is updated in step
with `loading.py`, so `iris.mesh.load_meshes` accepts Zarr stores too. The
existing `_uri_is_nczarr` specification is untouched: NCZarr keeps working
exactly as it does now, and the native reader is only reached for stores that
are not addressed with an NCZarr URL fragment.

`docs/src/user_manual/tutorial/s3_io.rst` currently says S3 is unsupported and
recommends FUSE mounting. It is updated to describe the Zarr route.

### 4.8 Deprecations

These names move and gain warn-on-use aliases at their old locations, following
the pattern in `iris/experimental/ugrid.py`:

| Old name | New name |
|---|---|
| `iris.fileformats.netcdf.loader.CHUNK_CONTROL` | `iris.fileformats.cf.loader.CHUNK_CONTROL` |
| `iris.fileformats.netcdf.loader.ChunkControl` | `iris.fileformats.cf.loader.ChunkControl` |
| `iris.fileformats.netcdf.CFNameCoordMap` | `iris.fileformats.cf.saver.CFNameCoordMap` |
| `iris.fileformats.netcdf.CF_CONVENTIONS_VERSION` | `iris.fileformats.cf.saver.CF_CONVENTIONS_VERSION` |
| `iris.fileformats.netcdf.MESH_ELEMENTS` | `iris.fileformats.cf.saver.MESH_ELEMENTS` |
| `iris.fileformats.netcdf.SPATIO_TEMPORAL_AXES` | `iris.fileformats.cf.saver.SPATIO_TEMPORAL_AXES` |

`CHUNK_CONTROL` is the one that matters: it is genuinely shared, and users
reach it by module path. The alias is the **same object**, not a copy, so
existing code keeps working including inside a `with CHUNK_CONTROL.set(...)`.

These names do **not** move, because they are netCDF API and should stay that
way: `netcdf.save`, `netcdf.Saver`, `netcdf.load_cubes`, `netcdf.save_mesh`,
`netcdf.NetCDFDataProxy`, `netcdf.DEBUG`.

Warnings are emitted on **use**, not on import, so simply importing
`iris.fileformats.netcdf` stays quiet. The deprecation is marked
`.. deprecated:: 3.17` and follows NEP 29.

---

## 5. The programme

Seven pull requests against `brownfield`. Each carries the `Agentic` and
`Type: Feature Branch` labels, attributes the contribution to Claude, and adds
a changelog fragment crediting `` :user:`claude` ``.

### PR 1 — `iris.fileformats.cf` becomes a package, with tests first

`git mv cf.py cf/_variables.py` and split out `_ugrid.py`, `_group.py` and
`_reader.py` along the boundaries in §4.1, with an `__init__.py` that
re-exports the existing public names. Imports updated across the tree. **No
behaviour change and no API change**: the diff is a move plus import edits, so
a reviewer can skim it.

Brings in the unit test coverage from SciTools/iris#7259 by Martin Yeo, rebased
onto `brownfield` and credited in both the changelog and the pull request body.
That branch is named `cf_reader_zarr` and was written for exactly this
refactor; it adds roughly 2300 lines of coverage across nineteen test modules.
Landing it *here*, against today's behaviour, is what makes PR 2 reviewable:
the tests become a regression net that was written before the rewrite and does
not move with it. A comment on #7259 explains the overlap before the pull
request is opened.

### PR 2 — `CFDataset`, and the CF variable classes rewritten against it

Closes **#6977**. Adds `cf/dataset.py` and `netcdf/_dataset.py`. Rewrites the
86 `getattr`/`hasattr` sites and the 57 netCDF-API sites in the `cf` package,
`_nc_load_rules/` and `netcdf/` to go through the new interface. Makes
`CFVariable.__getattr__` raise on non-netCDF backends. Fixes the
`_NCZARR_SCALAR_DIMENSION` gap in the three overriding `spans` methods, with
its own changelog entry.

No behaviour change other than the `spans` fix. The suite from PR 1 must pass
untouched.

### PR 3 — Relocate the CF loader

`git mv netcdf/loader.py cf/loader.py`, leaving a `netcdf/loader.py` that holds
the netCDF specifics and the deprecating aliases. Still netCDF-only behaviour;
the whole existing suite passes unchanged. Large, mechanical, independently
verifiable.

### PR 4 — Zarr loading

Closes **#6979**. Adds `zarr` to the optional dependencies section of
`requirements/py3{12,13,14}.yml` and regenerates the lock files, once, here.
Adds `fileformats/zarr/` with `_dataset.py`, `_decode.py` and `loader.py`, the
format specification, and the five load-chain changes in §4.7.

After this pull request, `iris.load("store.zarr")` and
`iris.load("s3://bucket/store.zarr")` work for version 2 and version 3 stores.

### PR 5 — Relocate the CF saver

`git mv netcdf/saver.py cf/saver.py`, same shape as PR 3. The largest diff of
the seven and the one with no behaviour change at all, which is precisely why
it is on its own.

### PR 6 — Zarr saving

Closes **#6980**. Adds `zarr/saver.py`, the `ZarrDataset` write path, JSON
attribute conversion, masked-data filling, deferred writes, encoding
translation and `consolidate_metadata`. Registers the `zarr` saver and updates
the S3 documentation.

After this pull request, `iris.save(cubes, "out.zarr")` works.

### PR 7 — Real-world test data and benchmarks

Integration tests against the cut-down NOAA GFS and ESA EOPF samples, plus ASV
coverage: `ZarrSave` beside `NetcdfSave` in `benchmarks/benchmarks/save.py`,
and Zarr variants of `LoadAndRealise` in `benchmarks/benchmarks/load/`.
Depends on a companion pull request to `SciTools/iris-test-data`.

**Ordering rationale.** Refactor and feature alternate so that no feature pull
request is large, and every relocation is a separate, skimmable diff. Tests
land before the rewrite they protect. Loading lands before saving because it is
the higher-value half if the programme is cut short. PR 1 to PR 3 are
independent of any Zarr decision and could merge before the rest is agreed.

---

## 6. Testing

Per `lib/iris/tests/AGENTS.md`: pytest style, no network, no unittest classes
in new files.

**Unit.** `lib/iris/tests/unit/fileformats/zarr/` mirrors the module layout.
The `CFDataset` contract gets one shared test body run against both
implementations, so a netCDF/Zarr divergence fails loudly. Stores are built
in-memory or in `tmp_path`.

**Integration.** `lib/iris/tests/integration/zarr/` covers round trips through
`stock.realistic_4d_w_everything()`, following the shape of the existing
`integration/netcdf/test_nczarr.py`.

Note that the `["nczarr", "xarray"]` parametrisation in that module is **not**
about the xarray package: both are netCDF-c NCZarr URL modes, and `xarray` is
the mode that writes `_ARRAY_DIMENSIONS`.

**Cross-reader testing.** xarray is added as a **test-only** dependency, so
that a handful of integration tests can assert that xarray reads what Iris
writes. This is the check that justifies the JSON-native attribute decision in
§4.5: if xarray cannot read Iris output, the decision was wrong. The cost has
been measured — `conda install -n iris-dev -c conda-forge xarray` adds
**exactly one package** and updates nothing **[verified]**, because xarray's
runtime requirements (NumPy, pandas, packaging) are already Iris dependencies.
No `skip_gdal`-style guard is therefore needed, and xarray goes into the test
section of `requirements/py3{12,13,14}.yml` alongside the other test-only
packages. It lands in PR 6, with the saver it validates.

**Version 2 reading** is tested two ways. Synthetic fixtures are written by
zarr-python with `zarr_format=2` and `_ARRAY_DIMENSIONS` set by hand, since
zarr-python refuses to write version 2 dimension names itself **[verified]**.
Separately, the existing NCZarr `mode=xarray` save path already produces a
version 2 store with `_ARRAY_DIMENSIONS`, so it generates realistic version 2
fixtures with no new dependency, and gives a direct check that the native
reader and the NCZarr writer agree.

**Remote stores** are covered by an in-memory store and a local directory
store only. No test touches the network. The URL path is exercised by checking
that the right store is constructed, not by fetching.

**Skipping.** A single `pytest.importorskip("zarr")` fixture in the Zarr test
package. Iris without zarr installed must show no new failures.

---

## 7. Real-world test data

Two sources, both cut down before contribution to `SciTools/iris-test-data`
(fork `bjlittle/iris-test-data`, default branch `master`).

**NOAA GFS, dynamical.org** — the version 3 fixture. Confirmed by fetching
`https://data.dynamical.org/noaa/gfs/forecast/latest.zarr/zarr.json`:
`zarr_format: 3`, 33 consolidated members, CF attributes including
`standard_name`, `units`, `axis`, `calendar` and `_FillValue`, version 3
`dimension_names`, sharded Blosc/zstd codecs, and a scalar `spatial_ref`
variable carrying `crs_wkt`. Licensed CC BY 4.0; the attribution string and
DOI 10.5281/zenodo.18777399 travel with the data.

**This endpoint is documented as closing on 30 September 2026.** The subset has
therefore already been cut, ahead of the rest of the programme, and is held at
`~/projects/iris-zarr-testdata/gfs_forecast_sample.zarr`: 1.4 MB, 35 files, 12
members, 2 forecast initialisations x 8 lead times x 120 x 120 points, for
`temperature_2m`, `wind_u_10m`, `wind_v_10m` and `precipitation_surface`. It
keeps the sharding codec, the scalar `spatial_ref` grid mapping, the
two-dimensional `valid_time` auxiliary coordinate, all source attributes
verbatim including the base64 `_FillValue`, and the `CC-BY-4.0` licence and
attribution in the root group.

**ESA EOPF Sentinel samples** — the version 2 and deep-group-hierarchy
fixture, exercising the `group=` keyword and the `_ARRAY_DIMENSIONS` path.

**Confirmed version 2 by reading the store** **[verified]**. For a Sentinel-2
L2A product under
`https://data.eodc.eu/collections/EOPF_ZARR/products/cpm_v270/S02MSIL2A/`,
`zarr.json` returns HTTP 404 while `.zgroup` returns HTTP 200 containing
`{"zarr_format": 2}`; opening the root with zarr-python reports
`zarr_format: 2`. Dimension names are carried on `_ARRAY_DIMENSIONS`
throughout. The service is migrating to version 3; until it does, this is a
more realistic version 2 sample than anything synthetic.

Two properties of this store earn it a place beyond "it is version 2":

- **The root group holds no arrays at all**, only subgroups:
  `conditions/geometry`, `conditions/mask/detector_footprint/{r10m,r20m,r60m}`,
  `conditions/meteorology/{cams,ecmwf}` and
  `measurements/reflectance/{r10m,r20m,r60m}`. Loading the default root group
  yields nothing, which makes the §4.6 design — raise a message naming the
  groups that *do* hold data — a necessity rather than a nicety.
- It uses `<U7` and `<U3` fixed-length unicode dtypes, which #6961 notes are
  not strictly Zarr-supported. A real store already contains them, so the
  reader needs a defined answer.

The `measurements/reflectance/r10m` group holds `b02`, `b03`, `b04` and `b08`
as `(10980, 10980)` `uint16` with `fill_value` 0, plus `x` and `y` coordinate
arrays — so a cropped window of one or two bands is the natural cut-down.
Each band carries `scale_factor`, `add_offset`, `units`, `long_name`,
`valid_min` and `valid_max` as flat, correct CF attributes — so this fixture
exercises the `zarr/_decode.py` unpacking path on real data — **and** a
redundant `_eopf_attrs` object duplicating them one level down
**[verified]**. The nested object is producer bookkeeping, not a relocation of
the CF metadata. It is the §4.4 nested-attribute case in its mildest form:
Iris reads the flat attributes and must simply carry the nested one through
without choking on it.

Target size is a few megabytes each: one or two variables, a handful of time
steps, a cropped spatial window, with the group structure and codec variety
preserved because that is what is being tested.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| dynamical.org closes before the subset is pulled | Pull within days; the design does not otherwise depend on it |
| PR 5 is a 3000-line diff | Pure `git mv` plus import edits, no behaviour change, full suite green |
| CF-Zarr conventions are still a moving target | Iris writes plain unprefixed CF attributes, which is what the draft and xarray both do; no `zarr_conventions` block is written until the specification settles |
| zarr-python uses Effective Effort Versioning, not semantic versioning | Pin `>=3.0.8`, which is above the known data-loss bug, and rely on CI to catch drift |
| Attribute dtype loss surprises someone | Documented; round-trip tests assert values not dtypes; xarray reads the output (§6); base64 remains available as a later opt-in |
| Designing against one publisher's files produces brittle code | Every behaviour in §4 is justified from the Zarr and CF specifications first; a real file is cited only as evidence that a path will be exercised. Where a file is non-conforming it is labelled a malformation and handled with a warning, never by bending the reader (§4.4) |
| Consolidated metadata is not in the version 3 specification | Written by default because remote stores are unusable without it; controllable by keyword; readers that ignore it still work |

---

## 9. Out of scope

- Writing Zarr version 2.
- Walking or loading multiple groups in one call.
- GeoZarr and the proposed Zarr-CS coordinate-system convention.
- Icechunk, virtual Zarr and Kerchunk reference stores.
- Multi-process write coordination beyond what zarr-python guarantees.
- Any change to the existing NCZarr path other than the `spans` bug fix.
- Adding an fsspec, s3fs or gcsfs dependency to Iris.

---

## 10. Decisions taken at review, and what is still parked

**Settled.**

1. **Module layout** — `iris.fileformats.cf` becomes a package. §4.1 rewritten;
   PR 1 does the move on its own, with no behaviour change.
2. **Cross-reader testing** — xarray is added as a test-only dependency. The
   cost was measured at exactly one extra conda package, so the `skip_gdal`
   fallback pattern is not needed. §6.
3. **EOPF's role** — confirmed Zarr version 2 by reading the store, so it keeps
   its role as the version 2 and deep-group fixture. §7.

**Parked for a dedicated discussion: the CF attribute model on JSON.**

This is one question with two faces, and the design currently answers both
provisionally:

- *Writing* (§4.5): JSON-native conversion, losing the NumPy dtype of
  attributes, versus a base64 envelope carrying `dtype` and `raw` as offered in
  #6961.
- *Reading* (§4.4): what to do with an attribute value that is a legal JSON
  object or array and has no CF equivalent — carry it opaquely, flatten it,
  drop it with a warning, or define a convention.

The provisional answer in both cases is **JSON-native**: convert on write,
carry through untouched on read. It is what xarray does and what makes Iris
output readable by the wider Zarr ecosystem. The design is deliberately
arranged so that changing this answer later touches only the attribute
conversion in `zarr/_dataset.py` — nothing in the loader, the saver or the
`CFDataset` interface depends on it.

A separate issue tracks the discussion, with the relevant specification
citations (Zarr v3 "arbitrary JSON literal"; CF §2.5.1 "of the same type as its
variable") and the two real-world examples from §4.4.

---

## 11. References

- SciTools/iris#6961 — Zarr I/O, parent issue
- SciTools/iris#6977, #6979, #6980 — the three sub-issues
- SciTools/iris#7259 — `cf.py` unit test coverage, by Martin Yeo
- Zarr specifications: https://zarr-specs.readthedocs.io/
- zarr-python: https://zarr.readthedocs.io/
- CF conventions for Zarr: https://github.com/zarr-conventions/CF
- NCZarr: https://docs.unidata.ucar.edu/nug/current/ncZarr_head.html
- NOAA GFS archive: https://data.dynamical.org/noaa/gfs/forecast/
