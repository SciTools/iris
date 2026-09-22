# Native Zarr I/O for Iris

> **This is a living document.** §12 is the single progress record for the
> programme — pull request states, open questions and the decision log. Update
> §12 as work lands; do not track status anywhere else.

| | |
|---|---|
| **Phase** | Design, awaiting approval |
| **Progress** | 0 of 7 pull requests raised; merge-back not started — see §12.1 |
| **Next action** | Spec approval, then the implementation plan |
| **Blocked on** | Nothing |
| **Branch** | `zarr-io-design` on `bjlittle/iris`, targeting `SciTools/iris:brownfield` |
| **Delivery** | Seven pull requests into `brownfield`, then one merge-back into `main` — §5 |
| **Resolves** | SciTools/iris#6977, #6979, #6980 — at merge-back, not before |

**Created:** 2026-09-21 · **Last updated:** 2026-09-22 (see §12.6)
**Baseline:** `brownfield` at `b0ab90b98`, Iris `3.17.0.dev4`
**Verified against:** zarr-python 3.4.0, numcodecs 0.16.5, Python 3.13

Line references in this document are accurate as of the baseline commit.
Claims marked **[verified]** were checked by running code against the versions
above; everything else is read from source or specification.

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
        _variables.py    # every CFVariable subclass, classic and UGRID (was cf.py)
        _group.py        # CFGroup                                      (was cf.py)
        _reader.py       # CFReader                                     (was cf.py)
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

The private split cuts at the three seams already in `cf.py`, each of which
is a different kind of thing: the variable classifiers (cf.py:78-1116), the
`CFGroup` mapping that collects them (cf.py:1117-1280) and the `CFReader`
that drives the whole thing (cf.py:1281-1721). That gives `_variables.py` at
roughly 1120 lines, `_group.py` at 180 and `_reader.py` at 460.

`_variables.py` sits just over the ~1000-line aim in `lib/iris/AGENTS.md`,
and stays there deliberately. An earlier draft of this section put the three
`CFUGrid*` classes in a fourth module to get under the number; that was a
line count in search of a rationale. The UGRID classes are not a separate
concern from the classic ones — they are `CFVariable` subclasses with the
same shape (`cf_identity`, `cf_identities`, a classmethod `identify`), doing
the same job of classifying a CF-netCDF variable by its attributes.
`CFReader._variable_types` lists all ten in one tuple (cf.py:1291-1302);
`CFGroup.non_data_variable_names` enumerates `connectivities`, `ugrid_coords`
and `meshes` alongside `bounds`, `labels` and `cell_measures`; and the
module-private `_is_str_dtype` (cf.py:78) is called from both sets, so the
split would export a private helper across a module boundary to buy nothing.
The `iris.mesh` analogy does not hold either: that package holds cube-level
data-model objects, whereas these are variable classifiers doing the work of
`CFAuxiliaryCoordinateVariable`. `lib/iris/AGENTS.md` settles it — *"Cohesion
wins. Do not fragment into micro-modules purely to shrink files."*

The re-export layer in `__init__.py` is the ordinary Iris pattern, not a
concession. `iris.mesh` re-exports from four modules, two of them in a
different subtree entirely (`iris.fileformats.netcdf.saver`,
`iris.fileformats.netcdf.ugrid_load`), so that users import `MeshXY` and
`save_mesh` from one place. Surfacing objects where users expect them is a
genuine convenience, and it is what leaves the private file layout free to
change. `lib/iris/AGENTS.md` says so explicitly; its "do not add indirection"
rule governs implementation layers, not API surface.

The re-exports are verbatim and static — `from ._reader import CFReader`, name
for name, gathered into `__all__` — so `class CFReader` still finds the
definition in one grep. No renaming, no conditional imports, no logic in
`__init__.py`.

`_variables.py`, `_group.py` and `_reader.py` are private because
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
    attributes: MutableMapping[str, Any]   # materialised once; tracks reads

    def __getitem__(self, keys) -> np.ndarray: ...
    def __setitem__(self, keys, values) -> None: ...

    def write_handle(self) -> Any: ...  # picklable __setitem__ target; see 4.5


class CFDataset(ABC):
    """A CF-conforming array store, open for reading or writing."""
    location: str                       # path or URL, for messages and proxies
    mode: str                           # "r" | "r+" | "a" | "w" | "w-"; 4.5
    variables: Mapping[str, CFDatasetVariable]
    dimensions: Mapping[str, int]
    attributes: MutableMapping[str, Any]

    def create_dimension(self, name: str, size: int) -> None: ...
    def create_variable(self, name, dtype, dimensions, *,
                        fill_value=None, **encoding) -> CFDatasetVariable: ...
    def sync(self) -> None: ...
    def finalise(self) -> None: ...     # one-shot; NOT part of close()
    def close(self) -> None: ...
```

Three members exist only to keep multi-process writing reachable later, and
are explained in §4.5: `write_handle`, `mode` and `finalise`. They cost
almost nothing now — `mode` is a string the implementations already track
internally, `write_handle` returns `self` for Zarr, and `finalise` is a no-op
for netCDF — and their absence is what would force a breaking interface
change later.

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

`CFVariable.__getattr__` (cf.py:198) does two unrelated jobs today, and the
refactor separates them. Counting what reaches it from inside Iris:

| Reached via `__getattr__` | Count | Job |
|---|---|---|
| `.dimensions` | 28 | proxy the backing netCDF4 variable |
| `.shape` | 11 | proxy |
| `.dtype` | 11 | proxy |
| `.getncattr` | 6 | proxy |
| `.size`, `.group`, `.chunking` | 7 | proxy |
| CF attribute names (`units`, `bounds`, `coordinates`, ...) | the rest | open-world data |

`self.getncattr(attr)` at cf.py:223 is the clearest case: `CFVariable` does not
define `getncattr`, so the call resolves through `__getattr__` onto the netCDF4
object. Everything in the "proxy" rows is exactly what `CFDatasetVariable`
(§4.2) declares explicitly and typed, so the refactor does not have to defend
`__getattr__` - it shrinks it to the one job that justifies it.

**The surviving job is legitimate.** CF attributes are an open-ended set of
data keys read from a file; there is no schema to enumerate, so this is the
narrow exception `lib/iris/AGENTS.md` now carves out. The condition attached to
that exception is that `.attributes` is the path library code takes, and
`__getattr__` is only a convenience skin over it.

So, on `CFVariable`:

- `__getattr__` resolves against `self.attributes` - one implementation on the
  base class, **identical for netCDF and Zarr**. An earlier draft had it raise
  `TypeError` for Zarr-backed variables; that put a hole in the most-used
  public accessor of the very abstraction §4.2 exists to provide, and was
  wrong. A missing key raises plain `AttributeError`, which is correct:
  `getattr(nc_var, "bounds", None)` is a deliberate idiom throughout the loader
  and depends on `hasattr` semantics working.
- **No `setattr(self, name, value)` caching.** Today's version caches by
  mutating `self.__dict__`, so an instance's shape depends on its access
  history - that genuinely is the "dynamically generated attributes"
  anti-pattern. A read-through to a mapping materialised once at construction
  is faster than today's *first* access and stays inspectable.
- On a miss, a netCDF-backed variable falls back to the backing netCDF4 object
  and issues a deprecation warning naming the `CFDatasetVariable` member that
  replaces it, so `cf_var.getncattr("units")` and `cf_var.ndim` keep working
  for one cycle. A Zarr-backed variable has no such object, so the fallback
  simply does not apply - consistent, not a special case.

**Known limitation, documented rather than fixed.** `__getattr__` fires only
when normal lookup fails, so a file carrying an attribute named `filename`,
`cf_name` or `spans` is silently shadowed by the class member of that name. CF
reserves no namespace, so this cannot be ruled out. `.attributes` has no such
ambiguity, which is the practical argument for making the mapping the library
path.

**Consequence for attribute tracking, the sharpest edge in PR 2.** The
`_cf_attrs` "which attributes were read" set is currently a side effect of
`__getattr__`, and `cf_attrs_unused()` has exactly one consumer:
`netcdf/loader.py:190`, which decides which attributes survive onto the cube.
Move the reads to `.attributes` and the tracking must move with them, or every
CF-reserved attribute silently leaks onto loaded cubes. `.attributes` is
therefore a tracking mapping, not a plain `dict`, and PR 1's tests must pin
`cf_attrs_unused()` before PR 2 touches it.

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
NCZarr path. It is fixed in PR 2, and is the only `bugfix` fragment the
merge-back owes: everything else in the programme is new capability, a
relocation or a deprecation.

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
answer; the durable one needs the attribute-model discussion in
SciTools/iris#7288 (§10).

**Masking and unpacking.** netCDF4 applies `_FillValue`/`missing_value`
masking, `valid_min`/`valid_max`/`valid_range` masking and
`scale_factor`/`add_offset` unpacking for free. Zarr does none of it, so
`zarr/_decode.py` implements the same CF rules. It is Zarr-side rather than
shared because the netCDF path gets the behaviour from the C library and has
nothing to reuse.

Decoding is applied as **explicit dask graph layers** over the raw array —
`da.map_blocks(_mask_and_scale, raw, ...)` — not hidden inside an opaque
`__getitem__`. Each rule is a small pure function of `(block, parameters)`,
unit-testable on a plain ndarray with no store at all; the layers are visible
in the graph, fuse with neighbours, and keep the lazy contract. A proxy whose
`__getitem__` silently returned something other than what the store holds is
exactly the "behaviour not derivable from the file in front of you" that
`lib/iris/AGENTS.md` warns against.

**Laziness: there is no `ZarrDataProxy`.** An earlier draft of this spec said
the Zarr reader would mirror `NetCDFDataProxy`. That was wrong, and the reason
is worth recording, because every reason `NetCDFDataProxy` exists is a
netCDF4/HDF5 reason:

- `netCDF4.Variable` is not picklable, so it cannot travel in a dask graph.
- HDF5 is not thread-safe, so every read serialises on `_GLOBAL_NETCDF4_LOCK`.
- A `Dataset` is an open OS file handle that must not be held for the lifetime
  of a lazy graph — hence the open/read/close on every `__getitem__`
  (`_thread_safe_nc.py:353-366`).

None of the three holds for Zarr. `zarr.Array` **is** picklable and round-trips
shape and chunks **[verified]**. A chunk is an independent object in a
key-value store, so concurrent reads of distinct chunks need no lock. A store
is a namespace, not an open handle, so there is nothing to leave open. And this
is the mainstream path rather than a novelty: `dask.array.from_zarr` is a thin
wrapper that calls `from_array` on the `zarr.Array` itself **[verified]**.

So `zarr/loader.py` hands the `zarr.Array` to `as_lazy_data` directly and adds
the decode layers on top. The three things the earlier draft claimed a proxy
preserved survive without one:

1. **`CHUNK_CONTROL`.** Chunking is computed by the loader from
   `CFDatasetVariable.chunking` *before* `as_lazy_data` is called. It never
   depended on proxy-ness.
2. **Masked dask meta.** `meta` is an argument to `as_lazy_data`, not something
   a proxy supplies. With decoding as a graph layer the mask arrives with the
   decode layer, so a masked meta is correct by construction. Note the honest
   ordering — passing a masked meta with no decode layer beneath it would be a
   meta that lies about its own graph.
3. **The dask cache key.** This is the one real point, and it wants a hashable
   identity, not a class. See below.

**Replacing the `NetCDFDataProxy` cache special case.** `iris/_lazy_data.py`
imports `NetCDFDataProxy` from `iris.fileformats.netcdf` (line 302) purely to
recognise it (lines 333-337) and build a cache key from `repr(data)`, so that
many cubes sharing a coordinate array share one dask array. That is a backwards
import — the lazy-data layer reaching into a file format — and adding a second
`isinstance` branch for Zarr would double it.

PR 3 replaces it with a caller-supplied `cache_key=` keyword on
`as_lazy_data`. The netCDF loader passes `repr(proxy)`, which is
byte-for-byte the key computed today, so the change is behaviour-preserving and
testable as such. The Zarr loader passes `(store location, array path, zarr
format)`. `_lazy_data` then knows about no file format at all.

**Read-side chunk alignment.** §4.5 fixes a write-side invariant: the dask
chunking must tile the Zarr chunk grid exactly. Reading needs the same
invariant for a different reason, and it is *not* free.

A Zarr chunk is the atomic unit of storage: a read of any part of it fetches
and decompresses the whole object. So if the dask chunking subdivides a store
chunk into N pieces, N tasks each pull and decompress that entire chunk — N
times the bytes and N times the CPU, and over a remote store, N HTTP GETs for
the same object.

`_optimum_chunksize` does the right thing in the ordinary case, expanding to
whole multiples of the store chunk **[verified]**:

| store chunk | array shape | dask chunk | multiple |
|---|---|---|---|
| `(512, 512)` | `(4096, 4096)` | `(4096, 4096)` | 8 × 8 |
| `(1000, 1000)` | `(10000, 10000)` | `(1000, 10000)` | 1 × 10 |

But when the store chunk is *itself* larger than the `dask.array.chunk-size`
target (128 MiB by default) it shrinks below it **[verified]**:

| store chunk | size | array shape | dask chunk | result |
|---|---|---|---|---|
| `(8000, 8000)` | 488 MiB | `(16000, 16000)` | `(2000, 8000)` | 4 × re-read |
| `(6000, 6000)` | 275 MiB | `(6000, 6000)` | `(2000, 6000)` | 3 × re-read |

Subdividing buys nothing here. Peak memory per task is still a whole
decompressed chunk, because that is what the store hands back; only the
I/O and decompression multiply. So on the Zarr path the loader rounds the
computed chunking **up** to a whole multiple of the store chunk, never below
it, and emits an `IrisLoadWarning` naming the variable when the store's own
chunking exceeds the dask target — the store's layout is then the binding
constraint and the user should know. An explicit `CHUNK_CONTROL` setting still
wins, since that is a deliberate instruction, but it warns on the same terms.

This trap is not new and not Zarr-specific: `NetCDFDataProxy` reopens the
`Dataset` on every `__getitem__`, so HDF5's chunk cache is discarded between
tasks and the netCDF path re-reads sub-chunks too. It simply costs far more
over a network than over a page cache. Changing the netCDF path is out of scope
here — the relocation PRs must not alter behaviour — but it is recorded as an
open question in §12.3.

`CFDatasetVariable.chunking` returns the store's own chunk shape, which
`cf/loader.py` uses exactly as it uses `chunking()` today. For a sharded
version 3 array it returns `Array.shards`, which is `None` on an unsharded
array **[verified]**, falling back to `Array.chunks`. The shard is the unit a
reader actually fetches, so it is the right unit for both the chunk
calculation and the alignment rule above.

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
paragraph in §4.4 runs into from the read side. §10 and SciTools/iris#7288
record it as parked for a dedicated discussion; nothing else in the design
depends on how it is settled.

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
returning a `Delayed`, and reopening per chunk under a whole-file lock. Zarr
needs no reopen: the store stays valid, and writes to *distinct chunks* are
independent objects. The Zarr saver builds one
`da.store(..., compute=False, lock=False)` over all lazy sources and returns it
from `save(..., compute=False)`, matching the netCDF signature. This is also
why the NCZarr special case in `Saver.__exit__` (saver.py:485-498) is
untouched: it is a different, older path.

`lock=False` is licensed by an invariant, not by Zarr being inherently safe.
Two dask tasks that touch the *same* Zarr chunk race: each reads the chunk,
modifies its slice and writes the whole chunk back, so one update is silently
lost. Therefore:

> **Chunk-alignment invariant.** Every Zarr array Iris writes has a chunk grid
> that the lazy source's dask chunking tiles exactly.

`create_variable` derives `chunks` from the source's dask chunking by default,
so the invariant holds by construction. Where the caller forces `chunksizes=`
that the source does not tile, the saver rechunks the dask source to match
before storing. It never issues an unaligned concurrent write. This is the same
guarantee xarray spells `safe_chunks`, and it is a correctness requirement
today, not only a precondition for the future work below.

**Consolidation is a separate, one-shot step.** `CFDataset.finalise()` writes
consolidated metadata; `close()` only releases resources. They are split
because a worker that writes one slab of a store must not consolidate — N
workers consolidating the same single metadata object is a race. Under
`compute=False` the returned `Delayed` owns the `finalise()` call, after every
chunk write. For netCDF, `finalise()` is a no-op.

#### The write proxy: why netCDF needs one, and Zarr does not

`NetCDFWriteProxy` is easy to read as multi-process machinery, because it
carries the file lock. That is its *second* job. Its first is more basic: to be
a `__setitem__` target that survives the file being closed.

The netCDF deferred save closes the dataset before the `Delayed` is computed —
`Saver.delayed_completion` says so explicitly, and computing with the file
still open hangs. A `netCDF4.Variable` cannot outlive its `Dataset`, and is not
picklable, so it cannot be the target handed to `da.store`. The proxy replaces
it with `(path, varname, lock)` and reopens per chunk, retrying up to five
times because HDF5 sometimes refuses a file Python believes it has released
(`_thread_safe_nc.py:412-433`). So the proxy is needed for a **single-threaded**
deferred save too, not only a distributed one. The lock is what makes it also
work across workers.

Neither job exists for Zarr. There is no handle to close, so nothing has to
outlive one; `zarr.Array` is picklable and keeps writing after a pickle
round trip **[verified]**; and distinct chunks are distinct objects, so the
alignment invariant above removes the need for a lock rather than deferring it.
The end-to-end property was checked directly: a `da.store(..., compute=False,
lock=False)` over a `zarr.Array` still lands correctly after every in-process
reference to the group is dropped, **and the whole delayed graph survives
`pickle.dumps`/`loads` before computing** **[verified]** — which is the
property a `distributed` scheduler actually requires.

**Native Zarr therefore restores a feature NCZarr had to give up.** The
existing saver cannot defer NCZarr writes at all: `Saver.__exit__`
(saver.py:485-498) computes them eagerly with `da.store` while the file is
still open, commenting that "the deferred reopen-write pattern used for netCDF
is not supported". That is a limitation of reaching Zarr through netCDF-c, not
of Zarr. Going direct, `iris.save(cubes, "out.zarr", compute=False)` returns a
deferred save like the netCDF path — this is a *gain* over the status quo for
the distributed-write use case, and PR 6 must include a test asserting it.

So `write_handle()` earns its place, but not as future-proofing: it is needed
today, by netCDF, for exactly the reason above. `NetCDFDatasetVariable` returns
the write proxy; `ZarrDatasetVariable` returns its `zarr.Array`. `cf/saver.py`
asks for a handle and stores into it, knowing nothing about locks, reopening or
schedulers.

One correction to the accounting in §4.5 below: library code never instantiates
`_thread_safe_nc.NetCDFWriteProxy` directly. `_lazy_stream_data` builds
`_bytecoding_datasets.EncodedNetCDFWriteProxy`, a subclass, and says why
(saver.py:2638-2643). The base class survives as public API and as the
superclass. The relocation must preserve both.

**A live defect in the machinery this inherits.** With `dask 2026.7.1`,
`da.store(sources, targets, compute=False)` returns a **tuple** when `sources`
is a sequence, not a `Delayed` **[verified]**. `Saver.delayed_completion` is
annotated `-> Delayed` and returns that value unchanged, so
`iris.save(cubes, path, compute=False)` returns a tuple and the documented
idiom fails:

```
>>> result = iris.save(cubes, path, compute=False)
>>> result.compute()
AttributeError: 'tuple' object has no attribute 'compute'
```

`dask.compute(result)` works, which is why nothing caught it. The cause is
known: #6451 adapted Iris to `dask/dask#11844` for dask 2025.4, correctly
switching every internal caller to `dask.compute()`, and relaxed the test
assertion to a helper accepting either shape. The public return type changed
with it, but `Saver.delayed_completion`, `save()` and
`docs/src/user_manual/explanation/real_and_lazy_data.rst` still promise a
`Delayed` completed by `result.compute()`. Raised as
[#7291](https://github.com/SciTools/iris/issues/7291).

This is a netCDF-side defect, not a Zarr one, and it is **not** fixed by this
programme — PR 5 relocates the saver without changing behaviour. But it is
recorded here because the Zarr saver inherits the same shape, so `zarr/saver.py`
wraps its `da.store` result in `dask.delayed` and returns a real `Delayed`, and
its tests assert `.compute()` works. Tracked as Q5 in §12.3 and as #7291.

#### Durability: what an interrupted save leaves behind

Zarr has no multi-object transaction. A store's metadata and its chunks are
separate objects, so an edit that touches both can be interrupted between them
and leave the two disagreeing. Two consequences, both confirmed against
`zarr 3.4.0` **[verified]**:

- A missing chunk is **not an error**. Remove one chunk object from a complete
  array and it reads back as `fill_value` with no warning. An interrupted save
  therefore leaves a store that opens cleanly and silently under-reports data.
- Consolidated metadata goes **silently stale**. Add an array to a consolidated
  store without re-consolidating, and a default reader — which uses
  consolidated metadata when it is present — does not see the new array at all.
  An unconsolidated reader of the same store does.

Neither is fixable from inside Iris; they are properties of the format. What
Iris can do is never create the conditions for them.

> **Create-once invariant.** Array metadata is written when the array is
> created and never edited afterwards. Iris does not resize, rechunk, or change
> the dtype, `fill_value`, codecs or dimension names of an array that already
> exists.

That is what makes `iris.save(cubes, "new.zarr")` desynchronisation-free by
construction rather than by luck: the chain is create-arrays, write-chunks,
consolidate, with metadata written once at the front. It is stated here as a
constraint on future work, not merely as a description of today's.

**Writing into a store that already exists** is the case the invariant does not
cover, and `save(..., group=...)` reaches it within the current scope. Rules:

- `save` takes `mode`, using zarr's vocabulary. The default is **`"w-"`:
  create, and raise if the target already exists.** This deliberately differs
  from the netCDF saver, which clobbers, because a Zarr clobber is many
  non-atomic deletes: an interrupt destroys the old store without completing
  the new one. `mode="w"` opts back in to replacement; `mode="a"` adds to an
  existing store and is what `group=` needs. The error message names `mode`.
- When Iris adds to a store that already carries consolidated metadata, it
  **must** re-consolidate as its final act. If it cannot — the root is not
  writable, say — it raises rather than returning, because leaving stale
  consolidated metadata hides the data just written.
- `finalise()` is always the last write. Under `compute=False` the returned
  `Delayed` owns it, so consolidation follows the last chunk rather than
  preceding it.

**What Iris does not promise.** An interrupted save leaves a partial store, and
Iris offers no torn-write detection or rollback. The usual mitigation — write
to a temporary name and rename on success — is not adopted: directory rename is
atomic on POSIX but object stores have no equivalent, so the guarantee would
evaporate precisely where Zarr is most used, which is worse than not offering
it. Callers needing transactional writes want Icechunk, which is out of scope
(§9). The Zarr documentation section says so plainly rather than leaving users
to discover it.

**Tests.** Deleting a chunk object from a saved store and asserting the read is
silently filled; adding an array to a consolidated store and asserting the
default reader misses it until re-consolidation; and `mode="w-"` raising on an
existing target. These pin format behaviour Iris depends on, so they should
fail loudly if a zarr-python release changes it.

#### Multi-process writes: kept reachable, not built

Coordinated writing to one target from many processes is an Iris
differentiator for netCDF and is explicitly wanted for Zarr. It is out of
scope here (§9) purely for size. These are the provisions that keep it a
later addition rather than a redesign.

**zarr-python will not do it for us.** Version 3.4.0 dropped the version 2
synchroniser machinery: `zarr.create_array` has no `synchronizer` parameter at
all, and `zarr.open_group(synchronizer=...)` is accepted but warns
`"synchronizer is not yet implemented"` **[verified]**. `zarr.ProcessSynchronizer`
and `zarr.sync` no longer exist. Any coordination Iris offers must therefore be
Iris's own, exactly as `_dask_locks.py` is for netCDF today. §9's bullet is
worded accordingly.

1. **`_dask_locks.py` moves to `cf/`, in PR 5.** Despite its netCDF-flavoured
   docstring it imports only `threading` and four `dask` modules — **no netCDF
   whatsoever** **[verified]**. It is scheduler-aware, file-identity-based
   locking that is generic already. Relocating it costs one `git mv` and an
   updated docstring, and it is the difference between a future Zarr
   implementation reaching for an existing toolkit and growing a parallel one.

2. **`CFDatasetVariable.write_handle()` is the coordination seam.** It returns
   a picklable object supporting `__setitem__`, which is what a dask worker
   receives. netCDF returns today's write proxy — in practice
   `EncodedNetCDFWriteProxy` — which carries the `distributed.Lock` keyed on
   the file path. Zarr returns the `zarr.Array`
   itself — verified picklable, round-tripping shape and chunks under
   `zarr 3.4.0` **[verified]** — because no lock is needed while the alignment
   invariant holds. If Zarr later needs coordination, the Zarr implementation
   returns a lock-carrying proxy instead: **no change to the interface and no
   change to `cf/saver.py`**. This is the precise sense in which the design
   does not preclude the feature.

3. **The lock lives in the dataset implementation, never in `cf/saver.py`.**
   The generic saver asks for write handles and stores into them; it holds no
   lock and knows no scheduler. That is what lets the two backends coordinate
   differently — and lets Zarr coordinate *better*, since a whole-store lock
   would throw away the parallel throughput that is the reason to write Zarr.

4. **`mode` is on `CFDataset` from the start.** The Zarr shape of this feature
   is a region write: open an existing store `"r+"` and have each process fill
   its own slab, rather than serialise every worker behind one lock. That needs
   a dataset that can be opened for update without creating anything, which an
   interface whose only verbs are `create_dimension` and `create_variable`
   cannot express. Adding `mode` afterwards would be a breaking change to a
   published ABC with two implementations and deprecation aliases pointing at
   it; adding it now is one attribute.

Nothing here builds the feature. Together they mean building it later is new
code in `zarr/_dataset.py` plus a keyword on `save`, with the generic layer
and the netCDF path untouched.

The durability hazards above get sharper under parallelism, and the
create-once invariant is most of the answer there too: workers that only fill
chunks of arrays created up front never touch metadata, so the one shared
mutable object is the consolidated metadata, written once at the end by the
process that owns `finalise()`. A design that let workers create or resize
arrays would have no such property, which is a further reason the invariant is
written down rather than assumed.

**Encoding.** `zlib`/`complevel` map to a Blosc or Gzip codec, `shuffle` to
Blosc shuffle, `chunksizes` to `chunks`, `fletcher32` to Crc32c. `contiguous`,
`endian` and `least_significant_digit` have no Zarr equivalent and raise
`TypeError` if passed. Zarr adds `shards=` and `compressors=`, and
`consolidate_metadata=True` (default `True`, since consolidation is what makes
remote stores usable, with the zarr-python warning suppressed and explained).

### 4.6 Groups

`load_cubes(..., group=None)` and `save(..., group=None)` both default to the
root group. A non-`None` `group` is a `/`-separated path within the store.

Writing to a non-root group means opening a store that may already exist, so
`save(..., group=...)` is governed by the existing-store rules in §4.5 — it
implies `mode="a"`, and re-consolidation is mandatory.

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

Seven pull requests into `brownfield`, each opened separately so that every
step gets its own CI signal and can be verified on its own, followed by a
single merge-back into `main`. Each carries the `Agentic` and
`Type: Feature Branch` labels and attributes the contribution to Claude. This
is the delivery plan, not a rehearsal for one: the pull requests below are the
work, and the merge-back is how it reaches users.

**Issues resolve at merge-back, not before.** A GitHub closing keyword only
fires when the pull request merges into the repository's *default* branch, so
`Closes #6977` on a pull request based on `brownfield` would link the issue and
never close it. The feature-branch pull requests therefore write
`Part of #6977`, and the merge-back pull request carries the closing keywords
for all three sub-issues. This is bookkeeping, not caution: #6977, #6979 and
#6980 are genuinely resolved by this programme, just not one branch merge
earlier than they should be. Issue *state* is not tracked in this document at
all — that lives on GitHub (§12.2).

**Changelog fragments land with the merge-back.** Fragments are named
`<PR-number>.<type>.rst` (`changelog/AGENTS.md`), and a fragment numbered for a
feature-branch pull request would advertise a change that `main` has not yet
received. Established practice on this branch agrees: the two pull requests
already merged into `brownfield`, #7285 and #7287, carry no fragment, while
#7267 into `main` does **[verified]**. Each feature-branch pull request body
therefore states that its fragment is deferred, so the omission does not read
as an oversight, and §5's merge-back subsection lists the fragments owed.

### PR 1 — `iris.fileformats.cf` becomes a package, with tests first

`git mv cf.py cf/_variables.py` and split out `_group.py` and `_reader.py`
along the boundaries in §4.1, with an `__init__.py` that
re-exports the existing public names. Imports updated across the tree. **No
behaviour change and no API change**: the diff is a move plus import edits, so
a reviewer can skim it.

Brings in the unit test coverage from SciTools/iris#7259 by Martin Yeo, rebased
onto `brownfield` and credited in the pull request body.
That branch is named `cf_reader_zarr` and was written for exactly this
refactor; it adds roughly 2300 lines of coverage across nineteen test modules.
Landing it *here*, against today's behaviour, is what makes PR 2 reviewable:
the tests become a regression net that was written before the rewrite and does
not move with it. A comment on #7259 explains the overlap before the pull
request is opened.

### PR 2 — `CFDataset`, and the CF variable classes rewritten against it

The subject of **#6977**. Adds `cf/dataset.py` and `netcdf/_dataset.py`. Rewrites the
86 `getattr`/`hasattr` sites and the 57 netCDF-API sites in the `cf` package,
`_nc_load_rules/` and `netcdf/` to go through the new interface. Rewires
`CFVariable.__getattr__` onto `.attributes`, with the deprecating
fallback and the attribute-read tracking of §4.3. Fixes the
`_NCZARR_SCALAR_DIMENSION` gap in the three overriding `spans` methods.

No behaviour change other than the `spans` fix. The suite from PR 1 must pass
untouched.

### PR 3 — Relocate the CF loader

`git mv netcdf/loader.py cf/loader.py`, leaving a `netcdf/loader.py` that holds
the netCDF specifics and the deprecating aliases. Still netCDF-only behaviour;
the whole existing suite passes unchanged. Large, mechanical, independently
verifiable.

Also adds the `cache_key=` keyword to `as_lazy_data` and deletes the
`isinstance(data, NetCDFDataProxy)` special case in `iris/_lazy_data.py`
(§4.4), removing that module's import of `iris.fileformats.netcdf`. The
netCDF loader passes the same `repr(proxy)` key it produces today, so this
too is behaviour-preserving — and a test asserts the key is unchanged.

### PR 4 — Zarr loading

The subject of **#6979**. Adds `zarr` to the optional dependencies section of
`requirements/py3{12,13,14}.yml` and regenerates the lock files, once, here.
Adds `fileformats/zarr/` with `_dataset.py`, `_decode.py` and `loader.py`, the
format specification, and the five load-chain changes in §4.7. Documents Zarr
loading in the user guide and the `iris.fileformats` API reference, including
the remote-store URL forms and the `group=` keyword.

After this pull request, `iris.load("store.zarr")` and
`iris.load("s3://bucket/store.zarr")` work for version 2 and version 3 stores.

### PR 5 — Relocate the CF saver

`git mv netcdf/saver.py cf/saver.py`, same shape as PR 3. The largest diff of
the seven and the one with no behaviour change at all, which is precisely why
it is on its own.

Also `git mv`s `netcdf/_dask_locks.py` to `cf/_dask_locks.py` and rewrites its
netCDF-specific docstring. The code imports only `threading` and four `dask`
modules **[verified]**, so this is a relocation with no behaviour change — it
puts the scheduler-aware locking toolkit where a future Zarr implementation
can reach it (§4.5).

### PR 6 — Zarr saving

The subject of **#6980**. Adds `zarr/saver.py`, the `ZarrDataset` write path, JSON
attribute conversion, masked-data filling, deferred writes, encoding
translation and `consolidate_metadata`. Registers the `zarr` saver, and
documents saving alongside the loading documentation from PR 4, including the
S3 pages.

Includes a test that `iris.save(cubes, "out.zarr", compute=False)` returns a
`Delayed` whose `.compute()` completes the store — the deferred save that the
NCZarr path cannot offer (§4.5), and the contract the netCDF path is currently
breaking (§12.3 Q5).

After this pull request, `iris.save(cubes, "out.zarr")` works.

### PR 7 — Real-world test data and benchmarks

Integration tests against the cut-down NOAA GFS and ESA EOPF samples, plus ASV
coverage: `ZarrSave` beside `NetcdfSave` in `benchmarks/benchmarks/save.py`,
and Zarr variants of `LoadAndRealise` in `benchmarks/benchmarks/load/`.
Depends on a companion pull request to `SciTools/iris-test-data`.

### Merge back — `brownfield` into `main`

Labelled `Type: Merge Back`. This is where the programme becomes a release, and
where three obligations deferred by the feature-branch pull requests are met.

**Closing keywords** for #6977, #6979 and #6980.

**Changelog fragments**, numbered for the merge-back pull request. Per
`changelog/AGENTS.md`, one pull request may own several fragments of different
types, which is what this needs:

| Type | Covers |
|---|---|
| `feature` | Native Zarr version 2 and version 3 reading, and version 3 writing |
| `dependency` | `zarr` added as an optional dependency; xarray as a test dependency |
| `deprecation` | The relocated public names in §4.8, which warn on use |
| `bugfix` | The `_NCZARR_SCALAR_DIMENSION` gap in the three overriding `spans` methods |
| `internal` | The `cf` package split and the CF loader/saver relocation |

Every fragment credits `` :user:`claude` `` per `changelog/AGENTS.md`.

**Documentation** is written in PRs 4 and 6, alongside the code it describes,
and becomes visible to users only here. The merge-back checks it builds clean
against `main` with `-W`, since the two branches can drift.

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

**Chunking and decoding** are unit-testable without a store, which is the main
practical dividend of dropping the proxy (§4.4). Each CF decode rule is a pure
function of `(block, parameters)` and is tested on plain ndarrays. The
alignment guard is tested on the `chunking` values alone:

- a store chunk smaller than the dask target expands to a whole multiple;
- a store chunk larger than the dask target is **not** subdivided, and warns;
- an explicit `CHUNK_CONTROL` setting that subdivides is honoured, and warns;
- the resulting dask chunks tile the store chunk grid exactly, in every case.

**The `cache_key=` change** (§4.4, PR 3) is pinned by asserting that the netCDF
loader produces the identical key to today's `repr(proxy)`, so the removal of
the `isinstance` branch in `iris/_lazy_data.py` cannot silently lose the
array-sharing it exists to provide.

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
therefore already been cut, ahead of the rest of the programme; §12.5 holds its
current location. It is 1.4 MB, 35 files, 12
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
- Icechunk, virtual Zarr and Kerchunk reference stores. Icechunk is the
  ecosystem's answer to transactional Zarr writes; §4.5 says so where the
  limitation bites.
- **Building** multi-process write coordination for Zarr — see below.
- Any change to the existing NCZarr path other than the `spans` bug fix.
- Adding an fsspec, s3fs or gcsfs dependency to Iris.

Multi-process write coordination is deferred for size only. It is an Iris
differentiator for netCDF, it is asked for by name, and zarr-python offers
nothing to build on — version 3.4.0 warns `"synchronizer is not yet
implemented"` **[verified]**, so it will be Iris's own machinery when it comes.
§4.5 lists the four provisions that keep it a later addition rather than a
redesign, and the chunk-alignment invariant it depends on is a correctness
requirement of the first release regardless. Anything in review that would
break those provisions should be treated as in scope, not out of it.

---

## 10. The parked question: the CF attribute model on JSON

Everything else raised at review is settled and recorded in the decision log
at §12.4. This one is not, and it is left open deliberately.

It is one question with two faces, and the design currently answers both
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

**SciTools/iris#7288** tracks the discussion, with the relevant specification
citations (Zarr v3 "arbitrary JSON literal"; CF §2.5.1 "of the same type as its
variable") and the two real-world examples from §4.4.

---

## 11. References

- SciTools/iris#6961 — Zarr I/O, parent issue
- SciTools/iris#6977, #6979, #6980 — the three sub-issues
- SciTools/iris#7259 — `cf.py` unit test coverage, by Martin Yeo
- SciTools/iris#7288 — the parked CF attribute model question (§10)
- Zarr specifications: https://zarr-specs.readthedocs.io/
- zarr-python: https://zarr.readthedocs.io/
- CF conventions for Zarr: https://github.com/zarr-conventions/CF
- NCZarr: https://docs.unidata.ucar.edu/nug/current/ncZarr_head.html
- NOAA GFS archive: https://data.dynamical.org/noaa/gfs/forecast/

---

## 12. Progress

The live record for the programme. Everything with a state belongs here;
the rest of this document describes the design and should change only when
the design changes.

### 12.1 Pull requests

Scope for each is in §5. Dependencies are strictly sequential within the
`cf`/`netcdf` refactor chain (1 → 2 → 3 → 5); the Zarr features (4, 6) depend
on the relocation before them, and PR 7 depends on everything.

| # | Title | State | Link |
|---|---|---|---|
| 1 | `iris.fileformats.cf` becomes a package, with tests first | Not started | — |
| 2 | `CFDataset`, and the CF variable classes rewritten against it | Not started | — |
| 3 | Relocate the CF loader | Not started | — |
| 4 | Zarr loading | Not started | — |
| 5 | Relocate the CF saver | Not started | — |
| 6 | Zarr saving | Not started | — |
| 7 | Real-world test data and benchmarks | Not started | — |
| — | **Merge back** `brownfield` → `main`: closing keywords, changelog fragments, docs | Not started | — |

Companion work outside `SciTools/iris`:

| Repository | Purpose | State | Link |
|---|---|---|---|
| `SciTools/iris-test-data` | Cut-down NOAA GFS (v3) and ESA EOPF (v2) fixtures for PR 7 | Not started | — |

States are `Not started`, `Drafted`, `In review`, `Changes requested`,
`Merged` or `Abandoned`. Record the pull request number and its state as soon
as it is opened, not when it merges.

There is deliberately no `Closes` column and no changelog-fragment column:
both obligations belong to the merge-back, not to the feature-branch pull
requests. See §5.

### 12.2 Related work

Issue state is tracked on GitHub, not here. This table says what each item is
*for* in this programme, and nothing about whether it is open. The three
sub-issues are resolved by the merge-back pull request, which is where the
closing keywords live (§5).

| Item | What it is to this programme |
|---|---|
| [#6961](https://github.com/SciTools/iris/issues/6961) | The requirement. Parent issue for Zarr I/O |
| [#6977](https://github.com/SciTools/iris/issues/6977) | Implemented by PR 2; closed at merge-back |
| [#6979](https://github.com/SciTools/iris/issues/6979) | Implemented by PR 4; closed at merge-back |
| [#6980](https://github.com/SciTools/iris/issues/6980) | Implemented by PR 6; closed at merge-back |
| [#7288](https://github.com/SciTools/iris/issues/7288) | The parked attribute-model question this design raised (§10) |
| [#7291](https://github.com/SciTools/iris/issues/7291) | The deferred-save return type this design turned up (§4.5, Q5). Not fixed by this programme |
| [#7259](https://github.com/SciTools/iris/pull/7259) | @trexfeathers' unit test coverage for `cf.py` (`cf_reader_zarr`, +2361/-643), carried into PR 1 with credit. Post a courtesy comment before opening PR 1 |

### 12.3 Open questions

| # | Question | Provisional answer | Tracked | Blocks |
|---|---|---|---|---|
| Q1 | How should the CF attribute model map onto JSON, on write and on read? | JSON-native both ways (§4.4, §4.5) | #7288, §10 | Nothing. Changing it touches only the attribute conversion in `zarr/_dataset.py` |
| Q2 | Are `<U7`-style fixed-length unicode dtypes, which #6961 notes are not strictly Zarr-supported and which the EOPF store contains, readable as-is? | Unverified — read them before PR 4 | §7 | PR 4 |
| Q3 | Does the `dimension_names`-absent error in §4.4 need an escape hatch for stores that are otherwise loadable? | No; synthesising names produces silently wrong cubes | §4.4 | Nothing |
| Q4 | The netCDF read path has the same sub-chunk re-read trap as Zarr, because `NetCDFDataProxy` reopens the `Dataset` on every `__getitem__` and so discards HDF5's chunk cache. Should it get the same alignment guard? | Out of scope here — the relocation pull requests must not change behaviour. Raise it separately once §4.4's Zarr guard has proven itself | §4.4 | Nothing |
| Q5 | `iris.save(..., compute=False)` returns a tuple, not the documented `Delayed`, so `result.compute()` raises `AttributeError` **[verified]** | Caused by #6451 adapting to dask/dask#11844; the code is right and the docstrings were left behind. Not fixed here — PR 5 is behaviour-preserving. `zarr/saver.py` returns a real `Delayed` and tests it | [#7291](https://github.com/SciTools/iris/issues/7291) | Nothing |

Close a question by moving it to §12.4 with the date and the answer. Do not
delete it.

### 12.4 Decision log

Append-only. Each entry is the decision, not the discussion.

**2026-09-21 — before design, agreed with @trexfeathers**

- `zarr` is an optional, lazily imported dependency.
- Extract the format-agnostic CF machinery out of `iris.fileformats.netcdf`,
  with warn-on-use deprecation aliases at the old public names.
- Write Zarr version 3 only, shaped so version 2 write is a later small change.
- Read both version 2 and version 3.
- Pass remote URLs straight to zarr-python; add no fsspec/s3fs dependency and
  handle no credentials.
- Load and save the root group by default, with a `group=` keyword.
- Source real-world test data from NOAA GFS (dynamical.org) and the ESA EOPF
  Sentinel samples.
- Carry the test modules from #7259 into the programme, with credit.

**2026-09-21 — at review of the first draft, agreed with @trexfeathers**

- **Module layout:** `iris.fileformats.cf` becomes a package, not three flat
  sibling modules. §4.1. The move became PR 1, on its own, so the programme
  went from six pull requests to seven.
- **Cross-reader testing:** xarray is added as a test-only dependency. Measured
  at exactly one extra conda package in `iris-dev`, so no `skip_gdal`-style
  guard is needed. §6.
- **EOPF's role:** confirmed Zarr version 2 by reading `.zgroup`, so it keeps
  its role as the version 2 and deep-group fixture. §7.
- **Attribute model:** parked rather than decided; JSON-native is provisional.
  Q1 above, issue #7288, §10.
- **Method:** behaviour is justified from the Zarr and CF specifications, and a
  real file is cited only as evidence that a code path gets exercised. Two
  earlier claims credited to "real data" were re-examined under this rule: the
  base64 `_FillValue` turned out to be a publisher's CF violation, and nested
  JSON attributes turned out to be already in the Zarr specification. Now
  recorded in `lib/iris/AGENTS.md`.

**2026-09-22 — agreed with @trexfeathers**

- ~~**This programme is a proof of concept**, with official pull requests to
  follow.~~ **Superseded the same day** — see the next entry.
- **Issue state is not tracked in this document.** It lives on GitHub. §12.2.
  *(Survives the retraction above; it is a rule about what a spec is for.)*
- **Seven separate pull requests**, not a collapsed branch. Per-step CI and
  independent verification are worth the extra cycles.

**2026-09-22 — retraction, agreed with @trexfeathers**

- **This is the delivery plan, not a proof of concept.** @trexfeathers: "I was
  mistaken to introduce talk of proof-of-concept. This spec is THE spec we will
  be using, and I need to see your strategy for real, even if everything is
  being isolated to the feature branch." The programme is unchanged in shape;
  what changes is that it is expected to reach users.
- **Issues resolve at merge-back.** A closing keyword only fires on merge into
  the default branch, so the feature-branch pull requests write `Part of
  #NNNN` and the merge-back carries `Closes`. §5.
- **Changelog fragments land with the merge-back**, for the same structural
  reason and matching established practice on this branch: #7285 and #7287
  carry no fragment, #7267 into `main` does **[verified]**. The merge-back owes
  five fragments; §5 lists them. This supersedes the earlier "changelog no"
  reasoning, which was justified by the proof-of-concept framing rather than by
  how the repository works.
- **Tests are required on every pull request**, unchanged. `tests/AGENTS.md`.

**2026-09-22 — cross-cutting design review, with @trexfeathers**

- **The `CFUGrid*` classes stay with the other variable classes.** @trexfeathers
  asked why they warranted a private module of their own; they did not. They
  are peers of the classic classifiers, share the private `_is_str_dtype`
  helper, and appear in the same tuples in `CFReader` and `CFGroup`. The
  earlier four-way split was a line count in search of a rationale. `cf`
  splits three ways instead, and `_variables.py` is allowed over the
  ~1000-line aim. §4.1.
- **`CFVariable.__getattr__` works identically for both backends.** @trexfeathers
  asked whether `lib/iris/AGENTS.md` overreached in banning `__getattr__`. It
  did, in one place, and the spec was worse: it had the accessor raise
  `TypeError` for Zarr-backed variables, which holed the abstraction §4.2
  exists to provide. `__getattr__` now reads through `.attributes` on the base
  class for netCDF and Zarr alike, drops the `setattr` instance caching, and
  keeps a one-cycle deprecating fallback to the netCDF4 object. §4.3.
- **The `getattr` rules in `lib/iris/AGENTS.md` are narrowed, not lifted.**
  `getattr(var, "cf_role", "")` was never banned — the rule says *computed*
  names and *string dispatch*, and `iris.fileformats` has 42 constant-name
  lookups and **zero** dispatch sites **[verified]**. That stands, with a
  clarifying clause. The blanket `__getattr__` ban is replaced by a checkable
  exception: open-world data keys read from a file, forwarding to a declared
  `Mapping`, with that `Mapping` as the path library code takes.
- **Re-exporting from `cf/__init__.py` needs no justification.** @trexfeathers
  noted that `lib/iris/AGENTS.md` made surfacing low-level objects at a higher
  level read as naughty, and that the spec had written a defensive paragraph in
  response. Neither was right: `iris.mesh` already re-exports from four
  modules across two subtrees, and the "do not add indirection" rule is about
  implementation layers. `AGENTS.md` now states the pattern positively, with
  the conditions that keep it greppable — verbatim, static, `__all__`, no
  renaming, no conditional imports. §4.1.
- **Multi-process writing stays out of scope, but the design must not
  preclude it.** @trexfeathers: it is "SPECIFICALLY DESIRED by multiple users, and
  is a USP of Iris when it comes to NetCDF". Four provisions added in §4.5:
  `_dask_locks.py` relocates to `cf/` in PR 5, `CFDatasetVariable.write_handle`
  becomes the coordination seam, the lock stays inside the dataset
  implementation, and `CFDataset` carries `mode` from the start.
- **The chunk-alignment invariant is a correctness requirement now.** Writing
  `da.store(..., lock=False)` is only safe where the lazy source tiles the Zarr
  chunk grid exactly; two tasks sharing a chunk silently lose one update. The
  saver derives `chunks` from the source's dask chunking, and rechunks the
  source when the caller forces a conflicting `chunksizes=`. §4.5.
- **`finalise()` is separate from `close()`.** Consolidating metadata is
  one-shot and belongs to the returned `Delayed` under `compute=False`; N
  workers consolidating one metadata object would race. §4.2, §4.5.
- **Store durability was an oversight; now covered in §4.5.** @trexfeathers asked
  whether metadata/data desynchronisation is impossible in the Iris write
  chain. It is for a fresh store — metadata is written once at array creation
  and never edited — but not for `save(..., group=...)`, which writes into a
  store that may already exist. Both hazards confirmed against zarr 3.4.0
  **[verified]**: a missing chunk reads back as `fill_value` with no warning,
  and consolidated metadata goes silently stale so a default reader cannot see
  arrays added without re-consolidation.
- **The create-once invariant is written down as a constraint**, not left as an
  accident of the current chain: Iris never resizes, rechunks or changes the
  dtype, `fill_value`, codecs or dimension names of an existing array. It is
  also most of the safety argument for future parallel writes. §4.5.
- **`save` defaults to `mode="w-"` — raise if the target exists.** Chosen by
  @trexfeathers over matching the netCDF saver's clobber, because a Zarr clobber is
  many non-atomic deletes and an interrupt destroys the old store without
  completing the new one. `mode="w"` opts back in; `mode="a"` is what `group=`
  needs. Re-consolidation is mandatory when adding to a consolidated store.
- **No torn-write detection, and write-to-temp-then-rename is rejected.**
  Directory rename is atomic on POSIX but object stores have no equivalent, so
  the guarantee would evaporate where Zarr is most used. Icechunk is the
  ecosystem's answer and stays out of scope; the documentation says so. §4.5.
- **There is no `ZarrDataProxy`.** @trexfeathers asked whether the netCDF proxy
  classes are actually necessary for Zarr, given that chunks are inherent to
  the store. They are not. Every reason `NetCDFDataProxy` exists is a
  netCDF4/HDF5 reason — an unpicklable `Variable`, a thread-unsafe library, an
  open file handle — and none holds for Zarr, where `zarr.Array` is picklable
  **[verified]** and a chunk is an independent object needing no lock. Two of
  the draft's three justifications for a proxy were false on inspection; the
  third, the dask cache key, wants a hashable identity rather than a class.
  §4.4.
- **CF decoding is an explicit dask graph layer**, not hidden inside a proxy's
  `__getitem__`: lazy, inspectable, fuseable, and testable without a store.
  §4.4.
- **`as_lazy_data` gains `cache_key=`**, and `iris/_lazy_data.py` stops
  importing `NetCDFDataProxy` from `iris.fileformats.netcdf`. The netCDF loader
  passes the key it produces today, so the change is behaviour-preserving.
  PR 3. §4.4.
- **Read-side chunk alignment, mirroring the write-side invariant.** A Zarr
  chunk is the atomic unit of storage, so dask chunking must never subdivide
  it. `_optimum_chunksize` ordinarily expands to whole multiples, but **shrinks
  below the store chunk when the store chunk already exceeds the 128 MiB dask
  target** — `(8000, 8000)` becomes `(2000, 8000)`, so four tasks each fetch
  and decompress the same 488 MiB object **[verified]**. The Zarr loader rounds
  up instead, and warns when the store's layout is the binding constraint.
  §4.4.
- **No `ZarrWriteProxy` either, but `write_handle()` stays.** @trexfeathers asked
  whether the write proxy is purely multi-process machinery and so purely
  future scope. It is not: `NetCDFWriteProxy`'s first job is to be a
  `__setitem__` target that outlives the closed `Dataset`, which a deferred
  save needs even single-threaded. The lock is its second job. Zarr needs
  neither — nothing to close, `zarr.Array` picklable, distinct chunks
  independent — and the delayed graph was verified to survive
  `pickle.dumps`/`loads` before computing. `write_handle()` is therefore
  justified by present netCDF need, not by future-proofing. §4.5.
- **Native Zarr restores deferred saving, which NCZarr gave up.**
  `Saver.__exit__` computes NCZarr writes eagerly because netCDF-c cannot
  reopen a Zarr store for deferred writes. Going direct removes that
  limitation, so `compute=False` works for Zarr; PR 6 tests it. §4.5.
- **Found in passing, not fixed:** `iris.save(..., compute=False)` returns a
  tuple rather than the documented `Delayed`, so the documented
  `result.compute()` raises **[verified]**. Untested because every internal
  caller uses `dask.compute`. Q5 in §12.3; a netCDF issue of its own.

### 12.5 Artefacts

| Artefact | Location | State |
|---|---|---|
| Cut-down NOAA GFS fixture, 1.4 MB, 12 members, Zarr v3 | `~/projects/iris-zarr-testdata/gfs_forecast_sample.zarr` | Cut, not yet contributed to `iris-test-data` |
| Cut-down ESA EOPF fixture, Zarr v2 | — | Not started |

The GFS subset was cut ahead of the rest of the programme because its source
endpoint is documented as closing on **30 September 2026** (§8).

### 12.6 Document history

| Date | Change |
|---|---|
| 2026-09-21 | First draft: design, six-pull-request programme, four open decisions. |
| 2026-09-21 | Grounded against the real NOAA GFS store. |
| 2026-09-21 | Package layout adopted; programme grew to seven pull requests; the two "real data" findings re-framed as one specification fact and one publisher malformation; EOPF confirmed version 2; three decisions settled and the attribute-model question parked to #7288. |
| 2026-09-21 | Added this progress record (§12). |
| 2026-09-22 | Reframed as a proof of concept; removed issue-closure claims and issue-state tracking. |
| 2026-09-22 | Confirmed seven separate pull requests; tests required, changelog fragments deferred. |
| 2026-09-22 | Proof-of-concept framing retracted. Added the merge-back step (§5) that carries the closing keywords, the five changelog fragments and the documentation. |
| 2026-09-22 | Dropped the proposed `cf/_ugrid.py`; the `cf` package splits three ways, not four. |
| 2026-09-22 | Rewrote §4.3: `__getattr__` separated from the backend-proxy job, made backend-agnostic, and the attribute-tracking consequence for PR 2 called out. `lib/iris/AGENTS.md` narrowed to match. |
| 2026-09-22 | Dropped the apologetic framing of the `cf/__init__.py` re-exports; `lib/iris/AGENTS.md` now endorses the pattern. |
| 2026-09-22 | Multi-process writing kept reachable: chunk-alignment invariant, `write_handle` seam, `_dask_locks` relocation, `mode` and `finalise` on the ABC. |
| 2026-09-22 | Added store durability (§4.5): the create-once invariant, `mode="w-"` by default, mandatory re-consolidation, and the two format hazards verified against zarr 3.4.0. |
| 2026-09-22 | Dropped `ZarrDataProxy` (§4.4): decoding becomes an explicit dask graph layer, `as_lazy_data` gains `cache_key=`, and the read-side chunk-alignment guard was added after `_optimum_chunksize` was shown to subdivide large store chunks. |
| 2026-09-22 | Covered the write proxy (§4.5): no Zarr equivalent needed, `write_handle()` justified by present netCDF need, native Zarr regains the deferred saving NCZarr gave up, and the `da.store` return-type defect recorded as Q5. |
