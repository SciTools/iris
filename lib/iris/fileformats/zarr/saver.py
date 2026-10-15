# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Save cubes to a Zarr store.

.. z_reference:: iris.fileformats.zarr.saver
   :tags: topic_load_save

   API reference

Writes Zarr version 3 stores. The CF work - dimension names, coordinate,
bounds, mesh and grid-mapping variables, attribute compliance - belongs to
:class:`iris.fileformats.cf.saver.CFSaver`, which this builds on. What is here
is what a Zarr store does differently: it is created rather than opened, it
holds text natively, and it finishes with a consolidated metadata document.

**The storage keywords are zarr-python's own**: ``chunks``, ``shards``,
``compressors``, ``filters``, ``serializer``. A netCDF keyword raises and names
the Zarr replacement. Translating ``zlib=True`` onto a Blosc codec would sell a
false equivalence, and would hand this new API the shape the netCDF saver is
stuck with. See §4.5 of the native Zarr I/O design,
``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

**The default mode is ``"w-"``**, which refuses an existing target, where the
netCDF saver clobbers one. A Zarr clobber is many non-atomic deletes, so an
interrupted one destroys the old store without completing the new.

"""

from pathlib import Path

import dask
from dask.delayed import Delayed
import numpy as np

import iris
from iris._lazy_data import is_lazy_data, is_lazy_masked_data
from iris.cube import CubeList
from iris.fileformats.cf import saver as cf_saver
from iris.fileformats.cf.saver import CFSaver

from ._dataset import ZarrDataset, check_encoding_keywords, existing_name_message

__all__ = ["Saver", "save"]


def _tiles_the_grid(chunks, grid) -> bool:
    """Whether every chunk boundary falls on a write-grid boundary.

    Parameters
    ----------
    chunks : tuple of tuple of int
        A dask array's ``chunks``: the lengths along each dimension.
    grid : tuple of int
        The write-grid cell shape.

    Returns
    -------
    bool
        True when no grid cell is touched by two chunks. A truncated final
        cell is permitted, so the last boundary is not checked.

    """
    for lengths, cell in zip(chunks, grid, strict=True):
        offset = 0
        for length in lengths[:-1]:
            offset += length
            if offset % cell:
                return False
    return True


def _align_to_write_grid(data, cf_var):
    """Return `data` chunked so that concurrent writes cannot collide.

    Parameters
    ----------
    data : dask.array.Array
        The lazy source about to be stored.
    cf_var : :class:`~iris.fileformats.zarr._dataset.ZarrDatasetVariable`
        The array it is going into.

    Returns
    -------
    dask.array.Array
        `data` itself when it already tiles the write grid - a needless
        rechunk is a needless graph layer - and a rechunked view otherwise.

    """
    grid = cf_var.write_grid
    if not grid or _tiles_the_grid(data.chunks, grid):
        return data
    return data.rechunk(grid)


def _complete(_written, dataset, consolidating) -> None:
    """Consolidate `dataset`'s metadata, once its delayed writes have run.

    A module-level function, rather than a closure, so that the
    :class:`~dask.delayed.Delayed` built from it survives a plain
    ``pickle.dumps``/``pickle.loads`` round trip and not only a cloudpickle
    one - what a :mod:`distributed` scheduler would use regardless.

    Parameters
    ----------
    _written : Any
        The result of the delayed writes this is chained after. Unused,
        beyond making this task depend on them.
    dataset : :class:`~iris.fileformats.zarr._dataset.ZarrDataset`
        The store to consolidate.
    consolidating : bool
        Whether to consolidate at all.

    """
    if consolidating:
        dataset.finalise()


class Saver(CFSaver):
    """A manager for saving cubes to a Zarr store."""

    def __init__(
        self,
        filename,
        compute=True,
        *,
        mode=None,
        group=None,
        consolidate_metadata=True,
    ):
        """Manage saving cubes into a Zarr store.

        Parameters
        ----------
        filename : str or pathlib.Path
            The store's path or URL. A URL is handed to zarr-python
            unaltered.
        compute : bool, default=True
            If ``True``, lazy data is written and the metadata consolidated
            on exit from the saver context. If ``False``, those writes are
            returned as a :class:`~dask.delayed.Delayed` by
            :meth:`delayed_completion`, which also owns the consolidation.
        mode : str, optional
            One of ``"w-"``, ``"w"``, ``"a"`` or ``"r+"``. The default is
            ``"w-"``, which refuses an existing target, or ``"a"`` when
            `group` is given, since adding a group to a store means the store
            may already be there.
        group : str, optional
            A ``/``-separated path to a group within the store. The root
            group by default.
        consolidate_metadata : bool, default=True
            Whether to write a consolidated metadata document at the root
            when the save completes. Consolidation is what makes a remote
            store usable; declining it suits a store that will be added to
            again before anyone reads it.

        Examples
        --------
        >>> import iris  # doctest: +SKIP
        >>> from iris.fileformats.zarr.saver import Saver  # doctest: +SKIP
        >>> with Saver("out.zarr") as saver:  # doctest: +SKIP
        ...     for cube in cubes:  # doctest: +SKIP
        ...         saver.write(cube)  # doctest: +SKIP

        """
        if mode is None:
            mode = "a" if group is not None else "w-"
        dataset = ZarrDataset(filename, mode=mode, group=group)
        super().__init__(dataset, compute=compute)
        #: Whether the completed store gets a consolidated metadata document.
        self.consolidate_metadata = consolidate_metadata
        #: Names the group already held when this `Saver` opened it.
        #: Snapshotted once, here, rather than read live from
        #: `self._dataset.variables`: that mapping is invalidated on every
        #: `create_variable` call, so a live read would also catch a name
        #: this same call created moments before - right for something the
        #: group held before this call, wrong for two cubes of one call
        #: sharing a name.
        self._pre_existing_names = frozenset(dataset.variables)

    def write(self, cube, *args, **kwargs):
        """Write a cube into the store.

        Takes what :meth:`iris.fileformats.cf.saver.CFSaver.write` takes,
        with Zarr's storage keywords in ``**encoding``. ``chunks`` defaults to
        the cube's own lazy chunking, so that the data is written in the shape
        it is held in, and so that the dask source tiles the array's write grid
        without being rechunked.

        Raises
        ------
        ValueError
            If the cube's data variable would collide with something the
            group already held before this call, under a mode that adds to
            an existing store. Left to the generic layer, such a collision
            is disambiguated by renaming rather than raised: right for two
            cubes of one call sharing a name, wrong for a store this call did
            not just create, where adding is not a merge.

        """
        if self._dataset.mode in ("a", "r+"):
            cf_name = self._get_element_variable_name(cube, cube)
            if cf_name in self._pre_existing_names:
                raise ValueError(existing_name_message(cf_name, self._dataset.location))
        if "chunks" not in kwargs and cube.has_lazy_data():
            kwargs["chunks"] = cube.lazy_data().chunksize
        super().write(cube, *args, **kwargs)

    def _create_string_variable(
        self, cube_or_mesh, cf_name, element, element_dims, data
    ):
        """Create a variable-length string array.

        Zarr has a string type, so unlike netCDF there is no trailing
        character dimension to invent and `element_dims` is left alone.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved. Unused here.
        cf_name : str
            The name the variable is created with.
        element : :class:`iris.coords._DimensionalMetadata`
            The element being saved. Unused here.
        element_dims : list of str
            The variable's dimension names. **Not** appended to.
        data : :class:`numpy.ndarray`
            The string values. Unused here: a Zarr string array has no width.

        Returns
        -------
        :class:`~iris.fileformats.zarr._dataset.ZarrDatasetVariable`

        """
        return self._dataset.create_variable(cf_name, str, element_dims)

    def _backend_write(self, data, cf_var) -> bool:
        """Declare a mask, align a lazy source, and collect it.

        Parameters
        ----------
        data : array-like
            The values to write.
        cf_var : :class:`~iris.fileformats.zarr._dataset.ZarrDatasetVariable`
            The variable to write them into.

        Returns
        -------
        bool
            True when the write was collected here, which it is for lazy
            data. Real data is left to the base class, which writes it
            straight in.

        """
        # `np.ma.is_masked` is true only when something is actually masked,
        # unlike `np.ma.isMaskedArray`, which is true of an np.ma array whose
        # mask is all-False - and Iris produces those routinely, so that
        # test would declare a missing-value sentinel on data the source
        # never marked missing. Lazy data cannot be inspected without
        # computing it, so `is_lazy_masked_data` is the safe answer there:
        # declaring unnecessarily costs an attribute, where not declaring
        # could lose a real mask silently.
        if (
            is_lazy_masked_data(data) or np.ma.is_masked(data)
        ) and "_FillValue" not in cf_var.attributes:
            # On version 3 the attribute is the masking declaration and the
            # storage fill is not, so a store written with the fill alone
            # reads back with every masked point unmasked. Declared here
            # rather than when the data is written, because under
            # compute=False that is after the metadata is consolidated.
            cf_var.attributes["_FillValue"] = cf_var.fill_value

        if not is_lazy_data(data):
            return False

        self._delayed_writes.append(
            (_align_to_write_grid(data, cf_var), cf_var.write_handle())
        )
        return True

    def delayed_completion(self) -> Delayed:
        """Return a :class:`~dask.delayed.Delayed` that completes the store.

        The lazy writes first, then the metadata consolidation, which is the
        store's last write: a store whose deferred save never runs should not
        carry a metadata document claiming it did.

        Returns
        -------
        :class:`dask.delayed.Delayed`
            A real one. The inherited
            :meth:`~iris.fileformats.cf.saver.CFSaver.delayed_completion`
            returns a :class:`dask.array.core.Array` for one delayed write,
            or a tuple of them for several - never actually a
            :class:`~dask.delayed.Delayed`, despite its own annotation -
            which is what makes the documented ``result.compute()`` fail on
            the netCDF path
            (`SciTools/iris#7291 <https://github.com/SciTools/iris/issues/7291>`_).
            Wrapping it here is what keeps the promise this saver's docstring
            makes. Plain ``pickle`` survives the round trip too, not only
            cloudpickle: the completion is :func:`_complete`, a module-level
            function with the dataset and the consolidation flag bound as
            plain values, never reached through ``self`` - the saver holds
            cubes, which must not be dragged into the graph.

        """
        complete = dask.delayed(_complete)
        return complete(
            super().delayed_completion(), self._dataset, self.consolidate_metadata
        )


def _per_cube(value, count, name):
    """Return one `value` per cube, from one value or a sequence of them."""
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            message = (
                f"If {name} is a sequence, it must have one element per cube: "
                f"got {len(value)} for {count} cubes."
            )
            raise ValueError(message)
        return list(value)
    return [value] * count


def save(
    cube,
    filename,
    *,
    mode=None,
    group=None,
    consolidate_metadata=True,
    local_keys=None,
    fill_value=None,
    packing=None,
    compute=True,
    **encoding,
):
    """Save cube(s) to a Zarr store.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`
        The cube, or cubes, to save. A whole :class:`~iris.cube.CubeList`
        goes into one store, which is opened once.
    filename : str or pathlib.Path
        The store's path or URL.
    mode : str, optional
        See :class:`Saver`. ``"w-"`` by default, ``"a"`` when `group` is
        given.
    group : str, optional
        A ``/``-separated path to a group within the store.
    consolidate_metadata : bool, default=True
        See :class:`Saver`.
    local_keys : iterable of str, optional
        Cube attribute names that become attributes of the data variable
        rather than global attributes.
    fill_value : optional
        The value of the data variable's ``_FillValue`` attribute, and of the
        array's own fill. One value, or one per cube.
    packing : type or str or dict or list, optional
        How to pack the data, if at all. See
        :meth:`iris.fileformats.netcdf.saver.Saver.write`.
    compute : bool, default=True
        If ``False``, returns a :class:`~dask.delayed.Delayed` that completes
        the store when computed, instead of writing lazy data here.
    **encoding
        Zarr's own storage keywords for the data variable: ``chunks``,
        ``shards``, ``compressors``, ``filters``, ``serializer``, ``order``.
        A netCDF keyword raises :class:`TypeError` naming the Zarr
        equivalent.

    Returns
    -------
    None or :class:`dask.delayed.Delayed`
        A ``Delayed`` if `compute` is ``False``, otherwise ``None``.

    Notes
    -----
    An interrupted save leaves a partial store. Iris offers no rollback: a
    directory rename is atomic on a local filesystem but has no equivalent on
    an object store, so the guarantee would evaporate where Zarr is most used.

    """
    # Validated before anything is created, so that a mistyped keyword does
    # not leave an empty store behind.
    check_encoding_keywords(encoding)

    cubes = CubeList([cube]) if isinstance(cube, iris.cube.Cube) else CubeList(cube)
    fill_values = _per_cube(fill_value, len(cubes), "fill_value")
    packspecs = _per_cube(packing, len(cubes), "packing")

    with Saver(
        filename,
        compute=compute,
        mode=mode,
        group=group,
        consolidate_metadata=consolidate_metadata,
    ) as saver:
        for one, packspec, one_fill in zip(cubes, packspecs, fill_values):
            saver.write(
                one,
                local_keys=local_keys,
                packing=packspec,
                fill_value=one_fill,
                **encoding,
            )

        conventions = cf_saver.CF_CONVENTIONS_VERSION
        if iris.FUTURE.save_split_attrs:
            saver.update_global_attributes({"Conventions": conventions})
        else:
            saver.update_global_attributes(Conventions=conventions)

    return None if compute else saver.delayed_completion()
