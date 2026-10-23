# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Load Iris cubes from NetCDF files and OPeNDAP URLs.

.. deprecated:: 3.17
    ``CHUNK_CONTROL`` and ``ChunkControl`` have moved to
    :mod:`iris.fileformats.cf.loader`, because chunking policy is not specific
    to netCDF. They can still be reached here, with a warning, for one
    deprecation cycle.

.. z_reference:: iris.fileformats.netcdf.loader
   :tags: topic_load_save

   API reference

What is left here is what is genuinely netCDF's: opening the file sources, and
the proxy class that reads from them. Interpreting what is inside is
:mod:`iris.fileformats.cf.loader`'s job.

See : `NetCDF User's Guide <https://docs.unidata.ucar.edu/nug/current/>`_
and `netCDF4 python module <https://github.com/Unidata/netcdf4-python>`_.

Also : `CF Conventions <https://cfconventions.org/>`_.

"""

from collections.abc import Iterable
import warnings

from iris._deprecation import warn_deprecated
from iris.fileformats.cf.loader import (
    _actions_engine,
    _load_aux_factory,
    _load_cube,
    _translate_constraints_to_var_callback,
)
import iris.warnings

# Show actions activation statistics.
# NOTE: read by iris.fileformats.cf.loader._load_cube_inner, through this
# module rather than by value, so that assigning to it here still works.
DEBUG = False

# Get _bytecoding_datasets, needed by NetCDFDataProxy just below.
from . import _bytecoding_datasets

# An expected part of the public loader API, but includes thread safety
#  concerns so is housed in _thread_safe_nc.
# NOTE: this is the *default*, as required for public legacy api
#  - in practice, when creating our proxies we dynamically choose between this and
#    :class:`_thread_safe_nc.DatasetWrapper`, depending on
#    :data:`_bytecoding_datasets.DECODE_TO_STRINGS_ON_READ`
NetCDFDataProxy = _bytecoding_datasets.EncodedNetCDFDataProxy

#: Names that moved to :mod:`iris.fileformats.cf.loader` in Iris 3.17 and are
#: still readable here, with a warning, for one deprecation cycle.
#:
#: Read-only: a module ``__getattr__`` cannot intercept assignment, so a name
#: that users *set* must keep its definition here instead. ``DEBUG`` is the
#: one such name, and is deliberately absent from this set.
#:
#: The same limitation means ``mock.patch("iris.fileformats.netcdf.loader."
#: "CHUNK_CONTROL", ...)`` succeeds and is a silent no-op: ``mock.patch``
#: creates the attribute on this module, while production code reads
#: ``iris.fileformats.cf.loader.CHUNK_CONTROL``. Patch
#: :mod:`iris.fileformats.cf.loader` instead.
_RELOCATED_NAMES = frozenset({"CHUNK_CONTROL", "ChunkControl"})


def __getattr__(name):
    """Forward a relocated name to :mod:`iris.fileformats.cf.loader`.

    Parameters
    ----------
    name : str
        The attribute being looked up. Python calls this only for names that
        are not already module attributes.

    Returns
    -------
    Any
        The object of that name in :mod:`iris.fileformats.cf.loader` - the
        same object, not a copy.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the relocated names. Raised without warning,
        so that a ``hasattr`` probe that comes back False stays quiet.

    """
    if name not in _RELOCATED_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Deferred so that importing this module stays quiet and cheap.
    from iris.fileformats.cf import loader as cf_loader

    warn_deprecated(
        f"iris.fileformats.netcdf.loader.{name} has moved to "
        f"iris.fileformats.cf.loader.{name}, because it is not specific to "
        "netCDF. The name here will be removed in a future release.",
        # warn_deprecated's default stacklevel=2 points at the line below,
        # inside this function - PEP 562 means that line is a genuine call
        # frame, not the user's code. One level further out reaches the
        # line that actually read the attribute.
        stacklevel=3,
    )
    return getattr(cf_loader, name)


def __dir__():
    """List module attributes, including the relocated names.

    A module ``__getattr__`` forwards reads of the relocated names, but does
    not make them discoverable: PEP 562 does not extend to ``dir()``,
    :func:`inspect.getmembers`, or tab-completion, so this fills the gap.

    Returns
    -------
    list of str
        The real module globals, unioned with the relocated names.

    """
    return sorted(set(globals()) | _RELOCATED_NAMES)


def load_cubes(file_sources, callback=None, constraints=None):
    """Load cubes from a list of NetCDF filenames/OPeNDAP URLs.

    Also supports Zarr files in the NcZarr URL format, e.g.
    ``file:///path/to/file#mode=nczarr,file``. Note that NcZarr is limited to
    Zarr Storage Specification version 2. See the
    `NcZarr docs <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`
    for more.

    Parameters
    ----------
    file_sources : str or list
        One or more NetCDF filenames/OPeNDAP URLs/NcZarr URLs to load from.
        OR open datasets.
    callback : function, optional
        Function which can be passed on to :func:`iris.io.run_callback`.
    constraints : optional

    Returns
    -------
    Generator of loaded NetCDF/NcZarr :class:`iris.cube.Cube`.

    """
    # Deferred import to avoid circular imports.
    from iris.fileformats.cf import CFReader
    from iris.io import run_callback

    from .ugrid_load import (
        _attach_mesh_coords,
        _meshes_from_cf,
        _prepare_mesh_coords,
    )

    # Create a low-level data-var filter from the original load constraints, if they are suitable.
    var_callback = _translate_constraints_to_var_callback(constraints)

    # Create an actions engine.
    engine = _actions_engine()

    if isinstance(file_sources, str) or not isinstance(file_sources, Iterable):
        file_sources = [file_sources]

    for file_source in file_sources:
        # Ingest the file.  At present may be a filepath or an open netCDF4.Dataset.
        with CFReader(file_source) as cf:
            meshes = _meshes_from_cf(cf)

            # Process each CF data variable.
            data_variables = list(cf.cf_group.data_variables.values()) + list(
                cf.cf_group.promoted.values()
            )
            for cf_var in data_variables:
                if var_callback and not var_callback(cf_var):
                    # Deliver only selected results.
                    continue

                # MeshCoords are built before the cube and attached after it:
                #  see _prepare_mesh_coords for why the two cannot be one step.
                pending_mesh_coords = _prepare_mesh_coords(cf, meshes, cf_var)

                cube = _load_cube(engine, cf, cf_var, cf.filename)

                if pending_mesh_coords is not None:
                    _attach_mesh_coords(pending_mesh_coords, cube)

                # Process any associated formula terms and attach
                # the corresponding AuxCoordFactory.
                try:
                    _load_aux_factory(engine, cube)
                except ValueError as e:
                    warnings.warn(
                        "{}".format(e),
                        category=iris.warnings.IrisLoadWarning,
                    )

                # Perform any user registered callback function.
                cube = run_callback(callback, cube, cf_var, file_source)

                # Callback mechanism may return None, which must not be yielded
                if cube is None:
                    continue

                yield cube
