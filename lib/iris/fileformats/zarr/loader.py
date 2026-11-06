# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Load Iris cubes from Zarr stores.

.. z_reference:: iris.fileformats.zarr.loader
   :tags: topic_load_save

   API reference

Iris loads **one** group per call. A Zarr store can legitimately hold
unrelated datasets, and merging them into one ``CubeList`` would be wrong, so
nested groups are not walked and ``group=`` selects which one to read.

UGRID meshes load from a Zarr store exactly as they do from a netCDF file.

.. maintainer-notes::

   A thin loop: open the store as a
   :class:`~iris.fileformats.zarr._dataset.ZarrDataset`, hand it to
   :class:`~iris.fileformats.cf.CFReader`, and build one cube per CF data
   variable with :mod:`iris.fileformats.cf.loader` - the same machinery the
   netCDF loader runs. Everything Zarr-specific happens below this module, in
   :mod:`iris.fileformats.zarr._dataset`.

   Nothing in the mesh machinery is netCDF-specific - it reads a
   :class:`~iris.fileformats.cf.CFReader`, and a ``CFReader`` over a
   ``ZarrDataset`` answers the same questions - so this module calls into
   :mod:`iris.fileformats.netcdf.ugrid_load` rather than growing a second
   copy. That module's name is the historical home of the code, not a
   statement about what it can read.

   See sections 4.6 and 4.7 of
   ``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

"""

from collections.abc import Iterable
import warnings

from iris.exceptions import TranslationError
from iris.fileformats.cf import CFReader
from iris.fileformats.cf.loader import (
    _actions_engine,
    _load_aux_factory,
    _load_cube,
    _translate_constraints_to_var_callback,
)
import iris.warnings

from ._dataset import ZarrDataset


def _groups_yielding_cubes(file_source, dataset) -> tuple:
    """Return the store's groups that would load as cubes, shallowest first.

    Candidates come from
    :attr:`~iris.fileformats.zarr._dataset.ZarrDataset.groups_holding_arrays`,
    and each is classified as a load of it would be. A group holding only
    coordinate variables makes no cubes, so suggesting it would raise the
    error being explained.

    Descendants of a group that loads are not examined. Costs each
    candidate's metadata.
    """
    found: list[str] = []
    for path in dataset.groups_holding_arrays:
        if any(path.startswith(f"{loadable}/") for loadable in found):
            continue
        with warnings.catch_warnings():
            # A pass over groups the user did not ask for; they will hear
            # anything that matters when they load one.
            warnings.simplefilter("ignore")
            with ZarrDataset(file_source, group=path) as candidate:
                with CFReader(candidate) as reader:
                    if reader.cf_group.data_variables or reader.cf_group.promoted:
                        found.append(path)
    return tuple(found)


def _check_has_data_variables(reader, dataset, file_source) -> None:
    """Raise if this group holds nothing to make a cube from.

    Returning an empty ``CubeList`` from a store that plainly holds data
    elsewhere leaves the user nothing to act on, and a store whose root group
    holds only subgroups is a real shape - the ESA EOPF Sentinel products are
    built that way.
    """
    if reader.cf_group.data_variables or reader.cf_group.promoted:
        return

    subgroups = _groups_yielding_cubes(file_source, dataset)
    if subgroups:
        message = (
            f"Zarr group {dataset.location!r} has no data variables. Load "
            "one of its groups instead, with the 'group' keyword: "
            f"{', '.join(subgroups)}"
        )
    else:
        message = f"Zarr store {dataset.location!r} has no data variables."
    raise TranslationError(message)


def load_cubes(file_sources, callback=None, constraints=None, group=None):
    """Load cubes from one or more Zarr stores.

    Parameters
    ----------
    file_sources : str or iterable of str
        The stores to load. A local directory path, or a URL that
        zarr-python can open - ``s3://``, ``gs://``, ``az://`` and ``abfs://``
        need the matching store library installed, which Iris does not
        depend on.
    callback : callable, optional
        Passed to :func:`iris.io.run_callback`.
    constraints : optional
        Used to skip data variables that cannot match, before they are built.
    group : str, optional
        A ``/``-separated path to a group within each store. The root group by
        default.

    Yields
    ------
    iris.cube.Cube

    Raises
    ------
    iris.exceptions.TranslationError
        If the selected group holds no data variables. The message names the
        store's other groups, when it has any.

    Notes
    -----
    This function maintains laziness; it does not realise data.

    """
    from iris.fileformats.netcdf.ugrid_load import (
        _attach_mesh_coords,
        _meshes_from_cf,
        _prepare_mesh_coords,
    )
    from iris.io import run_callback

    var_callback = _translate_constraints_to_var_callback(constraints)
    engine = _actions_engine()

    if isinstance(file_sources, str) or not isinstance(file_sources, Iterable):
        file_sources = [file_sources]

    for file_source in file_sources:
        with ZarrDataset(file_source, group=group) as dataset:
            with CFReader(dataset) as cf:
                _check_has_data_variables(cf, dataset, file_source)

                meshes = _meshes_from_cf(cf)

                data_variables = list(cf.cf_group.data_variables.values()) + list(
                    cf.cf_group.promoted.values()
                )
                for cf_var in data_variables:
                    if var_callback and not var_callback(cf_var):
                        continue

                    # MeshCoords are built before the cube and attached after
                    # it: see _prepare_mesh_coords for why the two cannot be
                    # one step.
                    pending_mesh_coords = _prepare_mesh_coords(cf, meshes, cf_var)

                    cube = _load_cube(engine, cf, cf_var, cf.filename)

                    if pending_mesh_coords is not None:
                        _attach_mesh_coords(pending_mesh_coords, cube)

                    try:
                        _load_aux_factory(engine, cube)
                    except ValueError as exception:
                        warnings.warn(
                            str(exception),
                            category=iris.warnings.IrisLoadWarning,
                        )

                    cube = run_callback(callback, cube, cf_var, file_source)
                    if cube is None:
                        # The callback rejected it.
                        continue

                    yield cube
