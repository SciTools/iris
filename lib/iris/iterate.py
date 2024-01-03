# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Cube functions for iteration in step."""

from collections.abc import Iterator
import itertools
import warnings

import numpy as np

from iris.exceptions import IrisUserWarning

__all__ = ["izip"]


def izip(*cubes, **kwargs):
    """Return an iterator for iterating over a collection of cubes in step.

    If the input cubes have dimensions for which there are no common
    coordinates, those dimensions will be treated as orthogonal. The
    resulting iterator will step through combinations of the associated
    coordinates.

    Parameters
    ----------
    cubes : :class:`iris.cube.Cube`
        One or more :class:`iris.cube.Cube` instances over which to iterate in
        step. Each cube should be provided as a separate argument e.g.
        ``iris.iterate.izip(cube_a, cube_b, cube_c, ...)``.
    coords : str, coord or a list of strings/coords
        Coordinate names/coordinates of the desired subcubes (i.e. those
        that are not iterated over). They must all be orthogonal (i.e. point
        to different dimensions).
    ordered : bool, optional
        If True (default), the order of the coordinates in the resulting
        subcubes will match the order of the coordinates in the coords
        keyword argument. If False, the order of the coordinates will
        be preserved and will match that of the input cubes.

    Returns
    -------
    An iterator over a collection of tuples that contain the resulting subcubes.

    Examples
    --------
        >>> e_content, e_density = iris.load_cubes(
        ...     iris.sample_data_path('space_weather.nc'),
        ...     ['total electron content', 'electron density'])
        >>> for tslice, hslice in iris.iterate.izip(e_content, e_density,
        ...                                         coords=['grid_latitude',
        ...                                                 'grid_longitude']):
        ...    pass

    Notes
    -----
    This function maintains laziness when called; it does not realise data.
    See more at :doc:`/userguide/real_and_lazy_data`.

    """
    if not cubes:
        raise TypeError("Expected one or more cubes.")

    ordered = kwargs.get("ordered", True)
    if not isinstance(ordered, bool):
        raise TypeError("Expected bool ordered parameter, got %r" % ordered)

    # Convert any coordinate names to coordinates (and ensure each cube has
    # requested slice coords).
    coords_to_slice = kwargs.get("coords")
    coords_by_cube = []

    for cube in cubes:
        if coords_to_slice is None or not coords_to_slice:
            coords_by_cube.append([])
        else:
            coords_by_cube.append(cube._as_list_of_coords(coords_to_slice))

    # For each input cube, generate the union of all describing dimensions for
    # the resulting subcube.
    requested_dims_by_cube = []
    for coords, cube in zip(coords_by_cube, cubes):
        requested_dims = set()
        for coord in coords:
            requested_dims.update(cube.coord_dims(coord))

        # Make sure this cube has no shared dimensions between the requested
        # coords.
        if len(requested_dims) != sum(
            (len(cube.coord_dims(coord)) for coord in coords)
        ):
            msg = (
                "The requested coordinates (%r) of cube (%r) are not "
                "orthogonal." % ([coord.name() for coord in coords], cube)
            )
            raise ValueError(msg)

        requested_dims_by_cube.append(requested_dims)

    # Checks on coordinates you are going to iterate over.
    # Create a list of sets (one set per cube), with each set containing the
    # dimensioned coordinates that will be iterated over (i.e exclude slice
    # coords).
    dimensioned_iter_coords_by_cube = []
    for requested_dims, cube in zip(requested_dims_by_cube, cubes):
        dimensioned_iter_coords = set()
        # Loop over dimensioned coords in each cube.
        for dim in range(len(cube.shape)):
            if dim not in requested_dims:
                dimensioned_iter_coords.update(cube.coords(contains_dimension=dim))
        dimensioned_iter_coords_by_cube.append(dimensioned_iter_coords)

    # Check for multidimensional coords - current implementation cannot
    # iterate over multidimensional coords.
    # for dimensioned_iter_coords in dimensioned_iter_coords_by_cube:
    #    for coord in dimensioned_iter_coords:
    #        if coord.ndim > 1:
    #            raise iris.exceptions.CoordinateMultiDimError(coord)

    # Iterate through all the possible pairs of cubes to compare dimensioned
    # coordinates.
    pairs_iter = itertools.combinations(dimensioned_iter_coords_by_cube, 2)
    for dimensioned_iter_coords_a, dimensioned_iter_coords_b in pairs_iter:
        coords_by_def_a = set(
            _CoordWrapper(coord) for coord in dimensioned_iter_coords_a
        )
        coords_by_def_b = set(
            _CoordWrapper(coord) for coord in dimensioned_iter_coords_b
        )

        # Check that the dimensioned coords that are common across the cubes
        # (i.e. have same definition/metadata) have the same shape. If this is
        # not the case it makes no sense to iterate through the coordinate in
        # step and an exception is raised.
        common = coords_by_def_a & coords_by_def_b
        for definition_coord in common:
            # Extract matching coord from dimensioned_iter_coords_a and
            # dimensioned_iter_coords_b to access shape.
            coord_a = next(
                coord
                for coord in dimensioned_iter_coords_a
                if definition_coord == coord
            )
            coord_b = next(
                coord
                for coord in dimensioned_iter_coords_b
                if definition_coord == coord
            )
            if coord_a.shape != coord_b.shape:
                raise ValueError(
                    "Shape of common dimensioned coordinate '%s' "
                    "does not match across all cubes. Unable "
                    "to iterate over this coordinate in "
                    "step." % coord_a.name()
                )
            if coord_a != coord_b:
                warnings.warn(
                    "Iterating over coordinate '%s' in step whose "
                    "definitions match but whose values "
                    "differ." % coord_a.name(),
                    category=IrisUserWarning,
                )

    return _ZipSlicesIterator(cubes, requested_dims_by_cube, ordered, coords_by_cube)


class _ZipSlicesIterator(Iterator):
    """Extension to _SlicesIterator (see cube.py) to support iteration over a
    collection of cubes in step.

    """

    def __init__(self, cubes, requested_dims_by_cube, ordered, coords_by_cube):
        self._cubes = cubes
        self._requested_dims_by_cube = requested_dims_by_cube
        self._ordered = ordered
        self._coords_by_cube = coords_by_cube

        # Check that the requested_dims_by_cube and coords_by_cube lists are
        # the same length as cubes so it is feasible that there is a 1-1
        # mapping of values (itertool.izip won't catch this).
        if len(requested_dims_by_cube) != len(cubes):
            raise ValueError(
                "requested_dims_by_cube parameter is not the same length as cubes."
            )
        if len(coords_by_cube) != len(cubes):
            raise ValueError(
                "coords_by_cube parameter is not the same length as cubes."
            )

        # Create an all encompassing dims_index called master_dims_index that
        # is iterated over (using np.ndindex) and from which the indices of the
        # subcubes can be extracted using offsets i.e. position of the
        # associated coord in the master_dims_index.
        master_dimensioned_coord_list = []
        master_dims_index = []
        self._offsets_by_cube = []
        for requested_dims, cube in zip(requested_dims_by_cube, cubes):
            # Create a list of the shape of each cube, and set the dimensions
            # which have been requested to length 1.
            dims_index = list(cube.shape)
            for dim in requested_dims:
                dims_index[dim] = 1
            offsets = []
            # Loop over dimensions in each cube.
            for i in range(len(cube.shape)):
                # Obtain the coordinates for this dimension.
                cube_coords = cube.coords(dimensions=i)
                found = False
                # Loop over coords in this dimension (could be just one).
                for coord in cube_coords:
                    # Search for coord in master_dimensioned_coord_list.
                    for j, master_coords in enumerate(master_dimensioned_coord_list):
                        # Use coord wrapper with desired equality
                        # functionality.
                        if _CoordWrapper(coord) in master_coords:
                            offsets.append(j)
                            found = True
                            break
                    if found:
                        break
                # If a coordinate with an equivalent definition (i.e. same
                # metadata) is not found in the master_dimensioned_coord_list,
                # add the coords associated with the dimension to the list,
                # add the size of the dimension to the master_dims_index and
                # store the offset.
                if not found:
                    master_dimensioned_coord_list.append(
                        set((_CoordWrapper(coord) for coord in cube_coords))
                    )
                    master_dims_index.append(dims_index[i])
                    offsets.append(len(master_dims_index) - 1)
            # Store the offsets for each cube so they can be used in
            # _ZipSlicesIterator.next().
            self._offsets_by_cube.append(offsets)

        # Let Numpy do some work in providing all of the permutations of our
        # data shape based on the combination of dimension sizes called
        # master_dims_index. This functionality is something like:
        # ndindex(2, 1, 3) -> [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0),
        # (1, 0, 1), (1, 0, 2)]
        self._ndindex = np.ndindex(*master_dims_index)

    def __next__(self):
        # When self._ndindex runs out it will raise StopIteration for us.
        master_index_tuple = next(self._ndindex)

        subcubes = []
        for offsets, requested_dims, coords, cube in zip(
            self._offsets_by_cube,
            self._requested_dims_by_cube,
            self._coords_by_cube,
            self._cubes,
        ):
            # Extract the index_list for each cube from the master index using
            # the offsets and for each of the spanning dimensions requested,
            # replace the index_list value (will be a zero from np.ndindex())
            # with a spanning slice.
            index_list = [master_index_tuple[x] for x in offsets]
            for dim in requested_dims:
                index_list[dim] = slice(None, None)
            # Extract slices from the cube
            subcube = cube[tuple(index_list)]
            # Call transpose if necessary (taken from _SlicesIterator in
            # cube.py).
            if self._ordered is True:
                transpose_order = []
                for coord in coords:
                    transpose_order += sorted(subcube.coord_dims(coord))
                if transpose_order != list(range(subcube.ndim)):
                    subcube.transpose(transpose_order)
            subcubes.append(subcube)

        return tuple(subcubes)

    next = __next__


class _CoordWrapper:
    """Class for creating a coordinate wrapper that allows the use of an
    alternative equality function based on metadata rather than
    metadata + points/bounds.

    .. note::

        Uses a lightweight/incomplete implementation of the Decorator
        pattern.

    """

    def __init__(self, coord):
        self._coord = coord

    # Methods of contained class we need to expose/use.
    @property
    def metadata(self):
        return self._coord.metadata

    # Methods of contained class we want to override/customise.
    def __eq__(self, other):
        return self._coord.metadata == other.metadata

    # Force use of __eq__ for set operations.
    def __hash__(self):
        return 1
