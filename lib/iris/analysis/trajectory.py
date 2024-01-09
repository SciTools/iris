# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Defines a Trajectory class, and a routine to extract a sub-cube along a
trajectory.

"""

import math

import numpy as np
from scipy.spatial import cKDTree

import iris.coords


class _Segment:
    """A single trajectory line segment.

    Two points, as described in the Trajectory class.
    """

    def __init__(self, p0, p1):
        # check keys
        if sorted(p0.keys()) != sorted(p1.keys()):
            raise ValueError("keys do not match")

        self.pts = [p0, p1]

        # calculate our length
        squares = 0
        for key in self.pts[0].keys():
            delta = self.pts[1][key] - self.pts[0][key]
            squares += delta * delta
        self.length = math.sqrt(squares)


class Trajectory:
    """A series of given waypoints with pre-calculated sample points."""

    def __init__(self, waypoints, sample_count=10):
        """Defines a trajectory using a sequence of waypoints.

        Parameters
        ----------
        waypoints :
            A sequence of dictionaries, mapping coordinate names to values.
        sample_count : int, optional, default=10
            The number of sample positions to use along the trajectory.

        Examples
        --------
        ::

            waypoints = [{'latitude': 45, 'longitude': -60},
                         {'latitude': 45, 'longitude': 0}]
            Trajectory(waypoints)

        .. note:: All the waypoint dictionaries must contain the same
            coordinate names.

        """
        self.waypoints = waypoints
        self.sample_count = sample_count

        # create line segments from the waypoints
        segments = [
            _Segment(self.waypoints[i], self.waypoints[i + 1])
            for i in range(len(self.waypoints) - 1)
        ]

        # calculate our total length
        self.length = sum([seg.length for seg in segments])

        # generate our sampled points
        # The trajectory points, as dictionaries of {coord_name: value}.
        self.sampled_points = []
        sample_step = self.length / (self.sample_count - 1)

        # start with the first segment
        cur_seg_i = 0
        cur_seg = segments[cur_seg_i]
        len_accum = cur_seg.length
        for p in range(self.sample_count):
            # calculate the sample position along our total length
            sample_at_len = p * sample_step

            # skip forward to the containing segment
            while len_accum < sample_at_len and cur_seg_i < len(segments):
                cur_seg_i += 1
                cur_seg = segments[cur_seg_i]
                len_accum += cur_seg.length

            # how far through the segment is our sample point?
            seg_start_len = len_accum - cur_seg.length
            seg_frac = (sample_at_len - seg_start_len) / cur_seg.length

            # sample each coordinate in this segment, to create a new
            # sampled point
            new_sampled_point = {}
            for key in cur_seg.pts[0].keys():
                seg_coord_delta = cur_seg.pts[1][key] - cur_seg.pts[0][key]
                new_sampled_point.update(
                    {key: cur_seg.pts[0][key] + seg_frac * seg_coord_delta}
                )

            # add this new sampled point
            self.sampled_points.append(new_sampled_point)

    def __repr__(self):
        return "Trajectory(%s, sample_count=%s)" % (
            self.waypoints,
            self.sample_count,
        )

    def _get_interp_points(self):
        """Translate `self.sampled_points` to the format expected by the
        interpolator.

        Returns
        -------
        `self.sampled points`
            `self.sampled points` in the format required by
            `:func:`~iris.analysis.trajectory.interpolate`.

        """
        points = {
            k: [point_dict[k] for point_dict in self.sampled_points]
            for k in self.sampled_points[0].keys()
        }
        return [(k, v) for k, v in points.items()]

    def _src_cube_anon_dims(self, cube):
        """A helper method to locate the index of anonymous dimensions on the
        interpolation target, ``cube``.

        Returns
        -------
        The index of any anonymous dimensions in ``cube``.

        """
        named_dims = [cube.coord_dims(c)[0] for c in cube.dim_coords]
        return list(set(range(cube.ndim)) - set(named_dims))

    def interpolate(self, cube, method=None):
        """Calls :func:`~iris.analysis.trajectory.interpolate` to interpolate
        ``cube`` on the defined trajectory.

        Assumes that the coordinate names supplied in the waypoints
        dictionaries match to coordinate names in `cube`, and that points are
        supplied in the same coord_system as in `cube`, where appropriate (i.e.
        for horizontal coordinate points).

        Parameters
        ----------
        cube :
             The source Cube to interpolate.
        method :
            The interpolation method to use; "linear" (default) or "nearest".
            Only nearest is available when specifying multi-dimensional
            coordinates.

        """
        sample_points = self._get_interp_points()
        interpolated_cube = interpolate(cube, sample_points, method=method)
        # Add an "index" coord to name the anonymous dimension produced by
        # the interpolation, if present.
        if len(interpolated_cube.dim_coords) < interpolated_cube.ndim:
            # Add a new coord `index` to describe the new dimension created by
            # interpolating.
            index_coord = iris.coords.DimCoord(
                range(self.sample_count), long_name="index"
            )
            # Make sure anonymous dims in `cube` do not mistakenly get labelled
            # as the new `index` dimension created by interpolating.
            src_anon_dims = self._src_cube_anon_dims(cube)
            interp_anon_dims = self._src_cube_anon_dims(interpolated_cube)
            (anon_dim_index,) = list(set(interp_anon_dims) - set(src_anon_dims))
            # Add the new coord to the interpolated cube.
            interpolated_cube.add_dim_coord(index_coord, anon_dim_index)
        return interpolated_cube


def interpolate(cube, sample_points, method=None):
    """Extract a sub-cube at the given n-dimensional points.

    Parameters
    ----------
    cube :
        The source Cube.
    sample_points :
        A sequence of coordinate (name) - values pairs.
    method : optional, default=None
        Request "linear" interpolation (default) or "nearest" neighbour.
        Only nearest neighbour is available when specifying multi-dimensional
        coordinates.

    Examples
    --------
    ::

        sample_points = [('latitude', [45, 45, 45]),
        ('longitude', [-60, -50, -40])]
        interpolated_cube = interpolate(cube, sample_points)

    Notes
    -----
    This function does not maintain laziness when called; it realises data.
    See more at :doc:`/userguide/real_and_lazy_data`.
    """
    from iris.analysis import Linear

    if method not in [None, "linear", "nearest"]:
        raise ValueError("Unhandled interpolation specified : %s" % method)

    # Convert any coordinate names to coords
    points = []
    for coord, values in sample_points:
        if isinstance(coord, str):
            coord = cube.coord(coord)
        points.append((coord, values))
    sample_points = points

    # Do all value sequences have the same number of values?
    coord, values = sample_points[0]
    trajectory_size = len(values)
    for coord, values in sample_points[1:]:
        if len(values) != trajectory_size:
            raise ValueError("Lengths of coordinate values are inconsistent.")

    # Which dimensions are we squishing into the last dimension?
    squish_my_dims = set()
    for coord, values in sample_points:
        dims = cube.coord_dims(coord)
        for dim in dims:
            squish_my_dims.add(dim)

    # Derive the new cube's shape by filtering out all the dimensions we're
    # about to sample,
    # and then adding a new dimension to accommodate all the sample points.
    remaining = [
        (dim, size) for dim, size in enumerate(cube.shape) if dim not in squish_my_dims
    ]
    new_data_shape = [size for dim, size in remaining]
    new_data_shape.append(trajectory_size)

    # Start with empty data and then fill in the "column" of values for each
    # trajectory point.
    new_cube = iris.cube.Cube(np.empty(new_data_shape))
    new_cube.metadata = cube.metadata

    # Derive the mapping from the non-trajectory source dimensions to their
    # corresponding destination dimensions.
    remaining_dims = [dim for dim, size in remaining]
    dimension_remap = {dim: i for i, dim in enumerate(remaining_dims)}

    # Record a mapping from old coordinate IDs to new coordinates,
    # for subsequent use in creating updated aux_factories.
    coord_mapping = {}

    # Create all the non-squished coords
    for coord in cube.dim_coords:
        src_dims = cube.coord_dims(coord)
        if squish_my_dims.isdisjoint(src_dims):
            dest_dims = [dimension_remap[dim] for dim in src_dims]
            new_coord = coord.copy()
            new_cube.add_dim_coord(new_coord, dest_dims)
            coord_mapping[id(coord)] = new_coord
    for coord in cube.aux_coords:
        src_dims = cube.coord_dims(coord)
        if squish_my_dims.isdisjoint(src_dims):
            dest_dims = [dimension_remap[dim] for dim in src_dims]
            new_coord = coord.copy()
            new_cube.add_aux_coord(new_coord, dest_dims)
            coord_mapping[id(coord)] = new_coord

    # Create all the squished (non derived) coords, not filled in yet.
    trajectory_dim = len(remaining_dims)
    for coord in cube.dim_coords + cube.aux_coords:
        src_dims = cube.coord_dims(coord)
        if not squish_my_dims.isdisjoint(src_dims):
            points = np.array([coord.points.flatten()[0]] * trajectory_size)
            new_coord = iris.coords.AuxCoord(
                points,
                var_name=coord.var_name,
                standard_name=coord.standard_name,
                long_name=coord.long_name,
                units=coord.units,
                bounds=None,
                attributes=coord.attributes,
                coord_system=coord.coord_system,
            )
            new_cube.add_aux_coord(new_coord, trajectory_dim)
            coord_mapping[id(coord)] = new_coord

    for factory in cube.aux_factories:
        new_cube.add_aux_factory(factory.updated(coord_mapping))

    # Are the given coords all 1-dimensional? (can we do linear interp?)
    for coord, values in sample_points:
        if coord.ndim > 1:
            if method == "linear":
                msg = (
                    "Cannot currently perform linear interpolation for "
                    "multi-dimensional coordinates."
                )
                raise iris.exceptions.CoordinateMultiDimError(msg)
            method = "nearest"
            break

    if method in ["linear", None]:
        # Using cube.interpolate will generate extra values that we don't need
        # as it makes a grid from the provided coordinates (like a meshgrid)
        # and then does interpolation for all of them. This is memory
        # inefficient, but significantly more time efficient than calling
        # cube.interpolate (or the underlying method on the interpolator)
        # repeatedly, so using this approach for now. In future, it would be
        # ideal if we only interpolated at the points we care about
        columns = cube.interpolate(sample_points, Linear())
        # np.einsum(a, [0, 0], [0]) is like np.diag(a)
        # We're using einsum here to do an n-dimensional diagonal, leaving the
        # other dimensions unaffected and putting the diagonal's direction on
        # the final axis
        initial_inds = list(range(1, columns.ndim + 1))
        for ind in squish_my_dims:
            initial_inds[ind] = 0
        final_inds = list(filter(lambda x: x != 0, initial_inds)) + [0]
        new_cube.data = np.einsum(columns.data, initial_inds, final_inds)

        # Fill in the empty squashed (non derived) coords.
        # We're using the same einstein summation plan as for the cube, but
        # redoing those indices to match the indices in the coordinates
        for columns_coord in columns.dim_coords + columns.aux_coords:
            src_dims = cube.coord_dims(columns_coord)
            if not squish_my_dims.isdisjoint(src_dims):
                # Mapping the cube indices onto the coord
                initial_coord_inds = [initial_inds[ind] for ind in src_dims]
                # Making the final ones the same way as for the cube
                # 0 will always appear in the initial ones because we know this
                # coord overlaps the squish dims
                final_coord_inds = list(
                    filter(lambda x: x != 0, initial_coord_inds)
                ) + [0]
                new_coord_points = np.einsum(
                    columns_coord.points, initial_coord_inds, final_coord_inds
                )
                # Check we're not overwriting coord.points with the wrong shape
                if (
                    not new_cube.coord(columns_coord.name()).points.shape
                    == new_coord_points.shape
                ):
                    msg = (
                        "Coord {} was expected to have new points of shape {}. "
                        "Found shape of {}."
                    )
                    raise ValueError(
                        msg.format(
                            columns_coord.name(),
                            new_cube.coord(columns_coord.name()).points.shape,
                            new_coord_points.shape,
                        )
                    )
                # Replace the points
                new_cube.coord(columns_coord.name()).points = new_coord_points

    elif method == "nearest":
        # Use a cache with _nearest_neighbour_indices_ndcoords()
        cache = {}
        column_indexes = _nearest_neighbour_indices_ndcoords(
            cube, sample_points, cache=cache
        )

        # Construct "fancy" indexes, so we can create the result data array in
        # a single numpy indexing operation.
        # ALSO: capture the index range in each dimension, so that we can fetch
        # only a required (square) sub-region of the source data.
        fancy_source_indices = []
        region_slices = []
        n_index_length = len(column_indexes[0])
        dims_reduced = [False] * n_index_length
        for i_ind in range(n_index_length):
            contents = [column_index[i_ind] for column_index in column_indexes]
            each_used = [content != slice(None) for content in contents]
            if np.all(each_used):
                # This dimension is addressed : use a list of indices.
                dims_reduced[i_ind] = True
                # Select the region by min+max indices.
                start_ind = np.min(contents)
                stop_ind = 1 + np.max(contents)
                region_slice = slice(start_ind, stop_ind)
                # Record point indices with start subtracted from all of them.
                fancy_index = list(np.array(contents) - start_ind)
            elif not np.any(each_used):
                # This dimension is not addressed by the operation.
                # Use a ":" as the index.
                fancy_index = slice(None)
                # No sub-region selection for this dimension.
                region_slice = slice(None)
            else:
                # Should really never happen, if _ndcoords is right.
                msg = (
                    "Internal error in trajectory interpolation : point "
                    "selection indices should all have the same form."
                )
                raise ValueError(msg)

            fancy_source_indices.append(fancy_index)
            region_slices.append(region_slice)

        # Fetch the required (square-section) region of the source data.
        # NOTE: This is not quite as good as only fetching the individual
        # points used, but it avoids creating a sub-cube for each point,
        # which is very slow, especially when points are re-used a lot ...
        source_area_indices = tuple(region_slices)
        source_data = cube[source_area_indices].data

        # Transpose source data before indexing it to get the final result.
        # Because.. the fancy indexing will replace the indexed (horizontal)
        # dimensions with a new single dimension over trajectory points.
        # Move those dimensions to the end *first* : this ensures that the new
        # dimension also appears at the end, which is where we want it.
        # Make a list of dims with the reduced ones last.
        dims_reduced = np.array(dims_reduced)
        dims_order = np.arange(n_index_length)
        dims_order = np.concatenate(
            (dims_order[~dims_reduced], dims_order[dims_reduced])
        )
        # Rearrange the data dimensions and the fancy indices into that order.
        source_data = source_data.transpose(dims_order)
        fancy_source_indices = [fancy_source_indices[i_dim] for i_dim in dims_order]

        # Apply the fancy indexing to get all the result data points.
        new_cube.data = source_data[tuple(fancy_source_indices)]

        # Fill in the empty squashed (non derived) coords.
        column_coords = [
            coord
            for coord in cube.dim_coords + cube.aux_coords
            if not squish_my_dims.isdisjoint(cube.coord_dims(coord))
        ]
        new_cube_coords = [
            new_cube.coord(column_coord.name()) for column_coord in column_coords
        ]
        all_point_indices = np.array(column_indexes)
        single_point_test_cube = cube[column_indexes[0]]
        for new_cube_coord, src_coord in zip(new_cube_coords, column_coords):
            # Check structure of the indexed coord (at one selected point).
            point_coord = single_point_test_cube.coord(src_coord)
            if len(point_coord.points) != 1:
                msg = (
                    "Coord {} at one x-y position has the shape {}, "
                    "instead of being a single point. "
                )
                raise ValueError(msg.format(src_coord.name(), src_coord.shape))

            # Work out which indices apply to the input coord.
            # NOTE: we know how to index the source cube to get a cube with a
            # single point for each coord, but this is very inefficient.
            # So here, we translate cube indexes into *coord* indexes.
            src_coord_dims = cube.coord_dims(src_coord)
            fancy_coord_index_arrays = [
                list(all_point_indices[:, src_dim]) for src_dim in src_coord_dims
            ]

            # Fill the new coord with all the correct points from the old one.
            new_cube_coord.points = src_coord.points[tuple(fancy_coord_index_arrays)]
            # NOTE: the new coords do *not* have bounds.

    return new_cube


def _ll_to_cart(lon, lat):
    # Based on cartopy.img_transform.ll_to_cart().
    x = np.sin(np.deg2rad(90 - lat)) * np.cos(np.deg2rad(lon))
    y = np.sin(np.deg2rad(90 - lat)) * np.sin(np.deg2rad(lon))
    z = np.cos(np.deg2rad(90 - lat))
    return (x, y, z)


def _cartesian_sample_points(sample_points, sample_point_coord_names):
    """Replace geographic lat/lon with cartesian xyz.
    Generates coords suitable for nearest point calculations with
    `scipy.spatial.cKDTree`.

    Parameters
    ----------
    sample_points :
        [coord][datum] list of sample_positions for each datum, formatted for
        fast use of :func:`_ll_to_cart()`.
    sample_point_coord_names :
        [coord] list of n coord names

    Returns
    -------
    list of [x,y,z,t,etc] positions, formatted for kdtree.

    """
    # Find lat and lon coord indices
    i_lat = i_lon = None
    i_non_latlon = list(range(len(sample_point_coord_names)))
    for i, name in enumerate(sample_point_coord_names):
        if "latitude" in name:
            i_lat = i
            i_non_latlon.remove(i_lat)
        if "longitude" in name:
            i_lon = i
            i_non_latlon.remove(i_lon)

    if i_lat is None or i_lon is None:
        return sample_points.transpose()

    num_points = len(sample_points[0])
    cartesian_points = [None] * num_points

    # Get the point coordinates without the latlon
    for p in range(num_points):
        cartesian_points[p] = [sample_points[c][p] for c in i_non_latlon]

    # Add cartesian xyz coordinates from latlon
    x, y, z = _ll_to_cart(sample_points[i_lon], sample_points[i_lat])
    for p in range(num_points):
        cartesian_point = cartesian_points[p]
        cartesian_point.append(x[p])
        cartesian_point.append(y[p])
        cartesian_point.append(z[p])

    return cartesian_points


def _nearest_neighbour_indices_ndcoords(cube, sample_points, cache=None):
    """Returns the indices to select the data value(s) closest to the given
    coordinate point values.

    'sample_points' is of the form [[coord-or-coord-name, point-value(s)]*].
    The lengths of all the point-values sequences must be equal.

    This function is adapted for points sampling a multi-dimensional coord,
    and can currently only do nearest neighbour interpolation.

    Because this function can be slow for multidimensional coordinates,
    a 'cache' dictionary can be provided by the calling code.

    .. Note::

        If the points are longitudes/latitudes, these are handled correctly as
        points on the sphere, but the values must be in 'degrees'.

    Developer notes:
    A "sample space cube" is made which only has the coords and dims we are
    sampling on.
    We get the nearest neighbour using this sample space cube.

    """
    if sample_points:
        try:
            coord, value = sample_points[0]
        except (KeyError, ValueError):
            emsg = (
                "Sample points must be a list of "
                "(coordinate, value) pairs, got {!r}."
            )
            raise TypeError(emsg.format(sample_points))

    # Convert names to coords in sample_point and reformat sample point values
    # for use in `_cartesian_sample_points()`.
    coord_values = []
    sample_point_coords = []
    sample_point_coord_names = []
    ok_coord_ids = set(map(id, cube.dim_coords + cube.aux_coords))
    for coord, value in sample_points:
        coord = cube.coord(coord)
        if id(coord) not in ok_coord_ids:
            msg = (
                "Invalid sample coordinate {!r}: derived coordinates are"
                " not allowed.".format(coord.name())
            )
            raise ValueError(msg)
        sample_point_coords.append(coord)
        sample_point_coord_names.append(coord.name())
        value = np.array(value, ndmin=1)
        coord_values.append(value)

    coord_point_lens = np.array([len(value) for value in coord_values])
    if not np.all(coord_point_lens == coord_point_lens[0]):
        msg = "All coordinates must have the same number of sample points."
        raise ValueError(msg)

    coord_values = np.array(coord_values)

    # Which dims are we sampling?
    sample_dims = set()
    for coord in sample_point_coords:
        for dim in cube.coord_dims(coord):
            sample_dims.add(dim)
    sample_dims = sorted(list(sample_dims))

    # Extract a sub cube that lives in just the sampling space.
    sample_space_slice = [0] * cube.ndim
    for sample_dim in sample_dims:
        sample_space_slice[sample_dim] = slice(None, None)
    sample_space_slice = tuple(sample_space_slice)
    sample_space_cube = cube[sample_space_slice]

    # Just the sampling coords.
    for coord in sample_space_cube.coords():
        if not coord.name() in sample_point_coord_names:
            sample_space_cube.remove_coord(coord)

    # Order the sample point coords according to the sample space cube coords.
    sample_space_coord_names = [coord.name() for coord in sample_space_cube.coords()]
    new_order = [
        sample_space_coord_names.index(name) for name in sample_point_coord_names
    ]
    coord_values = np.array([coord_values[i] for i in new_order])
    sample_point_coord_names = [sample_point_coord_names[i] for i in new_order]

    sample_space_coords = sample_space_cube.dim_coords + sample_space_cube.aux_coords
    sample_space_coords_and_dims = [
        (coord, sample_space_cube.coord_dims(coord)) for coord in sample_space_coords
    ]

    if cache is not None and cube in cache:
        kdtree = cache[cube]
    else:
        # Create a "sample space position" for each
        # `datum.sample_space_data_positions[coord_index][datum_index]`.
        sample_space_data_positions = np.empty(
            (len(sample_space_coords_and_dims), sample_space_cube.data.size),
            dtype=float,
        )
        for d, ndi in enumerate(np.ndindex(sample_space_cube.data.shape)):
            for c, (coord, coord_dims) in enumerate(sample_space_coords_and_dims):
                # Index of this datum along this coordinate (could be n-D).
                if coord_dims:
                    keys = tuple(ndi[ind] for ind in coord_dims)
                else:
                    keys = slice(None, None)
                # Position of this datum along this coordinate.
                sample_space_data_positions[c][d] = coord.points[keys]

        # Convert to cartesian coordinates. Flatten for kdtree compatibility.
        cartesian_space_data_coords = _cartesian_sample_points(
            sample_space_data_positions, sample_point_coord_names
        )

        # Create a kdtree for the nearest-distance lookup to these 3d points.
        kdtree = cKDTree(cartesian_space_data_coords)
        # This can find the nearest datum point to any given target point,
        # which is the goal of this function.

    # Update cache.
    if cache is not None:
        cache[cube] = kdtree

    # Convert the sample points to cartesian (3d) coords.
    # If there is no latlon within the coordinate there will be no change.
    # Otherwise, geographic latlon is replaced with cartesian xyz.
    cartesian_sample_points = _cartesian_sample_points(
        coord_values, sample_point_coord_names
    )

    # Use kdtree to get the nearest sourcepoint index for each target point.
    _, datum_index_lists = kdtree.query(cartesian_sample_points)

    # Convert flat indices back into multidimensional sample-space indices.
    sample_space_dimension_indices = np.unravel_index(
        datum_index_lists, sample_space_cube.data.shape
    )
    # Convert this from "pointwise list of index arrays for each dimension",
    # to "list of cube indices for each point".
    sample_space_ndis = np.array(sample_space_dimension_indices).transpose()

    # For the returned result, we must convert these indices into the source
    # (sample-space) cube, to equivalent indices into the target 'cube'.

    # Make a result array: (cube.ndim * <index>), per sample point.
    n_points = coord_values.shape[-1]
    main_cube_slices = np.empty((n_points, cube.ndim), dtype=object)
    # Initialise so all unused indices are ":".
    main_cube_slices[:] = slice(None)

    # Move result indices according to the source (sample) and target (cube)
    # dimension mappings.
    for sample_coord, sample_coord_dims in sample_space_coords_and_dims:
        # Find the coord in the main cube
        main_coord = cube.coord(sample_coord.name())
        main_coord_dims = cube.coord_dims(main_coord)
        # Fill nearest-point data indices for each coord dimension.
        for sample_i, main_i in zip(sample_coord_dims, main_coord_dims):
            main_cube_slices[:, main_i] = sample_space_ndis[:, sample_i]

    # Return as a list of **tuples** : required for correct indexing usage.
    result = [tuple(inds) for inds in main_cube_slices]
    return result


class UnstructuredNearestNeigbourRegridder:
    """Encapsulate the operation of :meth:`iris.analysis.trajectory.interpolate`
    with given source and target grids.

    This is the type used by the :class:`~iris.analysis.UnstructuredNearest`
    regridding scheme.

    """

    # TODO: cache the necessary bits of the operation so reuse can actually
    # be more efficient.
    def __init__(self, src_cube, target_grid_cube):
        """A nearest-neighbour regridder to perform regridding from the source
        grid to the target grid.

        This can then be applied to any source data with the same structure as
        the original 'src_cube'.

        Parameters
        ----------
        src_cube : :class:`~iris.cube.Cube`
            The :class:`~iris.cube.Cube` defining the source grid.
            The X and Y coordinates can have any shape, but must be mapped over
            the same cube dimensions.
        target_grid_cube : :class:`~iris.cube.Cube`
            A :class:`~iris.cube.Cube`, whose X and Y coordinates specify a
            desired target grid.
            The X and Y coordinates must be one-dimensional dimension
            coordinates, mapped to different dimensions.
            All other cube components are ignored.

        Returns
        -------
        regridder (object)
            A callable object with the interface::

                result_cube = regridder(data)

            where `data` is a cube with the same grid as the original
            `src_cube`, that is to be regridded to the `target_grid_cube`.

        Notes
        -----
        .. Note::

            For latitude-longitude coordinates, the nearest-neighbour distances
            are computed on the sphere, otherwise flat Euclidean distances are
            used.

            The source and target X and Y coordinates must all have the same
            coordinate system, which may also be None.
            If any X and Y coordinates are latitudes or longitudes, they *all*
            must be.  Otherwise, the corresponding X and Y coordinates must
            have the same units in the source and grid cubes.

        """
        from iris.analysis._interpolation import snapshot_grid
        from iris.util import _meshgrid

        # Make a copy of the source cube, so we can convert coordinate units.
        src_cube = src_cube.copy()

        # Snapshot the target grid and check it is a "normal" grid.
        tgt_x_coord, tgt_y_coord = snapshot_grid(target_grid_cube)

        # Check that the source has unique X and Y coords over common dims.
        if not src_cube.coords(axis="x") or not src_cube.coords(axis="y"):
            msg = "Source cube must have X- and Y-axis coordinates."
            raise ValueError(msg)
        src_x_coord = src_cube.coord(axis="x")
        src_y_coord = src_cube.coord(axis="y")
        if src_cube.coord_dims(src_x_coord) != src_cube.coord_dims(src_y_coord):
            msg = "Source cube X and Y coordinates must have the same cube dimensions."
            raise ValueError(msg)

        # Record *copies* of the original grid coords, in the desired
        # dimension order.
        # This lets us convert the actual ones in use to units of "degrees".
        self.src_grid_coords = [src_y_coord.copy(), src_x_coord.copy()]
        self.tgt_grid_coords = [tgt_y_coord.copy(), tgt_x_coord.copy()]

        # Check that all XY coords have suitable coordinate systems and units.
        coords_all = [src_x_coord, src_y_coord, tgt_x_coord, tgt_y_coord]
        cs = coords_all[0].coord_system
        if not all(coord.coord_system == cs for coord in coords_all):
            msg = (
                "Source and target cube X and Y coordinates must all have "
                "the same coordinate system."
            )
            raise ValueError(msg)

        # Check *all* X and Y coords are lats+lons, if any are.
        latlons = [
            "latitude" in coord.name() or "longitude" in coord.name()
            for coord in coords_all
        ]
        if any(latlons) and not all(latlons):
            msg = (
                "If any X and Y coordinates are latitudes/longitudes, "
                "then they all must be."
            )
            raise ValueError(msg)

        self.grid_is_latlon = any(latlons)
        if self.grid_is_latlon:
            # Convert all XY coordinates to units of "degrees".
            # N.B. already copied the target grid, so the result matches that.
            for coord in coords_all:
                try:
                    coord.convert_units("degrees")
                except ValueError:
                    msg = (
                        "Coordinate {!r} has units of {!r}, which does not "
                        'convert to "degrees".'
                    )
                    raise ValueError(msg.format(coord.name(), str(coord.units)))
        else:
            # Check that source and target have the same X and Y units.
            if (
                src_x_coord.units != tgt_x_coord.units
                or src_y_coord.units != tgt_y_coord.units
            ):
                msg = (
                    "Source and target cube X and Y coordinates must "
                    "have the same units."
                )
                raise ValueError(msg)

        # Record the resulting grid shape.
        self.tgt_grid_shape = tgt_y_coord.shape + tgt_x_coord.shape

        # Calculate sample points as 2d arrays, like broadcast (NY,1)*(1,NX).
        x_2d, y_2d = _meshgrid(tgt_x_coord.points, tgt_y_coord.points)
        # Cast as a "trajectory", to suit the method used.
        self.trajectory = (
            (tgt_x_coord.name(), x_2d.flatten()),
            (tgt_y_coord.name(), y_2d.flatten()),
        )

    def __call__(self, src_cube):
        # Check the source cube X and Y coords match the original.
        # Note: for now, this is sufficient to ensure a valid trajectory
        # interpolation, but if in future we save and reuse the cache context
        # for the 'interpolate' call, we may need more checks here.

        # Check the given cube against the original.
        x_cos = src_cube.coords(axis="x")
        y_cos = src_cube.coords(axis="y")
        if (
            not x_cos
            or not y_cos
            or y_cos != [self.src_grid_coords[0]]
            or x_cos != [self.src_grid_coords[1]]
        ):
            msg = (
                "The given cube is not defined on the same source "
                "grid as this regridder."
            )
            raise ValueError(msg)

        # Convert source XY coordinates to degrees if required.
        if self.grid_is_latlon:
            src_cube = src_cube.copy()
            src_cube.coord(axis="x").convert_units("degrees")
            src_cube.coord(axis="y").convert_units("degrees")

        # Get the basic interpolated results.
        result_trajectory_cube = interpolate(
            src_cube, self.trajectory, method="nearest"
        )

        # Reconstruct this as a cube "like" the source data.
        # TODO: handle all aux-coords, cell measures ??

        # The shape is that of the basic result, minus the trajectory (last)
        # dimension, plus the target grid dimensions.
        target_shape = result_trajectory_cube.shape[:-1] + self.tgt_grid_shape
        data_2d_x_and_y = result_trajectory_cube.data.reshape(target_shape)

        # Make a new result cube with the reshaped data.
        result_cube = iris.cube.Cube(data_2d_x_and_y)
        result_cube.metadata = src_cube.metadata

        # Copy all the coords from the trajectory result.
        i_trajectory_dim = result_trajectory_cube.ndim - 1
        for coord in result_trajectory_cube.dim_coords:
            dims = result_trajectory_cube.coord_dims(coord)
            if i_trajectory_dim not in dims:
                result_cube.add_dim_coord(coord.copy(), dims)
        for coord in result_trajectory_cube.aux_coords:
            dims = result_trajectory_cube.coord_dims(coord)
            if i_trajectory_dim not in dims:
                result_cube.add_aux_coord(coord.copy(), dims)

        # Add the X+Y grid coords from the grid cube, mapped to the new Y and X
        # dimensions, i.e. the last 2.
        for i_dim, coord in enumerate(self.tgt_grid_coords):
            result_cube.add_dim_coord(coord.copy(), i_dim + i_trajectory_dim)

        return result_cube
