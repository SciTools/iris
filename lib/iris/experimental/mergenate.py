# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Combined merge and concatenate of a cubelist over the given coordinate(s) or on
a new anonymous leading dimension.

"""

import dask.array as da
import numpy as np

from iris.cube import Cube
import iris.exceptions

COORD_TYPE = 0
ANC_VAR_TYPE = 1
CELL_MEAS_TYPE = 2
AUX_FACT_TYPE = 3

dim_func = {
    COORD_TYPE: Cube.coord_dims,
    ANC_VAR_TYPE: Cube.ancillary_variable_dims,
    CELL_MEAS_TYPE: Cube.cell_measure_dims,
}


def _get_all_underived_vars(cube, give_types=False):
    coords = cube.coords()
    # Filter derived coordinates because they'll get recreated
    coords = [coord for coord in coords if coord not in cube.derived_coords]

    anc_vars = cube.ancillary_variables()
    cm_vars = cube.cell_measures()
    af_vars = list(cube.aux_factories)

    vars = coords + anc_vars + cm_vars + af_vars

    if not give_types:
        return vars
    else:
        types = (
            [COORD_TYPE] * len(coords)
            + [ANC_VAR_TYPE] * len(anc_vars)
            + [CELL_MEAS_TYPE] * len(cm_vars)
            + [AUX_FACT_TYPE] * len(af_vars)
        )
        return vars, types


def _make_coord_table(cubes):

    # Build our prototypes from the first cube

    cube_0_vars, var_types = _get_all_underived_vars(cubes[0], give_types=True)

    coord_table = [cube_0_vars]

    coord_count = len(coord_table[0])

    # Stick everything else in
    for cube_ind, cube in enumerate(cubes[1:]):
        coord_table.append([None] * coord_count)

        for cube_coord in _get_all_underived_vars(cube):
            for ii, prototype_coord in enumerate(coord_table[0]):
                if type(cube_coord) == type(
                    prototype_coord
                ) and cube_coord.metadata.equal(prototype_coord.metadata):
                    coord_table[-1][ii] = cube_coord
                    break
            else:
                # Error if we haven't used any of our coords in this cube
                raise iris.exceptions.MergeError(
                    (
                        f"{cube_coord.name()} in cube {cube_ind+1} doesn't match any coord in cube 0",
                    )
                )

        missing_coord_inds = [
            ii for ii in range(coord_count) if coord_table[-1][ii] is None
        ]
        if missing_coord_inds:
            error_messages = [
                f"{coord_table[0][ind].name()} in cube 0 doesn't match any coord in cube {cube_ind+1}"
                for ind in missing_coord_inds
            ]
            raise iris.exceptions.MergeError(error_messages)

    return np.array(coord_table), var_types


def _merge_metadata(objs, err_name="objects"):
    # TODO: Check with Bill that I'm doing this right

    new_metadata = objs[0].metadata
    for obj in objs:
        if not new_metadata.equal(obj.metadata):
            raise iris.exceptions.MergeError(
                (
                    f"Inconsistent metadata between merging {err_name}.\nDifference is {new_metadata.difference(obj.metadata)}",
                )
            )
        new_metadata = new_metadata.combine(obj.metadata)

    return new_metadata


def _build_new_coord_pieces(coords, new_dims, cube_depths, extend_coords):
    prototype_coord = coords[0]

    if np.all(coords[1:] == coords[:-1]):
        # All coordinates are identical
        new_points = prototype_coord.points
        new_bounds = prototype_coord.bounds

    elif 0 in new_dims:
        # Coordinate already spans merge dimension
        new_points = []
        has_bounds = coords[0].has_bounds()
        if has_bounds:
            new_bounds = []
        else:
            new_bounds = None

        for coord in coords:
            new_points.append(coord.points)

            if has_bounds:
                new_bounds.append(coord.bounds)

        new_points = np.concatenate(new_points, axis=0)
        if has_bounds:
            new_bounds = np.concatenate(new_bounds, axis=0)

    elif not extend_coords:
        # The coordinate doesn't span the merge dimension and we shouldn't make
        # it
        raise iris.exceptions.MergeError(
            (
                "Different points or bounds in "
                f"{coords[0].name()} coords, "
                "but not allowed to extend coords. Consider trying "
                "again with extend_coords=True",
            )
        )

    else:
        # We need to broadcast the coordinates to span the merge dimension
        new_dims = (0,) + new_dims
        new_points = []
        has_bounds = coords[0].has_bounds()
        if has_bounds:
            new_bounds = []
        else:
            new_bounds = None

        for cube_ind, coord in enumerate(coords):
            cube_depth = cube_depths[cube_ind]

            # If the coord was scalar, it will become 1D
            if new_dims == (0,):
                broadcast_shape = [cube_depth]
            else:
                broadcast_shape = [cube_depth] + list(coord.points.shape)

            new_coord_points = np.broadcast_to(
                coord.points, broadcast_shape, subok=True
            )
            new_points.append(new_coord_points)

            if has_bounds:

                broadcast_shape = [cube_depth] + list(coord.bounds.shape)

                new_coord_bounds = np.broadcast_to(
                    coord.bounds, broadcast_shape, subok=True
                )

                new_bounds.append(new_coord_bounds)

        new_points = np.concatenate(new_points, axis=0)
        if has_bounds:
            new_bounds = np.concatenate(new_bounds, axis=0)

    return new_points, new_bounds, new_dims


def _build_new_var_pieces(coords, new_dims, cube_depths, extend_coords):
    prototype_coord = coords[0]

    if np.all(coords[1:] == coords[:-1]):
        # All coordinates are identical
        new_points = prototype_coord.data

    elif 0 in new_dims:
        # Coordinate already spans merge dimension
        new_points = []

        for coord in coords:
            new_points.append(coord.data)

        new_points = np.concatenate(new_points, axis=0)

    elif not extend_coords:
        # The coordinate doesn't span the merge dimension and we shouldn't make
        # it
        raise iris.exceptions.MergeError(
            (
                "Different points or bounds in "
                f"{coords[0].name()} coords, "
                "but not allowed to extend coords. Consider trying "
                "again with extend_coords=True",
            )
        )

    else:
        # We need to broadcast the coordinates to span the merge dimension
        new_dims = (0,) + new_dims
        new_points = []

        for cube_ind, coord in enumerate(coords):
            cube_depth = cube_depths[cube_ind]

            # If the coord was scalar, it will become 1D
            if new_dims == (0,):
                broadcast_shape = [cube_depth]
            else:
                broadcast_shape = [cube_depth] + list(coord.data.shape)

            new_coord_points = np.broadcast_to(
                coord.data, broadcast_shape, subok=True
            )
            new_points.append(new_coord_points)

        new_points = np.concatenate(new_points, axis=0)

    return new_points, new_dims


def aux_factories_equal(aux_a, aux_b):
    if type(aux_a) != type(aux_b):
        return False
    if aux_a.metadata != aux_b.metadata:
        return False
    print(aux_a.metadata)
    print()
    print(aux_b.metadata)
    dependencies_a = {
        key: coord.metadata for key, coord in aux_a.dependencies.items()
    }
    dependencies_b = {
        key: coord.metadata for key, coord in aux_b.dependencies.items()
    }
    if len(dependencies_a) != len(dependencies_b):
        return False
    try:
        for key, metadata_a in dependencies_a.items():
            if metadata_a != dependencies_b[key]:
                return False
    except (TypeError, KeyError):
        return False
    return True


def _mergenate(cubes, extend_coords: bool, merge_coord=None):
    """Actually mergenate (on axis 0)"""

    try:
        new_data = da.concatenate([cube.core_data() for cube in cubes], axis=0)
    except ValueError:
        msg = "Failed to concatenate cube datas"
        raise iris.exceptions.MergeError((msg,))

    coord_table, coord_types = _make_coord_table(cubes)
    cube_metadata = _merge_metadata(cubes, err_name="cubes")
    cube_depths = [cube.shape[0] for cube in cubes]

    # Make the new coords
    new_coords_and_dims = []
    current_aux_factories = []
    for coord_col, coord_type in zip(coord_table.T, coord_types):

        if coord_type == AUX_FACT_TYPE:
            all_equal = True
            for ii in range(len(coord_col) - 1):
                aux_a = coord_col[ii]
                aux_b = coord_col[ii + 1]
                if not aux_factories_equal(aux_a, aux_b):
                    all_equal = False
                    break

            if not all_equal:
                raise iris.exceptions.MergeError(
                    ("Inconsistent AuxCoordFactories across cubes",)
                )
            current_aux_factories.append(coord_col[0])
            continue

        new_dims = dim_func[coord_type](cubes[0], coord_col[0])
        if coord_type == COORD_TYPE:
            new_points, new_bounds, new_dims = _build_new_coord_pieces(
                coord_col, new_dims, cube_depths, extend_coords
            )
        else:
            new_points, new_dims = _build_new_var_pieces(
                coord_col, new_dims, cube_depths, extend_coords
            )
            new_bounds = None

        new_coord = None
        if isinstance(coord_col[0], iris.coords.DimCoord):
            try:
                new_coord = iris.coords.DimCoord(new_points, bounds=new_bounds)
            except ValueError:
                new_coord = iris.coords.AuxCoord(new_points, bounds=new_bounds)
        if new_coord is None:
            if coord_type == COORD_TYPE:
                new_coord = coord_col[0].copy(
                    points=new_points, bounds=new_bounds
                )
            else:
                new_coord = coord_col[0].copy(values=new_points)

        new_coord.metadata = _merge_metadata(coord_col, err_name="coords")

        new_coords_and_dims.append((new_coord, new_dims))

    # Type isn't a sufficient indicator of which coords were originally the
    # cube's dim coords, so check explicitly
    original_dim_coords = cubes[0].coords(dim_coords=True)
    if merge_coord is not None:
        original_dim_coords.append(cubes[0].coord(merge_coord))
    dim_coord_indices = []
    for ii, coord in enumerate(coord_table[0]):
        if coord in original_dim_coords:
            dim_coord_indices.append(ii)

    new_dim_coords_and_dims = []
    new_aux_coords_and_dims = []
    new_cell_measures_and_dims = []
    new_ancillary_variables_and_dims = []
    # Order is important here - a CellMeasure is an AncillaryVariable so
    # check for instance of CellMeasure first
    type_lookup = [
        (
            (iris.coords.DimCoord, iris.coords.AuxCoord),
            new_aux_coords_and_dims,
        ),
        (iris.coords.CellMeasure, new_cell_measures_and_dims),
        (iris.coords.AncillaryVariable, new_ancillary_variables_and_dims),
    ]
    for coord_ind, (new_coord, new_dims) in enumerate(new_coords_and_dims):
        if (coord_ind in dim_coord_indices) and isinstance(
            new_coord, iris.coords.DimCoord
        ):
            new_dim_coords_and_dims.append(
                (
                    new_coord,
                    new_dims,
                )
            )
            continue
        for coord_type, coord_list in type_lookup:
            if isinstance(new_coord, coord_type):
                coord_list.append(
                    (
                        new_coord,
                        new_dims,
                    )
                )
                break
        else:
            raise iris.exceptions.MergeError(
                (f"Aux coord of no known type: {new_coord}",)
            )

    merged_cube = Cube(
        new_data,
        dim_coords_and_dims=new_dim_coords_and_dims,
        aux_coords_and_dims=new_aux_coords_and_dims,
        cell_measures_and_dims=new_cell_measures_and_dims,
        ancillary_variables_and_dims=new_ancillary_variables_and_dims,
    )

    # Deal with aux factories now everything else is set up
    nonderived_coords = cubes[0].dim_coords + cubes[0].aux_coords
    coord_mapping = {
        id(old_co): merged_cube.coord(old_co) for old_co in nonderived_coords
    }
    for factory in cubes[0].aux_factories:
        new_factory = factory.updated(coord_mapping)
        merged_cube.add_aux_factory(new_factory)

    merged_cube.metadata = cube_metadata

    return merged_cube


def _sort_cubes(cubes, coord):

    merge_coords_points = [cube.coord(coord).points for cube in cubes]

    # Check if the coords are ascending or descending
    ascending = None
    for ii, merge_coord_points in enumerate(merge_coords_points):

        if len(merge_coord_points) == 1:
            pass
        elif ascending is None:
            ascending = merge_coord_points[1] > merge_coord_points[0]
        elif ascending != (merge_coord_points[1] > merge_coord_points[0]):
            raise iris.exceptions.MergeError(
                ("Mixture of ascending and descending coordinate points",)
            )

    if ascending is None:
        ascending = True

    # The order the the cubes should be uesd is based on the first point in each
    # of their points arrays. This sorting is reversed if we've established all
    # coords are descending
    cube_order = sorted(
        [*range(len(cubes))],
        key=lambda x: merge_coords_points[x][0],
        reverse=not ascending,
    )

    # Check that our new coordinate will be monotonic
    for ii in range(len(cubes) - 1):
        if (
            merge_coords_points[cube_order[ii]][-1]
            < merge_coords_points[cube_order[ii + 1]][0]
        ) != ascending:
            # Question: Here we prevent people from effectively inserting a
            # coord within another one - we could allow that, but it seems
            # more confusing not to error than it is inconvenient to error.
            # Is that sensible?
            raise iris.exceptions.MergeError(
                (
                    "Coordinate points overlap so correct merge order is ambiguous",
                )
            )

    return [cubes[ii] for ii in cube_order]


def _calculate_reorders(cubes, coord):
    """Work out how cube axis orders should change for mergenate"""
    max_ndim = max([cube.ndim for cube in cubes])

    coord_dims = set([cube.coord_dims(coord) for cube in cubes])

    coord_dims.discard(())

    if not coord_dims:
        base_order = [*range(max_ndim + 1)]
        return base_order, base_order

    try:
        (coord_dim_tuple,) = coord_dims
    except ValueError:
        raise iris.exceptions.MergeError(
            ("Coord lies on different axes on different cubes",)
        )

    try:
        (coord_dim,) = coord_dim_tuple
    except ValueError:
        raise iris.exceptions.MergeError(
            ("Can only merge on 1D or 0D coordinates",)
        )

    forward_order = [*range(max_ndim)]
    forward_order.remove(coord_dim)
    forward_order = tuple([coord_dim] + forward_order)

    backward_order = [*range(1, max_ndim)]
    backward_order.insert(coord_dim, 0)
    backward_order = tuple(backward_order)

    return forward_order, backward_order


def _categorise_by_coord(cubes, coord):
    category_dict = {}
    for cube in cubes:
        match_coord = cube.coord(coord)
        for seen_coord, categorised_cubes in category_dict.items():
            if np.all(match_coord.points == seen_coord.points) and np.all(
                match_coord.bounds == seen_coord.bounds
            ):
                categorised_cubes.append(cube)
                break
        else:
            category_dict[match_coord] = [cube]
    return [*category_dict.values()]


def _validate_shapes(cubelist, coord):

    check_shapes = []
    for cube in cubelist:
        if coord is None:
            check_shapes.append(cube.shape)
        else:
            coord_dims = cube.coord_dims(coord)
            if coord_dims == ():
                check_shapes.append(cube.shape)
            else:
                try:
                    (coord_dim,) = coord_dims
                except ValueError:
                    msg = "Can't merge on a 2D coordinate"
                    raise iris.exceptions.MergeError((msg,))
                cube_shape = [*cube.shape]
                cube_shape.pop(coord_dim)
                check_shapes.append((*cube_shape,))

    expected_shape = check_shapes[0]
    for check_shape in check_shapes:
        if check_shape != expected_shape:
            msg = "The shapes of cubes to be concatenated can only differ on the affected dimensions"
            raise iris.exceptions.MergeError((msg,))


def mergenate(cubelist, coords=None, extend_coords=False):
    # Order is only guaranteed if coords are monotonic and all ascending or all descending

    if coords is None:
        _validate_shapes(cubelist, None)
        prepared_cubes = []
        for cube in cubelist:
            prepared_cubes.append(iris.util.new_axis(cube))
        out_cube = _mergenate(prepared_cubes, extend_coords=extend_coords)
        return out_cube

    elif (
        isinstance(coords, (iris.coords._DimensionalMetadata, str))
        or len(coords) == 1
    ):
        try:
            (coord,) = coords
        except ValueError:
            coord = coords

        _validate_shapes(cubelist, coord)

        prepared_cubes = []
        forward_reorder, backward_reorder = _calculate_reorders(
            cubelist, coord
        )
        for cube in cubelist:
            coord_dims = cube.coord_dims(coord)
            if coord_dims == ():
                prepared_cubes.append(
                    iris.util.new_axis(cube, scalar_coord=coord)
                )
            else:
                cube_copy = cube.copy()
                cube_copy.transpose(forward_reorder)
                prepared_cubes.append(cube_copy)
        prepared_cubes = _sort_cubes(prepared_cubes, coord)
        out_cube = _mergenate(
            prepared_cubes, extend_coords=extend_coords, merge_coord=coord
        )
        out_cube.transpose(backward_reorder)
        return out_cube

    else:

        prepared_cubes = []
        categorisation = _categorise_by_coord(cubelist, coords[-1])
        for cubelist in categorisation:
            prepared_cubes.append(
                mergenate(cubelist, coords[:-1], extend_coords=extend_coords)
            )
        return mergenate(
            prepared_cubes, coords[-1], extend_coords=extend_coords
        )
