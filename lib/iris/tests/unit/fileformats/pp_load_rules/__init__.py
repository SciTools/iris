# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.fileformats.pp_load_rules` module."""

# general utility functions for PP field tests
from iris.tests._shared_utils import assert_array_equal


def assert_test_for_coord(
    field, convert, coord_predicate, expected_points, expected_bounds
):
    (
        factories,
        references,
        standard_name,
        long_name,
        units,
        attributes,
        cell_methods,
        dim_coords_and_dims,
        aux_coords_and_dims,
    ) = convert(field)

    # Check for one and only one matching coordinate.
    coords_and_dims = dim_coords_and_dims + aux_coords_and_dims
    matching_coords = [coord for coord, _ in coords_and_dims if coord_predicate(coord)]
    assert len(matching_coords) == 1, str(matching_coords)
    coord = matching_coords[0]

    # Check points and bounds.
    if expected_points is not None:
        assert_array_equal(coord.points, expected_points)

    if expected_bounds is None:
        assert coord.bounds is None
    else:
        assert_array_equal(coord.bounds, expected_bounds)


def assert_coords_and_dims_lists_match(coords_and_dims_got, coords_and_dims_expected):
    """Check that coords_and_dims lists are equivalent.

    The arguments are lists of pairs of (coordinate, dimensions).
    The elements are compared one-to-one, by coordinate name (so the order
    of the lists is _not_ significant).
    It also checks that the coordinate types (DimCoord/AuxCoord) match.

    """

    def sorted_by_coordname(list):
        return sorted(list, key=lambda item: item[0].name())

    coords_and_dims_got = sorted_by_coordname(coords_and_dims_got)
    coords_and_dims_expected = sorted_by_coordname(coords_and_dims_expected)
    assert coords_and_dims_got == coords_and_dims_expected
    # Also check coordinate type equivalences (as Coord.__eq__ does not).
    assert [type(coord) for coord, dims in coords_and_dims_got] == [
        type(coord) for coord, dims in coords_and_dims_expected
    ]
