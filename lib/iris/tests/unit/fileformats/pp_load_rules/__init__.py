# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.fileformats.pp_load_rules` module."""


# a general utility function for PP field tests
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
