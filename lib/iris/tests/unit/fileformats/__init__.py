# (C) British Crown Copyright 2013 - 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the :mod:`iris.fileformats` package."""

import iris.tests as tests


class TestField(tests.IrisTest):
    def _test_for_coord(self, field, convert, coord_predicate, expected_points,
                        expected_bounds):
        (factories, references, standard_name, long_name, units,
         attributes, cell_methods, dim_coords_and_dims,
         aux_coords_and_dims) = convert(field)

        # Check for one and only one matching coordinate.
        coords_and_dims = dim_coords_and_dims + aux_coords_and_dims
        matching_coords = [coord for coord, _ in coords_and_dims if
                           coord_predicate(coord)]
        self.assertEqual(len(matching_coords), 1, str(matching_coords))
        coord = matching_coords[0]

        # Check points and bounds.
        if expected_points is not None:
            self.assertArrayEqual(coord.points, expected_points)

        if expected_bounds is None:
            self.assertIsNone(coord.bounds)
        else:
            self.assertArrayEqual(coord.bounds, expected_bounds)

    def assertCoordsAndDimsListsMatch(self, coords_and_dims_got,
                                      coords_and_dims_expected):
        """
        Check that coords_and_dims lists are equivalent.

        The arguments are lists of pairs of (coordinate, dimensions).
        The elements are compared one-to-one, by coordinate name (so the order
        of the lists is _not_ significant).

        """
        def sorted_by_coordname(list):
            return sorted(list, key=lambda item: item[0].name())

        coords_and_dims_got = sorted_by_coordname(coords_and_dims_got)
        coords_and_dims_expected = sorted_by_coordname(
            coords_and_dims_expected)
        self.assertEqual(coords_and_dims_got, coords_and_dims_expected)
