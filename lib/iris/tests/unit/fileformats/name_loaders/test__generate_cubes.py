# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :func:`iris.analysis.name_loaders._generate_cubes`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.fileformats.name_loaders import NAMECoord, _generate_cubes


class TestCellMethods(tests.IrisTest):
    def test_cell_methods(self):
        header = mock.MagicMock()
        column_headings = {
            "Species": [1, 2, 3],
            "Quantity": [4, 5, 6],
            "Units": ["m", "m", "m"],
            "Z": [1, 2, 3],
        }
        coords = mock.MagicMock()
        data_arrays = [mock.Mock(), mock.Mock()]
        cell_methods = ["cell_method_1", "cell_method_2"]

        self.patch("iris.fileformats.name_loaders._cf_height_from_name")
        self.patch("iris.cube.Cube")
        cubes = list(
            _generate_cubes(
                header, column_headings, coords, data_arrays, cell_methods
            )
        )

        cubes[0].assert_has_calls([mock.call.add_cell_method("cell_method_1")])
        cubes[1].assert_has_calls([mock.call.add_cell_method("cell_method_2")])


class TestCircularLongitudes(tests.IrisTest):
    def _simulate_with_coords(self, names, values, dimensions):
        header = mock.MagicMock()
        column_headings = {
            "Species": [1, 2, 3],
            "Quantity": [4, 5, 6],
            "Units": ["m", "m", "m"],
            "Z": [1, 2, 3],
        }
        coords = [
            NAMECoord(name, dim, vals)
            for name, vals, dim in zip(names, values, dimensions)
        ]
        data_arrays = [mock.Mock()]

        self.patch("iris.fileformats.name_loaders._cf_height_from_name")
        self.patch("iris.cube.Cube")
        cubes = list(
            _generate_cubes(header, column_headings, coords, data_arrays)
        )
        return cubes

    def test_non_circular(self):
        results = self._simulate_with_coords(
            names=["longitude"], values=[[1, 7, 23]], dimensions=[(0,)]
        )
        self.assertEqual(len(results), 1)
        add_coord_calls = results[0].add_dim_coord.call_args_list
        self.assertEqual(len(add_coord_calls), 1)
        coord = add_coord_calls[0][0][0]
        self.assertEqual(coord.circular, False)

    def test_circular(self):
        results = self._simulate_with_coords(
            names=["longitude"],
            values=[[5.0, 95.0, 185.0, 275.0]],
            dimensions=[(0,)],
        )
        self.assertEqual(len(results), 1)
        add_coord_calls = results[0].add_dim_coord.call_args_list
        self.assertEqual(len(add_coord_calls), 1)
        coord = add_coord_calls[0][0][0]
        self.assertEqual(coord.circular, True)

    def test_lat_lon_byname(self):
        results = self._simulate_with_coords(
            names=["longitude", "latitude"],
            values=[[5.0, 95.0, 185.0, 275.0], [5.0, 95.0, 185.0, 275.0]],
            dimensions=[(0,), (1,)],
        )
        self.assertEqual(len(results), 1)
        add_coord_calls = results[0].add_dim_coord.call_args_list
        self.assertEqual(len(add_coord_calls), 2)
        lon_coord = add_coord_calls[0][0][0]
        lat_coord = add_coord_calls[1][0][0]
        self.assertEqual(lon_coord.circular, True)
        self.assertEqual(lat_coord.circular, False)


if __name__ == "__main__":
    tests.main()
