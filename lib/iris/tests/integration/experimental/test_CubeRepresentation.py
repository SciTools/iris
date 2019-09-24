# (C) British Crown Copyright 2018 - 2019, Met Office
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
"""Integration tests for cube html representation."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import skip

from iris.cube import Cube
from iris.tests import mock
import iris.tests.stock as stock
import numpy as np

from iris.experimental.representation import CubeRepresentation


@tests.skip_data
class TestNoMetadata(tests.IrisTest):
    # Test the situation where we have a cube with no metadata at all.
    def setUp(self):
        self.shape = (2, 3, 4)
        self.cube = Cube(np.arange(24).reshape(self.shape))
        self.representer = CubeRepresentation(self.cube)
        self.representer.repr_html()

    def test_cube_name(self):
        expected = 'Unknown'  # This cube has no metadata.
        result = self.representer.name
        self.assertEqual(expected, result)

    def test_cube_units(self):
        expected = 'unknown'  # This cube has no metadata.
        result = self.representer.units
        self.assertEqual(expected, result)

    def test_dim_names(self):
        expected = ['--'] * len(self.shape)
        result = self.representer.names
        self.assertEqual(expected, result)

    def test_shape(self):
        result = self.representer.shapes
        self.assertEqual(result, self.shape)


@tests.skip_data
class TestMissingMetadata(tests.IrisTest):
    def setUp(self):
        self.cube = stock.realistic_3d()

    def test_no_coords(self):
        all_coords = [coord.name() for coord in self.cube.coords()]
        for coord in all_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertNotIn('dimension coordinates', result)
        self.assertNotIn('auxiliary coordinates', result)
        self.assertNotIn('scalar coordinates', result)
        self.assertIn('attributes', result)

    def test_no_dim_coords(self):
        dim_coords = [c.name() for c in self.cube.coords(dim_coords=True)]
        for coord in dim_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertNotIn('dimension coordinates', result)
        self.assertIn('auxiliary coordinates', result)
        self.assertIn('scalar coordinates', result)
        self.assertIn('attributes', result)

    def test_no_aux_coords(self):
        aux_coords = ['forecast_period']
        for coord in aux_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertIn('dimension coordinates', result)
        self.assertNotIn('auxiliary coordinates', result)
        self.assertIn('scalar coordinates', result)
        self.assertIn('attributes', result)

    def test_no_scalar_coords(self):
        aux_coords = ['air_pressure']
        for coord in aux_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertIn('dimension coordinates', result)
        self.assertIn('auxiliary coordinates', result)
        self.assertNotIn('scalar coordinates', result)
        self.assertIn('attributes', result)

    def test_no_attrs(self):
        self.cube.attributes = {}
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertIn('dimension coordinates', result)
        self.assertIn('auxiliary coordinates', result)
        self.assertIn('scalar coordinates', result)
        self.assertNotIn('attributes', result)

    def test_no_cell_methods(self):
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertNotIn('cell methods', result)


@tests.skip_data
class TestScalarCube(tests.IrisTest):
    def setUp(self):
        self.cube = stock.realistic_3d()[0, 0, 0]
        self.representer = CubeRepresentation(self.cube)
        self.representer.repr_html()

    def test_identfication(self):
        # Is this scalar cube accurately identified?
        self.assertTrue(self.representer.scalar_cube)

    def test_header__name(self):
        header = self.representer._make_header()
        expected_name = self.cube.name().title().replace('_', ' ')
        self.assertIn(expected_name, header)

    def test_header__units(self):
        header = self.representer._make_header()
        expected_units = self.cube.units.symbol
        self.assertIn(expected_units, header)

    def test_header__scalar_str(self):
        # Check that 'scalar cube' is placed in the header.
        header = self.representer._make_header()
        expected_str = '(scalar cube)'
        self.assertIn(expected_str, header)

    def test_content__scalars(self):
        # Check an element "Scalar coordinates" is present in the main content.
        content = self.representer._make_content()
        expected_str = 'Scalar coordinates'
        self.assertIn(expected_str, content)

    def test_content__specific_scalar_coord(self):
        # Check a specific scalar coord is present in the main content.
        content = self.representer._make_content()
        expected_coord = self.cube.coords()[0]
        expected_coord_name = expected_coord.name()
        self.assertIn(expected_coord_name, content)
        expected_coord_val = str(expected_coord.points[0])
        self.assertIn(expected_coord_val, content)

    def test_content__attributes(self):
        # Check an element "attributes" is present in the main content.
        content = self.representer._make_content()
        expected_str = 'Attributes'
        self.assertIn(expected_str, content)


@tests.skip_data
class TestLazyDataRepr(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lazy_data_cube()

    def test_not_lazy(self):
        cube = stock.lat_lon_cube()
        representer = CubeRepresentation(cube)
        result = representer.repr_html()
        exp_str = '&nbsp;'
        self.assertIn(exp_str, result)

    @skip('Dask version does not include html repr')
    def test_lazy_data_repr(self):
        # Dask array repr uses an SVG.
        exp_str = '<svg'
        representer = CubeRepresentation(self.cube)
        result = representer._lazy_data_repr()
        self.assertIn(exp_str, result)

    def test_cannot_repr(self):
        # Test cases where the dask array repr cannot be made.
        cube = mock.MagicMock(spec=self.cube)
        cube.lazy_data()._repr_html_ = \
            mock.MagicMock(side_effect=AttributeError)
        representer = CubeRepresentation(cube)
        result = representer._lazy_data_repr()
        exp_str = '&nbsp;'
        self.assertIn(exp_str, result)


if __name__ == '__main__':
    tests.main()
