# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the `iris.cube.Cube` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np
import warnings

import iris
from iris.analysis import WeightedAggregator, Aggregator


class Test_xml(tests.IrisTest):
    def test_checksum_ignores_masked_values(self):
        # Mask out an single element.
        data = np.ma.arange(12).reshape(3, 4)
        data[1, 2] = np.ma.masked
        cube = iris.cube.Cube(data)
        self.assertCML(cube, ('unit', 'cube', 'Cube', 'xml', 'mask.cml'))

        # If we change the underlying value before masking it, the
        # checksum should be unaffected.
        data = np.ma.arange(12).reshape(3, 4)
        data[1, 2] = 42
        data[1, 2] = np.ma.masked
        cube = iris.cube.Cube(data)
        self.assertCML(cube, ('unit', 'cube', 'Cube', 'xml', 'mask.cml'))


class Test_collapsed__warning(tests.IrisTest):
    def setUp(self):
        self.cube = iris.cube.Cube([[1, 2], [1, 2]])
        lat = iris.coords.DimCoord([1, 2], standard_name='latitude')
        lon = iris.coords.DimCoord([1, 2], standard_name='longitude')
        grid_lat = iris.coords.AuxCoord([1, 2], standard_name='grid_latitude')
        grid_lon = iris.coords.AuxCoord([1, 2], standard_name='grid_longitude')
        wibble = iris.coords.AuxCoord([1, 2], long_name='wibble')

        self.cube.add_dim_coord(lat, 0)
        self.cube.add_dim_coord(lon, 1)
        self.cube.add_aux_coord(grid_lat, 0)
        self.cube.add_aux_coord(grid_lon, 1)
        self.cube.add_aux_coord(wibble, 1)

    def test_lat_lon_noweighted_aggregator(self):
        # Collapse latitude coordinate with unweighted aggregator.
        warnings.simplefilter("always")
        aggregator_instance = mock.Mock(spec=Aggregator)
        aggregator_instance.cell_method = None
        coords = ['latitude', 'longitude']

        with warnings.catch_warnings(record=True) as w:
            self.cube.collapsed(coords, aggregator_instance, somekeyword='bla')

        msg = "Collapsing spatial coordinate '{}' without weighting"
        for coord in coords:
            self.assertNotIn(
                msg.format(coord), [str(warn.message) for warn in w])

    def test_lat_lon_weighted_aggregator(self):
        # Collapse latitude coordinate with weighted aggregator without
        # providing weights.
        warnings.simplefilter("always")
        aggregator_instance = mock.Mock(spec=WeightedAggregator)
        aggregator_instance.cell_method = None
        aggregator_instance.uses_weighting = mock.Mock(
            return_value=False)
        coords = ['latitude', 'longitude']

        with warnings.catch_warnings(record=True) as w:
            self.cube.collapsed(coords, aggregator_instance)

        msg = "Collapsing spatial coordinate '{}' without weighting"
        coords = filter(lambda coord: 'latitude' in coord, coords)
        for coord in coords:
            self.assertIn(
                msg.format(coord), [str(warn.message) for warn in w])

    def test_lat_lon_weighted_aggregator_with_weights(self):
        # Collapse latitude coordinate with a weighted aggregators and
        # providing suitable weights.
        weights = np.array([[0.1, 0.5], [0.3, 0.2]])
        warnings.simplefilter("always")
        aggregator_instance = mock.Mock(spec=WeightedAggregator)
        aggregator_instance.cell_method = None
        aggregator_instance.uses_weighting = mock.Mock(
            return_value=True)
        coords = ['latitude', 'longitude']

        with warnings.catch_warnings(record=True) as w:
            self.cube.collapsed(coords, aggregator_instance, weights=weights)

        msg = "Collapsing spatial coordinate '{}' without weighting"
        for coord in coords:
            self.assertNotIn(
                msg.format(coord), [str(warn.message) for warn in w])

    def test_lat_lon_weighted_aggregator_alt(self):
        # Collapse grid_latitude coordinate with weighted aggregator without
        # providing weights.  Tests coordinate matching logic.
        warnings.simplefilter("always")
        aggregator_instance = mock.Mock(spec=WeightedAggregator)
        aggregator_instance.cell_method = None
        aggregator_instance.uses_weighting = mock.Mock(
            return_value=False)
        coords = ['grid_latitude', 'grid_longitude']

        with warnings.catch_warnings(record=True) as w:
            self.cube.collapsed(coords, aggregator_instance)

        msg = "Collapsing spatial coordinate '{}' without weighting"
        coords = filter(lambda coord: 'latitude' in coord, coords)
        for coord in coords:
            self.assertIn(
                msg.format(coord), [str(warn.message) for warn in w])

    def test_no_lat_weighted_aggregator_mixed(self):
        # Collapse grid_latitude and an unmatched coordinate (not lat/lon)
        # with weighted aggregator without providing weights.
        # Tests coordinate matching logic.
        warnings.simplefilter("always")
        aggregator_instance = mock.Mock(spec=WeightedAggregator)
        aggregator_instance.cell_method = None
        aggregator_instance.uses_weighting = mock.Mock(
            return_value=False)
        coords = ['wibble']

        with warnings.catch_warnings(record=True) as w:
            self.cube.collapsed(coords, aggregator_instance)

        msg = "Collapsing spatial coordinate '{}' without weighting"
        for coord in coords:
            self.assertNotIn(
                msg.format(coord), [str(warn.message) for warn in w])


if __name__ == "__main__":
    tests.main()
