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

    def _aggregator(self, uses_weighting):
        # Returns a mock aggregator with a mocked method (uses_weighting)
        # which returns the given True/False condition.
        aggregator = mock.Mock(spec=WeightedAggregator)
        aggregator.cell_method = None
        aggregator.uses_weighting = mock.Mock(return_value=uses_weighting)

        return aggregator

    def _assert_warn_collapse_without_weight(self, coords, warnings):
        # Ensure that warning is raised.
        msg = "Collapsing spatial coordinate '{!r}' without weighting"
        for coord in coords:
            self.assertIn(
                msg.format(coord), [str(warn.message) for warn in warnings])

    def _assert_nowarn_collapse_without_weight(self, coords, warnings):
        # Ensure that warning is not rised.
        msg = "Collapsing spatial coordinate '{!r}' without weighting"
        for coord in coords:
            self.assertNotIn(
                msg.format(coord), [str(warn.message) for warn in warnings])

    def test_lat_lon_noweighted_aggregator(self):
        # Collapse latitude coordinate with unweighted aggregator.
        aggregator = mock.Mock(spec=Aggregator)
        aggregator.cell_method = None
        coords = ['latitude', 'longitude']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cube.collapsed(coords, aggregator, somekeyword='bla')

        msg = "Collapsing spatial coordinate '{!r}' without weighting"
        for coord in coords:
            self.assertNotIn(
                msg.format(coord), [str(warn.message) for warn in w])

    def test_lat_lon_weighted_aggregator(self):
        # Collapse latitude coordinate with weighted aggregator without
        # providing weights.
        aggregator = self._aggregator(False)
        coords = ['latitude', 'longitude']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cube.collapsed(coords, aggregator)

        coords = filter(lambda coord: 'latitude' in coord, coords)
        self._assert_warn_collapse_without_weight(coords, w)

    def test_lat_lon_weighted_aggregator_with_weights(self):
        # Collapse latitude coordinate with a weighted aggregators and
        # providing suitable weights.
        weights = np.array([[0.1, 0.5], [0.3, 0.2]])
        aggregator = self._aggregator(True)
        coords = ['latitude', 'longitude']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cube.collapsed(coords, aggregator, weights=weights)

        self._assert_nowarn_collapse_without_weight(coords, w)

    def test_lat_lon_weighted_aggregator_alt(self):
        # Collapse grid_latitude coordinate with weighted aggregator without
        # providing weights.  Tests coordinate matching logic.
        aggregator = self._aggregator(False)
        coords = ['grid_latitude', 'grid_longitude']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cube.collapsed(coords, aggregator)

        coords = filter(lambda coord: 'latitude' in coord, coords)
        self._assert_warn_collapse_without_weight(coords, w)

    def test_no_lat_weighted_aggregator_mixed(self):
        # Collapse grid_latitude and an unmatched coordinate (not lat/lon)
        # with weighted aggregator without providing weights.
        # Tests coordinate matching logic.
        aggregator = self._aggregator(False)
        coords = ['wibble']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cube.collapsed(coords, aggregator)

        self._assert_nowarn_collapse_without_weight(coords, w)


if __name__ == "__main__":
    tests.main()
