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


class Test_collapsed_warning(tests.IrisTest):
    def setUp(self):
        self.cube = iris.cube.Cube([[1, 2], [1, 2]])
        lat = iris.coords.DimCoord([1, 2], standard_name='latitude')
        lon = iris.coords.DimCoord([1, 2], standard_name='longitude')
        grid_lat = iris.coords.DimCoord([1, 2], standard_name='grid_latitude')
        grid_lon = iris.coords.DimCoord([1, 2], standard_name='grid_longitude')

        self.cube.add_dim_coord(lat, 0)
        self.cube.add_dim_coord(lon, 1)
        self.cube.add_aux_coord(grid_lat, 0)
        self.cube.add_aux_coord(grid_lon, 1)

    def test_collapse_lat_noweighted_aggregator(self):
        # Collapse latitude coordinate with unweighted aggregator with or
        # without weights.
        with mock.patch(__name__ + '.Aggregator', spec=True) as aggregator,\
                warnings.catch_warnings(record=True) as w:

            warnings.simplefilter("always")
            aggregator_instance = aggregator()
            aggregator_instance.cell_method = None

            self.cube.collapsed(
                'latitude', aggregator_instance, somekeyword='bla')

            msg = "Collapsing spatial coordinate 'latitude' without weighting"
            self.assertFalse(msg in [str(warn.message) for warn in w])

    def test_collapse_lat_weighted_aggregator(self):
        # Collapse latitude coordinate with weighted aggregator without
        # providing weights.
        with mock.patch(__name__ + '.WeightedAggregator',
                        spec=True) as aggregator, \
                warnings.catch_warnings(record=True) as w:

            warnings.simplefilter("always")
            aggregator_instance = aggregator()
            aggregator_instance.cell_method = None
            aggregator_instance.uses_weighting = mock.Mock(
                return_value=False)

            self.cube.collapsed(
                'latitude', aggregator_instance, somekeyword='bla')

            aggregator_instance.uses_weighting.assert_called_once_with(
                somekeyword='bla')

            msg = "Collapsing spatial coordinate 'latitude' without weighting"
            self.assertTrue(msg in [str(warn.message) for warn in w])
            index = [str(warn.message) for warn in w].index(msg)
            self.assertTrue(issubclass(w[index].category, UserWarning))

    def test_collapse_lat_weighted_aggregator_with_weights(self):
        # Collapse latitude coordinate with a weighted aggregators and
        # providing suitable weights.
        with mock.patch(__name__ + '.WeightedAggregator',
                        spec=True) as aggregator, \
                warnings.catch_warnings(record=True) as w:

            weights = np.array([[0.1, 0.5], [0.3, 0.2]])
            warnings.simplefilter("always")
            aggregator_instance = aggregator()
            aggregator_instance.cell_method = None
            aggregator_instance.uses_weighting = mock.Mock(
                return_value=True)

            self.cube.collapsed(
                'latitude', aggregator_instance, somekeyword='bla',
                weights=weights)

            aggregator_instance.uses_weighting.assert_called_once_with(
                somekeyword='bla', weights=weights)

            msg = "Collapsing spatial coordinate 'latitude' without weighting"
            self.assertFalse(msg in [str(warn.message) for warn in w])

    def test_collapse_lat_weighted_aggregator_alt(self):
        # Collapse grid_latitude coordinate with weighted aggregator without
        # providing weights.  Tests coordinate matching logic.
        with mock.patch(__name__ + '.WeightedAggregator',
                        spec=True) as aggregator, \
                warnings.catch_warnings(record=True) as w:

            warnings.simplefilter("always")
            aggregator_instance = aggregator()
            aggregator_instance.cell_method = None
            aggregator_instance.uses_weighting = mock.Mock(
                return_value=False)

            self.cube.collapsed(
                'grid_latitude', aggregator_instance, somekeyword='bla')

            aggregator_instance.uses_weighting.assert_called_once_with(
                somekeyword='bla')

            msg = ("Collapsing spatial coordinate 'grid_latitude' without "
                   'weighting')
            self.assertTrue(msg in [str(warn.message) for warn in w])
            index = [str(warn.message) for warn in w].index(msg)
            self.assertTrue(issubclass(w[index].category, UserWarning))


if __name__ == "__main__":
    tests.main()
