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


class Test_weighted_collapsed(tests.IrisTest):
    def setUp(self):
        _collapsed_patch = mock.patch('iris.cube.Cube.collapsed')
        _collapsed_patch.start()
        self.addCleanup(_collapsed_patch.stop)

        self.coords = (mock.Mock(), mock.Mock())
        _getcoord_patch = mock.patch(
            'iris.analysis.cartography.get_lat_lon_coords',
            return_value=self.coords)
        _getcoord_patch.start()
        self.addCleanup(_getcoord_patch.stop)

        self.area_weight = [mock.Mock(), mock.Mock()]
        _areaweight_patch = mock.patch(
            'iris.analysis.cartography.area_weights',
            return_value=self.area_weight)
        _areaweight_patch.start()
        self.addCleanup(_areaweight_patch.stop)

        self.cube = iris.cube.Cube([1])
        self.coords = 'latitude'

    def test_provide_unweighted_aggregator(self):
        # Providing unweighted aggregator to collapse with no weights keyword
        # supplied.
        with mock.patch(__name__ + '.Aggregator', spec=True) as aggregator,\
                mock.patch('warnings.warn') as warn:
            aggregator_instance = aggregator()

            self.cube.weighted_collapsed(
                self.coords, aggregator_instance, somekeyword='bla')

            self.assertEqual(warn.call_count, 1)
            warn_msg = warn.call_args[0][0]
            msg = ('Aggregation function does not support spacial area '
                   'weights, continuing without determining these weights')
            self.assertTrue(warn_msg.startswith(msg))

            iris.cube.Cube.collapsed.assert_called_once_with(
                self.coords, aggregator_instance, somekeyword='bla')

            self.assertFalse(iris.analysis.cartography.area_weights.called)
            self.assertFalse(
                iris.analysis.cartography.get_lat_lon_coords.called)

    def test_provide_weighted_aggregator_withweight(self):
        # Providing a weighted aggregator and supplying overiding weights
        # keyword.
        weights = [1, 2, 3, 4]
        with mock.patch(__name__ + '.WeightedAggregator',
                        spec=True) as aggregator,\
                mock.patch('warnings.warn') as warn:
            aggregator_instance = aggregator()

            self.cube.weighted_collapsed(
                self.coords, aggregator_instance, somekeyword='bla',
                weights=weights)

            self.assertEqual(warn.call_count, 1)
            warn_msg = warn.call_args[0][0]
            msg = ('Specified weights overide spacial area weights')
            self.assertTrue(warn_msg.startswith(msg))

            iris.cube.Cube.collapsed.assert_called_once_with(
                self.coords, aggregator_instance, somekeyword='bla',
                weights=weights)

            self.assertFalse(iris.analysis.cartography.area_weights.called)
            self.assertFalse(
                iris.analysis.cartography.get_lat_lon_coords.called)

    def test_provide_weighted_aggregator_noweight(self):
        # Providing a weighted aggregator and supplying no overiding weights
        # keyword.
        with mock.patch(__name__ + '.WeightedAggregator',
                        spec=True) as aggregator:
            aggregator_instance = aggregator()

            self.cube.weighted_collapsed(
                self.coords, aggregator_instance, somekeyword='bla')

            iris.cube.Cube.collapsed.assert_called_once_with(
                self.coords, aggregator_instance, somekeyword='bla',
                weights=self.area_weight)

            self.assertTrue(iris.analysis.cartography.area_weights.called)
            self.assertTrue(
                iris.analysis.cartography.get_lat_lon_coords.called)


if __name__ == "__main__":
    tests.main()
