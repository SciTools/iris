# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""
Unit tests for the
`iris.aux_factory.OceanSigmaFactory` class.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.aux_factory import OceanSigmaFactory
from iris.coords import AuxCoord, DimCoord
from iris.tests import mock
from cf_units import Unit


class Test___init__(tests.IrisTest):
    def setUp(self):
        self.sigma = mock.Mock(units=Unit('1'), nbounds=0)
        self.eta = mock.Mock(units=Unit('m'), nbounds=0)
        self.depth = mock.Mock(units=Unit('m'), nbounds=0)
        self.kwargs = dict(sigma=self.sigma, eta=self.eta,
                           depth=self.depth)

    def test_insufficient_coordinates(self):
        with self.assertRaises(ValueError):
            OceanSigmaFactory()
        with self.assertRaises(ValueError):
            OceanSigmaFactory(sigma=None, eta=self.eta, depth=self.depth)
        with self.assertRaises(ValueError):
            OceanSigmaFactory(sigma=self.sigma, eta=None, depth=self.depth)
        with self.assertRaises(ValueError):
            OceanSigmaFactory(sigma=self.sigma, eta=self.eta, depth=None)

    def test_sigma_too_many_bounds(self):
        self.sigma.nbounds = 4
        with self.assertRaises(ValueError):
            OceanSigmaFactory(**self.kwargs)

    def test_sigma_incompatible_units(self):
        self.sigma.units = Unit('km')
        with self.assertRaises(ValueError):
            OceanSigmaFactory(**self.kwargs)

    def test_eta_incompatible_units(self):
        self.eta.units = Unit('km')
        with self.assertRaises(ValueError):
            OceanSigmaFactory(**self.kwargs)

    def test_depth_incompatible_units(self):
        self.depth.units = Unit('km')
        with self.assertRaises(ValueError):
            OceanSigmaFactory(**self.kwargs)


class Test_dependencies(tests.IrisTest):
    def setUp(self):
        self.sigma = mock.Mock(units=Unit('1'), nbounds=0)
        self.eta = mock.Mock(units=Unit('m'), nbounds=0)
        self.depth = mock.Mock(units=Unit('m'), nbounds=0)
        self.kwargs = dict(sigma=self.sigma, eta=self.eta, depth=self.depth)

    def test_values(self):
        factory = OceanSigmaFactory(**self.kwargs)
        self.assertEqual(factory.dependencies, self.kwargs)


class Test_make_coord(tests.IrisTest):
    @staticmethod
    def coord_dims(coord):
        mapping = dict(sigma=(0,), eta=(1, 2), depth=(1, 2))
        return mapping[coord.name()]

    @staticmethod
    def derive(sigma, eta, depth, coord=True):
        result = eta + sigma * (depth + eta)
        if coord:
            name = 'sea_surface_height_above_reference_ellipsoid'
            result = AuxCoord(result,
                              standard_name=name,
                              units='m',
                              attributes=dict(positive='up'))
        return result

    def setUp(self):
        self.sigma = DimCoord(np.linspace(-0.05, -1, 5),
                              long_name='sigma', units='1')
        self.eta = AuxCoord(np.arange(-1, 3, dtype=np.float).reshape(2, 2),
                            long_name='eta', units='m')
        self.depth = AuxCoord(np.arange(4, dtype=np.float).reshape(2, 2) * 1e3,
                              long_name='depth', units='m')
        self.kwargs = dict(sigma=self.sigma, eta=self.eta, depth=self.depth)

    def test_derived_points(self):
        # Broadcast expected points given the known dimensional mapping.
        sigma = self.sigma.points[..., np.newaxis, np.newaxis]
        eta = self.eta.points[np.newaxis, ...]
        depth = self.depth.points[np.newaxis, ...]
        # Calculate the expected result.
        expected_coord = self.derive(sigma, eta, depth)
        # Calculate the actual result.
        factory = OceanSigmaFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        self.assertEqual(expected_coord, coord)


class Test_update(tests.IrisTest):
    def setUp(self):
        self.sigma = mock.Mock(units=Unit('1'), nbounds=0)
        self.eta = mock.Mock(units=Unit('m'), nbounds=0)
        self.depth = mock.Mock(units=Unit('m'), nbounds=0)
        self.kwargs = dict(sigma=self.sigma, eta=self.eta, depth=self.depth)
        self.factory = OceanSigmaFactory(**self.kwargs)

    def test_sigma(self):
        new_sigma = mock.Mock(units=Unit('1'), nbounds=0)
        self.factory.update(self.sigma, new_sigma)
        self.assertIs(self.factory.sigma, new_sigma)

    def test_sigma_too_many_bounds(self):
        new_sigma = mock.Mock(units=Unit('1'), nbounds=4)
        with self.assertRaises(ValueError):
            self.factory.update(self.sigma, new_sigma)

    def test_sigma_incompatible_units(self):
        new_sigma = mock.Mock(units=Unit('Pa'), nbounds=0)
        with self.assertRaises(ValueError):
            self.factory.update(self.sigma, new_sigma)

    def test_eta(self):
        new_eta = mock.Mock(units=Unit('m'), nbounds=0)
        self.factory.update(self.eta, new_eta)
        self.assertIs(self.factory.eta, new_eta)

    def test_eta_incompatible_units(self):
        new_eta = mock.Mock(units=Unit('Pa'), nbounds=0)
        with self.assertRaises(ValueError):
            self.factory.update(self.eta, new_eta)

    def test_depth(self):
        new_depth = mock.Mock(units=Unit('m'), nbounds=0)
        self.factory.update(self.depth, new_depth)
        self.assertIs(self.factory.depth, new_depth)

    def test_depth_incompatible_units(self):
        new_depth = mock.Mock(units=Unit('Pa'), nbounds=0)
        with self.assertRaises(ValueError):
            self.factory.update(self.depth, new_depth)


if __name__ == "__main__":
    tests.main()
