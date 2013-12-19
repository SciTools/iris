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
"""Unit tests for the `iris.cube..............................................` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris
from iris.aux_factory import HybridPressureFactoryWithReferencePressure as \
    HybridFactory


class Test___init__(tests.IrisTest):
    def setUp(self):
        self.delta = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.reference_pressure = mock.Mock(units=iris.unit.Unit('Pa'))
        self.sigma = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.surface_air_pressure = mock.Mock(units=iris.unit.Unit('Pa'))

    def test_insufficient_coords(self):
        with self.assertRaises(ValueError):
            HybridFactory()
        with self.assertRaises(ValueError):
            HybridFactory(delta=None,
                          reference_pressure=None,
                          sigma=self.sigma,
                          surface_air_pressure=self.reference_pressure)
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=None,
                          sigma=self.sigma,
                          surface_air_pressure=None)
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=None,
                          sigma=None,
                          surface_air_pressure=self.reference_pressure)
    
    def test_incompatible_delta_units(self):
        self.delta.units = iris.unit.Unit('m')
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=self.reference_pressure,
                          sigma=self.sigma,
                          surface_air_pressure=self.surface_air_pressure)

    def test_incompatible_sigma_units(self):
        self.sigma.units = iris.unit.Unit('degrees')
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=self.reference_pressure,
                          sigma=self.sigma,
                          surface_air_pressure=self.surface_air_pressure)

    def test_incompatible_reference_pressure_units(self):
        self.sigma.units = iris.unit.Unit('1')
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=self.reference_pressure,
                          sigma=self.sigma,
                          surface_air_pressure=self.surface_air_pressure)

    def test_incompatible_surface_air_pressure_units(self):
        self.sigma.units = iris.unit.Unit('unknown')
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=self.reference_pressure,
                          sigma=self.sigma,
                          surface_air_pressure=self.surface_air_pressure)

    def test_different_pressure_units(self):
        self.reference_pressure.units = iris.unit.Unit('hPa')
        self.surface_air_pressure.units = iris.unit.Unit('Pa')
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=self.reference_pressure,
                          sigma=self.sigma,
                          surface_air_pressure=self.surface_air_pressure)

    def test_factory_metadata(self):
        factory = HybridFactory(delta=self.delta,
                                reference_pressure=self.reference_pressure,
                                sigma=self.sigma,
                                surface_air_pressure=self.surface_air_pressure)
        self.assertEqual(factory.standard_name, 'air_pressure')
        self.assertIsNone(factory.long_name)
        self.assertIsNone(factory.var_name)
        self.assertEqual(factory.units, self.reference_pressure.units)
        self.assertEqual(factory.units, self.surface_air_pressure.units)
        self.assertIsNone(factory.coord_system)
        self.assertEqual(factory.attributes, {})
        

class Test_dependencies(tests.IrisTest):
    def setUp(self):
        self.delta = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.reference_pressure = mock.Mock(units=iris.unit.Unit('Pa'))
        self.sigma = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.surface_air_pressure = mock.Mock(units=iris.unit.Unit('Pa'))

    def test_value(self):
        kwargs = dict(delta=self.delta,
                      reference_pressure=self.reference_pressure,
                      sigma=self.sigma,
                      surface_air_pressure=self.surface_air_pressure)
        factory = HybridFactory(**kwargs)
        self.assertEqual(factory.dependencies(), kwargs)


class Test_make_coord(tests.IrisTest):
    def setUp(self):
        self.delta = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.reference_pressure = mock.Mock(units=iris.unit.Unit('Pa'))
        self.sigma = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.surface_air_pressure = mock.Mock(units=iris.unit.Unit('Pa'))

    def test_value(self):
        pass


class Test_update(tests.IrisTest):
    def setUp(self):
        self.delta = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.reference_pressure = mock.Mock(units=iris.unit.Unit('Pa'))
        self.sigma = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.surface_air_pressure = mock.Mock(units=iris.unit.Unit('Pa'))

    def test_good_delta(self):
        factory = HybridFactory(delta=self.delta,
                                reference_pressure=self.reference_pressure,
                                sigma=self.sigma,
                                surface_air_pressure=self.surface_air_pressure)
        new_delta_coord = mock.Mock(nbounds=0)
        factory.update(self.delta, new_delta_coord)
        self.assertIs(factory.delta, new_delta_coord)


if __name__ == "__main__":
    tests.main()
