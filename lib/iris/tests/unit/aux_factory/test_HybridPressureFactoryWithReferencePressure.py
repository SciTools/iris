# (C) British Crown Copyright 2014, Met Office
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
`iris.aux_factory.HybridPressureFactoryWithReferencePressure` class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import numpy as np

import iris
from iris.aux_factory import HybridPressureFactoryWithReferencePressure as \
    HybridFactory


class Test___init__(tests.IrisTest):
    def setUp(self):
        self.delta = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.reference_pressure = mock.Mock(units=iris.unit.Unit('Pa'),
                                            nbounds=0)
        self.sigma = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.surface_air_pressure = mock.Mock(units=iris.unit.Unit('Pa'),
                                              nbounds=0)

    def test_insufficient_coords(self):
        with self.assertRaises(ValueError):
            HybridFactory()
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
        self.sigma.units = iris.unit.Unit('Pa')
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=self.reference_pressure,
                          sigma=self.sigma,
                          surface_air_pressure=self.surface_air_pressure)

    def test_incompatible_reference_pressure_units(self):
        self.reference_pressure.units = iris.unit.Unit('1')
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=self.reference_pressure,
                          sigma=self.sigma,
                          surface_air_pressure=self.surface_air_pressure)

    def test_incompatible_surface_air_pressure_units(self):
        self.surface_air_pressure.units = iris.unit.Unit('unknown')
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

    def test_too_many_delta_bounds(self):
        self.delta.nbounds = 4
        with self.assertRaises(ValueError):
            HybridFactory(delta=self.delta,
                          reference_pressure=self.reference_pressure,
                          sigma=self.sigma,
                          surface_air_pressure=self.surface_air_pressure)

    def test_too_many_sigma_bounds(self):
        self.sigma.nbounds = 4
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
        self.reference_pressure = mock.Mock(units=iris.unit.Unit('Pa'),
                                            nbounds=0)
        self.sigma = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.surface_air_pressure = mock.Mock(units=iris.unit.Unit('Pa'),
                                              nbounds=0)

    def test_value(self):
        kwargs = dict(delta=self.delta,
                      reference_pressure=self.reference_pressure,
                      sigma=self.sigma,
                      surface_air_pressure=self.surface_air_pressure)
        factory = HybridFactory(**kwargs)
        self.assertEqual(factory.dependencies, kwargs)


class Test_make_coord(tests.IrisTest):
    def test_points_only(self):
        delta = iris.coords.DimCoord(
            [0.0, 1.0, 2.0], long_name='delta')
        reference_pressure = iris.coords.DimCoord(
            2.0, long_name='p_zero', units='Pa')
        sigma = iris.coords.DimCoord(
            [1.0, 0.9, 0.8], long_name='sigma')
        surface_air_pressure = iris.coords.AuxCoord(
            np.arange(4).reshape(2, 2), 'surface_air_pressure',
            units='Pa')

        def coords_dims_func(coord):
            mapping = dict(delta=(0,), p_zero=(), sigma=(0,),
                           surface_air_pressure=(1, 2))
            return mapping[coord.name()]

        # Determine expected coord by manually broadcasting coord points
        # knowing the dimension mapping.
        delta_pts = delta.points[..., np.newaxis, np.newaxis]
        ref_p_pts = reference_pressure.points[..., np.newaxis, np.newaxis]
        sigma_pts = sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts * ref_p_pts + sigma_pts * surf_pts
        expected_coord = iris.coords.AuxCoord(expected_points,
                                              standard_name='air_pressure',
                                              units='Pa')

        factory = HybridFactory(delta=delta,
                                reference_pressure=reference_pressure,
                                sigma=sigma,
                                surface_air_pressure=surface_air_pressure)
        derived_coord = factory.make_coord(coords_dims_func)
        self.assertEqual(expected_coord, derived_coord)

    def test_none_delta(self):
        reference_pressure = iris.coords.DimCoord(
            2.0, long_name='p_zero', units='Pa')
        sigma = iris.coords.DimCoord(
            [1.0, 0.9, 0.8], long_name='sigma')
        surface_air_pressure = iris.coords.AuxCoord(
            np.arange(4).reshape(2, 2), 'surface_air_pressure',
            units='Pa')

        def coords_dims_func(coord):
            mapping = dict(p_zero=(), sigma=(0,),
                           surface_air_pressure=(1, 2))
            return mapping[coord.name()]

        sigma_pts = sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = surface_air_pressure.points[np.newaxis, ...]
        expected_points = sigma_pts * surf_pts
        expected_coord = iris.coords.AuxCoord(expected_points,
                                              standard_name='air_pressure',
                                              units='Pa')

        factory = HybridFactory(reference_pressure=reference_pressure,
                                sigma=sigma,
                                surface_air_pressure=surface_air_pressure)
        derived_coord = factory.make_coord(coords_dims_func)
        self.assertEqual(expected_coord, derived_coord)

    def test_none_reference_pressure(self):
        delta = iris.coords.DimCoord(
            [0.0, 1.0, 2.0], long_name='delta')
        sigma = iris.coords.DimCoord(
            [1.0, 0.9, 0.8], long_name='sigma')
        surface_air_pressure = iris.coords.AuxCoord(
            np.arange(4).reshape(2, 2), 'surface_air_pressure',
            units='Pa')

        def coords_dims_func(coord):
            mapping = dict(delta=(0,), sigma=(0,),
                           surface_air_pressure=(1, 2))
            return mapping[coord.name()]

        # Determine expected coord by manually broadcasting coord points
        # knowing the dimension mapping.
        sigma_pts = sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = surface_air_pressure.points[np.newaxis, ...]
        expected_points = sigma_pts * surf_pts
        expected_coord = iris.coords.AuxCoord(expected_points,
                                              standard_name='air_pressure',
                                              units='Pa')

        factory = HybridFactory(delta=delta,
                                sigma=sigma,
                                surface_air_pressure=surface_air_pressure)
        derived_coord = factory.make_coord(coords_dims_func)
        self.assertEqual(expected_coord, derived_coord)

    def test_none_sigma(self):
        delta = iris.coords.DimCoord(
            [0.0, 1.0, 2.0], long_name='delta')
        reference_pressure = iris.coords.DimCoord(
            2.0, long_name='p_zero', units='Pa')
        surface_air_pressure = iris.coords.AuxCoord(
            np.arange(4).reshape(2, 2), 'surface_air_pressure',
            units='Pa')

        def coords_dims_func(coord):
            mapping = dict(delta=(0,), p_zero=(),
                           surface_air_pressure=(1, 2))
            return mapping[coord.name()]

        delta_pts = delta.points[..., np.newaxis, np.newaxis]
        ref_p_pts = reference_pressure.points[..., np.newaxis, np.newaxis]
        expected_points = delta_pts * ref_p_pts
        expected_coord = iris.coords.AuxCoord(expected_points,
                                              standard_name='air_pressure',
                                              units='Pa')
        factory = HybridFactory(delta=delta,
                                reference_pressure=reference_pressure,
                                surface_air_pressure=surface_air_pressure)
        derived_coord = factory.make_coord(coords_dims_func)
        self.assertEqual(expected_coord, derived_coord)

    def test_none_surface_air_pressure(self):
        delta = iris.coords.DimCoord(
            [0.0, 1.0, 2.0], long_name='delta')
        reference_pressure = iris.coords.DimCoord(
            2.0, long_name='p_zero', units='Pa')
        sigma = iris.coords.DimCoord(
            [1.0, 0.9, 0.8], long_name='sigma')

        def coords_dims_func(coord):
            mapping = dict(delta=(0,), p_zero=(), sigma=(0,))
            return mapping[coord.name()]

        delta_pts = delta.points
        ref_p_pts = reference_pressure.points
        expected_points = delta_pts * ref_p_pts
        expected_coord = iris.coords.AuxCoord(expected_points,
                                              standard_name='air_pressure',
                                              units='Pa')
        factory = HybridFactory(delta=delta,
                                reference_pressure=reference_pressure,
                                sigma=sigma)
        derived_coord = factory.make_coord(coords_dims_func)
        self.assertEqual(expected_coord, derived_coord)

    def test_with_bounds(self):
        delta = iris.coords.DimCoord(
            [0.0, 1.0, 2.0], long_name='delta')
        delta.guess_bounds(0)
        reference_pressure = iris.coords.DimCoord(
            2.0, long_name='p_zero', units='Pa')
        sigma = iris.coords.DimCoord(
            [1.0, 0.9, 0.8], long_name='sigma')
        sigma.guess_bounds(0.5)
        surface_air_pressure = iris.coords.AuxCoord(
            np.arange(4).reshape(2, 2), 'surface_air_pressure',
            units='Pa')

        def coords_dims_func(coord):
            mapping = dict(delta=(0,), p_zero=(), sigma=(0,),
                           surface_air_pressure=(1, 2))
            return mapping[coord.name()]

        # Determine expected coord by manually broadcasting coord points
        # and bounds knowing the dimension mapping.
        delta_pts = delta.points[..., np.newaxis, np.newaxis]
        ref_p_pts = reference_pressure.points[..., np.newaxis, np.newaxis]
        sigma_pts = sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts * ref_p_pts + sigma_pts * surf_pts
        delta_vals = delta.bounds.reshape(3, 1, 1, 2)
        ref_p_vals = reference_pressure.points.reshape(1, 1, 1, 1)
        sigma_vals = sigma.bounds.reshape(3, 1, 1, 2)
        surf_vals = surface_air_pressure.points.reshape(1, 2, 2, 1)
        expected_points = delta_pts * ref_p_pts + sigma_pts * surf_pts
        expected_bounds = delta_vals * ref_p_vals + sigma_vals * surf_vals
        expected_coord = iris.coords.AuxCoord(expected_points,
                                              standard_name='air_pressure',
                                              units='Pa',
                                              bounds=expected_bounds)

        factory = HybridFactory(delta=delta,
                                reference_pressure=reference_pressure,
                                sigma=sigma,
                                surface_air_pressure=surface_air_pressure)
        derived_coord = factory.make_coord(coords_dims_func)
        self.assertEqual(expected_coord, derived_coord)


class Test_update(tests.IrisTest):
    def setUp(self):
        self.delta = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.reference_pressure = mock.Mock(units=iris.unit.Unit('Pa'),
                                            nbounds=0)
        self.sigma = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.surface_air_pressure = mock.Mock(units=iris.unit.Unit('Pa'),
                                              nbounds=0)

        self.factory = HybridFactory(
            delta=self.delta, reference_pressure=self.reference_pressure,
            sigma=self.sigma, surface_air_pressure=self.surface_air_pressure)

    def test_good_delta(self):
        new_delta_coord = mock.Mock(units=iris.unit.Unit('1'), nbounds=0)
        self.factory.update(self.delta, new_delta_coord)
        self.assertIs(self.factory.delta, new_delta_coord)

    def test_bad_delta(self):
        new_delta_coord = mock.Mock(units=iris.unit.Unit('Pa'), nbounds=0)
        with self.assertRaises(ValueError):
            self.factory.update(self.delta, new_delta_coord)

    def test_alternative_bad_delta(self):
        new_delta_coord = mock.Mock(units=iris.unit.Unit('1'), nbounds=4)
        with self.assertRaises(ValueError):
            self.factory.update(self.delta, new_delta_coord)

    def test_good_ref_pressure(self):
        new_ref_p_coord = mock.Mock(units=iris.unit.Unit('Pa'), nbounds=0)
        self.factory.update(self.reference_pressure, new_ref_p_coord)
        self.assertIs(self.factory.reference_pressure, new_ref_p_coord)

    def test_bad_ref_pressure(self):
        new_ref_p_coord = mock.Mock(units=iris.unit.Unit('m'), nbounds=0)
        with self.assertRaises(ValueError):
            self.factory.update(self.reference_pressure, new_ref_p_coord)

    def test_good_surface_pressure(self):
        new_surface_p_coord = mock.Mock(units=iris.unit.Unit('Pa'), nbounds=0)
        self.factory.update(self.surface_air_pressure, new_surface_p_coord)
        self.assertIs(self.factory.surface_air_pressure, new_surface_p_coord)

    def test_bad_surface_pressure(self):
        new_surface_p_coord = mock.Mock(units=iris.unit.Unit('km'), nbounds=0)
        with self.assertRaises(ValueError):
            self.factory.update(self.surface_air_pressure, new_surface_p_coord)

    def test_non_dependency(self):
        old_coord = mock.Mock()
        new_coord = mock.Mock()
        orig_dependencies = self.factory.dependencies
        self.factory.update(old_coord, new_coord)
        self.assertEqual(orig_dependencies, self.factory.dependencies)

    def test_none_delta(self):
        self.factory.update(self.delta, None)
        self.assertIsNone(self.factory.delta)

    def test_none_sigma(self):
        self.factory.update(self.sigma, None)
        self.assertIsNone(self.factory.sigma)

    def test_insufficient_coords(self):
        self.factory.update(self.delta, None)
        with self.assertRaises(ValueError):
            self.factory.update(self.surface_air_pressure, None)


if __name__ == "__main__":
    tests.main()
