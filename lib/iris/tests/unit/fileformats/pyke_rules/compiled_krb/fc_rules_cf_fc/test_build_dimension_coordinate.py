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
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.build_dimension_coordinate`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import warnings

import numpy as np
import mock

from iris.coords import AuxCoord, DimCoord
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    build_dimension_coordinate


class RulesTestMixin(object):
    def setUp(self):
        # Create dummy pyke engine.
        self.engine = mock.Mock(
            cube=mock.Mock(),
            cf_var=mock.Mock(dimensions=('foo', 'bar')),
            filename='DUMMY',
            provides=dict(coordinates=[]))

        # Create patch for deferred loading that prevents attempted
        # file access. This assumes that self.cf_coord_var and
        # self.cf_bounds_var are defined in the test case.
        def patched__getitem__(proxy_self, keys):
            for var in (self.cf_coord_var, self.cf_bounds_var):
                if proxy_self.variable_name == var.cf_name:
                    return var[keys]
            raise RuntimeError()

        self.deferred_load_patch = mock.patch(
            'iris.fileformats.netcdf.NetCDFDataProxy.__getitem__',
            new=patched__getitem__)

        # Patch the helper function that retrieves the bounds cf variable.
        # This avoids the need for setting up further mocking of cf objects.
        def get_cf_bounds_var(coord_var):
            return self.cf_bounds_var

        self.get_cf_bounds_var_patch = mock.patch(
            'iris.fileformats._pyke_rules.compiled_krb.'
            'fc_rules_cf_fc.get_cf_bounds_var',
            new=get_cf_bounds_var)


class TestCoordConstruction(tests.IrisTest, RulesTestMixin):
    def setUp(self):
        # Call parent setUp explicitly, because of how unittests work.
        RulesTestMixin.setUp(self)

        bounds = np.arange(12).reshape(6, 2)
        self.cf_bounds_var = mock.Mock(
            dimensions=('x', 'nv'),
            cf_name='wibble_bnds',
            shape=bounds.shape,
            __getitem__=lambda self, key: bounds[key])
        self.bounds = bounds

    def _set_cf_coord_var(self, points):
        self.cf_coord_var = mock.Mock(
            dimensions=('foo',),
            cf_name='wibble',
            standard_name=None,
            long_name='wibble',
            units='m',
            shape=points.shape,
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key])

    def test_dim_coord_construction(self):
        self._set_cf_coord_var(np.arange(6))

        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=self.bounds)

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(
                expected_coord, [0])

    def test_aux_coord_construction(self):
        # Use non monotonically increasing coordinates to force aux coord
        # construction.
        self._set_cf_coord_var(np.array([1, 3, 2, 4, 6, 5]))

        expected_coord = AuxCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=self.bounds)

        warning_patch = mock.patch('warnings.warn')

        # Asserts must lie within context manager because of deferred loading.
        with warning_patch, self.deferred_load_patch, \
                self.get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_aux_coord.assert_called_with(
                expected_coord, [0])
            self.assertIn("creating 'wibble' auxiliary coordinate instead",
                          warnings.warn.call_args[0][0])


class TestBoundsVertexDim(tests.IrisTest, RulesTestMixin):
    def setUp(self):
        # Call parent setUp explicitly, because of how unittests work.
        RulesTestMixin.setUp(self)
        # Create test coordinate cf variable.
        points = np.arange(6)
        self.cf_coord_var = mock.Mock(
            dimensions=('foo',),
            cf_name='wibble',
            standard_name=None,
            long_name='wibble',
            units='m',
            shape=points.shape,
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key])

    def test_slowest_varying_vertex_dim(self):
        # Create the bounds cf variable.
        bounds = np.arange(12).reshape(2, 6)
        self.cf_bounds_var = mock.Mock(
            dimensions=('nv', 'foo'),
            cf_name='wibble_bnds',
            shape=bounds.shape,
            __getitem__=lambda self, key: bounds[key])

        # Expected bounds on the resulting coordinate should be rolled so that
        # the vertex dimension is at the end.
        expected_bounds = bounds.transpose()
        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=expected_bounds)

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(
                expected_coord, [0])

            # Test that engine.provides container is correctly populated.
            expected_list = [(expected_coord, self.cf_coord_var.cf_name)]
            self.assertEqual(self.engine.provides['coordinates'],
                             expected_list)

    def test_fastest_varying_vertex_dim(self):
        bounds = np.arange(12).reshape(6, 2)
        self.cf_bounds_var = mock.Mock(
            dimensions=('foo', 'nv'),
            cf_name='wibble_bnds',
            shape=bounds.shape,
            __getitem__=lambda self, key: bounds[key])

        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=bounds)

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(
                expected_coord, [0])

            # Test that engine.provides container is correctly populated.
            expected_list = [(expected_coord, self.cf_coord_var.cf_name)]
            self.assertEqual(self.engine.provides['coordinates'],
                             expected_list)

    def test_fastest_with_different_dim_names(self):
        # Despite the dimension names 'x' differing from the coord's
        # which is 'foo' (as permitted by the cf spec),
        # this should still work because the vertex dim is the fastest varying.
        bounds = np.arange(12).reshape(6, 2)
        self.cf_bounds_var = mock.Mock(
            dimensions=('x', 'nv'),
            cf_name='wibble_bnds',
            shape=bounds.shape,
            __getitem__=lambda self, key: bounds[key])

        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=bounds)

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(
                expected_coord, [0])

            # Test that engine.provides container is correctly populated.
            expected_list = [(expected_coord, self.cf_coord_var.cf_name)]
            self.assertEqual(self.engine.provides['coordinates'],
                             expected_list)


class TestCircular(tests.IrisTest, RulesTestMixin):
    # Test the rules logic for marking a coordinate "circular".
    def setUp(self):
        # Call parent setUp explicitly, because of how unittests work.
        RulesTestMixin.setUp(self)
        self.cf_bounds_var = None

    def _make_vars(self, points, bounds=None, units='degrees'):
        points = np.array(points)
        self.cf_coord_var = mock.Mock(
            dimensions=('foo',),
            cf_name='wibble',
            standard_name=None,
            long_name='wibble',
            units=units,
            shape=points.shape,
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key])
        if bounds:
            bounds = np.array(bounds).reshape(
                self.cf_coord_var.shape + (2,))
            self.cf_bounds_var = mock.Mock(
                dimensions=('x', 'nv'),
                cf_name='wibble_bnds',
                shape=bounds.shape,
                __getitem__=lambda self, key: bounds[key])

    def _check_circular(self, circular, *args, **kwargs):
        if 'coord_name' in kwargs:
            coord_name = kwargs.pop('coord_name')
        else:
            coord_name = 'longitude'
        self._make_vars(*args, **kwargs)
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var,
                                       coord_name=coord_name)
            self.assertEqual(self.engine.cube.add_dim_coord.call_count, 1)
            coord, dims = self.engine.cube.add_dim_coord.call_args[0]
        self.assertEqual(coord.circular, circular)

    def check_circular(self, *args, **kwargs):
        self._check_circular(True, *args, **kwargs)

    def check_noncircular(self, *args, **kwargs):
        self._check_circular(False, *args, **kwargs)

    def test_single_zero_noncircular(self):
        self.check_noncircular([0.0])

    def test_single_lt_modulus_noncircular(self):
        self.check_noncircular([-1.0])

    def test_single_eq_modulus_circular(self):
        self.check_circular([360.0])

    def test_single_gt_modulus_circular(self):
        self.check_circular([361.0])

    def test_single_bounded_noncircular(self):
        self.check_noncircular([180.0], bounds=[90.0, 240.0])

    def test_single_bounded_circular(self):
        self.check_circular([180.0], bounds=[90.0, 450.0])

    def test_multiple_unbounded_circular(self):
        self.check_circular([0.0, 90.0, 180.0, 270.0])

    def test_non_angle_noncircular(self):
        points = [0.0, 90.0, 180.0, 270.0]
        self.check_noncircular(points, units='m')

    def test_non_longitude_noncircular(self):
        points = [0.0, 90.0, 180.0, 270.0]
        self.check_noncircular(points, coord_name='depth')

    def test_multiple_unbounded_irregular_noncircular(self):
        self.check_noncircular([0.0, 90.0, 189.999, 270.0])

    def test_multiple_unbounded_offset_circular(self):
        self.check_circular([45.0, 135.0, 225.0, 315.0])

    def test_multiple_unbounded_shortrange_circular(self):
        self.check_circular([0.0, 90.0, 180.0, 269.9999])

    def test_multiple_bounded_circular(self):
        self.check_circular([0.0, 120.3, 240.0],
                            bounds=[[-45.0, 50.0],
                                    [100.0, 175.0],
                                    [200.0, 315.0]])

    def test_multiple_bounded_noncircular(self):
        self.check_noncircular([0.0, 120.3, 240.0],
                               bounds=[[-45.0, 50.0],
                                       [100.0, 175.0],
                                       [200.0, 355.0]])


class TestCircularScalar(tests.IrisTest, RulesTestMixin):
    def setUp(self):
        RulesTestMixin.setUp(self)

    def _make_vars(self, bounds):
        # Create cf vars for the coordinate and its bounds.
        # Note that for a scalar the shape of the array from
        # the cf var is (), rather than (1,).
        points = np.array([0.])
        self.cf_coord_var = mock.Mock(
            dimensions=(),
            cf_name='wibble',
            standard_name=None,
            long_name='wibble',
            units='degrees',
            shape=(),
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key])

        bounds = np.array(bounds)
        self.cf_bounds_var = mock.Mock(
            dimensions=(u'bnds'),
            cf_name='wibble_bnds',
            shape=bounds.shape,
            __getitem__=lambda self, key: bounds[key])

    def _assert_circular(self, value):
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var,
                                       coord_name='longitude')
            self.assertEqual(self.engine.cube.add_aux_coord.call_count, 1)
            coord, dims = self.engine.cube.add_aux_coord.call_args[0]
        self.assertEqual(coord.circular, value)

    def test_two_bounds_noncircular(self):
        self._make_vars([0., 180.])
        self._assert_circular(False)

    def test_two_bounds_circular(self):
        self._make_vars([0., 360.])
        self._assert_circular(True)

    def test_two_bounds_circular_decreasing(self):
        self._make_vars([360., 0.])
        self._assert_circular(True)

    def test_two_bounds_circular_alt(self):
        self._make_vars([-180., 180.])
        self._assert_circular(True)

    def test_two_bounds_circular_alt_decreasing(self):
        self._make_vars([180., -180.])
        self._assert_circular(True)

    def test_four_bounds(self):
        self._make_vars([0., 10., 20., 30.])
        self._assert_circular(False)


if __name__ == '__main__':
    tests.main()
