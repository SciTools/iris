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
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.build_dimension_coordinate`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import warnings

import numpy as np
import mock

from iris.coords import AuxCoord, DimCoord
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    build_dimension_coordinate


class _CommonMockTemplate(tests.IrisTest):
    def setUp(self):
        # Create coordinate cf variables and pyke engine.
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

        self.engine = mock.Mock(
            cube=mock.Mock(),
            cf_var=mock.Mock(dimensions=('foo', 'bar')),
            filename='DUMMY',
            provides=dict(coordinates=[]))

        # Create patch for deferred loading that prevents attempted
        # file access. This assumes that self.cf_bounds_var is
        # defined in the test case.
        def patched__getitem__(proxy_self, keys):
            variable = None
            for var in (self.cf_coord_var, self.cf_bounds_var):
                if proxy_self.variable_name == var.cf_name:
                    return var[keys]
            raise RuntimeError()

        self.deferred_load_patch = mock.patch(
            'iris.fileformats.netcdf.NetCDFDataProxy.__getitem__',
            new=patched__getitem__)


class TestCoordConstruction(_CommonMockTemplate):
    # Ensure that the coordinates constructed have all the relevant cf
    # attributes populated.
    def test_dim_coord_construction(self):
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

        get_cf_bounds_var_patch = mock.patch(
            'iris.fileformats._pyke_rules.compiled_krb.'
            'fc_rules_cf_fc.get_cf_bounds_var',
            return_value=self.cf_bounds_var)

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(
                expected_coord, [0])

    def test_aux_coord_construction(self):
        # Use non monotonically increasing coordinates to force aux coord
        # construction.
        points = np.array([1, 3, 2, 4, 6, 5])
        self.cf_coord_var.__getitem__ = lambda self, key: points[key]
        bounds = np.arange(12).reshape(6, 2)
        self.cf_bounds_var = mock.Mock(
            dimensions=('x', 'nv'),
            cf_name='wibble_bnds',
            shape=bounds.shape,
            __getitem__=lambda self, key: bounds[key])

        expected_coord = AuxCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=bounds)

        get_cf_bounds_var_patch = mock.patch(
            'iris.fileformats._pyke_rules.compiled_krb.'
            'fc_rules_cf_fc.get_cf_bounds_var',
            return_value=self.cf_bounds_var)
        warning_patch = mock.patch('warnings.warn')

        # Asserts must lie within context manager because of deferred loading.
        with warning_patch, self.deferred_load_patch, get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_aux_coord.assert_called_with(
                expected_coord, [0])
            self.assertIn("creating 'wibble' auxiliary coordinate instead",
                          warnings.warn.call_args[0][0])


class TestBoundsVertexDim(_CommonMockTemplate):
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

        # Patch the helper function that retrieves the bounds cf variable.
        # This avoids the need for setting up further mocking of cf objects.
        get_cf_bounds_var_patch = mock.patch(
            'iris.fileformats._pyke_rules.compiled_krb.'
            'fc_rules_cf_fc.get_cf_bounds_var',
            return_value=self.cf_bounds_var)

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, get_cf_bounds_var_patch:
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

        get_cf_bounds_var_patch = mock.patch(
            'iris.fileformats._pyke_rules.compiled_krb.'
            'fc_rules_cf_fc.get_cf_bounds_var',
            return_value=self.cf_bounds_var)

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, get_cf_bounds_var_patch:
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

        get_cf_bounds_var_patch = mock.patch(
            'iris.fileformats._pyke_rules.compiled_krb.'
            'fc_rules_cf_fc.get_cf_bounds_var',
            return_value=self.cf_bounds_var)

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, get_cf_bounds_var_patch:
            build_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(
                expected_coord, [0])

            # Test that engine.provides container is correctly populated.
            expected_list = [(expected_coord, self.cf_coord_var.cf_name)]
            self.assertEqual(self.engine.provides['coordinates'],
                             expected_list)


if __name__ == '__main__':
    tests.main()
