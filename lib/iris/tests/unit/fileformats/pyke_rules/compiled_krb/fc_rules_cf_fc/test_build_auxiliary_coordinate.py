# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.build_auxilliary_coordinate`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

from unittest import mock

import numpy as np

from iris.coords import AuxCoord
from iris.fileformats.cf import CFVariable
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    build_auxiliary_coordinate


# from iris.tests.unit.fileformats.pyke_rules.compiled_krb\
#     .fc_rules_cf_fc.test_build_dimension_coordinate import RulesTestMixin

class TestBoundsVertexDim(tests.IrisTest):
    # Lookup for various tests (which change the dimension order).
    dim_names_lens = {
        'foo': 2, 'bar': 3, 'nv': 4,
        # 'x' and 'y' used as aliases for 'foo' and 'bar'
        'x': 2, 'y': 3}

    def setUp(self):
        # Create coordinate cf variables and pyke engine.
        dimension_names = ('foo', 'bar')
        points, cf_data = self._make_array_and_cf_data(dimension_names)
        self.cf_coord_var = mock.Mock(
            spec=CFVariable,
            dimensions=dimension_names,
            cf_name='wibble',
            cf_data=cf_data,
            standard_name=None,
            long_name='wibble',
            units='m',
            shape=points.shape,
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key])

        expected_bounds, _ = self._make_array_and_cf_data(
            dimension_names=('foo', 'bar', 'nv'))
        self.expected_coord = AuxCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=expected_bounds)

        self.engine = mock.Mock(
            cube=mock.Mock(),
            cf_var=mock.Mock(dimensions=('foo', 'bar'),
                             cf_data=cf_data),
            filename='DUMMY',
            cube_parts=dict(coordinates=[]))

        # Patch the deferred loading that prevents attempted file access.
        # This assumes that self.cf_bounds_var is defined in the test case.
        def patched__getitem__(proxy_self, keys):
            for var in (self.cf_coord_var, self.cf_bounds_var):
                if proxy_self.variable_name == var.cf_name:
                    return var[keys]
            raise RuntimeError()

        self.patch('iris.fileformats.netcdf.NetCDFDataProxy.__getitem__',
                   new=patched__getitem__)

        # Patch the helper function that retrieves the bounds cf variable,
        # and a False flag for climatological.
        # This avoids the need for setting up further mocking of cf objects.
        def _get_per_test_bounds_var(_coord_unused):
            # Return the 'cf_bounds_var' created by the current test.
            return (self.cf_bounds_var, False)

        self.patch('iris.fileformats._pyke_rules.compiled_krb.'
                   'fc_rules_cf_fc.get_cf_bounds_var',
                   new=_get_per_test_bounds_var)

    @classmethod
    def _make_array_and_cf_data(cls, dimension_names):
        shape = tuple(cls.dim_names_lens[name]
                      for name in dimension_names)
        cf_data = mock.MagicMock(_FillValue=None, spec=[])
        cf_data.chunking = mock.MagicMock(return_value=shape)
        return np.zeros(shape), cf_data

    def _make_cf_bounds_var(self, dimension_names):
        # Create the bounds cf variable.
        bounds, cf_data = self._make_array_and_cf_data(dimension_names)
        cf_bounds_var = mock.Mock(
            spec=CFVariable,
            dimensions=dimension_names,
            cf_name='wibble_bnds',
            cf_data=cf_data,
            shape=bounds.shape,
            dtype=bounds.dtype,
            __getitem__=lambda self, key: bounds[key])

        return bounds, cf_bounds_var

    def _check_case(self, dimension_names):
        bounds, self.cf_bounds_var = self._make_cf_bounds_var(
            dimension_names=dimension_names)

        # Asserts must lie within context manager because of deferred loading.
        build_auxiliary_coordinate(self.engine, self.cf_coord_var)

        # Test that expected coord is built and added to cube.
        self.engine.cube.add_aux_coord.assert_called_with(
            self.expected_coord, [0, 1])

        # Test that engine.cube_parts container is correctly populated.
        expected_list = [(self.expected_coord, self.cf_coord_var.cf_name)]
        self.assertEqual(self.engine.cube_parts['coordinates'],
                         expected_list)

    def test_fastest_varying_vertex_dim(self):
        # The usual order.
        self._check_case(dimension_names=('foo', 'bar', 'nv'))

    def test_slowest_varying_vertex_dim(self):
        # Bounds in the first (slowest varying) dimension.
        self._check_case(dimension_names=('nv', 'foo', 'bar'))

    def test_fastest_with_different_dim_names(self):
        # Despite the dimension names ('x', and 'y') differing from the coord's
        # which are 'foo' and 'bar' (as permitted by the cf spec),
        # this should still work because the vertex dim is the fastest varying.
        self._check_case(dimension_names=('x', 'y', 'nv'))


class TestDtype(tests.IrisTest):
    def setUp(self):
        # Create coordinate cf variables and pyke engine.
        points = np.arange(6).reshape(2, 3)
        cf_data = mock.MagicMock(_FillValue=None)
        cf_data.chunking = mock.MagicMock(return_value=points.shape)

        self.cf_coord_var = mock.Mock(
            spec=CFVariable,
            dimensions=('foo', 'bar'),
            cf_name='wibble',
            cf_data=cf_data,
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
            cube_parts=dict(coordinates=[]))

        def patched__getitem__(proxy_self, keys):
            if proxy_self.variable_name == self.cf_coord_var.cf_name:
                return self.cf_coord_var[keys]
            raise RuntimeError()

        self.deferred_load_patch = mock.patch(
            'iris.fileformats.netcdf.NetCDFDataProxy.__getitem__',
            new=patched__getitem__)

    def test_scale_factor_add_offset_int(self):
        self.cf_coord_var.scale_factor = 3
        self.cf_coord_var.add_offset = 5

        with self.deferred_load_patch:
            build_auxiliary_coordinate(self.engine, self.cf_coord_var)

        coord, _ = self.engine.cube_parts['coordinates'][0]
        self.assertEqual(coord.dtype.kind, 'i')

    def test_scale_factor_float(self):
        self.cf_coord_var.scale_factor = 3.

        with self.deferred_load_patch:
            build_auxiliary_coordinate(self.engine, self.cf_coord_var)

        coord, _ = self.engine.cube_parts['coordinates'][0]
        self.assertEqual(coord.dtype.kind, 'f')

    def test_add_offset_float(self):
        self.cf_coord_var.add_offset = 5.

        with self.deferred_load_patch:
            build_auxiliary_coordinate(self.engine, self.cf_coord_var)

        coord, _ = self.engine.cube_parts['coordinates'][0]
        self.assertEqual(coord.dtype.kind, 'f')


class TestCoordConstruction(tests.IrisTest):
    def setUp(self):
        # Create dummy pyke engine.
        self.engine = mock.Mock(
            cube=mock.Mock(),
            cf_var=mock.Mock(dimensions=('foo', 'bar')),
            filename='DUMMY',
            cube_parts=dict(coordinates=[]))

        points = np.arange(6)
        self.cf_coord_var = mock.Mock(
            dimensions=('foo',),
            scale_factor=1,
            add_offset=0,
            cf_name='wibble',
            cf_data=mock.MagicMock(chunking=mock.Mock(return_value=None), spec=[]),
            standard_name=None,
            long_name='wibble',
            units='days since 1970-01-01',
            calendar=None,
            shape=points.shape,
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key])

        bounds = np.arange(12).reshape(6, 2)
        self.cf_bounds_var = mock.Mock(
            dimensions=('x', 'nv'),
            scale_factor=1,
            add_offset=0,
            cf_name='wibble_bnds',
            cf_data=mock.MagicMock(chunking=mock.Mock(return_value=None)),
            shape=bounds.shape,
            dtype=bounds.dtype,
            __getitem__=lambda self, key: bounds[key])
        self.bounds = bounds

        # Create patch for deferred loading that prevents attempted
        # file access. This assumes that self.cf_coord_var and
        # self.cf_bounds_var are defined in the test case.
        def patched__getitem__(proxy_self, keys):
            for var in (self.cf_coord_var, self.cf_bounds_var):
                if proxy_self.variable_name == var.cf_name:
                    return var[keys]
            raise RuntimeError()

        self.patch('iris.fileformats.netcdf.NetCDFDataProxy.__getitem__',
                   new=patched__getitem__)

        # Patch the helper function that retrieves the bounds cf variable.
        # This avoids the need for setting up further mocking of cf objects.
        self.use_climatology_bounds = False  # Set this when you need to.

        def get_cf_bounds_var(coord_var):
            return self.cf_bounds_var, self.use_climatology_bounds

        self.patch('iris.fileformats._pyke_rules.compiled_krb.'
                   'fc_rules_cf_fc.get_cf_bounds_var',
                   new=get_cf_bounds_var)

    def check_case_aux_coord_construction(self, climatology=False):
        # Test a generic auxiliary coordinate, with or without
        # a climatological coord.
        self.use_climatology_bounds = climatology

        expected_coord = AuxCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=self.bounds,
            climatological=climatology)

        build_auxiliary_coordinate(self.engine, self.cf_coord_var)

        # Test that expected coord is built and added to cube.
        self.engine.cube.add_aux_coord.assert_called_with(
            expected_coord, [0])

    def test_aux_coord_construction(self):
        self.check_case_aux_coord_construction(climatology=False)

    def test_aux_coord_construction__climatology(self):
        self.check_case_aux_coord_construction(climatology=True)


if __name__ == '__main__':
    tests.main()
