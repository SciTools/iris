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
"""Unit tests for the `iris.fileformats.netcdf._load_aux_factory` function."""

from __future__ import (absolute_import, division, print_function)

import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np
import warnings

from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.netcdf import _load_aux_factory


class TestAtmosphereHybridSigmaPressureCoordinate(tests.IrisTest):
    def setUp(self):
        standard_name = 'atmosphere_hybrid_sigma_pressure_coordinate'
        self.requires = dict(formula_type=standard_name)
        coordinates = [(mock.sentinel.b, 'b'), (mock.sentinel.ps, 'ps')]
        self.provides = dict(coordinates=coordinates)
        self.engine = mock.Mock(requires=self.requires, provides=self.provides)
        self.cube = mock.create_autospec(Cube, spec_set=True, instance=True)
        # Patch out the check_dependencies functionality.
        func = 'iris.aux_factory.HybridPressureFactory._check_dependencies'
        patcher = mock.patch(func)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_formula_terms_ap(self):
        self.provides['coordinates'].append((mock.sentinel.ap, 'ap'))
        self.requires['formula_terms'] = dict(ap='ap', b='b', ps='ps')
        _load_aux_factory(self.engine, self.cube)
        # Check cube.add_aux_coord method.
        self.assertEqual(self.cube.add_aux_coord.call_count, 0)
        # Check cube.add_aux_factory method.
        self.assertEqual(self.cube.add_aux_factory.call_count, 1)
        args, _ = self.cube.add_aux_factory.call_args
        self.assertEqual(len(args), 1)
        factory = args[0]
        self.assertEqual(factory.delta, mock.sentinel.ap)
        self.assertEqual(factory.sigma, mock.sentinel.b)
        self.assertEqual(factory.surface_air_pressure, mock.sentinel.ps)

    def test_formula_terms_a_p0(self):
        coord_a = DimCoord(range(5), units='Pa')
        coord_p0 = DimCoord(10, units='1')
        coord_expected = DimCoord(np.arange(5) * 10, units='Pa',
                                  long_name='vertical pressure', var_name='ap')
        self.provides['coordinates'].extend([(coord_a, 'a'), (coord_p0, 'p0')])
        self.requires['formula_terms'] = dict(a='a', b='b', ps='ps', p0='p0')
        _load_aux_factory(self.engine, self.cube)
        # Check cube.coord_dims method.
        self.assertEqual(self.cube.coord_dims.call_count, 1)
        args, _ = self.cube.coord_dims.call_args
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], coord_a)
        # Check cube.add_aux_coord method.
        self.assertEqual(self.cube.add_aux_coord.call_count, 1)
        args, _ = self.cube.add_aux_coord.call_args
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0], coord_expected)
        self.assertIsInstance(args[1], mock.Mock)
        # Check cube.add_aux_factory method.
        self.assertEqual(self.cube.add_aux_factory.call_count, 1)
        args, _ = self.cube.add_aux_factory.call_args
        self.assertEqual(len(args), 1)
        factory = args[0]
        self.assertEqual(factory.delta, coord_expected)
        self.assertEqual(factory.sigma, mock.sentinel.b)
        self.assertEqual(factory.surface_air_pressure, mock.sentinel.ps)

    def test_formula_terms_p0_non_scalar(self):
        coord_p0 = DimCoord(range(5))
        self.provides['coordinates'].append((coord_p0, 'p0'))
        self.requires['formula_terms'] = dict(p0='p0')
        with self.assertRaises(ValueError):
            _load_aux_factory(self.engine, self.cube)

    def test_formula_terms_p0_bounded(self):
        coord_a = DimCoord(range(5))
        coord_p0 = DimCoord(1, bounds=[0, 2], var_name='p0')
        self.provides['coordinates'].extend([(coord_a, 'a'), (coord_p0, 'p0')])
        self.requires['formula_terms'] = dict(a='a', b='b', ps='ps', p0='p0')
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            _load_aux_factory(self.engine, self.cube)
            self.assertEqual(len(warn), 1)
            msg = 'Ignoring atmosphere hybrid sigma pressure scalar ' \
                'coordinate {!r} bounds.'.format(coord_p0.name())
            self.assertEqual(msg, warn[0].message.message)

    def test_formula_terms_ap_missing_coords(self):
        coordinates = [(mock.sentinel.b, 'b'), (mock.sentinel.ps, 'ps')]
        self.provides = dict(coordinates=coordinates)
        self.requires['formula_terms'] = dict(ap='ap', b='b', ps='ps')
        with mock.patch('warnings.warn') as warn:
            _load_aux_factory(self.engine, self.cube)
        warn.assert_called_once_with("Unable to find coordinate for variable "
                                     "'ap'")

if __name__ == '__main__':
    tests.main()
