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
Test function :func:`iris.fileformats.grib._load_convert.vertical_coords`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import deepcopy
import mock

from iris.coords import DimCoord
from iris.exceptions import TranslationError
from iris.fileformats.grib._load_convert import vertical_coords

from iris.fileformats.grib._load_convert import \
    _TYPE_OF_FIXED_SURFACE_MISSING as MISSING_SURFACE, \
    _MDI as MISSING_LEVEL


class Test(tests.IrisTest):
    def setUp(self):
        self.metadata = {'factories': [], 'references': [],
                         'standard_name': None,
                         'long_name': None, 'units': None, 'attributes': {},
                         'cell_methods': [], 'dim_coords_and_dims': [],
                         'aux_coords_and_dims': []}

    def test_hybrid_factories(self):
        metadata = deepcopy(self.metadata)
        section = {'NV': 1}
        this = 'iris.fileformats.grib._load_convert.hybrid_factories'
        factory = mock.sentinel.factory
        func = lambda section, metadata: metadata['factories'].append(factory)
        with mock.patch(this, side_effect=func) as hybrid_factories:
            vertical_coords(section, metadata)
            self.assertTrue(hybrid_factories.called)
            self.assertEqual(metadata['factories'], [factory])

    def test_no_first_fixed_surface(self):
        metadata = deepcopy(self.metadata)
        section = {'NV': 0,
                   'typeOfFirstFixedSurface': MISSING_SURFACE,
                   'scaledValueOfFirstFixedSurface': MISSING_LEVEL}
        vertical_coords(section, metadata)
        self.assertEqual(metadata, self.metadata)

    def _check(self, value, msg):
        this = 'iris.fileformats.grib._load_convert.options'
        with mock.patch('warnings.warn') as warn:
            with mock.patch(this) as options:
                for request_warning in [False, True]:
                    options.warn_on_unsupported = request_warning
                    metadata = deepcopy(self.metadata)
                    section = {'NV': 0,
                               'typeOfFirstFixedSurface': 0,
                               'scaledValueOfFirstFixedSurface': value}
                    # The call being tested.
                    vertical_coords(section, metadata)
                    self.assertEqual(metadata, self.metadata)
                    if request_warning:
                        self.assertEqual(len(warn.mock_calls), 1)
                        args, _ = warn.call_args
                        self.assertIn(msg, args[0])
                    else:
                        self.assertEqual(len(warn.mock_calls), 0)

    def test_unknown_first_fixed_surface_with_missing_scaled_value(self):
        self._check(MISSING_LEVEL, 'surface with missing scaled value')

    def test_unknown_first_fixed_surface_with_scaled_value(self):
        self._check(0, 'surface with scaled value')

    def test_pressure_with_no_second_fixed_surface(self):
        metadata = deepcopy(self.metadata)
        section = {'NV': 0,
                   'typeOfFirstFixedSurface': 100,  # pressure / Pa
                   'scaledValueOfFirstFixedSurface': 10,
                   'scaleFactorOfFirstFixedSurface': 1,
                   'typeOfSecondFixedSurface': MISSING_SURFACE}
        vertical_coords(section, metadata)
        coord = DimCoord(1.0, long_name='pressure', units='Pa')
        expected = deepcopy(self.metadata)
        expected['aux_coords_and_dims'].append((coord, None))
        self.assertEqual(metadata, expected)

    def test_height_with_no_second_fixed_surface(self):
        metadata = deepcopy(self.metadata)
        section = {'NV': 0,
                   'typeOfFirstFixedSurface': 103,  # height / m
                   'scaledValueOfFirstFixedSurface': 100,
                   'scaleFactorOfFirstFixedSurface': 2,
                   'typeOfSecondFixedSurface': MISSING_SURFACE}
        vertical_coords(section, metadata)
        coord = DimCoord(1.0, long_name='height', units='m')
        expected = deepcopy(self.metadata)
        expected['aux_coords_and_dims'].append((coord, None))
        self.assertEqual(metadata, expected)

    def test_different_fixed_surfaces(self):
        section = {'NV': 0,
                   'typeOfFirstFixedSurface': 100,
                   'scaledValueOfFirstFixedSurface': None,
                   'scaleFactorOfFirstFixedSurface': None,
                   'typeOfSecondFixedSurface': 0}
        emsg = 'different types of first and second fixed surface'
        with self.assertRaisesRegexp(TranslationError, emsg):
            vertical_coords(section, None)

    def test_same_fixed_surfaces_missing_second_scaled_value(self):
        section = {'NV': 0,
                   'typeOfFirstFixedSurface': 100,
                   'scaledValueOfFirstFixedSurface': None,
                   'scaleFactorOfFirstFixedSurface': None,
                   'typeOfSecondFixedSurface': 100,
                   'scaledValueOfSecondFixedSurface': MISSING_LEVEL}
        emsg = 'missing scaled value of second fixed surface'
        with self.assertRaisesRegexp(TranslationError, emsg):
            vertical_coords(section, None)

    def test_pressure_with_second_fixed_surface(self):
        metadata = deepcopy(self.metadata)
        section = {'NV': 0,
                   'typeOfFirstFixedSurface': 100,
                   'scaledValueOfFirstFixedSurface': 10,
                   'scaleFactorOfFirstFixedSurface': 1,
                   'typeOfSecondFixedSurface': 100,
                   'scaledValueOfSecondFixedSurface': 30,
                   'scaleFactorOfSecondFixedSurface': 1}
        vertical_coords(section, metadata)
        coord = DimCoord(2.0, long_name='pressure', units='Pa',
                         bounds=[1.0, 3.0])
        expected = deepcopy(self.metadata)
        expected['aux_coords_and_dims'].append((coord, None))
        self.assertEqual(metadata, expected)

    def test_height_with_second_fixed_surface(self):
        metadata = deepcopy(self.metadata)
        section = {'NV': 0,
                   'typeOfFirstFixedSurface': 103,
                   'scaledValueOfFirstFixedSurface': 1000,
                   'scaleFactorOfFirstFixedSurface': 2,
                   'typeOfSecondFixedSurface': 103,
                   'scaledValueOfSecondFixedSurface': 3000,
                   'scaleFactorOfSecondFixedSurface': 2}
        vertical_coords(section, metadata)
        coord = DimCoord(20.0, long_name='height', units='m',
                         bounds=[10.0, 30.0])
        expected = deepcopy(self.metadata)
        expected['aux_coords_and_dims'].append((coord, None))
        self.assertEqual(metadata, expected)


if __name__ == '__main__':
    tests.main()
