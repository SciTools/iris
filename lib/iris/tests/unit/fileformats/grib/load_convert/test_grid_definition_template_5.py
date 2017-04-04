# (C) British Crown Copyright 2014 - 2017, Met Office
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
Test function
:func:`iris.fileformats.grib._load_convert.grid_definition_template_5`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris.tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import deepcopy
import mock

from iris.fileformats.grib._load_convert import grid_definition_template_5


class Test(tests.IrisGribTest):
    def setUp(self):
        def func(s, m, y, x, c):
            return m['dim_coords_and_dims'].append(item)

        module = 'iris.fileformats.grib._load_convert'

        self.major = mock.sentinel.major
        self.minor = mock.sentinel.minor
        self.radius = mock.sentinel.radius

        mfunc = '{}.ellipsoid_geometry'.format(module)
        return_value = (self.major, self.minor, self.radius)
        self.patch(mfunc, return_value=return_value)

        mfunc = '{}.ellipsoid'.format(module)
        self.ellipsoid = mock.sentinel.ellipsoid
        self.patch(mfunc, return_value=self.ellipsoid)

        mfunc = '{}.grid_definition_template_4_and_5'.format(module)
        self.coord = mock.sentinel.coord
        self.dim = mock.sentinel.dim
        item = (self.coord, self.dim)
        self.patch(mfunc, side_effect=func)

        mclass = 'iris.coord_systems.RotatedGeogCS'
        self.cs = mock.sentinel.cs
        self.patch(mclass, return_value=self.cs)

        self.metadata = {'factories': [], 'references': [],
                         'standard_name': None,
                         'long_name': None, 'units': None, 'attributes': {},
                         'cell_methods': [], 'dim_coords_and_dims': [],
                         'aux_coords_and_dims': []}

    def test(self):
        metadata = deepcopy(self.metadata)
        angleOfRotation = mock.sentinel.angleOfRotation
        shapeOfTheEarth = mock.sentinel.shapeOfTheEarth
        section = {'latitudeOfSouthernPole': 45000000,
                   'longitudeOfSouthernPole': 90000000,
                   'angleOfRotation': angleOfRotation,
                   'shapeOfTheEarth': shapeOfTheEarth}
        # The called being tested.
        grid_definition_template_5(section, metadata)
        from iris.fileformats.grib._load_convert import \
            ellipsoid_geometry, \
            ellipsoid, \
            grid_definition_template_4_and_5 as gdt_4_5
        self.assertEqual(ellipsoid_geometry.call_count, 1)
        ellipsoid.assert_called_once_with(shapeOfTheEarth, self.major,
                                          self.minor, self.radius)
        from iris.coord_systems import RotatedGeogCS
        RotatedGeogCS.assert_called_once_with(-45.0, 270.0, angleOfRotation,
                                              self.ellipsoid)
        gdt_4_5.assert_called_once_with(section, metadata, 'grid_latitude',
                                        'grid_longitude', self.cs)
        expected = deepcopy(self.metadata)
        expected['dim_coords_and_dims'].append((self.coord, self.dim))
        self.assertEqual(metadata, expected)


if __name__ == '__main__':
    tests.main()
