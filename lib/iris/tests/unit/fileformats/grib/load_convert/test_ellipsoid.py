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
Test function :func:`iris.fileformats.grib._load_convert.ellipsoid.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy.ma as ma

import iris.coord_systems as icoord_systems
from iris.exceptions import TranslationError
from iris.fileformats.grib._load_convert import ellipsoid


# Reference GRIB2 Code Table 3.2 - Shape of the Earth.


MDI = ma.masked


class Test(tests.IrisTest):
    def test_shape_unsupported(self):
        unsupported = [2, 4, 5, 8, 9, 10, MDI]
        emsg = 'unsupported shape of the earth'
        for shape in unsupported:
            with self.assertRaisesRegexp(TranslationError, emsg):
                ellipsoid(shape, MDI, MDI, MDI)

    def test_spherical_default_supported(self):
        cs_by_shape = {0: icoord_systems.GeogCS(6367470),
                       6: icoord_systems.GeogCS(6371229)}
        for shape, expected in cs_by_shape.items():
            result = ellipsoid(shape, MDI, MDI, MDI)
            self.assertEqual(result, expected)

    def test_spherical_shape_1_no_radius(self):
        shape = 1
        emsg = 'radius to be specified'
        with self.assertRaisesRegexp(ValueError, emsg):
            ellipsoid(shape, MDI, MDI, MDI)

    def test_spherical_shape_1(self):
        shape = 1
        radius = 10
        result = ellipsoid(shape, MDI, MDI, radius)
        expected = icoord_systems.GeogCS(radius)
        self.assertEqual(result, expected)

    def test_oblate_shape_3_7_no_axes(self):
        for shape in [3, 7]:
            emsg = 'axis to be specified'
            with self.assertRaisesRegexp(ValueError, emsg):
                ellipsoid(shape, MDI, MDI, MDI)

    def test_oblate_shape_3_7_no_major(self):
        for shape in [3, 7]:
            emsg = 'major axis to be specified'
            with self.assertRaisesRegexp(ValueError, emsg):
                ellipsoid(shape, MDI, 1, MDI)

    def test_oblate_shape_3_7_no_minor(self):
        for shape in [3, 7]:
            emsg = 'minor axis to be specified'
            with self.assertRaisesRegexp(ValueError, emsg):
                ellipsoid(shape, 1, MDI, MDI)

    def test_oblate_shape_3_7(self):
        for shape in [3, 7]:
            major, minor = 1, 10
            scale = 1
            result = ellipsoid(shape, major, minor, MDI)
            if shape == 3:
                # Convert km to m.
                scale = 1000
            expected = icoord_systems.GeogCS(major * scale, minor * scale)
            self.assertEqual(result, expected)


if __name__ == '__main__':
    tests.main()
