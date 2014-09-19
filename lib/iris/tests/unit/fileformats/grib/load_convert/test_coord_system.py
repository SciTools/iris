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
Test function :func:`iris.fileformats.grib._load_convert.coord_system.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import iris
import iris.coord_systems as icoord_systems
from iris.exceptions import TranslationError
from iris.fileformats.grib._load_convert import coord_system


# Reference GRIB2 Code Table 3.2 - Shape of the Earth.


class Test(tests.IrisTest):
    def test_shape_invalid(self):
        shape = 10
        emsg = 'invalid shape of the earth'
        with self.assertRaisesRegexp(TranslationError, emsg):
            coord_system(shape)

    def test_shape_unsupported(self):
        unsupported = [1, 2, 3, 4, 5, 7, 8, 9]
        emsg = 'unsupported shape of the earth'
        for shape in unsupported:
            with self.assertRaisesRegexp(TranslationError, emsg):
                coord_system(shape)

    def test_shape_supported(self):
        cs_by_shape = {0: icoord_systems.GeogCS(6367470),
                       6: icoord_systems.GeogCS(6371229),
                       }
        for shape, expected in cs_by_shape.items():
            self.assertEqual(coord_system(shape), expected)


if __name__ == '__main__':
    tests.main()
