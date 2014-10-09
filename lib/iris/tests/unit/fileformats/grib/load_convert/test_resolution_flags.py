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
Test function :func:`iris.fileformats.grib._load_convert.resolution_flags.`

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.fileformats.grib._load_convert import \
    resolution_flags, ResolutionFlags


class Test(tests.IrisTest):
    def test_unset(self):
        expected = ResolutionFlags(False, False, False)
        self.assertEqual(resolution_flags(0x0), expected)

    def test_i_increments_given(self):
        expected = ResolutionFlags(True, False, False)
        self.assertEqual(resolution_flags(0x20), expected)

    def test_j_increments_given(self):
        expected = ResolutionFlags(False, True, False)
        self.assertEqual(resolution_flags(0x10), expected)

    def test_uv_resolved(self):
        expected = ResolutionFlags(False, False, True)
        self.assertEqual(resolution_flags(0x08), expected)


if __name__ == '__main__':
    tests.main()
