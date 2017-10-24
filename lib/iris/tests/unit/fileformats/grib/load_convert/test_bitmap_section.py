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
Test function :func:`iris.fileformats.grib._load_convert.bitmap_section.`

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.exceptions import TranslationError
from iris.fileformats.grib._load_convert import bitmap_section
from iris.tests.unit.fileformats.grib import _make_test_message


class Test(tests.IrisTest):
    def test_bitmap_unsupported(self):
        # bitMapIndicator in range 1-254.
        # Note that bitMapIndicator = 1-253 and bitMapIndicator = 254 mean two
        # different things, but load_convert treats them identically.
        message = _make_test_message({6: {'bitMapIndicator': 100,
                                          'bitmap': None}})
        with self.assertRaisesRegexp(TranslationError, 'unsupported bitmap'):
            bitmap_section(message.sections[6])


if __name__ == '__main__':
    tests.main()
