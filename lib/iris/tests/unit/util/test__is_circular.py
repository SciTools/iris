# (C) British Crown Copyright 2016, Met Office
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
"""Test function :func:`iris.util._is_circular`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.util import _is_circular


class Test(tests.IrisTest):
    def test_simple(self):
        data = np.arange(12) * 30
        self.assertTrue(_is_circular(data, 360))

    def test_negative_diff(self):
        data = (np.arange(96) * -3.749998) + 3.56249908e+02
        self.assertTrue(_is_circular(data, 360))


if __name__ == '__main__':
    tests.main()
