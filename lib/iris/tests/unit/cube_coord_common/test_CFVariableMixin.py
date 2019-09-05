# (C) British Crown Copyright 2019, Met Office
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
Unit tests for the :class:`iris._cube_coord_common.CFVariableMixin`.
"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris._cube_coord_common import CFVariableMixin


class Test_standard_name__setter(tests.IrisTest):
    def test_valid_standard_name(self):
        cf_var = CFVariableMixin()
        cf_var.standard_name = 'air_temperature'

    def test_invalid_standard_name(self):
        cf_var = CFVariableMixin()
        emsg = "'not_a_standard_name' is not a valid standard_name"
        with self.assertRaisesRegexp(ValueError, emsg):
            cf_var.standard_name = 'not_a_standard_name'

    def test_none_standard_name(self):
        cf_var = CFVariableMixin()
        cf_var.standard_name = None


if __name__ == '__main__':
    tests.main()
