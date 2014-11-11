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
Unit tests for :class:`iris.experimental.um.Field`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import mock

from iris.experimental.um import Field


class Test_read_data(tests.IrisTest):
    def test_None(self):
        field = Field([], [], None)
        self.assertIsNone(field.read_data())

    def test_provider(self):
        provider = mock.Mock(read_data=lambda field: mock.sentinel.DATA)
        field = Field([], [], provider)
        self.assertIs(field.read_data(), mock.sentinel.DATA)


if __name__ == '__main__':
    tests.main()
