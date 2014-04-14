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
"""Unit tests for :func:`iris.fileformats.pp_rules._model_level_number`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.fileformats.pp_rules import _model_level_number


class Test_9999(tests.IrisTest):
    def test(self):
        field = mock.Mock(lblev=9999)
        self.assertEqual(_model_level_number(field), 0)


class Test_lblev(tests.IrisTest):
    def test(self):
        for val in xrange(9999):
            field = mock.Mock(lblev=val)
            self.assertEqual(_model_level_number(field), val)


if __name__ == "__main__":
    tests.main()
