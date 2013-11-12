# (C) British Crown Copyright 2013, Met Office
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
Unit tests for :func:`iris.analysis.name_loaders._generate_cubes`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris.cube
from iris.fileformats.name_loaders import _build_cell_methods


class Tests(tests.IrisTest):
    def setUp(self):
        self.mock_cell = mock.patch('iris.coords.CellMethod')
        self.mock_cell.start()
        self.addCleanup(self.mock_cell.stop)

    def test_nameII_average(self):
        av_or_int = ['something average ob bla'] * 3
        res = _build_cell_methods(av_or_int)
        iris.coords.CellMethod.assert_called('average', 'time')

    def test_nameIII_averaged(self):
        av_or_int = ['something averaged ob bla'] * 3
        res = _build_cell_methods(av_or_int)
        iris.coords.CellMethod.assert_called('average', 'time')

    def test_nameII_integral(self):
        av_or_int = ['something integral ob bla'] * 3
        res = _build_cell_methods(av_or_int)
        iris.coords.CellMethod.assert_called('sum', 'time')

    def test_nameIII_integrated(self):
        av_or_int = ['something integrated ob bla'] * 3
        res = _build_cell_methods(av_or_int)
        iris.coords.CellMethod.assert_called('sum', 'time')

    def test_unrecognised(self):
        av_or_int = ['something average', 'something integral', 'bla else']
        with self.assertRaises(iris.exceptions.TranslationError):
            res = _build_cell_methods(av_or_int)


if __name__ == "__main__":
    tests.main()
