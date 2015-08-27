# (C) British Crown Copyright 2013 - 2015, Met Office
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
Unit tests for :func:`iris.fileformats.name_loaders._build_cell_methods`.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris.coords

from iris.fileformats.name_loaders import _build_cell_methods


class Tests(tests.IrisTest):
    def test_nameII_average(self):
        av_or_int = ['something average ob bla']
        coord_name = 'foo'
        res = _build_cell_methods(av_or_int, coord_name)
        self.assertEqual(res, [iris.coords.CellMethod('mean', 'foo')])

    def test_nameIII_averaged(self):
        av_or_int = ['something averaged ob bla']
        coord_name = 'bar'
        res = _build_cell_methods(av_or_int, coord_name)
        self.assertEqual(res, [iris.coords.CellMethod('mean', 'bar')])

    def test_nameII_integral(self):
        av_or_int = ['something integral ob bla']
        coord_name = 'ensemble'
        res = _build_cell_methods(av_or_int, coord_name)
        self.assertEqual(res, [iris.coords.CellMethod('sum', 'ensemble')])

    def test_nameIII_integrated(self):
        av_or_int = ['something integrated ob bla']
        coord_name = 'time'
        res = _build_cell_methods(av_or_int, coord_name)
        self.assertEqual(res, [iris.coords.CellMethod('sum', 'time')])

    def test_no_averaging(self):
        av_or_int = ['No foo averaging',
                     'No bar averaging',
                     'no',
                     '',
                     'no averaging',
                     'no anything at all averaging']
        coord_name = 'time'
        res = _build_cell_methods(av_or_int, coord_name)
        self.assertEqual(res, [None] * len(av_or_int))

    def test_nameII_mixed(self):
        av_or_int = ['something integral ob bla',
                     'no averaging',
                     'other average']
        coord_name = 'ensemble'
        res = _build_cell_methods(av_or_int, coord_name)
        self.assertEqual(res, [iris.coords.CellMethod('sum', 'ensemble'),
                               None,
                               iris.coords.CellMethod('mean', 'ensemble')])

    def test_nameIII_mixed(self):
        av_or_int = ['something integrated ob bla',
                     'no averaging',
                     'other averaged']
        coord_name = 'ensemble'
        res = _build_cell_methods(av_or_int, coord_name)
        self.assertEqual(res, [iris.coords.CellMethod('sum', 'ensemble'),
                               None,
                               iris.coords.CellMethod('mean', 'ensemble')])

    def test_unrecognised(self):
        unrecognised_heading = 'bla else'
        av_or_int = ['something average',
                     unrecognised_heading,
                     'something integral']
        coord_name = 'foo'
        with mock.patch('warnings.warn') as warn:
            res = _build_cell_methods(av_or_int, coord_name)
        expected_msg = 'Unknown {} statistic: {!r}. Unable to ' \
                       'create cell method.'.format(coord_name,
                                                    unrecognised_heading)
        warn.assert_called_with(expected_msg)

    def test_unrecognised_similar_to_no_averaging(self):
        unrecognised_headings = ['not averaging',
                                 'this is not a valid no',
                                 'nope',
                                 'no daveraging',
                                 'no averagingg',
                                 'no something',
                                 'noaveraging']
        for unrecognised_heading in unrecognised_headings:
            av_or_int = ['something average',
                         unrecognised_heading,
                         'something integral']
            coord_name = 'foo'
            with mock.patch('warnings.warn') as warn:
                res = _build_cell_methods(av_or_int, coord_name)
            expected_msg = 'Unknown {} statistic: {!r}. Unable to ' \
                           'create cell method.'.format(coord_name,
                                                        unrecognised_heading)
            warn.assert_called_with(expected_msg)


if __name__ == "__main__":
    tests.main()
