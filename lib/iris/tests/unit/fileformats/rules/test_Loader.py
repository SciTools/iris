# (C) British Crown Copyright 2015 - 2018, Met Office
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
"""Unit tests for :class:`iris.fileformats.rules.Loader`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.fileformats.rules import Loader
from iris.tests import mock


class Test___init__(tests.IrisTest):
    def test_normal(self):
        with mock.patch('warnings.warn') as warn:
            loader = Loader(mock.sentinel.GEN_FUNC,
                            mock.sentinel.GEN_FUNC_KWARGS,
                            mock.sentinel.CONVERTER)
        self.assertEqual(warn.call_count, 0)
        self.assertIs(loader.field_generator, mock.sentinel.GEN_FUNC)
        self.assertIs(loader.field_generator_kwargs,
                      mock.sentinel.GEN_FUNC_KWARGS)
        self.assertIs(loader.converter, mock.sentinel.CONVERTER)

    def test_normal_with_explicit_none(self):
        with mock.patch('warnings.warn') as warn:
            loader = Loader(mock.sentinel.GEN_FUNC,
                            mock.sentinel.GEN_FUNC_KWARGS,
                            mock.sentinel.CONVERTER)
        self.assertEqual(warn.call_count, 0)
        self.assertIs(loader.field_generator, mock.sentinel.GEN_FUNC)
        self.assertIs(loader.field_generator_kwargs,
                      mock.sentinel.GEN_FUNC_KWARGS)
        self.assertIs(loader.converter, mock.sentinel.CONVERTER)


if __name__ == '__main__':
    tests.main()
