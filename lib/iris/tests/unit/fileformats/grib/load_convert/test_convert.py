# (C) British Crown Copyright 2014 - 2017, Met Office
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
"""Test function :func:`iris.fileformats.grib._load_convert.convert`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris.tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import mock

from iris.exceptions import TranslationError

from iris.fileformats.grib._load_convert import convert
from iris.tests.unit.fileformats.grib import _make_test_message


class TestGribMessage(tests.IrisTest):
    def test_edition_2(self):
        def func(field, metadata):
            return metadata['factories'].append(factory)

        sections = [{'editionNumber': 2}]
        field = _make_test_message(sections)
        this = 'iris.fileformats.grib._load_convert.grib2_convert'
        factory = mock.sentinel.factory
        with mock.patch(this, side_effect=func) as grib2_convert:
            # The call being tested.
            result = convert(field)
            self.assertTrue(grib2_convert.called)
            metadata = ([factory], [], None, None, None, {}, [], [], [])
            self.assertEqual(result, metadata)

    def test_edition_1_bad(self):
        sections = [{'editionNumber': 1}]
        field = _make_test_message(sections)
        emsg = 'edition 1 is not supported'
        with self.assertRaisesRegexp(TranslationError, emsg):
            convert(field)


class TestGribWrapper(tests.IrisTest):
    def test_edition_2_bad(self):
        # Test object with no '.sections', and '.edition' ==2.
        field = mock.Mock(edition=2, spec=('edition'))
        emsg = 'edition 2 is not supported'
        with self.assertRaisesRegexp(TranslationError, emsg):
            convert(field)

    def test_edition_1(self):
        # Test object with no '.sections', and '.edition' ==1.
        field = mock.Mock(edition=1, spec=('edition'))
        func = 'iris.fileformats.grib._load_convert.grib1_convert'
        metadata = mock.sentinel.metadata
        with mock.patch(func, return_value=metadata) as grib1_convert:
            result = convert(field)
            grib1_convert.assert_called_once_with(field)
            self.assertEqual(result, metadata)


if __name__ == '__main__':
    tests.main()
