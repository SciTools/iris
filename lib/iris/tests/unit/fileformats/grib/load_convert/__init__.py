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
"""Unit tests for the :mod:`iris.fileformats.grib._load_convert` package."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from collections import OrderedDict


def empty_metadata():
    metadata = OrderedDict()
    metadata['factories'] = []
    metadata['references'] = []
    metadata['standard_name'] = None
    metadata['long_name'] = None
    metadata['units'] = None
    metadata['attributes'] = {}
    metadata['cell_methods'] = []
    metadata['dim_coords_and_dims'] = []
    metadata['aux_coords_and_dims'] = []
    return metadata


class LoadConvertTest(tests.IrisTest):
    def assertMetadataEqual(self, result, expected):
        # Compare two metadata dictionaries. Gives slightly more
        # helpful error message than: self.assertEqual(result, expected)
        self.assertEqual(result.keys(), expected.keys())
        for key in result.keys():
            self.assertEqual(result[key], expected[key])
