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
Unit tests for :class:`iris.experimental.um.FieldsFileVariant`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import os.path
import shutil
import tempfile
import unittest

import numpy as np

from iris.experimental.um import FieldsFileVariant, Field, Field3

try:
    import mo_pack
except ImportError as err:
    if err.message.startswith('No module named'):
        # Disable all these tests if mo_pack is not installed.
        mo_pack = None
    else:
        raise

skip_mo_pack = unittest.skipIf(mo_pack is None,
                               'Test(s) require "mo_pack", '
                               'which is not available.')


class Test___init__(tests.IrisTest):
    def test_invalid_mode(self):
        with self.assertRaisesRegexp(ValueError, 'access mode'):
            FieldsFileVariant('/fake/path', mode='g')

    def test_missing_file(self):
        dir_path = tempfile.mkdtemp()
        try:
            file_path = os.path.join(dir_path, 'missing')
            with self.assertRaisesRegexp(IOError, 'No such file'):
                FieldsFileVariant(file_path, mode=FieldsFileVariant.READ_MODE)
        finally:
            shutil.rmtree(dir_path)

    def test_new_file(self):
        with self.temp_filename() as temp_path:
            ffv = FieldsFileVariant(temp_path,
                                    mode=FieldsFileVariant.CREATE_MODE)
            self.assertArrayEqual(ffv.fixed_length_header.raw, [-32768] * 256)
            self.assertIsNone(ffv.integer_constants)
            self.assertIsNone(ffv.real_constants)
            self.assertIsNone(ffv.level_dependent_constants)
            self.assertIsNone(ffv.row_dependent_constants)
            self.assertIsNone(ffv.column_dependent_constants)
            self.assertIsNone(ffv.fields_of_constants)
            self.assertIsNone(ffv.extra_constants)
            self.assertIsNone(ffv.temp_historyfile)
            self.assertIsNone(ffv.compressed_field_index1)
            self.assertIsNone(ffv.compressed_field_index2)
            self.assertIsNone(ffv.compressed_field_index3)
            self.assertEqual(ffv.fields, [])
            del ffv


@tests.skip_data
class Test_filename(tests.IrisTest):
    def test(self):
        path = tests.get_data_path(('FF', 'n48_multi_field'))
        ffv = FieldsFileVariant(path)
        self.assertEqual(ffv.filename, path)


@tests.skip_data
class Test_class_assignment(tests.IrisTest):
    @skip_mo_pack
    def test_lbrel_class(self):
        path = tests.get_data_path(('FF', 'lbrel_test_data'))
        ffv = FieldsFileVariant(path)
        self.assertEqual(type(ffv.fields[0]), Field)
        self.assertEqual(type(ffv.fields[1]), Field3)
        self.assertEqual(ffv.fields[0].int_headers[Field.LBREL_OFFSET], -32768)
        self.assertEqual(ffv.fields[1].int_headers[Field.LBREL_OFFSET], 3)


class Test_mode(tests.IrisTest):
    @tests.skip_data
    def test_read(self):
        path = tests.get_data_path(('FF', 'n48_multi_field'))
        ffv = FieldsFileVariant(path)
        self.assertIs(ffv.mode, FieldsFileVariant.READ_MODE)

    @tests.skip_data
    def test_append(self):
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path,
                                    mode=FieldsFileVariant.UPDATE_MODE)
            self.assertIs(ffv.mode, FieldsFileVariant.UPDATE_MODE)
            del ffv

    def test_write(self):
        with self.temp_filename() as temp_path:
            ffv = FieldsFileVariant(temp_path,
                                    mode=FieldsFileVariant.CREATE_MODE)
            self.assertIs(ffv.mode, FieldsFileVariant.CREATE_MODE)
            del ffv


if __name__ == '__main__':
    tests.main()
