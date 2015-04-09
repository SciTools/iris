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

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import os.path
import shutil
import tempfile

from contextlib import contextmanager
import numpy as np

from iris.experimental.um import FieldsFileVariant, cutout, Field, Field3,\
    FixedLengthHeader


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
    def test_lbrel_class(self):
        path = tests.get_data_path(('FF', 'lbrel_test_data'))
        ffv = FieldsFileVariant(path)
        self.assertEquals(type(ffv.fields[0]), Field)
        self.assertEquals(type(ffv.fields[1]), Field3)
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


class Test_cutout(tests.IrisTest):
    def simple_p_grid(self, ffv, nx, ny):
        ffv.fixed_length_header.horiz_grid_type = 0
        ffv.fixed_length_header.sub_model = 1
        ffv.integer_constants = np.arange(46, dtype='>i8')
        ffv.real_constants = np.arange(38, dtype='>f8')
        ffv.integer_constants[5] = nx
        ffv.integer_constants[6] = ny
        ffv.real_constants[0] = 5
        ffv.real_constants[1] = 10
        ffv.real_constants[2] = 0
        ffv.real_constants[3] = 0

        field = Field3(np.arange(45, dtype='>i8'),
                       np.arange(19, dtype='>f8'),
                       np.arange(nx * ny, dtype='>f8').reshape([ny, nx])
                       )
        field.lbhem = 0
        field.bdx = 5
        field.bzx = 0
        field.bdy = 10
        field.bzy = 0
        field.lbnpt = nx
        field.lbrow = ny
        field.lbcode = 1
        field.lbext = 0
        ffv.fields.append(field)

    def setUp(self):
        # Create a temporary test directory.
        self.temp_dirpath = tempfile.mkdtemp()
        self.input_ffv_path = os.path.join(self.temp_dirpath, 'in_ff')
        self.output_ffv_path = os.path.join(self.temp_dirpath, 'out_ff')

        # Create a test input file (and leave this *open*).
        self.input_ffv = FieldsFileVariant(self.input_ffv_path,
                                           mode=FieldsFileVariant.CREATE_MODE)

        # Define a standard grid on the input file.
        self.simple_p_grid(self.input_ffv, 10, 12)

    def tearDown(self):
        # Close the test input file and delete the whole test directory.
        # NOTE: any created ffvs already got closed, as they left scope on exit
        # from the test routine.
        self.input_ffv.close()
        shutil.rmtree(self.temp_dirpath)

    def test_fixed_length_header(self):
        ffv_dest = cutout(self.input_ffv, self.output_ffv_path, [2, 1, 4, 5])
        self.assertEqual(ffv_dest.fixed_length_header.sub_model, 1)

    def test_horiz_grid_type(self):
        ffv_dest = cutout(self.input_ffv, self.output_ffv_path, [2, 1, 4, 5])
        self.assertEqual(ffv_dest.fixed_length_header.horiz_grid_type, 3)

    def test_integer_constants(self):
        ffv_dest = cutout(self.input_ffv, self.output_ffv_path, [2, 1, 4, 5])
        self.assertEqual(ffv_dest.integer_constants[5], 4)
        self.assertEqual(ffv_dest.integer_constants[6], 5)

    def test_real_constants(self):
        ffv_dest = cutout(self.input_ffv, self.output_ffv_path, [2, 1, 4, 5])
        self.assertEqual(ffv_dest.real_constants[2], 10)
        self.assertEqual(ffv_dest.real_constants[3], 10)

    def test_lbhem(self):
        ffv_dest = cutout(self.input_ffv, self.output_ffv_path, [2, 1, 4, 5])
        self.assertEqual(ffv_dest.fields[0].lbhem, 3)

    def test_lbnpt_lbrow(self):
        ffv_dest = cutout(self.input_ffv, self.output_ffv_path, [2, 1, 4, 5])
        self.assertEqual(ffv_dest.fields[0].lbnpt, 4)
        self.assertEqual(ffv_dest.fields[0].lbrow, 5)

    def test_bzx_bzy(self):
        ffv_dest = cutout(self.input_ffv, self.output_ffv_path, [2, 1, 4, 5])
        self.assertEqual(ffv_dest.fields[0].bzx, 10)
        self.assertEqual(ffv_dest.fields[0].bzy, 10)

    def test_fail_too_many_nx_ny(self):
        msg_re = 'cutout .* outside the dimensions of the grid'
        with self.assertRaisesRegexp(ValueError, msg_re):
            ffv_dest = cutout(self.input_ffv, self.output_ffv_path,
                              [5, 5, 100, 100])

    def test_get_data(self):
        ffv_dest = cutout(self.input_ffv, self.output_ffv_path, [1, 0, 2, 1])
        array = np.array([[1., 2.]])
        self.assertArrayEqual(ffv_dest.fields[0].get_data(), array)

    def test_fail_fixed_dx_0(self):
        self.input_ffv.real_constants[0] = 0.0
        msg_re = 'Source grid in header is not regular'
        with self.assertRaisesRegexp(ValueError, msg_re):
            ffv_dest = cutout(self.input_ffv, self.output_ffv_path,
                              [2, 1, 4, 5])

    def test_fail_fixed_dy_rmdi(self):
        self.input_ffv.real_constants[0] = -2.0e30
        msg_re = 'Source grid in header is not regular'
        with self.assertRaisesRegexp(ValueError, msg_re):
            ffv_dest = cutout(self.input_ffv, self.output_ffv_path,
                              [2, 1, 4, 5])

    def test_fail_field_dx_0(self):
        self.input_ffv.fields[0].bdx = 0.0
        msg_re = 'Source grid in field#0 is not regular'
        with self.assertRaisesRegexp(ValueError, msg_re):
            ffv_dest = cutout(self.input_ffv, self.output_ffv_path,
                              [2, 1, 4, 5])

    def test_fail_field2_dy_mdi(self):
        field1 = self.input_ffv.fields[0]
        field2 = Field3(field1.int_headers.copy(),
                        field1.real_headers.copy(),
                        field1.get_data())
        self.input_ffv.fields = [field1, field2]
        field2.bmdi = 123.45
        field2.bdy = 123.45
        msg_re = 'Source grid in field#1 is not regular'
        with self.assertRaisesRegexp(ValueError, msg_re):
            ffv_dest = cutout(self.input_ffv, self.output_ffv_path,
                              [2, 1, 4, 5])

    def test_fail_field_extension_data(self):
        self.input_ffv.fields[0].lbext = 1
        msg_re = 'field#0 has extension data'
        with self.assertRaisesRegexp(ValueError, msg_re):
            ffv_dest = cutout(self.input_ffv, self.output_ffv_path,
                              [2, 1, 4, 5])


if __name__ == '__main__':
    tests.main()
