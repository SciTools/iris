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

from iris.experimental.um import FieldsFileVariant, cutout, Field3


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
    @contextmanager
    def temp_ff(self):
        with self.temp_filename() as temp_path:
            ffv = FieldsFileVariant(temp_path,
                                    mode=FieldsFileVariant.CREATE_MODE)
            yield ffv

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
        ffv.fields.append(field)

    def test_fixed_length_header(self):
        with self.temp_ff() as ffv:
            with self.temp_filename() as temp_path:
                self.simple_p_grid(ffv, 10, 12)
                ffv_dest = cutout(ffv, temp_path, [2, 1, 4, 5])
                self.assertEqual(ffv_dest.fixed_length_header.sub_model, 1)

    def test_horiz_grid_type(self):
        with self.temp_ff() as ffv:
            with self.temp_filename() as temp_path:
                self.simple_p_grid(ffv, 10, 12)
                ffv_dest = cutout(ffv, temp_path, [2, 1, 4, 5])
                self.assertEqual(ffv_dest.fixed_length_header.horiz_grid_type,
                                 3)

    def test_integer_constants(self):
        with self.temp_ff() as ffv:
            with self.temp_filename() as temp_path:
                self.simple_p_grid(ffv, 10, 12)
                ffv_dest = cutout(ffv, temp_path, [2, 1, 4, 5])
                self.assertEqual(ffv_dest.integer_constants[5], 4)
                self.assertEqual(ffv_dest.integer_constants[6], 5)

    def test_real_constants(self):
        with self.temp_ff() as ffv:
            with self.temp_filename() as temp_path:
                self.simple_p_grid(ffv, 10, 12)
                ffv_dest = cutout(ffv, temp_path, [2, 1, 4, 5])
                self.assertEqual(ffv_dest.real_constants[2], 10)
                self.assertEqual(ffv_dest.real_constants[3], 10)

    def test_lbhem(self):
        with self.temp_ff() as ffv:
            with self.temp_filename() as temp_path:
                self.simple_p_grid(ffv, 10, 12)
                ffv_dest = cutout(ffv, temp_path, [2, 1, 4, 5])
                self.assertEqual(ffv_dest.fields[0].lbhem, 3)

    def test_lbnpt_lbrow(self):
        with self.temp_ff() as ffv:
            with self.temp_filename() as temp_path:
                self.simple_p_grid(ffv, 10, 12)
                ffv_dest = cutout(ffv, temp_path, [2, 1, 4, 5])
                self.assertEqual(ffv_dest.fields[0].lbnpt, 4)
                self.assertEqual(ffv_dest.fields[0].lbrow, 5)

    def test_bzx_bzy(self):
        with self.temp_ff() as ffv:
            with self.temp_filename() as temp_path:
                self.simple_p_grid(ffv, 10, 12)
                ffv_dest = cutout(ffv, temp_path, [2, 1, 4, 5])
                self.assertEqual(ffv_dest.fields[0].bzx, 10)
                self.assertEqual(ffv_dest.fields[0].bzy, 10)

    def test_too_many_nx_ny(self):
        with self.temp_ff() as ffv:
            with self.temp_filename() as temp_path:
                self.simple_p_grid(ffv, 10, 12)
                ffv_dest = cutout(ffv, temp_path, [5, 5, 100, 100])
                self.assertEqual(ffv_dest.integer_constants[5], 5)
                self.assertEqual(ffv_dest.integer_constants[6], 7)
                # Add test to check warning is raised

    def test_get_data(self):
        with self.temp_ff() as ffv:
            with self.temp_filename() as temp_path:
                self.simple_p_grid(ffv, 10, 12)
                ffv_dest = cutout(ffv, temp_path, [1, 0, 2, 1])
                array = np.array([[1., 2.]])
                self.assertArrayEqual(ffv_dest.fields[0].get_data(), array)


if __name__ == '__main__':
    tests.main()
