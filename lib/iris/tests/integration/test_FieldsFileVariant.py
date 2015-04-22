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
"""Integration tests for loading UM FieldsFile variants."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import shutil
import tempfile

import numpy as np

from iris.experimental.um import (Field, Field2, Field3, FieldsFileVariant,
                                  FixedLengthHeader)


IMDI = -32768
RMDI = -1073741824.0


@tests.skip_data
class TestRead(tests.IrisTest):
    def load(self):
        path = tests.get_data_path(('FF', 'n48_multi_field'))
        return FieldsFileVariant(path)

    def test_fixed_length_header(self):
        ffv = self.load()
        self.assertEqual(ffv.fixed_length_header.dataset_type, 3)
        self.assertEqual(ffv.fixed_length_header.lookup_shape, (64, 5))

    def test_integer_constants(self):
        ffv = self.load()
        expected = [IMDI, IMDI, IMDI, IMDI, IMDI,  # 1 - 5
                    96, 73, 70, 70, 4,             # 6 - 10
                    IMDI, 70, 50, IMDI, IMDI,      # 11 - 15
                    IMDI, 2, IMDI, IMDI, IMDI,     # 16 - 20
                    IMDI, IMDI, IMDI, 50, 2381,    # 21 - 25
                    IMDI, IMDI, 4, IMDI, IMDI,     # 26 - 30
                    IMDI, IMDI, IMDI, IMDI, IMDI,  # 31 - 35
                    IMDI, IMDI, IMDI, IMDI, IMDI,  # 36 - 40
                    IMDI, IMDI, IMDI, IMDI, IMDI,  # 41 - 45
                    IMDI]                          # 46
        self.assertArrayEqual(ffv.integer_constants, expected)

    def test_real_constants(self):
        ffv = self.load()
        expected = [3.75, 2.5, -90.0, 0.0, 90.0,      # 1 - 5
                    0.0, RMDI, RMDI, RMDI, RMDI,      # 6 - 10
                    RMDI, RMDI, RMDI, RMDI, RMDI,     # 11 - 15
                    80000.0, RMDI, RMDI, RMDI, RMDI,  # 16 - 20
                    RMDI, RMDI, RMDI, RMDI, RMDI,     # 21 - 25
                    RMDI, RMDI, RMDI, RMDI, RMDI,     # 26 - 30
                    RMDI, RMDI, RMDI, RMDI, RMDI,     # 31 - 35
                    RMDI, RMDI, RMDI]                 # 36 - 38
        self.assertArrayEqual(ffv.real_constants, expected)

    def test_level_dependent_constants(self):
        ffv = self.load()
        # To make sure we have the correct Fortran-order interpretation
        # we just check the overall shape and a few of the values.
        self.assertEqual(ffv.level_dependent_constants.shape, (71, 8))
        expected = [0.92, 0.918, 0.916, 0.912, 0.908]
        self.assertArrayEqual(ffv.level_dependent_constants[:5, 2], expected)

    def test_fields__length(self):
        ffv = self.load()
        self.assertEqual(len(ffv.fields), 5)

    def test_fields__superclass(self):
        ffv = self.load()
        fields = ffv.fields
        for field in fields:
            self.assertIsInstance(field, Field)

    def test_fields__specific_classes(self):
        ffv = self.load()
        fields = ffv.fields
        for i in range(4):
            self.assertIs(type(fields[i]), Field3)
        self.assertIs(type(fields[4]), Field)

    def test_fields__header(self):
        ffv = self.load()
        self.assertEqual(ffv.fields[0].lbfc, 16)

    def test_fields__data_wgdos(self):
        ffv = self.load()
        data = ffv.fields[0].get_data()
        self.assertEqual(data.shape, (73, 96))
        self.assertArrayEqual(data[2, :3], [223.5, 223.0, 222.5])

    def test_fields__data_not_packed(self):
        path = tests.get_data_path(('FF', 'ancillary', 'qrparm.mask'))
        ffv = FieldsFileVariant(path)
        data = ffv.fields[0].get_data()
        expected = [[1, 1, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 0, 1]]
        self.assertArrayEqual(data[:11, 605:608], expected)


@tests.skip_data
class TestUpdate(tests.IrisTest):
    def test_fixed_length_header(self):
        # Check that tweaks to the fixed length header are reflected in
        # the output file.
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            self.assertEqual(ffv.fixed_length_header.sub_model, 1)
            ffv.fixed_length_header.sub_model = 2
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            self.assertEqual(ffv.fixed_length_header.sub_model, 2)

    def test_fixed_length_header_wrong_dtype(self):
        # Check that using the wrong dtype in the fixed length header
        # doesn't confuse things.
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            header_values = ffv.fixed_length_header.raw
            self.assertEqual(header_values.dtype, '>i8')
            header = FixedLengthHeader(header_values.astype('<i4'))
            ffv.fixed_length_header = header
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            # If the header was written out with the wrong dtype this
            # value will go crazy - so check that it's still OK.
            self.assertEqual(ffv.fixed_length_header.sub_model, 1)

    def test_integer_constants(self):
        # Check that tweaks to the integer constants are reflected in
        # the output file.
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            self.assertEqual(ffv.integer_constants[5], 96)
            ffv.integer_constants[5] = 95
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            self.assertEqual(ffv.integer_constants[5], 95)

    def test_integer_constants_wrong_dtype(self):
        # Check that using the wrong dtype in the integer constants
        # doesn't cause mayhem!
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            self.assertEqual(ffv.integer_constants.dtype, '>i8')
            ffv.integer_constants = ffv.integer_constants.astype('<f4')
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            # If the integer constants were written out with the wrong
            # dtype this value will go crazy - so check that it's still
            # OK.
            self.assertEqual(ffv.integer_constants[5], 96)

    def test_real_constants(self):
        # Check that tweaks to the real constants are reflected in the
        # output file.
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            self.assertEqual(ffv.real_constants[1], 2.5)
            ffv.real_constants[1] = 14.75
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            self.assertEqual(ffv.real_constants[1], 14.75)

    def test_real_constants_wrong_dtype(self):
        # Check that using the wrong dtype in the real constants doesn't
        # cause mayhem!
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            self.assertEqual(ffv.real_constants.dtype, '>f8')
            ffv.real_constants = ffv.real_constants.astype('<i4')
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            # If the real constants were written out with the wrong
            # dtype this value will go crazy - so check that it's still
            # OK.
            self.assertEqual(ffv.real_constants[1], 2)

    def test_level_dependent_constants(self):
        # Check that tweaks to the level dependent constants are
        # reflected in the output file.
        # NB. Because it is a multi-dimensional component, this is
        # sensitive to the Fortran vs C array ordering used to write
        # the file.
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            self.assertEqual(ffv.level_dependent_constants[3, 2], 0.912)
            ffv.level_dependent_constants[3, 2] = 0.913
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            self.assertEqual(ffv.level_dependent_constants[3, 2], 0.913)

    def test_field_data(self):
        # Check that tweaks to field data are reflected in the output
        # file.
        src_path = tests.get_data_path(('FF', 'ancillary', 'qrparm.mask'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            field = ffv.fields[0]
            self.assertArrayEqual(field.get_data()[0, 604:607], [0, 1, 1])
            field.set_data(field.get_data() + 10)
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            field = ffv.fields[0]
            self.assertArrayEqual(field.get_data()[0, 604:607], [10, 11, 11])

    def test_large_lookup(self):
        # Check more space is allocated for the lookups when a lot of blank
        # lookups are added.
        src_path = tests.get_data_path(('FF', 'ancillary', 'qrparm.mask'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            field = ffv.fields[0]
            original_field_data = field.get_data()
            blank_int_headers = field.int_headers.copy()
            blank_real_headers = field.real_headers.copy()
            blank_int_headers[:] = -99  # The 'invalid' signature
            blank_real_headers[:] = 0.0
            # Work out how many lookups fills a file 'sector'.
            lookups_per_sector = ffv._WORDS_PER_SECTOR / field.num_values()
            # Make a new fields list with many "blank" and one "real" field.
            n_blank_lookups = 2 * int(np.ceil(lookups_per_sector))
            new_fields = [Field(int_headers=blank_int_headers,
                                real_headers=blank_real_headers,
                                data_provider=None)
                          for _ in range(n_blank_lookups)]
            new_fields.append(field)
            ffv.fields = new_fields
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            self.assertEqual(len(ffv.fields), n_blank_lookups + 1)
            # Check that the data of the last ("real") field is correct.
            field = ffv.fields[-1]
            self.assertArrayEqual(field.get_data(), original_field_data)

    def test_getdata_while_open(self):
        # Check that data is read from the original file if it is still open.
        src_path = tests.get_data_path(('FF', 'ancillary', 'qrparm.mask'))
        with self.temp_filename() as temp_path:
            # Make a copy and open for UPDATE (maybe not strictly necessary?)
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            test_field = ffv.fields[0]
            original_file = test_field._data_provider.source
            # Fetch data.
            test_field.get_data()
            # Check that it used the existing open file.
            last_used_file = ffv.fields[0]._data_provider.source
            self.assertEqual(last_used_file, original_file)
            self.assertFalse(original_file.closed)
            ffv.close()

    def test_getdata_after_close(self):
        # Check that data is read from a new file if the original is closed,
        # and that it is was then closed.
        src_path = tests.get_data_path(('FF', 'ancillary', 'qrparm.mask'))
        with self.temp_filename() as temp_path:
            # Make a copy and open for UPDATE (maybe not strictly necessary?)
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            test_field = ffv.fields[0]
            original_file = test_field._data_provider.source
            ffv.close()
            # Fetch data.
            test_field.get_data()
            # Check that it used the existing open file.
            last_used_file = ffv.fields[0]._data_provider.source
            self.assertNotEqual(last_used_file, original_file)
            self.assertTrue(last_used_file.closed)

    def test_save_packed_as_unpacked(self):
        # Check that we can successfully re-save a packed datafile as unpacked.
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            # Make a copy and open for UPDATE (maybe not strictly necessary?)
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            self.assertEqual(ffv.fields[0].lbpack, 1)
            old_sizes = [fld.lbnrec for fld in ffv.fields
                         if hasattr(fld, 'lbpack')]
            test_data_old = ffv.fields[0].get_data()
            for fld in ffv.fields:
                if hasattr(fld, 'lbpack'):
                    fld.lbpack = 0
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            new_sizes = [fld.lbnrec for fld in ffv.fields
                         if hasattr(fld, 'lbpack')]
            for i_field, (new_size, old_size) in enumerate(zip(new_sizes,
                                                               old_sizes)):
                msg = 'unpacked LBNREC({}) is < packed({})'
                self.assertGreaterEqual(new_size, old_size,
                                        msg=msg.format(new_size, old_size))

            test_data_new = ffv.fields[0].get_data()
            self.assertArrayAllClose(test_data_old, test_data_new)

    def test_save_packed_unchanged(self):
        # Check that we can copy packed fields without ever unpacking them.
        original_getdata_call = Field.get_data
        patch_getdata_call = self.patch('iris.experimental.um.Field.get_data')
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            # Make a copy and open for UPDATE.
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            self.assertEqual(ffv.fields[0].lbpack, 1)
            old_size = ffv.fields[0].lbnrec
            test_data_old = original_getdata_call(ffv.fields[0])
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            self.assertEqual(ffv.fields[0].lbpack, 1)
            new_size = ffv.fields[0].lbnrec
            msg = 'unpacked LBNREC({}) is != packed({})'
            self.assertEqual(new_size, old_size,
                             msg=msg.format(new_size, old_size))
            test_data_new = original_getdata_call(ffv.fields[0])
            self.assertArrayAllClose(test_data_old, test_data_new)
        # Finally, check we never fetched any field data.
        self.assertEqual(patch_getdata_call.call_count, 0)

    def test_save_with_wgdos_packing(self):
        # Check that we can pack data with the WGDOS method.
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            # Make a copy and open for UPDATE (maybe not strictly necessary?)
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)
            # Grab a single field to fiddle with (as a shortcut).
            field = ffv.fields[0]
            # Make a new test data array, compatible with the existing header.
            shape = (field.lbrow, field.lbnpt)
            data = np.arange(np.prod(shape))
            data = data.reshape(shape)
            # Make two copied Fields, with the new data.
            field1 = field.__class__(field.int_headers.copy(),
                                     field.real_headers.copy(),
                                     data.copy())
            field2 = field.__class__(field.int_headers.copy(),
                                     field.real_headers.copy(),
                                     data.copy())
            # Setup one to be wgdos-packed and one to be left unpacked.
            field1.lbpack = 1
            field1.bacc = -13
            field2.lbpack = 0
            # Rewrite the test file with just these fields.
            ffv.fields = [field1, field2]
            ffv.close()

            # Read the test file back in, and check all is as expected.
            ffv = FieldsFileVariant(temp_path)
            self.assertEqual(len(ffv.fields), 2)
            # Check lbpacks.
            self.assertEqual(ffv.fields[0].lbpack, 1)
            self.assertEqual(ffv.fields[1].lbpack, 0)
            # Check that the data arrays are approximately the same.
            self.assertArrayAllClose(ffv.fields[0].get_data(),
                                     ffv.fields[1].get_data())

    def test_save_packed_mixed(self):
        # Check all save options, and show we can "partially" unpack a file.
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            # Make a copy and open for UPDATE.
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)

            # Reduce to only the first 3 fields.
            ffv.fields = ffv.fields[:3]
            # Check that these fields are all WGDOS packed.
            self.assertTrue(all(fld.lbpack == 1 for fld in ffv.fields))

            # Modify the fields to exercise all 3 saving 'styles'.
            # Field#0 : store packed as unpacked.
            data_0 = ffv.fields[0].get_data()
            ffv.fields[0].lbpack = 2000
            # Field#1 : pass-through packed as packed.
            data_1 = ffv.fields[1].get_data()
            # Field#2 : save array as unpacked.
            shape2 = (ffv.fields[2].lbrow, ffv.fields[2].lbnpt)
            data_2 = np.arange(np.prod(shape2)).reshape(shape2)
            ffv.fields[2].set_data(data_2)
            ffv.fields[2].lbpack = 3000
            ffv.close()

            # Read the test file back in, and check all is as expected.
            ffv = FieldsFileVariant(temp_path)
            self.assertEqual(len(ffv.fields), 3)
            # Field#0.
            self.assertEqual(ffv.fields[0].lbpack, 2000)
            self.assertArrayAllClose(ffv.fields[0].get_data(), data_0)
            # Field#1.
            self.assertEqual(ffv.fields[1].lbpack, 1)
            self.assertArrayAllClose(ffv.fields[1].get_data(), data_1)
            # Field#2.
            self.assertEqual(ffv.fields[2].lbpack, 3000)
            self.assertArrayAllClose(ffv.fields[2].get_data(), data_2)

    def test_fail_save_with_unknown_packing(self):
        # Check that trying to save data as packed causes an error.
        src_path = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            # Make a copy and open for UPDATE.
            shutil.copyfile(src_path, temp_path)
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.UPDATE_MODE)

            # Reduce to just one field.
            field = ffv.fields[0]
            ffv.fields = [field]
            # Actualise the data to a concrete array.
            field.set_data(field.get_data())
            # Attempt to save back to an unrecognised packed format.
            field.lbpack = 7
            msg = 'Cannot save.*lbpack=7.*unsupported'
            with self.assertRaisesRegexp(ValueError, msg):
                ffv.close()


class TestCreate(tests.IrisTest):
    @tests.skip_data
    def test_copy(self):
        # Checks that copying all the attributes to a new file
        # re-creates the original with minimal differences.
        src_path = tests.get_data_path(('FF', 'ancillary', 'qrparm.mask'))
        ffv_src = FieldsFileVariant(src_path, FieldsFileVariant.READ_MODE)
        with self.temp_filename() as temp_path:
            ffv_dest = FieldsFileVariant(temp_path,
                                         FieldsFileVariant.CREATE_MODE)
            ffv_dest.fixed_length_header = ffv_src.fixed_length_header
            for name, kind in FieldsFileVariant._COMPONENTS:
                setattr(ffv_dest, name, getattr(ffv_src, name))
            ffv_dest.fields = ffv_src.fields
            ffv_dest.close()

            # Compare the files at a binary level.
            src = np.fromfile(src_path, dtype='>i8', count=-1)
            dest = np.fromfile(temp_path, dtype='>i8', count=-1)
            changed_indices = np.where(src != dest)[0]
            # Allow for acceptable differences.
            self.assertArrayEqual(changed_indices, [110, 111, 125, 126, 130,
                                                    135, 140, 142, 144, 160])
            # All but the last difference is from the use of IMDI
            # instead of 1 to mark an unused dimension length.
            self.assertArrayEqual(dest[changed_indices[:-1]], [IMDI] * 9)
            # The last difference is to the length of the DATA component
            # because we've padded the last field.
            self.assertEqual(dest[160], 956416)

    def test_create(self):
        # Check we can create a new file from scratch, with the correct
        # cross-referencing automatically applied to the headers to
        # enable it to load again.
        with self.temp_filename() as temp_path:
            ffv = FieldsFileVariant(temp_path, FieldsFileVariant.CREATE_MODE)
            ffv.fixed_length_header = FixedLengthHeader([-1] * 256)
            ffv.fixed_length_header.data_set_format_version = 20
            ffv.fixed_length_header.sub_model = 1
            ffv.fixed_length_header.dataset_type = 3
            constants = IMDI * np.ones(46, dtype=int)
            constants[5] = 4
            constants[6] = 5
            ffv.integer_constants = constants
            ints = IMDI * np.ones(45, dtype=int)
            ints[17] = 4  # LBROW
            ints[18] = 5  # LBNPT
            ints[20] = 0  # LBPACK
            ints[21] = 2  # LBREL
            ints[38] = 1  # LBUSER(1)
            reals = range(19)
            src_data = np.arange(20, dtype='f4').reshape((4, 5))
            ffv.fields = [Field2(ints, reals, src_data)]
            ffv.close()

            ffv = FieldsFileVariant(temp_path)
            # Fill with -1 instead of IMDI so we can detect where IMDI
            # values are being automatically set.
            expected = -np.ones(256, dtype=int)
            expected[0] = 20
            expected[1] = 1
            expected[4] = 3
            expected[99:101] = (257, 46)  # Integer constants
            expected[104:106] = IMDI
            expected[109:112] = IMDI
            expected[114:117] = IMDI
            expected[119:122] = IMDI
            expected[124:127] = IMDI
            expected[129:131] = IMDI
            expected[134:136] = IMDI
            expected[139:145] = IMDI
            expected[149:152] = (303, 64, 1)  # 303 = 256 + 46 + 1
            expected[159:161] = (2049, 2048)
            # Compare using lists because we get more helpful error messages!
            self.assertEqual(list(ffv.fixed_length_header.raw), list(expected))
            self.assertArrayEqual(ffv.integer_constants, constants)
            self.assertIsNone(ffv.real_constants)
            self.assertEqual(len(ffv.fields), 1)
            for field in ffv.fields:
                data = field.get_data()
                self.assertArrayEqual(data, src_data)


if __name__ == '__main__':
    tests.main()
