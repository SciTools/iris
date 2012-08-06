# (C) British Crown Copyright 2010 - 2012, Met Office
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
Test the Fieldsfile file loading plugin and FFHeader.

"""


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os.path

import iris
import iris.fileformats.ff as ff


@iris.tests.skip_data
class TestFFHeaderGet(tests.IrisTest):
    def setUp(self):
        self.filename = tests.get_data_path(('ssps', 'qtgl.ssps_006'))
        self.ff_header = ff.FFHeader(self.filename)

    def test_unit_pass_0(self):
        """Test FieldsFile header attribute offset."""
        ff_header_index = [(name, tuple([position-ff.UM_TO_FF_HEADER_OFFSET for position in positions])) for name, positions in ff.UM_FIXED_LENGTH_HEADER]
        self.assertEqual(ff.FF_HEADER, ff_header_index)

    def test_unit_pass_1(self):
        """Test FieldsFile header attribute lookup."""
        self.assertEqual(self.ff_header.data_set_format_version, 20)
        self.assertEqual(self.ff_header.sub_model, 1)
        self.assertEqual(self.ff_header.vert_coord_type, 5)
        self.assertEqual(self.ff_header.horiz_grid_type, 0)
        self.assertEqual(self.ff_header.dataset_type, 3)
        self.assertEqual(self.ff_header.run_identifier, 1)
        self.assertEqual(self.ff_header.experiment_number, -32768)
        self.assertEqual(self.ff_header.calendar, 1)
        self.assertEqual(self.ff_header.grid_staggering, 3)
        self.assertEqual(self.ff_header.time_type, -32768)
        self.assertEqual(self.ff_header.projection_number, -32768)
        self.assertEqual(self.ff_header.model_version, 707)
        self.assertEqual(self.ff_header.obs_file_type, -32768)
        self.assertEqual(self.ff_header.last_fieldop_type, -32768)
        self.assertEqual(self.ff_header.first_validity_time, (2011, 4, 18, 6, 0, 0, 108))
        self.assertEqual(self.ff_header.last_validity_time, (2011, 4, 18, 9, 0, 0, 108))
        self.assertEqual(self.ff_header.misc_validity_time, (2011, 4, 18, 12, 59, 19, -32768))
        self.assertEqual(self.ff_header.integer_constants, (257, 46))
        self.assertEqual(self.ff_header.real_constants, (303, 38))
        self.assertEqual(self.ff_header.level_dependent_constants, (341, 71, 8))
        self.assertEqual(self.ff_header.row_dependent_constants, (0, -1073741824, -1073741824))
        self.assertEqual(self.ff_header.column_dependent_constants, (0, -1073741824, -1073741824))
        self.assertEqual(self.ff_header.fields_of_constants, (0, -1073741824, -1073741824))
        self.assertEqual(self.ff_header.extra_constants, (0, -1073741824))
        self.assertEqual(self.ff_header.temp_historyfile, (0, -1073741824))
        self.assertEqual(self.ff_header.compressed_field_index1, (0, -1073741824))
        self.assertEqual(self.ff_header.compressed_field_index2, (0, -1073741824))
        self.assertEqual(self.ff_header.compressed_field_index3, (0, -1073741824))
        self.assertEqual(self.ff_header.lookup_table, (909, 64, 4096))
        self.assertEqual(self.ff_header.total_prognostic_fields, 2970)
        self.assertEqual(self.ff_header.data, (264193, 2477557843, -32768))

    def test_unit_pass_2(self):
        """Test FieldsFile header for valid pointer attributes."""
        self.assertTrue(self.ff_header.valid('integer_constants'))
        self.assertTrue(self.ff_header.valid('real_constants'))
        self.assertTrue(self.ff_header.valid('level_dependent_constants'))
        self.assertFalse(self.ff_header.valid('row_dependent_constants'))
        self.assertFalse(self.ff_header.valid('column_dependent_constants'))
        self.assertFalse(self.ff_header.valid('fields_of_constants'))
        self.assertFalse(self.ff_header.valid('extra_constants'))
        self.assertFalse(self.ff_header.valid('temp_historyfile'))
        self.assertFalse(self.ff_header.valid('compressed_field_index1'))
        self.assertFalse(self.ff_header.valid('compressed_field_index2'))
        self.assertFalse(self.ff_header.valid('compressed_field_index3'))
        self.assertTrue(self.ff_header.valid('lookup_table'))
        self.assertTrue(self.ff_header.valid('data'))

    def test_unit_pass_3(self):
        """Test FieldsFile header pointer attribute addresses."""
        self.assertEqual(self.ff_header.address('integer_constants'), self.ff_header.integer_constants[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('real_constants'), self.ff_header.real_constants[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('level_dependent_constants'), self.ff_header.level_dependent_constants[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('row_dependent_constants'), self.ff_header.row_dependent_constants[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('column_dependent_constants'), self.ff_header.column_dependent_constants[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('fields_of_constants'), self.ff_header.fields_of_constants[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('extra_constants'), self.ff_header.extra_constants[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('temp_historyfile'), self.ff_header.temp_historyfile[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('compressed_field_index1'), self.ff_header.compressed_field_index1[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('compressed_field_index2'), self.ff_header.compressed_field_index2[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('compressed_field_index3'), self.ff_header.compressed_field_index3[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('lookup_table'), self.ff_header.lookup_table[0] * ff.FF_WORD_DEPTH)
        self.assertEqual(self.ff_header.address('data'), self.ff_header.data[0] * ff.FF_WORD_DEPTH)

    def test_unit_pass_4(self):
        """Test FieldsFile header pointer attribute shape."""
        self.assertEqual(self.ff_header.shape('integer_constants'), self.ff_header.integer_constants[1:])
        self.assertEqual(self.ff_header.shape('real_constants'), self.ff_header.real_constants[1:])
        self.assertEqual(self.ff_header.shape('level_dependent_constants'), self.ff_header.level_dependent_constants[1:])
        self.assertEqual(self.ff_header.shape('row_dependent_constants'), self.ff_header.row_dependent_constants[1:])
        self.assertEqual(self.ff_header.shape('column_dependent_constants'), self.ff_header.column_dependent_constants[1:])
        self.assertEqual(self.ff_header.shape('fields_of_constants'), self.ff_header.fields_of_constants[1:])
        self.assertEqual(self.ff_header.shape('extra_constants'), self.ff_header.extra_constants[1:])
        self.assertEqual(self.ff_header.shape('temp_historyfile'), self.ff_header.temp_historyfile[1:])
        self.assertEqual(self.ff_header.shape('compressed_field_index1'), self.ff_header.compressed_field_index1[1:])
        self.assertEqual(self.ff_header.shape('compressed_field_index2'), self.ff_header.compressed_field_index2[1:])
        self.assertEqual(self.ff_header.shape('compressed_field_index3'), self.ff_header.compressed_field_index3[1:])
        self.assertEqual(self.ff_header.shape('lookup_table'), self.ff_header.lookup_table[1:])
        self.assertEqual(self.ff_header.shape('data'), self.ff_header.data[1:])


@iris.tests.skip_data
class TestFF2PP2Cube(tests.IrisTest):
    def setUp(self):
        self.filename = tests.get_data_path(('ssps', 'qtgl.ssps_006'))

    def assertCML(self, cube, path, *args, **kwargs):
        try:
            coord = cube.coord('forecast_period')
            coord._TEST_COMPAT_force_explicit = True
        except iris.exceptions.CoordinateNotFoundError:
            pass
        try:
            coord = cube.coord('time')
            coord._TEST_COMPAT_force_explicit = True
        except iris.exceptions.CoordinateNotFoundError:
            pass
        try:
            coord = cube.coord('model_level_number')
            coord._TEST_COMPAT_force_explicit = True
        except iris.exceptions.CoordinateNotFoundError:
            pass
        try:
            coord = cube.coord('height')
            coord._TEST_COMPAT_override_axis = 'z'
        except iris.exceptions.CoordinateNotFoundError:
            pass
        super(TestFF2PP2Cube, self).assertCML(cube, path, *args, **kwargs)

    def test_unit_pass_0(self):
        """Test FieldsFile to PPFields cube load."""
        # Adding the surface_altitude to all 4000(?) fields causes a
        # massive memory overhead - so until that's resolved, throw
        # away all the cubes we're not interested.
        standard_names = ['air_temperature', 'soil_temperature']
        def callback(cube, field, filename):
            if cube.standard_name not in standard_names + ['surface_altitude']:
                raise iris.exceptions.IgnoreCubeException
        #
        # CML comparision causes packed cube data to be read from disk,
        # so we must constrain the load otherwise this test will
        # take a _very_ long time to execute!
        #
        cube_by_name = {}
        cubes = iris.load(self.filename, standard_names, callback=callback)
        # Re-order to match the old cube-merge order.
        cubes = [cubes[0], cubes[3], cubes[2], cubes[4], cubes[1]] + cubes[5:]
        while cubes:
            cube = cubes.pop(0)
            standard_name = cube.standard_name
            v = cube_by_name.setdefault(standard_name, None)
            if v is None:
                cube_by_name[standard_name] = 0
            else:
                cube_by_name[standard_name] += 1
            self.assertCML(cube, ('FF', '%s_%s_%d.cml' % (os.path.basename(self.filename), standard_name, cube_by_name[standard_name])))


if __name__ == '__main__':
    tests.main()
