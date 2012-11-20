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

import collections
import os.path

import mock
import numpy

import iris
import iris.fileformats.ff as ff
import iris.fileformats.pp as pp


_MOCK_FIELD = collections.namedtuple('MockField',
                                     'lbext lblrec lbnrec lbpack lbuser')
_MOCK_LBPACK = collections.namedtuple('MockPack', 'n1')

# PP-field: LBPACK N1 values.
_UNPACKED = 0
_WGDOS = 1
_CRAY = 2
_GRIB = 3  # Not implemented.
_RLE = 4   # Not supported, deprecated FF format.

# PP-field: LBUSER(1) values.
_REAL = 1
_INTEGER = 2
_LOGICAL = 3  # Not implemented.


class TestFF_HEADER(tests.IrisTest):
    def test_initialisation(self):
        self.assertEqual(ff.FF_HEADER[0], ('data_set_format_version', (0,)))
        self.assertEqual(ff.FF_HEADER[17], ('integer_constants', (99, 100)))

    def test_size(self):
        self.assertEqual(len(ff.FF_HEADER), 31)


@iris.tests.skip_data
class TestFFHeader(tests.IrisTest):
    def setUp(self):
        self.filename = tests.get_data_path(('FF', 'n48_multi_field'))
        self.ff_header = ff.FFHeader(self.filename)
        self.valid_headers = (
            'integer_constants', 'real_constants', 'level_dependent_constants',
            'lookup_table', 'data'
        )
        self.invalid_headers = (
            'row_dependent_constants', 'column_dependent_constants',
            'fields_of_constants', 'extra_constants', 'temp_historyfile',
            'compressed_field_index1', 'compressed_field_index2',
            'compressed_field_index3'
        )

    def test_constructor(self):
        """Test FieldsFile header attribute lookup."""
        self.assertEqual(self.ff_header.data_set_format_version, 20)
        self.assertEqual(self.ff_header.sub_model, 1)
        self.assertEqual(self.ff_header.vert_coord_type, 5)
        self.assertEqual(self.ff_header.horiz_grid_type, 0)
        self.assertEqual(self.ff_header.dataset_type, 3)
        self.assertEqual(self.ff_header.run_identifier, 0)
        self.assertEqual(self.ff_header.experiment_number, -32768)
        self.assertEqual(self.ff_header.calendar, 1)
        self.assertEqual(self.ff_header.grid_staggering, 3)
        self.assertEqual(self.ff_header.time_type, -32768)
        self.assertEqual(self.ff_header.projection_number, -32768)
        self.assertEqual(self.ff_header.model_version, 802)
        self.assertEqual(self.ff_header.obs_file_type, -32768)
        self.assertEqual(self.ff_header.last_fieldop_type, -32768)
        self.assertEqual(self.ff_header.first_validity_time,
                         (2011, 7, 10, 18, 0, 0, 191))
        self.assertEqual(self.ff_header.last_validity_time,
                         (2011, 7, 10, 21, 0, 0, 191))
        self.assertEqual(self.ff_header.misc_validity_time,
                         (2012, 4, 30, 18, 12, 13, -32768))
        self.assertEqual(self.ff_header.integer_constants, (257, 46))
        self.assertEqual(self.ff_header.real_constants, (303, 38))
        self.assertEqual(self.ff_header.level_dependent_constants, (341, 71, 8))
        self.assertEqual(self.ff_header.row_dependent_constants,
                         (0, -1073741824, -1073741824))
        self.assertEqual(self.ff_header.column_dependent_constants,
                         (0, -1073741824, -1073741824))
        self.assertEqual(self.ff_header.fields_of_constants,
                         (0, -1073741824, -1073741824))
        self.assertEqual(self.ff_header.extra_constants, (0, -1073741824))
        self.assertEqual(self.ff_header.temp_historyfile, (0, -1073741824))
        self.assertEqual(self.ff_header.compressed_field_index1,
                         (0, -1073741824))
        self.assertEqual(self.ff_header.compressed_field_index2,
                         (0, -1073741824))
        self.assertEqual(self.ff_header.compressed_field_index3,
                         (0, -1073741824))
        self.assertEqual(self.ff_header.lookup_table, (909, 64, 5))
        self.assertEqual(self.ff_header.total_prognostic_fields, 3119)
        self.assertEqual(self.ff_header.data, (2049, 2961, -32768))

    def test_str(self):
        target = """FF Header:
    data_set_format_version: 20
    sub_model: 1
    vert_coord_type: 5
    horiz_grid_type: 0
    dataset_type: 3
    run_identifier: 0
    experiment_number: -32768
    calendar: 1
    grid_staggering: 3
    time_type: -32768
    projection_number: -32768
    model_version: 802
    obs_file_type: -32768
    last_fieldop_type: -32768
    first_validity_time: (2011, 7, 10, 18, 0, 0, 191)
    last_validity_time: (2011, 7, 10, 21, 0, 0, 191)
    misc_validity_time: (2012, 4, 30, 18, 12, 13, -32768)
    integer_constants: (257, 46)
    real_constants: (303, 38)
    level_dependent_constants: (341, 71, 8)
    row_dependent_constants: (0, -1073741824, -1073741824)
    column_dependent_constants: (0, -1073741824, -1073741824)
    fields_of_constants: (0, -1073741824, -1073741824)
    extra_constants: (0, -1073741824)
    temp_historyfile: (0, -1073741824)
    compressed_field_index1: (0, -1073741824)
    compressed_field_index2: (0, -1073741824)
    compressed_field_index3: (0, -1073741824)
    lookup_table: (909, 64, 5)
    total_prognostic_fields: 3119
    data: (2049, 2961, -32768)"""
        self.assertEqual(str(self.ff_header), target)

    def test_repr(self):
        target = "FFHeader('" + self.filename + "')"
        self.assertEqual(repr(self.ff_header), target)

    def test_valid(self):
        for header in self.valid_headers:
            self.assertTrue(self.ff_header.valid(header))
        for header in self.invalid_headers:
            self.assertFalse(self.ff_header.valid(header))
        with self.assertRaises(AttributeError):
            self.ff_header.valid('foobar')

    def test_address_coverage(self):
        for header in self.valid_headers + self.invalid_headers:
            self.assertIsInstance(self.ff_header.address(header), int)
        with self.assertRaises(AttributeError):
            self.ff_header.address('not_a_real_header')

    def test_address_value(self):
        self.assertEqual(self.ff_header.address('integer_constants'), 2056)
        for header in self.invalid_headers:
            self.assertEqual(self.ff_header.address(header), 0)

    def test_shape(self):
        self.assertEqual(self.ff_header.shape('integer_constants'), (46,))
        self.assertEqual(self.ff_header.shape('data'), (2961, -32768))


@iris.tests.skip_data
class TestFF2PP2Cube(tests.IrisTest):
    def setUp(self):
        self.filename = tests.get_data_path(('FF', 'n48_multi_field'))

    def test_unit_pass_0(self):
        """Test FieldsFile to PPFields cube load."""
        cube_by_name = collections.defaultdict(int)
        cubes = iris.load(self.filename)
        while cubes:
            cube = cubes.pop(0)
            standard_name = cube.standard_name
            cube_by_name[standard_name] += 1
            filename = '{}_{}.cml'.format(standard_name,
                                             cube_by_name[standard_name])
            self.assertCML(cube, ('FF', filename))


class TestFFPayload(tests.IrisTest):
    filename = 'mockery'

    def _test_payload(self, mock_field, expected_depth, expected_type):
        with mock.patch('iris.fileformats.ff.FFHeader') as mock_header:
            mock_header.return_value = None
            ff2pp = ff.FF2PP(self.filename)
            data_depth, data_type = ff2pp._payload(mock_field)
            self.assertEqual(data_depth, expected_depth)
            self.assertEqual(data_type, expected_type)

    def test_payload_unpacked_real(self):
        mock_field = _MOCK_FIELD(lbext=0, lblrec=100, lbnrec=-1,
                                 lbpack=_MOCK_LBPACK(_UNPACKED),
                                 lbuser=[_REAL])
        self._test_payload(mock_field, 800, ff._LBUSER_DTYPE_LOOKUP[_REAL])

    def test_payload_unpacked_real_ext(self):
        mock_field = _MOCK_FIELD(lbext=50, lblrec=100, lbnrec=-1,
                                 lbpack=_MOCK_LBPACK(_UNPACKED),
                                 lbuser=[_REAL])
        self._test_payload(mock_field, 400, ff._LBUSER_DTYPE_LOOKUP[_REAL])

    def test_payload_unpacked_integer(self):
        mock_field = _MOCK_FIELD(lbext=0, lblrec=200, lbnrec=-1,
                                 lbpack=_MOCK_LBPACK(_UNPACKED),
                                 lbuser=[_INTEGER])
        self._test_payload(mock_field, 1600, ff._LBUSER_DTYPE_LOOKUP[_INTEGER])

    def test_payload_unpacked_integer_ext(self):
        mock_field = _MOCK_FIELD(lbext=100, lblrec=200, lbnrec=-1,
                                 lbpack=_MOCK_LBPACK(_UNPACKED),
                                 lbuser=[_INTEGER])
        self._test_payload(mock_field, 800, ff._LBUSER_DTYPE_LOOKUP[_INTEGER])

    def test_payload_wgdos_real(self):
        mock_field = _MOCK_FIELD(lbext=0, lblrec=-1, lbnrec=100,
                                 lbpack=_MOCK_LBPACK(_WGDOS),
                                 lbuser=[_REAL])
        self._test_payload(mock_field, 796, pp.LBUSER_DTYPE_LOOKUP[_REAL])

    def test_payload_wgdos_real_ext(self):
        mock_field = _MOCK_FIELD(lbext=50, lblrec=-1, lbnrec=100,
                                 lbpack=_MOCK_LBPACK(_WGDOS),
                                 lbuser=[_REAL])
        self._test_payload(mock_field, 796, pp.LBUSER_DTYPE_LOOKUP[_REAL])

    def test_payload_wgdos_integer(self):
        mock_field = _MOCK_FIELD(lbext=0, lblrec=-1, lbnrec=200,
                                 lbpack=_MOCK_LBPACK(_WGDOS),
                                 lbuser=[_INTEGER])
        self._test_payload(mock_field, 1596, pp.LBUSER_DTYPE_LOOKUP[_INTEGER])

    def test_payload_wgdos_integer_ext(self):
        mock_field = _MOCK_FIELD(lbext=100, lblrec=-1, lbnrec=200,
                                 lbpack=_MOCK_LBPACK(_WGDOS),
                                 lbuser=[_INTEGER])
        self._test_payload(mock_field, 1596, pp.LBUSER_DTYPE_LOOKUP[_INTEGER])

    def test_payload_cray_real(self):
        mock_field = _MOCK_FIELD(lbext=0, lblrec=100, lbnrec=-1,
                                 lbpack=_MOCK_LBPACK(_CRAY),
                                 lbuser=[_REAL])
        self._test_payload(mock_field, 400, pp.LBUSER_DTYPE_LOOKUP[_REAL])

    def test_payload_cray_real_ext(self):
        mock_field = _MOCK_FIELD(lbext=50, lblrec=100, lbnrec=-1,
                                 lbpack=_MOCK_LBPACK(_CRAY),
                                 lbuser=[_REAL])
        self._test_payload(mock_field, 200, pp.LBUSER_DTYPE_LOOKUP[_REAL])

    def test_payload_cray_integer(self):
        mock_field = _MOCK_FIELD(lbext=0, lblrec=200, lbnrec=-1,
                                 lbpack=_MOCK_LBPACK(_CRAY),
                                 lbuser=[_INTEGER])
        self._test_payload(mock_field, 800, pp.LBUSER_DTYPE_LOOKUP[_INTEGER])

    def test_payload_cray_integer_ext(self):
        mock_field = _MOCK_FIELD(lbext=100, lblrec=200, lbnrec=-1,
                                 lbpack=_MOCK_LBPACK(_CRAY),
                                 lbuser=[_INTEGER])
        self._test_payload(mock_field, 400, pp.LBUSER_DTYPE_LOOKUP[_INTEGER])


if __name__ == '__main__':
    tests.main()
