# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the :class:`iris.fileformat.FF2PP` class instance."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import contextlib

import mock
import numpy as np
import warnings

import iris.fileformats.ff as ff
import iris.fileformats.pp as pp


class Test_FF2PP___iter__(tests.IrisTest):
    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_call_structure(self, _FFHeader):
        # Check that the iter method calls the two necessary utility
        # functions
        extract_result = mock.Mock()
        interpret_patch = mock.patch('iris.fileformats.pp._interpret_fields',
                                     autospec=True, return_value=iter([]))
        extract_patch = mock.patch('iris.fileformats.ff.FF2PP._extract_field',
                                   autospec=True, return_value=extract_result)

        FF2PP_instance = ff.FF2PP('mock')
        with interpret_patch as interpret, extract_patch as extract:
            list(iter(FF2PP_instance))

        interpret.assert_called_once_with(extract_result)
        extract.assert_called_once_with(FF2PP_instance)


class Test_FF2PP__extract_field__LBC_format(tests.IrisTest):
    @contextlib.contextmanager
    def mock_for_extract_field(self, fields):
        """
        A context manager to ensure FF2PP._extract_field gets a field
        instance looking like the next one in the "fields" iterable from
        the "make_pp_field" call.

        """
        with mock.patch('iris.fileformats.ff.FFHeader'):
            ff2pp = ff.FF2PP('mock')
            ff2pp._ff_header.lookup_table = [0, 0, len(fields)]

        with mock.patch('numpy.fromfile', return_value=[0]), \
                mock.patch('__builtin__.open'), \
                mock.patch('struct.unpack_from', return_value=[4]), \
                mock.patch('iris.fileformats.pp.make_pp_field',
                           side_effect=fields), \
                mock.patch('iris.fileformats.ff.FF2PP._payload',
                           return_value=(0, 0)):
            yield ff2pp

    def test_LBC_header(self):
        bzx, bzy = -10, 15
        field = mock.Mock(lbegin=0, stash='m01s00i001',
                          lbrow=10, lbnpt=12, bdx=1, bdy=1, bzx=bzx, bzy=bzy,
                          lbuser=[None, None, 121416])
        with self.mock_for_extract_field([field]) as ff2pp:
            ff2pp._ff_header.dataset_type = 5
            result = list(ff2pp._extract_field())

        self.assertEqual([field], result)
        self.assertEqual(field.lbrow, 10 + 14 * 2)
        self.assertEqual(field.lbnpt, 12 + 16 * 2)

        name_mapping_dict = dict(rim_width=slice(4, 6), y_halo=slice(2, 4),
                                 x_halo=slice(0, 2))
        boundary_packing = pp.SplittableInt(121416, name_mapping_dict)

        self.assertEqual(field.lbpack.boundary_packing, boundary_packing)
        self.assertEqual(field.bzy, bzy - boundary_packing.y_halo * field.bdy)
        self.assertEqual(field.bzx, bzx - boundary_packing.x_halo * field.bdx)

    def check_non_trivial_coordinate_warning(self, field):
        field.lbegin = 0
        field.stash = 'm01s31i020'
        field.lbrow = 10
        field.lbnpt = 12
        field.lbuser = [None, None, 121416]
        orig_bdx, orig_bdy = field.bdx, field.bdy

        with self.mock_for_extract_field([field]) as ff2pp:
            ff2pp._ff_header.dataset_type = 5
            with mock.patch('warnings.warn') as warn:
                list(ff2pp._extract_field())

        # Check the values are unchanged.
        self.assertEqual(field.bdy, orig_bdy)
        self.assertEqual(field.bdx, orig_bdx)

        # Check a warning was raised with a suitable message.
        warn_error_tmplt = 'Unexpected warning message: {}'
        non_trivial_coord_warn_msg = warn.call_args[0][0]
        msg = 'The LBC field has non-trivial x or y coordinates'
        self.assertTrue(non_trivial_coord_warn_msg.startswith(msg),
                        warn_error_tmplt.format(non_trivial_coord_warn_msg))

    def test_LBC_header_non_trivial_coords_both(self):
        # Check a warning is raised when both bdx and bdy are bad.
        field = mock.Mock(bdx=0, bdy=0, bzx=10, bzy=10)
        self.check_non_trivial_coordinate_warning(field)

        field.bdy = field.bdx = field.bmdi
        self.check_non_trivial_coordinate_warning(field)

    def test_LBC_header_non_trivial_coords_x(self):
        # Check a warning is raised when bdx is bad.
        field = mock.Mock(bdx=0, bdy=10, bzx=10, bzy=10)
        self.check_non_trivial_coordinate_warning(field)

        field.bdx = field.bmdi
        self.check_non_trivial_coordinate_warning(field)

    def test_LBC_header_non_trivial_coords_y(self):
        # Check a warning is raised when bdy is bad.
        field = mock.Mock(bdx=10, bdy=0, bzx=10, bzy=10)
        self.check_non_trivial_coordinate_warning(field)

        field.bdy = field.bmdi
        self.check_non_trivial_coordinate_warning(field)

    def test_negative_bdy(self):
        # Check a warning is raised when bdy is negative,
        # we don't yet know what "north" means in this case.
        field = mock.Mock(bdx=10, bdy=-10, bzx=10, bzy=10, lbegin=0,
                          lbuser=[0, 0, 121416], lbrow=10, lbnpt=12)
        with self.mock_for_extract_field([field]) as ff2pp:
            ff2pp._ff_header.dataset_type = 5
            with mock.patch('warnings.warn') as warn:
                list(ff2pp._extract_field())
        msg = 'The LBC has a bdy less than 0.'
        self.assertTrue(warn.call_args[0][0].startswith(msg),
                        'Northwards bdy warning not correctly raised.')


class Test__det_typeC_vpole_grid_coord(tests.IrisTest):
    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_no_col_dep_no_row_dep(self, _FFHEADER):
        # No column/row dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        FF2PP_instance._ff_header.column_dependent_constants = None
        FF2PP_instance._ff_header.row_dependent_constants = None

        res = FF2PP_instance._det_typeC_vpole_grid_coord()
        com = (None, None, None, None)
        self.assertEqual(res, com)

    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_yes_col_dep_no_row_dep_1dim(self, _FFHEADER):
        # 1-D column dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        col_cons = np.array([[0], [1], [2], [3]])
        row_cons = None
        FF2PP_instance._ff_header.column_dependent_constants = col_cons
        FF2PP_instance._ff_header.row_dependent_constants = row_cons

        res = FF2PP_instance._det_typeC_vpole_grid_coord()
        com = (np.array([0, 1, 2, 3]), None, None, None)
        for coord_ind in range(len(res)):
            self.assertArrayEqual(res[coord_ind], com[coord_ind])

    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_no_col_dep_yes_row_dep_1dim(self, _FFHEADER):
        # 1-D row dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        col_cons = None
        row_cons = np.array([[0], [1], [2], [3]])
        FF2PP_instance._ff_header.column_dependent_constants = col_cons
        FF2PP_instance._ff_header.row_dependent_constants = row_cons

        res = FF2PP_instance._det_typeC_vpole_grid_coord()
        com = (None, np.array([0, 1, 2]), None, None)
        for coord_ind in range(len(res)):
            self.assertArrayEqual(res[coord_ind], com[coord_ind])

    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_yes_col_dep_no_row_dep_2dim(self, _FFHEADER):
        # 2-D column dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        col_cons = np.array([[0, 0], [1, 10], [2, 20], [3, 30]])
        row_cons = None
        FF2PP_instance._ff_header.column_dependent_constants = col_cons
        FF2PP_instance._ff_header.row_dependent_constants = row_cons

        res = FF2PP_instance._det_typeC_vpole_grid_coord()
        com = (np.array([0, 1, 2, 3]), None,
               np.array([0, 10, 20, 30]), None)
        for coord_ind in range(len(res)):
            self.assertArrayEqual(res[coord_ind], com[coord_ind])

    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_no_col_dep_yes_row_dep_2dim(self, _FFHEADER):
        # 2-D row dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        col_cons = None
        row_cons = np.array([[0, 0], [1, 10], [2, 20], [3, 30]])
        FF2PP_instance._ff_header.column_dependent_constants = col_cons
        FF2PP_instance._ff_header.row_dependent_constants = row_cons

        res = FF2PP_instance._det_typeC_vpole_grid_coord()
        com = (None, np.array([0, 1, 2]),
               None, np.array([0, 10, 20, 30]))
        for coord_ind in range(len(res)):
            self.assertArrayEqual(res[coord_ind], com[coord_ind])


class Test__det_typeC_grid_coord(tests.IrisTest):
    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_no_col_dep_no_row_dep(self, _FFHEADER):
        # No column/row dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        FF2PP_instance._ff_header.column_dependent_constants = None
        FF2PP_instance._ff_header.row_dependent_constants = None

        res = FF2PP_instance._det_typeC_grid_coord()
        com = (None, None, None, None)
        self.assertEqual(res, com)

    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_yes_col_dep_no_row_dep_1dim(self, _FFHEADER):
        # 1-D column dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        col_cons = np.array([[0], [1], [2], [3]])
        row_cons = None
        FF2PP_instance._ff_header.column_dependent_constants = col_cons
        FF2PP_instance._ff_header.row_dependent_constants = row_cons

        res = FF2PP_instance._det_typeC_grid_coord()
        com = (np.array([0, 1, 2, 3]), None, None, None)
        for coord_ind in range(len(res)):
            self.assertArrayEqual(res[coord_ind], com[coord_ind])

    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_no_col_dep_yes_row_dep_1dim(self, _FFHEADER):
        # 1-D row dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        col_cons = None
        row_cons = np.array([[0], [1], [2], [3]])
        FF2PP_instance._ff_header.column_dependent_constants = col_cons
        FF2PP_instance._ff_header.row_dependent_constants = row_cons

        res = FF2PP_instance._det_typeC_grid_coord()
        com = (None, np.array([0, 1, 2, 3]), None, None)
        for coord_ind in range(len(res)):
            self.assertArrayEqual(res[coord_ind], com[coord_ind])

    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_yes_col_dep_no_row_dep_2dim(self, _FFHEADER):
        # 2-D column dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        col_cons = np.array([[0, 0], [1, 10], [2, 20], [3, 30]])
        row_cons = None
        FF2PP_instance._ff_header.column_dependent_constants = col_cons
        FF2PP_instance._ff_header.row_dependent_constants = row_cons

        res = FF2PP_instance._det_typeC_grid_coord()
        com = (np.array([0, 1, 2, 3]), None,
               np.array([0, 10, 20, 30]), None)
        for coord_ind in range(len(res)):
            self.assertArrayEqual(res[coord_ind], com[coord_ind])

    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_no_col_dep_yes_row_dep_2dim(self, _FFHEADER):
        # 2-D row dependent constants present.
        FF2PP_instance = ff.FF2PP('mock')
        col_cons = None
        row_cons = np.array([[0, 0], [1, 10], [2, 20], [3, 30]])
        FF2PP_instance._ff_header.column_dependent_constants = col_cons
        FF2PP_instance._ff_header.row_dependent_constants = row_cons

        res = FF2PP_instance._det_typeC_grid_coord()
        com = (None, np.array([0, 1, 2, 3]),
               None, np.array([0, 10, 20]))
        for coord_ind in range(len(res)):
            self.assertArrayEqual(res[coord_ind], com[coord_ind])


class Test__extract_field__grid_staggering(tests.IrisTest):
    def setUp(self):
        _ppfield_patch = mock.patch('iris.fileformats.pp.make_pp_field')
        _ppfield_patch.start()
        self.addCleanup(_ppfield_patch.stop)

        _open_patch = mock.patch('__builtin__.open')
        _open_patch.start()
        self.addCleanup(_open_patch.stop)

        _fromfile_patch = mock.patch('numpy.fromfile', return_value=[0])
        _fromfile_patch.start()
        self.addCleanup(_fromfile_patch.stop)

        _FFHEADER_patch = mock.patch('iris.fileformats.ff.FFHeader')
        _FFHEADER_patch.start()
        self.addCleanup(_FFHEADER_patch.stop)

        _typeC_grid_patch = mock.patch(
            'iris.fileformats.ff.FF2PP._det_typeC_grid_coord',
            return_value=(None, None, None, None))
        _typeC_grid_patch.start()
        self.addCleanup(_typeC_grid_patch.stop)

        _typeC_grid_vpole_patch = mock.patch(
            'iris.fileformats.ff.FF2PP._det_typeC_vpole_grid_coord',
            return_value=(None, None, None, None))
        _typeC_grid_vpole_patch.start()
        self.addCleanup(_typeC_grid_vpole_patch.stop)

        FF2PP_instance = ff.FF2PP('mock')
        FF2PP_instance._ff_header.lookup_table = (0, 0, 1)
        FF2PP_instance._payload = mock.Mock(return_value=(0, 0))
        self.FF2PP_instance = FF2PP_instance

    def test_no_grid_staggering(self):
        # No grid staggering specified.
        self.FF2PP_instance._ff_header.grid_staggering = None

        res = self.FF2PP_instance._extract_field()
        warning_msg = ('Staggered grid type: None not currently interpreted, '
                       'assuming standard C-grid')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res.next()
            self.assertTrue(warning_msg in [str(warn.message) for warn in w])
            index = [str(warn.message) for warn in w].index(warning_msg)
            self.assertTrue(issubclass(w[index].category, UserWarning))
        self.assertFalse(ff.FF2PP._det_typeC_vpole_grid_coord.called)
        ff.FF2PP._det_typeC_grid_coord.assert_called_once_with()

    def test_grid_stagger_eq3(self):
        # Arakawa type C grid.
        self.FF2PP_instance._ff_header.grid_staggering = 3

        res = self.FF2PP_instance._extract_field()
        res.next()
        self.assertFalse(ff.FF2PP._det_typeC_vpole_grid_coord.called)
        ff.FF2PP._det_typeC_grid_coord.assert_called_once_with()

    def test_grid_stagger_eq6(self):
        # Arakawa type C grid with v at pole.
        self.FF2PP_instance._ff_header.grid_staggering = 6

        res = self.FF2PP_instance._extract_field()
        res.next()
        self.assertFalse(ff.FF2PP._det_typeC_grid_coord.called)
        ff.FF2PP._det_typeC_vpole_grid_coord.assert_called_once_with()


if __name__ == "__main__":
    tests.main()
