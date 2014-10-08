# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for the :class:`iris.fileformat.ff.FF2PP` class."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import contextlib

import mock
import numpy as np

from iris.fileformats.ff import FF2PP
import iris.fileformats.ff as ff
import iris.fileformats.pp as pp


class Test____iter__(tests.IrisTest):
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


class Test__extract_field__LBC_format(tests.IrisTest):
    @contextlib.contextmanager
    def mock_for_extract_field(self, fields, x=None, y=None):
        """
        A context manager to ensure FF2PP._extract_field gets a field
        instance looking like the next one in the "fields" iterable from
        the "make_pp_field" call.

        """
        with mock.patch('iris.fileformats.ff.FFHeader'):
            ff2pp = ff.FF2PP('mock')
        ff2pp._ff_header.lookup_table = [0, 0, len(fields)]
        grid = mock.Mock()
        grid.vectors = mock.Mock(return_value=(x, y))
        ff2pp._ff_header.grid = mock.Mock(return_value=grid)

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
        # stash m01s00i001
        lbuser = [None, None, 121416, 1, None, None, 1]
        field = mock.Mock(lbegin=0,
                          lbrow=10, lbnpt=12, bdx=1, bdy=1, bzx=bzx, bzy=bzy,
                          lbuser=lbuser)
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
        field.lbrow = 10
        field.lbnpt = 12
        # stash m01s31i020
        field.lbuser = [None, None, 121416, 20, None, None, 1]
        orig_bdx, orig_bdy = field.bdx, field.bdy

        x = np.array([1, 2, 6])
        y = np.array([1, 2, 6])
        with self.mock_for_extract_field([field], x, y) as ff2pp:
            ff2pp._ff_header.dataset_type = 5
            with mock.patch('warnings.warn') as warn:
                list(ff2pp._extract_field())

        # Check the values are unchanged.
        self.assertEqual(field.bdy, orig_bdy)
        self.assertEqual(field.bdx, orig_bdx)

        # Check a warning was raised with a suitable message.
        warn_error_tmplt = 'Unexpected warning message: {}'
        non_trivial_coord_warn_msg = warn.call_args[0][0]
        msg = ('The x or y coordinates of your boundary condition field may '
               'be incorrect, not having taken into account the boundary '
               'size.')
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
                          lbuser=[0, 0, 121416, 0, None, None, 0],
                          lbrow=10, lbnpt=12)
        with self.mock_for_extract_field([field]) as ff2pp:
            ff2pp._ff_header.dataset_type = 5
            with mock.patch('warnings.warn') as warn:
                list(ff2pp._extract_field())
        msg = 'The LBC has a bdy less than 0.'
        self.assertTrue(warn.call_args[0][0].startswith(msg),
                        'Northwards bdy warning not correctly raised.')


class Test__det_border(tests.IrisTest):
    def setUp(self):
        _FFH_patch = mock.patch('iris.fileformats.ff.FFHeader')
        _FFH_patch.start()
        self.addCleanup(_FFH_patch.stop)

    def test_unequal_spacing_eitherside(self):
        # Ensure that we do not interpret the case where there is not the same
        # spacing on the lower edge as the upper edge.
        ff2pp = FF2PP('dummy')
        field_x = np.array([1, 2, 10])

        msg = ('The x or y coordinates of your boundary condition field may '
               'be incorrect, not having taken into account the boundary '
               'size.')

        with mock.patch('warnings.warn') as warn:
            result = ff2pp._det_border(field_x, None)
        warn.assert_called_with(msg)
        self.assertIs(result, field_x)

    def test_increasing_field_values(self):
        # Field where its values a increasing.
        ff2pp = FF2PP('dummy')
        field_x = np.array([1, 2, 3])
        com = np.array([0, 1, 2, 3, 4])
        result = ff2pp._det_border(field_x, 1)
        self.assertArrayEqual(result, com)

    def test_decreasing_field_values(self):
        # Field where its values a decreasing.
        ff2pp = FF2PP('dummy')
        field_x = np.array([3, 2, 1])
        com = np.array([4, 3, 2, 1, 0])
        result = ff2pp._det_border(field_x, 1)
        self.assertArrayEqual(result, com)


if __name__ == "__main__":
    tests.main()
