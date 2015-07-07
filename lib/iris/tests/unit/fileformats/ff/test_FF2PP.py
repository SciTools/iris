# (C) British Crown Copyright 2013 - 2015, Met Office
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
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import contextlib
import copy
import mock
import numpy as np
import warnings

from iris.exceptions import NotYetImplementedError
import iris.fileformats.ff as ff
import iris.fileformats.pp as pp

from iris.fileformats.ff import FF2PP


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
        # Fake level constants, with shape specifying just one model-level.
        ff2pp._ff_header.level_dependent_constants = np.zeros(1)
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

    def _mock_lbc(self, **kwargs):
        """Return a Mock object representing an LBC field."""
        # Default kwargs for a valid LBC field mapping just 1 model-level.
        field_kwargs = dict(lbtim=0, lblev=7777, lbvc=0, lbhem=101)
        # Apply provided args (replacing any defaults if specified).
        field_kwargs.update(kwargs)
        # Return a mock with just those properties pre-defined.
        return mock.Mock(**field_kwargs)

    def test_LBC_header(self):
        bzx, bzy = -10, 15
        # stash m01s00i001
        lbuser = [None, None, 121416, 1, None, None, 1]
        field = self._mock_lbc(lbegin=0,
                               lbrow=10, lbnpt=12,
                               bdx=1, bdy=1, bzx=bzx, bzy=bzy,
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

        self.assertEqual(field.boundary_packing, boundary_packing)
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
        field = self._mock_lbc(bdx=0, bdy=0, bzx=10, bzy=10)
        self.check_non_trivial_coordinate_warning(field)

        field.bdy = field.bdx = field.bmdi
        self.check_non_trivial_coordinate_warning(field)

    def test_LBC_header_non_trivial_coords_x(self):
        # Check a warning is raised when bdx is bad.
        field = self._mock_lbc(bdx=0, bdy=10, bzx=10, bzy=10)
        self.check_non_trivial_coordinate_warning(field)

        field.bdx = field.bmdi
        self.check_non_trivial_coordinate_warning(field)

    def test_LBC_header_non_trivial_coords_y(self):
        # Check a warning is raised when bdy is bad.
        field = self._mock_lbc(bdx=10, bdy=0, bzx=10, bzy=10)
        self.check_non_trivial_coordinate_warning(field)

        field.bdy = field.bmdi
        self.check_non_trivial_coordinate_warning(field)

    def test_negative_bdy(self):
        # Check a warning is raised when bdy is negative,
        # we don't yet know what "north" means in this case.
        field = self._mock_lbc(bdx=10, bdy=-10, bzx=10, bzy=10, lbegin=0,
                               lbuser=[0, 0, 121416, 0, None, None, 0],
                               lbrow=10, lbnpt=12)
        with self.mock_for_extract_field([field]) as ff2pp:
            ff2pp._ff_header.dataset_type = 5
            with mock.patch('warnings.warn') as warn:
                list(ff2pp._extract_field())
        msg = 'The LBC has a bdy less than 0.'
        self.assertTrue(warn.call_args[0][0].startswith(msg),
                        'Northwards bdy warning not correctly raised.')


class Test__payload(tests.IrisTest):
    def setUp(self):
        # Patch FFHeader to produce a mock header instead of opening a file.
        self.mock_ff_header = mock.Mock()
        self.mock_ff = self.patch('iris.fileformats.ff.FFHeader',
                                  return_value=self.mock_ff_header)
        # Make it look like a 32-bit LBC file.
        self.mock_ff_header.word_depth = 8
        self.mock_ff_header.dataset_type = 5

        # Create a mock LBC type PPField.
        self.mock_field = mock.Mock()
        field = self.mock_field
        field.raw_lbpack = 0
        field.lbuser = [0]
        field.lblrec = 777
        field.lbext = 222
        field.lbnrec = 50
        field.boundary_packing = None

    def test__basic(self):
        ff2pp = FF2PP('dummy_filename')
        field = self.mock_field
        data_depth, data_type = ff2pp._payload(field)
        self.assertEqual(data_depth, 4440)
        self.assertEqual(data_type, np.dtype('>f8'))

    def test__word_depth(self):
        ff2pp = FF2PP('dummy_filename', word_depth=4)
        field = self.mock_field
        data_depth, data_type = ff2pp._payload(field)
        self.assertEqual(data_depth, 2220)
        self.assertEqual(data_type, np.dtype('>f4'))

    def test__lbpack_1(self):
        ff2pp = FF2PP('dummy_filename')
        field = self.mock_field
        field.raw_lbpack = 1
        data_depth, data_type = ff2pp._payload(field)
        self.assertEqual(data_depth, 396)
        self.assertEqual(data_type, np.dtype('>f4'))

    def test__lbpack_2(self):
        ff2pp = FF2PP('dummy_filename')
        field = self.mock_field
        field.raw_lbpack = 2
        data_depth, data_type = ff2pp._payload(field)
        self.assertEqual(data_depth, 2220)
        self.assertEqual(data_type, np.dtype('>f4'))

    def test__lbpack_unsupported(self):
        ff2pp = FF2PP('dummy_filename')
        field = self.mock_field
        field.raw_lbpack = 1239
        with self.assertRaisesRegexp(
                NotYetImplementedError,
                'PP fields with LBPACK of 1239 are not supported.'):
            ff2pp._payload(field)

    def test__lbpack_lbc_error(self):
        ff2pp = FF2PP('dummy_filename')
        field = self.mock_field
        field.raw_lbpack = 1
        field.boundary_packing = 0  # Anything not None will do here.
        with self.assertRaisesRegexp(
                ValueError,
                'packed LBC data is not supported'):
            ff2pp._payload(field)


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


class Test__adjust_field_for_lbc(tests.IrisTest):
    def setUp(self):
        # Patch FFHeader to produce a mock header instead of opening a file.
        self.mock_ff_header = mock.Mock()
        self.mock_ff_header.dataset_type = 5
        self.mock_ff = self.patch('iris.fileformats.ff.FFHeader',
                                  return_value=self.mock_ff_header)

        # Create a mock LBC type PPField.
        self.mock_field = mock.Mock()
        field = self.mock_field
        field.lbtim = 0
        field.lblev = 7777
        field.lbvc = 0
        field.lbnpt = 1001
        field.lbrow = 2001
        field.lbuser = (None, None, 80504)
        field.lbpack = pp.SplittableInt(0)
        field.boundary_packing = None
        field.bdx = 1.0
        field.bzx = 0.0
        field.bdy = 1.0
        field.bzy = 0.0

    def test__basic(self):
        ff2pp = FF2PP('dummy_filename')
        field = self.mock_field
        ff2pp._adjust_field_for_lbc(field)
        self.assertEqual(field.lbtim, 11)
        self.assertEqual(field.lbvc, 65)
        self.assertEqual(field.boundary_packing.rim_width, 8)
        self.assertEqual(field.boundary_packing.y_halo, 5)
        self.assertEqual(field.boundary_packing.x_halo, 4)
        self.assertEqual(field.lbnpt, 1009)
        self.assertEqual(field.lbrow, 2011)

    def test__bad_lbtim(self):
        self.mock_field.lbtim = 717
        ff2pp = FF2PP('dummy_filename')
        with self.assertRaisesRegexp(ValueError,
                                     'LBTIM of 717, expected only 0 or 11'):
            ff2pp._adjust_field_for_lbc(self.mock_field)

    def test__bad_lbvc(self):
        self.mock_field.lbvc = 312
        ff2pp = FF2PP('dummy_filename')
        with self.assertRaisesRegexp(ValueError,
                                     'LBVC of 312, expected only 0 or 65'):
            ff2pp._adjust_field_for_lbc(self.mock_field)


class Test__fields_over_all_levels(tests.IrisTest):
    def setUp(self):
        # Patch FFHeader to produce a mock header instead of opening a file.
        self.mock_ff_header = mock.Mock()
        self.mock_ff_header.dataset_type = 5

        # Fake the level constants to look like 3 model levels.
        self.n_all_levels = 3
        self.mock_ff_header.level_dependent_constants = \
            np.zeros((self.n_all_levels))
        self.mock_ff = self.patch('iris.fileformats.ff.FFHeader',
                                  return_value=self.mock_ff_header)

        # Create a simple mock for a test field.
        self.mock_field = mock.Mock()
        field = self.mock_field
        field.lbhem = 103
        self.original_lblev = mock.sentinel.untouched_lbev
        field.lblev = self.original_lblev

    def _check_expected_levels(self, results, n_levels):
        if n_levels is 0:
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].lblev, self.original_lblev)
        else:
            self.assertEqual(len(results), n_levels)
            self.assertEqual([fld.lblev for fld in results],
                             list(range(n_levels)))

    def test__is_lbc(self):
        ff2pp = FF2PP('dummy_filename')
        field = self.mock_field
        results = list(ff2pp._fields_over_all_levels(field))
        self._check_expected_levels(results, 3)

    def test__lbhem_too_small(self):
        ff2pp = FF2PP('dummy_filename')
        field = self.mock_field
        field.lbhem = 100
        with self.assertRaisesRegexp(
                ValueError,
                'hence >= 101'):
            results = list(ff2pp._fields_over_all_levels(field))

    def test__lbhem_too_large(self):
        ff2pp = FF2PP('dummy_filename')
        field = self.mock_field
        field.lbhem = 105
        with self.assertRaisesRegexp(
                ValueError,
                'more than the total number of levels in the file = 3'):
            results = list(ff2pp._fields_over_all_levels(field))


if __name__ == "__main__":
    tests.main()
