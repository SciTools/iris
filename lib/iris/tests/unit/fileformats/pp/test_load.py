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
"""Unit tests for the `iris.fileformats.pp.load` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import contextlib
from copy import deepcopy

import mock
import numpy as np

import iris.fileformats.pp as pp


class PPLoadTest(tests.IrisTest):
    @contextlib.contextmanager
    def mock_field_load_context(self, fields):
        side_effect_fields = list(fields)[:]

        def fields_returning_side_effect(*args):
            # Iterates over the fields passed to this context manager,
            # until there are no more, upon which the np.fromfile
            # returns an empty list and the while loop in load() is
            # broken.
            result = side_effect_fields.pop(0)
            if not side_effect_fields:
                np.fromfile.return_value = []
            return result

        with mock.patch('numpy.fromfile', return_value=[0]), \
                mock.patch('__builtin__.open'), \
                mock.patch('struct.unpack_from', return_value=[4]), \
                mock.patch('iris.fileformats.pp.make_pp_field',
                           side_effect=fields_returning_side_effect):
            yield


class Test__interpret_fields__land_packed_fields(PPLoadTest):
    def setUp(self):
        self.pp_field = mock.Mock(lblrec=1, lbext=0, lbuser=[0],
                                  lbrow=0, lbnpt=0,
                                  lbpack=mock.Mock(n2=2))
        self.land_mask_field = mock.Mock(lblrec=1, lbext=0, lbuser=[0],
                                         lbrow=3, lbnpt=4,
                                         stash='m01s00i030',
                                         data=np.empty((3, 4)))

    def test_non_deferred_fix_lbrow_lbnpt(self):
        # Checks the fix_lbrow_lbnpt is applied to fields which are not
        # deferred.
        f1, mask = self.pp_field, self.land_mask_field
        self.assertEqual(f1.lbrow, 0)
        self.assertEqual(f1.lbnpt, 0)
        list(pp._interpret_fields([mask, f1]))
        self.assertEqual(f1.lbrow, 3)
        self.assertEqual(f1.lbnpt, 4)
        # Check the data manager's shape has been updated too.
        self.assertEqual(f1._data_manager._orig_data_shape, (3, 4))

    def test_fix_lbrow_lbnpt_no_mask_available(self):
#        with self.assertRaises(ValueError):
        with mock.patch('warnings.warn') as warn:
            list(pp._interpret_fields([self.pp_field]))
        self.assertEqual(warn.call_count, 1)
        warn_msg = warn.call_args[0][0]
        self.assertTrue(warn_msg.startswith('Landmask compressed fields '
                                            'existed without a landmask'),
                        'Unexpected warning message: {!r}'.format(warn_msg))

    def test_deferred_mask_field(self):
        # Check that the order of the load is yielded last if the mask
        # hasn't yet been seen.
        result = list(pp._interpret_fields([self.pp_field,
                                            self.land_mask_field]))
        self.assertEqual(result, [self.land_mask_field, self.pp_field])

    def test_not_deferred_mask_field(self):
        # Check that the order of the load is unchanged if a land mask
        # has already been seen.
        f1, mask = self.pp_field, self.land_mask_field
        mask2 = deepcopy(mask)
        result = list(pp._interpret_fields([mask, f1, mask2]))
        self.assertEqual(result, [mask, f1, mask2])

    def test_deferred_fix_lbrow_lbnpt(self):
        # Check the fix is also applied to fields which are deferred.
        f1, mask = self.pp_field, self.land_mask_field
        self.assertEqual(f1.lbrow, 0)
        self.assertEqual(f1.lbnpt, 0)
        list(pp._interpret_fields([mask, f1]))
        self.assertEqual(f1.lbrow, 3)
        self.assertEqual(f1.lbnpt, 4)

    def test_shared_land_mask_field(self):
        # Check that multiple land masked fields share the
        # land mask field instance.
        f1 = deepcopy(self.pp_field)
        f2 = deepcopy(self.pp_field)
        self.assertIsNot(f1, f2)
        list(pp._interpret_fields([f1, self.land_mask_field, f2]))
        self.assertIs(f1._data.item().mask,
                      f2._data.item().mask)


class Test__field_gen(PPLoadTest):
    def mock_field_gen(self, fields):
        with self.mock_field_load_context(fields):
            return list(pp._field_gen('mocked', 'mocked'))

    def test_lblrec_invalid(self):
        pp_field = mock.Mock(lblrec=2,
                             lbext=0)
        with self.assertRaises(ValueError) as err:
            self.mock_field_gen([pp_field])
        self.assertEqual(str(err.exception),
                         ('LBLREC has a different value to the integer '
                          'recorded after the header in the file (8 '
                          'and 4).'))

    def test_read_headers_call(self):
        # Checks that the two calls to np.fromfile are called in the
        # expected way.
        pp_field = mock.Mock(lblrec=1,
                             lbext=0,
                             lbuser=[0])
        with self.mock_field_load_context([pp_field]):
            open_fh = mock.Mock()
            open.return_value = open_fh
            self.mock_field_gen([pp_field])
            next(pp.load('mocked'))
            calls = [mock.call(open_fh, count=45, dtype='>i4'),
                     mock.call(open_fh, count=19, dtype='>f4')]
            np.fromfile.assert_has_calls(calls)


class Test_load(tests.IrisTest):
    def test_call_structure(self):
        # Check that the load function calls the two necessary utility
        # functions.
        extract_result = mock.Mock()
        with mock.patch('iris.fileformats.pp._interpret_fields',
                        autospec=True, return_value=iter([])) as interpret:
            with mock.patch('iris.fileformats.pp._field_gen', autospec=True,
                            return_value=extract_result) as _field_gen:
                pp.load('mock', read_data=True)

        interpret.assert_called_once_with(extract_result)
        _field_gen.assert_called_once_with('mock', read_data_bytes=True)


if __name__ == "__main__":
    tests.main()
