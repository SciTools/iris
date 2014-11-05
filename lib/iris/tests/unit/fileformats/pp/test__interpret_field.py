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
"""Unit tests for the `iris.fileformats.pp._interpret_field` function."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from copy import deepcopy

import mock

import iris.fileformats.pp as pp


class Test__interpret_fields__land_packed_fields(tests.IrisTest):
    def setUp(self):
        # A field packed using a land/sea mask.
        self.pp_field = mock.Mock(lblrec=1, lbext=0, lbuser=[0] * 7,
                                  lbrow=0, lbnpt=0,
                                  raw_lbpack=20,
                                  _my_data=('dummy', 0, 0, 0))
        # The field specifying the land/seamask.
        lbuser = [None, None, None, 30, None, None, 1]  # m01s00i030
        self.land_mask_field = mock.Mock(lblrec=1, lbext=0, lbuser=lbuser,
                                         lbrow=3, lbnpt=4,
                                         raw_lbpack=0,
                                         _my_data=('dummy', 0, 0, 0))

    def test_non_deferred_fix_lbrow_lbnpt(self):
        # Checks the fix_lbrow_lbnpt is applied to fields which are not
        # deferred.
        f1, mask = self.pp_field, self.land_mask_field
        self.assertEqual(f1.lbrow, 0)
        self.assertEqual(f1.lbnpt, 0)
        list(pp._interpret_fields([mask, f1]))
        self.assertEqual(f1.lbrow, 3)
        self.assertEqual(f1.lbnpt, 4)
        # Check the data's shape has been updated too.
        self.assertEqual(f1._my_data.shape, (3, 4))

    def test_fix_lbrow_lbnpt_no_mask_available(self):
        # Check a warning is issued when loading a land masked field
        # without a land mask.
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
        list(pp._interpret_fields([f1, mask]))
        self.assertEqual(f1.lbrow, 3)
        self.assertEqual(f1.lbnpt, 4)

    def test_shared_land_mask_field(self):
        # Check that multiple land masked fields share the
        # land mask field instance.
        f1 = deepcopy(self.pp_field)
        f2 = deepcopy(self.pp_field)
        self.assertIsNot(f1, f2)
        with mock.patch('iris.fileformats.pp.PPDataProxy') as PPDataProxy:
            PPDataProxy.return_value = mock.MagicMock()
            list(pp._interpret_fields([f1, self.land_mask_field, f2]))
        for call in PPDataProxy.call_args_list:
            positional_args = call[0]
            self.assertIs(positional_args[7], self.land_mask_field)


if __name__ == "__main__":
    tests.main()
