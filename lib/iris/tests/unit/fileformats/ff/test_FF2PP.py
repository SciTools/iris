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
"""Unit tests for the `iris.fileformats.ff._read_data` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from copy import deepcopy

import mock
import numpy as np

import iris.fileformats.ff as ff
import iris.fileformats.pp as pp


class Test_FF2PP_extract_field(tests.IrisTest):
    def setUp(self):
        self.pp_field = mock.Mock(name='PP field', lblrec=1, lbext=0,
                                  lbuser=[0], lbrow=0, lbnpt=0, lbegin=0,
                                  lbpack=self.create_lbpack(120),
                                  stash='m01s00i024')
        self.land_mask_field = mock.Mock(name='Land mask PP field',
                                         lblrec=1, lbext=0, lbuser=[0],
                                         stash='m01s00i030', lbegin=0,
                                         lbrow=3, lbnpt=4,
                                         data=np.empty((3, 4)))
        with mock.patch('iris.fileformats.ff.FFHeader'):
            self.ff2pp = ff.FF2PP(None, False, 4)
        self.ff2pp._ff_header.lookup_table = [1, 2, 3]

    def create_lbpack(self, value):
        name_mapping = dict(n5=slice(4, None), n4=3, n3=2, n2=1, n1=0)
        return pp.SplittableInt(value, name_mapping)

    def mock_field_load(self, header, fields):
        side_effect_fields = list(fields)[:]

        def fields_returning_side_effect(*args):
            # Iterates over the fields passed to this context manager,
            # until there are no more, upon which the np.fromfile
            # returns an empty list and the while loop in load() is
            # broken.
            result = side_effect_fields.pop(0)
            if not side_effect_fields:
                np.fromfile.return_value = [ff._FF_LOOKUP_TABLE_TERMINATE]
            return result

        with mock.patch('numpy.fromfile', return_value=(0, )), \
                mock.patch('__builtin__.open'), \
                mock.patch('struct.unpack_from', return_value=[4]), \
                mock.patch('iris.fileformats.ff.FF2PP._payload',
                           return_value=[4, '>f4']), \
                mock.patch('iris.fileformats.pp.make_pp_field',
                           side_effect=fields_returning_side_effect):
            return list(self.ff2pp._extract_field())

    def test_no_land_mask(self):
        # Check behaviour of loading a file containing a mask packed field
        # but which has no mask.
        fields_header = mock.Mock()
        fields = [self.pp_field]
        with self.assertRaises(ValueError):
            self.mock_field_load(fields_header, fields)

    def test_non_deferred_fix_lbrow_lbnpt(self):
        # Checks the fix_lbrow_lbnpt is applied to fields which are not
        # deferred.
        fields_header = mock.Mock()
        f1 = self.pp_field
        fields = [self.land_mask_field, f1]
        self.assertEqual(f1.lbrow, 0)
        self.assertEqual(f1.lbnpt, 0)
        self.assertEqual([self.land_mask_field, f1],
                         self.mock_field_load(fields_header, fields))
        self.assertEqual(f1.lbrow, 3)
        self.assertEqual(f1.lbnpt, 4)
        # Check the data manager's shape has been updated too.
        self.assertEqual(f1._data_manager._orig_data_shape, (3, 4))

    def test_deferred_fix_lbrow_lbnpt(self):
        # Check the fix is also applied to fields which are deferred.
        fields_header = mock.Mock()
        f1 = self.pp_field
        self.assertEqual(f1.lbrow, 0)
        self.assertEqual(f1.lbnpt, 0)
        self.mock_field_load(fields_header, [f1, self.land_mask_field])
        self.assertEqual(f1.lbrow, 3)
        self.assertEqual(f1.lbnpt, 4)

    def test_shared_land_mask_instance(self):
        # Check that multiple land masked fields share the
        # _LandMask instance.
        f1 = deepcopy(self.pp_field)
        f2 = deepcopy(self.pp_field)
        self.assertIsNot(f1, f2)
        self.mock_field_load(mock.Mock(), [f1, self.land_mask_field, f2])
        self.assertIs(f1._data.item().mask.land_mask_field,
                      f2._data.item().mask.land_mask_field)


if __name__ == "__main__":
    tests.main()
