# (C) British Crown Copyright 2017, Met Office
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
Test function :func:`iris.util._slice_data_with_keys`.

Note: much of the functionality really belongs to the other routines,
:func:`iris.util._build_full_slice_given_keys`, and
:func:`column_slices_generator`.
However, it is relatively simple to test multiple aspects of all three here
in combination.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.util import _slice_data_with_keys


class DummyArray(object):
    # A dummy array-like that records the keys of indexing calls.
    def __init__(self, shape, _indexing_record_list=None):
        self.shape = shape
        self.ndim = len(shape)
        if _indexing_record_list is None:
            _indexing_record_list = []
        self._getitem_call_keys = _indexing_record_list

    def __getitem__(self, keys):
        # Add the indexing keys to the call list.
        self._getitem_call_keys.append(keys)
        # Return a new object with the correct derived shape, and record its
        # indexing operations in the same key list as this.
        shape_array = np.zeros(self.shape)
        shape_array = shape_array.__getitem__(keys)
        new_shape = shape_array.shape
        return DummyArray(new_shape,
                          _indexing_record_list=self._getitem_call_keys)


class Indexer(object):
    # An object to make __getitem__ arglists from indexing operations.
    def __getitem__(self, keys):
        return keys


# An Indexer object for generating indexing keys in a nice visible way.
Index = Indexer()


class MixinIndexingTest(object):
    def check(self, shape, keys, expect_call_keys=None, expect_map=None):
        data = DummyArray(shape)
        dim_map, _ = _slice_data_with_keys(data, keys)
        if expect_call_keys is not None:
            calls_got = data._getitem_call_keys
            # Check that the indexing keys applied were the expected ones.
            equal = len(calls_got) == len(expect_call_keys)
            for act_call, expect_call in zip(calls_got, expect_call_keys):
                equal &= len(act_call) == len(expect_call)
                # A problem here is that in each call, some keys may be
                # *arrays*, and arrays can't be compared in the "normal"
                # way.  So we must use np.all for comparison :-(
                for act_key, expect_key in zip(act_call, expect_call):
                    equal &= (np.asanyarray(act_key).dtype ==
                              np.asanyarray(expect_key).dtype and
                              np.all(act_key == expect_key))
            errmsg = 'Different key lists:\n{!s}\n!=\n{!s}\n'

            def showkeys(keys_list):
                msg = '[\n  '
                msg += '\n  '.join(str(x) for x in keys_list)
                msg += '\n]'
                return msg

            self.assertTrue(equal, errmsg.format(showkeys(calls_got),
                                                 showkeys(expect_call_keys)))
        if expect_map is not None:
            self.assertEqual(dim_map, expect_map)


class Test_indexing(MixinIndexingTest, tests.IrisTest):
    # Check the indexing operations performed for various requested keys.

    def test_0d_nokeys(self):
        # Performs *no* underlying indexing operation.
        self.check((), Index[()],
                   [])

    def test_1d_int(self):
        self.check((4,), Index[2],
                   [(2,)])

    def test_1d_float(self):
        # Number types are not cast.
        self.check((4,), Index[3.1],
                   [(3.1,)])

    def test_1d_all(self):
        self.check((3,), Index[:],
                   [(slice(None),)])

    def test_1d_tuple(self):
        # The call makes tuples into 1-D arrays, and a trailing Ellipsis is
        # added (for the 1-D case only).
        self.check((3,), Index[(2, 0, 1), ],
                   [(np.array([2, 0, 1]), Ellipsis)])

    def test_fail_1d_2keys(self):
        msg = 'More slices .* than dimensions'
        with self.assertRaisesRegexp(IndexError, msg):
            self.check((3,), Index[1, 2])

    def test_fail_empty_slice(self):
        msg = 'Cannot index with zero length slice'
        with self.assertRaisesRegexp(IndexError, msg):
            self.check((3,), Index[1:1])

    def test_2d_tuple(self):
        # Like the above, but there is an extra no-op at the start and no
        # trailing Ellipsis is generated.
        self.check((3, 2), Index[(2, 0, 1), ],
                   [(slice(None), slice(None)),
                    (np.array([2, 0, 1]), slice(None))])

    def test_2d_two_tuples(self):
        # Could be treated as fancy indexing, but must not be !
        # Two separate 2-D indexing operations.
        self.check((3, 2), Index[(2, 0, 1, 1), (0, 1, 0, 1)],
                   [(np.array([2, 0, 1, 1]), slice(None)),
                    (slice(None), np.array([0, 1, 0, 1]))])

    def test_2d_tuple_and_value(self):
        # The two keys are applied in separate operations, and in the reverse
        # order (?) :  The second op is then slicing a 1-D array, not 2-D.
        self.check((3, 5), Index[(2, 0, 1), 3],
                   [(slice(None), 3),
                    (np.array([2, 0, 1]), Ellipsis)])

    def test_2d_single_int(self):
        self.check((3, 4), Index[2],
                   [(2, slice(None))])

    def test_2d_multiple_int(self):
        self.check((3, 4), Index[2, 1:3],
                   [(2, slice(1, 3))])

    def test_3d_1int(self):
        self.check((3, 4, 5), Index[2],
                   [(2, slice(None), slice(None))])

    def test_3d_2int(self):
        self.check((3, 4, 5), Index[2, 3],
                   [(2, 3, slice(None))])

    def test_3d_tuple_and_value(self):
        # The two keys are applied in separate operations, and in the reverse
        # order (?) : The second op is slicing a 2-D array, not 3-D.
        self.check((3, 5, 7), Index[(2, 0, 1), 4],
                   [(slice(None), 4, slice(None)),
                    (np.array([2, 0, 1]), slice(None))])

    def test_3d_ellipsis_last(self):
        self.check((3, 4, 5), Index[2, ...],
                   [(2, slice(None), slice(None))])

    def test_3d_ellipsis_first_1int(self):
        self.check((3, 4, 5), Index[..., 2],
                   [(slice(None), slice(None), 2)])

    def test_3d_ellipsis_first_2int(self):
        self.check((3, 4, 5), Index[..., 2, 3],
                   [(slice(None), 2, 3)])

    def test_3d_multiple_tuples(self):
        # Where there are TWO or more tuple keys, this could be misinterpreted
        # as 'fancy' indexing :  It should resolve into multiple calls.
        self.check((3, 4, 5), Index[(1, 2, 1), :, (2, 2, 3)],
                   [(slice(None), slice(None), slice(None)),
                    (np.array([1, 2, 1]), slice(None), slice(None)),
                    (slice(None), slice(None), np.array([2, 2, 3])),
                    ])
        # NOTE: there seem to be an extra initial [:, :, :].
        # That's just what it does at present.


class Test_dimensions_mapping(MixinIndexingTest, tests.IrisTest):
    # Check the dimensions map returned for various requested keys.

    def test_1d_nochange(self):
        self.check((3,), Index[1:2],
                   expect_map={None: None, 0: 0})

    def test_1d_1int_losedim0(self):
        self.check((3,), Index[1],
                   expect_map={None: None, 0: None})

    def test_1d_tuple_nochange(self):
        # A selection index leaves the dimension intact.
        self.check((3,), Index[(1, 0, 1, 2), ],
                   expect_map={None: None, 0: 0})

    def test_1d_1tuple_nochange(self):
        # A selection index with only one value in it *still* leaves the
        # dimension intact.
        self.check((3,), Index[(2,), ],
                   expect_map={None: None, 0: 0})

    def test_1d_slice_nochange(self):
        # A slice leaves the dimension intact.
        self.check((3,), Index[1:7],
                   expect_map={None: None, 0: 0})

    def test_2d_nochange(self):
        self.check((3, 4), Index[:, :],
                   expect_map={None: None, 0: 0, 1: 1})

    def test_2d_losedim0(self):
        self.check((3, 4), Index[1, :],
                   expect_map={None: None, 0: None, 1: 0})

    def test_2d_losedim1(self):
        self.check((3, 4), Index[1:4, 2],
                   expect_map={None: None, 0: 0, 1: None})

    def test_2d_loseboth(self):
        # Two indices give scalar result.
        self.check((3, 4), Index[1, 2],
                   expect_map={None: None, 0: None, 1: None})

    def test_3d_losedim1(self):
        # Cutting out the middle dim.
        self.check((3, 4, 2), Index[:, 2],
                   expect_map={None: None, 0: 0, 1: None, 2: 1})


if __name__ == '__main__':
    tests.main()
