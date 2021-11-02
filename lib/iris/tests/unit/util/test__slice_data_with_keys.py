# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.util._slice_data_with_keys`.

Note: much of the functionality really belongs to the other routines,
:func:`iris.util._build_full_slice_given_keys`, and
:func:`column_slices_generator`.
However, it is relatively simple to test multiple aspects of all three here
in combination.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

from iris._lazy_data import as_concrete_data, as_lazy_data
from iris.util import _slice_data_with_keys


class DummyArray:
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
        return DummyArray(
            new_shape, _indexing_record_list=self._getitem_call_keys
        )


class Indexer:
    # An object to make __getitem__ arglists from indexing operations.
    def __getitem__(self, keys):
        return keys


# An Indexer object for generating indexing keys in a nice visible way.
Index = Indexer()


class MixinIndexingTest:
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
                    equal &= np.asanyarray(act_key).dtype == np.asanyarray(
                        expect_key
                    ).dtype and np.all(act_key == expect_key)
            errmsg = "Different key lists:\n{!s}\n!=\n{!s}\n"

            def showkeys(keys_list):
                msg = "[\n  "
                msg += "\n  ".join(str(x) for x in keys_list)
                msg += "\n]"
                return msg

            self.assertTrue(
                equal,
                errmsg.format(showkeys(calls_got), showkeys(expect_call_keys)),
            )
        if expect_map is not None:
            self.assertEqual(dim_map, expect_map)


class Test_indexing(MixinIndexingTest, tests.IrisTest):
    # Check the indexing operations performed for various requested keys.

    def test_0d_nokeys(self):
        # Performs *no* underlying indexing operation.
        self.check((), Index[()], [])

    def test_1d_int(self):
        self.check((4,), Index[2], [(2,)])

    def test_1d_all(self):
        self.check((3,), Index[:], [(slice(None),)])

    def test_1d_tuple(self):
        # The call makes tuples into 1-D arrays, and a trailing Ellipsis is
        # added (for the 1-D case only).
        self.check(
            (3,), Index[((2, 0, 1),)], [(np.array([2, 0, 1]), Ellipsis)]
        )

    def test_fail_1d_2keys(self):
        msg = "More slices .* than dimensions"
        with self.assertRaisesRegex(IndexError, msg):
            self.check((3,), Index[1, 2])

    def test_fail_empty_slice(self):
        msg = "Cannot index with zero length slice"
        with self.assertRaisesRegex(IndexError, msg):
            self.check((3,), Index[1:1])

    def test_2d_tuple(self):
        # Like the above, but there is an extra no-op at the start and no
        # trailing Ellipsis is generated.
        self.check(
            (3, 2),
            Index[((2, 0, 1),)],
            [(slice(None), slice(None)), (np.array([2, 0, 1]), slice(None))],
        )

    def test_2d_two_tuples(self):
        # Could be treated as fancy indexing, but must not be !
        # Two separate 2-D indexing operations.
        self.check(
            (3, 2),
            Index[(2, 0, 1, 1), (0, 1, 0, 1)],
            [
                (np.array([2, 0, 1, 1]), slice(None)),
                (slice(None), np.array([0, 1, 0, 1])),
            ],
        )

    def test_2d_tuple_and_value(self):
        # The two keys are applied in separate operations, and in the reverse
        # order (?) :  The second op is then slicing a 1-D array, not 2-D.
        self.check(
            (3, 5),
            Index[(2, 0, 1), 3],
            [(slice(None), 3), (np.array([2, 0, 1]), Ellipsis)],
        )

    def test_2d_single_int(self):
        self.check((3, 4), Index[2], [(2, slice(None))])

    def test_2d_multiple_int(self):
        self.check((3, 4), Index[2, 1:3], [(2, slice(1, 3))])

    def test_3d_1int(self):
        self.check((3, 4, 5), Index[2], [(2, slice(None), slice(None))])

    def test_3d_2int(self):
        self.check((3, 4, 5), Index[2, 3], [(2, 3, slice(None))])

    def test_3d_tuple_and_value(self):
        # The two keys are applied in separate operations, and in the reverse
        # order (?) : The second op is slicing a 2-D array, not 3-D.
        self.check(
            (3, 5, 7),
            Index[(2, 0, 1), 4],
            [
                (slice(None), 4, slice(None)),
                (np.array([2, 0, 1]), slice(None)),
            ],
        )

    def test_3d_ellipsis_last(self):
        self.check((3, 4, 5), Index[2, ...], [(2, slice(None), slice(None))])

    def test_3d_ellipsis_first_1int(self):
        self.check((3, 4, 5), Index[..., 2], [(slice(None), slice(None), 2)])

    def test_3d_ellipsis_first_2int(self):
        self.check((3, 4, 5), Index[..., 2, 3], [(slice(None), 2, 3)])

    def test_3d_multiple_tuples(self):
        # Where there are TWO or more tuple keys, this could be misinterpreted
        # as 'fancy' indexing :  It should resolve into multiple calls.
        self.check(
            (3, 4, 5),
            Index[(1, 2, 1), :, (2, 2, 3)],
            [
                (slice(None), slice(None), slice(None)),
                (np.array([1, 2, 1]), slice(None), slice(None)),
                (slice(None), slice(None), np.array([2, 2, 3])),
            ],
        )
        # NOTE: there seem to be an extra initial [:, :, :].
        # That's just what it does at present.


class Test_dimensions_mapping(MixinIndexingTest, tests.IrisTest):
    # Check the dimensions map returned for various requested keys.

    def test_1d_nochange(self):
        self.check((3,), Index[1:2], expect_map={None: None, 0: 0})

    def test_1d_1int_losedim0(self):
        self.check((3,), Index[1], expect_map={None: None, 0: None})

    def test_1d_tuple_nochange(self):
        # A selection index leaves the dimension intact.
        self.check((3,), Index[((1, 0, 1, 2),)], expect_map={None: None, 0: 0})

    def test_1d_1tuple_nochange(self):
        # A selection index with only one value in it *still* leaves the
        # dimension intact.
        self.check((3,), Index[((2,),)], expect_map={None: None, 0: 0})

    def test_1d_slice_nochange(self):
        # A slice leaves the dimension intact.
        self.check((3,), Index[1:7], expect_map={None: None, 0: 0})

    def test_2d_nochange(self):
        self.check((3, 4), Index[:, :], expect_map={None: None, 0: 0, 1: 1})

    def test_2d_losedim0(self):
        self.check((3, 4), Index[1, :], expect_map={None: None, 0: None, 1: 0})

    def test_2d_losedim1(self):
        self.check(
            (3, 4), Index[1:4, 2], expect_map={None: None, 0: 0, 1: None}
        )

    def test_2d_loseboth(self):
        # Two indices give scalar result.
        self.check(
            (3, 4), Index[1, 2], expect_map={None: None, 0: None, 1: None}
        )

    def test_3d_losedim1(self):
        # Cutting out the middle dim.
        self.check(
            (3, 4, 2),
            Index[:, 2],
            expect_map={None: None, 0: 0, 1: None, 2: 1},
        )


class TestResults(tests.IrisTest):
    # Integration-style test, exercising (mostly) the same cases as above,
    # but checking actual results, for both real and lazy array inputs.

    def check(self, real_data, keys, expect_result, expect_map):
        real_data = np.array(real_data)
        lazy_data = as_lazy_data(real_data, real_data.shape)
        real_dim_map, real_result = _slice_data_with_keys(real_data, keys)
        lazy_dim_map, lazy_result = _slice_data_with_keys(lazy_data, keys)
        lazy_result = as_concrete_data(lazy_result)
        self.assertArrayEqual(real_result, expect_result)
        self.assertArrayEqual(lazy_result, expect_result)
        self.assertEqual(real_dim_map, expect_map)
        self.assertEqual(lazy_dim_map, expect_map)

    def test_1d_int(self):
        self.check([1, 2, 3, 4], Index[2], [3], {None: None, 0: None})

    def test_1d_all(self):
        self.check([1, 2, 3], Index[:], [1, 2, 3], {None: None, 0: 0})

    def test_1d_tuple(self):
        self.check(
            [1, 2, 3], Index[((2, 0, 1, 0),)], [3, 1, 2, 1], {None: None, 0: 0}
        )

    def test_fail_1d_2keys(self):
        msg = "More slices .* than dimensions"
        with self.assertRaisesRegex(IndexError, msg):
            self.check([1, 2, 3], Index[1, 2], None, None)

    def test_fail_empty_slice(self):
        msg = "Cannot index with zero length slice"
        with self.assertRaisesRegex(IndexError, msg):
            self.check([1, 2, 3], Index[1:1], None, None)

    def test_2d_tuple(self):
        self.check(
            [[11, 12], [21, 22], [31, 32]],
            Index[((2, 0, 1),)],
            [[31, 32], [11, 12], [21, 22]],
            {None: None, 0: 0, 1: 1},
        )

    def test_2d_two_tuples(self):
        # Could be treated as fancy indexing, but must not be !
        # Two separate 2-D indexing operations.
        self.check(
            [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            Index[(2, 0), (0, 1, 0, 1)],
            [[31, 32, 31, 32], [11, 12, 11, 12]],
            {None: None, 0: 0, 1: 1},
        )

    def test_2d_tuple_and_value(self):
        # The two keys are applied in separate operations, and in the reverse
        # order (?) :  The second op is then slicing a 1-D array, not 2-D.
        self.check(
            [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]],
            Index[(2, 0, 1), 3],
            [34, 14, 24],
            {None: None, 0: 0, 1: None},
        )

    def test_2d_single_int(self):
        self.check(
            [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            Index[1],
            [21, 22, 23],
            {None: None, 0: None, 1: 0},
        )

    def test_2d_int_slice(self):
        self.check(
            [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            Index[2, 1:3],
            [32, 33],
            {None: None, 0: None, 1: 0},
        )

    def test_3d_1int(self):
        self.check(
            [
                [[111, 112, 113], [121, 122, 123]],
                [[211, 212, 213], [221, 222, 223]],
                [[311, 312, 313], [321, 322, 323]],
            ],
            Index[1],
            [[211, 212, 213], [221, 222, 223]],
            {None: None, 0: None, 1: 0, 2: 1},
        )

    def test_3d_2int(self):
        self.check(
            [
                [[111, 112, 113], [121, 122, 123], [131, 132, 133]],
                [[211, 212, 213], [221, 222, 223], [231, 232, 233]],
            ],
            Index[1, 2],
            [231, 232, 233],
            {None: None, 0: None, 1: None, 2: 0},
        )

    def test_3d_tuple_and_value(self):
        # The two keys are applied in separate operations, and in the reverse
        # order (?) : The second op is slicing a 2-D array, not 3-D.
        self.check(
            [
                [[111, 112, 113, 114], [121, 122, 123, 124]],
                [[211, 212, 213, 214], [221, 222, 223, 224]],
                [[311, 312, 313, 314], [321, 322, 323, 324]],
            ],
            Index[(2, 0, 1), 1],
            [[321, 322, 323, 324], [121, 122, 123, 124], [221, 222, 223, 224]],
            {None: None, 0: 0, 1: None, 2: 1},
        )

    def test_3d_ellipsis_last(self):
        self.check(
            [
                [[111, 112, 113], [121, 122, 123]],
                [[211, 212, 213], [221, 222, 223]],
                [[311, 312, 313], [321, 322, 323]],
            ],
            Index[2, ...],
            [[311, 312, 313], [321, 322, 323]],
            {None: None, 0: None, 1: 0, 2: 1},
        )

    def test_3d_ellipsis_first_1int(self):
        self.check(
            [
                [[111, 112, 113, 114], [121, 122, 123, 124]],
                [[211, 212, 213, 214], [221, 222, 223, 224]],
                [[311, 312, 313, 314], [321, 322, 323, 324]],
            ],
            Index[..., 2],
            [[113, 123], [213, 223], [313, 323]],
            {None: None, 0: 0, 1: 1, 2: None},
        )

    def test_3d_ellipsis_mid_1int(self):
        self.check(
            [
                [[111, 112, 113], [121, 122, 123]],
                [[211, 212, 213], [221, 222, 223]],
                [[311, 312, 313], [321, 322, 323]],
            ],
            Index[..., 1, ...],
            [[121, 122, 123], [221, 222, 223], [321, 322, 323]],
            {None: None, 0: 0, 1: None, 2: 1},
        )

    def test_3d_ellipsis_first_2int(self):
        self.check(
            [
                [[111, 112, 113], [121, 122, 123]],
                [[211, 212, 213], [221, 222, 223]],
                [[311, 312, 313], [321, 322, 323]],
            ],
            Index[..., 1, 2],
            [123, 223, 323],
            {None: None, 0: 0, 1: None, 2: None},
        )

    def test_3d_multiple_tuples(self):
        # Where there are TWO or more tuple keys, this could be misinterpreted
        # as 'fancy' indexing :  It should resolve into multiple calls.
        self.check(
            [
                [[111, 112, 113, 114], [121, 122, 123, 124]],
                [[211, 212, 213, 214], [221, 222, 223, 224]],
                [[311, 312, 313, 314], [321, 322, 323, 324]],
            ],
            Index[(1, 2, 1), :, (2, 2, 3)],
            [
                [[213, 213, 214], [223, 223, 224]],
                [[313, 313, 314], [323, 323, 324]],
                [[213, 213, 214], [223, 223, 224]],
            ],
            {None: None, 0: 0, 1: 1, 2: 2},
        )
        # NOTE: there seem to be an extra initial [:, :, :].
        # That's just what it does at present.


if __name__ == "__main__":
    tests.main()
