# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.pp.PPDataProxy` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.fileformats.pp import PPDataProxy, SplittableInt


class Test_lbpack(tests.IrisTest):
    def test_lbpack_SplittableInt(self):
        lbpack = mock.Mock(spec_set=SplittableInt)
        proxy = PPDataProxy(None, None, None, None, None, lbpack, None, None)
        self.assertEqual(proxy.lbpack, lbpack)
        self.assertIs(proxy.lbpack, lbpack)

    def test_lbpack_raw(self):
        lbpack = 4321
        proxy = PPDataProxy(None, None, None, None, None, lbpack, None, None)
        self.assertEqual(proxy.lbpack, lbpack)
        self.assertIsNot(proxy.lbpack, lbpack)
        self.assertIsInstance(proxy.lbpack, SplittableInt)
        self.assertEqual(proxy.lbpack.n1, lbpack % 10)
        self.assertEqual(proxy.lbpack.n2, lbpack // 10 % 10)
        self.assertEqual(proxy.lbpack.n3, lbpack // 100 % 10)
        self.assertEqual(proxy.lbpack.n4, lbpack // 1000 % 10)


class SliceTranslator:
    """
    Class to translate an array-indexing expression into a tuple of keys.

    An instance just returns the argument of its __getitem__ call.

    """

    def __getitem__(self, keys):
        return keys


# A multidimensional-indexable object that returns its index keys, so we can
# use multidimensional-indexing notation to specify a slicing expression.
Slices = SliceTranslator()


class Test__getitem__slicing(tests.IrisTest):
    def _check_slicing(
        self, test_shape, indices, result_shape, data_was_fetched=True
    ):
        # Check behaviour of the getitem call with specific slicings.
        # Especially: check cases where a fetch does *not* read from the file.
        # This is necessary because, since Dask 2.0, the "from_array" function
        # takes a zero-length slice of its array argument, to capture array
        # metadata, and in those cases we want to avoid file access.
        test_dtype = np.dtype(np.float32)
        proxy = PPDataProxy(
            shape=test_shape,
            src_dtype=test_dtype,
            path=None,
            offset=None,
            data_len=None,
            lbpack=0,  # Note: a 'real' value is needed.
            boundary_packing=None,
            mdi=None,
        )

        # Mock out the file-open call, to see if the file would be read.
        builtin_open_func_name = "builtins.open"
        mock_fileopen = self.patch(builtin_open_func_name)

        # Also mock out the 'databytes_to_shaped_array' call, to fake minimal
        # operation in the cases where file-open *does* get called.
        fake_data = np.zeros(test_shape, dtype=test_dtype)
        self.patch(
            "iris.fileformats.pp._data_bytes_to_shaped_array",
            mock.MagicMock(return_value=fake_data),
        )

        # Test the requested indexing operation.
        result = proxy.__getitem__(indices)

        # Check the behaviour and results were as expected.
        self.assertEqual(mock_fileopen.called, data_was_fetched)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, test_dtype)
        self.assertEqual(result.shape, result_shape)

    def test_slicing_1d_normal(self):
        # A 'normal' 1d testcase with no empty slices.
        self._check_slicing(
            test_shape=(3,),
            indices=Slices[1:10],
            result_shape=(2,),
            data_was_fetched=True,
        )

    def test_slicing_1d_empty(self):
        # A 1d testcase with an empty slicing.
        self._check_slicing(
            test_shape=(3,),
            indices=Slices[0:0],
            result_shape=(0,),
            data_was_fetched=False,
        )

    def test_slicing_2d_normal(self):
        # A 2d testcase with no empty slices.
        self._check_slicing(
            test_shape=(3, 4),
            indices=Slices[2, :3],
            result_shape=(3,),
            data_was_fetched=True,
        )

    def test_slicing_2d_allempty(self):
        # A 2d testcase with all empty slices.
        self._check_slicing(
            test_shape=(3, 4),
            indices=Slices[0:0, 0:0],
            result_shape=(0, 0),
            data_was_fetched=False,
        )

    def test_slicing_2d_empty_dim0(self):
        # A 2d testcase with an empty slice.
        self._check_slicing(
            test_shape=(3, 4),
            indices=Slices[0:0],
            result_shape=(0, 4),
            data_was_fetched=False,
        )

    def test_slicing_2d_empty_dim1(self):
        # A 2d testcase with an empty slice, and an integer index.
        self._check_slicing(
            test_shape=(3, 4),
            indices=Slices[1, 0:0],
            result_shape=(0,),
            data_was_fetched=False,
        )

    def test_slicing_complex(self):
        # Multiple dimensions with multiple empty slices.
        self._check_slicing(
            test_shape=(3, 4, 2, 5, 6, 3, 7),
            indices=Slices[1:3, 2, 0:0, :, 1:1, :100],
            result_shape=(2, 0, 5, 0, 3, 7),
            data_was_fetched=False,
        )


if __name__ == "__main__":
    tests.main()
