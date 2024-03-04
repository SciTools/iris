# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris._lazy data.is_lazy_data`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import dask.array as da
import numpy as np

from iris._lazy_data import is_lazy_data


class Test_is_lazy_data(tests.IrisTest):
    def test_lazy(self):
        values = np.arange(30).reshape((2, 5, 3))
        lazy_array = da.from_array(values, chunks="auto")
        self.assertTrue(is_lazy_data(lazy_array))

    def test_real(self):
        real_array = np.arange(24).reshape((2, 3, 4))
        self.assertFalse(is_lazy_data(real_array))


if __name__ == "__main__":
    tests.main()
