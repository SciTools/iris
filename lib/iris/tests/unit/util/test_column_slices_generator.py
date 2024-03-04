# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.column_slices_generator`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

from iris.util import column_slices_generator


class Test_int_types(tests.IrisTest):
    def _test(self, key):
        full_slice = (key,)
        ndims = 1
        mapping, iterable = column_slices_generator(full_slice, ndims)
        self.assertEqual(mapping, {0: None, None: None})
        self.assertEqual(list(iterable), [(0,)])

    def test_int(self):
        self._test(0)

    def test_int_32(self):
        self._test(np.int32(0))

    def test_int_64(self):
        self._test(np.int64(0))


if __name__ == "__main__":
    tests.main()
