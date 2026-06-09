# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.column_slices_generator`."""

import numpy as np
import pytest

from iris.util import column_slices_generator


class Test_int_types:
    @pytest.mark.parametrize("key", [0, np.int32(0), np.int64(0)])
    def test(self, key):
        full_slice = (key,)
        ndims = 1
        mapping, iterable = column_slices_generator(full_slice, ndims)
        assert mapping == {0: None, None: None}
        assert list(iterable) == [(0,)]
