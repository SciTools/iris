# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util._is_circular`."""

import numpy as np

from iris.util import _is_circular


class Test:
    def test_simple(self):
        data = np.arange(12) * 30
        assert _is_circular(data, 360)

    def test_negative_diff(self):
        data = (np.arange(96) * -3.749998) + 3.56249908e02
        assert _is_circular(data, 360)
