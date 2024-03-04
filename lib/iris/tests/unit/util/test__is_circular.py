# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util._is_circular`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

from iris.util import _is_circular


class Test(tests.IrisTest):
    def test_simple(self):
        data = np.arange(12) * 30
        self.assertTrue(_is_circular(data, 360))

    def test_negative_diff(self):
        data = (np.arange(96) * -3.749998) + 3.56249908e02
        self.assertTrue(_is_circular(data, 360))


if __name__ == "__main__":
    tests.main()
