# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._lazy data.stack`."""

import dask.array as da
import numpy as np

from iris._lazy_data import stack


def test_stack():
    seq = [
        da.arange(2),
        da.ma.masked_array(da.arange(2)),
    ]
    result = stack(seq)

    assert isinstance(result[0].compute(), np.ma.MaskedArray)
    assert isinstance(result[1].compute(), np.ma.MaskedArray)
    np.testing.assert_array_equal(stack(seq).compute(), da.stack(seq).compute())
