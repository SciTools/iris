# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis.GMEAN` aggregator."""

import numpy as np
import numpy.ma as ma

from iris.analysis import GMEAN


def test_name():
    assert GMEAN.name() == "geometric_mean"


def test_fully_masked_slice_is_masked():
    """A slice with no unmasked points comes out masked, not as NaN.

    scipy.stats.mstats returns a plain array with NaN there (SciPy 1.18).
    """
    data = ma.masked_array(
        [[1.0, 2.0], [4.0, 8.0]], mask=[[False, True], [False, True]]
    )
    result = GMEAN.aggregate(data, axis=0)
    assert ma.isMaskedArray(result)
    np.testing.assert_array_equal(ma.getmaskarray(result), [False, True])


def test_partly_masked_slice_ignores_masked_points():
    data = ma.masked_array(
        [[1.0, 2.0], [4.0, 8.0]], mask=[[False, False], [False, True]]
    )
    result = GMEAN.aggregate(data, axis=0)
    assert not ma.getmaskarray(result).any()
    np.testing.assert_allclose(result[1], 2.0)


def test_unmasked_input_gives_unmasked_output():
    result = GMEAN.aggregate(np.array([[1.0, 2.0], [4.0, 8.0]]), axis=0)
    assert not ma.isMaskedArray(result)
