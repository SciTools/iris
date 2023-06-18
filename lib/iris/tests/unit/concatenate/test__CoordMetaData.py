# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit-tests for :class:`iris._concatenate._CoordMetaData`."""

import numpy as np
import pytest

from iris._concatenate import (
    _CONSTANT,
    _DECREASING,
    _INCREASING,
    _CoordMetaData,
)

from . import ExpectedItem, create_metadata


def check(actual: _CoordMetaData, expected: ExpectedItem) -> None:
    """Assert actual and expected results."""
    assert actual.defn == expected.defn
    assert actual.dims == expected.dims
    assert actual.points_dtype == expected.points_dtype
    assert actual.bounds_dtype == expected.bounds_dtype
    assert actual.kwargs == expected.kwargs


@pytest.mark.parametrize("order", [_DECREASING, _INCREASING])
@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_dim(
    order: int, circular: bool, dtype: np.dtype, lazy: bool, with_bounds: bool
) -> None:
    """Test :class:`iris._concatenate._CoordMetaData` with dim coord."""
    metadata = create_metadata(
        dim=True,
        scalar=False,
        order=order,
        circular=circular,
        dtype=dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    actual = _CoordMetaData(coord=metadata.coord, dims=metadata.dims)
    check(actual, metadata.expected)


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_dim__scalar(
    circular: bool, dtype: np.dtype, lazy: bool, with_bounds: bool
) -> None:
    """Test :class:`iris._concatenate._CoordMetaData` with scalar dim coord."""
    metadata = create_metadata(
        dim=True,
        scalar=True,
        order=_CONSTANT,
        circular=circular,
        dtype=dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    actual = _CoordMetaData(coord=metadata.coord, dims=metadata.dims)
    check(actual, metadata.expected)


@pytest.mark.parametrize("order", [_DECREASING, _INCREASING])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_aux(
    order: int, dtype: np.dtype, lazy: bool, with_bounds: bool
) -> None:
    """Test :class:`iris._concatenate._CoordMetaData` with aux coord."""
    metadata = create_metadata(
        dim=False,
        scalar=False,
        order=order,
        circular=None,
        dtype=dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    actual = _CoordMetaData(coord=metadata.coord, dims=metadata.dims)
    check(actual, metadata.expected)


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_aux__scalar(dtype: np.dtype, lazy: bool, with_bounds: bool) -> None:
    """Test :class:`iris._concatenate._CoordMetaData` with scalar aux coord."""
    metadata = create_metadata(
        dim=False,
        scalar=True,
        order=_CONSTANT,
        circular=None,
        dtype=dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    actual = _CoordMetaData(coord=metadata.coord, dims=metadata.dims)
    check(actual, metadata.expected)
