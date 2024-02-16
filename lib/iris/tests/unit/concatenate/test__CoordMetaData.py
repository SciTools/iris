# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit-tests for :class:`iris._concatenate._CoordMetaData`."""

from __future__ import annotations

import numpy as np
import pytest

from iris._concatenate import _CONSTANT, _DECREASING, _INCREASING, _CoordMetaData

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
@pytest.mark.parametrize("coord_dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_dim(
    order: int,
    circular: bool,
    coord_dtype: np.dtype,
    lazy: bool,
    with_bounds: bool,
) -> None:
    """Test :class:`iris._concatenate._CoordMetaData` with dim coord."""
    metadata = create_metadata(
        dim_coord=True,
        scalar=False,
        order=order,
        circular=circular,
        coord_dtype=coord_dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    actual = _CoordMetaData(coord=metadata.coord, dims=metadata.dims)
    check(actual, metadata.expected)


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("coord_dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_dim__scalar(
    circular: bool, coord_dtype: np.dtype, lazy: bool, with_bounds: bool
) -> None:
    """Test :class:`iris._concatenate._CoordMetaData` with scalar dim coord."""
    metadata = create_metadata(
        dim_coord=True,
        scalar=True,
        order=_CONSTANT,
        circular=circular,
        coord_dtype=coord_dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    actual = _CoordMetaData(coord=metadata.coord, dims=metadata.dims)
    check(actual, metadata.expected)


@pytest.mark.parametrize("order", [_DECREASING, _INCREASING])
@pytest.mark.parametrize("coord_dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_aux(order: int, coord_dtype: np.dtype, lazy: bool, with_bounds: bool) -> None:
    """Test :class:`iris._concatenate._CoordMetaData` with aux coord."""
    metadata = create_metadata(
        dim_coord=False,
        scalar=False,
        order=order,
        circular=None,
        coord_dtype=coord_dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    actual = _CoordMetaData(coord=metadata.coord, dims=metadata.dims)
    check(actual, metadata.expected)


@pytest.mark.parametrize("coord_dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_aux__scalar(coord_dtype: np.dtype, lazy: bool, with_bounds: bool) -> None:
    """Test :class:`iris._concatenate._CoordMetaData` with scalar aux coord."""
    metadata = create_metadata(
        dim_coord=False,
        scalar=True,
        order=_CONSTANT,
        circular=None,
        coord_dtype=coord_dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    actual = _CoordMetaData(coord=metadata.coord, dims=metadata.dims)
    check(actual, metadata.expected)
