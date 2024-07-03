# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit-tests for :class:`iris._concatenate._CoordSignature`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from iris._concatenate import (
    _CONSTANT,
    _DECREASING,
    _INCREASING,
    _CoordExtent,
    _CoordMetaData,
    _CoordSignature,
    _Extent,
)
from iris.coords import DimCoord

from . import N_POINTS, SCALE_FACTOR, create_metadata


@dataclass
class MockCubeSignature:
    """Simple mock of :class:`iris._concatenate._CubeSignature`."""

    aux_coords_and_dims: bool | None = None
    cell_measures_and_dims: bool | None = None
    ancillary_variables_and_dims: bool | None = None
    derived_coords_and_dims: bool | None = None
    dim_coords: list[DimCoord] = field(default_factory=list)
    dim_mapping: bool | None = None
    dim_extents: list[_Extent] = field(default_factory=list)
    dim_order: list[int] = field(default_factory=list)
    dim_metadata: list[_CoordMetaData] = field(default_factory=list)


@pytest.mark.parametrize("order", [_DECREASING, _INCREASING])
@pytest.mark.parametrize("coord_dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_dim(order: int, coord_dtype, lazy: bool, with_bounds: bool) -> None:
    """Test extent calculation of vector dimension coordinates."""
    metadata = create_metadata(
        dim_coord=True,
        scalar=False,
        order=order,
        coord_dtype=coord_dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    assert isinstance(metadata.coord, DimCoord)  # Type hint for linters.
    dim_metadata = [_CoordMetaData(metadata.coord, metadata.dims)]
    cube_signature = MockCubeSignature(
        dim_coords=[metadata.coord], dim_metadata=dim_metadata
    )
    coord_signature = _CoordSignature(cube_signature)
    assert len(coord_signature.dim_extents) == 1
    (actual,) = coord_signature.dim_extents
    first, last = coord_dtype(0), coord_dtype((N_POINTS - 1) * SCALE_FACTOR)
    if order == _CONSTANT:
        emsg = f"Expected 'order' of '{_DECREASING}' or '{_INCREASING}', got '{order}'."
        raise ValueError(emsg)
    points_extent = _Extent(min=first, max=last)
    bounds_extent = None
    if with_bounds:
        offset = SCALE_FACTOR // 2
        if order == _INCREASING:
            bounds_extent = (
                _Extent(min=first - offset, max=last - offset),
                _Extent(min=first + offset, max=last + offset),
            )
        else:
            bounds_extent = (
                _Extent(min=first + offset, max=last + offset),
                _Extent(min=first - offset, max=last - offset),
            )
    expected = _CoordExtent(points=points_extent, bounds=bounds_extent)
    assert actual == expected


@pytest.mark.parametrize("coord_dtype", [np.int32, np.float32])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("with_bounds", [False, True])
def test_dim__scalar(coord_dtype, lazy: bool, with_bounds: bool) -> None:
    """Test extent calculation of scalar dimension coordinates."""
    metadata = create_metadata(
        dim_coord=True,
        scalar=True,
        order=_CONSTANT,
        coord_dtype=coord_dtype,
        lazy=lazy,
        with_bounds=with_bounds,
    )
    assert isinstance(metadata.coord, DimCoord)  # Hint for mypy.
    dim_metadata = [_CoordMetaData(metadata.coord, metadata.dims)]
    cube_signature = MockCubeSignature(
        dim_coords=[metadata.coord], dim_metadata=dim_metadata
    )
    coord_signature = _CoordSignature(cube_signature)
    assert len(coord_signature.dim_extents) == 1
    (actual,) = coord_signature.dim_extents
    point = coord_dtype(1)
    points_extent = _Extent(min=point, max=point)
    bounds_extent = None
    if with_bounds:
        first, last = coord_dtype(0), coord_dtype(2)
        bounds_extent = (
            _Extent(min=first, max=first),
            _Extent(min=last, max=last),
        )
    expected = _CoordExtent(points=points_extent, bounds=bounds_extent)
    assert actual == expected
