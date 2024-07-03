# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit-test infrastructure for the :mod:`iris._concatenate` package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dask.array as da
import numpy as np

from iris._concatenate import _CONSTANT, _DECREASING, _INCREASING
import iris.common
from iris.coords import AuxCoord, DimCoord

__all__ = ["ExpectedItem", "N_POINTS", "SCALE_FACTOR", "create_metadata"]

# number of coordinate points
N_POINTS: int = 10

# coordinate points multiplication scale factor
SCALE_FACTOR: int = 10


METADATA = {
    "standard_name": "air_temperature",
    "long_name": "air temperature",
    "var_name": "atemp",
    "units": "kelvin",
    "attributes": {},
    "coord_system": None,
    "climatological": False,
    "circular": False,
}


@dataclass
class ExpectedItem:
    """Expected test result components of :class:`iris._concatenate._CoordMetaData`."""

    defn: iris.common.DimCoordMetadata | iris.common.CoordMetadata
    dims: tuple[int, ...]
    points_dtype: np.dtype
    bounds_dtype: np.dtype | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaDataItem:
    """Test input and expected output from :class:`iris._concatenate._CoordMetaData`."""

    coord: AuxCoord | DimCoord
    dims: tuple[int, ...]
    expected: ExpectedItem


def create_metadata(
    dim_coord: bool = True,
    scalar: bool = False,
    order: int | None = None,
    circular: bool | None = False,
    coord_dtype=None,
    lazy: bool = True,
    with_bounds: bool | None = False,
) -> MetaDataItem:
    """Construct payload for :class:`iris._concatenate.CoordMetaData` testing."""
    if coord_dtype is None:
        coord_dtype = np.float32

    if order is None:
        order = _INCREASING

    array_lib = da if lazy else np
    bounds = None

    if scalar:
        points = array_lib.ones(1, dtype=coord_dtype)
        order = _CONSTANT

        if with_bounds:
            bounds = array_lib.array([0, 2], dtype=coord_dtype).reshape(1, 2)
    else:
        if order == _CONSTANT:
            points = array_lib.ones(N_POINTS, dtype=coord_dtype)
        else:
            if order == _DECREASING:
                start, stop, step = N_POINTS - 1, -1, -1
            else:
                start, stop, step = 0, N_POINTS, 1
            points = (
                array_lib.arange(start, stop, step, dtype=coord_dtype) * SCALE_FACTOR
            )

        if with_bounds:
            offset = SCALE_FACTOR // 2
            bounds = array_lib.vstack(
                [points.copy() - offset, points.copy() + offset]
            ).T

    bounds_dtype = coord_dtype if with_bounds else None

    values = METADATA.copy()
    values["circular"] = circular
    CoordClass = DimCoord if dim_coord else AuxCoord
    coord = CoordClass(points, bounds=bounds)
    if dim_coord and lazy:
        # creating a DimCoord *always* results in realized points/bounds.
        assert not coord.has_lazy_points()
        if with_bounds:
            assert not coord.has_lazy_bounds()
    metadata = iris.common.DimCoordMetadata(**values)

    if dim_coord:
        coord.metadata = metadata
    else:
        # convert the DimCoordMetadata to a CoordMetadata instance
        # and assign to the AuxCoord
        coord.metadata = iris.common.CoordMetadata.from_metadata(metadata)

    dims = tuple([dim for dim in range(coord.ndim)])
    kwargs: dict[str, Any] = {"scalar": scalar}

    if dim_coord:
        kwargs["circular"] = circular
        kwargs["order"] = order

    expected = ExpectedItem(
        defn=metadata,
        dims=dims,
        points_dtype=coord_dtype,
        bounds_dtype=bounds_dtype,
        kwargs=kwargs,
    )

    return MetaDataItem(coord=coord, dims=dims, expected=expected)
