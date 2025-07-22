# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test class :class:`iris._concatenate._CubeSignature`."""

# import iris tests first so that some things can be initialised
# before importing anything else.
from dataclasses import dataclass

from cf_units import Unit
import numpy as np
import pytest

from iris._concatenate import _CubeSignature as CubeSignature
from iris.coords import DimCoord
from iris.cube import Cube
from iris.util import new_axis


@dataclass
class SampleData:
    series_inc: CubeSignature | None = None
    series_inc_cube: Cube | None = None
    series_dec: CubeSignature | None = None
    series_dec_cube: Cube | None = None
    scalar_cube: Cube | None = None


class Test__coordinate_dim_metadata_equality:
    @pytest.fixture()
    def sample_data(self) -> SampleData:
        # Return a standard set of test items, wrapped in a data object

        data = SampleData()

        nt = 10
        cube_data = np.arange(nt, dtype=np.float32)
        cube = Cube(cube_data, standard_name="air_temperature", units="K")
        # Temporal coordinate.
        t_units = Unit("hours since 1970-01-01 00:00:00", calendar="standard")
        t_coord = DimCoord(points=np.arange(nt), standard_name="time", units=t_units)
        cube.add_dim_coord(t_coord, 0)
        # Increasing 1D time-series cube.
        data.series_inc_cube = cube
        data.series_inc = CubeSignature(data.series_inc_cube)

        # Decreasing 1D time-series cube.
        data.series_dec_cube = data.series_inc_cube.copy()
        data.series_dec_cube.remove_coord("time")
        t_tmp = DimCoord(
            points=t_coord.points[::-1], standard_name="time", units=t_units
        )
        data.series_dec_cube.add_dim_coord(t_tmp, 0)
        data.series_dec = CubeSignature(data.series_dec_cube)

        # Scalar 0D time-series cube with scalar time coordinate.
        cube = Cube(0, standard_name="air_temperature", units="K")
        cube.add_aux_coord(DimCoord(points=nt, standard_name="time", units=t_units))
        data.scalar_cube = cube
        return data

    def test_scalar_non_common_axis(self, sample_data):
        scalar = CubeSignature(sample_data.scalar_cube)
        assert sample_data.series_inc.dim_metadata != scalar.dim_metadata
        assert sample_data.series_dec.dim_metadata != scalar.dim_metadata

    def test_1d_single_value_common_axis(self, sample_data):
        # Manually promote scalar time cube to be a 1d cube.
        single = CubeSignature(new_axis(sample_data.scalar_cube, "time"))
        assert sample_data.series_inc.dim_metadata == single.dim_metadata
        assert sample_data.series_dec.dim_metadata == single.dim_metadata

    def test_increasing_common_axis(self, sample_data):
        series_inc = sample_data.series_inc
        series_dec = sample_data.series_dec
        assert series_inc.dim_metadata == series_inc.dim_metadata
        assert series_inc.dim_metadata != series_dec.dim_metadata

    def test_decreasing_common_axis(self, sample_data):
        series_inc = sample_data.series_inc
        series_dec = sample_data.series_dec
        assert series_dec.dim_metadata != series_inc.dim_metadata
        assert series_dec.dim_metadata == series_dec.dim_metadata

    def test_circular(self, sample_data):
        series_inc = sample_data.series_inc
        circular_cube = sample_data.series_inc_cube.copy()
        circular_cube.coord("time").circular = True
        circular = CubeSignature(circular_cube)
        assert circular.dim_metadata != series_inc.dim_metadata
        assert circular.dim_metadata == circular.dim_metadata
