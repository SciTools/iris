# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.analysis.cartography.project`."""

import cartopy.crs as ccrs
import numpy as np
import pytest

from iris.analysis.cartography import project
import iris.coord_systems
import iris.coords
import iris.cube
import iris.tests
from iris.tests import _shared_utils
import iris.tests.stock
from iris.warnings import IrisDefaultingWarning

ROBINSON = ccrs.Robinson()


@pytest.fixture
def low_res_4d():
    cube = iris.tests.stock.realistic_4d_no_derived()
    cube = cube[0:2, 0:3, ::10, ::10]
    cube.remove_coord("surface_altitude")
    return cube


class TestAll:
    @pytest.fixture(autouse=True)
    def _setup(self):
        cs = iris.coord_systems.GeogCS(6371229)
        self.cube = iris.cube.Cube(np.zeros(25).reshape(5, 5))
        self.cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(5),
                standard_name="latitude",
                units="degrees",
                coord_system=cs,
            ),
            0,
        )
        self.cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(5),
                standard_name="longitude",
                units="degrees",
                coord_system=cs,
            ),
            1,
        )

        self.tcs = iris.coord_systems.GeogCS(6371229)

    def test_is_iris_coord_system(self):
        res, _ = project(self.cube, self.tcs)
        assert res.coord("projection_y_coordinate").coord_system == self.tcs
        assert res.coord("projection_x_coordinate").coord_system == self.tcs

        assert res.coord("projection_y_coordinate").coord_system is not self.tcs
        assert res.coord("projection_x_coordinate").coord_system is not self.tcs

    @_shared_utils.skip_data
    def test_bad_resolution_negative(self, low_res_4d):
        with pytest.raises(ValueError, match="must be non-negative"):
            project(low_res_4d, ROBINSON, nx=-200, ny=200)

    @_shared_utils.skip_data
    def test_bad_resolution_non_numeric(self, low_res_4d):
        with pytest.raises(TypeError):
            project(low_res_4d, ROBINSON, nx=200, ny="abc")

    @_shared_utils.skip_data
    def test_missing_lat(self, low_res_4d):
        low_res_4d.remove_coord("grid_latitude")
        with pytest.raises(
            ValueError, match="Cannot get latitude/longitude coordinates"
        ):
            project(low_res_4d, ROBINSON)

    @_shared_utils.skip_data
    def test_missing_lon(self, low_res_4d):
        low_res_4d.remove_coord("grid_longitude")
        with pytest.raises(
            ValueError, match="Cannot get latitude/longitude coordinates"
        ):
            project(low_res_4d, ROBINSON)

    @_shared_utils.skip_data
    def test_missing_latlon(self, low_res_4d):
        low_res_4d.remove_coord("grid_longitude")
        low_res_4d.remove_coord("grid_latitude")
        with pytest.raises(
            ValueError, match="Cannot get latitude/longitude coordinates"
        ):
            project(low_res_4d, ROBINSON)

    @_shared_utils.skip_data
    def test_default_resolution(self, low_res_4d):
        new_cube, extent = project(low_res_4d, ROBINSON)
        assert new_cube.shape == low_res_4d.shape

    @_shared_utils.skip_data
    def test_explicit_resolution(self, low_res_4d):
        nx, ny = 5, 4
        new_cube, extent = project(low_res_4d, ROBINSON, nx=nx, ny=ny)
        assert new_cube.shape == low_res_4d.shape[:2] + (ny, nx)

    @_shared_utils.skip_data
    def test_explicit_resolution_single_point(self, low_res_4d):
        nx, ny = 1, 1
        new_cube, extent = project(low_res_4d, ROBINSON, nx=nx, ny=ny)
        assert new_cube.shape == low_res_4d.shape[:2] + (ny, nx)

    @_shared_utils.skip_data
    def test_mismatched_coord_systems(self, low_res_4d):
        low_res_4d.coord("grid_longitude").coord_system = None
        with pytest.raises(ValueError, match="different coordinates systems"):
            project(low_res_4d, ROBINSON)

    @_shared_utils.skip_data
    def test_extent(self, low_res_4d):
        _, extent = project(low_res_4d, ROBINSON)
        assert extent == [
            -17005833.33052523,
            17005833.33052523,
            -8625154.6651,
            8625154.6651,
        ]

    @_shared_utils.skip_data
    def test_cube(self, request, low_res_4d):
        new_cube, _ = project(low_res_4d, ROBINSON)
        _shared_utils.assert_CML(request, new_cube, approx_data=True)

    @_shared_utils.skip_data
    def test_no_coord_system(self, low_res_4d):
        low_res_4d.coord("grid_longitude").coord_system = None
        low_res_4d.coord("grid_latitude").coord_system = None
        message = (
            "Coordinate system of latitude and longitude coordinates is not "
            "specified. Assuming WGS84 Geodetic."
        )
        with pytest.warns(IrisDefaultingWarning, match=message):
            _, _ = project(low_res_4d, ROBINSON)
