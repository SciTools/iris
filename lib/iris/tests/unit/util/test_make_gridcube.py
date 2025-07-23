# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.util.make_gridcube` function."""

import numpy as np
import pytest

from iris.coord_systems import GeogCS, LambertConformal, RotatedGeogCS
from iris.cube import Cube
from iris.fileformats.pp import EARTH_RADIUS
from iris.util import make_gridcube

_GLOBE = GeogCS(EARTH_RADIUS)
_CS_ROTATED = RotatedGeogCS(
    grid_north_pole_latitude=70.0, grid_north_pole_longitude=125.0, ellipsoid=_GLOBE
)
_CS_LAMBERT = LambertConformal(
    central_lat=70.0, central_lon=130.0, secant_latitudes=[40.0, 60.0], ellipsoid=_GLOBE
)


class TestMakeGridcube:
    def test_default(self):
        cube = make_gridcube()
        assert isinstance(cube, Cube)
        assert cube.standard_name is None
        assert cube.long_name == "grid_cube"
        assert cube.coord_system() == _GLOBE

        assert len(cube.coords()) == 2
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]

        assert co_x.standard_name == "longitude"
        assert co_x.units == "degrees"
        assert co_x.shape == (30,)
        assert not co_x.has_bounds()
        assert np.all(co_x.points == np.linspace(0.0, 360.0, 30))

        assert co_y.standard_name == "latitude"
        assert co_y.units == "degrees"
        assert co_y.shape == (20,)
        assert not co_y.has_bounds()
        assert np.all(co_y.points == np.linspace(-90.0, 90.0, 20))

        assert cube.has_lazy_data()
        assert cube.shape == (20, 30)
        assert np.all(cube.data == 0)

    def test_points_region(self):
        """Check use of n? and ?lims args."""
        cube = make_gridcube(xlims=(20, 30), nx=5, ylims=(10, 25), ny=4)
        assert cube.shape == (4, 5)
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]
        assert np.allclose(co_x.points, [20.0, 22.5, 25.0, 27.5, 30.0])
        assert np.allclose(co_y.points, [10.0, 15.0, 20.0, 25.0])

    def test_points_positional(self):
        """Check function of positional arguments."""
        cube = make_gridcube(3, 4, (20, 40), (10, 25))
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]
        assert np.allclose(co_x.points, [20.0, 30.0, 40.0])
        assert np.allclose(co_y.points, [10.0, 15.0, 20.0, 25.0])

    def test_points_values(self):
        """Check use of full (irregular) points arrays."""
        xpts = [1.0, 2.0, 4.0, 5.0]
        ypts = [10.0, -13.0, -20.0]
        # NB we set the nx/ny/xlims/ylims, to show that they are ignored.
        cube = make_gridcube(
            nx=7, ny=5, xlims=(0, 10), ylims=(0, 10.0), x_points=xpts, y_points=ypts
        )
        assert cube.shape == (3, 4)
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]
        assert np.allclose(co_x.points, xpts)
        assert np.allclose(co_y.points, ypts)

    def test_points_mixed(self):
        """Check that you can have X and Y specified in different ways."""
        xpts = [1.0, 2.0, 5.0, 15.0]
        cube = make_gridcube(x_points=xpts, ny=3, ylims=(0.0, 20.0))
        assert cube.shape == (3, 4)
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]
        assert np.allclose(co_x.points, xpts)
        assert np.allclose(co_y.points, [0.0, 10.0, 20.0])

    @pytest.mark.parametrize("cs", ["latlon", "rotated", "projection"])
    def test_coord_system(self, cs):
        """Test with different coord-system types.

        That is, with the three different possible coordinate standard_name choices.
        """
        coord_system = {
            "latlon": _GLOBE,
            "rotated": _CS_ROTATED,
            "projection": _CS_LAMBERT,
        }[cs]

        cube = make_gridcube(coord_system=coord_system)

        assert cube.coord_system() == coord_system
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]

        expect_names = {
            "latlon": ["longitude", "latitude"],
            "rotated": ["grid_longitude", "grid_latitude"],
            "projection": ["projection_x_coordinate", "projection_y_coordinate"],
        }[cs]
        assert [co.standard_name for co in (co_x, co_y)] == expect_names

        expect_units = {
            "latlon": "degrees",
            "rotated": "degrees",
            "projection": "m",
        }[cs]
        assert co_x.units == expect_units
        assert co_y.units == expect_units
