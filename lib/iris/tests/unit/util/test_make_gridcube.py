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
        assert cube.is_dataless()
        assert cube.shape == (20, 30)

        assert len(cube.coords()) == 2
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]

        assert co_x.standard_name == "longitude"
        assert co_x.units == "degrees"
        assert co_x.shape == (30,)
        assert not co_x.has_bounds()
        assert np.all(co_x.points == np.linspace(0.0, 360.0, 30))
        assert co_x.points.dtype == np.dtype("f8")

        assert co_y.standard_name == "latitude"
        assert co_y.units == "degrees"
        assert co_y.shape == (20,)
        assert not co_y.has_bounds()
        assert np.all(co_y.points == np.linspace(-90.0, 90.0, 20))
        assert co_y.points.dtype == np.dtype("f8")

    def test_regular_region(self):
        """Check use of n? and ?lims args."""
        cube = make_gridcube(xlims=(20, 30), nx=5, ylims=(10, 25), ny=4)
        assert cube.shape == (4, 5)
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]
        assert np.allclose(co_x.points, [20.0, 22.5, 25.0, 27.5, 30.0])
        assert np.allclose(co_y.points, [10.0, 15.0, 20.0, 25.0])

    def test_regular_positional(self):
        """Check function of positional arguments."""
        cube = make_gridcube(3, 4, (20, 40), (10, 25))
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]
        assert np.allclose(co_x.points, [20.0, 30.0, 40.0])
        assert np.allclose(co_y.points, [10.0, 15.0, 20.0, 25.0])

    @pytest.mark.parametrize("nname", ["nx", "ny"])
    @pytest.mark.parametrize(
        "num", ["none", "object", "list", "array", "np_scalar", "-1", "0"]
    )
    def test_regular_badnumber__fail(self, num, nname):
        """Check errors from bad 'nx'/'ny'."""
        if num == "none":
            val = None
        elif num == "object":
            val = {}
        elif num == "list":
            val = [3]
        elif num == "array":
            val = np.array([3])
        elif num == "np_scalar":
            val = np.array(3)
        elif num in ("-1", "0"):
            # Set of obvious bad values
            val = int(num)
        else:
            raise ValueError(f"Unrecognised parameter : num = {num!r}")

        msg = rf"Bad value for '{nname}' arg.*Must be an integer >= 1"
        kwargs = {nname: val}
        with pytest.raises(ValueError, match=msg):
            make_gridcube(**kwargs)

    @pytest.mark.parametrize("usecase", ["single", "decrease", "single_zerorange"])
    def test_regular_cases(self, usecase):
        """Some specific testcases which should work."""
        if usecase == "single":
            nx = 1
            xlims = [10, 100]
            expect_xpts = [10]
        elif usecase == "decrease":
            nx = 3
            xlims = [40, -20]
            expect_xpts = [40, 10, -20]
        elif usecase == "single_zerorange":
            nx = 1
            xlims = [40, 40]
            expect_xpts = [40]
        else:
            raise ValueError(f"Unrecognised parameter : usecase = {usecase!r}")

        cube = make_gridcube(nx=nx, xlims=xlims)
        assert np.all(cube.coord(axis="x").points == expect_xpts)

    @pytest.mark.parametrize("lims", ["none", "object", "1pt", "3pt", "equal"])
    def test_regular_badlims__fail(self, lims):
        """Some input cases that should fail."""
        if lims == "none":
            lims = None
        elif lims == "object":
            lims = {}
        elif lims == "1pt":
            lims = [3]
        elif lims == "3pt":
            lims = [1, 2, 3]
        elif lims == "equal":
            lims = [10, 10]
        else:
            raise ValueError(f"Unrecognised parameter : lims = {lims!r}")

        msg = (
            "Bad value for 'xlims' arg.*"
            "Must be a pair of floats or ints, different unless `nx`=1"
        )
        with pytest.raises(ValueError, match=msg):
            make_gridcube(xlims=lims)

    @pytest.fixture(params=["int", "float", "i2", "i4", "i8", "f2", "f4", "f8"])
    def arg_dtype(self, request):
        """Check all valid numeric argument types."""
        yield request.param

    @staticmethod
    def f4_promoted_dtype(typename):
        """How ?points/?lims dtypes are promoted to define the coord dtype."""
        return {
            "i2": "f4",
            "i4": "f8",
            "i8": "f8",
            "f2": "f4",
            "f4": "f4",
            "f8": "f8",
        }[typename]

    def test_lims_types(self, arg_dtype):
        vals = [1, 2]
        expect_dtype = np.dtype("f8")
        if arg_dtype == "int":
            # Python ints
            xlims = vals
        elif arg_dtype == "float":
            # Python floats
            xlims = [float(x) for x in vals]
        else:
            # Various numpy dtypes
            xlims = np.asarray(vals, dtype=arg_dtype)
            # The dtypes here are the outcome of np.linspace:  OK with these.
            expect_dtype = self.f4_promoted_dtype(arg_dtype)

        # All valid arg dtypes are acceptable, and equivalent.
        cube = make_gridcube(xlims=xlims)
        # Point values are float64 in all cases.
        assert cube.coord(axis="x").points.dtype == expect_dtype

    def test_points(self):
        """Check use of full (irregular) points arrays."""
        # NB also show that either tuples/lists of floats/ints work.
        xpts = (1, 2, 4, 5)
        ypts = [10.0, -13.0, -20.0]
        # NB we set the nx/ny/xlims/ylims, to show that they are ignored.
        cube = make_gridcube(
            nx=7, ny=5, xlims=(0, 10), ylims=(0, 10.0), x_points=xpts, y_points=ypts
        )
        assert cube.shape == (3, 4)
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]
        assert np.allclose(co_x.points, xpts)
        assert np.allclose(co_y.points, ypts)

    def test_xregular_ypoints(self):
        """Check different spec types : X-regular / Y-points combination."""
        xpts = [1.0, 2.0, 5.0, 15.0]
        cube = make_gridcube(x_points=xpts, ny=3, ylims=(0.0, 20.0))
        assert cube.shape == (3, 4)
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]
        assert np.allclose(co_x.points, xpts)
        assert np.allclose(co_y.points, [0.0, 10.0, 20.0])

    def test_xpoints_yregular(self):
        """Check different spec types : X-points / Y-regular combination."""
        ypts = [1.0, 2.0, 5.0, 15.0]
        cube = make_gridcube(nx=3, xlims=(0.0, 20.0), y_points=ypts)
        assert cube.shape == (4, 3)
        co_x, co_y = [cube.coord(axis=ax) for ax in "xy"]
        assert np.allclose(co_x.points, [0.0, 10.0, 20.0])
        assert np.allclose(co_y.points, ypts)

    def test_points_types(self, arg_dtype):
        # Check type handling of points array creation
        vals = [1, 2, 5, 7]
        expect_dtype = np.dtype("f8")
        if arg_dtype == "int":
            # Python ints
            xpts = vals
        elif arg_dtype == "float":
            # Python floats
            xpts = [float(x) for x in vals]
        else:
            # Various numpy dtypes
            xpts = np.asarray(vals, dtype=arg_dtype)
            expect_dtype = self.f4_promoted_dtype(arg_dtype)

        cube = make_gridcube(x_points=xpts)
        co_x = cube.coord(axis="x")
        assert co_x.points.dtype == expect_dtype

    _POINTS_FAIL_MSG = (
        "Bad value for 'x_points' arg.*"
        "Must be a monotonic 1-d array-like of at least 1 floats or ints"
    )

    @pytest.mark.parametrize(
        "ptype",
        ["noniterable", "string", "2d", "strings", "objects"],
    )
    def test_points_badtypes__fail(self, ptype):
        # Check various bad types for points array arg.
        if ptype == "noniterable":
            pts = 17
        elif ptype == "string":
            pts = "this"
        elif ptype == "2d":
            pts = [[1, 2], [3, 4]]
        elif ptype == "strings":
            pts = ["ab", "cde"]
        elif ptype == "objects":
            pts = [None, {}]
        else:
            raise ValueError(f"Unrecognised parameter : ptype = {ptype!r}")

        with pytest.raises(ValueError, match=self._POINTS_FAIL_MSG):
            make_gridcube(x_points=pts)

    @pytest.mark.parametrize(
        "pvals", ["rising", "falling", "0_pts", "1_pts", "2_pts", "repeat", "nonmono"]
    )
    def test_points_values(self, pvals):
        # Check various cases where points values are valid or invalid.
        expect_ok = True
        if pvals == "rising":
            pts = [-3, 1, 2, 3]
        elif pvals == "falling":
            pts = [3, 2, 1, -4]
        elif pvals.endswith("_pts"):
            # zero or one point is not allowed, 2 is OK.
            n_pts = int(pvals[:1])
            pts = [3, 2, 1, -4][:n_pts]
            expect_ok = n_pts >= 1
        elif pvals == "repeat":
            # Repeated value (or pause in rise/fall) is an error.
            pts = [1, 2, 2, 3]
            expect_ok = False
        elif pvals == "nonmono":
            # Change in rise/fall is an error.
            pts = [1, 2, 3, 2]
            expect_ok = False
        else:
            raise ValueError(f"Unrecognised parameter : pvals = {pvals!r}")

        if expect_ok:
            assert make_gridcube(x_points=pts) is not None
        else:
            with pytest.raises(ValueError, match=self._POINTS_FAIL_MSG):
                make_gridcube(x_points=pts)

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
