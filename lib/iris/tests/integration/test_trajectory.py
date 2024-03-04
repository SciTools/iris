# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for :mod:`iris.analysis.trajectory`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

import iris
from iris._lazy_data import as_lazy_data
from iris.analysis.trajectory import Trajectory
from iris.analysis.trajectory import interpolate as traj_interpolate
import iris.tests.stock as istk


@tests.skip_data
class TestColpex(tests.IrisTest):
    def setUp(self):
        # Load the COLPEX data => TZYX
        path = tests.get_data_path(
            ["PP", "COLPEX", "theta_and_orog_subset.pp"]
        )
        cube = iris.load_cube(path, "air_potential_temperature")
        cube.coord("grid_latitude").bounds = None
        cube.coord("grid_longitude").bounds = None
        # TODO: Workaround until regrid can handle factories
        cube.remove_aux_factory(cube.aux_factories[0])
        cube.remove_coord("surface_altitude")
        self.cube = cube

    def test_trajectory_extraction(self):
        # Pull out a single point - no interpolation required
        single_point = traj_interpolate(
            self.cube,
            [("grid_latitude", [-0.1188]), ("grid_longitude", [359.57958984])],
        )
        expected = self.cube[..., 10, 0].data
        self.assertArrayAllClose(
            single_point[..., 0].data, expected, rtol=2.0e-7
        )
        self.assertCML(
            single_point, ("trajectory", "single_point.cml"), checksum=False
        )

    def test_trajectory_extraction_calc(self):
        # Pull out another point and test against a manually calculated result.
        single_point = [
            ["grid_latitude", [-0.1188]],
            ["grid_longitude", [359.584090412]],
        ]
        scube = self.cube[0, 0, 10:11, 4:6]
        x0 = scube.coord("grid_longitude")[0].points
        x1 = scube.coord("grid_longitude")[1].points
        y0 = scube.data[0, 0]
        y1 = scube.data[0, 1]
        expected = y0 + ((y1 - y0) * ((359.584090412 - x0) / (x1 - x0)))
        trajectory_cube = traj_interpolate(scube, single_point)
        self.assertArrayAllClose(trajectory_cube.data, expected, rtol=2.0e-7)

    def _traj_to_sample_points(self, trajectory):
        sample_points = []
        src_points = trajectory.sampled_points
        for name in src_points[0].keys():
            values = [point[name] for point in src_points]
            sample_points.append((name, values))
        return sample_points

    def test_trajectory_extraction_axis_aligned(self):
        # Extract a simple, axis-aligned trajectory that is similar to an
        # indexing operation.
        # (It's not exactly the same because the source cube doesn't have
        # regular spacing.)
        waypoints = [
            {"grid_latitude": -0.1188, "grid_longitude": 359.57958984},
            {"grid_latitude": -0.1188, "grid_longitude": 359.66870117},
        ]
        trajectory = Trajectory(waypoints, sample_count=100)
        sample_points = self._traj_to_sample_points(trajectory)
        trajectory_cube = traj_interpolate(self.cube, sample_points)
        self.assertCML(
            trajectory_cube, ("trajectory", "constant_latitude.cml")
        )

    def test_trajectory_extraction_zigzag(self):
        # Extract a zig-zag trajectory
        waypoints = [
            {"grid_latitude": -0.1188, "grid_longitude": 359.5886},
            {"grid_latitude": -0.0828, "grid_longitude": 359.6606},
            {"grid_latitude": -0.0468, "grid_longitude": 359.6246},
        ]
        trajectory = Trajectory(waypoints, sample_count=20)
        sample_points = self._traj_to_sample_points(trajectory)
        trajectory_cube = traj_interpolate(self.cube[0, 0], sample_points)
        expected = np.array(
            [
                287.95953369,
                287.9190979,
                287.95550537,
                287.93240356,
                287.83850098,
                287.87869263,
                287.90942383,
                287.9463501,
                287.74365234,
                287.68856812,
                287.75588989,
                287.54611206,
                287.48522949,
                287.53356934,
                287.60217285,
                287.43795776,
                287.59701538,
                287.52468872,
                287.45025635,
                287.52716064,
            ],
            dtype=np.float32,
        )

        self.assertCML(
            trajectory_cube, ("trajectory", "zigzag.cml"), checksum=False
        )
        self.assertArrayAllClose(trajectory_cube.data, expected, rtol=2.0e-7)

    def test_colpex__nearest(self):
        # Check a smallish nearest-neighbour interpolation against a result
        # snapshot.
        test_cube = self.cube[0][0]
        # Test points on a regular grid, a bit larger than the source region.
        xmin, xmax = [
            fn(test_cube.coord(axis="x").points) for fn in (np.min, np.max)
        ]
        ymin, ymax = [
            fn(test_cube.coord(axis="x").points) for fn in (np.min, np.max)
        ]
        fractions = [-0.23, -0.01, 0.27, 0.624, 0.983, 1.052, 1.43]
        x_points = [xmin + frac * (xmax - xmin) for frac in fractions]
        y_points = [ymin + frac * (ymax - ymin) for frac in fractions]
        x_points, y_points = np.meshgrid(x_points, y_points)
        sample_points = [
            ("grid_longitude", x_points.flatten()),
            ("grid_latitude", y_points.flatten()),
        ]
        result = traj_interpolate(test_cube, sample_points, method="nearest")
        expected = [
            288.07168579,
            288.07168579,
            287.9367981,
            287.82736206,
            287.78564453,
            287.8374939,
            287.8374939,
            288.07168579,
            288.07168579,
            287.9367981,
            287.82736206,
            287.78564453,
            287.8374939,
            287.8374939,
            288.07168579,
            288.07168579,
            287.9367981,
            287.82736206,
            287.78564453,
            287.8374939,
            287.8374939,
            288.07168579,
            288.07168579,
            287.9367981,
            287.82736206,
            287.78564453,
            287.8374939,
            287.8374939,
            288.07168579,
            288.07168579,
            287.9367981,
            287.82736206,
            287.78564453,
            287.8374939,
            287.8374939,
            288.07168579,
            288.07168579,
            287.9367981,
            287.82736206,
            287.78564453,
            287.8374939,
            287.8374939,
            288.07168579,
            288.07168579,
            287.9367981,
            287.82736206,
            287.78564453,
            287.8374939,
            287.8374939,
        ]
        self.assertArrayAllClose(result.data, expected)


@tests.skip_data
class TestTriPolar(tests.IrisTest):
    def setUp(self):
        # load data
        cubes = iris.load(
            tests.get_data_path(["NetCDF", "ORCA2", "votemper.nc"])
        )
        cube = cubes[0]
        # The netCDF file has different data types for the points and
        # bounds of 'depth'. This wasn't previously supported, so we
        # emulate that old behaviour.
        b32 = cube.coord("depth").bounds.astype(np.float32)
        cube.coord("depth").bounds = b32
        self.cube = cube
        # define a latitude trajectory (put coords in a different order
        # to the cube, just to be awkward) although avoid south pole
        # singularity as a sample point and the issue of snapping to
        # multi-equidistant closest points from within orca antarctic hole
        latitudes = list(range(-80, 90, 2))
        longitudes = [-90] * len(latitudes)
        self.sample_points = [
            ("longitude", longitudes),
            ("latitude", latitudes),
        ]

    def test_tri_polar(self):
        # extract
        sampled_cube = traj_interpolate(
            self.cube, self.sample_points, method="nearest"
        )
        self.assertCML(
            sampled_cube, ("trajectory", "tri_polar_latitude_slice.cml")
        )

    def test_tri_polar_method_linear_fails(self):
        # Try to request linear interpolation.
        # Not allowed, as we have multi-dimensional coords.
        self.assertRaises(
            iris.exceptions.CoordinateMultiDimError,
            traj_interpolate,
            self.cube,
            self.sample_points,
            method="linear",
        )

    def test_tri_polar_method_unknown_fails(self):
        # Try to request unknown interpolation.
        self.assertRaises(
            ValueError,
            traj_interpolate,
            self.cube,
            self.sample_points,
            method="linekar",
        )

    def test_tri_polar__nearest(self):
        # Check a smallish nearest-neighbour interpolation against a result
        # snapshot.
        test_cube = self.cube
        # Use just one 2d layer, just to be faster.
        test_cube = test_cube[0][0]
        # Fix the fill value of the data to zero, just so that we get the same
        # result under numpy < 1.11 as with 1.11.
        # NOTE: numpy<1.11 *used* to assign missing data points into an
        # unmasked array as =0.0, now =fill-value.
        # TODO: arguably, we should support masked data properly in the
        # interpolation routine.  In the legacy code, that is unfortunately
        # just not the case.
        test_cube.data.fill_value = 0

        # Test points on a regular global grid, with unrelated steps + offsets
        # and an extended range of longitude values.
        x_points = np.arange(-185.23, +360.0, 73.123)
        y_points = np.arange(-89.12, +90.0, 42.847)
        x_points, y_points = np.meshgrid(x_points, y_points)
        sample_points = [
            ("longitude", x_points.flatten()),
            ("latitude", y_points.flatten()),
        ]
        result = traj_interpolate(test_cube, sample_points, method="nearest")
        expected = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            12.13186264,
            10.69991493,
            9.86881161,
            7.08723927,
            9.04308414,
            12.56258678,
            10.63761806,
            9.19426727,
            28.93525505,
            23.85289955,
            26.94649506,
            0.0,
            27.88831711,
            28.65439224,
            23.39414215,
            26.78363228,
            13.53453922,
            0.0,
            17.41485596,
            0.0,
            0.0,
            13.0413475,
            0.0,
            17.10849571,
            -1.67040622,
            -1.64783156,
            0.0,
            -1.97898054,
            -1.67642927,
            -1.65173221,
            -1.623945,
            0.0,
        ]

        self.assertArrayAllClose(result.data, expected)


class TestLazyData(tests.IrisTest):
    def test_hybrid_height(self):
        cube = istk.simple_4d_with_hybrid_height()
        # Put a lazy array into the cube so we can test deferred loading.
        cube.data = as_lazy_data(cube.data)

        # Use opionated grid-latitudes to avoid the issue of platform
        # specific behaviour within SciPy cKDTree choosing a different
        # equi-distant nearest neighbour point when there are multiple
        # valid candidates.
        traj = (
            ("grid_latitude", [20.4, 21.6, 22.6, 23.6]),
            ("grid_longitude", [31, 32, 33, 34]),
        )
        xsec = traj_interpolate(cube, traj, method="nearest")

        # Check that creating the trajectory hasn't led to the original
        # data being loaded.
        self.assertTrue(cube.has_lazy_data())
        self.assertCML([cube, xsec], ("trajectory", "hybrid_height.cml"))


if __name__ == "__main__":
    tests.main()
