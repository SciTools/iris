# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for regridding."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

import iris
from iris.analysis import UnstructuredNearest
from iris.analysis._regrid import RectilinearRegridder as Regridder
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests.stock import global_pp, simple_3d


@tests.skip_data
class TestOSGBToLatLon(tests.IrisTest):
    def setUp(self):
        path = tests.get_data_path(
            (
                "NIMROD",
                "uk2km",
                "WO0000000003452",
                "201007020900_u1096_ng_ey00_visibility0180_screen_2km",
            )
        )
        self.src = iris.load_cube(path)[0]
        # Cast up to float64, to work around numpy<=1.8 bug with means of
        # arrays of 32bit floats.
        self.src.data = self.src.data.astype(np.float64)
        self.grid = Cube(np.empty((73, 96)))
        cs = GeogCS(6370000)
        lat = DimCoord(
            np.linspace(46, 65, 73),
            "latitude",
            units="degrees",
            coord_system=cs,
        )
        lon = DimCoord(
            np.linspace(-14, 8, 96),
            "longitude",
            units="degrees",
            coord_system=cs,
        )
        self.grid.add_dim_coord(lat, 0)
        self.grid.add_dim_coord(lon, 1)

    def _regrid(self, method):
        regridder = Regridder(self.src, self.grid, method, "mask")
        result = regridder(self.src)
        return result

    def test_linear(self):
        res = self._regrid("linear")
        self.assertArrayShapeStats(res, (73, 96), 17799.296120, 11207.701323)

    def test_nearest(self):
        res = self._regrid("nearest")
        self.assertArrayShapeStats(res, (73, 96), 17808.068828, 11225.314310)


@tests.skip_data
class TestGlobalSubsample(tests.IrisTest):
    def setUp(self):
        self.src = global_pp()
        _ = self.src.data
        # Cast up to float64, to work around numpy<=1.8 bug with means of
        # arrays of 32bit floats.
        self.src.data = self.src.data.astype(np.float64)
        # Subsample and shift the target grid so that we can see a visual
        # difference between regridding scheme methods.
        grid = self.src[1::2, 1::3]
        grid.coord("latitude").points = grid.coord("latitude").points + 1
        grid.coord("longitude").points = grid.coord("longitude").points + 1
        self.grid = grid

    def _regrid(self, method):
        regridder = Regridder(self.src, self.grid, method, "mask")
        result = regridder(self.src)
        return result

    def test_linear(self):
        res = self._regrid("linear")
        self.assertArrayShapeStats(res, (36, 32), 280.35907, 15.997223)

    def test_nearest(self):
        res = self._regrid("nearest")
        self.assertArrayShapeStats(res, (36, 32), 280.33726, 16.064001)


@tests.skip_data
class TestUnstructured(tests.IrisTest):
    def setUp(self):
        path = tests.get_data_path(
            ("NetCDF", "unstructured_grid", "theta_nodal_not_ugrid.nc")
        )
        self.src = iris.load_cube(path, "Potential Temperature")
        self.grid = simple_3d()[0, :, :]

    def test_nearest(self):
        res = self.src.regrid(self.grid, UnstructuredNearest())
        self.assertArrayShapeStats(res, (1, 6, 3, 4), 315.890808, 11.000724)


class TestZonalMean_global(tests.IrisTest):
    def setUp(self):
        self.src = iris.cube.Cube(
            np.random.default_rng().integers(0, 10, size=(140, 1))
        )
        s_crs = iris.coord_systems.GeogCS(6371229.0)
        sy_coord = iris.coords.DimCoord(
            np.linspace(-90, 90, 140),
            standard_name="latitude",
            units="degrees",
            coord_system=s_crs,
        )
        sx_coord = iris.coords.DimCoord(
            -180,
            bounds=[-180, 180],
            standard_name="longitude",
            units="degrees",
            circular=True,
            coord_system=s_crs,
        )
        self.src.add_dim_coord(sy_coord, 0)
        self.src.add_dim_coord(sx_coord, 1)

    def test_linear_same_crs_global(self):
        # Regrid the zonal mean onto an identical coordinate system target, but
        # on a different set of longitudes - which should result in no change.
        points = [-150, -90, -30, 30, 90, 150]
        bounds = [
            [-180, -120],
            [-120, -60],
            [-60, 0],
            [0, 60],
            [60, 120],
            [120, 180],
        ]
        sx_coord = self.src.coord(axis="x")
        sy_coord = self.src.coord(axis="y")
        x_coord = sx_coord.copy(points, bounds=bounds)
        grid = iris.cube.Cube(np.zeros([sy_coord.points.size, x_coord.points.size]))
        grid.add_dim_coord(sy_coord, 0)
        grid.add_dim_coord(x_coord, 1)

        res = self.src.regrid(grid, iris.analysis.Linear())

        # Ensure data remains unchanged.
        # (the same along each column)
        self.assertTrue(
            np.array(
                [
                    (res.data[:, 0] - res.data[:, i]).max()
                    for i in range(1, res.shape[1])
                ]
            ).max()
            < 1e-10
        )
        self.assertArrayAlmostEqual(res.data[:, 0], self.src.data.reshape(-1))


class TestZonalMean_regional(TestZonalMean_global, tests.IrisTest):
    def setUp(self):
        super().setUp()

        # Define a target grid and a target result (what we expect the
        # regridder to return).
        sx_coord = self.src.coord(axis="x")
        sy_coord = self.src.coord(axis="y")
        grid_crs = iris.coord_systems.RotatedGeogCS(
            37.5, 177.5, ellipsoid=iris.coord_systems.GeogCS(6371229.0)
        )
        grid_x = sx_coord.copy(np.linspace(350, 370, 100))
        grid_x.circular = False
        grid_x.coord_system = grid_crs
        grid_y = sy_coord.copy(np.linspace(-10, 10, 100))
        grid_y.coord_system = grid_crs
        grid = iris.cube.Cube(np.zeros([grid_y.points.size, grid_x.points.size]))
        grid.add_dim_coord(grid_y, 0)
        grid.add_dim_coord(grid_x, 1)

        # The target result is derived by regridding a multi-column version of
        # the source to the target (i.e. turning a zonal mean regrid into a
        # conventional regrid).
        self.tar = self.zonal_mean_as_multi_column(self.src).regrid(
            grid, iris.analysis.Linear()
        )
        self.grid = grid

    def zonal_mean_as_multi_column(self, src_cube):
        # Munge the source (duplicate source latitudes) so that we can
        # utilise linear regridding as a conventional problem (that is, to
        # duplicate columns so that it is no longer a zonal mean problem).
        src_cube2 = src_cube.copy()
        src_cube2.coord(axis="x").points = -90
        src_cube2.coord(axis="x").bounds = [-180, 0]
        src_cube.coord(axis="x").points = 90
        src_cube.coord(axis="x").bounds = [0, 180]
        src_cubes = iris.cube.CubeList([src_cube, src_cube2])
        return src_cubes.concatenate_cube()

    def test_linear_rotated_regional(self):
        # Ensure that zonal mean source data is linearly interpolated onto a
        # high resolution target.
        regridder = iris.analysis.Linear()
        res = self.src.regrid(self.grid, regridder)
        self.assertArrayAlmostEqual(res.data, self.tar.data)

    def test_linear_rotated_regional_no_extrapolation(self):
        # Capture the case where our source remains circular but we don't use
        # extrapolation.
        regridder = iris.analysis.Linear(extrapolation_mode="nan")
        res = self.src.regrid(self.grid, regridder)
        self.assertArrayAlmostEqual(res.data, self.tar.data)

    def test_linear_rotated_regional_not_circular(self):
        # Capture the case where our source is not circular but we utilise
        # extrapolation.
        regridder = iris.analysis.Linear()
        self.src.coord(axis="x").circular = False
        res = self.src.regrid(self.grid, regridder)
        self.assertArrayAlmostEqual(res.data, self.tar.data)

    def test_linear_rotated_regional_no_extrapolation_not_circular(self):
        # Confirm how zonal mean actually works in so far as, that
        # extrapolation and circular source handling is the means by which
        # these usecases are supported.
        # In the case where the source is neither using extrapolation and is
        # not circular, then 'nan' values will result (as we would expect).
        regridder = iris.analysis.Linear(extrapolation_mode="nan")
        self.src.coord(axis="x").circular = False
        res = self.src.regrid(self.grid, regridder)
        self.assertTrue(np.isnan(res.data).all())


if __name__ == "__main__":
    tests.main()
