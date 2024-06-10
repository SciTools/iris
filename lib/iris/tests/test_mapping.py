# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests map creation."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs
import numpy as np
import numpy.testing as np_testing

import iris
import iris.coord_systems
import iris.cube
import iris.tests.stock

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt

    import iris.plot as iplt


# A specific cartopy Globe matching the iris RotatedGeogCS default.
_DEFAULT_GLOBE = ccrs.Globe(
    semimajor_axis=6371229.0, semiminor_axis=6371229.0, ellipse=None
)


@tests.skip_plot
@tests.skip_data
class TestBasic(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
        self.cube = iris.tests.stock.realistic_4d()

    def test_contourf(self):
        cube = self.cube[0, 0]
        iplt.contourf(cube)
        self.check_graphic()

    def test_pcolor(self):
        cube = self.cube[0, 0]
        iplt.pcolor(cube)
        self.check_graphic()

    def test_unmappable(self):
        cube = self.cube[0, 0]
        cube.coord("grid_longitude").standard_name = None
        iplt.contourf(cube)
        self.check_graphic()

    def test_default_projection_and_extent(self):
        self.assertEqual(
            iplt.default_projection(self.cube),
            ccrs.RotatedPole(357.5 - 180, 37.5, globe=_DEFAULT_GLOBE),
        )

        np_testing.assert_array_almost_equal(
            iplt.default_projection_extent(self.cube),
            (3.59579163e02, 3.59669159e02, -1.28250003e-01, -3.82499993e-02),
            decimal=3,
        )


@tests.skip_data
@tests.skip_plot
class TestUnmappable(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
        src_cube = iris.tests.stock.global_pp()

        # Make a cube that can't be located on the globe.
        cube = iris.cube.Cube(src_cube.data)
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(96, dtype=np.float32) * 100, long_name="x", units="m"
            ),
            1,
        )
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(73, dtype=np.float32) * 100, long_name="y", units="m"
            ),
            0,
        )
        cube.standard_name = "air_temperature"
        cube.units = "K"
        self.cube = cube

    def test_simple(self):
        iplt.contourf(self.cube, coords=["y", "x"])
        self.check_graphic()


@tests.skip_data
@tests.skip_plot
class TestMappingSubRegion(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
        cube_path = tests.get_data_path(("PP", "aPProt1", "rotatedMHtimecube.pp"))
        cube = iris.load_cube(cube_path)[0]
        # make the data smaller to speed things up.
        self.cube = cube[::10, ::10]

    def test_simple(self):
        # First sub-plot
        plt.subplot(221)
        plt.title("Default")
        iplt.contourf(self.cube)
        plt.gca().coastlines("110m")

        # Second sub-plot
        plt.subplot(222, projection=ccrs.Mollweide(central_longitude=120))
        plt.title("Molleweide")
        iplt.contourf(self.cube)
        plt.gca().coastlines("110m")

        # Third sub-plot (the projection part is redundant, but a useful
        # test none-the-less)
        ax = plt.subplot(223, projection=iplt.default_projection(self.cube))
        plt.title("Native")
        iplt.contour(self.cube)
        ax.coastlines("110m")

        # Fourth sub-plot
        ax = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
        plt.title("PlateCarree")
        iplt.contourf(self.cube)
        ax.coastlines("110m")

        self.check_graphic()

    def test_default_projection_and_extent(self):
        self.assertEqual(
            iplt.default_projection(self.cube),
            ccrs.RotatedPole(357.5 - 180, 37.5, globe=_DEFAULT_GLOBE),
        )

        np_testing.assert_array_almost_equal(
            iplt.default_projection_extent(self.cube),
            (313.01998901, 391.11999512, -22.48999977, 24.80999947),
        )


@tests.skip_data
@tests.skip_plot
class TestLowLevel(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
        self.cube = iris.tests.stock.global_pp()
        self.few = 4
        self.few_levels = list(range(280, 300, 5))
        self.many_levels = np.linspace(self.cube.data.min(), self.cube.data.max(), 40)

    def test_simple(self):
        iplt.contour(self.cube)
        self.check_graphic()

    def test_params(self):
        iplt.contourf(self.cube, self.few)
        self.check_graphic()

        iplt.contourf(self.cube, self.few_levels)
        self.check_graphic()

        iplt.contourf(self.cube, self.many_levels)
        self.check_graphic()

    def test_keywords(self):
        iplt.contourf(self.cube, levels=self.few_levels)
        self.check_graphic()

        iplt.contourf(self.cube, levels=self.many_levels, alpha=0.5)
        self.check_graphic()


@tests.skip_data
@tests.skip_plot
class TestBoundedCube(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
        self.cube = iris.tests.stock.global_pp()
        # Add some bounds to this data (this will actually make the bounds
        # invalid as they will straddle the north pole and overlap on the
        # dateline, but that doesn't matter for this test.)
        self.cube.coord("latitude").guess_bounds()
        self.cube.coord("longitude").guess_bounds()

    def test_pcolormesh(self):
        # pcolormesh can only be drawn in native coordinates (or more
        # specifically, in coordinates that don't wrap).
        plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        iplt.pcolormesh(self.cube)
        self.check_graphic()

    def test_grid(self):
        iplt.outline(self.cube)
        self.check_graphic()

    def test_default_projection_and_extent(self):
        self.assertEqual(
            iplt.default_projection(self.cube),
            ccrs.PlateCarree(
                globe=self.cube.coord_system("CoordSystem").as_cartopy_globe()
            ),
        )
        np_testing.assert_array_almost_equal(
            iplt.default_projection_extent(self.cube),
            [0.0, 360.0, -89.99995422, 89.99998474],
        )
        np_testing.assert_array_almost_equal(
            iplt.default_projection_extent(self.cube, mode=iris.coords.BOUND_MODE),
            [-1.875046, 358.124954, -90, 90],
        )


@tests.skip_data
@tests.skip_plot
class TestLimitedAreaCube(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
        cube_path = tests.get_data_path(("PP", "aPProt1", "rotated.pp"))
        self.cube = iris.load_cube(cube_path)[::20, ::20]
        self.cube.coord("grid_latitude").guess_bounds()
        self.cube.coord("grid_longitude").guess_bounds()

    def test_pcolormesh(self):
        iplt.pcolormesh(self.cube)
        self.check_graphic()

    def test_outline(self):
        iplt.outline(self.cube)
        self.check_graphic()

    def test_scatter(self):
        iplt.points(self.cube)
        plt.gca().coastlines("110m")
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
