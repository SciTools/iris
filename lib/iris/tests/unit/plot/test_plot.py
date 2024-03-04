# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.plot.plot` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

import iris.coord_systems as ics
import iris.coords as coords
from iris.tests.unit.plot import TestGraphicStringCoord

if tests.MPL_AVAILABLE:
    import cartopy.crs as ccrs
    import cartopy.mpl.geoaxes
    from matplotlib.path import Path
    import matplotlib.pyplot as plt

    import iris.plot as iplt


@tests.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def setUp(self):
        super().setUp()
        self.cube = self.cube[0, :]
        self.lat_lon_cube = self.lat_lon_cube[0, :]

    def test_yaxis_labels(self):
        iplt.plot(self.cube, self.cube.coord("str_coord"))
        self.assertBoundsTickLabels("yaxis")

    def test_xaxis_labels(self):
        iplt.plot(self.cube.coord("str_coord"), self.cube)
        self.assertBoundsTickLabels("xaxis")

    def test_yaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        iplt.plot(self.cube, self.cube.coord("str_coord"), axes=ax)
        plt.close(fig)
        self.assertBoundsTickLabels("yaxis", ax)

    def test_xaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        iplt.plot(self.cube.coord("str_coord"), self.cube, axes=ax)
        plt.close(fig)
        self.assertBoundsTickLabels("xaxis", ax)

    def test_plot_longitude(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        iplt.plot(
            self.lat_lon_cube.coord("longitude"), self.lat_lon_cube, axes=ax
        )
        plt.close(fig)


@tests.skip_plot
class TestTrajectoryWrap(tests.IrisTest):
    """
    Test that a line plot of geographic coordinates wraps around the end of the
    coordinates rather than plotting across the map.

    """

    def setUp(self):
        plt.figure()
        self.geog_cs = ics.GeogCS(6371229.0)
        self.plate_carree = self.geog_cs.as_cartopy_projection()

    def lon_lat_coords(self, lons, lats, cs=None):
        if cs is None:
            cs = self.geog_cs
        return (
            coords.AuxCoord(
                lons, "longitude", units="degrees", coord_system=cs
            ),
            coords.AuxCoord(
                lats, "latitude", units="degrees", coord_system=cs
            ),
        )

    def assertPathsEqual(self, expected, actual):
        """
        Assert that the given paths are equal once STOP vertices have been
        removed

        """
        expected = expected.cleaned()
        actual = actual.cleaned()
        # Remove Path.STOP vertices
        everts = expected.vertices[np.where(expected.codes != Path.STOP)]
        averts = actual.vertices[np.where(actual.codes != Path.STOP)]
        self.assertArrayAlmostEqual(everts, averts)
        self.assertArrayEqual(expected.codes, actual.codes)

    def check_paths(self, expected_path, expected_path_crs, lines, axes):
        """
        Check that the paths in `lines` match the given expected paths when
        plotted on the given geoaxes

        """

        self.assertEqual(
            1, len(lines), "Expected a single line, got {}".format(len(lines))
        )
        (line,) = lines
        inter_proj_transform = cartopy.mpl.geoaxes.InterProjectionTransform(
            expected_path_crs, axes.projection
        )
        ax_transform = inter_proj_transform + axes.transData

        expected = ax_transform.transform_path(expected_path)
        actual = line.get_transform().transform_path(line.get_path())

        self.assertPathsEqual(expected, actual)

    def test_simple(self):
        lon, lat = self.lon_lat_coords([359, 1], [0, 0])
        expected_path = Path([[-1, 0], [1, 0]], [Path.MOVETO, Path.LINETO])

        lines = iplt.plot(lon, lat)

        self.check_paths(expected_path, self.plate_carree, lines, plt.gca())

    def test_reverse(self):
        lon, lat = self.lon_lat_coords([1, 359], [0, 0])
        expected_path = Path([[1, 0], [-1, 0]], [Path.MOVETO, Path.LINETO])

        lines = iplt.plot(lon, lat)

        self.check_paths(expected_path, self.plate_carree, lines, plt.gca())

    def test_multi(self):
        lon, lat = self.lon_lat_coords([1, 359, 2, 358], [0, 0, 0, 0])
        expected_path = Path(
            [[1, 0], [-1, 0], [2, 0], [-2, 0]],
            [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO],
        )

        lines = iplt.plot(lon, lat)

        self.check_paths(expected_path, self.plate_carree, lines, plt.gca())

    def test_many_wraps(self):
        lon, lat = self.lon_lat_coords(
            [350, 10, 180, 350, 10, 180, 10, 350], [0, 0, 0, 0, 0, 0, 0, 0]
        )
        expected_path = Path(
            [
                [350, 0],
                [370, 0],
                [540, 0],
                [710, 0],
                [730, 0],
                [900, 0],
                [730, 0],
                [710, 0],
            ],
            [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
            ],
        )

        lines = iplt.plot(lon, lat)

        self.check_paths(expected_path, self.plate_carree, lines, plt.gca())

    def test_180(self):
        lon, lat = self.lon_lat_coords([179, -179], [0, 0])
        expected_path = Path([[179, 0], [181, 0]], [Path.MOVETO, Path.LINETO])

        lines = iplt.plot(lon, lat)

        self.check_paths(expected_path, self.plate_carree, lines, plt.gca())

    def test_shifted_projection(self):
        lon, lat = self.lon_lat_coords([359, 1], [0, 0])
        expected_path = Path([[-1, 0], [1, 0]], [Path.MOVETO, Path.LINETO])

        shifted_plate_carree = ccrs.PlateCarree(180)

        plt.axes(projection=shifted_plate_carree)
        lines = iplt.plot(lon, lat)

        self.check_paths(expected_path, self.plate_carree, lines, plt.gca())

    def test_shifted_projection_180(self):
        lon, lat = self.lon_lat_coords([179, -179], [0, 0])
        expected_path = Path([[179, 0], [181, 0]], [Path.MOVETO, Path.LINETO])

        shifted_plate_carree = ccrs.PlateCarree(180)

        plt.axes(projection=shifted_plate_carree)
        lines = iplt.plot(lon, lat)

        self.check_paths(expected_path, self.plate_carree, lines, plt.gca())

    def test_long(self):
        lon, lat = self.lon_lat_coords([271, 89], [0, 0])
        expected_path = Path([[-89, 0], [89, 0]], [Path.MOVETO, Path.LINETO])

        lines = iplt.plot(lon, lat)

        self.check_paths(expected_path, self.plate_carree, lines, plt.gca())

    def _test_rotated(
        self,
        grid_north_pole_latitude=90,
        grid_north_pole_longitude=0,
        north_pole_grid_longitude=0,
    ):
        cs = ics.RotatedGeogCS(
            grid_north_pole_latitude,
            grid_north_pole_longitude,
            north_pole_grid_longitude,
        )
        glon = coords.AuxCoord(
            [359, 1], "grid_longitude", units="degrees", coord_system=cs
        )
        glat = coords.AuxCoord(
            [0, 0], "grid_latitude", units="degrees", coord_system=cs
        )
        expected_path = Path([[-1, 0], [1, 0]], [Path.MOVETO, Path.LINETO])

        plt.figure()
        lines = iplt.plot(glon, glat)
        # Matplotlib won't immediately set up the correct transform to allow us
        # to compare paths. Calling set_global(), which calls set_xlim() and
        # set_ylim(), will trigger Matplotlib to set up the transform.
        ax = plt.gca()
        ax.set_global()

        crs = cs.as_cartopy_crs()
        self.check_paths(expected_path, crs, lines, ax)

    def test_rotated_90(self):
        self._test_rotated(north_pole_grid_longitude=90)

    def test_rotated_180(self):
        self._test_rotated(north_pole_grid_longitude=180)

    def test_rotated(self):
        self._test_rotated(
            grid_north_pole_latitude=-30,
            grid_north_pole_longitude=120,
            north_pole_grid_longitude=45,
        )


if __name__ == "__main__":
    tests.main()
