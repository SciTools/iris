# (C) British Crown Copyright 2013 - 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.experimental.regrid import \
    regrid_bilinear_rectilinear_src_and_grid as regrid
from iris.aux_factory import HybridHeightFactory
from iris.coord_systems import GeogCS, OSGB
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests.stock import global_pp, realistic_4d

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import iris.quickplot as qplt


RESULT_DIR = ('experimental', 'regrid',
              'regrid_bilinear_rectilinear_src_and_grid')


@tests.skip_data
class TestInvalid(tests.IrisTest):
    def test_non_cube(self):
        array = np.zeros((3, 4))
        cube = global_pp()
        with self.assertRaises(TypeError):
            regrid(array, cube)
        with self.assertRaises(TypeError):
            regrid(cube, array)
        with self.assertRaises(TypeError):
            regrid(42, cube)
        with self.assertRaises(TypeError):
            regrid(cube, 42)

    def test_non_rectilinear_src(self):
        ok = global_pp()

        # Lat and/or lon missing
        bad = global_pp()
        bad.remove_coord('latitude')
        with self.assertRaises(ValueError):
            regrid(bad, ok)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

        bad = global_pp()
        bad.remove_coord('longitude')
        with self.assertRaises(ValueError):
            regrid(bad, ok)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

        bad = global_pp()
        bad.remove_coord('latitude')
        bad.remove_coord('longitude')
        with self.assertRaises(ValueError):
            regrid(bad, ok)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

        # Lat/lon not a DimCoord
        def demote_coord(coord_name):
            bad = global_pp()
            coord = bad.coord(coord_name)
            dims = bad.coord_dims(coord)
            bad.remove_coord(coord_name)
            aux_coord = AuxCoord.from_coord(coord)
            bad.add_aux_coord(aux_coord, dims)
            with self.assertRaises(ValueError):
                regrid(bad, ok)
            with self.assertRaises(ValueError):
                regrid(ok, bad)
        demote_coord('latitude')
        demote_coord('longitude')

        # Lat/lon share a single dimension
        bad = global_pp()
        lat = bad.coord('latitude')
        bad = bad[0, :lat.shape[0]]
        bad.remove_coord('latitude')
        bad.add_aux_coord(lat, 0)
        with self.assertRaises(ValueError):
            regrid(bad, ok)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_bad_georeference(self):
        ok = global_pp()

        bad = global_pp()
        bad.coord('latitude').coord_system = None
        bad.coord('longitude').coord_system = None
        with self.assertRaises(ValueError):
            regrid(bad, ok)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

        bad = global_pp()
        bad.coord('latitude').coord_system = None
        bad.coord('longitude').coord_system = GeogCS(6371000)
        with self.assertRaises(ValueError):
            regrid(bad, ok)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

        bad = global_pp()
        bad.coord('latitude').coord_system = GeogCS(6370000)
        bad.coord('longitude').coord_system = GeogCS(6371000)
        with self.assertRaises(ValueError):
            regrid(bad, ok)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_bad_units(self):
        ok = global_pp()

        bad = global_pp()
        bad.coord('longitude').units = 'radians'
        with self.assertRaises(ValueError):
            regrid(bad, ok)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

        bad = Cube(np.arange(12, dtype=np.float32).reshape(3, 4))
        cs = OSGB()
        y_coord = DimCoord(range(3), 'projection_y_coordinate', units='m',
                           coord_system=cs)
        x_coord = DimCoord(range(4), 'projection_x_coordinate', units='km',
                           coord_system=cs)
        bad.add_dim_coord(y_coord, 0)
        bad.add_dim_coord(x_coord, 1)
        with self.assertRaises(ValueError):
            regrid(bad, ok)
        with self.assertRaises(ValueError):
            regrid(ok, bad)


@tests.skip_data
class TestRotatedToLatLon(tests.IrisTest):
    def setUp(self):
        self.src = realistic_4d()[:5, :2, ::40, ::30]

    def test_single_point(self):
        src = self.src[0, 0]
        grid = global_pp()[:1, :1]
        # These coordinate values have been derived by converting the
        # rotated coordinates of src[1, 1] into lat/lon by using cs2cs.
        grid.coord('longitude').points = -3.144870
        grid.coord('latitude').points = 52.406444
        result = regrid(src, grid)
        self.assertEqual(src.data[1, 1], result.data)

    def test_transposed_src(self):
        # The source dimensions are in a non-standard order.
        src = self.src
        src.transpose([3, 1, 2, 0])
        grid = self._grid_subset()
        result = regrid(src, grid)
        result.transpose([3, 1, 2, 0])
        self.assertCMLApproxData(result, RESULT_DIR + ('subset.cml',))

    def _grid_subset(self):
        # The destination grid points are entirely contained within the
        # src grid points.
        grid = global_pp()[:4, :5]
        grid.coord('longitude').points = np.linspace(-3.182, -3.06, 5)
        grid.coord('latitude').points = np.linspace(52.372, 52.44, 4)
        return grid

    def test_reversed(self):
        src = self.src
        grid = self._grid_subset()
        result = regrid(src, grid[::-1])
        self.assertCMLApproxData(result[:, :, ::-1],
                                 RESULT_DIR + ('subset.cml',))
        result = regrid(src[:, :, ::-1], grid[::-1])
        self.assertCMLApproxData(result[:, :, ::-1],
                                 RESULT_DIR + ('subset.cml',))
        result = regrid(src[:, :, :, ::-1], grid[::-1])
        self.assertCMLApproxData(result[:, :, ::-1],
                                 RESULT_DIR + ('subset.cml',))
        result = regrid(src[:, :, ::-1, ::-1], grid[::-1])
        self.assertCMLApproxData(result[:, :, ::-1],
                                 RESULT_DIR + ('subset.cml',))

        result = regrid(src, grid[:, ::-1])
        self.assertCMLApproxData(result[:, :, :, ::-1],
                                 RESULT_DIR + ('subset.cml',))
        result = regrid(src[:, :, ::-1], grid[:, ::-1])
        self.assertCMLApproxData(result[:, :, :, ::-1],
                                 RESULT_DIR + ('subset.cml',))
        result = regrid(src[:, :, :, ::-1], grid[:, ::-1])
        self.assertCMLApproxData(result[:, :, :, ::-1],
                                 RESULT_DIR + ('subset.cml',))
        result = regrid(src[:, :, ::-1, ::-1], grid[:, ::-1])
        self.assertCMLApproxData(result[:, :, :, ::-1],
                                 RESULT_DIR + ('subset.cml',))

        result = regrid(src, grid[::-1, ::-1])
        self.assertCMLApproxData(result[:, :, ::-1, ::-1],
                                 RESULT_DIR + ('subset.cml',))
        result = regrid(src[:, :, ::-1], grid[::-1, ::-1])
        self.assertCMLApproxData(result[:, :, ::-1, ::-1],
                                 RESULT_DIR + ('subset.cml',))
        result = regrid(src[:, :, :, ::-1], grid[::-1, ::-1])
        self.assertCMLApproxData(result[:, :, ::-1, ::-1],
                                 RESULT_DIR + ('subset.cml',))
        result = regrid(src[:, :, ::-1, ::-1], grid[::-1, ::-1])
        self.assertCMLApproxData(result[:, :, ::-1, ::-1],
                                 RESULT_DIR + ('subset.cml',))

    def test_grid_subset(self):
        # The destination grid points are entirely contained within the
        # src grid points.
        grid = self._grid_subset()
        result = regrid(self.src, grid)
        self.assertCMLApproxData(result, RESULT_DIR + ('subset.cml',))

    def _big_grid(self):
        grid = self._grid_subset()
        big_grid = Cube(np.zeros((5, 10, 3, 4, 5)))
        big_grid.add_dim_coord(grid.coord('latitude'), 3)
        big_grid.add_dim_coord(grid.coord('longitude'), 4)
        return big_grid

    def test_grid_subset_big(self):
        # Add some extra dimensions to the destination Cube and
        # these should be safely ignored.
        big_grid = self._big_grid()
        result = regrid(self.src, big_grid)
        self.assertCMLApproxData(result, RESULT_DIR + ('subset.cml',))

    def test_grid_subset_big_transposed(self):
        # The order of the grid's dimensions (including the X and Y
        # dimensions) must not affect the result.
        big_grid = self._big_grid()
        big_grid.transpose([4, 0, 3, 1, 2])
        result = regrid(self.src, big_grid)
        self.assertCMLApproxData(result, RESULT_DIR + ('subset.cml',))

    def test_grid_subset_anon(self):
        # Must cope OK with anonymous source dimensions.
        src = self.src
        src.remove_coord('time')
        grid = self._grid_subset()
        result = regrid(src, grid)
        self.assertCMLApproxData(result, RESULT_DIR + ('subset_anon.cml',))

    def test_grid_subset_missing_data_1(self):
        # The destination grid points are entirely contained within the
        # src grid points AND we have missing data.
        src = self.src
        src.data = np.ma.MaskedArray(src.data)
        src.data[:, :, 0, 0] = np.ma.masked
        grid = self._grid_subset()
        result = regrid(src, grid)
        self.assertCMLApproxData(result, RESULT_DIR + ('subset_masked_1.cml',))

    def test_grid_subset_missing_data_2(self):
        # The destination grid points are entirely contained within the
        # src grid points AND we have missing data.
        src = self.src
        src.data = np.ma.MaskedArray(src.data)
        src.data[:, :, 1, 2] = np.ma.masked
        grid = self._grid_subset()
        result = regrid(src, grid)
        self.assertCMLApproxData(result, RESULT_DIR + ('subset_masked_2.cml',))

    def test_grid_partial_overlap(self):
        # The destination grid points are partially contained within the
        # src grid points.
        grid = global_pp()[:4, :4]
        grid.coord('longitude').points = np.linspace(-3.3, -3.06, 4)
        grid.coord('latitude').points = np.linspace(52.377, 52.43, 4)
        result = regrid(self.src, grid)
        self.assertCMLApproxData(result, RESULT_DIR + ('partial_overlap.cml',))

    def test_grid_no_overlap(self):
        # The destination grid points are NOT contained within the
        # src grid points.
        grid = global_pp()[:4, :4]
        grid.coord('longitude').points = np.linspace(-3.3, -3.2, 4)
        grid.coord('latitude').points = np.linspace(52.377, 52.43, 4)
        result = regrid(self.src, grid)
        self.assertCMLApproxData(result, RESULT_DIR + ('no_overlap.cml',))

    def test_grid_subset_missing_data_aux(self):
        # The destination grid points are entirely contained within the
        # src grid points AND we have missing data on the aux coordinate.
        src = self.src
        src.coord('surface_altitude').points[1, 2] = np.ma.masked
        grid = self._grid_subset()
        result = regrid(src, grid)
        self.assertCMLApproxData(result, RESULT_DIR + ('masked_altitude.cml',))


class TestNOP(tests.IrisTest):
    def test_nop(self):
        # The destination grid points are exactly the same as the
        # src grid points.
        src = realistic_4d()[:5, :2, ::40, ::30]
        grid = src.copy()
        result = regrid(src, grid)
        self.assertEqual(result, src)


@tests.skip_data
class TestCircular(tests.IrisTest):
    def setUp(self):
        src = global_pp()[::10, ::10]
        level_height = AuxCoord(0, long_name='level_height', units='m',
                                attributes={'positive': 'up'})
        sigma = AuxCoord(1, long_name='sigma')
        surface_altitude = AuxCoord((src.data - src.data.min()) * 50,
                                    'surface_altitude', units='m')
        src.add_aux_coord(level_height)
        src.add_aux_coord(sigma)
        src.add_aux_coord(surface_altitude, [0, 1])
        hybrid_height = HybridHeightFactory(level_height, sigma,
                                            surface_altitude)
        src.add_aux_factory(hybrid_height)
        self.src = src

        grid = global_pp()[:4, :4]
        grid.coord('longitude').points = grid.coord('longitude').points - 5
        self.grid = grid

    def test_non_circular(self):
        # Non-circular src -> non-circular grid
        result = regrid(self.src, self.grid)
        self.assertFalse(result.coord('longitude').circular)
        self.assertCMLApproxData(result, RESULT_DIR + ('non_circular.cml',))

    def test_circular_src(self):
        # Circular src -> non-circular grid
        src = self.src
        src.coord('longitude').circular = True
        result = regrid(src, self.grid)
        self.assertFalse(result.coord('longitude').circular)
        self.assertCMLApproxData(result, RESULT_DIR + ('circular_src.cml',))

    def test_circular_grid(self):
        # Non-circular src -> circular grid
        grid = self.grid
        grid.coord('longitude').circular = True
        result = regrid(self.src, grid)
        self.assertTrue(result.coord('longitude').circular)
        self.assertCMLApproxData(result, RESULT_DIR + ('circular_grid.cml',))

    def test_circular_src_and_grid(self):
        # Circular src -> circular grid
        src = self.src
        src.coord('longitude').circular = True
        grid = self.grid
        grid.coord('longitude').circular = True
        result = regrid(src, grid)
        self.assertTrue(result.coord('longitude').circular)
        self.assertCMLApproxData(result, RESULT_DIR + ('both_circular.cml',))


@tests.skip_data
@tests.skip_plot
class TestVisual(tests.GraphicsTest):
    def test_osgb_to_latlon(self):
        path = tests.get_data_path(
            ('NIMROD', 'uk2km', 'WO0000000003452',
             '201007020900_u1096_ng_ey00_visibility0180_screen_2km'))
        src = iris.load_cube(path)[0]
        src.data = src.data.astype(np.float32)
        grid = Cube(np.empty((73, 96)))
        cs = GeogCS(6370000)
        lat = DimCoord(np.linspace(46, 65, 73), 'latitude', units='degrees',
                       coord_system=cs)
        lon = DimCoord(np.linspace(-14, 8, 96), 'longitude', units='degrees',
                       coord_system=cs)
        grid.add_dim_coord(lat, 0)
        grid.add_dim_coord(lon, 1)
        result = regrid(src, grid)
        qplt.pcolor(result, antialiased=False)
        qplt.plt.gca().coastlines()
        self.check_graphic()

    def test_subsample(self):
        src = global_pp()
        grid = src[::2, ::3]
        result = regrid(src, grid)
        qplt.pcolormesh(result)
        qplt.plt.gca().coastlines()
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
