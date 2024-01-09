# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test area weighted regridding."""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import copy
import random

import dask.array as da
import numpy as np
import numpy.ma as ma

import iris.analysis._interpolation
from iris.experimental.regrid import (
    regrid_area_weighted_rectilinear_src_and_grid as regrid_area_weighted,
)
import iris.tests.stock

RESULT_DIR = (
    "experimental",
    "regrid",
    "regrid_area_weighted_rectilinear_src_and_grid",
)


def _scaled_and_offset_grid(
    cube, x_scalefactor, y_scalefactor, x_offset=0.0, y_offset=0.0
):
    """Return a cube with a horizontal grid that is scaled and offset
    from the horizontal grid of `src`.

    """
    x, y = iris.analysis._interpolation.get_xy_dim_coords(cube)
    new_cube = cube.copy()
    new_cube.replace_coord(x * x_scalefactor + x_offset)
    new_cube.replace_coord(y * y_scalefactor + y_offset)
    return new_cube


def _subsampled_coord(coord, subsamplefactor):
    """Return a coordinate that is a subsampled copy of `coord`.

    .. note:: `subsamplefactor` must be an integer >= 1.

    """
    if not isinstance(subsamplefactor, int):
        raise ValueError("subsamplefactor must be an integer.")
    if subsamplefactor < 1:
        raise ValueError("subsamplefactor must be >= 1.")
    if not coord.has_bounds():
        raise ValueError("The coordinate must have bounds.")
    new_coord = coord[::subsamplefactor]
    new_bounds = new_coord.bounds.copy()
    new_bounds[:, 1] = coord.bounds[(subsamplefactor - 1) :: subsamplefactor, 1]
    new_bounds[-1, 1] = coord.bounds[-1, 1]
    new_coord = coord.copy(points=new_coord.points, bounds=new_bounds)
    return new_coord


def _subsampled_grid(cube, x_subsamplefactor, y_subsamplefactor):
    """Return a cube that has a horizontal grid that is a subsampled
    version of the horizontal grid of `cube`.

    .. note:: The two subsamplefactors must both be integers >= 1.

    .. note:: The data of the returned cube is populated with zeros.

    """
    x, y = iris.analysis._interpolation.get_xy_dim_coords(cube)
    x_dim = cube.coord_dims(x)[0]
    y_dim = cube.coord_dims(y)[0]
    new_x = _subsampled_coord(x, x_subsamplefactor)
    new_y = _subsampled_coord(y, y_subsamplefactor)
    new_shape = list(cube.shape)
    new_shape[x_dim] = len(new_x.points)
    new_shape[y_dim] = len(new_y.points)
    new_data = np.zeros(new_shape)
    new_cube = iris.cube.Cube(new_data)
    new_cube.metadata = cube.metadata
    new_cube.add_dim_coord(new_y, y_dim)
    new_cube.add_dim_coord(new_x, x_dim)
    return new_cube


def _resampled_coord(coord, samplefactor):
    """Return a coordinate that has the same extent as `coord` but has
    `samplefactor` times as many points and bounds.

    """
    bounds = coord.bounds
    lower = bounds[0, 0]
    upper = bounds[-1, 1]
    # Prevent fp-precision increasing the extent by "squeezing" the grid.
    delta = 0.00001 * np.sign(upper - lower) * abs(bounds[0, 1] - bounds[0, 0])
    lower = lower + delta
    upper = upper - delta
    samples = int(len(bounds) * samplefactor)
    new_points, step = np.linspace(lower, upper, samples, endpoint=False, retstep=True)
    new_points += step * 0.5
    new_coord = coord.copy(points=new_points)
    new_coord.guess_bounds()
    return new_coord


def _resampled_grid(cube, x_samplefactor, y_samplefactor):
    """Return a cube that has the same horizontal extent as `cube` but has
    a reduced (or increased) number of points (and bounds) along the X and Y
    dimensions.

    The resulting number of points for each dimension is determined by::

        int(len(coord.points) * samplefactor)

    This will be truncated if the result is not an integer.

    .. note:: The data of the returned cube is populated with zeros.

    """
    x, y = iris.analysis._interpolation.get_xy_dim_coords(cube)
    x_dim = cube.coord_dims(x)[0]
    y_dim = cube.coord_dims(y)[0]
    new_x = _resampled_coord(x, x_samplefactor)
    new_y = _resampled_coord(y, y_samplefactor)
    new_shape = list(cube.shape)
    new_shape[x_dim] = len(new_x.points)
    new_shape[y_dim] = len(new_y.points)
    new_data = np.zeros(new_shape)
    new_cube = iris.cube.Cube(new_data)
    new_cube.metadata = cube.metadata
    new_cube.add_dim_coord(new_y, y_dim)
    new_cube.add_dim_coord(new_x, x_dim)
    return new_cube


@tests.skip_data
class TestAreaWeightedRegrid(tests.IrisTest):
    def setUp(self):
        # A cube with a hybrid height derived coordinate.
        self.realistic_cube = iris.tests.stock.realistic_4d()[:2, :5, :20, :30]
        # A simple (3, 4) cube.
        self.simple_cube = iris.tests.stock.lat_lon_cube()
        self.simple_cube.coord("latitude").guess_bounds(0.0)
        self.simple_cube.coord("longitude").guess_bounds(0.0)

    def test_no_bounds(self):
        src = self.simple_cube.copy()
        src.coord("latitude").bounds = None
        dest = self.simple_cube.copy()
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)

        src = self.simple_cube.copy()
        src.coord("longitude").bounds = None
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)

        src = self.simple_cube.copy()
        dest = self.simple_cube.copy()
        dest.coord("latitude").bounds = None
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)

        dest = self.simple_cube.copy()
        dest.coord("longitude").bounds = None
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)

    def test_non_contiguous_bounds(self):
        src = self.simple_cube.copy()
        bounds = src.coord("latitude").bounds.copy()
        bounds[1, 1] -= 0.1
        src.coord("latitude").bounds = bounds
        dest = self.simple_cube.copy()
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)

        src = self.simple_cube.copy()
        dest = self.simple_cube.copy()
        bounds = dest.coord("longitude").bounds.copy()
        bounds[1, 1] -= 0.1
        dest.coord("longitude").bounds = bounds
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)

    def test_missing_coords(self):
        dest = self.simple_cube.copy()
        # Missing src_x.
        src = self.simple_cube.copy()
        src.remove_coord("longitude")
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)
        # Missing src_y.
        src = self.simple_cube.copy()
        src.remove_coord("latitude")
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)
        # Missing dest_x.
        src = self.simple_cube.copy()
        dest = self.simple_cube.copy()
        dest.remove_coord("longitude")
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)
        # Missing dest_y.
        src = self.simple_cube.copy()
        dest = self.simple_cube.copy()
        dest.remove_coord("latitude")
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)

    def test_different_cs(self):
        src = self.simple_cube.copy()
        src_cs = copy.copy(src.coord("latitude").coord_system)
        src_cs.semi_major_axis = 7000000
        src.coord("longitude").coord_system = src_cs
        src.coord("latitude").coord_system = src_cs
        dest = self.simple_cube.copy()
        dest_cs = copy.copy(src_cs)
        dest_cs.semi_major_axis = 7000001
        dest.coord("longitude").coord_system = dest_cs
        dest.coord("latitude").coord_system = dest_cs
        with self.assertRaises(ValueError):
            regrid_area_weighted(src, dest)

    def test_regrid_to_same_grid(self):
        src = self.simple_cube
        res = regrid_area_weighted(src, src)
        self.assertEqual(res, src)
        self.assertCMLApproxData(res, RESULT_DIR + ("simple.cml",))

    def test_equal_area_numbers(self):
        # Remove coords system and units so it is no longer spherical.
        self.simple_cube.coord("latitude").coord_system = None
        self.simple_cube.coord("latitude").units = None
        self.simple_cube.coord("longitude").coord_system = None
        self.simple_cube.coord("longitude").units = None
        # Reduce to a single cell
        src = self.simple_cube.copy()
        dest = _subsampled_grid(src, 4, 3)
        res = regrid_area_weighted(src, dest)
        expected_val = np.mean(src.data)
        self.assertAlmostEqual(expected_val, res.data)

        # Reduce to two cells along x
        src = self.simple_cube.copy()
        dest = _subsampled_grid(src, 2, 3)
        res = regrid_area_weighted(src, dest)
        expected_val_left = np.mean(src.data[:, 0:2])
        self.assertEqual(expected_val_left, res.data[0])
        expected_val_right = np.mean(src.data[:, 2:4])
        self.assertAlmostEqual(expected_val_right, res.data[1])

        # Reduce to two cells along x, one three times the size
        # of the other.
        src = self.simple_cube.copy()
        dest = _subsampled_grid(src, 2, 3)
        lon = dest.coord("longitude")
        points = lon.points.copy()
        bounds = [[-1, 0], [0, 3]]
        lon = lon.copy(points=points, bounds=bounds)
        dest.replace_coord(lon)
        res = regrid_area_weighted(src, dest)
        expected_val_left = np.mean(src.data[:, 0:1])
        self.assertEqual(expected_val_left, res.data[0])
        expected_val_right = np.mean(src.data[:, 1:4])
        self.assertAlmostEqual(expected_val_right, res.data[1])

    def test_unqeual_area_numbers(self):
        # Remove coords system and units so it is no longer spherical.
        self.simple_cube.coord("latitude").coord_system = None
        self.simple_cube.coord("latitude").units = None
        self.simple_cube.coord("longitude").coord_system = None
        self.simple_cube.coord("longitude").units = None
        # Reduce src to two cells along x, one three times the size
        # of the other.
        src = self.simple_cube.copy()
        src = _subsampled_grid(src, 2, 2)
        lon = src.coord("longitude")
        points = lon.points.copy()
        bounds = [[-1, 0], [0, 3]]
        lon = lon.copy(points=points, bounds=bounds)
        src.replace_coord(lon)
        # Reduce src to two cells along y, one 2 times the size
        # of the other.
        lat = src.coord("latitude")
        points = lat.points.copy()
        bounds = [[-1, 0], [0, 2]]
        lat = lat.copy(points=points, bounds=bounds)
        src.replace_coord(lat)
        # Populate with data
        src.data = np.arange(src.data.size).reshape(src.shape) + 1.23
        # dest is a single cell over the whole area.
        dest = _subsampled_grid(self.simple_cube, 4, 3)
        res = regrid_area_weighted(src, dest)
        expected_val = (
            1.0 / 12.0 * src.data[0, 0]
            + 2.0 / 12.0 * np.mean(src.data[1:, 0])
            + 3.0 / 12.0 * np.mean(src.data[0, 1:])
            + 6.0 / 12.0 * np.mean(src.data[1:, 1:])
        )
        self.assertAlmostEqual(expected_val, res.data)

    def test_regrid_latlon_reduced_res(self):
        src = self.simple_cube
        # Reduce from (3, 4) to (2, 2).
        dest = _subsampled_grid(src, 2, 2)
        res = regrid_area_weighted(src, dest)
        self.assertCMLApproxData(res, RESULT_DIR + ("latlonreduced.cml",))

    def test_regrid_reorder_axis(self):
        src = self.realistic_cube[0, :4, :3, :2]
        z = src.coord("model_level_number")
        lat = src.coord("grid_latitude")
        lon = src.coord("grid_longitude")
        dest = _resampled_grid(self.realistic_cube[0, 0, :3, :2], 3, 3)
        res = regrid_area_weighted(src, dest)
        self.assertArrayShapeStats(src, (4, 3, 2), 288.08868, 0.008262919)
        self.assertArrayShapeStats(res, (4, 9, 6), 288.08865, 0.00826281)
        # Reshape src so that the coords are ordered [x, z, y],
        # the mean and std statistics should be the same
        data = np.moveaxis(src.data.copy(), 2, 0)
        src = iris.cube.Cube(data)
        src.add_dim_coord(lat, 2)
        src.add_dim_coord(z, 1)
        src.add_dim_coord(lon, 0)
        res = regrid_area_weighted(src, dest)
        self.assertArrayShapeStats(src, (2, 4, 3), 288.08868, 0.008262919)
        self.assertArrayShapeStats(res, (6, 4, 9), 288.08865, 0.00826281)
        # Reshape src so that the coords are ordered [y, x, z],
        # the mean and std statistics should be the same
        data = np.moveaxis(src.data.copy(), 2, 0)
        src = iris.cube.Cube(data)
        src.add_dim_coord(z, 2)
        src.add_dim_coord(lon, 1)
        src.add_dim_coord(lat, 0)
        dest = _resampled_grid(self.realistic_cube[0, 0, :3, :2], 3, 3)
        res = regrid_area_weighted(src, dest)
        self.assertArrayShapeStats(src, (3, 2, 4), 288.08868, 0.008262919)
        self.assertArrayShapeStats(res, (9, 6, 4), 288.08865, 0.00826281)

    def test_regrid_lon_to_half_res(self):
        src = self.simple_cube
        dest = _resampled_grid(src, 0.5, 1.0)
        res = regrid_area_weighted(src, dest)
        self.assertCMLApproxData(res, RESULT_DIR + ("lonhalved.cml",))

    def test_regrid_to_non_int_frac(self):
        # Create dest such that bounds do not line up
        # with src: src.shape = (3, 4), dest.shape = (2, 3)
        src = self.simple_cube
        dest = _resampled_grid(src, 0.75, 0.67)
        res = regrid_area_weighted(src, dest)
        self.assertCMLApproxData(res, RESULT_DIR + ("lower.cml",))

    def test_regrid_to_higher_res(self):
        src = self.simple_cube
        frac = 3.5
        dest = _resampled_grid(src, frac, frac)
        res = regrid_area_weighted(src, dest)
        self.assertCMLApproxData(res, RESULT_DIR + ("higher.cml",))

    def test_hybrid_height(self):
        src = self.realistic_cube
        dest = _resampled_grid(src, 0.7, 0.8)
        res = regrid_area_weighted(src, dest)
        self.assertCMLApproxData(res, RESULT_DIR + ("hybridheight.cml",))

    def test_missing_data(self):
        src = self.simple_cube.copy()
        src.data = ma.masked_array(src.data, fill_value=999)
        src.data[1, 2] = ma.masked
        dest = _resampled_grid(self.simple_cube, 2.3, 2.4)
        res = regrid_area_weighted(src, dest)
        mask = np.zeros((7, 9), bool)
        mask[slice(2, 5), slice(4, 7)] = True
        self.assertArrayEqual(res.data.mask, mask)
        self.assertArrayEqual(res.data.fill_value, 999)

    def test_masked_data_all_false(self):
        src = self.simple_cube.copy()
        src.data = ma.masked_array(src.data, mask=False, fill_value=999)
        dest = _resampled_grid(self.simple_cube, 2.3, 2.4)
        res = regrid_area_weighted(src, dest)
        self.assertArrayEqual(res.data.mask, False)
        self.assertArrayEqual(res.data.fill_value, 999)

    def test_no_x_overlap(self):
        src = self.simple_cube
        dest = _scaled_and_offset_grid(
            src,
            1.0,
            1.0,
            (
                np.max(src.coord("longitude").bounds)
                - np.min(src.coord("longitude").bounds)
            ),
            0.0,
        )
        res = regrid_area_weighted(src, dest)
        self.assertTrue(res.data.mask.all())

    def test_no_y_overlap(self):
        src = self.simple_cube
        dest = _scaled_and_offset_grid(
            src,
            1.0,
            1.0,
            0.0,
            (
                np.max(src.coord("latitude").bounds)
                - np.min(src.coord("latitude").bounds)
            ),
        )
        res = regrid_area_weighted(src, dest)
        self.assertTrue(res.data.mask.all())

    def test_scalar(self):
        src = self.realistic_cube
        i = 2
        j = 3
        dest = src[0, 0, i, j]
        res = regrid_area_weighted(src, dest)
        self.assertEqual(res, src[:, :, i, j])

    def test_one_point(self):
        src = self.simple_cube.copy()
        for n in range(10):
            i = random.randint(0, src.shape[0] - 1)
            j = random.randint(0, src.shape[1] - 1)
            indices = tuple([slice(i, i + 1), slice(j, j + 1)])
            dest = src[indices]
            res = regrid_area_weighted(src, dest)
            self.assertTrue(res, src[indices])

    def test_ten_by_ten_subset(self):
        src = _resampled_grid(self.simple_cube, 20, 20)
        for n in range(10):
            i = random.randint(0, src.shape[0] - 10)
            j = random.randint(0, src.shape[1] - 10)
            indices = tuple([slice(i, i + 10), slice(j, j + 10)])
            dest = src[indices]
            res = regrid_area_weighted(src, dest)
            self.assertEqual(res, src[indices])

    def test_lazy_nop(self):
        src = self.realistic_cube[:2, :3, :10, :10]
        src.data = da.asarray(src.data, chunks=((1, 1), (2, 1), (10,), (10,)))
        res = regrid_area_weighted(src, src)
        self.assertTrue(res.has_lazy_data())
        self.assertEqual(res, src)

    def test_cross_section(self):
        # Slice to get a cross section.
        # Constant latitude
        src = self.realistic_cube[0, :, 10, :]
        lon = _resampled_coord(src.coord("grid_longitude"), 0.6)
        shape = list(src.shape)
        shape[1] = len(lon.points)
        data = np.zeros(shape)
        dest = iris.cube.Cube(data)
        dest.add_dim_coord(lon, 1)
        dest.add_aux_coord(src.coord("grid_latitude").copy(), None)
        res = regrid_area_weighted(src, dest)
        self.assertCMLApproxData(res, RESULT_DIR + ("const_lat_cross_section.cml",))
        # Constant latitude, data order [x, z]
        # Using original and transposing the result should give the
        # same answer.
        src.transpose()
        dest.transpose()
        res = regrid_area_weighted(src, dest)
        res.transpose()
        self.assertCMLApproxData(res, RESULT_DIR + ("const_lat_cross_section.cml",))

        # Constant longitude
        src = self.realistic_cube[0, :, :, 10]
        lat = _resampled_coord(src.coord("grid_latitude"), 0.6)
        shape = list(src.shape)
        shape[1] = len(lat.points)
        data = np.zeros(shape)
        dest = iris.cube.Cube(data)
        dest.add_dim_coord(lat, 1)
        dest.add_aux_coord(src.coord("grid_longitude").copy(), None)
        res = regrid_area_weighted(src, dest)
        self.assertCMLApproxData(res, RESULT_DIR + ("const_lon_cross_section.cml",))
        # Constant longitude, data order [y, z]
        # Using original and transposing the result should give the
        # same answer.
        src.transpose()
        dest.transpose()
        res = regrid_area_weighted(src, dest)
        res.transpose()
        self.assertCMLApproxData(res, RESULT_DIR + ("const_lon_cross_section.cml",))

    def test_scalar_source_cube(self):
        src = self.simple_cube[1, 2]
        # Extend dest beyond src grid
        dest = src.copy()
        dest.coord("latitude").bounds = np.array([[-0.5, 1.5]])
        res = regrid_area_weighted(src, dest)
        self.assertTrue(res.data.mask.all())
        # Shrink dest to 1/4 of src
        dest = src.copy()
        dest.coord("latitude").bounds = np.array([[0.25, 0.75]])
        dest.coord("longitude").bounds = np.array([[1.25, 1.75]])
        res = regrid_area_weighted(src, dest)
        self.assertEqual(res.data, src.data)

    @tests.skip_data
    def test_global_data_reduce_res(self):
        src = iris.tests.stock.global_pp()
        src.coord("latitude").guess_bounds()
        src.coord("longitude").guess_bounds()
        dest = _resampled_grid(src, 0.4, 0.3)
        res = regrid_area_weighted(src, dest)
        self.assertArrayShapeStats(res, (21, 38), 280.484932, 15.831545)

    @tests.skip_data
    def test_global_data_increase_res(self):
        src = iris.tests.stock.global_pp()
        src.coord("latitude").guess_bounds()
        src.coord("longitude").guess_bounds()
        dest = _resampled_grid(src, 1.5, 1.5)
        res = regrid_area_weighted(src, dest)
        self.assertArrayShapeStats(res, (109, 144), 280.349625, 16.073397)

    @tests.skip_data
    def test_global_data_same_res(self):
        src = iris.tests.stock.global_pp()
        src.coord("latitude").guess_bounds()
        src.coord("longitude").guess_bounds()
        res = regrid_area_weighted(src, src)
        self.assertArrayShapeStats(res, (73, 96), 279.945160, 16.345842)

    @tests.skip_data
    def test_global_data_subset(self):
        src = iris.tests.stock.global_pp()
        src.coord("latitude").guess_bounds()
        src.coord("longitude").guess_bounds()
        dest_lat = src.coord("latitude")[0:40]
        dest_lon = iris.coords.DimCoord(
            np.linspace(-160, -70, 30),
            standard_name="longitude",
            units="degrees",
            coord_system=dest_lat.coord_system,
        )
        # Note target grid (in -180 to 180) src in 0 to 360
        dest_lon.guess_bounds()
        data = np.zeros((dest_lat.shape[0], dest_lon.shape[0]))
        dest = iris.cube.Cube(data)
        dest.add_dim_coord(dest_lat, 0)
        dest.add_dim_coord(dest_lon, 1)

        res = regrid_area_weighted(src, dest)
        self.assertArrayShapeStats(res, (40, 30), 280.979310, 16.640421)

    @tests.skip_data
    def test_circular_subset(self):
        src = iris.tests.stock.global_pp()
        src.coord("latitude").guess_bounds()
        src.coord("longitude").guess_bounds()
        dest_lat = src.coord("latitude")[0:40]
        dest_lon = iris.coords.DimCoord(
            [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0],
            standard_name="longitude",
            units="degrees",
            coord_system=dest_lat.coord_system,
        )
        # Note target grid (in -180 to 180) src in 0 to 360
        dest_lon.guess_bounds()
        data = np.zeros((dest_lat.shape[0], dest_lon.shape[0]))
        dest = iris.cube.Cube(data)
        dest.add_dim_coord(dest_lat, 0)
        dest.add_dim_coord(dest_lon, 1)

        res = regrid_area_weighted(src, dest)
        self.assertArrayShapeStats(res, (40, 7), 285.653960, 15.212710)

    @tests.skip_data
    def test_non_circular_subset(self):
        """Test regridding behaviour when the source grid has circular latitude.

        This tests the specific case when the longitude coordinate of the
        source grid has the `circular` attribute as `False` but otherwise spans
        the full 360 degrees.

        Note: the previous behaviour was to always mask target cells when they
        spanned the boundary of max/min longitude and `circular` was `False`,
        however this has been changed so that such cells will only be masked
        when there is a gap between max longitude and min longitude. In this
        test these cells are expected to be unmasked and therefore the result
        will be equal to the above test for circular longitudes.
        """
        src = iris.tests.stock.global_pp()
        src.coord("latitude").guess_bounds()
        src.coord("longitude").guess_bounds()
        src.coord("longitude").circular = False
        dest_lat = src.coord("latitude")[0:40]
        dest_lon = iris.coords.DimCoord(
            [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0],
            standard_name="longitude",
            units="degrees",
            coord_system=dest_lat.coord_system,
        )
        # Note target grid (in -180 to 180) src in 0 to 360
        dest_lon.guess_bounds()
        data = np.zeros((dest_lat.shape[0], dest_lon.shape[0]))
        dest = iris.cube.Cube(data)
        dest.add_dim_coord(dest_lat, 0)
        dest.add_dim_coord(dest_lon, 1)

        res = regrid_area_weighted(src, dest)
        self.assertArrayShapeStats(res, (40, 7), 285.653960, 15.212710)

    @tests.skip_data
    def test__proper_non_circular_subset(self):
        """Test regridding behaviour when the source grid has circular latitude.

        This tests the specific case when the longitude coordinate of the
        source grid does not span the full 360 degrees. Target cells which span
        the boundary of max/min longitude will contain a section which is out
        of bounds from the source grid and are therefore expected to be masked.
        """
        src = iris.tests.stock.global_pp()
        src.coord("latitude").guess_bounds()
        src.coord("longitude").guess_bounds()
        src_lon_bounds = src.coord("longitude").bounds.copy()
        # Leave a small gap between the first and last longitude value.
        src_lon_bounds[0, 0] += 0.001
        src_lon = src.coord("longitude").copy(
            points=src.coord("longitude").points, bounds=src_lon_bounds
        )
        src.remove_coord("longitude")
        src.add_dim_coord(src_lon, 1)
        dest_lat = src.coord("latitude")[0:40]
        dest_lon = iris.coords.DimCoord(
            [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0],
            standard_name="longitude",
            units="degrees",
            coord_system=dest_lat.coord_system,
        )
        # Note target grid (in -180 to 180) src in 0 to 360
        dest_lon.guess_bounds()
        data = np.zeros((dest_lat.shape[0], dest_lon.shape[0]))
        dest = iris.cube.Cube(data)
        dest.add_dim_coord(dest_lat, 0)
        dest.add_dim_coord(dest_lon, 1)

        res = regrid_area_weighted(src, dest)
        self.assertArrayShapeStats(res, (40, 7), 285.550814, 15.190245)

        # The target cells straddling the gap between min and max source
        # longitude should be masked.
        expected_mask = np.zeros(res.shape)
        expected_mask[:, 3] = 1
        assert np.array_equal(expected_mask, res.data.mask)


if __name__ == "__main__":
    tests.main()
