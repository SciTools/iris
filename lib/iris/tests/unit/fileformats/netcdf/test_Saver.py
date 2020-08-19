# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.netcdf.Saver` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from contextlib import contextmanager
from unittest import mock

import netCDF4 as nc
import numpy as np
from numpy import ma

import iris
from iris._lazy_data import as_lazy_data
from iris.coord_systems import (
    GeogCS,
    TransverseMercator,
    RotatedGeogCS,
    LambertConformal,
    Mercator,
    Stereographic,
    LambertAzimuthalEqualArea,
    AlbersEqualArea,
    VerticalPerspective,
    Geostationary,
)
from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.netcdf import Saver
import iris.tests.stock as stock


class Test_write(tests.IrisTest):
    # -------------------------------------------------------------------------
    # It is not considered necessary to have integration tests for saving
    # EVERY coordinate system. A subset are tested below.
    # -------------------------------------------------------------------------
    def _transverse_mercator_cube(self, ellipsoid=None):
        data = np.arange(12).reshape(3, 4)
        cube = Cube(data, "air_pressure_anomaly")
        trans_merc = TransverseMercator(
            49.0, -2.0, -400000.0, 100000.0, 0.9996012717, ellipsoid
        )
        coord = DimCoord(
            np.arange(3),
            "projection_y_coordinate",
            units="m",
            coord_system=trans_merc,
        )
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(
            np.arange(4),
            "projection_x_coordinate",
            units="m",
            coord_system=trans_merc,
        )
        cube.add_dim_coord(coord, 1)
        return cube

    def _mercator_cube(self, ellipsoid=None):
        data = np.arange(12).reshape(3, 4)
        cube = Cube(data, "air_pressure_anomaly")
        merc = Mercator(49.0, ellipsoid)
        coord = DimCoord(
            np.arange(3),
            "projection_y_coordinate",
            units="m",
            coord_system=merc,
        )
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(
            np.arange(4),
            "projection_x_coordinate",
            units="m",
            coord_system=merc,
        )
        cube.add_dim_coord(coord, 1)
        return cube

    def _stereo_cube(self, ellipsoid=None):
        data = np.arange(12).reshape(3, 4)
        cube = Cube(data, "air_pressure_anomaly")
        stereo = Stereographic(
            -10.0, 20.0, 500000.0, -200000.0, None, ellipsoid
        )
        coord = DimCoord(
            np.arange(3),
            "projection_y_coordinate",
            units="m",
            coord_system=stereo,
        )
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(
            np.arange(4),
            "projection_x_coordinate",
            units="m",
            coord_system=stereo,
        )
        cube.add_dim_coord(coord, 1)
        return cube

    def test_transverse_mercator(self):
        # Create a Cube with a transverse Mercator coordinate system.
        ellipsoid = GeogCS(6377563.396, 6356256.909)
        cube = self._transverse_mercator_cube(ellipsoid)
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    def test_transverse_mercator_no_ellipsoid(self):
        # Create a Cube with a transverse Mercator coordinate system.
        cube = self._transverse_mercator_cube()
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    def test_mercator(self):
        # Create a Cube with a Mercator coordinate system.
        ellipsoid = GeogCS(6377563.396, 6356256.909)
        cube = self._mercator_cube(ellipsoid)
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    def test_stereographic(self):
        # Create a Cube with a stereographic coordinate system.
        ellipsoid = GeogCS(6377563.396, 6356256.909)
        cube = self._stereo_cube(ellipsoid)
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    def test_mercator_no_ellipsoid(self):
        # Create a Cube with a Mercator coordinate system.
        cube = self._mercator_cube()
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    def test_stereographic_no_ellipsoid(self):
        # Create a Cube with a stereographic coordinate system.
        cube = self._stereo_cube()
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    def _simple_cube(self, dtype):
        data = np.arange(12, dtype=dtype).reshape(3, 4)
        points = np.arange(3, dtype=dtype)
        bounds = np.arange(6, dtype=dtype).reshape(3, 2)
        cube = Cube(data, "air_pressure_anomaly")
        coord = DimCoord(points, bounds=bounds, units="1")
        cube.add_dim_coord(coord, 0)
        return cube

    def test_little_endian(self):
        # Create a Cube with little-endian data.
        cube = self._simple_cube("<f4")
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            result_path = self.result_path("endian", "cdl")
            self.assertCDL(nc_path, result_path, flags="")

    def test_big_endian(self):
        # Create a Cube with big-endian data.
        cube = self._simple_cube(">f4")
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            result_path = self.result_path("endian", "cdl")
            self.assertCDL(nc_path, result_path, flags="")

    def test_zlib(self):
        cube = self._simple_cube(">f4")
        api = self.patch("iris.fileformats.netcdf.netCDF4")
        with Saver("/dummy/path", "NETCDF4") as saver:
            saver.write(cube, zlib=True)
        dataset = api.Dataset.return_value
        create_var_call = mock.call(
            "air_pressure_anomaly",
            np.dtype("float32"),
            ["dim0", "dim1"],
            fill_value=None,
            shuffle=True,
            least_significant_digit=None,
            contiguous=False,
            zlib=True,
            fletcher32=False,
            endian="native",
            complevel=4,
            chunksizes=None,
        )
        self.assertIn(create_var_call, dataset.createVariable.call_args_list)

    def test_least_significant_digit(self):
        cube = Cube(
            np.array([1.23, 4.56, 7.89]),
            standard_name="surface_temperature",
            long_name=None,
            var_name="temp",
            units="K",
        )
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, least_significant_digit=1)
            cube_saved = iris.load_cube(nc_path)
            self.assertEqual(
                cube_saved.attributes["least_significant_digit"], 1
            )
            self.assertFalse(np.all(cube.data == cube_saved.data))
            self.assertArrayAllClose(cube.data, cube_saved.data, 0.1)

    def test_default_unlimited_dimensions(self):
        # Default is no unlimited dimensions.
        cube = self._simple_cube(">f4")
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            ds = nc.Dataset(nc_path)
            self.assertFalse(ds.dimensions["dim0"].isunlimited())
            self.assertFalse(ds.dimensions["dim1"].isunlimited())
            ds.close()

    def test_no_unlimited_dimensions(self):
        cube = self._simple_cube(">f4")
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=None)
            ds = nc.Dataset(nc_path)
            for dim in ds.dimensions.values():
                self.assertFalse(dim.isunlimited())
            ds.close()

    def test_invalid_unlimited_dimensions(self):
        cube = self._simple_cube(">f4")
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                # should not raise an exception
                saver.write(cube, unlimited_dimensions=["not_found"])

    def test_custom_unlimited_dimensions(self):
        cube = self._transverse_mercator_cube()
        unlimited_dimensions = [
            "projection_y_coordinate",
            "projection_x_coordinate",
        ]
        # test coordinates by name
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=unlimited_dimensions)
            ds = nc.Dataset(nc_path)
            for dim in unlimited_dimensions:
                self.assertTrue(ds.dimensions[dim].isunlimited())
            ds.close()
        # test coordinate arguments
        with self.temp_filename(".nc") as nc_path:
            coords = [cube.coord(dim) for dim in unlimited_dimensions]
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=coords)
            ds = nc.Dataset(nc_path)
            for dim in unlimited_dimensions:
                self.assertTrue(ds.dimensions[dim].isunlimited())
            ds.close()

    def test_reserved_attributes(self):
        cube = self._simple_cube(">f4")
        cube.attributes["dimensions"] = "something something_else"
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            ds = nc.Dataset(nc_path)
            res = ds.getncattr("dimensions")
            ds.close()
            self.assertEqual(res, "something something_else")

    def test_with_climatology(self):
        cube = stock.climatology_3d()
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            self.assertCDL(nc_path)


class Test__create_cf_bounds(tests.IrisTest):
    def _check_bounds_setting(self, climatological=False):
        # Generic test that can run with or without a climatological coord.
        cube = stock.climatology_3d()
        coord = cube.coord("time").copy()
        # Over-write original value from stock.climatology_3d with test value.
        coord.climatological = climatological

        # Set up expected strings.
        if climatological:
            property_name = "climatology"
            varname_extra = "climatology"
        else:
            property_name = "bounds"
            varname_extra = "bnds"
        boundsvar_name = "time_" + varname_extra

        # Set up arguments for testing _create_cf_bounds.
        saver = mock.MagicMock(spec=Saver)
        # NOTE: 'saver' must have spec=Saver to fake isinstance(save, Saver),
        # so it can pass as 'self' in the call to _create_cf_cbounds.
        # Mock a '_dataset' property; not automatic because 'spec=Saver'.
        saver._dataset = mock.MagicMock()
        # Mock the '_ensure_valid_dtype' method to return an object with a
        # suitable 'shape' and 'dtype'.
        saver._ensure_valid_dtype.return_value = mock.Mock(
            shape=coord.bounds.shape, dtype=coord.bounds.dtype
        )
        var = mock.MagicMock(spec=nc.Variable)

        # Make the main call.
        Saver._create_cf_bounds(saver, coord, var, "time")

        # Test the call of _setncattr in _create_cf_bounds.
        setncattr_call = mock.call(
            property_name, boundsvar_name.encode(encoding="ascii")
        )
        self.assertEqual(setncattr_call, var.setncattr.call_args)

        # Test the call of createVariable in _create_cf_bounds.
        dataset = saver._dataset
        expected_dimensions = var.dimensions + ("bnds",)
        create_var_call = mock.call(
            boundsvar_name, coord.bounds.dtype, expected_dimensions
        )
        self.assertEqual(create_var_call, dataset.createVariable.call_args)

    def test_set_bounds_default(self):
        self._check_bounds_setting(climatological=False)

    def test_set_bounds_climatology(self):
        self._check_bounds_setting(climatological=True)


class Test_write__valid_x_cube_attributes(tests.IrisTest):
    """Testing valid_range, valid_min and valid_max attributes."""

    def test_valid_range_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        vrange = np.array([1, 2], dtype="int32")
        cube.attributes["valid_range"] = vrange
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = nc.Dataset(nc_path)
            self.assertArrayEqual(ds.valid_range, vrange)
            ds.close()

    def test_valid_min_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        cube.attributes["valid_min"] = 1
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = nc.Dataset(nc_path)
            self.assertArrayEqual(ds.valid_min, 1)
            ds.close()

    def test_valid_max_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        cube.attributes["valid_max"] = 2
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = nc.Dataset(nc_path)
            self.assertArrayEqual(ds.valid_max, 2)
            ds.close()


class Test_write__valid_x_coord_attributes(tests.IrisTest):
    """Testing valid_range, valid_min and valid_max attributes."""

    def test_valid_range_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        vrange = np.array([1, 2], dtype="int32")
        cube.coord(axis="x").attributes["valid_range"] = vrange
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = nc.Dataset(nc_path)
            self.assertArrayEqual(
                ds.variables["longitude"].valid_range, vrange
            )
            ds.close()

    def test_valid_min_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        cube.coord(axis="x").attributes["valid_min"] = 1
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = nc.Dataset(nc_path)
            self.assertArrayEqual(ds.variables["longitude"].valid_min, 1)
            ds.close()

    def test_valid_max_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        cube.coord(axis="x").attributes["valid_max"] = 2
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = nc.Dataset(nc_path)
            self.assertArrayEqual(ds.variables["longitude"].valid_max, 2)
            ds.close()


class Test_write_fill_value(tests.IrisTest):
    def _make_cube(
        self, dtype, lazy=False, masked_value=None, masked_index=None
    ):
        data = np.arange(12, dtype=dtype).reshape(3, 4)
        if masked_value is not None:
            data = ma.masked_equal(data, masked_value)
        if masked_index is not None:
            data = np.ma.masked_array(data)
            data[masked_index] = ma.masked
        if lazy:
            data = as_lazy_data(data)
        lat = DimCoord(np.arange(3), "latitude", units="degrees")
        lon = DimCoord(np.arange(4), "longitude", units="degrees")
        return Cube(
            data,
            standard_name="air_temperature",
            units="K",
            dim_coords_and_dims=[(lat, 0), (lon, 1)],
        )

    @contextmanager
    def _netCDF_var(self, cube, **kwargs):
        # Get the netCDF4 Variable for a cube from a temp file
        standard_name = cube.standard_name
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, **kwargs)
            ds = nc.Dataset(nc_path)
            (var,) = [
                var
                for var in ds.variables.values()
                if var.standard_name == standard_name
            ]
            yield var

    def test_fill_value(self):
        # Test that a passed fill value is saved as a _FillValue attribute.
        cube = self._make_cube(">f4")
        fill_value = 12345.0
        with self._netCDF_var(cube, fill_value=fill_value) as var:
            self.assertEqual(fill_value, var._FillValue)

    def test_default_fill_value(self):
        # Test that if no fill value is passed then there is no _FillValue.
        # attribute.
        cube = self._make_cube(">f4")
        with self._netCDF_var(cube) as var:
            self.assertNotIn("_FillValue", var.ncattrs())

    def test_mask_fill_value(self):
        # Test that masked data saves correctly when given a fill value.
        index = (1, 1)
        fill_value = 12345.0
        cube = self._make_cube(">f4", masked_index=index)
        with self._netCDF_var(cube, fill_value=fill_value) as var:
            self.assertEqual(fill_value, var._FillValue)
            self.assertTrue(var[index].mask)

    def test_mask_default_fill_value(self):
        # Test that masked data saves correctly using the default fill value.
        index = (1, 1)
        cube = self._make_cube(">f4", masked_index=index)
        with self._netCDF_var(cube) as var:
            self.assertNotIn("_FillValue", var.ncattrs())
            self.assertTrue(var[index].mask)

    def test_mask_lazy_fill_value(self):
        # Test that masked lazy data saves correctly when given a fill value.
        index = (1, 1)
        fill_value = 12345.0
        cube = self._make_cube(">f4", masked_index=index, lazy=True)
        with self._netCDF_var(cube, fill_value=fill_value) as var:
            self.assertEqual(var._FillValue, fill_value)
            self.assertTrue(var[index].mask)

    def test_mask_lazy_default_fill_value(self):
        # Test that masked lazy data saves correctly using the default fill
        # value.
        index = (1, 1)
        cube = self._make_cube(">f4", masked_index=index, lazy=True)
        with self._netCDF_var(cube) as var:
            self.assertNotIn("_FillValue", var.ncattrs())
            self.assertTrue(var[index].mask)

    def test_contains_fill_value_passed(self):
        # Test that a warning is raised if the data contains the fill value.
        cube = self._make_cube(">f4")
        fill_value = 1
        with self.assertWarnsRegexp(
            "contains unmasked data points equal to the fill-value"
        ):
            with self._netCDF_var(cube, fill_value=fill_value):
                pass

    def test_contains_fill_value_byte(self):
        # Test that a warning is raised if the data contains the fill value
        # when it is of a byte type.
        cube = self._make_cube(">i1")
        fill_value = 1
        with self.assertWarnsRegexp(
            "contains unmasked data points equal to the fill-value"
        ):
            with self._netCDF_var(cube, fill_value=fill_value):
                pass

    def test_contains_default_fill_value(self):
        # Test that a warning is raised if the data contains the default fill
        # value if no fill_value argument is supplied.
        cube = self._make_cube(">f4")
        cube.data[0, 0] = nc.default_fillvals["f4"]
        with self.assertWarnsRegexp(
            "contains unmasked data points equal to the fill-value"
        ):
            with self._netCDF_var(cube):
                pass

    def test_contains_default_fill_value_byte(self):
        # Test that no warning is raised if the data contains the default fill
        # value if no fill_value argument is supplied when the data is of a
        # byte type.
        cube = self._make_cube(">i1")
        with self.assertNoWarningsRegexp(r"\(fill\|mask\)"):
            with self._netCDF_var(cube):
                pass

    def test_contains_masked_fill_value(self):
        # Test that no warning is raised if the data contains the fill_value at
        # a masked point.
        fill_value = 1
        cube = self._make_cube(">f4", masked_value=fill_value)
        with self.assertNoWarningsRegexp(r"\(fill\|mask\)"):
            with self._netCDF_var(cube, fill_value=fill_value):
                pass

    def test_masked_byte_default_fill_value(self):
        # Test that a warning is raised when saving masked byte data with no
        # fill value supplied.
        cube = self._make_cube(">i1", masked_value=1)
        with self.assertNoWarningsRegexp(r"\(fill\|mask\)"):
            with self._netCDF_var(cube):
                pass

    def test_masked_byte_fill_value_passed(self):
        # Test that no warning is raised when saving masked byte data with a
        # fill value supplied if the the data does not contain the fill_value.
        fill_value = 100
        cube = self._make_cube(">i1", masked_value=2)
        with self.assertNoWarningsRegexp(r"\(fill\|mask\)"):
            with self._netCDF_var(cube, fill_value=fill_value):
                pass


class Test_cf_valid_var_name(tests.IrisTest):
    def test_no_replacement(self):
        self.assertEqual(Saver.cf_valid_var_name("valid_Nam3"), "valid_Nam3")

    def test_special_chars(self):
        self.assertEqual(Saver.cf_valid_var_name("inv?alid"), "inv_alid")

    def test_leading_underscore(self):
        self.assertEqual(Saver.cf_valid_var_name("_invalid"), "var__invalid")

    def test_leading_number(self):
        self.assertEqual(Saver.cf_valid_var_name("2invalid"), "var_2invalid")

    def test_leading_invalid(self):
        self.assertEqual(Saver.cf_valid_var_name("?invalid"), "var__invalid")

    def test_no_hyphen(self):
        # CF explicitly prohibits hyphen, even though it is fine in NetCDF.
        self.assertEqual(
            Saver.cf_valid_var_name("valid-netcdf"), "valid_netcdf"
        )


class _Common__check_attribute_compliance:
    def setUp(self):
        self.container = mock.Mock(name="container", attributes={})
        self.data = np.array(1, dtype="int32")

        patch = mock.patch("netCDF4.Dataset")
        _ = patch.start()
        self.addCleanup(patch.stop)

    def set_attribute(self, value):
        self.container.attributes[self.attribute] = value

    def assertAttribute(self, value):
        self.assertEqual(
            np.asarray(self.container.attributes[self.attribute]).dtype, value
        )

    def check_attribute_compliance_call(self, value):
        self.set_attribute(value)
        with Saver(mock.Mock(), "NETCDF4") as saver:
            saver.check_attribute_compliance(self.container, self.data)


class Test_check_attribute_compliance__valid_range(
    _Common__check_attribute_compliance, tests.IrisTest
):
    @property
    def attribute(self):
        return "valid_range"

    def test_valid_range_type_coerce(self):
        value = np.array([1, 2], dtype="float")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(self.data.dtype)

    def test_valid_range_unsigned_int8_data_signed_range(self):
        self.data = self.data.astype("uint8")
        value = np.array([1, 2], dtype="int8")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(value.dtype)

    def test_valid_range_cannot_coerce(self):
        value = np.array([1.5, 2.5], dtype="float64")
        msg = '"valid_range" is not of a suitable value'
        with self.assertRaisesRegex(ValueError, msg):
            self.check_attribute_compliance_call(value)

    def test_valid_range_not_numpy_array(self):
        # Ensure we handle the case when not a numpy array is provided.
        self.data = self.data.astype("int8")
        value = [1, 2]
        self.check_attribute_compliance_call(value)
        self.assertAttribute(np.int64)


class Test_check_attribute_compliance__valid_min(
    _Common__check_attribute_compliance, tests.IrisTest
):
    @property
    def attribute(self):
        return "valid_min"

    def test_valid_range_type_coerce(self):
        value = np.array(1, dtype="float")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(self.data.dtype)

    def test_valid_range_unsigned_int8_data_signed_range(self):
        self.data = self.data.astype("uint8")
        value = np.array(1, dtype="int8")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(value.dtype)

    def test_valid_range_cannot_coerce(self):
        value = np.array(1.5, dtype="float64")
        msg = '"valid_min" is not of a suitable value'
        with self.assertRaisesRegex(ValueError, msg):
            self.check_attribute_compliance_call(value)

    def test_valid_range_not_numpy_array(self):
        # Ensure we handle the case when not a numpy array is provided.
        self.data = self.data.astype("int8")
        value = 1
        self.check_attribute_compliance_call(value)
        self.assertAttribute(np.int64)


class Test_check_attribute_compliance__valid_max(
    _Common__check_attribute_compliance, tests.IrisTest
):
    @property
    def attribute(self):
        return "valid_max"

    def test_valid_range_type_coerce(self):
        value = np.array(2, dtype="float")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(self.data.dtype)

    def test_valid_range_unsigned_int8_data_signed_range(self):
        self.data = self.data.astype("uint8")
        value = np.array(2, dtype="int8")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(value.dtype)

    def test_valid_range_cannot_coerce(self):
        value = np.array(2.5, dtype="float64")
        msg = '"valid_max" is not of a suitable value'
        with self.assertRaisesRegex(ValueError, msg):
            self.check_attribute_compliance_call(value)

    def test_valid_range_not_numpy_array(self):
        # Ensure we handle the case when not a numpy array is provided.
        self.data = self.data.astype("int8")
        value = 2
        self.check_attribute_compliance_call(value)
        self.assertAttribute(np.int64)


class Test_check_attribute_compliance__exception_handlng(
    _Common__check_attribute_compliance, tests.IrisTest
):
    def test_valid_range_and_valid_min_valid_max_provided(self):
        # Conflicting attributes should raise a suitable exception.
        self.data = self.data.astype("int8")
        self.container.attributes["valid_range"] = [1, 2]
        self.container.attributes["valid_min"] = [1]
        msg = 'Both "valid_range" and "valid_min"'
        with Saver(mock.Mock(), "NETCDF4") as saver:
            with self.assertRaisesRegex(ValueError, msg):
                saver.check_attribute_compliance(self.container, self.data)


class Test__cf_coord_identity(tests.IrisTest):
    def check_call(self, coord_name, coord_system, units, expected_units):
        coord = iris.coords.DimCoord(
            [30, 45], coord_name, units=units, coord_system=coord_system
        )
        result = Saver._cf_coord_identity(coord)
        self.assertEqual(
            result, (coord.standard_name, coord.long_name, expected_units)
        )

    def test_geogcs_latitude(self):
        crs = iris.coord_systems.GeogCS(60, 0)
        self.check_call(
            "latitude",
            coord_system=crs,
            units="degrees",
            expected_units="degrees_north",
        )

    def test_geogcs_longitude(self):
        crs = iris.coord_systems.GeogCS(60, 0)
        self.check_call(
            "longitude",
            coord_system=crs,
            units="degrees",
            expected_units="degrees_east",
        )

    def test_no_coord_system_latitude(self):
        self.check_call(
            "latitude",
            coord_system=None,
            units="degrees",
            expected_units="degrees_north",
        )

    def test_no_coord_system_longitude(self):
        self.check_call(
            "longitude",
            coord_system=None,
            units="degrees",
            expected_units="degrees_east",
        )

    def test_passthrough_units(self):
        crs = iris.coord_systems.LambertConformal(0, 20)
        self.check_call(
            "projection_x_coordinate",
            coord_system=crs,
            units="km",
            expected_units="km",
        )


class Test__create_cf_grid_mapping(tests.IrisTest):
    def _cube_with_cs(self, coord_system):
        """Return a simple 2D cube that uses the given coordinate system."""
        cube = stock.lat_lon_cube()
        x, y = cube.coord("longitude"), cube.coord("latitude")
        x.coord_system = y.coord_system = coord_system
        return cube

    def _grid_mapping_variable(self, coord_system):
        """
        Return a mock netCDF variable that represents the conversion
        of the given coordinate system.

        """
        cube = self._cube_with_cs(coord_system)

        class NCMock(mock.Mock):
            def setncattr(self, name, attr):
                setattr(self, name, attr)

        # Calls the actual NetCDF saver with appropriate mocking, returning
        # the grid variable that gets created.
        grid_variable = NCMock(name="NetCDFVariable")
        create_var_fn = mock.Mock(side_effect=[grid_variable])
        dataset = mock.Mock(variables=[], createVariable=create_var_fn)
        saver = mock.Mock(spec=Saver, _coord_systems=[], _dataset=dataset)
        variable = NCMock()

        # This is the method we're actually testing!
        Saver._create_cf_grid_mapping(saver, cube, variable)

        self.assertEqual(create_var_fn.call_count, 1)
        self.assertEqual(
            variable.grid_mapping, grid_variable.grid_mapping_name
        )
        return grid_variable

    def _variable_attributes(self, coord_system):
        """
        Return the attributes dictionary for the grid mapping variable
        that is created from the given coordinate system.

        """
        mock_grid_variable = self._grid_mapping_variable(coord_system)

        # Get the attributes defined on the mock object.
        attributes = sorted(mock_grid_variable.__dict__.keys())
        attributes = [name for name in attributes if not name.startswith("_")]
        attributes.remove("method_calls")
        return {key: getattr(mock_grid_variable, key) for key in attributes}

    def _test(self, coord_system, expected):
        actual = self._variable_attributes(coord_system)

        # To see obvious differences, check that they keys are the same.
        self.assertEqual(sorted(actual.keys()), sorted(expected.keys()))
        # Now check that the values are equivalent.
        self.assertEqual(actual, expected)

    def test_rotated_geog_cs(self):
        coord_system = RotatedGeogCS(37.5, 177.5, ellipsoid=GeogCS(6371229.0))
        expected = {
            "grid_mapping_name": b"rotated_latitude_longitude",
            "north_pole_grid_longitude": 0.0,
            "grid_north_pole_longitude": 177.5,
            "grid_north_pole_latitude": 37.5,
            "longitude_of_prime_meridian": 0.0,
            "earth_radius": 6371229.0,
        }
        self._test(coord_system, expected)

    def test_spherical_geog_cs(self):
        coord_system = GeogCS(6371229.0)
        expected = {
            "grid_mapping_name": b"latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "earth_radius": 6371229.0,
        }
        self._test(coord_system, expected)

    def test_elliptic_geog_cs(self):
        coord_system = GeogCS(637, 600)
        expected = {
            "grid_mapping_name": b"latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_minor_axis": 600.0,
            "semi_major_axis": 637.0,
        }
        self._test(coord_system, expected)

    def test_lambert_conformal(self):
        coord_system = LambertConformal(
            central_lat=44,
            central_lon=2,
            false_easting=-2,
            false_northing=-5,
            secant_latitudes=(38, 50),
            ellipsoid=GeogCS(6371000),
        )
        expected = {
            "grid_mapping_name": b"lambert_conformal_conic",
            "latitude_of_projection_origin": 44,
            "longitude_of_central_meridian": 2,
            "false_easting": -2,
            "false_northing": -5,
            "standard_parallel": (38, 50),
            "earth_radius": 6371000,
            "longitude_of_prime_meridian": 0,
        }
        self._test(coord_system, expected)

    def test_laea_cs(self):
        coord_system = LambertAzimuthalEqualArea(
            latitude_of_projection_origin=52,
            longitude_of_projection_origin=10,
            false_easting=100,
            false_northing=200,
            ellipsoid=GeogCS(6377563.396, 6356256.909),
        )
        expected = {
            "grid_mapping_name": b"lambert_azimuthal_equal_area",
            "latitude_of_projection_origin": 52,
            "longitude_of_projection_origin": 10,
            "false_easting": 100,
            "false_northing": 200,
            "semi_major_axis": 6377563.396,
            "semi_minor_axis": 6356256.909,
            "longitude_of_prime_meridian": 0,
        }
        self._test(coord_system, expected)

    def test_aea_cs(self):
        coord_system = AlbersEqualArea(
            latitude_of_projection_origin=52,
            longitude_of_central_meridian=10,
            false_easting=100,
            false_northing=200,
            standard_parallels=(38, 50),
            ellipsoid=GeogCS(6377563.396, 6356256.909),
        )
        expected = {
            "grid_mapping_name": b"albers_conical_equal_area",
            "latitude_of_projection_origin": 52,
            "longitude_of_central_meridian": 10,
            "false_easting": 100,
            "false_northing": 200,
            "standard_parallel": (38, 50),
            "semi_major_axis": 6377563.396,
            "semi_minor_axis": 6356256.909,
            "longitude_of_prime_meridian": 0,
        }
        self._test(coord_system, expected)

    def test_vp_cs(self):
        latitude_of_projection_origin = 1.0
        longitude_of_projection_origin = 2.0
        perspective_point_height = 2000000.0
        false_easting = 100.0
        false_northing = 200.0

        semi_major_axis = 6377563.396
        semi_minor_axis = 6356256.909
        ellipsoid = GeogCS(semi_major_axis, semi_minor_axis)

        coord_system = VerticalPerspective(
            latitude_of_projection_origin=latitude_of_projection_origin,
            longitude_of_projection_origin=longitude_of_projection_origin,
            perspective_point_height=perspective_point_height,
            false_easting=false_easting,
            false_northing=false_northing,
            ellipsoid=ellipsoid,
        )
        expected = {
            "grid_mapping_name": b"vertical_perspective",
            "latitude_of_projection_origin": latitude_of_projection_origin,
            "longitude_of_projection_origin": longitude_of_projection_origin,
            "perspective_point_height": perspective_point_height,
            "false_easting": false_easting,
            "false_northing": false_northing,
            "semi_major_axis": semi_major_axis,
            "semi_minor_axis": semi_minor_axis,
            "longitude_of_prime_meridian": 0,
        }
        self._test(coord_system, expected)

    def test_geo_cs(self):
        latitude_of_projection_origin = 0.0
        longitude_of_projection_origin = 2.0
        perspective_point_height = 2000000.0
        sweep_angle_axis = "x"
        false_easting = 100.0
        false_northing = 200.0

        semi_major_axis = 6377563.396
        semi_minor_axis = 6356256.909
        ellipsoid = GeogCS(semi_major_axis, semi_minor_axis)

        coord_system = Geostationary(
            latitude_of_projection_origin=latitude_of_projection_origin,
            longitude_of_projection_origin=longitude_of_projection_origin,
            perspective_point_height=perspective_point_height,
            sweep_angle_axis=sweep_angle_axis,
            false_easting=false_easting,
            false_northing=false_northing,
            ellipsoid=ellipsoid,
        )
        expected = {
            "grid_mapping_name": b"geostationary",
            "latitude_of_projection_origin": latitude_of_projection_origin,
            "longitude_of_projection_origin": longitude_of_projection_origin,
            "perspective_point_height": perspective_point_height,
            "sweep_angle_axis": sweep_angle_axis,
            "false_easting": false_easting,
            "false_northing": false_northing,
            "semi_major_axis": semi_major_axis,
            "semi_minor_axis": semi_minor_axis,
            "longitude_of_prime_meridian": 0,
        }
        self._test(coord_system, expected)


class Test__create_cf_cell_measure_variable(tests.IrisTest):
    # Saving of masked data is disallowed.
    def setUp(self):
        self.cube = stock.lat_lon_cube()
        self.names_map = ["latitude", "longitude"]
        masked_array = np.ma.masked_array([0, 1, 2], mask=[True, False, True])
        self.cm = iris.coords.CellMeasure(masked_array, var_name="cell_area")
        self.cube.add_cell_measure(self.cm, data_dims=0)
        self.exp_emsg = "Cell measures with missing data are not supported."

    def test_masked_data__insitu(self):
        # Test that the error is raised in the right place.
        with self.temp_filename(".nc") as nc_path:
            saver = Saver(nc_path, "NETCDF4")
            with self.assertRaisesRegex(ValueError, self.exp_emsg):
                saver._create_cf_cell_measure_variable(
                    self.cube, self.names_map, self.cm
                )

    def test_masked_data__save_pipeline(self):
        # Test that the right error is raised by the saver pipeline.
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                with self.assertRaisesRegex(ValueError, self.exp_emsg):
                    saver.write(self.cube)


if __name__ == "__main__":
    tests.main()
