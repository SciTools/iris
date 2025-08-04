# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.fileformats.netcdf.Saver` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
from types import ModuleType

import iris.tests as tests  # isort:skip

import collections
from contextlib import contextmanager
from unittest import mock

import numpy as np
from numpy import ma
import pytest

import iris
from iris.coord_systems import (
    AlbersEqualArea,
    GeogCS,
    Geostationary,
    LambertAzimuthalEqualArea,
    LambertConformal,
    Mercator,
    ObliqueMercator,
    RotatedGeogCS,
    RotatedMercator,
    Stereographic,
    TransverseMercator,
    VerticalPerspective,
)
from iris.coords import AncillaryVariable, AuxCoord, DimCoord
from iris.cube import Cube
from iris.fileformats.netcdf import Saver, _thread_safe_nc
from iris.tests._shared_utils import assert_CDL
import iris.tests.stock as stock


class Test_write(tests.IrisTest):
    # -------------------------------------------------------------------------
    # It is not considered necessary to have integration tests for saving
    # EVERY coordinate system. A subset are tested below.
    # -------------------------------------------------------------------------

    # Attribute is substituted in test_Saver__lazy.
    array_lib: ModuleType = np

    def _transverse_mercator_cube(self, ellipsoid=None):
        data = self.array_lib.arange(12).reshape(3, 4)
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
        data = self.array_lib.arange(12).reshape(3, 4)
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

    def _stereo_cube(self, ellipsoid=None, scale_factor=None):
        data = self.array_lib.arange(12).reshape(3, 4)
        cube = Cube(data, "air_pressure_anomaly")
        stereo = Stereographic(
            -10.0,
            20.0,
            500000.0,
            -200000.0,
            None,
            ellipsoid,
            scale_factor_at_projection_origin=scale_factor,
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

    def test_stereographic_scale_factor(self):
        # Create a Cube with a stereographic coordinate system.
        cube = self._stereo_cube(scale_factor=1.3)
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    @staticmethod
    def _filter_compression_calls(patch, compression_kwargs, mismatch=False):
        result = []
        for call in patch.call_args_list:
            kwargs = call.kwargs
            if all(kwargs.get(k) == v for k, v in compression_kwargs.items()):
                if not mismatch:
                    result.append(call.args[0])
            elif mismatch:
                result.append(call.args[0])
        return result

    def _simple_cube(self, dtype):
        data = self.array_lib.arange(12, dtype=dtype).reshape(3, 4)
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
        api = self.patch("iris.fileformats.netcdf.saver._thread_safe_nc")
        # Define mocked default fill values to prevent deprecation warning (#4374).
        api.default_fillvals = collections.defaultdict(lambda: -99.0)
        # Mock the apparent dtype of mocked variables, to avoid an error.
        ref = api.DatasetWrapper.return_value
        ref = ref.createVariable.return_value
        ref.dtype = np.dtype(np.float32)
        # NOTE: use compute=False as otherwise it gets in a pickle trying to construct
        # a fill-value report on a non-compliant variable in a non-file (!)
        with Saver("/dummy/path", "NETCDF4", compute=False) as saver:
            saver.write(cube, zlib=True)
        dataset = api.DatasetWrapper.return_value
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

    def test_compression(self):
        cube = self._simple_cube(">f4")
        data_dims, shape = range(cube.ndim), cube.shape

        # add an auxiliary coordinate to test compression
        aux_coord = AuxCoord(np.zeros(shape), var_name="compress_aux", units="1")
        cube.add_aux_coord(aux_coord, data_dims=data_dims)

        # add an ancillary variable to test compression
        anc_coord = AncillaryVariable(
            np.zeros(shape), var_name="compress_anc", units="1"
        )
        cube.add_ancillary_variable(anc_coord, data_dims=data_dims)

        patch = self.patch(
            "iris.fileformats.netcdf.saver._thread_safe_nc.DatasetWrapper.createVariable"
        )
        compression_kwargs = {
            "complevel": 9,
            "fletcher32": True,
            "shuffle": True,
            "zlib": True,
        }

        with self.temp_filename(suffix=".nc") as nc_path:
            with Saver(nc_path, "NETCDF4", compute=False) as saver:
                saver.write(cube, **compression_kwargs)

        self.assertEqual(5, patch.call_count)
        result = self._filter_compression_calls(patch, compression_kwargs)
        self.assertEqual(3, len(result))
        self.assertEqual({cube.name(), aux_coord.name(), anc_coord.name()}, set(result))

    def test_non_compression__shape(self):
        cube = self._simple_cube(">f4")
        data_dims, shape = (0, 1), cube.shape

        # add an auxiliary coordinate to test non-compression (shape)
        aux_coord = AuxCoord(np.zeros(shape[0]), var_name="non_compress_aux", units="1")
        cube.add_aux_coord(aux_coord, data_dims=data_dims[0])

        # add an ancillary variable to test non-compression (shape)
        anc_coord = AncillaryVariable(
            np.zeros(shape[1]), var_name="non_compress_anc", units="1"
        )
        cube.add_ancillary_variable(anc_coord, data_dims=data_dims[1])

        patch = self.patch(
            "iris.fileformats.netcdf.saver._thread_safe_nc.DatasetWrapper.createVariable"
        )
        compression_kwargs = {
            "complevel": 9,
            "fletcher32": True,
            "shuffle": True,
            "zlib": True,
        }

        with self.temp_filename(suffix=".nc") as nc_path:
            with Saver(nc_path, "NETCDF4", compute=False) as saver:
                saver.write(cube, **compression_kwargs)

        self.assertEqual(5, patch.call_count)
        result = self._filter_compression_calls(
            patch, compression_kwargs, mismatch=True
        )
        self.assertEqual(4, len(result))
        # the aux coord and ancil variable are not compressed due to shape, and
        # the dim coord and its associated bounds are also not compressed
        expected = {aux_coord.name(), anc_coord.name(), "dim0", "dim0_bnds"}
        self.assertEqual(expected, set(result))

    def test_non_compression__dtype(self):
        cube = self._simple_cube(">f4")
        data_dims, shape = (0, 1), cube.shape

        # add an auxiliary coordinate to test non-compression (dtype)
        data = np.array(["."] * np.prod(shape)).reshape(shape)
        aux_coord = AuxCoord(data, var_name="non_compress_aux", units="1")
        cube.add_aux_coord(aux_coord, data_dims=data_dims)

        patch = self.patch(
            "iris.fileformats.netcdf.saver._thread_safe_nc.DatasetWrapper.createVariable"
        )
        patch.return_value = mock.MagicMock(dtype=np.dtype("S1"))
        compression_kwargs = {
            "complevel": 9,
            "fletcher32": True,
            "shuffle": True,
            "zlib": True,
        }

        with self.temp_filename(suffix=".nc") as nc_path:
            with Saver(nc_path, "NETCDF4", compute=False) as saver:
                saver.write(cube, **compression_kwargs)

        self.assertEqual(4, patch.call_count)
        result = self._filter_compression_calls(
            patch, compression_kwargs, mismatch=True
        )
        self.assertEqual(3, len(result))
        # the aux coord is not compressed due to its string dtype, and
        # the dim coord and its associated bounds are also not compressed
        expected = {aux_coord.name(), "dim0", "dim0_bnds"}
        self.assertEqual(expected, set(result))

    def test_least_significant_digit(self):
        cube = Cube(
            self.array_lib.array([1.23, 4.56, 7.89]),
            standard_name="surface_temperature",
            long_name=None,
            var_name="temp",
            units="K",
        )
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, least_significant_digit=1)
            cube_saved = iris.load_cube(nc_path)
            self.assertEqual(cube_saved.attributes["least_significant_digit"], 1)
            self.assertFalse(np.all(cube.data == cube_saved.data))
            self.assertArrayAllClose(cube.data, cube_saved.data, 0.1)

    def test_default_unlimited_dimensions(self):
        # Default is no unlimited dimensions.
        cube = self._simple_cube(">f4")
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            self.assertFalse(ds.dimensions["dim0"].isunlimited())
            self.assertFalse(ds.dimensions["dim1"].isunlimited())
            ds.close()

    def test_no_unlimited_dimensions(self):
        cube = self._simple_cube(">f4")
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=None)
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
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
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            for dim in unlimited_dimensions:
                self.assertTrue(ds.dimensions[dim].isunlimited())
            ds.close()
        # test coordinate arguments
        with self.temp_filename(".nc") as nc_path:
            coords = [cube.coord(dim) for dim in unlimited_dimensions]
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=coords)
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            for dim in unlimited_dimensions:
                self.assertTrue(ds.dimensions[dim].isunlimited())
            ds.close()

    def test_reserved_attributes(self):
        cube = self._simple_cube(">f4")
        cube.attributes["dimensions"] = "something something_else"
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            res = ds.getncattr("dimensions")
            ds.close()
            self.assertEqual(res, "something something_else")

    def test_with_climatology(self):
        cube = stock.climatology_3d()
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    def test_dimensional_to_scalar(self):
        # Bounds for 1 point are still in a 2D array.
        scalar_bounds = self.array_lib.arange(2).reshape(1, 2)
        scalar_point = scalar_bounds.mean()
        scalar_data = self.array_lib.zeros(1)
        scalar_coord = AuxCoord(points=scalar_point, bounds=scalar_bounds)
        cube = Cube(scalar_data, aux_coords_and_dims=[(scalar_coord, 0)])[0]
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube)
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            # Confirm that the only dimension is the one denoting the number
            #  of bounds - have successfully saved the 2D bounds array into 1D.
            self.assertEqual(["bnds"], list(ds.dimensions.keys()))
            ds.close()


class Test__create_cf_bounds(tests.IrisTest):
    # Method is substituted in test_Saver__lazy.
    @staticmethod
    def climatology_3d():
        return stock.climatology_3d()

    def _check_bounds_setting(self, climatological=False):
        # Generic test that can run with or without a climatological coord.
        cube = self.climatology_3d()
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
        var = mock.MagicMock(spec=_thread_safe_nc.VariableWrapper)

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

    # Attribute is substituted in test_Saver__lazy.
    array_lib: ModuleType = np

    def test_valid_range_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        vrange = self.array_lib.array([1, 2], dtype="int32")
        cube.attributes["valid_range"] = vrange
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            self.assertArrayEqual(ds.valid_range, vrange)
            ds.close()

    def test_valid_min_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        cube.attributes["valid_min"] = 1
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            self.assertArrayEqual(ds.valid_min, 1)
            ds.close()

    def test_valid_max_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        cube.attributes["valid_max"] = 2
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            self.assertArrayEqual(ds.valid_max, 2)
            ds.close()


class Test_write__valid_x_coord_attributes(tests.IrisTest):
    """Testing valid_range, valid_min and valid_max attributes."""

    # Attribute is substituted in test_Saver__lazy.
    array_lib: ModuleType = np

    def test_valid_range_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        vrange = self.array_lib.array([1, 2], dtype="int32")
        cube.coord(axis="x").attributes["valid_range"] = vrange
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            self.assertArrayEqual(ds.variables["longitude"].valid_range, vrange)
            ds.close()

    def test_valid_min_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        cube.coord(axis="x").attributes["valid_min"] = 1
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            self.assertArrayEqual(ds.variables["longitude"].valid_min, 1)
            ds.close()

    def test_valid_max_saved(self):
        cube = tests.stock.lat_lon_cube()
        cube.data = cube.data.astype("int32")

        cube.coord(axis="x").attributes["valid_max"] = 2
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
            self.assertArrayEqual(ds.variables["longitude"].valid_max, 2)
            ds.close()


class Test_write_fill_value(tests.IrisTest):
    # Attribute is substituted in test_Saver__lazy.
    array_lib: ModuleType = np

    def _make_cube(self, dtype, masked_value=None, masked_index=None):
        data = self.array_lib.arange(12, dtype=dtype).reshape(3, 4)
        if masked_value is not None:
            data = ma.masked_equal(data, masked_value)
        if masked_index is not None:
            data = self.array_lib.ma.masked_array(data)
            data[masked_index] = ma.masked
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
            ds = _thread_safe_nc.DatasetWrapper(nc_path)
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
        self.assertEqual(Saver.cf_valid_var_name("valid-netcdf"), "valid_netcdf")


class _Common__check_attribute_compliance:
    # Attribute is substituted in test_Saver__lazy.
    array_lib: ModuleType = np

    def setUp(self):
        self.container = mock.Mock(name="container", attributes={})
        self.data_dtype = np.dtype("int32")

        # We need to create mock datasets which look like they are closed.
        dataset_class = mock.Mock(
            return_value=mock.Mock(
                # Mock dataset : the isopen() call should return 0.
                isopen=mock.Mock(return_value=0)
            )
        )
        patch = mock.patch(
            "iris.fileformats.netcdf._thread_safe_nc.DatasetWrapper",
            dataset_class,
        )
        _ = patch.start()
        self.addCleanup(patch.stop)

    def set_attribute(self, value):
        self.container.attributes[self.attribute] = value

    def assertAttribute(self, value):
        self.assertEqual(
            np.asarray(self.container.attributes[self.attribute]).dtype, value
        )

    def check_attribute_compliance_call(self, value, file_type="NETCDF4"):
        self.set_attribute(value)
        with Saver("nonexistent test file", file_type) as saver:
            # Get the Mock to work properly.
            saver._dataset.file_format = file_type
            saver.check_attribute_compliance(self.container, self.data_dtype)


class Test_check_attribute_compliance__valid_range(
    _Common__check_attribute_compliance, tests.IrisTest
):
    @property
    def attribute(self):
        return "valid_range"

    def test_valid_range_type_coerce(self):
        value = self.array_lib.array([1, 2], dtype="float")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(self.data_dtype)

    def test_valid_range_unsigned_int8_data_signed_range(self):
        self.data_dtype = np.dtype("uint8")
        value = self.array_lib.array([1, 2], dtype="int8")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(value.dtype)

    def test_valid_range_cannot_coerce(self):
        value = self.array_lib.array([1.5, 2.5], dtype="float64")
        msg = '"valid_range" is not of a suitable value'
        with self.assertRaisesRegex(ValueError, msg):
            self.check_attribute_compliance_call(value)

    def test_valid_range_not_numpy_array(self):
        # Ensure we handle the case when not a numpy array is provided.
        self.data_dtype = np.dtype("int8")
        value = [1, 2]
        self.check_attribute_compliance_call(value)
        self.assertAttribute(np.int64)

    def test_uncastable_dtype(self):
        self.data_dtype = np.dtype("int64")
        value = [0, np.iinfo(self.data_dtype).max]
        with self.assertRaisesRegex(ValueError, "cannot be safely cast"):
            self.check_attribute_compliance_call(value, file_type="NETCDF4_CLASSIC")


class Test_check_attribute_compliance__valid_min(
    _Common__check_attribute_compliance, tests.IrisTest
):
    @property
    def attribute(self):
        return "valid_min"

    def test_valid_range_type_coerce(self):
        value = self.array_lib.array(1, dtype="float")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(self.data_dtype)

    def test_valid_range_unsigned_int8_data_signed_range(self):
        self.data_dtype = np.dtype("uint8")
        value = self.array_lib.array(1, dtype="int8")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(value.dtype)

    def test_valid_range_cannot_coerce(self):
        value = self.array_lib.array(1.5, dtype="float64")
        msg = '"valid_min" is not of a suitable value'
        with self.assertRaisesRegex(ValueError, msg):
            self.check_attribute_compliance_call(value)

    def test_valid_range_not_numpy_array(self):
        # Ensure we handle the case when not a numpy array is provided.
        self.data_dtype = np.dtype("int8")
        value = 1
        self.check_attribute_compliance_call(value)
        self.assertAttribute(np.int64)

    def test_uncastable_dtype(self):
        self.data_dtype = np.dtype("int64")
        value = np.iinfo(self.data_dtype).min
        with self.assertRaisesRegex(ValueError, "cannot be safely cast"):
            self.check_attribute_compliance_call(value, file_type="NETCDF4_CLASSIC")


class Test_check_attribute_compliance__valid_max(
    _Common__check_attribute_compliance, tests.IrisTest
):
    @property
    def attribute(self):
        return "valid_max"

    def test_valid_range_type_coerce(self):
        value = self.array_lib.array(2, dtype="float")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(self.data_dtype)

    def test_valid_range_unsigned_int8_data_signed_range(self):
        self.data_dtype = np.dtype("uint8")
        value = self.array_lib.array(2, dtype="int8")
        self.check_attribute_compliance_call(value)
        self.assertAttribute(value.dtype)

    def test_valid_range_cannot_coerce(self):
        value = self.array_lib.array(2.5, dtype="float64")
        msg = '"valid_max" is not of a suitable value'
        with self.assertRaisesRegex(ValueError, msg):
            self.check_attribute_compliance_call(value)

    def test_valid_range_not_numpy_array(self):
        # Ensure we handle the case when not a numpy array is provided.
        self.data_dtype = np.dtype("int8")
        value = 2
        self.check_attribute_compliance_call(value)
        self.assertAttribute(np.int64)

    def test_uncastable_dtype(self):
        self.data_dtype = np.dtype("int64")
        value = np.iinfo(self.data_dtype).max
        with self.assertRaisesRegex(ValueError, "cannot be safely cast"):
            self.check_attribute_compliance_call(value, file_type="NETCDF4_CLASSIC")


class Test_check_attribute_compliance__exception_handling(
    _Common__check_attribute_compliance, tests.IrisTest
):
    def test_valid_range_and_valid_min_valid_max_provided(self):
        # Conflicting attributes should raise a suitable exception.
        self.data_dtype = np.dtype("int8")
        self.container.attributes["valid_range"] = [1, 2]
        self.container.attributes["valid_min"] = [1]
        msg = 'Both "valid_range" and "valid_min"'
        with Saver("nonexistent test file", "NETCDF4") as saver:
            with self.assertRaisesRegex(ValueError, msg):
                saver.check_attribute_compliance(self.container, self.data_dtype)


class Test__cf_coord_identity(tests.IrisTest):
    def check_call(self, coord_name, coord_system, units, expected_units):
        coord = iris.coords.DimCoord(
            [30, 45], coord_name, units=units, coord_system=coord_system
        )
        result = Saver._cf_coord_standardised_units(coord)
        self.assertEqual(result, expected_units)

    def test_geogcs_latitude(self):
        crs = iris.coord_systems.GeogCS(60, 30)
        self.check_call(
            "latitude",
            coord_system=crs,
            units="degrees",
            expected_units="degrees_north",
        )

    def test_geogcs_longitude(self):
        crs = iris.coord_systems.GeogCS(60, 30)
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


@pytest.fixture
def transverse_mercator_cube_multi_cs():
    """A transverse mercator cube with an auxiliary GeogGS coordinate system."""
    data = np.arange(12).reshape(3, 4)
    cube = Cube(data, "air_pressure_anomaly")
    cube.extended_grid_mapping = True

    geog_cs = GeogCS(6377563.396, 6356256.909)
    trans_merc = TransverseMercator(
        49.0, -2.0, -400000.0, 100000.0, 0.9996012717, geog_cs
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

    # Add auxiliary lat/lon coords with a GeogCS coord system
    coord = AuxCoord(
        np.arange(3 * 4).reshape((3, 4)),
        "longitude",
        units="degrees",
        coord_system=geog_cs,
    )
    cube.add_aux_coord(coord, (0, 1))

    coord = AuxCoord(
        np.arange(3 * 4).reshape((3, 4)),
        "latitude",
        units="degrees",
        coord_system=geog_cs,
    )
    cube.add_aux_coord(coord, (0, 1))

    return cube


class Test_write_extended_grid_mapping:
    def test_multi_cs(self, transverse_mercator_cube_multi_cs, tmp_path, request):
        """Test writing a cube with multiple coordinate systems.
        Should generate a grid mapping using extended syntax that references
        both coordinate systems and the coords.
        """
        cube = transverse_mercator_cube_multi_cs
        nc_path = tmp_path / "tmp.nc"
        with Saver(nc_path, "NETCDF4") as saver:
            saver.write(cube)
        assert_CDL(request, nc_path)

    def test_no_aux_cs(self, transverse_mercator_cube_multi_cs, tmp_path, request):
        """Test when DimCoords have coord system, but AuxCoords do not.
        Should write extended grid mapping for just DimCoords.
        """
        cube = transverse_mercator_cube_multi_cs
        cube.coord("latitude").coord_system = None
        cube.coord("longitude").coord_system = None

        nc_path = tmp_path / "tmp.nc"
        with Saver(nc_path, "NETCDF4") as saver:
            saver.write(cube)
        assert_CDL(request, nc_path)

    def test_multi_cs_missing_coord(
        self, transverse_mercator_cube_multi_cs, tmp_path, request
    ):
        """Test when we have a missing coordinate.
        Grid mapping will fall back to simple mapping to DimCoord CS (no coords referenced).
        """
        cube = transverse_mercator_cube_multi_cs
        cube.remove_coord("latitude")
        nc_path = tmp_path / "tmp.nc"
        with Saver(nc_path, "NETCDF4") as saver:
            saver.write(cube)
        assert_CDL(request, nc_path)

    def test_no_cs(self, transverse_mercator_cube_multi_cs, tmp_path, request):
        """Test when no coordinate systems associated with cube coords.
        Grid mapping will not be generated at all.
        """
        cube = transverse_mercator_cube_multi_cs
        for coord in cube.coords():
            coord.coord_system = None

        nc_path = tmp_path / "tmp.nc"
        with Saver(nc_path, "NETCDF4") as saver:
            saver.write(cube)
        assert_CDL(request, nc_path)


class Test_create_cf_grid_mapping:
    """Tests correct generation of CF grid_mapping variable attributes.

    Note: The first 3 tests are run with the "extended grid" mapping
    both enabled (the default for all these tests) and disabled. This
    controls the output of the WKT attribute.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._extended_grid_mapping = True  # forces WKT strings to be written

    def _cube_with_cs(self, coord_system):
        """Return a simple 2D cube that uses the given coordinate system."""
        cube = stock.lat_lon_cube()
        x, y = cube.coord("longitude"), cube.coord("latitude")
        x.coord_system = y.coord_system = coord_system
        cube.extended_grid_mapping = self._extended_grid_mapping
        return cube

    def _grid_mapping_variable(self, coord_system):
        """Return a mock netCDF variable that represents the conversion
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
        variable = NCMock()

        saver = Saver(dataset, "NETCDF4", compute=False)

        # The method we want to test:
        saver._create_cf_grid_mapping(cube, variable)

        assert create_var_fn.call_count == 1
        assert variable.grid_mapping, grid_variable.grid_mapping_name
        return grid_variable

    def _variable_attributes(self, coord_system):
        """Return the attributes dictionary for the grid mapping variable
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
        assert sorted(actual.keys()) == sorted(expected.keys())
        # Now check that the values are equivalent.
        assert actual == expected

    def test_rotated_geog_cs(self, extended_grid_mapping):
        self._extended_grid_mapping = extended_grid_mapping
        coord_system = RotatedGeogCS(37.5, 177.5, ellipsoid=GeogCS(6371229.0))

        expected = {
            "grid_mapping_name": b"rotated_latitude_longitude",
            "north_pole_grid_longitude": 0.0,
            "grid_north_pole_longitude": 177.5,
            "grid_north_pole_latitude": 37.5,
            "longitude_of_prime_meridian": 0.0,
            "earth_radius": 6371229.0,
        }
        if extended_grid_mapping:
            expected["crs_wkt"] = (
                'GEOGCRS["unnamed",BASEGEOGCRS["unknown",DATUM["unknown",'
                'ELLIPSOID["unknown",6371229,0,LENGTHUNIT["metre",1,ID['
                '"EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",'
                '0.0174532925199433],ID["EPSG",8901]]],DERIVINGCONVERSION['
                '"unknown",METHOD["PROJ ob_tran o_proj=latlon"],PARAMETER['
                '"o_lon_p",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG"'
                ',9122]]],PARAMETER["o_lat_p",37.5,ANGLEUNIT["degree",'
                '0.0174532925199433,ID["EPSG",9122]]],PARAMETER["lon_0",357.5'
                ',ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]],CS['
                'ellipsoidal,2],AXIS["longitude",east,ORDER[1],ANGLEUNIT['
                '"degree",0.0174532925199433,ID["EPSG",9122]]],AXIS["latitude"'
                ',north,ORDER[2],ANGLEUNIT["degree",0.0174532925199433,ID['
                '"EPSG",9122]]]]'
            )

        self._test(coord_system, expected)

    def test_spherical_geog_cs(self, extended_grid_mapping):
        self._extended_grid_mapping = extended_grid_mapping
        coord_system = GeogCS(6371229.0)
        expected = {
            "grid_mapping_name": b"latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "earth_radius": 6371229.0,
        }
        if extended_grid_mapping:
            expected["crs_wkt"] = (
                'GEOGCRS["unknown",DATUM["unknown",ELLIPSOID["unknown",6371229'
                ',0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenwich"'
                ',0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8901]],CS'
                '[ellipsoidal,2],AXIS["longitude",east,ORDER[1],ANGLEUNIT['
                '"degree",0.0174532925199433,ID["EPSG",9122]]],AXIS["latitude"'
                ',north,ORDER[2],ANGLEUNIT["degree",0.0174532925199433,ID['
                '"EPSG",9122]]]]'
            )
        self._test(coord_system, expected)

    def test_elliptic_geog_cs(self, extended_grid_mapping):
        self._extended_grid_mapping = extended_grid_mapping
        coord_system = GeogCS(637, 600)
        expected = {
            "grid_mapping_name": b"latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_minor_axis": 600.0,
            "semi_major_axis": 637.0,
        }
        if extended_grid_mapping:
            expected["crs_wkt"] = (
                'GEOGCRS["unknown",DATUM["unknown",ELLIPSOID["unknown",637,'
                '17.2162162162162,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],'
                'PRIMEM["Reference meridian",0,ANGLEUNIT["degree",'
                '0.0174532925199433,ID["EPSG",9122]]],CS[ellipsoidal,2],AXIS'
                '["longitude",east,ORDER[1],ANGLEUNIT["degree",'
                '0.0174532925199433,ID["EPSG",9122]]],AXIS["latitude",north,'
                'ORDER[2],ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",'
                "9122]]]]"
            )
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
            "crs_wkt": (
                'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["unknown",'
                'ELLIPSOID["unknown",6371000,0,LENGTHUNIT["metre",1,ID["EPSG"'
                ',9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",'
                '0.0174532925199433],ID["EPSG",8901]]],CONVERSION["unknown",'
                'METHOD["Lambert Conic Conformal (2SP)",ID["EPSG",9802]],'
                'PARAMETER["Latitude of false origin",44,ANGLEUNIT["degree",'
                '0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitude of '
                'false origin",2,ANGLEUNIT["degree",0.0174532925199433],ID['
                '"EPSG",8822]],PARAMETER["Latitude of 1st standard parallel",'
                '38,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],'
                'PARAMETER["Latitude of 2nd standard parallel",50,ANGLEUNIT'
                '["degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER['
                '"Easting at false origin",-2,LENGTHUNIT["metre",1],ID["EPSG",'
                '8826]],PARAMETER["Northing at false origin",-5,LENGTHUNIT['
                '"metre",1],ID["EPSG",8827]]],CS[Cartesian,2],AXIS["(E)",east,'
                'ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",'
                'north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]'
            ),
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
            "crs_wkt": (
                'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["unknown",ELLIP'
                'SOID["unknown",6377563.396,299.324961266495,LENGTHUNIT["metre'
                '",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree'
                '",0.0174532925199433],ID["EPSG",8901]]],CONVERSION["unknown",'
                'METHOD["Lambert Azimuthal Equal Area",ID["EPSG",9820]],PARAME'
                'TER["Latitude of natural origin",52,ANGLEUNIT["degree",0.0174'
                '532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natura'
                'l origin",10,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG"'
                ',8802]],PARAMETER["False easting",100,LENGTHUNIT["metre",1],I'
                'D["EPSG",8806]],PARAMETER["False northing",200,LENGTHUNIT["me'
                'tre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["(E)",east,ORD'
                'ER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north'
                ',ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]'
            ),
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
            "crs_wkt": (
                'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["unknown",ELLIP'
                'SOID["unknown",6377563.396,299.324961266495,LENGTHUNIT["metre'
                '",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree'
                '",0.0174532925199433],ID["EPSG",8901]]],CONVERSION["unknown",'
                'METHOD["Albers Equal Area",ID["EPSG",9822]],PARAMETER["Latitu'
                'de of false origin",52,ANGLEUNIT["degree",0.0174532925199433]'
                ',ID["EPSG",8821]],PARAMETER["Longitude of false origin",10,AN'
                'GLEUNIT["degree",0.0174532925199433],ID["EPSG",8822]],PARAMET'
                'ER["Latitude of 1st standard parallel",38,ANGLEUNIT["degree",'
                '0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2'
                'nd standard parallel",50,ANGLEUNIT["degree",0.017453292519943'
                '3],ID["EPSG",8824]],PARAMETER["Easting at false origin",100,L'
                'ENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at '
                'false origin",200,LENGTHUNIT["metre",1],ID["EPSG",8827]]],CS['
                'Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID'
                '["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",'
                '1,ID["EPSG",9001]]]]'
            ),
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
            "crs_wkt": (
                'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["unknown",ELLIP'
                'SOID["unknown",6377563.396,299.324961266495,LENGTHUNIT["metre'
                '",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree'
                '",0.0174532925199433],ID["EPSG",8901]]],CONVERSION["unknown",'
                'METHOD["Vertical Perspective",ID["EPSG",9838]],PARAMETER["Lat'
                'itude of topocentric origin",1,ANGLEUNIT["degree",0.017453292'
                '5199433],ID["EPSG",8834]],PARAMETER["Longitude of topocentric'
                ' origin",2,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8'
                '835]],PARAMETER["Ellipsoidal height of topocentric origin",0,'
                'LENGTHUNIT["metre",1],ID["EPSG",8836]],PARAMETER["Viewpoint h'
                'eight",2000000,LENGTHUNIT["metre",1],ID["EPSG",8840]],PARAMET'
                'ER["False easting",100,LENGTHUNIT["metre",1],ID["EPSG",8806]]'
                ',PARAMETER["False northing",200,LENGTHUNIT["metre",1],ID["EPS'
                'G",8807]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNI'
                'T["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGT'
                'HUNIT["metre",1,ID["EPSG",9001]]]]'
            ),
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
            "crs_wkt": (
                'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["unknown",ELLIP'
                'SOID["unknown",6377563.396,299.324961266495,LENGTHUNIT["metre'
                '",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree'
                '",0.0174532925199433],ID["EPSG",8901]]],CONVERSION["unknown",'
                'METHOD["Geostationary Satellite (Sweep X)"],PARAMETER["Longit'
                'ude of natural origin",2,ANGLEUNIT["degree",0.017453292519943'
                '3],ID["EPSG",8802]],PARAMETER["Satellite Height",2000000,LENG'
                'THUNIT["metre",1,ID["EPSG",9001]]],PARAMETER["False easting",'
                '100,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False n'
                'orthing",200,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Carte'
                'sian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPS'
                'G",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID['
                '"EPSG",9001]]],REMARK["PROJ CRS string: +proj=geos +a=6377563'
                ".396 +b=6356256.909 +lon_0=2.0 +lat_0=0.0 +h=2000000.0 +x_0=1"
                '00.0 +y_0=200.0 +units=m +sweep=x +no_defs"]]'
            ),
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

    def test_oblique_cs(self):
        # Some none-default settings to confirm all parameters are being
        #  handled.

        wkt_template = (
            'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["unknown",ELLIP'
            'SOID["unknown",1,0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PR'
            'IMEM["Reference meridian",0,ANGLEUNIT["degree",0.017453292519'
            '9433,ID["EPSG",9122]]]],CONVERSION["unknown",METHOD["Hotine O'
            'blique Mercator (variant B)",ID["EPSG",9815]],PARAMETER["Lati'
            'tude of projection centre",89.9,ANGLEUNIT["degree",0.01745329'
            '25199433],ID["EPSG",8811]],PARAMETER["Longitude of projection'
            ' centre",45,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",'
            '8812]],PARAMETER["Azimuth at projection centre",{angle},ANGLEUNIT['
            '"degree",0.0174532925199433],ID["EPSG",8813]],PARAMETER["Angl'
            'e from Rectified to Skew Grid",{angle},ANGLEUNIT["degree",0.017453'
            '2925199433],ID["EPSG",8814]],PARAMETER["Scale factor at proje'
            'ction centre",0.939692620786,SCALEUNIT["unity",1],ID["EPSG",8'
            '815]],PARAMETER["Easting at projection centre",1000000,LENGTH'
            'UNIT["metre",1],ID["EPSG",8816]],PARAMETER["Northing at proje'
            'ction centre",-2000000,LENGTHUNIT["metre",1],ID["EPSG",8817]]'
            '],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre"'
            ',1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["me'
            'tre",1,ID["EPSG",9001]]]]'
        )

        kwargs_rotated = dict(
            latitude_of_projection_origin=89.9,
            longitude_of_projection_origin=45.0,
            false_easting=1000000.0,
            false_northing=-2000000.0,
            scale_factor_at_projection_origin=0.939692620786,
            ellipsoid=GeogCS(1),
        )

        # Same as rotated, but with azimuth too.
        oblique_azimuth = dict(azimuth_of_central_line=45.0)
        kwargs_oblique = dict(kwargs_rotated, **oblique_azimuth)

        expected_rotated = dict(
            # Automatically converted to oblique_mercator in line with CF 1.11 .
            grid_mapping_name=b"oblique_mercator",
            # Azimuth and crs_wkt should be automatically populated.
            azimuth_of_central_line=90.0,
            crs_wkt=wkt_template.format(angle="89.999"),
            **kwargs_rotated,
        )
        # Convert the ellipsoid
        expected_rotated.update(
            dict(
                earth_radius=expected_rotated.pop("ellipsoid").semi_major_axis,
                longitude_of_prime_meridian=0.0,
            )
        )

        # Same as rotated, but different azimuth.
        expected_oblique = dict(expected_rotated, **oblique_azimuth)
        expected_oblique["crs_wkt"] = wkt_template.format(angle="45")

        oblique = ObliqueMercator(**kwargs_oblique)
        rotated = RotatedMercator(**kwargs_rotated)

        for coord_system, expected in [
            (oblique, expected_oblique),
            (rotated, expected_rotated),
        ]:
            self._test(coord_system, expected)


@pytest.fixture(
    params=[
        pytest.param(True, id="extended_grid_mapping"),
        pytest.param(False, id="no_extended_grid_mapping"),
    ]
)
def extended_grid_mapping(request):
    """Fixture for enabling/disabling extended grid mapping."""
    return request.param


if __name__ == "__main__":
    tests.main()
