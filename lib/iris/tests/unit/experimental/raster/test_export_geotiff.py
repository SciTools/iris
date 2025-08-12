# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.experimental.raster.export_geotiff` function."""

import re

import numpy as np
import pytest

try:
    from osgeo import gdal

    from iris.experimental.raster import export_geotiff
except ImportError:
    gdal = None

from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import _shared_utils


@_shared_utils.skip_gdal
class TestDtypeAndValues:
    def _cube(self, dtype):
        data = np.arange(12).reshape(3, 4).astype(dtype) + 20
        cube = Cube(data, "air_pressure_anomaly")
        coord = DimCoord(np.arange(3), "latitude", units="degrees")
        coord.guess_bounds()
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(np.arange(4), "longitude", units="degrees")
        coord.guess_bounds()
        cube.add_dim_coord(coord, 1)
        return cube

    def _check_dtype(self, dtype, gdal_dtype, tmp_path):
        cube = self._cube(dtype)
        temp_filename = str(tmp_path / "tmp.tif")
        export_geotiff(cube, temp_filename)
        dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
        band = dataset.GetRasterBand(1)
        assert band.DataType == gdal_dtype
        assert band.ComputeRasterMinMax(1) == (20, 31)

    def test_int16(self, tmp_path):
        self._check_dtype("<i2", gdal.GDT_Int16, tmp_path)

    def test_int16_big_endian(self, tmp_path):
        self._check_dtype(">i2", gdal.GDT_Int16, tmp_path)

    def test_int32(self, tmp_path):
        self._check_dtype("<i4", gdal.GDT_Int32, tmp_path)

    def test_int32_big_endian(self, tmp_path):
        self._check_dtype(">i4", gdal.GDT_Int32, tmp_path)

    def test_uint8(self, tmp_path):
        self._check_dtype("u1", gdal.GDT_Byte, tmp_path)

    def test_uint16(self, tmp_path):
        self._check_dtype("<u2", gdal.GDT_UInt16, tmp_path)

    def test_uint16_big_endian(self, tmp_path):
        self._check_dtype(">u2", gdal.GDT_UInt16, tmp_path)

    def test_uint32(self, tmp_path):
        self._check_dtype("<u4", gdal.GDT_UInt32, tmp_path)

    def test_uint32_big_endian(self, tmp_path):
        self._check_dtype(">u4", gdal.GDT_UInt32, tmp_path)

    def test_float32(self, tmp_path):
        self._check_dtype("<f4", gdal.GDT_Float32, tmp_path)

    def test_float32_big_endian(self, tmp_path):
        self._check_dtype(">f4", gdal.GDT_Float32, tmp_path)

    def test_float64(self, tmp_path):
        self._check_dtype("<f8", gdal.GDT_Float64, tmp_path)

    def test_float64_big_endian(self, tmp_path):
        self._check_dtype(">f8", gdal.GDT_Float64, tmp_path)

    def test_invalid(self, tmp_path):
        cube = self._cube("i1")
        temp_filename = str(tmp_path / "tmp.tif")
        with pytest.raises(ValueError, match="Unsupported data type"):
            export_geotiff(cube, temp_filename)


@_shared_utils.skip_gdal
class TestProjection:
    def _cube(self, ellipsoid=None):
        data = np.arange(12).reshape(3, 4).astype("u1")
        cube = Cube(data, "air_pressure_anomaly")
        coord = DimCoord(
            np.arange(3), "latitude", units="degrees", coord_system=ellipsoid
        )
        coord.guess_bounds()
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(
            np.arange(4), "longitude", units="degrees", coord_system=ellipsoid
        )
        coord.guess_bounds()
        cube.add_dim_coord(coord, 1)
        return cube

    def test_no_ellipsoid(self, tmp_path):
        cube = self._cube()
        temp_filename = str(tmp_path / "tmp.tif")
        export_geotiff(cube, temp_filename)
        dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
        assert dataset.GetProjection() == ""

    def test_sphere(self, tmp_path):
        cube = self._cube(GeogCS(6377000))
        temp_filename = str(tmp_path / "tmp.tif")
        export_geotiff(cube, temp_filename)
        dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
        projection_string = dataset.GetProjection()
        # String has embedded floating point values,
        # Test with values to N decimal places, using a regular expression.
        re_pattern = (
            r'GEOGCS\["unknown",DATUM\["unknown",'
            r'SPHEROID\["unknown",637....,0\]\],PRIMEM\["Greenwich",0\],'
            r'UNIT\["degree",0.01745[0-9]*,AUTHORITY\["EPSG","9122"\]\],'
            r'AXIS\["Latitude",NORTH\],AXIS\["Longitude",EAST\]\]'
        )
        re_exp = re.compile(re_pattern)
        assert re_exp.match(projection_string), (
            "projection string {!r} does not match {!r}".format(
                projection_string, re_pattern
            )
            is not None
        )

    def test_ellipsoid(self, tmp_path):
        cube = self._cube(GeogCS(6377000, 6360000))
        temp_filename = str(tmp_path / "tmp.tif")
        export_geotiff(cube, temp_filename)
        dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
        projection_string = dataset.GetProjection()
        # String has embedded floating point values,
        # Test with values to N decimal places, using a regular expression.
        re_pattern = (
            r'GEOGCS\["unknown",DATUM\["unknown",'
            r'SPHEROID\["unknown",637....,375.117[0-9]*\]\],'
            r'PRIMEM\["Greenwich",0\],UNIT\["degree",0.01745[0-9]*,'
            r'AUTHORITY\["EPSG","9122"\]\],AXIS\["Latitude",NORTH\],'
            r'AXIS\["Longitude",EAST\]\]'
        )
        re_exp = re.compile(re_pattern)
        assert re_exp.match(projection_string), (
            "projection string {!r} does not match {!r}".format(
                projection_string, re_pattern
            )
            is not None
        )


@_shared_utils.skip_gdal
class TestGeoTransform:
    def test_(self, tmp_path):
        data = np.arange(12).reshape(3, 4).astype(np.uint8)
        cube = Cube(data, "air_pressure_anomaly")
        coord = DimCoord([30, 40, 50], "latitude", units="degrees")
        coord.guess_bounds()
        cube.add_dim_coord(coord, 0)
        coord = DimCoord([-10, -5, 0, 5], "longitude", units="degrees")
        coord.guess_bounds()
        cube.add_dim_coord(coord, 1)
        temp_filename = str(tmp_path / "tmp.tif")
        export_geotiff(cube, temp_filename)
        dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
        assert dataset.GetGeoTransform() == (-12.5, 5, 0, 55, 0, -10)
