# (C) British Crown Copyright 2013 - 2015, Met Office
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
"""Unit tests for the `iris.experimental.raster.export_geotiff` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
try:
    from osgeo import gdal
    from iris.experimental.raster import export_geotiff
except ImportError as err:
    if err.message.startswith('No module named'):
        gdal = None
    else:
        raise

from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube


@tests.skip_gdal
class TestDtypeAndValues(tests.IrisTest):
    def _cube(self, dtype):
        data = np.arange(12).reshape(3, 4).astype(dtype) + 20
        cube = Cube(data, 'air_pressure_anomaly')
        coord = DimCoord(np.arange(3), 'latitude', units='degrees')
        coord.guess_bounds()
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(np.arange(4), 'longitude', units='degrees')
        coord.guess_bounds()
        cube.add_dim_coord(coord, 1)
        return cube

    def _check_dtype(self, dtype, gdal_dtype):
        cube = self._cube(dtype)
        with self.temp_filename('.tif') as temp_filename:
            export_geotiff(cube, temp_filename)
            dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
            band = dataset.GetRasterBand(1)
            self.assertEqual(band.DataType, gdal_dtype)
            self.assertEqual(band.ComputeRasterMinMax(1), (20, 31))

    def test_int16(self):
        self._check_dtype('<i2', gdal.GDT_Int16)

    def test_int16_big_endian(self):
        self._check_dtype('>i2', gdal.GDT_Int16)

    def test_int32(self):
        self._check_dtype('<i4', gdal.GDT_Int32)

    def test_int32_big_endian(self):
        self._check_dtype('>i4', gdal.GDT_Int32)

    def test_uint8(self):
        self._check_dtype('u1', gdal.GDT_Byte)

    def test_uint16(self):
        self._check_dtype('<u2', gdal.GDT_UInt16)

    def test_uint16_big_endian(self):
        self._check_dtype('>u2', gdal.GDT_UInt16)

    def test_uint32(self):
        self._check_dtype('<u4', gdal.GDT_UInt32)

    def test_uint32_big_endian(self):
        self._check_dtype('>u4', gdal.GDT_UInt32)

    def test_float32(self):
        self._check_dtype('<f4', gdal.GDT_Float32)

    def test_float32_big_endian(self):
        self._check_dtype('>f4', gdal.GDT_Float32)

    def test_float64(self):
        self._check_dtype('<f8', gdal.GDT_Float64)

    def test_float64_big_endian(self):
        self._check_dtype('>f8', gdal.GDT_Float64)

    def test_invalid(self):
        cube = self._cube('i1')
        with self.assertRaises(ValueError):
            with self.temp_filename('.tif') as temp_filename:
                export_geotiff(cube, temp_filename)


@tests.skip_gdal
class TestProjection(tests.IrisTest):
    def _cube(self, ellipsoid=None):
        data = np.arange(12).reshape(3, 4).astype('u1')
        cube = Cube(data, 'air_pressure_anomaly')
        coord = DimCoord(np.arange(3), 'latitude', units='degrees',
                         coord_system=ellipsoid)
        coord.guess_bounds()
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(np.arange(4), 'longitude', units='degrees',
                         coord_system=ellipsoid)
        coord.guess_bounds()
        cube.add_dim_coord(coord, 1)
        return cube

    def test_no_ellipsoid(self):
        cube = self._cube()
        with self.temp_filename('.tif') as temp_filename:
            export_geotiff(cube, temp_filename)
            dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
            self.assertEqual(dataset.GetProjection(), '')

    def test_sphere(self):
        cube = self._cube(GeogCS(6377000))
        with self.temp_filename('.tif') as temp_filename:
            export_geotiff(cube, temp_filename)
            dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
            self.assertEqual(
                dataset.GetProjection(),
                'GEOGCS["unnamed ellipse",DATUM["unknown",'
                'SPHEROID["unnamed",6377000,0]],PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433]]')

    def test_ellipsoid(self):
        cube = self._cube(GeogCS(6377000, 6360000))
        with self.temp_filename('.tif') as temp_filename:
            export_geotiff(cube, temp_filename)
            dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
            self.assertEqual(
                dataset.GetProjection(),
                'GEOGCS["unnamed ellipse",DATUM["unknown",'
                'SPHEROID["unnamed",6377000,375.117647058816]],'
                'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')


@tests.skip_gdal
class TestGeoTransform(tests.IrisTest):
    def test_(self):
        data = np.arange(12).reshape(3, 4).astype(np.uint8)
        cube = Cube(data, 'air_pressure_anomaly')
        coord = DimCoord([30, 40, 50], 'latitude', units='degrees')
        coord.guess_bounds()
        cube.add_dim_coord(coord, 0)
        coord = DimCoord([-10, -5, 0, 5], 'longitude', units='degrees')
        coord.guess_bounds()
        cube.add_dim_coord(coord, 1)
        with self.temp_filename('.tif') as temp_filename:
            export_geotiff(cube, temp_filename)
            dataset = gdal.Open(temp_filename, gdal.GA_ReadOnly)
            self.assertEqual(dataset.GetGeoTransform(),
                             (-12.5, 5, 0, 55, 0, -10))


if __name__ == "__main__":
    tests.main()
