# (C) British Crown Copyright 2013, Met Office
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
"""
Experimental module for importing/exporting raster data from Iris cubes using
the GDAL library.

See also: `GDAL - Geospatial Data Abstraction Library <http://www.gdal.org>`_.

TODO: If this module graduates from experimental the (optional) GDAL
      dependency should be added to INSTALL

"""
import numpy as np
from osgeo import gdal

import iris


def _gdal_write_array(cube_data, padf_transform, fname, ftype):
    """
    Use GDAL WriteArray to export cube_data as a 32-bit raster image.
    Requires the array data to be of the form: North-at-top
    and West-on-left.

    Args:
        * cube_data (numpy.ndarray): 2d array of values to export
        * padf_transform (tuple): coefficients for affine transformation
        * fname (string): Output file name.
        * ftype (string): Export file type.
        * dtype (numpy.dtype): Specify the data type to be written.
            By default a suitable type will be determined from cube_data.

    .. note::

        Projection information is currently not written to the output.

    """
    dtype = gdal.GDT_Float32

    driver = gdal.GetDriverByName(ftype)
    data = driver.Create(fname, cube_data.shape[1], cube_data.shape[0],
                         1, dtype)

    data.SetGeoTransform(padf_transform)
    band = data.GetRasterBand(1)

    if isinstance(cube_data, np.ma.core.MaskedArray):
        cube_data = cube_data.copy()
        cube_data[cube_data.mask] = cube_data.fill_value
        band.SetNoDataValue(cube_data.fill_value)

    band.WriteArray(cube_data)


def export_geotiff(cube, fname):
    """
    Writes cube data to raster file format as a PixelIsArea GeoTiff image.

    Args:
        * cube (Cube): The 2D regularly gridded cube slice to be exported.
                       The cube must have regular, contiguous bounds.
        * fname (string): Output file name.

    .. note::

        For more details on GeoTiff specification and PixelIsArea, see:
        http://www.remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2.2

    """
    if cube.ndim != 2:
        raise ValueError("The cube must be two dimensional.")

    coord_y = cube.coord(axis='Y')
    coord_x = cube.coord(axis='X')

    if (coord_y.bounds is None) or (coord_y.bounds is None):
        raise ValueError('Coordinates must have bounds, consider using '
                         'guess_bounds()')

    y_step = np.diff(coord_y.bounds[0])
    x_step = np.diff(coord_x.bounds[0])

    if (coord_y.coord_system and
            coord_y.coord_system != iris.coord_systems.GeogCS(6371229.0)):
        raise ValueError('Coordinates coord_system must be GeogCS or None')
    if coord_y.name() != 'latitude':
        raise ValueError('Y coordinate must have name "latitude"')
    if coord_x.name() != 'longitude':
        raise ValueError('X coordinate must have name "longitude"')
    if coord_y.units != 'degrees':
        raise ValueError('Y coordinate units must be "degrees"')
    if coord_x.units != 'degrees':
        raise ValueError('X coordinate units must be "degree"')
    if not (coord_y.is_contiguous() and coord_x.is_contiguous()):
        raise ValueError('Coordinate bounds must be contiguous')
    if not np.all(np.diff(coord_x.bounds) == x_step):
        raise ValueError('X coordinate bounds must be regularly spaced')
    if not np.all(np.diff(coord_y.bounds) == y_step):
        raise ValueError('Y coordinate bounds must be regularly spaced')
    if coord_x.points[0] > coord_x.points[-1]:
        raise ValueError('Longitude values must be monotonically increasing')

    bbox_top = np.max(coord_y.bounds)
    bbox_left = np.min(coord_x.bounds)

    padf_transform = (bbox_left, x_step, 0.0, bbox_top, 0.0, y_step)
    data = cube.data

    if coord_y.points[0] < coord_y.points[-1]:
        # Flip the data so North is at the top
        data = data[::-1, :]

    _gdal_write_array(data, padf_transform, fname, 'GTiff')
