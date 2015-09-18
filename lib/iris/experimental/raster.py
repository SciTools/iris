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
"""
Experimental module for importing/exporting raster data from Iris cubes using
the GDAL library.

See also: `GDAL - Geospatial Data Abstraction Library <http://www.gdal.org>`_.

TODO: If this module graduates from experimental the (optional) GDAL
      dependency should be added to INSTALL

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import warnings

from gdalconst import GA_ReadOnly
import numpy as np
from osgeo import gdal, osr

import iris
import iris.coord_systems
import iris.unit


_GDAL_DATATYPES = {
    'i2': gdal.GDT_Int16,
    'i4': gdal.GDT_Int32,
    'u1': gdal.GDT_Byte,
    'u2': gdal.GDT_UInt16,
    'u4': gdal.GDT_UInt32,
    'f4': gdal.GDT_Float32,
    'f8': gdal.GDT_Float64,
}


def _gdal_write_array(x_min, x_step, y_max, y_step, coord_system, data, fname,
                      ftype):
    """
    Use GDAL WriteArray to export data as a 32-bit raster image.
    Requires the array data to be of the form: North-at-top
    and West-on-left.

    Args:
        * x_min: Minimum X coordinate bounds value.
        * x_step: Change in X coordinate per cell.
        * y_max: Maximum Y coordinate bounds value.
        * y_step: Change in Y coordinate per cell.
        * coord_system (iris.coord_systems.CoordSystem):
            Coordinate system for X and Y.
        * data (numpy.ndarray): 2d array of values to export
        * fname (string): Output file name.
        * ftype (string): Export file type.

    .. note::

        Projection information is currently not written to the output.

    """
    byte_order = data.dtype.str[0]
    format = data.dtype.str[1:]
    dtype = _GDAL_DATATYPES.get(format)
    if dtype is None:
        raise ValueError('Unsupported data type: {}'.format(data.dtype))

    driver = gdal.GetDriverByName(ftype)
    gdal_dataset = driver.Create(fname, data.shape[1], data.shape[0],
                                 1, dtype)

    # Where possible, set the projection.
    if coord_system is not None:
        srs = osr.SpatialReference()
        proj4_defn = coord_system.as_cartopy_crs().proj4_init
        # GDAL can't cope with "+proj=lonlat" which Cartopy produces.
        proj4_defn = proj4_defn.replace('lonlat', 'longlat')
        if srs.ImportFromProj4(proj4_defn):
            msg = 'Unsupported coordinate system: {}'.format(coord_system)
            raise ValueError(msg)
        gdal_dataset.SetProjection(srs.ExportToWkt())

    # Set the affine transformation coefficients.
    padf_transform = (x_min, x_step, 0.0, y_max, 0.0, y_step)
    gdal_dataset.SetGeoTransform(padf_transform)

    band = gdal_dataset.GetRasterBand(1)
    if isinstance(data, np.ma.core.MaskedArray):
        data = data.copy()
        data[data.mask] = data.fill_value
        band.SetNoDataValue(float(data.fill_value))
    # GeoTIFF always needs little-endian data.
    if byte_order == '>':
        data = data.astype(data.dtype.newbyteorder('<'))
    band.WriteArray(data)


def import_raster(fname, header=None):
    """
    Imports raster images using gdal and constructs a cube.

    Parameters
    ----------
    fname : str
        Input file name.

    Returns
    -------
    cube : iris.cube.Cube
        A 2D regularly gridded cube.  The resulting cube has regular,
        contiguous bounds.

    .. note::

        Coordinate system information it not yet interpreted.
        Constrained raster import not yet supported.

    .. warning::

        Deferred loading not yet supported.

    """
    dataset = gdal.Open(fname, GA_ReadOnly)
    if dataset is None:
        raise IOError('gdal failed to open raster image')

    projection = dataset.GetProjection()
    if projection:
        warnings.warn('Currently the following projection information is '
                      'not interpreted: {}'.format(projection))

    # Get metadata applies to all raster bands
    transform = dataset.GetGeoTransform()
    origin_xy = (transform[0], transform[3])
    num_xy = (dataset.RasterXSize, dataset.RasterYSize)

    # This effectively indicates the bounds of the cells.
    pixel_width = (transform[1], transform[5])
    num_raster = dataset.RasterCount

    # Position of North 0, 0 is north-up
    rotation = (transform[2], transform[4])
    if rotation[0] != 0 or rotation[1] != 0:
        msg = ('Rotation not supported: ({}, {})'.format(rotation[0],
                                                         rotation[1]))
        raise ValueError(msg)

    if num_raster > 1:
        warnings.warn('Multiple raster band support ({}) is highly '
                      'experimental, use at your own risk'.format(num_raster))
    elif num_raster == 0:
        return None

    # Calculate coordinate points
    if transform is not None:
        points_origin = (origin_xy[0] + pixel_width[0]/2,
                         origin_xy[1] + pixel_width[1]/2)
        points_x = np.arange(points_origin[0], points_origin[0] +
                             pixel_width[0] * num_xy[0], pixel_width[0])
        points_y = np.arange(points_origin[1], points_origin[1] +
                             pixel_width[1] * num_xy[1], pixel_width[1])[::-1]
        x = iris.coords.DimCoord(points_x,
                                 standard_name='projection_x_coordinate')
        x.guess_bounds()
        y = iris.coords.DimCoord(points_y,
                                 standard_name='projection_y_coordinate')
        y.guess_bounds()

    # Load data for each raster band.
    cubes = iris.cube.CubeList()
    for iraster in range(num_raster):
        iband = dataset.GetRasterBand(iraster+1)
        # ReadAsArray(xoffset, yoffset, xsize, ysize)
        data = iband.ReadAsArray(0, 0, num_xy[0], num_xy[1])[::-1, :]
        mdi = iband.GetNoDataValue() or np.nan
        mask = data == mdi
        if mask.any():
            data = np.ma.masked_equal(data, mdi)
        cube = iris.cube.Cube(data)
        if transform is not None:
            cube.add_dim_coord(x, 1)
            cube.add_dim_coord(y, 0)
        cubes.append(cube)

    return cubes


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

    coord_x = cube.coord(axis='X', dim_coords=True)
    coord_y = cube.coord(axis='Y', dim_coords=True)

    if coord_x.bounds is None or coord_y.bounds is None:
        raise ValueError('Coordinates must have bounds, consider using '
                         'guess_bounds()')

    if coord_x is None or coord_y is None or \
       coord_x.coord_system != coord_y.coord_system:
        raise ValueError('The X and Y coordinates must share a CoordSystem.')

    xy_step = []
    for coord in [coord_x, coord_y]:
        name = coord.name()
        if coord.nbounds != 2:
            msg = 'Coordinate {!r} must have two bounds ' \
                'per point.'.format(name)
            raise ValueError(msg)
        if not (coord.units == iris.unit.Unit('degrees') or
                coord.units.is_convertible('meters')):
            raise ValueError('Coordinate {!r} units must be either degrees or '
                             'convertible to meters.'.format(name))
        if not coord.is_contiguous():
            raise ValueError('Coordinate {!r} bounds must be '
                             'contiguous.'.format(name))
        xy_step.append(np.diff(coord.bounds[0]))
        if not np.allclose(np.diff(coord.bounds), xy_step[-1]):
            msg = 'Coordinate {!r} bounds must be regularly ' \
                'spaced.'.format(name)
            raise ValueError(msg)

    if coord_x.points[0] > coord_x.points[-1]:
        raise ValueError('Coordinate {!r} x-points must be monotonically'
                         'increasing.'.format(name))

    data = cube.data

    # Make sure we have a YX data layout.
    if cube.coord_dims(coord_x) == 0:
        data = data.T

    x_step, y_step = xy_step
    if y_step > 0:
        # Flip the data so North is at the top.
        data = data[::-1, :]
        y_step *= -1

    coord_system = coord_x.coord_system
    x_bounds = coord_x.bounds
    if isinstance(coord_system, iris.coord_systems.GeogCS):
        big_indices = np.where(coord_x.points > 180)[0]
        n_big = len(big_indices)
        if n_big:
            data = np.roll(data, n_big, axis=1)
            x_bounds = x_bounds.copy()
            x_bounds[big_indices] -= 360

    x_min = np.min(x_bounds)
    y_max = np.max(coord_y.bounds)
    _gdal_write_array(x_min, x_step, y_max, y_step, coord_system, data, fname,
                      'GTiff')
