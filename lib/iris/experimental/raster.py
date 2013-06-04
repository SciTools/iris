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
import iris.coord_systems as ics
import iris.unit


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
        band.SetNoDataValue(float(cube_data.fill_value))

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

    coord_x = cube.coord(axis='X', dim_coords=True)
    coord_y = cube.coord(axis='Y', dim_coords=True)

    if coord_x.bounds is None or coord_y.bounds is None:
        raise ValueError('Coordinates must have bounds, consider using '
                         'guess_bounds()')
    if coord_x.bounds.shape[-1] != 2:
        raise ValueError('Coordinate {!r} x-bounds must have shape '
                         '(N, 2).'.format(coord_x.name()))
    if coord_y.bounds.shape[-1] != 2:
        raise ValueError('Coordinate {!r} y-bounds must have shape '
                         '(N, 2).'.format(coord_y.name()))

    xy_step = []
    for coord in [coord_x, coord_y]:
        name = coord.name()
        if coord.coord_system is not None and \
                not isinstance(coord.coord_system, ics.GeogCS):
            msg = 'Coordinate {!r} must be a geographic (ellipsoidal) ' \
                'coordinate system.'.format(name)
            raise ValueError(msg)
        if not (coord.units == iris.unit.Unit('degrees') or
                coord.units.is_convertible('meters')):
            raise ValueError('Coordinate {!r} units must be either degrees or '
                             'convertible to meters.'.format(name))
        if not coord.is_contiguous():
            raise ValueError('Coordinate {!r} bounds must be '
                             'contiguous.'.format(name))
        xy_step.append(np.diff(coord.bounds[0]))
        size = coord.points.size
        expected = np.array([xy_step[-1]] * size,
                            dtype=coord.bounds.dtype).reshape(size, 1)
        msg = 'Coordinate {!r} bounds must be regularly spaced.'.format(name)
        np.testing.assert_array_almost_equal(np.diff(coord.bounds), expected,
                                             err_msg=msg)

    if coord_x.points[0] > coord_x.points[-1]:
        raise ValueError('Coordinate {!r} x-points must be monotonically'
                         'increasing.'.format(name))

    data = cube.data

    if xy_step[1] > 0:
        # Flip the data so North is at the top.
        data = data[::-1, :]
        xy_step[1] *= -1

    bbox_top = np.max(coord_y.bounds)
    bbox_left = np.min(coord_x.bounds)
    padf_transform = (bbox_left, xy_step[0], 0.0, bbox_top, 0.0, xy_step[1])
    _gdal_write_array(data, padf_transform, fname, 'GTiff')
