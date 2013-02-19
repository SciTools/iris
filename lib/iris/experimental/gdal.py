# (C) British Crown Copyright 2010 - 2013, Met Office
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
Experimental code for interfacing Iris with the GDAL/OGR library

"""
import numpy as np
from osgeo import gdal

import iris.fileformats


def _raster_checker(cube):
    """
    Determines whether a cube is suitable for saving as a GDAL raster

    A suitable cube will be a single 2D slice that is described
    by DimCoords named 'latitude' and 'longitude' (i.e. is unprojected)
    and will have regularly gridded data.

    Currently, the coordinates must not be circular, and longitude values
    must be in the range -180 <= lon <= 180.

    Args:
        * cube (Cube): The cube to be tested.

        Returns:
            Boolean.
    """

    if cube.ndim != 2:
        raise Exception("Cube must have two dimensions")

    Y = cube.coord('latitude')
    X = cube.coord('longitude')

    if not (isinstance(X, iris.coords.DimCoord) and
            isinstance(Y, iris.coords.DimCoord)):
        raise Exception("Coordinates must be of type iris.coords.DimCoord")

    if hasattr(X, 'circular') and X.circular:
        raise Exception("Longitude coordinate must not be circular")
    if hasattr(Y, 'circular') and Y.circular:
        raise Exception("Latitude coordinate must not be circular")

    if not (iris.fileformats.rules.is_regular(X) and
            iris.fileformats.rules.is_regular(Y)):
        raise Exception("Both coordinates must be regular")

    if np.min(X.points) < -180 or np.max(X.points) > 180:
        raise Exception("Latitude coordinate values must be in the range"
                        "-180 <= Latitude <= 180")


def _get_geo_transform(px_width, px_height, bbox_left, bbox_bottom):
    """
    Get the affine transformation coefficients as a tuple of six values
    that are compatible with GDAL's SetGeoTransform()

    Args:
        * px_width (float): Width of each pixel
        * px_height (float): Height of each pixel
        * bbox_left (float): Left-most value of the bounding box
        * bbox_bottom (float): bottom-most value of the bounding box

    .. note::
        More details on the returned tuple are available at:
        http://www.gdal.org/classGDALDataset.html#af9593cc241e7d140f5f3c4798a43a668
    """

    return (bbox_left, px_width, 0.0, bbox_bottom, 0.0, px_height)


def export(cube, fname, ftype="GTiff"):
    """
    Determines whether a cube is suitable for saving as a raster with
    gdal.GetDriverByName.Create()

    A suitable cube will be a single 2D slice that is described
    by DimCoords named 'latitude' and 'longitude' and will have regularly
    gridded data.

    Currently, the coordinates must not be circular, and longitude values
    must be in the range -180 <= lon <= 180.

    Args:
        * cube (Cube): The regularly gridded cube slice to be exported.
        * fname (string): Output file name.
        * ftype (string): Export file type, see

        Returns:
            Boolean.

        .. note::
            See http://www.gdal.org/formats_list.html for supported
            output formats
    """

    _raster_checker(cube)

    Ny, Nx = cube.data.shape
    lat = cube.coord('latitude')
    lon = cube.coord('longitude')

    px_width = iris.fileformats.rules.regular_step(lon)
    px_height = iris.fileformats.rules.regular_step(lat)

    if not lat.has_bounds():
        lat.guess_bounds()
    if not lon.has_bounds():
        lon.guess_bounds()

    bbox_bottom = min(np.concatenate(lat.bounds[[0, -1]]))
    bbox_left = min(np.concatenate(lon.bounds[[0, -1]]))

    padfTransform = _get_geo_transform(px_width, px_height,
                                       bbox_left, bbox_bottom)

    # TODO: If latitude and/or longitude can be monotonically decreasing
    # (e.g. data[0][0] as the top-left corner, instead of the bottom-left)
    # then the data must be inverted and/or mirrored before being written.
    #data = cube.data.copy()
    #data = np.flipud(data)

    # TODO: Select data type based on type of cube.data
    # e.g. gdal.GDT_Byte for integer range 0 - 255

    driver = gdal.GetDriverByName(ftype)
    ds = driver.Create(fname, Nx, Ny, 1, gdal.GDT_Float32)

    ds.SetGeoTransform(padfTransform)
    ds.GetRasterBand(1).WriteArray(cube.data)

    # The GDAL gotchas suggest explicitly deallocating ds. Is this necessary
    # when it is about to go out of scope?
    # http://trac.osgeo.org/gdal/wiki/PythonGotchas
    del ds
