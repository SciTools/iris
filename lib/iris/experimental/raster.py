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
Experimental module for importing/exporting data from Iris cubes using the
GDAL library.

See also: `GDAL - Geospatial Data Abstraction Library <http://www.gdal.org>`_.

TODO: If this module graduates from experimental the (optional) gdal
      dependency should be added to INSTALL

"""
import numpy as np
import iris.fileformats

try:
    from osgeo import gdal
except:
    import gdal


def _get_geo_transform(px_width, px_height, bbox_left, bbox_bottom):
    """
    Get the affine transformation coefficients as a tuple of six values
    that are compatible with GDAL's SetGeoTransform()

    Args:
        * px_width (float): Width of each pixel
        * px_height (float): Height of each pixel
        * bbox_left (float): Left-most value of the bounding box
        * bbox_bottom (float): Bottom-most value of the bounding box

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
    by regularly 1d coordinates.

    Args:
        * cube (Cube): The regularly gridded cube slice to be exported.
        * fname (string): Output file name.
        * ftype (string): Export file type.

        .. note::

            See http://www.gdal.org/formats_list.html for supported
            output formats

    """
    if cube.ndim != 2:
        raise ValueError("Cube must have two dimensions")

    coord_y = cube.coord(axis="Y")
    coord_x = cube.coord(axis="X")
    px_width = iris.fileformats.rules.regular_step(coord_x)
    px_height = iris.fileformats.rules.regular_step(coord_y)

    x_points = coord_x.points
    if (coord_x.name().find('longitude') >= 0):
        if np.max(coord_x.points) > 180:
            # Attempt to correct for non-geographic longitude ranges.
            x_points -= 180

    bbox_bottom = np.min(coord_y.points) - (0.5 * px_height)
    bbox_left = np.min(x_points) - (0.5 * px_width)

    padfTransform = _get_geo_transform(px_width, px_height,
                                       bbox_left, bbox_bottom)

    # TODO: If Y (and/or X) can be monotonically decreasing, e.g. causing
    # data[0][0] to be the top-left corner, instead of the bottom-left,
    # then the data must be inverted and/or mirrored before being written.
    #data = cube.data.copy()
    #data = np.flipud(data)

    # TODO: Select data type based on type of cube.data
    # e.g. gdal.GDT_Byte for integer range 0 - 255
    driver = gdal.GetDriverByName(ftype)
    ds = driver.Create(fname, len(coord_x.points), len(coord_y.points),
                       1, gdal.GDT_Float32)

    ds.SetGeoTransform(padfTransform)
    ds.GetRasterBand(1).WriteArray(cube.data)

    # TODO: Specify the projection of the output raster
    # TODO: For test data with bbox_bottom = -90, GIS shows bbox_bottom
    #       is being written as -90.11515
