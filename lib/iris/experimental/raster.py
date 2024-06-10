# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Experimental module for importing/exporting raster data from Iris cubes using the GDAL library.

See also: `GDAL - Geospatial Data Abstraction Library <https://www.gdal.org>`_.

TODO: If this module graduates from experimental the (optional) GDAL
      dependency should be added to INSTALL

"""

import cf_units
import numpy as np
import numpy.ma as ma
from osgeo import gdal, osr

import iris
from iris._deprecation import warn_deprecated
import iris.coord_systems

wmsg = (
    "iris.experimental.raster is deprecated since version 3.2, and will be "
    "removed in a future release. If you make use of this functionality, "
    "please contact the Iris Developers to discuss how to retain it (which may "
    "involve reversing the deprecation)."
)
warn_deprecated(wmsg)

_GDAL_DATATYPES = {
    "i2": gdal.GDT_Int16,
    "i4": gdal.GDT_Int32,
    "u1": gdal.GDT_Byte,
    "u2": gdal.GDT_UInt16,
    "u4": gdal.GDT_UInt32,
    "f4": gdal.GDT_Float32,
    "f8": gdal.GDT_Float64,
}


def _gdal_write_array(x_min, x_step, y_max, y_step, coord_system, data, fname, ftype):
    """Use GDAL WriteArray to export data as a 32-bit raster image.

    Requires the array data to be of the form: North-at-top
    and West-on-left.

    Parameters
    ----------
    x_min :
        Minimum X coordinate bounds value.
    x_step :
        Change in X coordinate per cell.
    y_max :
        Maximum Y coordinate bounds value.
    y_step :
        Change in Y coordinate per cell.
    coord_system : iris.coord_systems.CoordSystem
        Coordinate system for X and Y.
    data : numpy.ndarray
        2d array of values to export.
    fname : str
        Output file name.
    ftype : str
        Export file type.

    Notes
    -----
    Projection information is currently not written to the output.

    """
    byte_order = data.dtype.str[0]
    format = data.dtype.str[1:]
    dtype = _GDAL_DATATYPES.get(format)
    if dtype is None:
        raise ValueError("Unsupported data type: {}".format(data.dtype))

    driver = gdal.GetDriverByName(ftype)
    gdal_dataset = driver.Create(fname, data.shape[1], data.shape[0], 1, dtype)

    # Where possible, set the projection.
    if coord_system is not None:
        srs = osr.SpatialReference()
        proj4_defn = coord_system.as_cartopy_crs().proj4_init
        # GDAL can't cope with "+proj=lonlat" which Cartopy produces.
        proj4_defn = proj4_defn.replace("lonlat", "longlat")
        if srs.ImportFromProj4(proj4_defn):
            msg = "Unsupported coordinate system: {}".format(coord_system)
            raise ValueError(msg)
        gdal_dataset.SetProjection(srs.ExportToWkt())

    # Set the affine transformation coefficients.
    padf_transform = (x_min, x_step, 0.0, y_max, 0.0, y_step)
    gdal_dataset.SetGeoTransform(padf_transform)

    band = gdal_dataset.GetRasterBand(1)
    if ma.isMaskedArray(data):
        data = data.copy()
        data[data.mask] = data.fill_value
        band.SetNoDataValue(float(data.fill_value))
    # GeoTIFF always needs little-endian data.
    if byte_order == ">":
        data = data.astype(data.dtype.newbyteorder("<"))
    band.WriteArray(data)


def export_geotiff(cube, fname):
    """Write cube data to raster file format as a PixelIsArea GeoTiff image.

    Parameters
    ----------
    cube : Cube
        The 2D regularly gridded cube slice to be exported.
        The cube must have regular, contiguous bounds.
    fname : str
        Output file name.

    Notes
    -----
    For more details on GeoTiff specification and PixelIsArea, see:
    https://www.remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2.2

    .. deprecated:: 3.2.0

        This method is scheduled to be removed in a future release, and no
        replacement is currently planned.
        If you make use of this functionality, please contact the Iris
        Developers to discuss how to retain it (which could include reversing
        the deprecation).

    """
    wmsg = (
        "iris.experimental.raster.export_geotiff has been deprecated, and will "
        "be removed in a future release.  Please consult the docstring for "
        "details."
    )
    warn_deprecated(wmsg)

    if cube.ndim != 2:
        raise ValueError("The cube must be two dimensional.")

    coord_x = cube.coord(axis="X", dim_coords=True)
    coord_y = cube.coord(axis="Y", dim_coords=True)

    if coord_x.bounds is None or coord_y.bounds is None:
        raise ValueError("Coordinates must have bounds, consider using guess_bounds()")

    if (
        coord_x is None
        or coord_y is None
        or coord_x.coord_system != coord_y.coord_system
    ):
        raise ValueError("The X and Y coordinates must share a CoordSystem.")

    xy_step = []
    for coord in [coord_x, coord_y]:
        name = coord.name()
        if coord.nbounds != 2:
            msg = "Coordinate {!r} must have two bounds per point.".format(name)
            raise ValueError(msg)
        if not (
            coord.units == cf_units.Unit("degrees")
            or coord.units.is_convertible("meters")
        ):
            raise ValueError(
                "Coordinate {!r} units must be either degrees or "
                "convertible to meters.".format(name)
            )
        if not coord.is_contiguous():
            raise ValueError("Coordinate {!r} bounds must be contiguous.".format(name))
        xy_step.append(np.diff(coord.bounds[0]))
        if not np.allclose(np.diff(coord.bounds), xy_step[-1]):
            msg = "Coordinate {!r} bounds must be regularly spaced.".format(name)
            raise ValueError(msg)

    if coord_x.points[0] > coord_x.points[-1]:
        raise ValueError(
            "Coordinate {!r} x-points must be monotonically increasing.".format(name)
        )

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
    _gdal_write_array(x_min, x_step, y_max, y_step, coord_system, data, fname, "GTiff")
