# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
A collection of routines which create standard Cubes for test purposes.

"""

from datetime import datetime
import os.path

import numpy as np
import numpy.ma as ma

from cf_units import Unit
from iris.cube import Cube
import iris.aux_factory
import iris.coords
import iris.coords as icoords
from iris.coords import DimCoord, AuxCoord, CellMethod
import iris.tests as tests
from iris.coord_systems import GeogCS, RotatedGeogCS
from ._stock_2d_latlons import (  # noqa
    sample_2d_latlons,
    make_bounds_discontiguous_at_point,
)


def lat_lon_cube():
    """
    Returns a cube with a latitude and longitude suitable for testing
    saving to PP/NetCDF etc.

    """
    cube = Cube(np.arange(12, dtype=np.int32).reshape((3, 4)))
    cs = GeogCS(6371229)
    coord = DimCoord(
        points=np.array([-1, 0, 1], dtype=np.int32),
        standard_name="latitude",
        units="degrees",
        coord_system=cs,
    )
    cube.add_dim_coord(coord, 0)
    coord = DimCoord(
        points=np.array([-1, 0, 1, 2], dtype=np.int32),
        standard_name="longitude",
        units="degrees",
        coord_system=cs,
    )
    cube.add_dim_coord(coord, 1)
    return cube


def global_pp():
    """
    Returns a two-dimensional cube derived from PP/aPPglob1/global.pp.

    The standard_name and unit attributes are added to compensate for the
    broken STASH encoding in that file.

    """

    def callback_global_pp(cube, field, filename):
        cube.standard_name = "air_temperature"
        cube.units = "K"

    path = tests.get_data_path(("PP", "aPPglob1", "global.pp"))
    cube = iris.load_cube(path, callback=callback_global_pp)
    return cube


def simple_pp():
    # Differs from global_pp()
    filename = tests.get_data_path(["PP", "simple_pp", "global.pp"])
    cube = iris.load_cube(filename)
    return cube


def simple_1d(with_bounds=True):
    """
    Returns an abstract, one-dimensional cube.

    >>> print(simple_1d())
    thingness                           (foo: 11)
         Dimension coordinates:
              foo                           x

    >>> print(repr(simple_1d().data))
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]

    """
    cube = Cube(np.arange(11, dtype=np.int32))
    cube.long_name = "thingness"
    cube.units = "1"
    points = np.arange(11, dtype=np.int32) + 1
    bounds = np.column_stack(
        [np.arange(11, dtype=np.int32), np.arange(11, dtype=np.int32) + 1]
    )
    coord = DimCoord(points, long_name="foo", units="1", bounds=bounds)
    cube.add_dim_coord(coord, 0)
    return cube


def simple_2d(with_bounds=True):
    """
    Returns an abstract, two-dimensional, optionally bounded, cube.

    >>> print(simple_2d())
    thingness                           (bar: 3; foo: 4)
         Dimension coordinates:
              bar                           x       -
              foo                           -       x

    >>> print(repr(simple_2d().data))
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]


    """
    cube = Cube(np.arange(12, dtype=np.int32).reshape((3, 4)))
    cube.long_name = "thingness"
    cube.units = "1"
    y_points = np.array([2.5, 7.5, 12.5])
    y_bounds = np.array([[0, 5], [5, 10], [10, 15]], dtype=np.int32)
    y_coord = DimCoord(
        y_points,
        long_name="bar",
        units="1",
        bounds=y_bounds if with_bounds else None,
    )
    x_points = np.array([-7.5, 7.5, 22.5, 37.5])
    x_bounds = np.array(
        [[-15, 0], [0, 15], [15, 30], [30, 45]], dtype=np.int32
    )
    x_coord = DimCoord(
        x_points,
        long_name="foo",
        units="1",
        bounds=x_bounds if with_bounds else None,
    )

    cube.add_dim_coord(y_coord, 0)
    cube.add_dim_coord(x_coord, 1)
    return cube


def simple_2d_w_multidim_coords(with_bounds=True):
    """
    Returns an abstract, two-dimensional, optionally bounded, cube.

    >>> print(simple_2d_w_multidim_coords())
    thingness                           (*ANONYMOUS*: 3; *ANONYMOUS*: 4)
         Auxiliary coordinates:
              bar                                   x               x
              foo                                   x               x

    >>> print(repr(simple_2d().data))
    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11]]

    """
    cube = simple_3d_w_multidim_coords(with_bounds)[0, :, :]
    cube.remove_coord("wibble")
    cube.data = np.arange(12, dtype=np.int32).reshape((3, 4))
    return cube


def simple_3d_w_multidim_coords(with_bounds=True):
    """
    Returns an abstract, two-dimensional, optionally bounded, cube.

    >>> print(simple_3d_w_multidim_coords())
    thingness                           (wibble: 2; *ANONYMOUS*: 3; *ANONYMOUS*: 4)
         Dimension coordinates:
              wibble                           x               -               -
         Auxiliary coordinates:
              bar                              -               x               x
              foo                              -               x               x

    >>> print(simple_3d_w_multidim_coords().data)
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]

     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]

    """
    cube = Cube(np.arange(24, dtype=np.int32).reshape((2, 3, 4)))
    cube.long_name = "thingness"
    cube.units = "1"

    y_points = np.array(
        [
            [2.5, 7.5, 12.5, 17.5],
            [10.0, 17.5, 27.5, 42.5],
            [15.0, 22.5, 32.5, 50.0],
        ]
    )
    y_bounds = np.array(
        [
            [[0, 5], [5, 10], [10, 15], [15, 20]],
            [[5, 15], [15, 20], [20, 35], [35, 50]],
            [[10, 20], [20, 25], [25, 40], [40, 60]],
        ],
        dtype=np.int32,
    )
    y_coord = AuxCoord(
        points=y_points,
        long_name="bar",
        units="1",
        bounds=y_bounds if with_bounds else None,
    )
    x_points = np.array(
        [
            [-7.5, 7.5, 22.5, 37.5],
            [-12.5, 4.0, 26.5, 47.5],
            [2.5, 14.0, 36.5, 44.0],
        ]
    )
    x_bounds = np.array(
        [
            [[-15, 0], [0, 15], [15, 30], [30, 45]],
            [[-25, 0], [0, 8], [8, 45], [45, 50]],
            [[-5, 10], [10, 18], [18, 55], [18, 70]],
        ],
        dtype=np.int32,
    )
    x_coord = AuxCoord(
        points=x_points,
        long_name="foo",
        units="1",
        bounds=x_bounds if with_bounds else None,
    )
    wibble_coord = DimCoord(
        np.array([10.0, 30.0], dtype=np.float32), long_name="wibble", units="1"
    )

    cube.add_dim_coord(wibble_coord, [0])
    cube.add_aux_coord(y_coord, [1, 2])
    cube.add_aux_coord(x_coord, [1, 2])
    return cube


def simple_3d():
    """
    Returns an abstract three dimensional cube.

    >>> print(simple_3d())
    thingness / (1)                     (wibble: 2; latitude: 3; longitude: 4)
     Dimension coordinates:
          wibble                           x            -             -
          latitude                         -            x             -
          longitude                        -            -             x

    >>> print(simple_3d().data)
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]

     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]

    """
    cube = Cube(np.arange(24, dtype=np.int32).reshape((2, 3, 4)))
    cube.long_name = "thingness"
    cube.units = "1"
    wibble_coord = DimCoord(
        np.array([10.0, 30.0], dtype=np.float32), long_name="wibble", units="1"
    )
    lon = DimCoord(
        [-180, -90, 0, 90],
        standard_name="longitude",
        units="degrees",
        circular=True,
    )
    lat = DimCoord([90, 0, -90], standard_name="latitude", units="degrees")
    cube.add_dim_coord(wibble_coord, [0])
    cube.add_dim_coord(lat, [1])
    cube.add_dim_coord(lon, [2])
    return cube


def simple_3d_mask():
    """
    Returns an abstract three dimensional cube that has data masked.

    >>> print(simple_3d_mask())
    thingness / (1)                     (wibble: 2; latitude: 3; longitude: 4)
     Dimension coordinates:
          wibble                           x            -             -
          latitude                         -            x             -
          longitude                        -            -             x

    >>> print(simple_3d_mask().data)
    [[[-- -- -- --]
      [-- -- -- --]
      [-- 9 10 11]]

    [[12 13 14 15]
     [16 17 18 19]
     [20 21 22 23]]]

    """
    cube = simple_3d()
    cube.data = ma.asanyarray(cube.data)
    cube.data = ma.masked_less_equal(cube.data, 8.0)
    return cube


def track_1d(duplicate_x=False):
    """
    Returns a one-dimensional track through two-dimensional space.

    >>> print(track_1d())
    air_temperature                     (y, x: 11)
         Dimensioned coords:
              x -> x
              y -> y
         Single valued coords:

    >>> print(repr(track_1d().data))
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    """
    cube = Cube(
        np.arange(11, dtype=np.int32),
        standard_name="air_temperature",
        units="K",
    )
    bounds = np.column_stack(
        [np.arange(11, dtype=np.int32), np.arange(11, dtype=np.int32) + 1]
    )
    pts = bounds[:, 1]
    coord = AuxCoord(pts, "projection_x_coordinate", units="1", bounds=bounds)
    cube.add_aux_coord(coord, [0])
    if duplicate_x:
        coord = AuxCoord(
            pts, "projection_x_coordinate", units="1", bounds=bounds
        )
        cube.add_aux_coord(coord, [0])
    coord = AuxCoord(
        pts * 2, "projection_y_coordinate", units="1", bounds=bounds * 2
    )
    cube.add_aux_coord(coord, 0)
    return cube


def simple_2d_w_multidim_and_scalars():
    data = np.arange(50, dtype=np.int32).reshape((5, 10))
    cube = iris.cube.Cube(
        data, long_name="test 2d dimensional cube", units="meters"
    )

    # DimCoords
    dim1 = DimCoord(
        np.arange(5, dtype=np.float32) * 5.1 + 3.0,
        long_name="dim1",
        units="meters",
    )
    dim2 = DimCoord(
        np.arange(10, dtype=np.int32),
        long_name="dim2",
        units="meters",
        bounds=np.arange(20, dtype=np.int32).reshape(10, 2),
    )

    # Scalars
    an_other = AuxCoord(3.0, long_name="an_other", units="meters")
    yet_an_other = DimCoord(
        23.3,
        standard_name="air_temperature",
        long_name="custom long name",
        var_name="custom_var_name",
        units="K",
    )

    # Multidim
    my_multi_dim_coord = AuxCoord(
        np.arange(50, dtype=np.int32).reshape(5, 10),
        long_name="my_multi_dim_coord",
        units="1",
        bounds=np.arange(200, dtype=np.int32).reshape(5, 10, 4),
    )

    cube.add_dim_coord(dim1, 0)
    cube.add_dim_coord(dim2, 1)
    cube.add_aux_coord(an_other)
    cube.add_aux_coord(yet_an_other)
    cube.add_aux_coord(my_multi_dim_coord, [0, 1])

    return cube


def hybrid_height():
    """
    Returns a two-dimensional (Z, X), hybrid-height cube.

    >>> print(hybrid_height())
    TODO: Update!
    air_temperature                     (level_height: 3; *ANONYMOUS*: 4)
         Dimension coordinates:
              level_height                           x               -
         Auxiliary coordinates:
              model_level_number                     x               -
              sigma                                  x               -
              surface_altitude                       -               x
         Derived coordinates:
              altitude                               x               x

    >>> print(hybrid_height().data)
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]

    """
    data = np.arange(12, dtype="i8").reshape((3, 4))

    orography = AuxCoord(
        [10, 25, 50, 5], standard_name="surface_altitude", units="m"
    )
    model_level = AuxCoord([2, 1, 0], standard_name="model_level_number")
    level_height = DimCoord(
        [100, 50, 10],
        long_name="level_height",
        units="m",
        attributes={"positive": "up"},
        bounds=[[150, 75], [75, 20], [20, 0]],
    )
    sigma = AuxCoord(
        [0.8, 0.9, 0.95],
        long_name="sigma",
        bounds=[[0.7, 0.85], [0.85, 0.97], [0.97, 1.0]],
    )
    hybrid_height = iris.aux_factory.HybridHeightFactory(
        level_height, sigma, orography
    )

    cube = iris.cube.Cube(
        data,
        standard_name="air_temperature",
        units="K",
        dim_coords_and_dims=[(level_height, 0)],
        aux_coords_and_dims=[(orography, 1), (model_level, 0), (sigma, 0)],
        aux_factories=[hybrid_height],
    )
    return cube


def simple_4d_with_hybrid_height():
    cube = iris.cube.Cube(
        np.arange(3 * 4 * 5 * 6, dtype="i8").reshape(3, 4, 5, 6),
        "air_temperature",
        units="K",
    )

    cube.add_dim_coord(
        DimCoord(np.arange(3, dtype="i8"), "time", units="hours since epoch"),
        0,
    )
    cube.add_dim_coord(
        DimCoord(
            np.arange(4, dtype="i8") + 10, "model_level_number", units="1"
        ),
        1,
    )
    cube.add_dim_coord(
        DimCoord(
            np.arange(5, dtype="i8") + 20, "grid_latitude", units="degrees"
        ),
        2,
    )
    cube.add_dim_coord(
        DimCoord(
            np.arange(6, dtype="i8") + 30, "grid_longitude", units="degrees"
        ),
        3,
    )

    cube.add_aux_coord(
        AuxCoord(
            np.arange(4, dtype="i8") + 40, long_name="level_height", units="m"
        ),
        1,
    )
    cube.add_aux_coord(
        AuxCoord(np.arange(4, dtype="i8") + 50, long_name="sigma", units="1"),
        1,
    )
    cube.add_aux_coord(
        AuxCoord(
            np.arange(5 * 6, dtype="i8").reshape(5, 6) + 100,
            long_name="surface_altitude",
            units="m",
        ),
        [2, 3],
    )

    cube.add_aux_factory(
        iris.aux_factory.HybridHeightFactory(
            delta=cube.coord("level_height"),
            sigma=cube.coord("sigma"),
            orography=cube.coord("surface_altitude"),
        )
    )
    return cube


def realistic_3d():
    """
    Returns a realistic 3d cube.

    >>> print(repr(realistic_3d()))
    <iris 'Cube' of air_potential_temperature (time: 7; grid_latitude: 9;
    grid_longitude: 11)>

    """
    data = np.arange(7 * 9 * 11).reshape((7, 9, 11))
    lat_pts = np.linspace(-4, 4, 9)
    lon_pts = np.linspace(-5, 5, 11)
    time_pts = np.linspace(394200, 394236, 7)
    forecast_period_pts = np.linspace(0, 36, 7)
    ll_cs = RotatedGeogCS(37.5, 177.5, ellipsoid=GeogCS(6371229.0))

    lat = icoords.DimCoord(
        lat_pts,
        standard_name="grid_latitude",
        units="degrees",
        coord_system=ll_cs,
    )
    lon = icoords.DimCoord(
        lon_pts,
        standard_name="grid_longitude",
        units="degrees",
        coord_system=ll_cs,
    )
    time = icoords.DimCoord(
        time_pts, standard_name="time", units="hours since 1970-01-01 00:00:00"
    )
    forecast_period = icoords.DimCoord(
        forecast_period_pts, standard_name="forecast_period", units="hours"
    )
    height = icoords.DimCoord(1000.0, standard_name="air_pressure", units="Pa")
    cube = iris.cube.Cube(
        data,
        standard_name="air_potential_temperature",
        units="K",
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
        aux_coords_and_dims=[(forecast_period, 0), (height, None)],
        attributes={"source": "Iris test case"},
    )
    return cube


def realistic_4d():
    """
    Returns a realistic 4d cube.

    >>> print(repr(realistic_4d()))
    <iris 'Cube' of air_potential_temperature (time: 6; model_level_number: 70;
    grid_latitude: 100; grid_longitude: 100)>

    """
    data_path = tests.get_data_path(("stock", "stock_arrays.npz"))
    if not os.path.isfile(data_path):
        raise IOError("Test data is not available at {}.".format(data_path))
    r = np.load(data_path)
    # sort the arrays based on the order they were originally given.
    # The names given are of the form 'arr_1' or 'arr_10'
    _, arrays = zip(*sorted(r.items(), key=lambda item: int(item[0][4:])))

    (
        lat_pts,
        lat_bnds,
        lon_pts,
        lon_bnds,
        level_height_pts,
        level_height_bnds,
        model_level_pts,
        sigma_pts,
        sigma_bnds,
        time_pts,
        _source_pts,
        forecast_period_pts,
        data,
        orography,
    ) = arrays

    ll_cs = RotatedGeogCS(37.5, 177.5, ellipsoid=GeogCS(6371229.0))

    lat = icoords.DimCoord(
        lat_pts,
        standard_name="grid_latitude",
        units="degrees",
        bounds=lat_bnds,
        coord_system=ll_cs,
    )
    lon = icoords.DimCoord(
        lon_pts,
        standard_name="grid_longitude",
        units="degrees",
        bounds=lon_bnds,
        coord_system=ll_cs,
    )
    level_height = icoords.DimCoord(
        level_height_pts,
        long_name="level_height",
        units="m",
        bounds=level_height_bnds,
        attributes={"positive": "up"},
    )
    model_level = icoords.DimCoord(
        model_level_pts,
        standard_name="model_level_number",
        units="1",
        attributes={"positive": "up"},
    )
    sigma = icoords.AuxCoord(
        sigma_pts, long_name="sigma", units="1", bounds=sigma_bnds
    )
    orography = icoords.AuxCoord(
        orography, standard_name="surface_altitude", units="m"
    )
    time = icoords.DimCoord(
        time_pts, standard_name="time", units="hours since 1970-01-01 00:00:00"
    )
    forecast_period = icoords.DimCoord(
        forecast_period_pts, standard_name="forecast_period", units="hours"
    )

    hybrid_height = iris.aux_factory.HybridHeightFactory(
        level_height, sigma, orography
    )

    cube = iris.cube.Cube(
        data,
        standard_name="air_potential_temperature",
        units="K",
        dim_coords_and_dims=[(time, 0), (model_level, 1), (lat, 2), (lon, 3)],
        aux_coords_and_dims=[
            (orography, (2, 3)),
            (level_height, 1),
            (sigma, 1),
            (forecast_period, None),
        ],
        attributes={"source": "Iris test case"},
        aux_factories=[hybrid_height],
    )
    return cube


def realistic_4d_no_derived():
    """
    Returns a realistic 4d cube without hybrid height

    >>> print(repr(realistic_4d()))
    <iris 'Cube' of air_potential_temperature (time: 6; model_level_number: 70;
    grid_latitude: 100; grid_longitude: 100)>

    """
    cube = realistic_4d()

    # TODO determine appropriate way to remove aux_factory from a cube
    cube._aux_factories = []

    return cube


def realistic_4d_w_missing_data():
    data_path = tests.get_data_path(("stock", "stock_mdi_arrays.npz"))
    if not os.path.isfile(data_path):
        raise IOError("Test data is not available at {}.".format(data_path))
    data_archive = np.load(data_path)
    data = ma.masked_array(data_archive["arr_0"], mask=data_archive["arr_1"])

    # sort the arrays based on the order they were originally given.
    # The names given are of the form 'arr_1' or 'arr_10'

    ll_cs = GeogCS(6371229)

    lat = DimCoord(
        np.arange(20, dtype=np.float32),
        standard_name="grid_latitude",
        units="degrees",
        coord_system=ll_cs,
    )
    lon = DimCoord(
        np.arange(20, dtype=np.float32),
        standard_name="grid_longitude",
        units="degrees",
        coord_system=ll_cs,
    )
    time = DimCoord(
        [1000.0, 1003.0, 1006.0],
        standard_name="time",
        units="hours since 1970-01-01 00:00:00",
    )
    forecast_period = DimCoord(
        [0.0, 3.0, 6.0], standard_name="forecast_period", units="hours"
    )
    pressure = DimCoord(
        np.array([800.0, 900.0, 1000.0], dtype=np.float32),
        long_name="pressure",
        units="hPa",
    )

    cube = iris.cube.Cube(
        data,
        long_name="missing data test data",
        units="K",
        dim_coords_and_dims=[(time, 0), (pressure, 1), (lat, 2), (lon, 3)],
        aux_coords_and_dims=[(forecast_period, 0)],
        attributes={"source": "Iris test case"},
    )
    return cube


def global_grib2():
    path = tests.get_data_path(("GRIB", "global_t", "global.grib2"))
    cube = iris.load_cube(path)
    return cube


def ocean_sigma_z():
    """
    Return a sample cube with an
    :class:`iris.aux_factory.OceanSigmaZFactory` vertical coordinate.

    This is a fairly small cube with real coordinate arrays.  The coordinate
    values are derived from the sample data linked at
    https://github.com/SciTools/iris/pull/509#issuecomment-23565381.

    """
    co_time = DimCoord([0.0, 1.0], standard_name="time", units="")
    co_lats = DimCoord(
        [-58.1, -52.7, -46.9], standard_name="latitude", units=Unit("degrees")
    )
    co_lons = DimCoord(
        [65.1, 72.9, 83.7, 96.5],
        standard_name="longitude",
        units=Unit("degrees"),
    )
    co_ssh = AuxCoord(
        [
            [
                [-0.63157895, -0.52631579, -0.42105263, -0.31578947],
                [-0.78947368, -0.68421053, -0.57894737, -0.47368421],
                [-0.94736842, -0.84210526, -0.73684211, -0.63157895],
            ],
            [
                [-0.84210526, -0.73684211, -0.63157895, -0.52631579],
                [-1.00000000, -0.89473684, -0.78947368, -0.68421053],
                [-1.15789474, -1.05263158, -0.94736842, -0.84210526],
            ],
        ],
        standard_name="sea_surface_height",
        units=Unit("m"),
    )

    co_sigma = AuxCoord(
        [0.0, -0.1, -0.6, -1.0, -1.0],
        standard_name="ocean_sigma_z_coordinate",
        units=Unit("1"),
        attributes={"positive": "up"},
    )

    co_zlay = AuxCoord(
        [-137.2, -137.3, -137.4, -368.4, -1495.6],
        long_name="layer_depth",
        units=Unit("m"),
    )
    co_depth = AuxCoord(
        [
            [1625.7, 3921.2, 4106.4, 5243.5],
            [3615.4, 4942.6, 3883.6, 4823.1],
            [3263.2, 2816.3, 2741.8, 3883.6],
        ],
        standard_name="depth",
        units=Unit("m"),
    )
    co_depthc = DimCoord(137.9, long_name="depth_c", units=Unit("m"))
    co_nsigma = DimCoord(3, long_name="nsigma")

    cube = Cube(np.zeros((2, 5, 3, 4)))
    cube.add_dim_coord(co_time, 0)
    cube.add_dim_coord(co_lats, 2)
    cube.add_dim_coord(co_lons, 3)
    cube.add_aux_coord(co_zlay, 1)
    cube.add_aux_coord(co_sigma, 1)
    cube.add_aux_coord(co_ssh, (0, 2, 3))
    cube.add_aux_coord(co_depth, (2, 3))
    cube.add_aux_coord(co_depthc)
    cube.add_aux_coord(co_nsigma)

    fact = iris.aux_factory.OceanSigmaZFactory(
        depth=co_depth,
        eta=co_ssh,
        depth_c=co_depthc,
        zlev=co_zlay,
        sigma=co_sigma,
        nsigma=co_nsigma,
    )
    cube.add_aux_factory(fact)
    return cube


def climatology_3d():
    def jan_offset(day, year):
        dt = datetime(year, 1, day) - datetime(1970, 1, 1)
        return dt.total_seconds() / (24.0 * 3600)

    days = range(10, 15)
    years = [[year, year + 10] for year in [2001] * 4]
    days_since = [
        [jan_offset(day, yr1), jan_offset(day, yr2)]
        for (day, [yr1, yr2]) in zip(days, years)
    ]
    time_bounds = np.array(days_since)
    time_points = time_bounds[..., 0]

    lon = np.linspace(-25, 25, 5)
    lat = np.linspace(0, 60, 3)

    time_dim = DimCoord(
        time_points,
        standard_name="time",
        bounds=time_bounds,
        units="days since 1970-01-01 00:00:00-00",
        climatological=True,
    )
    lon_dim = DimCoord(lon, standard_name="longitude", units="degrees")
    lat_dim = DimCoord(lat, standard_name="latitude", units="degrees")

    data_shape = (len(time_points), len(lat), len(lon))
    values = np.zeros(shape=data_shape, dtype=np.int8)
    cube = Cube(values)
    cube.add_dim_coord(time_dim, 0)
    cube.add_dim_coord(lat_dim, 1)
    cube.add_dim_coord(lon_dim, 2)
    cube.rename("climatology test")
    cube.units = "Kelvin"
    cube.add_cell_method(CellMethod("mean over years", coords="time"))

    return cube
