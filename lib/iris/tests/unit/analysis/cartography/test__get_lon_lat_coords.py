# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.analysis.cartography._get_lon_lat_coords"""

import pytest

from iris.analysis.cartography import _get_lon_lat_coords as g_lat_lon
from iris.coords import AuxCoord
from iris.tests.stock import lat_lon_cube

dim_only_cube = lat_lon_cube()
lon_dim = dim_only_cube.coord("longitude")
lat_dim = dim_only_cube.coord("latitude")


def test_dim_only():
    lon, lat = g_lat_lon(dim_only_cube)
    assert lon == lon_dim
    assert lat == lat_dim


dim_aux_cube = lat_lon_cube()
lat_aux = AuxCoord.from_coord(lat_dim)
lat_aux.standard_name = "grid_latitude"
lon_aux = AuxCoord.from_coord(lon_dim)
lon_aux.standard_name = "grid_longitude"
dim_aux_cube.add_aux_coord(lat_aux, 0)
dim_aux_cube.add_aux_coord(lon_aux, 1)


def test_dim_and_aux():
    lon, lat = g_lat_lon(dim_aux_cube)
    assert lon == lon_dim
    assert lat == lat_dim


aux_only_cube = dim_aux_cube.copy(dim_aux_cube.data)
aux_only_cube.remove_coord(lon_dim)
aux_only_cube.remove_coord(lat_dim)


def test_aux_only():
    lon, lat = g_lat_lon(aux_only_cube)
    assert lon == lon_aux
    assert lat == lat_aux


double_dim_cube = lat_lon_cube()
double_dim_cube.coord("latitude").standard_name = "grid_longitude"


def test_double_dim():
    with pytest.raises(
        ValueError, match="with multiple.*is currently disallowed"
    ):
        _ = g_lat_lon(double_dim_cube)


double_aux_cube = aux_only_cube.copy(aux_only_cube.data)
double_aux_cube.coord("grid_latitude").standard_name = "longitude"


def test_double_aux():
    with pytest.raises(
        ValueError, match="with multiple.*is currently disallowed"
    ):
        _ = g_lat_lon(double_aux_cube)


missing_lat_cube = lat_lon_cube()
missing_lat_cube.remove_coord("latitude")


def test_missing_coord():
    with pytest.raises(IndexError):
        _ = g_lat_lon(missing_lat_cube)
