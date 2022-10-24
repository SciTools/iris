# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.analysis.cartography._get_lon_lat_coords"""

import pytest

from iris.analysis.cartography import _get_lon_lat_coords as g_lon_lat
from iris.coords import AuxCoord
from iris.tests.stock import lat_lon_cube


@pytest.fixture
def dim_only_cube():
    return lat_lon_cube()


def test_dim_only(dim_only_cube):
    t_lat, t_lon = dim_only_cube.dim_coords

    lon, lat = g_lon_lat(dim_only_cube)

    assert lon == t_lon
    assert lat == t_lat


@pytest.fixture
def dim_aux_cube(dim_only_cube):
    lat_dim, lon_dim = dim_only_cube.dim_coords

    lat_aux = AuxCoord.from_coord(lat_dim)
    lat_aux.standard_name = "grid_latitude"
    lon_aux = AuxCoord.from_coord(lon_dim)
    lon_aux.standard_name = "grid_longitude"

    dim_aux_cube = dim_only_cube
    dim_aux_cube.add_aux_coord(lat_aux, 0)
    dim_aux_cube.add_aux_coord(lon_aux, 1)

    return dim_aux_cube


def test_dim_aux(dim_aux_cube):
    t_lat_dim, t_lon_dim = dim_aux_cube.dim_coords

    lon, lat = g_lon_lat(dim_aux_cube)

    assert lon == t_lon_dim
    assert lat == t_lat_dim


@pytest.fixture
def aux_only_cube(dim_aux_cube):
    lon_dim, lat_dim = dim_aux_cube.dim_coords

    aux_only_cube = dim_aux_cube
    aux_only_cube.remove_coord(lon_dim)
    aux_only_cube.remove_coord(lat_dim)

    return dim_aux_cube


def test_aux_only(aux_only_cube):
    aux_lat, aux_lon = aux_only_cube.aux_coords

    lon, lat = g_lon_lat(aux_only_cube)

    assert lon == aux_lon
    assert lat == aux_lat


@pytest.fixture
def double_dim_cube(dim_only_cube):
    double_dim_cube = dim_only_cube
    double_dim_cube.coord("latitude").standard_name = "grid_longitude"

    return double_dim_cube


def test_double_dim(double_dim_cube):
    t_error_message = "with multiple.*is currently disallowed"

    with pytest.raises(ValueError, match=t_error_message):
        g_lon_lat(double_dim_cube)


@pytest.fixture
def double_aux_cube(aux_only_cube):
    double_aux_cube = aux_only_cube
    double_aux_cube.coord("grid_latitude").standard_name = "longitude"

    return double_aux_cube


def test_double_aux(double_aux_cube):
    t_error_message = "with multiple.*is currently disallowed"

    with pytest.raises(ValueError, match=t_error_message):
        g_lon_lat(double_aux_cube)


@pytest.fixture
def missing_lat_cube(dim_only_cube):
    missing_lat_cube = dim_only_cube
    missing_lat_cube.remove_coord("latitude")

    return missing_lat_cube


def test_missing_coord(missing_lat_cube):
    with pytest.raises(IndexError):
        g_lon_lat(missing_lat_cube)
