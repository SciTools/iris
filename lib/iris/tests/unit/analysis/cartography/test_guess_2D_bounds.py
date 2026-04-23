# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the function :func:`iris.analysis.cartography.guess_2D_bounds`."""

import numpy as np
import pytest

from iris.analysis.cartography import _transform_xy
from iris.coord_systems import Mercator, RotatedGeogCS
from iris.coords import AuxCoord
from iris.tests.unit.analysis.cartography.test_gridcell_angles import (
    _2d_multicells_testcube,
)
from iris.analysis.cartography import guess_2D_bounds


def _2D_guess_bounds(cube, extrapolate=True, in_place=False):
    lon = cube.coord(axis="X")
    lat = cube.coord(axis="Y")
    new_lon, new_lat = guess_2D_bounds(
        lon, lat, extrapolate=extrapolate, in_place=in_place
    )
    if not in_place:
        for old, new in [[lon, new_lon], [lat, new_lat]]:
            dims = cube.coord_dims(old)
            cube.remove_coord(old)
            cube.add_aux_coord(new, dims)
    return cube


def test_2D_guess_bounds_contiguity():
    cube = _2d_multicells_testcube()
    assert not cube.coord("latitude").is_contiguous()
    assert not cube.coord("longitude").is_contiguous()

    result_extrap = _2D_guess_bounds(cube)
    assert result_extrap.coord("latitude").is_contiguous()
    assert result_extrap.coord("longitude").is_contiguous()

    result_clipped = _2D_guess_bounds(cube, extrapolate=False)
    assert result_clipped.coord("latitude").is_contiguous()
    assert result_clipped.coord("longitude").is_contiguous()


def test_2D_guess_bounds_rotational_equivalence():
    # Check that _2D_guess_bounds is rotationally equivalent.
    cube = _2d_multicells_testcube()

    # Define a rotation with a pair of coordinate systems.
    rotated_cs = RotatedGeogCS(20, 30).as_cartopy_crs()
    unrotated_cs = RotatedGeogCS(0, 0).as_cartopy_crs()

    # Guess the bounds before rotating the lat-lon points.
    _2D_guess_bounds(cube, extrapolate=True, in_place=True)
    lon_bounds_unrotated = cube.coord("longitude").bounds
    lat_bounds_unrotated = cube.coord("latitude").bounds

    # Rotate the guessed lat-lon bounds.
    rotated_lon_bounds, rotated_lat_bounds = _transform_xy(
        unrotated_cs,
        lon_bounds_unrotated.flatten(),
        lat_bounds_unrotated.flatten(),
        rotated_cs,
    )

    # Rotate the lat-lon points.
    lat = cube.coord("latitude")
    lon = cube.coord("longitude")
    lon.points, lat.points = _transform_xy(
        unrotated_cs, lon.points, lat.points, rotated_cs
    )

    # guess the bounds after rotating the lat-lon points.
    _2D_guess_bounds(cube, extrapolate=True, in_place=True)
    lon_bounds_from_rotated_points = cube.coord("longitude").bounds
    lat_bounds_from_rotated_points = cube.coord("latitude").bounds

    # Check that the results are equivalent.
    assert np.allclose(rotated_lon_bounds, lon_bounds_from_rotated_points.flatten())
    assert np.allclose(rotated_lat_bounds, lat_bounds_from_rotated_points.flatten())


def test_2D_guess_bounds_transpose_equivalence():
    # Check that _2D_guess_bounds is transpose equivalent.
    cube = _2d_multicells_testcube()
    cube_transposed = _2d_multicells_testcube()

    def transpose_2D_coord(coord):
        new_points = coord.points.transpose()
        new_bounds = coord.bounds.transpose((1, 0, 2))[:, :, (0, 3, 2, 1)]
        new_coord = AuxCoord(
            new_points,
            bounds=new_bounds,
            standard_name=coord.standard_name,
            units=coord.units,
        )
        return new_coord

    cube_transposed.transpose()

    new_lat = transpose_2D_coord(cube_transposed.coord("latitude"))
    new_lon = transpose_2D_coord(cube_transposed.coord("longitude"))
    cube_transposed.remove_coord("latitude")
    cube_transposed.remove_coord("longitude")
    cube_transposed.add_aux_coord(new_lat, (0, 1))
    cube_transposed.add_aux_coord(new_lon, (0, 1))

    _2D_guess_bounds(cube, extrapolate=True, in_place=True)
    _2D_guess_bounds(cube_transposed, extrapolate=True, in_place=True)

    cube_transposed.transpose()

    untransposed_lat = transpose_2D_coord(cube_transposed.coord("latitude"))
    untransposed_lon = transpose_2D_coord(cube_transposed.coord("longitude"))

    assert np.allclose(untransposed_lat.bounds, cube.coord("latitude").bounds)
    assert np.allclose(untransposed_lon.bounds, cube.coord("longitude").bounds)


def test_2D_guess_bounds_slice_equivalence():
    # Extrapolation should approximate expected values from an extended regular grid when points are
    # close enough together.
    shrink_factor = 1000
    cube = _2d_multicells_testcube()
    for coord in cube.coords():
        coord.points = coord.points / shrink_factor
    sub_cube = cube[1:, 1:-1]
    _2D_guess_bounds(cube)
    _2D_guess_bounds(sub_cube)
    assert np.allclose(
        cube[1:, 1:-1].coord("latitude").bounds, sub_cube.coord("latitude").bounds
    )
    assert np.allclose(
        cube[1:, 1:-1].coord("longitude").bounds, sub_cube.coord("longitude").bounds
    )


def test_2D_guess_bounds_coord_systems():
    rotated_cs = RotatedGeogCS(20, 30)
    mercator_cs = Mercator()

    rotated_cube = _2d_multicells_testcube()
    rotated_cube.coord("latitude").coord_system = rotated_cs
    rotated_cube.coord("latitude").standard_name = "grid_latitude"
    rotated_cube.coord("longitude").coord_system = rotated_cs
    rotated_cube.coord("longitude").standard_name = "grid_longitude"

    _2D_guess_bounds(rotated_cube, in_place=True)

    assert rotated_cube.coord("grid_latitude").is_contiguous()
    assert rotated_cube.coord("grid_longitude").is_contiguous()

    mercator_cube = _2d_multicells_testcube()
    mercator_cube.coord("latitude").coord_system = mercator_cs
    mercator_cube.coord("longitude").coord_system = mercator_cs

    with pytest.raises(ValueError, match="Coordinate systems are expected geodetic."):
        _2D_guess_bounds(mercator_cube)
