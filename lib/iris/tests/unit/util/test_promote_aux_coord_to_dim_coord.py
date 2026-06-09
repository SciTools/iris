# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.promote_aux_coord_to_dim_coord`."""

import re

import pytest

import iris
from iris.tests import _shared_utils
import iris.tests.stock as stock
from iris.util import promote_aux_coord_to_dim_coord


class Test:
    def test_dimension_already_has_dimcoord(self):
        cube_a = stock.hybrid_height()
        cube_b = cube_a.copy()
        promote_aux_coord_to_dim_coord(cube_b, "model_level_number")
        assert cube_b.dim_coords == (cube_a.coord("model_level_number"),)

    def test_old_dim_coord_is_now_aux_coord(self):
        cube_a = stock.hybrid_height()
        cube_b = cube_a.copy()
        promote_aux_coord_to_dim_coord(cube_b, "model_level_number")
        assert cube_a.coord("level_height") in cube_b.aux_coords

    @_shared_utils.skip_data
    def test_argument_is_coord_instance(self):
        cube_a = stock.realistic_4d()
        cube_b = cube_a.copy()
        promote_aux_coord_to_dim_coord(cube_b, cube_b.coord("level_height"))
        assert cube_b.dim_coords == (
            cube_a.coord("time"),
            cube_a.coord("level_height"),
            cube_a.coord("grid_latitude"),
            cube_a.coord("grid_longitude"),
        )

    @_shared_utils.skip_data
    def test_dimension_is_anonymous(self):
        cube_a = stock.realistic_4d()
        cube_b = cube_a.copy()
        cube_b.remove_coord("model_level_number")
        promote_aux_coord_to_dim_coord(cube_b, "level_height")
        assert cube_b.dim_coords == (
            cube_a.coord("time"),
            cube_a.coord("level_height"),
            cube_a.coord("grid_latitude"),
            cube_a.coord("grid_longitude"),
        )

    def test_already_a_dim_coord(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        cube_b = cube_a.copy()
        promote_aux_coord_to_dim_coord(cube_b, "dim1")
        assert cube_a == cube_b

    def test_coord_of_that_name_does_not_exist(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            promote_aux_coord_to_dim_coord(cube_a, "wibble")

    def test_coord_does_not_exist(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        coord = cube_a.coord("dim1").copy()
        coord.rename("new")
        msg = re.escape(
            "Attempting to promote an AuxCoord (new) which does not exist in the cube."
        )
        with pytest.raises(ValueError, match=msg):
            promote_aux_coord_to_dim_coord(cube_a, coord)

    def test_argument_is_wrong_type(self):
        cube_a = stock.simple_1d()
        with pytest.raises(TypeError):
            promote_aux_coord_to_dim_coord(cube_a, 0.0)

    def test_trying_to_promote_a_multidim_coord(self):
        cube_a = stock.simple_2d_w_multidim_coords()
        msg = re.escape(
            "Attempting to promote an AuxCoord (bar) which is associated with 2 dimensions."
        )
        with pytest.raises(ValueError, match=msg):
            promote_aux_coord_to_dim_coord(cube_a, "bar")

    def test_trying_to_promote_a_scalar_coord(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        msg = re.escape(
            "Attempting to promote an AuxCoord (an_other) which is associated with 0 dimensions."
        )
        with pytest.raises(ValueError, match=msg):
            promote_aux_coord_to_dim_coord(cube_a, "an_other")

    def test_trying_to_promote_a_nonmonotonic_coord(self):
        cube_a = stock.hybrid_height()
        msg = re.escape(
            "Attempt to promote an AuxCoord (surface_altitude) fails when attempting to create a DimCoord "
            "from the AuxCoord because: The 'surface_altitude' DimCoord points array must be strictly monotonic."
        )
        with pytest.raises(ValueError, match=msg):
            promote_aux_coord_to_dim_coord(cube_a, "surface_altitude")
