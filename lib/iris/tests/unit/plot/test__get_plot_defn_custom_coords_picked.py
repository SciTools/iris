# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot._get_plot_defn_custom_coords_picked`
function.
"""

import pytest

from iris.coords import BOUND_MODE, POINT_MODE
from iris.tests import _shared_utils
from iris.tests.stock import hybrid_height, simple_2d, simple_2d_w_multidim_coords

if _shared_utils.MPL_AVAILABLE:
    import iris.plot as iplt


@_shared_utils.skip_plot
class Test_get_plot_defn_custom_coords_picked:
    def test_1d_coords(self):
        cube = simple_2d()
        defn = iplt._get_plot_defn_custom_coords_picked(
            cube, ("foo", "bar"), POINT_MODE
        )
        assert [coord.name() for coord in defn.coords] == ["bar", "foo"]
        assert not defn.transpose

    def test_1d_coords_swapped(self):
        cube = simple_2d()
        defn = iplt._get_plot_defn_custom_coords_picked(
            cube, ("bar", "foo"), POINT_MODE
        )
        assert [coord.name() for coord in defn.coords] == ["foo", "bar"]
        assert defn.transpose

    def test_1d_coords_as_integers(self):
        cube = simple_2d()
        defn = iplt._get_plot_defn_custom_coords_picked(cube, (1, 0), POINT_MODE)
        assert [coord for coord in defn.coords] == [0, 1]
        assert not defn.transpose

    def test_1d_coords_as_integers_swapped(self):
        cube = simple_2d()
        defn = iplt._get_plot_defn_custom_coords_picked(cube, (0, 1), POINT_MODE)
        assert [coord for coord in defn.coords] == [1, 0]
        assert defn.transpose

    def test_2d_coords(self):
        cube = simple_2d_w_multidim_coords()
        defn = iplt._get_plot_defn_custom_coords_picked(
            cube, ("foo", "bar"), BOUND_MODE
        )
        assert [coord.name() for coord in defn.coords] == ["bar", "foo"]
        assert not defn.transpose

    def test_2d_coords_as_integers(self):
        cube = simple_2d_w_multidim_coords()
        defn = iplt._get_plot_defn_custom_coords_picked(cube, (0, 1), BOUND_MODE)
        assert [coord for coord in defn.coords] == [1, 0]
        assert defn.transpose

    def test_span_check(self):
        cube = hybrid_height()
        emsg = "don't span the 2 data dimensions"
        with pytest.raises(ValueError, match=emsg):
            iplt._get_plot_defn_custom_coords_picked(
                cube, ("sigma", "level_height"), POINT_MODE
            )
