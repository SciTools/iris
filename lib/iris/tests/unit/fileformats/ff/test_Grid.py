# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformat.ff.Grid`."""

import pytest

from iris.fileformats._ff import Grid
from iris.tests.unit.fileformats import MockerMixin


class Test___init__:
    def test_attributes(self, mocker):
        # Ensure the constructor initialises all the grid's attributes
        # correctly, including unpacking values from the REAL constants.
        reals = (
            mocker.sentinel.ew,
            mocker.sentinel.ns,
            mocker.sentinel.first_lat,
            mocker.sentinel.first_lon,
            mocker.sentinel.pole_lat,
            mocker.sentinel.pole_lon,
        )
        grid = Grid(
            mocker.sentinel.column,
            mocker.sentinel.row,
            reals,
            mocker.sentinel.horiz_grid_type,
        )
        assert grid.column_dependent_constants is mocker.sentinel.column
        assert grid.row_dependent_constants is mocker.sentinel.row
        assert grid.ew_spacing is mocker.sentinel.ew
        assert grid.ns_spacing is mocker.sentinel.ns
        assert grid.first_lat is mocker.sentinel.first_lat
        assert grid.first_lon is mocker.sentinel.first_lon
        assert grid.pole_lat is mocker.sentinel.pole_lat
        assert grid.pole_lon is mocker.sentinel.pole_lon
        assert grid.horiz_grid_type is mocker.sentinel.horiz_grid_type


class Test_vectors(MockerMixin):
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.xp = mocker.sentinel.xp
        self.xu = mocker.sentinel.xu
        self.yp = mocker.sentinel.yp
        self.yv = mocker.sentinel.yv

    def _test_subgrid_vectors(self, subgrid, expected):
        grid = Grid(None, None, (None,) * 6, None)
        grid._x_vectors = self.mocker.Mock(return_value=(self.xp, self.xu))
        grid._y_vectors = self.mocker.Mock(return_value=(self.yp, self.yv))
        result = grid.vectors(subgrid)
        assert result == expected

    def test_1(self):
        # Data on atmospheric theta points.
        self._test_subgrid_vectors(1, (self.xp, self.yp))

    def test_2(self):
        # Data on atmospheric theta points, values over land only.
        self._test_subgrid_vectors(2, (self.xp, self.yp))

    def test_3(self):
        # Data on atmospheric theta points, values over sea only.
        self._test_subgrid_vectors(3, (self.xp, self.yp))

    def test_4(self):
        # Data on atmospheric zonal theta points.
        self._test_subgrid_vectors(4, (self.xp, self.yp))

    def test_5(self):
        # Data on atmospheric meridional theta points.
        self._test_subgrid_vectors(5, (self.xp, self.yp))

    def test_11(self):
        # Data on atmospheric uv points.
        self._test_subgrid_vectors(11, (self.xu, self.yv))

    def test_18(self):
        # Data on atmospheric u points on the 'c' grid.
        self._test_subgrid_vectors(18, (self.xu, self.yp))

    def test_19(self):
        # Data on atmospheric v points on the 'c' grid.
        self._test_subgrid_vectors(19, (self.xp, self.yv))

    def test_26(self):
        # Lateral boundary data at atmospheric theta points.
        self._test_subgrid_vectors(26, (self.xp, self.yp))

    def test_27(self):
        # Lateral boundary data at atmospheric u points.
        self._test_subgrid_vectors(27, (self.xu, self.yp))

    def test_28(self):
        # Lateral boundary data at atmospheric v points.
        self._test_subgrid_vectors(28, (self.xp, self.yv))

    def test_29(self):
        # Orography field for atmospheric LBCs.
        self._test_subgrid_vectors(29, (self.xp, self.yp))
