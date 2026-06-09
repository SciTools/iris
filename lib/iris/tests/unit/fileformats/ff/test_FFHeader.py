# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformat.ff.FFHeader`."""

import collections

import numpy as np
import pytest

from iris.fileformats._ff import FFHeader, _WarnComboLoadingDefaulting
from iris.tests import _shared_utils
from iris.tests.unit.fileformats import MockerMixin

MyGrid = collections.namedtuple("MyGrid", "column row real horiz_grid_type")


class Test_grid(MockerMixin):
    def _header(self, grid_staggering):
        _ = self.mocker.patch.object(
            FFHeader, "__init__", self.mocker.Mock(return_value=None)
        )
        header = FFHeader()
        header.grid_staggering = grid_staggering
        header.column_dependent_constants = self.mocker.sentinel.column
        header.row_dependent_constants = self.mocker.sentinel.row
        header.real_constants = self.mocker.sentinel.real
        header.horiz_grid_type = self.mocker.sentinel.horiz_grid_type
        return header

    def _test_grid_staggering(self, grid_staggering):
        header = self._header(grid_staggering)
        _ = self.mocker.patch.dict(
            FFHeader.GRID_STAGGERING_CLASS, {grid_staggering: MyGrid}
        )
        grid = header.grid()
        assert isinstance(grid, MyGrid)
        assert grid.column is self.mocker.sentinel.column
        assert grid.row is self.mocker.sentinel.row
        assert grid.real is self.mocker.sentinel.real
        assert grid.horiz_grid_type is self.mocker.sentinel.horiz_grid_type

    def test_new_dynamics(self):
        self._test_grid_staggering(3)

    def test_end_game(self):
        self._test_grid_staggering(6)

    def test_unknown(self, mocker):
        header = self._header(0)
        _ = mocker.patch(
            "iris.fileformats._ff.NewDynamics",
            mocker.Mock(return_value=mocker.sentinel.grid),
        )
        msg = (
            "Staggered grid type: 0 not currently interpreted, assuming standard C-grid"
        )
        with pytest.warns(_WarnComboLoadingDefaulting, match=msg):
            grid = header.grid()
        assert grid is mocker.sentinel.grid


@_shared_utils.skip_data
class Test_integer_constants:
    def test_read_ints(self):
        test_file_path = _shared_utils.get_data_path(("FF", "structured", "small"))
        ff_header = FFHeader(test_file_path)
        assert ff_header.integer_constants.dtype == np.dtype(">i8")
