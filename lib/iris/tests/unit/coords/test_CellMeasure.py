# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coords.CellMeasure` class."""

import numpy as np
import pytest

from iris._lazy_data import as_lazy_data
from iris.coords import CellMeasure
from iris.cube import Cube
from iris.tests import _shared_utils


class TestCellMeasure:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.values = np.array((10.0, 12.0, 16.0, 9.0))
        self.measure = CellMeasure(
            self.values,
            units="m^2",
            standard_name="cell_area",
            long_name="measured_area",
            var_name="area",
            attributes={"notes": "1m accuracy"},
        )

    def test_invalid_measure(self):
        msg = "measure must be 'area' or 'volume', got 'length'"
        with pytest.raises(ValueError, match=msg):
            self.measure.measure = "length"

    def test_set_measure(self):
        v = "volume"
        self.measure.measure = v
        assert self.measure.measure == v

    def test_data(self):
        _shared_utils.assert_array_equal(self.measure.data, self.values)

    def test_set_data(self):
        new_vals = np.array((1.0, 2.0, 3.0, 4.0))
        self.measure.data = new_vals
        _shared_utils.assert_array_equal(self.measure.data, new_vals)

    def test_set_data__int(self):
        new_vals = np.array((1, 2, 3, 4), dtype=np.int32)
        self.measure.data = new_vals
        _shared_utils.assert_array_equal(self.measure.data, new_vals)

    def test_set_data__uint(self):
        new_vals = np.array((1, 2, 3, 4), dtype=np.uint32)
        self.measure.data = new_vals
        _shared_utils.assert_array_equal(self.measure.data, new_vals)

    def test_set_data__lazy(self):
        new_vals = as_lazy_data(np.array((1.0, 2.0, 3.0, 4.0)))
        self.measure.data = new_vals
        _shared_utils.assert_array_equal(self.measure.data, new_vals)

    def test_data_different_shape(self):
        new_vals = np.array((1.0, 2.0, 3.0))
        msg = "Require data with shape."
        with pytest.raises(ValueError, match=msg):
            self.measure.data = new_vals

    def test_shape(self):
        assert self.measure.shape == (4,)

    def test_ndim(self):
        assert self.measure.ndim == 1

    def test___getitem__(self):
        sub_measure = self.measure[2]
        _shared_utils.assert_array_equal(self.values[2], sub_measure.data)

    def test___getitem__data_copy(self):
        # Check that a sliced cell measure has independent data.
        sub_measure = self.measure[1:3]
        old_values = sub_measure.data.copy()
        # Change the original one.
        self.measure.data[:] = 0.0
        # Check the new one has not changed.
        _shared_utils.assert_array_equal(sub_measure.data, old_values)

    def test_copy(self):
        new_vals = np.array((7.0, 8.0))
        copy_measure = self.measure.copy(new_vals)
        _shared_utils.assert_array_equal(copy_measure.data, new_vals)

    def test___str__(self):
        expected = "\n".join(
            [
                "CellMeasure :  cell_area / (m^2)",
                "    data: [10., 12., 16.,  9.]",
                "    shape: (4,)",
                "    dtype: float64",
                "    standard_name: 'cell_area'",
                "    long_name: 'measured_area'",
                "    var_name: 'area'",
                "    attributes:",
                "        notes  '1m accuracy'",
                "    measure: 'area'",
            ]
        )
        assert self.measure.__str__() == expected

    def test___repr__(self):
        expected = "<CellMeasure: cell_area / (m^2)  [10., 12., 16., 9.]  shape(4,)>"
        assert expected == self.measure.__repr__()

    def test__eq__(self):
        assert self.measure == self.measure


class Test_cube_dims:
    def test_cube_dims(self, mocker):
        # Check that "coord.cube_dims(cube)" calls "cube.coord_dims(coord)".
        mock_dims_result = mocker.sentinel.CM_DIMS
        mock_dims_call = mocker.Mock(return_value=mock_dims_result)
        mock_cube = mocker.Mock(Cube, cell_measure_dims=mock_dims_call)
        test_cm = CellMeasure([1], long_name="test_name")

        result = test_cm.cube_dims(mock_cube)
        assert result == mock_dims_result
        assert mock_dims_call.call_args_list == [mocker.call(test_cm)]
