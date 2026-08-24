# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFCoordinateVariable`."""

import numpy as np
import numpy.ma as ma
import pytest

from iris.fileformats.cf import CFCoordinateVariable


class _CoordVariableStub:
    """Stub for a 1D netCDF variable acting as a coordinate variable."""

    def __init__(self, name, dimensions, data, dtype=float):
        self.name = name
        self.dtype = np.dtype(dtype)
        self.dimensions = dimensions
        self.ndim = len(dimensions)
        data_arr = np.asarray(data)
        self.shape = data_arr.shape or ()
        self._data = data

    def ncattrs(self):
        return []

    def __getitem__(self, key):
        if self._data.ndim == 0:
            return self._data
        return self._data[key]

    def __len__(self):
        return len(self._data)


def _make_coord_var(name, data, dtype=float):
    """Helper: create a valid 1D coord variable with name == dimension."""
    return _CoordVariableStub(name=name, dimensions=(name,), data=data, dtype=dtype)


class TestIdentify:
    def test_valid_coordinate_identified(self):
        nc_var = _make_coord_var("lat", [1.0, 2.0, 3.0])
        vars_all = {"lat": nc_var}

        result = CFCoordinateVariable.identify(vars_all)
        assert "lat" in result
        assert isinstance(result["lat"], CFCoordinateVariable)

    def test_string_dtype_rejected(self):
        nc_var = _CoordVariableStub(
            name="lat", dimensions=("lat",), data=["a", "b"], dtype=np.bytes_
        )
        vars_all = {"lat": nc_var}

        result = CFCoordinateVariable.identify(vars_all)
        assert {} == result

    def test_ndim_not_one_rejected(self):
        stub = _CoordVariableStub(
            name="lat", dimensions=("lat", "lon"), data=[[1.0, 2.0]], dtype=float
        )
        assert stub.ndim == 2
        assert stub.shape == (1, 2)
        vars_all = {"lat": stub}

        result = CFCoordinateVariable.identify(vars_all)
        assert {} == result

    def test_name_not_in_dimensions_rejected(self):
        stub = _CoordVariableStub(
            name="lat", dimensions=("x",), data=[1.0, 2.0], dtype=float
        )
        vars_all = {"lat": stub}

        result = CFCoordinateVariable.identify(vars_all)
        assert {} == result

    def test_ignored_name_excluded(self):
        nc_var = _make_coord_var("lat", [1.0, 2.0, 3.0])
        vars_all = {"lat": nc_var}

        result = CFCoordinateVariable.identify(vars_all, ignore=["lat"])
        assert {} == result

    def test_target_filters_to_named_var(self):
        lat = _make_coord_var("lat", [1.0, 2.0])
        lon = _make_coord_var("lon", [10.0, 20.0])
        vars_all = {"lat": lat, "lon": lon}

        result = CFCoordinateVariable.identify(vars_all, target="lat")
        assert "lat" in result
        assert "lon" not in result

    def test_target_unknown_raises(self):
        vars_all = {"lat": _make_coord_var("lat", [1.0])}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            CFCoordinateVariable.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self):
        vars_all = {"lat": _make_coord_var("lat", [1.0])}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            CFCoordinateVariable.identify(vars_all, target=object())


class TestIdentifyMonotonic:
    def test_monotonic_increasing_accepted(self):
        nc_var = _make_coord_var("lat", np.array([1.0, 2.0, 3.0]))
        vars_all = {"lat": nc_var}

        result = CFCoordinateVariable.identify(vars_all, monotonic=True)
        assert "lat" in result

    def test_monotonic_decreasing_accepted(self):
        nc_var = _make_coord_var("lat", np.array([3.0, 2.0, 1.0]))
        vars_all = {"lat": nc_var}

        result = CFCoordinateVariable.identify(vars_all, monotonic=True)
        assert "lat" in result

    def test_non_monotonic_rejected(self):
        nc_var = _make_coord_var("lat", np.array([1.0, 3.0, 2.0]))
        vars_all = {"lat": nc_var}

        result = CFCoordinateVariable.identify(vars_all, monotonic=True)
        assert {} == result

    def test_scalar_shape_accepted(self):
        """Shape () is always accepted under monotonic mode."""
        stub = _make_coord_var("lat", np.float64(1.0))
        vars_all = {"lat": stub}

        result = CFCoordinateVariable.identify(vars_all, monotonic=True)
        assert "lat" in result

    def test_single_element_shape_accepted(self):
        """Shape (1,) is always accepted under monotonic mode."""
        nc_var = _make_coord_var("lat", np.array([42.0]))
        vars_all = {"lat": nc_var}

        result = CFCoordinateVariable.identify(vars_all, monotonic=True)
        assert "lat" in result

    def test_masked_array_accepted_when_monotonic(self):
        """Masked arrays are filled before monotonic check."""
        data = ma.masked_array([1.0, 2.0, 3.0], mask=[False, False, False])
        nc_var = _make_coord_var("lat", data)
        vars_all = {"lat": nc_var}

        result = CFCoordinateVariable.identify(vars_all, monotonic=True)
        assert "lat" in result

    def test_masked_array_rejected_when_masked(self):
        """Masked arrays are rejected under monotonic mode if any elements are masked."""
        data = ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        nc_var = _make_coord_var("lat", data)
        vars_all = {"lat": nc_var}

        result = CFCoordinateVariable.identify(vars_all, monotonic=True)
        assert {} == result
