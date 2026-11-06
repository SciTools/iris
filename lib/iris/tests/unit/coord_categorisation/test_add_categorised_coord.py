# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.coord_categorisation.add_categorised_coord`."""

from cf_units import CALENDARS as calendars
from cf_units import Unit
import numpy as np
import pytest

from iris.coord_categorisation import add_categorised_coord, add_day_of_year
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import _shared_utils


class Test_add_categorised_coord:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        # Factor out common variables and objects.
        self.cube = mocker.Mock(name="cube", coords=mocker.Mock(return_value=[]))
        self.coord = mocker.Mock(name="coord", points=np.arange(12).reshape(3, 4))
        self.units = "units"
        self.vectorised = mocker.Mock(name="vectorized_result")

    def test_vectorise_call(self, mocker):
        # Check that the function being passed through gets called with
        # numpy.vectorize, before being applied to the points array.
        # The reason we use numpy.vectorize is to support multi-dimensional
        # coordinate points.
        def fn(coord, v):
            return v**2

        vectorise_patch = mocker.patch("numpy.vectorize", return_value=self.vectorised)
        aux_coord_constructor = mocker.patch("iris.coords.AuxCoord")

        add_categorised_coord(self.cube, "foobar", self.coord, fn, units=self.units)

        # Check the constructor of AuxCoord gets called with the
        # appropriate arguments.
        # Start with the vectorised function.
        vectorise_patch.assert_called_once_with(fn)
        # Check the vectorize wrapper gets called with the appropriate args.
        self.vectorised.assert_called_once_with(self.coord, self.coord.points)
        # Check the AuxCoord constructor itself.
        aux_coord_constructor.assert_called_once_with(
            self.vectorised(self.coord, self.coord.points),
            units=self.units,
            attributes=self.coord.attributes.copy(),
        )
        # And check adding the aux coord to the cube mock.
        self.cube.add_aux_coord.assert_called_once_with(
            aux_coord_constructor(), self.cube.coord_dims(self.coord)
        )

    def test_string_vectorised(self, mocker):
        # Check that special case handling of a vectorized string returning
        # function is taking place.
        def fn(coord, v):
            return "0123456789"[:v]

        vectorise_patch = mocker.patch("numpy.vectorize", return_value=self.vectorised)
        aux_coord_constructor = mocker.patch("iris.coords.AuxCoord")

        add_categorised_coord(self.cube, "foobar", self.coord, fn, units=self.units)

        assert aux_coord_constructor.call_args[0][0] == vectorise_patch(
            fn, otypes=[object]
        )(self.coord, self.coord.points).astype("|S64")


class Test_add_day_of_year:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.expected = {
            "standard": np.array(list(range(360, 367)) + list(range(1, 4))),
            "gregorian": np.array(list(range(360, 367)) + list(range(1, 4))),
            "proleptic_gregorian": np.array(list(range(360, 367)) + list(range(1, 4))),
            "noleap": np.array(list(range(359, 366)) + list(range(1, 4))),
            "julian": np.array(list(range(360, 367)) + list(range(1, 4))),
            "all_leap": np.array(list(range(360, 367)) + list(range(1, 4))),
            "365_day": np.array(list(range(359, 366)) + list(range(1, 4))),
            "366_day": np.array(list(range(360, 367)) + list(range(1, 4))),
            "360_day": np.array(list(range(355, 361)) + list(range(1, 5))),
        }

    def make_cube(self, calendar):
        n_times = 10
        cube = Cube(np.arange(n_times))
        time_coord = DimCoord(
            np.arange(n_times),
            standard_name="time",
            units=Unit("days since 1980-12-25", calendar=calendar),
        )
        cube.add_dim_coord(time_coord, 0)
        return cube

    def test_calendars(self):
        for calendar in calendars:
            # Skip the Julian calendar due to
            # https://github.com/Unidata/cftime/issues/13
            # Remove this if block once the issue is resolved.
            if calendar == "julian":
                continue
            cube = self.make_cube(calendar)
            add_day_of_year(cube, "time")
            points = cube.coord("day_of_year").points
            expected_points = self.expected[calendar]
            msg = "Test failed for the following calendar: {}."
            _shared_utils.assert_array_equal(
                points, expected_points, err_msg=msg.format(calendar)
            )
