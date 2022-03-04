# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.coord_categorisation.add_categorised_coord`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

from cf_units import CALENDARS as calendars
from cf_units import Unit
import numpy as np

from iris.coord_categorisation import add_categorised_coord, add_day_of_year
from iris.coords import DimCoord
from iris.cube import Cube


class Test_add_categorised_coord(tests.IrisTest):
    def setUp(self):
        # Factor out common variables and objects.
        self.cube = mock.Mock(name="cube", coords=mock.Mock(return_value=[]))
        self.coord = mock.Mock(
            name="coord", points=np.arange(12).reshape(3, 4)
        )
        self.units = "units"
        self.vectorised = mock.Mock(name="vectorized_result")

    def test_vectorise_call(self):
        # Check that the function being passed through gets called with
        # numpy.vectorize, before being applied to the points array.
        # The reason we use numpy.vectorize is to support multi-dimensional
        # coordinate points.
        def fn(coord, v):
            return v**2

        with mock.patch(
            "numpy.vectorize", return_value=self.vectorised
        ) as vectorise_patch:
            with mock.patch("iris.coords.AuxCoord") as aux_coord_constructor:
                add_categorised_coord(
                    self.cube, "foobar", self.coord, fn, units=self.units
                )

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

    def test_string_vectorised(self):
        # Check that special case handling of a vectorized string returning
        # function is taking place.
        def fn(coord, v):
            return "0123456789"[:v]

        with mock.patch(
            "numpy.vectorize", return_value=self.vectorised
        ) as vectorise_patch:
            with mock.patch("iris.coords.AuxCoord") as aux_coord_constructor:
                add_categorised_coord(
                    self.cube, "foobar", self.coord, fn, units=self.units
                )

        self.assertEqual(
            aux_coord_constructor.call_args[0][0],
            vectorise_patch(fn, otypes=[object])(
                self.coord, self.coord.points
            ).astype("|S64"),
        )


class Test_add_day_of_year(tests.IrisTest):
    def setUp(self):
        self.expected = {
            "standard": np.array(list(range(360, 367)) + list(range(1, 4))),
            "gregorian": np.array(list(range(360, 367)) + list(range(1, 4))),
            "proleptic_gregorian": np.array(
                list(range(360, 367)) + list(range(1, 4))
            ),
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
            # https://github.com/Unidata/netcdftime/issues/13
            # Remove this if block once the issue is resolved.
            if calendar == "julian":
                continue
            cube = self.make_cube(calendar)
            add_day_of_year(cube, "time")
            points = cube.coord("day_of_year").points
            expected_points = self.expected[calendar]
            msg = "Test failed for the following calendar: {}."
            self.assertArrayEqual(
                points, expected_points, err_msg=msg.format(calendar)
            )


if __name__ == "__main__":
    tests.main()
