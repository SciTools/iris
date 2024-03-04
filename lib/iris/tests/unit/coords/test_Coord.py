# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coords.Coord` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import collections
from unittest import mock
import warnings

import dask.array as da
import numpy as np

import iris
from iris.coords import AuxCoord, Coord, DimCoord
from iris.cube import Cube
from iris.exceptions import UnitConversionError
from iris.tests.unit.coords import CoordTestMixin

Pair = collections.namedtuple("Pair", "points bounds")


class Test_nearest_neighbour_index__ascending(tests.IrisTest):
    def setUp(self):
        points = [0.0, 90.0, 180.0, 270.0]
        self.coord = DimCoord(points, circular=False, units="degrees")

    def _test_nearest_neighbour_index(
        self, target, bounds=None, circular=False
    ):
        _bounds = [[-20, 10], [10, 100], [100, 260], [260, 340]]
        ext_pnts = [-70, -10, 110, 275, 370]
        if bounds is True:
            self.coord.bounds = _bounds
        else:
            self.coord.bounds = bounds
        self.coord.circular = circular
        results = [self.coord.nearest_neighbour_index(ind) for ind in ext_pnts]
        self.assertEqual(results, target)

    def test_nobounds(self):
        target = [0, 0, 1, 3, 3]
        self._test_nearest_neighbour_index(target)

    def test_nobounds_circular(self):
        target = [3, 0, 1, 3, 0]
        self._test_nearest_neighbour_index(target, circular=True)

    def test_bounded(self):
        target = [0, 0, 2, 3, 3]
        self._test_nearest_neighbour_index(target, bounds=True)

    def test_bounded_circular(self):
        target = [3, 0, 2, 3, 0]
        self._test_nearest_neighbour_index(target, bounds=True, circular=True)

    def test_bounded_overlapping(self):
        _bounds = [[-20, 50], [10, 150], [100, 300], [260, 340]]
        target = [0, 0, 1, 2, 3]
        self._test_nearest_neighbour_index(target, bounds=_bounds)

    def test_bounded_disjointed(self):
        _bounds = [[-20, 10], [80, 170], [180, 190], [240, 340]]
        target = [0, 0, 1, 3, 3]
        self._test_nearest_neighbour_index(target, bounds=_bounds)

    def test_scalar(self):
        self.coord = DimCoord([0], circular=False, units="degrees")
        target = [0, 0, 0, 0, 0]
        self._test_nearest_neighbour_index(target)

    def test_bounded_float_point(self):
        coord = DimCoord(1, bounds=[0, 2])
        result = coord.nearest_neighbour_index(2.5)
        self.assertEqual(result, 0)


class Test_nearest_neighbour_index__descending(tests.IrisTest):
    def setUp(self):
        points = [270.0, 180.0, 90.0, 0.0]
        self.coord = DimCoord(points, circular=False, units="degrees")

    def _test_nearest_neighbour_index(
        self, target, bounds=False, circular=False
    ):
        _bounds = [[340, 260], [260, 100], [100, 10], [10, -20]]
        ext_pnts = [-70, -10, 110, 275, 370]
        if bounds:
            self.coord.bounds = _bounds
        self.coord.circular = circular
        results = [self.coord.nearest_neighbour_index(ind) for ind in ext_pnts]
        self.assertEqual(results, target)

    def test_nobounds(self):
        target = [3, 3, 2, 0, 0]
        self._test_nearest_neighbour_index(target)

    def test_nobounds_circular(self):
        target = [0, 3, 2, 0, 3]
        self._test_nearest_neighbour_index(target, circular=True)

    def test_bounded(self):
        target = [3, 3, 1, 0, 0]
        self._test_nearest_neighbour_index(target, bounds=True)

    def test_bounded_circular(self):
        target = [0, 3, 1, 0, 3]
        self._test_nearest_neighbour_index(target, bounds=True, circular=True)


class Test_guess_bounds(tests.IrisTest):
    def setUp(self):
        self.coord = DimCoord(
            np.array([-160, -120, 0, 30, 150, 170]),
            units="degree",
            standard_name="longitude",
            circular=True,
        )

    def test_non_circular(self):
        self.coord.circular = False
        self.coord.guess_bounds()
        target = np.array(
            [
                [-180.0, -140.0],
                [-140.0, -60.0],
                [-60.0, 15.0],
                [15.0, 90.0],
                [90.0, 160.0],
                [160.0, 180.0],
            ]
        )
        self.assertArrayEqual(target, self.coord.bounds)

    def test_circular_increasing(self):
        self.coord.guess_bounds()
        target = np.array(
            [
                [-175.0, -140.0],
                [-140.0, -60.0],
                [-60.0, 15.0],
                [15.0, 90.0],
                [90.0, 160.0],
                [160.0, 185.0],
            ]
        )
        self.assertArrayEqual(target, self.coord.bounds)

    def test_circular_decreasing(self):
        self.coord.points = self.coord.points[::-1]
        self.coord.guess_bounds()
        target = np.array(
            [
                [185.0, 160.0],
                [160.0, 90.0],
                [90.0, 15.0],
                [15.0, -60.0],
                [-60.0, -140.0],
                [-140.0, -175.0],
            ]
        )
        self.assertArrayEqual(target, self.coord.bounds)

    def test_circular_increasing_alt_range(self):
        self.coord.points = np.array([10, 30, 90, 150, 210, 220])
        self.coord.guess_bounds()
        target = np.array(
            [
                [-65.0, 20.0],
                [20.0, 60.0],
                [60.0, 120.0],
                [120.0, 180.0],
                [180.0, 215.0],
                [215.0, 295.0],
            ]
        )
        self.assertArrayEqual(target, self.coord.bounds)

    def test_circular_decreasing_alt_range(self):
        self.coord.points = np.array([10, 30, 90, 150, 210, 220])[::-1]
        self.coord.guess_bounds()
        target = np.array(
            [
                [295, 215],
                [215, 180],
                [180, 120],
                [120, 60],
                [60, 20],
                [20, -65],
            ]
        )
        self.assertArrayEqual(target, self.coord.bounds)


class Test_guess_bounds__default_enabled_latitude_clipping(tests.IrisTest):
    def test_all_inside(self):
        lat = DimCoord([-10, 0, 20], units="degree", standard_name="latitude")
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-15, -5], [-5, 10], [10, 30]])

    def test_points_inside_bounds_outside(self):
        lat = DimCoord([-80, 0, 70], units="degree", standard_name="latitude")
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-90, -40], [-40, 35], [35, 90]])

    def test_points_inside_bounds_outside_grid_latitude(self):
        lat = DimCoord(
            [-80, 0, 70], units="degree", standard_name="grid_latitude"
        )
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-90, -40], [-40, 35], [35, 90]])

    def test_points_to_edges_bounds_outside(self):
        lat = DimCoord([-90, 0, 90], units="degree", standard_name="latitude")
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-90, -45], [-45, 45], [45, 90]])

    def test_points_outside(self):
        lat = DimCoord(
            [-100, 0, 120], units="degree", standard_name="latitude"
        )
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-150, -50], [-50, 60], [60, 180]])

    def test_points_inside_bounds_outside_wrong_unit(self):
        lat = DimCoord([-80, 0, 70], units="feet", standard_name="latitude")
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-120, -40], [-40, 35], [35, 105]])

    def test_points_inside_bounds_outside_wrong_name(self):
        lat = DimCoord([-80, 0, 70], units="degree", standard_name="longitude")
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-120, -40], [-40, 35], [35, 105]])

    def test_points_inside_bounds_outside_wrong_name_2(self):
        lat = DimCoord(
            [-80, 0, 70], units="degree", long_name="other_latitude"
        )
        lat.guess_bounds()
        self.assertArrayEqual(lat.bounds, [[-120, -40], [-40, 35], [35, 105]])


class Test_cell(tests.IrisTest):
    def _mock_coord(self):
        coord = mock.Mock(
            spec=Coord,
            ndim=1,
            points=np.array([mock.sentinel.time]),
            bounds=np.array([[mock.sentinel.lower, mock.sentinel.upper]]),
        )
        return coord

    def test_time_as_object(self):
        # Ensure Coord.cell() converts the point/bound values to
        # "datetime" objects.
        coord = self._mock_coord()
        coord.units.num2date = mock.Mock(
            side_effect=[
                mock.sentinel.datetime,
                (mock.sentinel.datetime_lower, mock.sentinel.datetime_upper),
            ]
        )
        cell = Coord.cell(coord, 0)
        self.assertIs(cell.point, mock.sentinel.datetime)
        self.assertEqual(
            cell.bound,
            (mock.sentinel.datetime_lower, mock.sentinel.datetime_upper),
        )
        self.assertEqual(
            coord.units.num2date.call_args_list,
            [
                mock.call((mock.sentinel.time,)),
                mock.call((mock.sentinel.lower, mock.sentinel.upper)),
            ],
        )


class Test_collapsed(tests.IrisTest, CoordTestMixin):
    def test_serialize(self):
        # Collapse a string AuxCoord, causing it to be serialised.
        string = Pair(
            np.array(["two", "four", "six", "eight"]),
            np.array(
                [
                    ["one", "three"],
                    ["three", "five"],
                    ["five", "seven"],
                    ["seven", "nine"],
                ]
            ),
        )
        string_nobounds = Pair(np.array(["ecks", "why", "zed"]), None)
        string_multi = Pair(
            np.array(["three", "six", "nine"]),
            np.array(
                [
                    ["one", "two", "four", "five"],
                    ["four", "five", "seven", "eight"],
                    ["seven", "eight", "ten", "eleven"],
                ]
            ),
        )

        def _serialize(data):
            return "|".join(str(item) for item in data.flatten())

        for units in ["unknown", "no_unit"]:
            for points, bounds in [string, string_nobounds, string_multi]:
                coord = AuxCoord(points=points, bounds=bounds, units=units)
                collapsed_coord = coord.collapsed()
                self.assertArrayEqual(
                    collapsed_coord.points, _serialize(points)
                )
                if bounds is not None:
                    for index in np.ndindex(bounds.shape[1:]):
                        index_slice = (slice(None),) + tuple(index)
                        self.assertArrayEqual(
                            collapsed_coord.bounds[index_slice],
                            _serialize(bounds[index_slice]),
                        )

    def test_dim_1d(self):
        # Numeric coords should not be serialised.
        coord = DimCoord(
            points=np.array([2, 4, 6, 8]),
            bounds=np.array([[1, 3], [3, 5], [5, 7], [7, 9]]),
        )
        for units in ["unknown", "no_unit", 1, "K"]:
            coord.units = units
            with self.assertNoWarningsRegexp():
                collapsed_coord = coord.collapsed()
            self.assertArrayEqual(
                collapsed_coord.points, np.mean(coord.points)
            )
            self.assertArrayEqual(
                collapsed_coord.bounds,
                [[coord.bounds.min(), coord.bounds.max()]],
            )

    def test_lazy_points(self):
        # Lazy points should stay lazy after collapse.
        coord = AuxCoord(points=da.from_array(np.arange(5), chunks=5))
        collapsed_coord = coord.collapsed()
        self.assertTrue(collapsed_coord.has_lazy_bounds())
        self.assertTrue(collapsed_coord.has_lazy_points())

    def test_numeric_nd(self):
        coord = AuxCoord(
            points=np.array([[1, 2, 4, 5], [4, 5, 7, 8], [7, 8, 10, 11]])
        )

        collapsed_coord = coord.collapsed()
        self.assertArrayEqual(collapsed_coord.points, np.array([6]))
        self.assertArrayEqual(collapsed_coord.bounds, np.array([[1, 11]]))

        # Test partially collapsing one dimension...
        collapsed_coord = coord.collapsed(1)
        self.assertArrayEqual(
            collapsed_coord.points, np.array([3.0, 6.0, 9.0])
        )
        self.assertArrayEqual(
            collapsed_coord.bounds, np.array([[1, 5], [4, 8], [7, 11]])
        )

        # ... and the other
        collapsed_coord = coord.collapsed(0)
        self.assertArrayEqual(collapsed_coord.points, np.array([4, 5, 7, 8]))
        self.assertArrayEqual(
            collapsed_coord.bounds,
            np.array([[1, 7], [2, 8], [4, 10], [5, 11]]),
        )

    def test_numeric_nd_bounds_all(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)

        collapsed_coord = coord.collapsed()
        self.assertArrayEqual(collapsed_coord.points, np.array([55]))
        self.assertArrayEqual(collapsed_coord.bounds, np.array([[-2, 112]]))

    def test_numeric_nd_bounds_second(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        collapsed_coord = coord.collapsed(1)
        self.assertArrayEqual(collapsed_coord.points, np.array([15, 55, 95]))
        self.assertArrayEqual(
            collapsed_coord.bounds, np.array([[-2, 32], [38, 72], [78, 112]])
        )

    def test_numeric_nd_bounds_first(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        # ... and the other..
        collapsed_coord = coord.collapsed(0)
        self.assertArrayEqual(
            collapsed_coord.points, np.array([40, 50, 60, 70])
        )
        self.assertArrayEqual(
            collapsed_coord.bounds,
            np.array([[-2, 82], [8, 92], [18, 102], [28, 112]]),
        )

    def test_numeric_nd_bounds_last(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        # ... and again with -ve dimension specification.
        collapsed_coord = coord.collapsed(-1)
        self.assertArrayEqual(collapsed_coord.points, np.array([15, 55, 95]))
        self.assertArrayEqual(
            collapsed_coord.bounds, np.array([[-2, 32], [38, 72], [78, 112]])
        )

    def test_lazy_nd_bounds_all(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)

        collapsed_coord = coord.collapsed()

        # Note that the new points get recalculated from the lazy bounds
        #  and so end up as lazy
        self.assertTrue(collapsed_coord.has_lazy_points())
        self.assertTrue(collapsed_coord.has_lazy_bounds())

        self.assertArrayEqual(collapsed_coord.points, np.array([55]))
        self.assertArrayEqual(collapsed_coord.bounds, da.array([[-2, 112]]))

    def test_lazy_nd_bounds_second(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)

        collapsed_coord = coord.collapsed(1)
        self.assertArrayEqual(collapsed_coord.points, np.array([15, 55, 95]))
        self.assertArrayEqual(
            collapsed_coord.bounds, np.array([[-2, 32], [38, 72], [78, 112]])
        )

    def test_lazy_nd_bounds_first(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)

        collapsed_coord = coord.collapsed(0)
        self.assertArrayEqual(
            collapsed_coord.points, np.array([40, 50, 60, 70])
        )
        self.assertArrayEqual(
            collapsed_coord.bounds,
            np.array([[-2, 82], [8, 92], [18, 102], [28, 112]]),
        )

    def test_lazy_nd_bounds_last(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)

        collapsed_coord = coord.collapsed(-1)
        self.assertArrayEqual(collapsed_coord.points, np.array([15, 55, 95]))
        self.assertArrayEqual(
            collapsed_coord.bounds, np.array([[-2, 32], [38, 72], [78, 112]])
        )

    def test_lazy_nd_points_and_bounds(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_lazy)

        collapsed_coord = coord.collapsed()

        self.assertTrue(collapsed_coord.has_lazy_points())
        self.assertTrue(collapsed_coord.has_lazy_bounds())

        self.assertArrayEqual(collapsed_coord.points, da.array([55]))
        self.assertArrayEqual(collapsed_coord.bounds, da.array([[-2, 112]]))

    def test_numeric_nd_multidim_bounds_warning(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_real, bounds=self.bds_real, long_name="y")

        msg = (
            "Collapsing a multi-dimensional coordinate. "
            "Metadata may not be fully descriptive for 'y'."
        )
        with self.assertWarnsRegex(UserWarning, msg):
            coord.collapsed()

    def test_lazy_nd_multidim_bounds_warning(self):
        self.setupTestArrays((3, 4))
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_lazy, long_name="y")

        msg = (
            "Collapsing a multi-dimensional coordinate. "
            "Metadata may not be fully descriptive for 'y'."
        )
        with self.assertWarnsRegex(UserWarning, msg):
            coord.collapsed()

    def test_numeric_nd_noncontiguous_bounds_warning(self):
        self.setupTestArrays((3))
        coord = AuxCoord(self.pts_real, bounds=self.bds_real, long_name="y")

        msg = (
            "Collapsing a non-contiguous coordinate. "
            "Metadata may not be fully descriptive for 'y'."
        )
        with self.assertWarnsRegex(UserWarning, msg):
            coord.collapsed()

    def test_lazy_nd_noncontiguous_bounds_warning(self):
        self.setupTestArrays((3))
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_lazy, long_name="y")

        msg = (
            "Collapsing a non-contiguous coordinate. "
            "Metadata may not be fully descriptive for 'y'."
        )
        with self.assertWarnsRegex(UserWarning, msg):
            coord.collapsed()

    def test_numeric_3_bounds(self):
        points = np.array([2.0, 6.0, 4.0])
        bounds = np.array([[1.0, 0.0, 3.0], [5.0, 4.0, 7.0], [3.0, 2.0, 5.0]])

        coord = AuxCoord(points, bounds=bounds, long_name="x")

        msg = (
            r"Cannot check if coordinate is contiguous: Invalid operation for "
            r"'x', with 3 bound\(s\). Contiguous bounds are only defined for "
            r"1D coordinates with 2 bounds. Metadata may not be fully "
            r"descriptive for 'x'. Ignoring bounds."
        )
        with self.assertWarnsRegex(UserWarning, msg):
            collapsed_coord = coord.collapsed()

        self.assertFalse(collapsed_coord.has_lazy_points())
        self.assertFalse(collapsed_coord.has_lazy_bounds())

        self.assertArrayAlmostEqual(collapsed_coord.points, np.array([4.0]))
        self.assertArrayAlmostEqual(
            collapsed_coord.bounds, np.array([[2.0, 6.0]])
        )

    def test_lazy_3_bounds(self):
        points = da.arange(3) * 2.0
        bounds = da.arange(3 * 3).reshape(3, 3)

        coord = AuxCoord(points, bounds=bounds, long_name="x")

        msg = (
            r"Cannot check if coordinate is contiguous: Invalid operation for "
            r"'x', with 3 bound\(s\). Contiguous bounds are only defined for "
            r"1D coordinates with 2 bounds. Metadata may not be fully "
            r"descriptive for 'x'. Ignoring bounds."
        )
        with self.assertWarnsRegex(UserWarning, msg):
            collapsed_coord = coord.collapsed()

        self.assertTrue(collapsed_coord.has_lazy_points())
        self.assertTrue(collapsed_coord.has_lazy_bounds())

        self.assertArrayAlmostEqual(collapsed_coord.points, da.array([2.0]))
        self.assertArrayAlmostEqual(
            collapsed_coord.bounds, da.array([[0.0, 4.0]])
        )


class Test_is_compatible(tests.IrisTest):
    def setUp(self):
        self.test_coord = AuxCoord([1.0])
        self.other_coord = self.test_coord.copy()

    def test_noncommon_array_attrs_compatible(self):
        # Non-common array attributes should be ok.
        self.test_coord.attributes["array_test"] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_coord.is_compatible(self.other_coord))

    def test_matching_array_attrs_compatible(self):
        # Matching array attributes should be ok.
        self.test_coord.attributes["array_test"] = np.array([1.0, 2, 3])
        self.other_coord.attributes["array_test"] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_coord.is_compatible(self.other_coord))

    def test_different_array_attrs_incompatible(self):
        # Differing array attributes should make coords incompatible.
        self.test_coord.attributes["array_test"] = np.array([1.0, 2, 3])
        self.other_coord.attributes["array_test"] = np.array([1.0, 2, 777.7])
        self.assertFalse(self.test_coord.is_compatible(self.other_coord))


class Test_contiguous_bounds(tests.IrisTest):
    def test_1d_coord_no_bounds_warning(self):
        coord = DimCoord([0, 1, 2], standard_name="latitude")
        msg = (
            "Coordinate 'latitude' is not bounded, guessing contiguous "
            "bounds."
        )
        with warnings.catch_warnings():
            # Cause all warnings to raise Exceptions
            warnings.simplefilter("error")
            with self.assertRaisesRegex(Warning, msg):
                coord.contiguous_bounds()

    def test_2d_coord_no_bounds_error(self):
        coord = AuxCoord(np.array([[0, 0], [5, 5]]), standard_name="latitude")
        emsg = "Guessing bounds of 2D coords is not currently supported"
        with self.assertRaisesRegex(ValueError, emsg):
            coord.contiguous_bounds()

    def test__sanity_check_bounds_call(self):
        coord = DimCoord([5, 15, 25], bounds=[[0, 10], [10, 20], [20, 30]])
        with mock.patch(
            "iris.coords.Coord._sanity_check_bounds"
        ) as bounds_check:
            coord.contiguous_bounds()
        bounds_check.assert_called_once()

    def test_1d_coord(self):
        coord = DimCoord(
            [2, 4, 6],
            standard_name="latitude",
            bounds=[[1, 3], [3, 5], [5, 7]],
        )
        expected = np.array([1, 3, 5, 7])
        result = coord.contiguous_bounds()
        self.assertArrayEqual(result, expected)

    def test_1d_coord_discontiguous(self):
        coord = DimCoord(
            [2, 4, 6],
            standard_name="latitude",
            bounds=[[1, 3], [4, 5], [5, 7]],
        )
        expected = np.array([1, 4, 5, 7])
        result = coord.contiguous_bounds()
        self.assertArrayEqual(result, expected)

    def test_2d_lon_bounds(self):
        coord = AuxCoord(
            np.array([[1, 3], [1, 3]]),
            bounds=np.array(
                [[[0, 2, 2, 0], [2, 4, 4, 2]], [[0, 2, 2, 0], [2, 4, 4, 2]]]
            ),
        )
        expected = np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]])
        result = coord.contiguous_bounds()
        self.assertArrayEqual(result, expected)

    def test_2d_lat_bounds(self):
        coord = AuxCoord(
            np.array([[1, 1], [3, 3]]),
            bounds=np.array(
                [[[0, 0, 2, 2], [0, 0, 2, 2]], [[2, 2, 4, 4], [2, 2, 4, 4]]]
            ),
        )
        expected = np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]])
        result = coord.contiguous_bounds()
        self.assertArrayEqual(result, expected)


class Test_is_contiguous(tests.IrisTest):
    def test_no_bounds(self):
        coord = DimCoord([1, 3])
        result = coord.is_contiguous()
        self.assertFalse(result)

    def test__discontiguity_in_bounds_call(self):
        # Check that :meth:`iris.coords.Coord._discontiguity_in_bounds` is
        # called.
        coord = DimCoord([1, 3], bounds=[[0, 2], [2, 4]])
        with mock.patch(
            "iris.coords.Coord._discontiguity_in_bounds"
        ) as discontiguity_check:
            # Discontiguity returns two objects that are unpacked in
            # `coord.is_contiguous`.
            discontiguity_check.return_value = [None, None]
            coord.is_contiguous(rtol=1e-1, atol=1e-3)
        discontiguity_check.assert_called_with(rtol=1e-1, atol=1e-3)


class Test__discontiguity_in_bounds(tests.IrisTest):
    def setUp(self):
        self.points_3by3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.lon_bounds_3by3 = np.array(
            [
                [[0, 2, 2, 0], [2, 4, 4, 2], [4, 6, 6, 4]],
                [[0, 2, 2, 0], [2, 4, 4, 2], [4, 6, 6, 4]],
                [[0, 2, 2, 0], [2, 4, 4, 2], [4, 6, 6, 4]],
            ]
        )
        self.lat_bounds_3by3 = np.array(
            [
                [[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]],
                [[2, 2, 4, 4], [2, 2, 4, 4], [2, 2, 4, 4]],
                [[4, 4, 6, 6], [4, 4, 6, 6], [4, 4, 6, 6]],
            ]
        )

    def test_1d_contiguous(self):
        coord = DimCoord(
            [-20, 0, 20], bounds=[[-30, -10], [-10, 10], [10, 30]]
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        self.assertTrue(contiguous)
        self.assertArrayEqual(diffs, np.zeros(2))

    def test_1d_discontiguous(self):
        coord = DimCoord([10, 20, 40], bounds=[[5, 15], [15, 25], [35, 45]])
        contiguous, diffs = coord._discontiguity_in_bounds()
        self.assertFalse(contiguous)
        self.assertArrayEqual(diffs, np.array([False, True]))

    def test_1d_one_cell(self):
        # Test a 1D coord with a single cell.
        coord = DimCoord(20, bounds=[[10, 30]])
        contiguous, diffs = coord._discontiguity_in_bounds()
        self.assertTrue(contiguous)
        self.assertArrayEqual(diffs, np.array([]))

    def test_2d_contiguous_both_dirs(self):
        coord = AuxCoord(self.points_3by3, bounds=self.lon_bounds_3by3)
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        self.assertTrue(contiguous)
        self.assertTrue(not diffs_along_x.any())
        self.assertTrue(not diffs_along_y.any())

    def test_2d_discontiguous_along_x(self):
        coord = AuxCoord(
            self.points_3by3[:, ::2], bounds=self.lon_bounds_3by3[:, ::2, :]
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        self.assertFalse(contiguous)
        self.assertArrayEqual(
            diffs_along_x, np.array([True, True, True]).reshape(3, 1)
        )
        self.assertTrue(not diffs_along_y.any())

    def test_2d_discontiguous_along_y(self):
        coord = AuxCoord(
            self.points_3by3[::2, :], bounds=self.lat_bounds_3by3[::2, :, :]
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        self.assertFalse(contiguous)
        self.assertTrue(not diffs_along_x.any())
        self.assertArrayEqual(diffs_along_y, np.array([[True, True, True]]))

    def test_2d_discontiguous_along_x_and_y(self):
        coord = AuxCoord(
            np.array([[1, 5], [3, 5]]),
            bounds=np.array(
                [[[0, 2, 2, 0], [4, 6, 6, 4]], [[2, 4, 4, 2], [4, 6, 6, 4]]]
            ),
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        exp_x_diffs = np.array([True, False]).reshape(2, 1)
        exp_y_diffs = np.array([True, False]).reshape(1, 2)
        self.assertFalse(contiguous)
        self.assertArrayEqual(diffs_along_x, exp_x_diffs)
        self.assertArrayEqual(diffs_along_y, exp_y_diffs)

    def test_2d_contiguous_along_x_atol(self):
        coord = AuxCoord(
            self.points_3by3[:, ::2], bounds=self.lon_bounds_3by3[:, ::2, :]
        )
        # Set a high atol that allows small discontiguities.
        contiguous, diffs = coord._discontiguity_in_bounds(atol=5)
        diffs_along_x, diffs_along_y = diffs
        self.assertTrue(contiguous)
        self.assertArrayEqual(
            diffs_along_x, np.array([False, False, False]).reshape(3, 1)
        )
        self.assertTrue(not diffs_along_y.any())

    def test_2d_one_cell(self):
        # Test a 2D coord with a single cell, where the coord has shape (1, 1).
        coord = AuxCoord(
            self.points_3by3[:1, :1], bounds=self.lon_bounds_3by3[:1, :1, :]
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        expected_diffs = np.array([], dtype=np.int64)
        self.assertTrue(contiguous)
        self.assertArrayEqual(diffs_along_x, expected_diffs.reshape(1, 0))
        self.assertArrayEqual(diffs_along_y, expected_diffs.reshape(0, 1))

    def test_2d_one_cell_along_x(self):
        # Test a 2D coord with a single cell along the x axis, where the coord
        # has shape (2, 1).
        coord = AuxCoord(
            self.points_3by3[:, :1], bounds=self.lat_bounds_3by3[:, :1, :]
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        self.assertTrue(contiguous)
        self.assertTrue(not diffs_along_x.any())
        self.assertArrayEqual(diffs_along_y, np.array([0, 0]).reshape(2, 1))

    def test_2d_one_cell_along_y(self):
        # Test a 2D coord with a single cell along the y axis, where the coord
        # has shape (1, 2).
        coord = AuxCoord(
            self.points_3by3[:1, :], bounds=self.lon_bounds_3by3[:1, :, :]
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        self.assertTrue(contiguous)
        self.assertTrue(not diffs_along_x.any())
        self.assertTrue(not diffs_along_y.any())

    def test_2d_contiguous_mod_360(self):
        # Test that longitude coordinates are adjusted by the 360 modulus when
        # calculating the discontiguities in contiguous bounds.
        coord = AuxCoord(
            [[175, -175], [175, -175]],
            standard_name="longitude",
            bounds=np.array(
                [
                    [[170, 180, 180, 170], [-180, -170, -170, -180]],
                    [[170, 180, 180, 170], [-180, -170, -170, -180]],
                ]
            ),
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        self.assertTrue(contiguous)
        self.assertTrue(not diffs_along_x.any())
        self.assertTrue(not diffs_along_y.any())

    def test_2d_discontiguous_mod_360(self):
        # Test that longitude coordinates are adjusted by the 360 modulus when
        # calculating the discontiguities in contiguous bounds.
        coord = AuxCoord(
            [[175, -175], [175, -175]],
            standard_name="longitude",
            bounds=np.array(
                [
                    [[170, 180, 180, 170], [10, 20, 20, 10]],
                    [[170, 180, 180, 170], [10, 20, 20, 10]],
                ]
            ),
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        self.assertFalse(contiguous)
        self.assertArrayEqual(diffs_along_x, np.array([[True], [True]]))
        self.assertTrue(not diffs_along_y.any())

    def test_2d_contiguous_mod_360_not_longitude(self):
        # Test that non-longitude coordinates are not adjusted by the 360
        # modulus when calculating the discontiguities in contiguous bounds.
        coord = AuxCoord(
            [[-150, 350], [-150, 350]],
            standard_name="height",
            bounds=np.array(
                [
                    [[-400, 100, 100, -400], [100, 600, 600, 100]],
                    [[-400, 100, 100, -400], [100, 600, 600, 100]],
                ]
            ),
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        self.assertTrue(contiguous)
        self.assertTrue(not diffs_along_x.any())
        self.assertTrue(not diffs_along_y.any())

    def test_2d_discontiguous_mod_360_not_longitude(self):
        # Test that non-longitude coordinates are not adjusted by the 360
        # modulus when calculating the discontiguities in discontiguous bounds.
        coord = AuxCoord(
            [[-150, 350], [-150, 350]],
            standard_name="height",
            bounds=np.array(
                [
                    [[-400, 100, 100, -400], [200, 600, 600, 200]],
                    [[-400, 100, 100, -400], [200, 600, 600, 200]],
                ]
            ),
        )
        contiguous, diffs = coord._discontiguity_in_bounds()
        diffs_along_x, diffs_along_y = diffs
        self.assertFalse(contiguous)
        self.assertArrayEqual(diffs_along_x, np.array([[True], [True]]))
        self.assertTrue(not diffs_along_y.any())


class Test__sanity_check_bounds(tests.IrisTest):
    def test_coord_1d_2_bounds(self):
        # Check that a 1d coord with 2 bounds does not raise an error.
        coord = iris.coords.DimCoord(
            [0, 1], standard_name="latitude", bounds=[[0, 1], [1, 2]]
        )
        coord._sanity_check_bounds()

    def test_coord_1d_no_bounds(self):
        coord = iris.coords.DimCoord([0, 1], standard_name="latitude")
        emsg = (
            "Contiguous bounds are only defined for 1D coordinates with "
            "2 bounds."
        )
        with self.assertRaisesRegex(ValueError, emsg):
            coord._sanity_check_bounds()

    def test_coord_1d_1_bounds(self):
        coord = iris.coords.DimCoord(
            [0, 1], standard_name="latitude", bounds=np.array([[0], [1]])
        )
        emsg = (
            "Contiguous bounds are only defined for 1D coordinates with "
            "2 bounds."
        )
        with self.assertRaisesRegex(ValueError, emsg):
            coord._sanity_check_bounds()

    def test_coord_2d_4_bounds(self):
        coord = iris.coords.AuxCoord(
            [[0, 0], [1, 1]],
            standard_name="latitude",
            bounds=np.array(
                [[[0, 0, 1, 1], [0, 0, 1, 1]], [[1, 1, 2, 2], [1, 1, 2, 2]]]
            ),
        )
        coord._sanity_check_bounds()

    def test_coord_2d_no_bounds(self):
        coord = iris.coords.AuxCoord(
            [[0, 0], [1, 1]], standard_name="latitude"
        )
        emsg = (
            "Contiguous bounds are only defined for 2D coordinates with "
            "4 bounds."
        )
        with self.assertRaisesRegex(ValueError, emsg):
            coord._sanity_check_bounds()

    def test_coord_2d_2_bounds(self):
        coord = iris.coords.AuxCoord(
            [[0, 0], [1, 1]],
            standard_name="latitude",
            bounds=np.array([[[0, 1], [0, 1]], [[1, 2], [1, 2]]]),
        )
        emsg = (
            "Contiguous bounds are only defined for 2D coordinates with "
            "4 bounds."
        )
        with self.assertRaisesRegex(ValueError, emsg):
            coord._sanity_check_bounds()

    def test_coord_3d(self):
        coord = iris.coords.AuxCoord(
            np.zeros((2, 2, 2)), standard_name="height"
        )
        emsg = (
            "Contiguous bounds are not defined for coordinates with more "
            "than 2 dimensions."
        )
        with self.assertRaisesRegex(ValueError, emsg):
            coord._sanity_check_bounds()


class Test_convert_units(tests.IrisTest):
    def test_convert_unknown_units(self):
        coord = iris.coords.AuxCoord(1, units="unknown")
        emsg = (
            "Cannot convert from unknown units. "
            'The "units" attribute may be set directly.'
        )
        with self.assertRaisesRegex(UnitConversionError, emsg):
            coord.convert_units("degrees")


class Test___str__(tests.IrisTest):
    def test_short_time_interval(self):
        coord = DimCoord(
            [5], standard_name="time", units="days since 1970-01-01"
        )
        expected = "\n".join(
            [
                "DimCoord :  time / (days since 1970-01-01, standard calendar)",
                "    points: [1970-01-06 00:00:00]",
                "    shape: (1,)",
                "    dtype: int64",
                "    standard_name: 'time'",
            ]
        )
        result = coord.__str__()
        self.assertEqual(expected, result)

    def test_short_time_interval__bounded(self):
        coord = DimCoord(
            [5, 6], standard_name="time", units="days since 1970-01-01"
        )
        coord.guess_bounds()
        expected = "\n".join(
            [
                "DimCoord :  time / (days since 1970-01-01, standard calendar)",
                "    points: [1970-01-06 00:00:00, 1970-01-07 00:00:00]",
                "    bounds: [",
                "        [1970-01-05 12:00:00, 1970-01-06 12:00:00],",
                "        [1970-01-06 12:00:00, 1970-01-07 12:00:00]]",
                "    shape: (2,)  bounds(2, 2)",
                "    dtype: int64",
                "    standard_name: 'time'",
            ]
        )
        result = coord.__str__()
        self.assertEqual(expected, result)

    def test_long_time_interval(self):
        coord = DimCoord(
            [5], standard_name="time", units="years since 1970-01-01"
        )
        expected = "\n".join(
            [
                "DimCoord :  time / (years since 1970-01-01, standard calendar)",
                "    points: [5]",
                "    shape: (1,)",
                "    dtype: int64",
                "    standard_name: 'time'",
            ]
        )
        result = coord.__str__()
        self.assertEqual(expected, result)

    def test_long_time_interval__bounded(self):
        coord = DimCoord(
            [5, 6], standard_name="time", units="years since 1970-01-01"
        )
        coord.guess_bounds()
        expected = "\n".join(
            [
                "DimCoord :  time / (years since 1970-01-01, standard calendar)",
                "    points: [5, 6]",
                "    bounds: [",
                "        [4.5, 5.5],",
                "        [5.5, 6.5]]",
                "    shape: (2,)  bounds(2, 2)",
                "    dtype: int64",
                "    standard_name: 'time'",
            ]
        )
        result = coord.__str__()
        self.assertEqual(expected, result)

    def test_non_time_unit(self):
        coord = DimCoord([1.0])
        expected = "\n".join(
            [
                "DimCoord :  unknown / (unknown)",
                "    points: [1.]",
                "    shape: (1,)",
                "    dtype: float64",
            ]
        )
        result = coord.__str__()
        self.assertEqual(expected, result)


class TestClimatology(tests.IrisTest):
    # Variety of tests for the climatological property of a coord.
    # Only using AuxCoord since there is no different behaviour between Aux
    # and DimCoords for this property.

    def test_create(self):
        coord = AuxCoord(
            points=[0, 1],
            bounds=[[0, 1], [1, 2]],
            units="days since 1970-01-01",
            climatological=True,
        )
        self.assertTrue(coord.climatological)

    def test_create_no_bounds_no_set(self):
        with self.assertRaisesRegex(ValueError, "Cannot set.*no bounds exist"):
            AuxCoord(
                points=[0, 1],
                units="days since 1970-01-01",
                climatological=True,
            )

    def test_create_no_time_no_set(self):
        emsg = "Cannot set climatological .* valid time reference units.*"
        with self.assertRaisesRegex(TypeError, emsg):
            AuxCoord(
                points=[0, 1], bounds=[[0, 1], [1, 2]], climatological=True
            )

    def test_absent(self):
        coord = AuxCoord(points=[0, 1], bounds=[[0, 1], [1, 2]])
        self.assertFalse(coord.climatological)

    def test_absent_no_bounds_no_set(self):
        coord = AuxCoord(points=[0, 1], units="days since 1970-01-01")
        with self.assertRaisesRegex(ValueError, "Cannot set.*no bounds exist"):
            coord.climatological = True

    def test_absent_no_time_no_set(self):
        coord = AuxCoord(points=[0, 1], bounds=[[0, 1], [1, 2]])
        emsg = "Cannot set climatological .* valid time reference units.*"
        with self.assertRaisesRegex(TypeError, emsg):
            coord.climatological = True

    def test_absent_no_bounds_unset(self):
        coord = AuxCoord(points=[0, 1])
        coord.climatological = False
        self.assertFalse(coord.climatological)

    def test_bounds_set(self):
        coord = AuxCoord(
            points=[0, 1],
            bounds=[[0, 1], [1, 2]],
            units="days since 1970-01-01",
        )
        coord.climatological = True
        self.assertTrue(coord.climatological)

    def test_bounds_unset(self):
        coord = AuxCoord(
            points=[0, 1],
            bounds=[[0, 1], [1, 2]],
            units="days since 1970-01-01",
            climatological=True,
        )
        coord.climatological = False
        self.assertFalse(coord.climatological)

    def test_remove_bounds(self):
        coord = AuxCoord(
            points=[0, 1],
            bounds=[[0, 1], [1, 2]],
            units="days since 1970-01-01",
            climatological=True,
        )
        coord.bounds = None
        self.assertFalse(coord.climatological)

    def test_change_units(self):
        coord = AuxCoord(
            points=[0, 1],
            bounds=[[0, 1], [1, 2]],
            units="days since 1970-01-01",
            climatological=True,
        )
        self.assertTrue(coord.climatological)
        coord.units = "K"
        self.assertFalse(coord.climatological)


class Test___init____abstractmethod(tests.IrisTest):
    def test(self):
        emsg = (
            "Can't instantiate abstract class Coord with abstract"
            " method.* __init__"
        )
        with self.assertRaisesRegex(TypeError, emsg):
            _ = Coord(points=[0, 1])


class Test_cube_dims(tests.IrisTest):
    def test(self):
        # Check that "coord.cube_dims(cube)" calls "cube.coord_dims(coord)".
        mock_dims_result = mock.sentinel.COORD_DIMS
        mock_dims_call = mock.Mock(return_value=mock_dims_result)
        mock_cube = mock.Mock(Cube, coord_dims=mock_dims_call)
        test_coord = AuxCoord([1], long_name="test_name")

        result = test_coord.cube_dims(mock_cube)
        self.assertEqual(result, mock_dims_result)
        self.assertEqual(
            mock_dims_call.call_args_list, [mock.call(test_coord)]
        )


if __name__ == "__main__":
    tests.main()
