# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.cube.Cube` class aggregated_by method."""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from cf_units import Unit
import numpy as np

from iris._lazy_data import as_lazy_data
import iris.analysis
from iris.analysis import MEAN, SUM, Aggregator, WeightedAggregator
import iris.aux_factory
import iris.coords
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
from iris.cube import Cube
import iris.exceptions
from iris.tests.stock import realistic_4d


class Test_aggregated_by(tests.IrisTest):
    def setUp(self):
        self.cube = Cube(np.arange(44).reshape(4, 11))

        val_coord = AuxCoord(
            [0, 0, 0, 1, 1, 2, 0, 0, 2, 0, 1], long_name="val"
        )
        label_coord = AuxCoord(
            [
                "alpha",
                "alpha",
                "beta",
                "beta",
                "alpha",
                "gamma",
                "alpha",
                "alpha",
                "alpha",
                "gamma",
                "beta",
            ],
            long_name="label",
            units="no_unit",
        )
        simple_agg_coord = AuxCoord([1, 1, 2, 2], long_name="simple_agg")
        spanning_coord = AuxCoord(
            np.arange(44).reshape(4, 11), long_name="spanning"
        )
        spanning_label_coord = AuxCoord(
            np.arange(1, 441, 10).reshape(4, 11).astype(str),
            long_name="span_label",
            units="no_unit",
        )

        self.cube.add_aux_coord(simple_agg_coord, 0)
        self.cube.add_aux_coord(val_coord, 1)
        self.cube.add_aux_coord(label_coord, 1)
        self.cube.add_aux_coord(spanning_coord, (0, 1))
        self.cube.add_aux_coord(spanning_label_coord, (0, 1))

        self.mock_agg = mock.Mock(spec=Aggregator)
        self.mock_agg.cell_method = []
        self.mock_agg.aggregate = mock.Mock(
            return_value=mock.Mock(dtype="object")
        )
        self.mock_agg.aggregate_shape = mock.Mock(return_value=())
        self.mock_agg.lazy_func = None
        self.mock_agg.post_process = mock.Mock(side_effect=lambda x, y, z: x)

        self.mock_weighted_agg = mock.Mock(spec=WeightedAggregator)
        self.mock_weighted_agg.cell_method = []

        def mock_weighted_aggregate(*_, **kwargs):
            if kwargs.get("returned", False):
                return (mock.Mock(dtype="object"), mock.Mock(dtype="object"))
            return mock.Mock(dtype="object")

        self.mock_weighted_agg.aggregate = mock.Mock(
            side_effect=mock_weighted_aggregate
        )
        self.mock_weighted_agg.aggregate_shape = mock.Mock(return_value=())
        self.mock_weighted_agg.lazy_func = None
        self.mock_weighted_agg.post_process = mock.Mock(
            side_effect=lambda x, y, z, **kwargs: y
        )

        self.ancillary_variable = AncillaryVariable(
            [0, 1, 2, 3], long_name="foo"
        )
        self.cube.add_ancillary_variable(self.ancillary_variable, 0)
        self.cell_measure = CellMeasure([0, 1, 2, 3], long_name="bar")
        self.cube.add_cell_measure(self.cell_measure, 0)

        self.simple_weights = np.array([1.0, 0.0, 2.0, 2.0])
        self.val_weights = np.ones_like(self.cube.data, dtype=np.float32)

    def test_2d_coord_simple_agg(self):
        # For 2d coords, slices of aggregated coord should be the same as
        # aggregated slices.
        res_cube = self.cube.aggregated_by("simple_agg", self.mock_agg)
        for res_slice, cube_slice in zip(
            res_cube.slices("simple_agg"), self.cube.slices("simple_agg")
        ):
            cube_slice_agg = cube_slice.aggregated_by(
                "simple_agg", self.mock_agg
            )
            self.assertEqual(
                res_slice.coord("spanning"), cube_slice_agg.coord("spanning")
            )
            self.assertEqual(
                res_slice.coord("span_label"),
                cube_slice_agg.coord("span_label"),
            )

    def test_agg_by_label(self):
        # Aggregate a cube on a string coordinate label where label
        # and val entries are not in step; the resulting cube has a val
        # coord of bounded cells and a label coord of single string entries.
        res_cube = self.cube.aggregated_by("label", self.mock_agg)
        val_coord = AuxCoord(
            np.array([1.0, 0.5, 1.0]),
            bounds=np.array([[0, 2], [0, 1], [0, 2]]),
            long_name="val",
        )
        label_coord = AuxCoord(
            np.array(["alpha", "beta", "gamma"]),
            long_name="label",
            units="no_unit",
        )
        self.assertEqual(res_cube.coord("val"), val_coord)
        self.assertEqual(res_cube.coord("label"), label_coord)

    def test_agg_by_label_bounded(self):
        # Aggregate a cube on a string coordinate label where label
        # and val entries are not in step; the resulting cube has a val
        # coord of bounded cells and a label coord of single string entries.
        val_points = self.cube.coord("val").points
        self.cube.coord("val").bounds = np.array(
            [val_points - 0.5, val_points + 0.5]
        ).T
        res_cube = self.cube.aggregated_by("label", self.mock_agg)
        val_coord = AuxCoord(
            np.array([1.0, 0.5, 1.0]),
            bounds=np.array([[-0.5, 2.5], [-0.5, 1.5], [-0.5, 2.5]]),
            long_name="val",
        )
        label_coord = AuxCoord(
            np.array(["alpha", "beta", "gamma"]),
            long_name="label",
            units="no_unit",
        )
        self.assertEqual(res_cube.coord("val"), val_coord)
        self.assertEqual(res_cube.coord("label"), label_coord)

    def test_2d_agg_by_label(self):
        res_cube = self.cube.aggregated_by("label", self.mock_agg)
        # For 2d coord, slices of aggregated coord should be the same as
        # aggregated slices.
        for res_slice, cube_slice in zip(
            res_cube.slices("val"), self.cube.slices("val")
        ):
            cube_slice_agg = cube_slice.aggregated_by("label", self.mock_agg)
            self.assertEqual(
                res_slice.coord("spanning"), cube_slice_agg.coord("spanning")
            )

    def test_agg_by_val(self):
        # Aggregate a cube on a numeric coordinate val where label
        # and val entries are not in step; the resulting cube has a label
        # coord with serialised labels from the aggregated cells.
        res_cube = self.cube.aggregated_by("val", self.mock_agg)
        val_coord = AuxCoord(np.array([0, 1, 2]), long_name="val")
        exp0 = "alpha|alpha|beta|alpha|alpha|gamma"
        exp1 = "beta|alpha|beta"
        exp2 = "gamma|alpha"
        label_coord = AuxCoord(
            np.array((exp0, exp1, exp2)), long_name="label", units="no_unit"
        )
        self.assertEqual(res_cube.coord("val"), val_coord)
        self.assertEqual(res_cube.coord("label"), label_coord)

    def test_2d_agg_by_val(self):
        res_cube = self.cube.aggregated_by("val", self.mock_agg)
        # For 2d coord, slices of aggregated coord should be the same as
        # aggregated slices.
        for res_slice, cube_slice in zip(
            res_cube.slices("val"), self.cube.slices("val")
        ):
            cube_slice_agg = cube_slice.aggregated_by("val", self.mock_agg)
            self.assertEqual(
                res_slice.coord("spanning"), cube_slice_agg.coord("spanning")
            )

    def test_single_string_aggregation(self):
        aux_coords = [
            (AuxCoord(["a", "b", "a"], long_name="foo"), 0),
            (AuxCoord(["a", "a", "a"], long_name="bar"), 0),
        ]
        cube = iris.cube.Cube(
            np.arange(12).reshape(3, 4), aux_coords_and_dims=aux_coords
        )
        result = cube.aggregated_by("foo", MEAN)
        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(
            result.coord("bar"), AuxCoord(["a|a", "a"], long_name="bar")
        )

    def test_ancillary_variables_and_cell_measures_kept(self):
        cube_agg = self.cube.aggregated_by("val", self.mock_agg)
        self.assertEqual(
            cube_agg.ancillary_variables(), [self.ancillary_variable]
        )
        self.assertEqual(cube_agg.cell_measures(), [self.cell_measure])

    def test_ancillary_variables_and_cell_measures_removed(self):
        cube_agg = self.cube.aggregated_by("simple_agg", self.mock_agg)
        self.assertEqual(cube_agg.ancillary_variables(), [])
        self.assertEqual(cube_agg.cell_measures(), [])

    def test_1d_weights(self):
        self.cube.aggregated_by(
            "simple_agg", self.mock_weighted_agg, weights=self.simple_weights
        )

        self.assertEqual(self.mock_weighted_agg.aggregate.call_count, 2)

        # A simple mock.assert_called_with does not work due to ValueError: The
        # truth value of an array with more than one element is ambiguous. Use
        # a.any() or a.all()
        call_1 = self.mock_weighted_agg.aggregate.mock_calls[0]
        np.testing.assert_array_equal(
            call_1.args[0],
            np.array(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                ]
            ),
        )
        self.assertEqual(call_1.kwargs["axis"], 0)
        np.testing.assert_array_almost_equal(
            call_1.kwargs["weights"],
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        call_2 = self.mock_weighted_agg.aggregate.mock_calls[1]
        np.testing.assert_array_equal(
            call_2.args[0],
            np.array(
                [
                    [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                    [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
                ]
            ),
        )
        self.assertEqual(call_2.kwargs["axis"], 0)
        np.testing.assert_array_almost_equal(
            call_2.kwargs["weights"],
            np.array(
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ]
            ),
        )

    def test_2d_weights(self):
        self.cube.aggregated_by(
            "val", self.mock_weighted_agg, weights=self.val_weights
        )

        self.assertEqual(self.mock_weighted_agg.aggregate.call_count, 3)

        # A simple mock.assert_called_with does not work due to ValueError: The
        # truth value of an array with more than one element is ambiguous. Use
        # a.any() or a.all()
        call_1 = self.mock_weighted_agg.aggregate.mock_calls[0]
        np.testing.assert_array_equal(
            call_1.args[0],
            np.array(
                [
                    [0, 1, 2, 6, 7, 9],
                    [11, 12, 13, 17, 18, 20],
                    [22, 23, 24, 28, 29, 31],
                    [33, 34, 35, 39, 40, 42],
                ]
            ),
        )
        self.assertEqual(call_1.kwargs["axis"], 1)
        np.testing.assert_array_almost_equal(
            call_1.kwargs["weights"], np.ones((4, 6))
        )

        call_2 = self.mock_weighted_agg.aggregate.mock_calls[1]
        np.testing.assert_array_equal(
            call_2.args[0],
            np.array([[3, 4, 10], [14, 15, 21], [25, 26, 32], [36, 37, 43]]),
        )
        self.assertEqual(call_2.kwargs["axis"], 1)
        np.testing.assert_array_almost_equal(
            call_2.kwargs["weights"], np.ones((4, 3))
        )

        call_3 = self.mock_weighted_agg.aggregate.mock_calls[2]
        np.testing.assert_array_equal(
            call_3.args[0], np.array([[5, 8], [16, 19], [27, 30], [38, 41]])
        )
        self.assertEqual(call_3.kwargs["axis"], 1)
        np.testing.assert_array_almost_equal(
            call_3.kwargs["weights"], np.ones((4, 2))
        )

    def test_returned(self):
        output = self.cube.aggregated_by(
            "simple_agg", self.mock_weighted_agg, returned=True
        )

        self.assertTrue(isinstance(output, tuple))
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape, (2, 11))
        self.assertEqual(output[1].shape, (2, 11))

    def test_fail_1d_weights_wrong_len(self):
        wrong_weights = np.array([1.0, 2.0])
        msg = (
            r"1D weights must have the same length as the dimension that is "
            r"aggregated, got 2, expected 11"
        )
        with self.assertRaisesRegex(ValueError, msg):
            self.cube.aggregated_by(
                "val", self.mock_weighted_agg, weights=wrong_weights
            )

    def test_fail_weights_wrong_shape(self):
        wrong_weights = np.ones((42, 1))
        msg = (
            r"Weights must either be 1D or have the same shape as the cube, "
            r"got shape \(42, 1\) for weights, \(4, 11\) for cube"
        )
        with self.assertRaisesRegex(ValueError, msg):
            self.cube.aggregated_by(
                "val", self.mock_weighted_agg, weights=wrong_weights
            )


class Test_aggregated_by__lazy(tests.IrisTest):
    def setUp(self):
        self.data = np.arange(44).reshape(4, 11)
        self.lazydata = as_lazy_data(self.data)
        self.cube = Cube(self.lazydata)

        val_coord = AuxCoord(
            [0, 0, 0, 1, 1, 2, 0, 0, 2, 0, 1], long_name="val"
        )
        label_coord = AuxCoord(
            [
                "alpha",
                "alpha",
                "beta",
                "beta",
                "alpha",
                "gamma",
                "alpha",
                "alpha",
                "alpha",
                "gamma",
                "beta",
            ],
            long_name="label",
            units="no_unit",
        )
        simple_agg_coord = AuxCoord([1, 1, 2, 2], long_name="simple_agg")

        self.label_mean = np.array(
            [
                [4.0 + 1.0 / 3.0, 5.0, 7.0],
                [15.0 + 1.0 / 3.0, 16.0, 18.0],
                [26.0 + 1.0 / 3.0, 27.0, 29.0],
                [37.0 + 1.0 / 3.0, 38.0, 40.0],
            ]
        )
        self.val_mean = np.array(
            [
                [4.0 + 1.0 / 6.0, 5.0 + 2.0 / 3.0, 6.5],
                [15.0 + 1.0 / 6.0, 16.0 + 2.0 / 3.0, 17.5],
                [26.0 + 1.0 / 6.0, 27.0 + 2.0 / 3.0, 28.5],
                [37.0 + 1.0 / 6.0, 38.0 + 2.0 / 3.0, 39.5],
            ]
        )

        self.cube.add_aux_coord(simple_agg_coord, 0)
        self.cube.add_aux_coord(val_coord, 1)
        self.cube.add_aux_coord(label_coord, 1)

        self.simple_weights = np.array([1.0, 0.0, 2.0, 2.0])
        self.val_weights = 2.0 * np.ones(self.cube.shape, dtype=np.float32)

    def test_agg_by_label__lazy(self):
        # Aggregate a cube on a string coordinate label where label
        # and val entries are not in step; the resulting cube has a val
        # coord of bounded cells and a label coord of single string entries.
        res_cube = self.cube.aggregated_by("label", MEAN)
        val_coord = AuxCoord(
            np.array([1.0, 0.5, 1.0]),
            bounds=np.array([[0, 2], [0, 1], [0, 2]]),
            long_name="val",
        )
        label_coord = AuxCoord(
            np.array(["alpha", "beta", "gamma"]),
            long_name="label",
            units="no_unit",
        )
        self.assertTrue(res_cube.has_lazy_data())
        self.assertEqual(res_cube.coord("val"), val_coord)
        self.assertEqual(res_cube.coord("label"), label_coord)
        self.assertArrayEqual(res_cube.data, self.label_mean)
        self.assertFalse(res_cube.has_lazy_data())

    def test_agg_by_val__lazy(self):
        # Aggregate a cube on a numeric coordinate val where label
        # and val entries are not in step; the resulting cube has a label
        # coord with serialised labels from the aggregated cells.
        res_cube = self.cube.aggregated_by("val", MEAN)
        val_coord = AuxCoord(np.array([0, 1, 2]), long_name="val")
        exp0 = "alpha|alpha|beta|alpha|alpha|gamma"
        exp1 = "beta|alpha|beta"
        exp2 = "gamma|alpha"
        label_coord = AuxCoord(
            np.array((exp0, exp1, exp2)), long_name="label", units="no_unit"
        )
        self.assertTrue(res_cube.has_lazy_data())
        self.assertEqual(res_cube.coord("val"), val_coord)
        self.assertEqual(res_cube.coord("label"), label_coord)
        self.assertArrayEqual(res_cube.data, self.val_mean)
        self.assertFalse(res_cube.has_lazy_data())

    def test_single_string_aggregation__lazy(self):
        aux_coords = [
            (AuxCoord(["a", "b", "a"], long_name="foo"), 0),
            (AuxCoord(["a", "a", "a"], long_name="bar"), 0),
        ]
        cube = iris.cube.Cube(
            as_lazy_data(np.arange(12).reshape(3, 4)),
            aux_coords_and_dims=aux_coords,
        )
        means = np.array([[4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0]])
        result = cube.aggregated_by("foo", MEAN)
        self.assertTrue(result.has_lazy_data())
        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(
            result.coord("bar"), AuxCoord(["a|a", "a"], long_name="bar")
        )
        self.assertArrayEqual(result.data, means)
        self.assertFalse(result.has_lazy_data())

    def test_1d_weights__lazy(self):
        self.assertTrue(self.cube.has_lazy_data())

        cube_agg = self.cube.aggregated_by(
            "simple_agg", SUM, weights=self.simple_weights
        )

        self.assertTrue(self.cube.has_lazy_data())
        self.assertTrue(cube_agg.has_lazy_data())
        self.assertEqual(cube_agg.shape, (2, 11))

        row_0 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        row_1 = [
            110.0,
            114.0,
            118.0,
            122.0,
            126.0,
            130.0,
            134.0,
            138.0,
            142.0,
            146.0,
            150.0,
        ]
        np.testing.assert_array_almost_equal(
            cube_agg.data, np.array([row_0, row_1])
        )

    def test_2d_weights__lazy(self):
        self.assertTrue(self.cube.has_lazy_data())

        cube_agg = self.cube.aggregated_by(
            "val", SUM, weights=self.val_weights
        )

        self.assertTrue(self.cube.has_lazy_data())
        self.assertTrue(cube_agg.has_lazy_data())

        self.assertEqual(cube_agg.shape, (4, 3))
        np.testing.assert_array_almost_equal(
            cube_agg.data,
            np.array(
                [
                    [50.0, 34.0, 26.0],
                    [182.0, 100.0, 70.0],
                    [314.0, 166.0, 114.0],
                    [446.0, 232.0, 158.0],
                ]
            ),
        )

    def test_returned__lazy(self):
        self.assertTrue(self.cube.has_lazy_data())

        output = self.cube.aggregated_by(
            "simple_agg", SUM, weights=self.simple_weights, returned=True
        )

        self.assertTrue(self.cube.has_lazy_data())

        self.assertTrue(isinstance(output, tuple))
        self.assertEqual(len(output), 2)

        cube = output[0]
        self.assertTrue(isinstance(cube, Cube))
        self.assertTrue(cube.has_lazy_data())
        self.assertEqual(cube.shape, (2, 11))
        row_0 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        row_1 = [
            110.0,
            114.0,
            118.0,
            122.0,
            126.0,
            130.0,
            134.0,
            138.0,
            142.0,
            146.0,
            150.0,
        ]
        np.testing.assert_array_almost_equal(
            cube.data, np.array([row_0, row_1])
        )

        weights = output[1]
        self.assertEqual(weights.shape, (2, 11))
        np.testing.assert_array_almost_equal(
            weights,
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                ]
            ),
        )


class Test_aggregated_by__climatology(tests.IrisTest):
    def setUp(self):
        self.data = np.arange(100).reshape(20, 5)
        self.aggregator = iris.analysis.MEAN

    def get_result(
        self,
        transpose: bool = False,
        second_categorised: bool = False,
        bounds: bool = False,
        partially_aligned: bool = False,
        partially_aligned_timelike: bool = False,
        invalid_units: bool = False,
        already_climatological: bool = False,
        climatological_op: bool = True,
    ) -> Cube:
        cube_data = self.data
        if transpose:
            cube_data = cube_data.T
            axes = [1, 0]
        else:
            axes = [0, 1]
        if not invalid_units:
            units = Unit("days since 1970-01-01")
        else:
            units = Unit("m")
        if partially_aligned_timelike:
            pa_units = Unit("days since 1970-01-01")
        else:
            pa_units = Unit("m")

        # DimCoords
        aligned_coord = DimCoord(
            np.arange(20),
            long_name="aligned",
            units=units,
        )
        orthogonal_coord = DimCoord(np.arange(5), long_name="orth")

        if bounds:
            aligned_coord.guess_bounds()

        aligned_coord.climatological = already_climatological

        dim_coords_and_dims = zip([aligned_coord, orthogonal_coord], axes)

        # AuxCoords
        categorised_coord1 = AuxCoord(
            np.tile([0, 1], 10), long_name="cat1", units=Unit("month")
        )

        if second_categorised:
            categorised_coord2 = AuxCoord(
                np.tile([0, 1, 2, 3, 4], 4), long_name="cat2"
            )
            categorised_coords = [categorised_coord1, categorised_coord2]
        else:
            categorised_coords = categorised_coord1

        aux_coords_and_dims = [
            (categorised_coord1, axes[0]),
        ]

        if second_categorised:
            aux_coords_and_dims.append((categorised_coord2, axes[0]))

        if partially_aligned:
            partially_aligned_coord = AuxCoord(
                cube_data + 1,
                long_name="part_aligned",
                units=pa_units,
            )
            aux_coords_and_dims.append((partially_aligned_coord, (0, 1)))

        # Build cube
        in_cube = iris.cube.Cube(
            cube_data,
            long_name="wibble",
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
        )

        out_cube = in_cube.aggregated_by(
            categorised_coords,
            self.aggregator,
            climatological=climatological_op,
        )

        return out_cube

    def test_basic(self):
        """
        Check the least complicated version works (set climatological, set
        points correctly).
        """
        result = self.get_result()

        aligned_coord = result.coord("aligned")
        self.assertArrayEqual(aligned_coord.points, np.arange(2))
        self.assertArrayEqual(
            aligned_coord.bounds, np.array([[0, 18], [1, 19]])
        )
        self.assertTrue(aligned_coord.climatological)
        self.assertIn(aligned_coord, result.dim_coords)

        categorised_coord = result.coord("cat1")
        self.assertArrayEqual(categorised_coord.points, np.arange(2))
        self.assertIsNone(categorised_coord.bounds)
        self.assertFalse(categorised_coord.climatological)

    def test_2d_other_coord(self):
        """
        Check that we can handle aggregation applying to a 2d AuxCoord that
        covers the aggregation dimension and another one.
        """
        result = self.get_result(partially_aligned=True)

        aligned_coord = result.coord("aligned")
        self.assertArrayEqual(aligned_coord.points, np.arange(2))
        self.assertArrayEqual(
            aligned_coord.bounds, np.array([[0, 18], [1, 19]])
        )
        self.assertTrue(aligned_coord.climatological)

        part_aligned_coord = result.coord("part_aligned")
        self.assertArrayEqual(
            part_aligned_coord.points, np.arange(46, 56).reshape(2, 5)
        )
        self.assertArrayEqual(
            part_aligned_coord.bounds,
            np.array([np.arange(1, 11), np.arange(91, 101)]).T.reshape(
                2, 5, 2
            ),
        )
        self.assertFalse(part_aligned_coord.climatological)

    def test_2d_timelike_other_coord(self):
        """
        Check that we can handle aggregation applying to a 2d AuxCoord that
        covers the aggregation dimension and another one.
        """
        result = self.get_result(
            partially_aligned=True, partially_aligned_timelike=True
        )

        aligned_coord = result.coord("aligned")
        self.assertArrayEqual(aligned_coord.points, np.arange(2))
        self.assertArrayEqual(
            aligned_coord.bounds, np.array([[0, 18], [1, 19]])
        )
        self.assertTrue(aligned_coord.climatological)

        part_aligned_coord = result.coord("part_aligned")
        self.assertArrayEqual(
            part_aligned_coord.points, np.arange(1, 11).reshape(2, 5)
        )
        self.assertArrayEqual(
            part_aligned_coord.bounds,
            np.array([np.arange(1, 11), np.arange(91, 101)]).T.reshape(
                2, 5, 2
            ),
        )
        self.assertTrue(part_aligned_coord.climatological)

    def test_transposed(self):
        """
        Check that we can handle the axis of aggregation being a different one.
        """
        result = self.get_result(transpose=True)

        aligned_coord = result.coord("aligned")
        self.assertArrayEqual(aligned_coord.points, np.arange(2))
        self.assertArrayEqual(
            aligned_coord.bounds, np.array([[0, 18], [1, 19]])
        )
        self.assertTrue(aligned_coord.climatological)

        categorised_coord = result.coord("cat1")
        self.assertArrayEqual(categorised_coord.points, np.arange(2))
        self.assertIsNone(categorised_coord.bounds)
        self.assertFalse(categorised_coord.climatological)

    def test_bounded(self):
        """Check that we handle bounds correctly."""
        result = self.get_result(bounds=True)

        aligned_coord = result.coord("aligned")
        self.assertArrayEqual(aligned_coord.points, [-0.5, 0.5])
        self.assertArrayEqual(
            aligned_coord.bounds, np.array([[-0.5, 18.5], [0.5, 19.5]])
        )
        self.assertTrue(aligned_coord.climatological)

    def test_multiple_agg_coords(self):
        """
        Check that we can aggregate on multiple coords on the same axis.
        """
        result = self.get_result(second_categorised=True)

        aligned_coord = result.coord("aligned")
        self.assertArrayEqual(aligned_coord.points, np.arange(10))
        self.assertArrayEqual(
            aligned_coord.bounds,
            np.array([np.arange(10), np.arange(10, 20)]).T,
        )
        self.assertTrue(aligned_coord.climatological)

        categorised_coord1 = result.coord("cat1")
        self.assertArrayEqual(
            categorised_coord1.points, np.tile(np.arange(2), 5)
        )
        self.assertIsNone(categorised_coord1.bounds)
        self.assertFalse(categorised_coord1.climatological)

        categorised_coord2 = result.coord("cat2")
        self.assertArrayEqual(
            categorised_coord2.points, np.tile(np.arange(5), 2)
        )
        self.assertIsNone(categorised_coord2.bounds)
        self.assertFalse(categorised_coord2.climatological)

    def test_non_climatological_units(self):
        """
        Check that the failure to set the climatological flag on an incompatible
        unit is handled quietly.
        """
        result = self.get_result(invalid_units=True)

        aligned_coord = result.coord("aligned")
        self.assertArrayEqual(aligned_coord.points, np.arange(9, 11))
        self.assertArrayEqual(
            aligned_coord.bounds, np.array([[0, 18], [1, 19]])
        )
        self.assertFalse(aligned_coord.climatological)

    def test_clim_in_clim_op(self):
        """
        Check the least complicated version works (set climatological, set
        points correctly). For the input coordinate to be climatological, it
        must have bounds
        """
        result = self.get_result(bounds=True, already_climatological=True)

        aligned_coord = result.coord("aligned")
        self.assertArrayEqual(aligned_coord.points, [-0.5, 0.5])
        self.assertArrayEqual(
            aligned_coord.bounds, np.array([[-0.5, 18.5], [0.5, 19.5]])
        )
        self.assertTrue(aligned_coord.climatological)

        categorised_coord = result.coord("cat1")
        self.assertArrayEqual(categorised_coord.points, np.arange(2))
        self.assertIsNone(categorised_coord.bounds)
        self.assertFalse(categorised_coord.climatological)

    def test_clim_in_no_clim_op(self):
        """
        Check the least complicated version works (set climatological, set
        points correctly). For the input coordinate to be climatological, it
        must have bounds.
        """
        result = self.get_result(
            bounds=True, already_climatological=True, climatological_op=False
        )

        aligned_coord = result.coord("aligned")
        self.assertArrayEqual(aligned_coord.points, np.arange(9, 11))
        self.assertArrayEqual(
            aligned_coord.bounds, np.array([[-0.5, 18.5], [0.5, 19.5]])
        )
        self.assertTrue(aligned_coord.climatological)

        categorised_coord = result.coord("cat1")
        self.assertArrayEqual(categorised_coord.points, np.arange(2))
        self.assertIsNone(categorised_coord.bounds)
        self.assertFalse(categorised_coord.climatological)


class Test_aggregated_by__derived(tests.IrisTest):
    def setUp(self):
        self.cube = realistic_4d()[:, :10, :6, :8]
        self.time_cat_coord = AuxCoord(
            [0, 0, 1, 1, 2, 2], long_name="time_cat"
        )
        self.cube.add_aux_coord(self.time_cat_coord, 0)
        height_data = np.zeros(self.cube.shape[1])
        height_data[5:] = 1
        self.height_cat_coord = AuxCoord(height_data, long_name="height_cat")
        self.cube.add_aux_coord(self.height_cat_coord, 1)
        self.aggregator = iris.analysis.MEAN

    def test_grouped_dim(self):
        """
        Check that derived coordinates are maintained when the coordinates they
        derive from are aggregated.
        """
        result = self.cube.aggregated_by(
            self.height_cat_coord,
            self.aggregator,
        )
        assert len(result.aux_factories) == 1
        altitude = result.coord("altitude")
        assert altitude.shape == (2, 6, 8)

        # Check the bounds are derived as expected.
        orig_alt_bounds = self.cube.coord("altitude").bounds
        bounds_0 = orig_alt_bounds[0::5, :, :, 0]
        bounds_1 = orig_alt_bounds[4::5, :, :, 1]
        expected_bounds = np.stack([bounds_0, bounds_1], axis=-1)
        assert np.array_equal(expected_bounds, result.coord("altitude").bounds)

    def test_ungrouped_dim(self):
        """
        Check that derived coordinates are preserved when aggregating along a
        different axis.
        """
        result = self.cube.aggregated_by(
            self.time_cat_coord,
            self.aggregator,
        )
        assert len(result.aux_factories) == 1
        altitude = result.coord("altitude")
        assert altitude == self.cube.coord("altitude")


if __name__ == "__main__":
    tests.main()
