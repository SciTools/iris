# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.cube.Cube` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from itertools import permutations
from unittest import mock

import numpy as np
import numpy.ma as ma

from cf_units import Unit

import iris.analysis
import iris.aux_factory
import iris.coords
import iris.exceptions
from iris.analysis import WeightedAggregator, Aggregator
from iris.analysis import MEAN
from iris.aux_factory import HybridHeightFactory
from iris.cube import Cube
from iris.coords import (
    AuxCoord,
    DimCoord,
    CellMeasure,
    AncillaryVariable,
    CellMethod,
)
from iris.exceptions import (
    CoordinateNotFoundError,
    CellMeasureNotFoundError,
    AncillaryVariableNotFoundError,
    UnitConversionError,
)
from iris._lazy_data import as_lazy_data
import iris.tests.stock as stock


class Test___init___data(tests.IrisTest):
    def test_ndarray(self):
        # np.ndarray should be allowed through
        data = np.arange(12).reshape(3, 4)
        cube = Cube(data)
        self.assertEqual(type(cube.data), np.ndarray)
        self.assertArrayEqual(cube.data, data)

    def test_masked(self):
        # ma.MaskedArray should be allowed through
        data = ma.masked_greater(np.arange(12).reshape(3, 4), 1)
        cube = Cube(data)
        self.assertEqual(type(cube.data), ma.MaskedArray)
        self.assertMaskedArrayEqual(cube.data, data)

    def test_masked_no_mask(self):
        # ma.MaskedArray should be allowed through even if it has no mask
        data = ma.masked_array(np.arange(12).reshape(3, 4), False)
        cube = Cube(data)
        self.assertEqual(type(cube.data), ma.MaskedArray)
        self.assertMaskedArrayEqual(cube.data, data)

    def test_matrix(self):
        # Subclasses of np.ndarray should be coerced back to np.ndarray.
        # (Except for np.ma.MaskedArray.)
        data = np.matrix([[1, 2, 3], [4, 5, 6]])
        cube = Cube(data)
        self.assertEqual(type(cube.data), np.ndarray)
        self.assertArrayEqual(cube.data, data)


class Test_data_dtype_fillvalue(tests.IrisTest):
    def _sample_data(
        self, dtype=("f4"), masked=False, fill_value=None, lazy=False
    ):
        data = np.arange(6).reshape((2, 3))
        dtype = np.dtype(dtype)
        data = data.astype(dtype)
        if masked:
            data = ma.masked_array(
                data, mask=[[0, 1, 0], [0, 0, 0]], fill_value=fill_value
            )
        if lazy:
            data = as_lazy_data(data)
        return data

    def _sample_cube(
        self, dtype=("f4"), masked=False, fill_value=None, lazy=False
    ):
        data = self._sample_data(
            dtype=dtype, masked=masked, fill_value=fill_value, lazy=lazy
        )
        cube = Cube(data)
        return cube

    def test_realdata_change(self):
        # Check re-assigning real data.
        cube = self._sample_cube()
        self.assertEqual(cube.dtype, np.float32)
        new_dtype = np.dtype("i4")
        new_data = self._sample_data(dtype=new_dtype)
        cube.data = new_data
        self.assertIs(cube.core_data(), new_data)
        self.assertEqual(cube.dtype, new_dtype)

    def test_realmaskdata_change(self):
        # Check re-assigning real masked data.
        cube = self._sample_cube(masked=True, fill_value=1234)
        self.assertEqual(cube.dtype, np.float32)
        new_dtype = np.dtype("i4")
        new_fill_value = 4321
        new_data = self._sample_data(
            masked=True, fill_value=new_fill_value, dtype=new_dtype
        )
        cube.data = new_data
        self.assertIs(cube.core_data(), new_data)
        self.assertEqual(cube.dtype, new_dtype)
        self.assertEqual(cube.data.fill_value, new_fill_value)

    def test_lazydata_change(self):
        # Check re-assigning lazy data.
        cube = self._sample_cube(lazy=True)
        self.assertEqual(cube.core_data().dtype, np.float32)
        new_dtype = np.dtype("f8")
        new_data = self._sample_data(new_dtype, lazy=True)
        cube.data = new_data
        self.assertIs(cube.core_data(), new_data)
        self.assertEqual(cube.dtype, new_dtype)

    def test_lazymaskdata_change(self):
        # Check re-assigning lazy masked data.
        cube = self._sample_cube(masked=True, fill_value=1234, lazy=True)
        self.assertEqual(cube.core_data().dtype, np.float32)
        new_dtype = np.dtype("f8")
        new_fill_value = 4321
        new_data = self._sample_data(
            dtype=new_dtype, masked=True, fill_value=new_fill_value, lazy=True
        )
        cube.data = new_data
        self.assertIs(cube.core_data(), new_data)
        self.assertEqual(cube.dtype, new_dtype)
        self.assertEqual(cube.data.fill_value, new_fill_value)

    def test_lazydata_realise(self):
        # Check touching lazy data.
        cube = self._sample_cube(lazy=True)
        data = cube.data
        self.assertIs(cube.core_data(), data)
        self.assertEqual(cube.dtype, np.float32)

    def test_lazymaskdata_realise(self):
        # Check touching masked lazy data.
        fill_value = 27.3
        cube = self._sample_cube(masked=True, fill_value=fill_value, lazy=True)
        data = cube.data
        self.assertIs(cube.core_data(), data)
        self.assertEqual(cube.dtype, np.float32)
        self.assertEqual(data.fill_value, np.float32(fill_value))

    def test_realmaskedconstantint_realise(self):
        masked_data = ma.masked_array([666], mask=True)
        masked_constant = masked_data[0]
        cube = Cube(masked_constant)
        data = cube.data
        self.assertTrue(ma.isMaskedArray(data))
        self.assertNotIsInstance(data, ma.core.MaskedConstant)

    def test_lazymaskedconstantint_realise(self):
        dtype = np.dtype("i2")
        masked_data = ma.masked_array([666], mask=True, dtype=dtype)
        masked_constant = masked_data[0]
        masked_constant_lazy = as_lazy_data(masked_constant)
        cube = Cube(masked_constant_lazy)
        data = cube.data
        self.assertTrue(ma.isMaskedArray(data))
        self.assertNotIsInstance(data, ma.core.MaskedConstant)

    def test_lazydata___getitem__dtype(self):
        fill_value = 1234
        dtype = np.dtype("int16")
        masked_array = ma.masked_array(
            np.arange(5),
            mask=[0, 0, 1, 0, 0],
            fill_value=fill_value,
            dtype=dtype,
        )
        lazy_masked_array = as_lazy_data(masked_array)
        cube = Cube(lazy_masked_array)
        subcube = cube[3:]
        self.assertEqual(subcube.dtype, dtype)
        self.assertEqual(subcube.data.fill_value, fill_value)


class Test_extract(tests.IrisTest):
    def test_scalar_cube_exists(self):
        # Ensure that extract is able to extract a scalar cube.
        constraint = iris.Constraint(name="a1")
        cube = Cube(1, long_name="a1")
        res = cube.extract(constraint)
        self.assertIs(res, cube)

    def test_scalar_cube_noexists(self):
        # Ensure that extract does not return a non-matching scalar cube.
        constraint = iris.Constraint(name="a2")
        cube = Cube(1, long_name="a1")
        res = cube.extract(constraint)
        self.assertIs(res, None)

    def test_scalar_cube_coord_match(self):
        # Ensure that extract is able to extract a scalar cube according to
        # constrained scalar coordinate.
        constraint = iris.Constraint(scalar_coord=0)
        cube = Cube(1, long_name="a1")
        coord = iris.coords.AuxCoord(0, long_name="scalar_coord")
        cube.add_aux_coord(coord, None)
        res = cube.extract(constraint)
        self.assertIs(res, cube)

    def test_scalar_cube_coord_nomatch(self):
        # Ensure that extract is not extracting a scalar cube with scalar
        # coordinate that does not match the constraint.
        constraint = iris.Constraint(scalar_coord=1)
        cube = Cube(1, long_name="a1")
        coord = iris.coords.AuxCoord(0, long_name="scalar_coord")
        cube.add_aux_coord(coord, None)
        res = cube.extract(constraint)
        self.assertIs(res, None)

    def test_1d_cube_exists(self):
        # Ensure that extract is able to extract from a 1d cube.
        constraint = iris.Constraint(name="a1")
        cube = Cube([1], long_name="a1")
        res = cube.extract(constraint)
        self.assertIs(res, cube)

    def test_1d_cube_noexists(self):
        # Ensure that extract does not return a non-matching 1d cube.
        constraint = iris.Constraint(name="a2")
        cube = Cube([1], long_name="a1")
        res = cube.extract(constraint)
        self.assertIs(res, None)


class Test_xml(tests.IrisTest):
    def test_checksum_ignores_masked_values(self):
        # Mask out an single element.
        data = ma.arange(12).reshape(3, 4)
        data[1, 2] = ma.masked
        cube = Cube(data)
        self.assertCML(cube)

        # If we change the underlying value before masking it, the
        # checksum should be unaffected.
        data = ma.arange(12).reshape(3, 4)
        data[1, 2] = 42
        data[1, 2] = ma.masked
        cube = Cube(data)
        self.assertCML(cube)

    def test_byteorder_default(self):
        cube = Cube(np.arange(3))
        self.assertIn("byteorder", cube.xml())

    def test_byteorder_false(self):
        cube = Cube(np.arange(3))
        self.assertNotIn("byteorder", cube.xml(byteorder=False))

    def test_byteorder_true(self):
        cube = Cube(np.arange(3))
        self.assertIn("byteorder", cube.xml(byteorder=True))

    def test_cell_measures(self):
        cube = stock.simple_3d_w_multidim_coords()
        cm_a = iris.coords.CellMeasure(
            np.zeros(cube.shape[-2:]), measure="area", units="1"
        )
        cube.add_cell_measure(cm_a, (1, 2))
        cm_v = iris.coords.CellMeasure(
            np.zeros(cube.shape),
            measure="volume",
            long_name="madeup",
            units="m3",
        )
        cube.add_cell_measure(cm_v, (0, 1, 2))
        self.assertCML(cube)

    def test_ancils(self):
        cube = stock.simple_2d_w_multidim_coords()
        av = iris.coords.AncillaryVariable(
            np.zeros(cube.shape), long_name="xy", var_name="vxy", units="1"
        )
        cube.add_ancillary_variable(av, (0, 1))
        self.assertCML(cube)


class Test_collapsed__lazy(tests.IrisTest):
    def setUp(self):
        self.data = np.arange(6.0).reshape((2, 3))
        self.lazydata = as_lazy_data(self.data)
        cube = Cube(self.lazydata)
        for i_dim, name in enumerate(("y", "x")):
            npts = cube.shape[i_dim]
            coord = DimCoord(np.arange(npts), long_name=name)
            cube.add_dim_coord(coord, i_dim)
        self.cube = cube

    def test_dim0_lazy(self):
        cube_collapsed = self.cube.collapsed("y", MEAN)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, [1.5, 2.5, 3.5])
        self.assertFalse(cube_collapsed.has_lazy_data())

    def test_dim1_lazy(self):
        cube_collapsed = self.cube.collapsed("x", MEAN)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, [1.0, 4.0])
        self.assertFalse(cube_collapsed.has_lazy_data())

    def test_multidims(self):
        # Check that MEAN works with multiple dims.
        cube_collapsed = self.cube.collapsed(("x", "y"), MEAN)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAllClose(cube_collapsed.data, 2.5)

    def test_non_lazy_aggregator(self):
        # An aggregator which doesn't have a lazy function should still work.
        dummy_agg = Aggregator(
            "custom_op", lambda x, axis=None: np.mean(x, axis=axis)
        )
        result = self.cube.collapsed("x", dummy_agg)
        self.assertFalse(result.has_lazy_data())
        self.assertArrayEqual(result.data, np.mean(self.data, axis=1))


class Test_collapsed__cellmeasure_ancils(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(6.0).reshape((2, 3)))
        for i_dim, name in enumerate(("y", "x")):
            npts = cube.shape[i_dim]
            coord = DimCoord(np.arange(npts), long_name=name)
            cube.add_dim_coord(coord, i_dim)
        self.ancillary_variable = AncillaryVariable([0, 1], long_name="foo")
        cube.add_ancillary_variable(self.ancillary_variable, 0)
        self.cell_measure = CellMeasure([0, 1], long_name="bar")
        cube.add_cell_measure(self.cell_measure, 0)
        self.cube = cube

    def test_ancillary_variables_and_cell_measures_kept(self):
        cube_collapsed = self.cube.collapsed("x", MEAN)
        self.assertEqual(
            cube_collapsed.ancillary_variables(), [self.ancillary_variable]
        )
        self.assertEqual(cube_collapsed.cell_measures(), [self.cell_measure])

    def test_ancillary_variables_and_cell_measures_removed(self):
        cube_collapsed = self.cube.collapsed("y", MEAN)
        self.assertEqual(cube_collapsed.ancillary_variables(), [])
        self.assertEqual(cube_collapsed.cell_measures(), [])


class Test_collapsed__warning(tests.IrisTest):
    def setUp(self):
        self.cube = Cube([[1, 2], [1, 2]])
        lat = DimCoord([1, 2], standard_name="latitude")
        lon = DimCoord([1, 2], standard_name="longitude")
        grid_lat = AuxCoord([1, 2], standard_name="grid_latitude")
        grid_lon = AuxCoord([1, 2], standard_name="grid_longitude")
        wibble = AuxCoord([1, 2], long_name="wibble")

        self.cube.add_dim_coord(lat, 0)
        self.cube.add_dim_coord(lon, 1)
        self.cube.add_aux_coord(grid_lat, 0)
        self.cube.add_aux_coord(grid_lon, 1)
        self.cube.add_aux_coord(wibble, 1)

    def _aggregator(self, uses_weighting):
        # Returns a mock aggregator with a mocked method (uses_weighting)
        # which returns the given True/False condition.
        aggregator = mock.Mock(spec=WeightedAggregator, lazy_func=None)
        aggregator.cell_method = None
        aggregator.uses_weighting = mock.Mock(return_value=uses_weighting)

        return aggregator

    def _assert_warn_collapse_without_weight(self, coords, warn):
        # Ensure that warning is raised.
        msg = "Collapsing spatial coordinate {!r} without weighting"
        for coord in coords:
            self.assertIn(mock.call(msg.format(coord)), warn.call_args_list)

    def _assert_nowarn_collapse_without_weight(self, coords, warn):
        # Ensure that warning is not rised.
        msg = "Collapsing spatial coordinate {!r} without weighting"
        for coord in coords:
            self.assertNotIn(mock.call(msg.format(coord)), warn.call_args_list)

    def test_lat_lon_noweighted_aggregator(self):
        # Collapse latitude coordinate with unweighted aggregator.
        aggregator = mock.Mock(spec=Aggregator, lazy_func=None)
        aggregator.cell_method = None
        coords = ["latitude", "longitude"]

        with mock.patch("warnings.warn") as warn:
            self.cube.collapsed(coords, aggregator, somekeyword="bla")

        self._assert_nowarn_collapse_without_weight(coords, warn)

    def test_lat_lon_weighted_aggregator(self):
        # Collapse latitude coordinate with weighted aggregator without
        # providing weights.
        aggregator = self._aggregator(False)
        coords = ["latitude", "longitude"]

        with mock.patch("warnings.warn") as warn:
            self.cube.collapsed(coords, aggregator)

        coords = [coord for coord in coords if "latitude" in coord]
        self._assert_warn_collapse_without_weight(coords, warn)

    def test_lat_lon_weighted_aggregator_with_weights(self):
        # Collapse latitude coordinate with a weighted aggregators and
        # providing suitable weights.
        weights = np.array([[0.1, 0.5], [0.3, 0.2]])
        aggregator = self._aggregator(True)
        coords = ["latitude", "longitude"]

        with mock.patch("warnings.warn") as warn:
            self.cube.collapsed(coords, aggregator, weights=weights)

        self._assert_nowarn_collapse_without_weight(coords, warn)

    def test_lat_lon_weighted_aggregator_alt(self):
        # Collapse grid_latitude coordinate with weighted aggregator without
        # providing weights.  Tests coordinate matching logic.
        aggregator = self._aggregator(False)
        coords = ["grid_latitude", "grid_longitude"]

        with mock.patch("warnings.warn") as warn:
            self.cube.collapsed(coords, aggregator)

        coords = [coord for coord in coords if "latitude" in coord]
        self._assert_warn_collapse_without_weight(coords, warn)

    def test_no_lat_weighted_aggregator_mixed(self):
        # Collapse grid_latitude and an unmatched coordinate (not lat/lon)
        # with weighted aggregator without providing weights.
        # Tests coordinate matching logic.
        aggregator = self._aggregator(False)
        coords = ["wibble"]

        with mock.patch("warnings.warn") as warn:
            self.cube.collapsed(coords, aggregator)

        self._assert_nowarn_collapse_without_weight(coords, warn)


class Test_summary(tests.IrisTest):
    def setUp(self):
        self.cube = Cube(0)

    def test_cell_datetime_objects(self):
        self.cube.add_aux_coord(AuxCoord(42, units="hours since epoch"))
        summary = self.cube.summary()
        self.assertIn("1970-01-02 18:00:00", summary)

    def test_scalar_str_coord(self):
        str_value = "foo"
        self.cube.add_aux_coord(AuxCoord(str_value))
        summary = self.cube.summary()
        self.assertIn(str_value, summary)

    def test_ancillary_variable(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        av = AncillaryVariable([1, 2], "status_flag")
        cube.add_ancillary_variable(av, 0)
        expected_summary = (
            "unknown / (unknown)                 (-- : 2; -- : 3)\n"
            "     Ancillary variables:\n"
            "          status_flag                   x       -"
        )
        self.assertEqual(cube.summary(), expected_summary)


class Test_is_compatible(tests.IrisTest):
    def setUp(self):
        self.test_cube = Cube([1.0])
        self.other_cube = self.test_cube.copy()

    def test_noncommon_array_attrs_compatible(self):
        # Non-common array attributes should be ok.
        self.test_cube.attributes["array_test"] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_cube.is_compatible(self.other_cube))

    def test_matching_array_attrs_compatible(self):
        # Matching array attributes should be ok.
        self.test_cube.attributes["array_test"] = np.array([1.0, 2, 3])
        self.other_cube.attributes["array_test"] = np.array([1.0, 2, 3])
        self.assertTrue(self.test_cube.is_compatible(self.other_cube))

    def test_different_array_attrs_incompatible(self):
        # Differing array attributes should make the cubes incompatible.
        self.test_cube.attributes["array_test"] = np.array([1.0, 2, 3])
        self.other_cube.attributes["array_test"] = np.array([1.0, 2, 777.7])
        self.assertFalse(self.test_cube.is_compatible(self.other_cube))


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

        self.ancillary_variable = AncillaryVariable(
            [0, 1, 2, 3], long_name="foo"
        )
        self.cube.add_ancillary_variable(self.ancillary_variable, 0)
        self.cell_measure = CellMeasure([0, 1, 2, 3], long_name="bar")
        self.cube.add_cell_measure(self.cell_measure, 0)

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
            bounds=np.array([[0, 2], [0, 1], [2, 0]]),
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

    def test_agg_by_label__lazy(self):
        # Aggregate a cube on a string coordinate label where label
        # and val entries are not in step; the resulting cube has a val
        # coord of bounded cells and a label coord of single string entries.
        res_cube = self.cube.aggregated_by("label", MEAN)
        val_coord = AuxCoord(
            np.array([1.0, 0.5, 1.0]),
            bounds=np.array([[0, 2], [0, 1], [2, 0]]),
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


class Test_rolling_window(tests.IrisTest):
    def setUp(self):
        self.cube = Cube(np.arange(6))
        self.multi_dim_cube = Cube(np.arange(36).reshape(6, 6))
        val_coord = DimCoord([0, 1, 2, 3, 4, 5], long_name="val")
        month_coord = AuxCoord(
            ["jan", "feb", "mar", "apr", "may", "jun"], long_name="month"
        )
        extra_coord = AuxCoord([0, 1, 2, 3, 4, 5], long_name="extra")
        self.cube.add_dim_coord(val_coord, 0)
        self.cube.add_aux_coord(month_coord, 0)
        self.multi_dim_cube.add_dim_coord(val_coord, 0)
        self.multi_dim_cube.add_aux_coord(extra_coord, 1)
        self.ancillary_variable = AncillaryVariable(
            [0, 1, 2, 0, 1, 2], long_name="foo"
        )
        self.multi_dim_cube.add_ancillary_variable(self.ancillary_variable, 1)
        self.cell_measure = CellMeasure([0, 1, 2, 0, 1, 2], long_name="bar")
        self.multi_dim_cube.add_cell_measure(self.cell_measure, 1)

        self.mock_agg = mock.Mock(spec=Aggregator)
        self.mock_agg.aggregate = mock.Mock(return_value=np.empty([4]))
        self.mock_agg.post_process = mock.Mock(side_effect=lambda x, y, z: x)

    def test_string_coord(self):
        # Rolling window on a cube that contains a string coordinate.
        res_cube = self.cube.rolling_window("val", self.mock_agg, 3)
        val_coord = DimCoord(
            np.array([1, 2, 3, 4]),
            bounds=np.array([[0, 2], [1, 3], [2, 4], [3, 5]]),
            long_name="val",
        )
        month_coord = AuxCoord(
            np.array(
                ["jan|feb|mar", "feb|mar|apr", "mar|apr|may", "apr|may|jun"]
            ),
            bounds=np.array(
                [
                    ["jan", "mar"],
                    ["feb", "apr"],
                    ["mar", "may"],
                    ["apr", "jun"],
                ]
            ),
            long_name="month",
        )
        self.assertEqual(res_cube.coord("val"), val_coord)
        self.assertEqual(res_cube.coord("month"), month_coord)

    def test_kwargs(self):
        # Rolling window with missing data not tolerated
        window = 2
        self.cube.data = ma.array(
            self.cube.data, mask=([True, False, False, False, True, False])
        )
        res_cube = self.cube.rolling_window(
            "val", iris.analysis.MEAN, window, mdtol=0
        )
        expected_result = ma.array(
            [-99.0, 1.5, 2.5, -99.0, -99.0],
            mask=[True, False, False, True, True],
            dtype=np.float64,
        )
        self.assertMaskedArrayEqual(expected_result, res_cube.data)

    def test_ancillary_variables_and_cell_measures_kept(self):
        res_cube = self.multi_dim_cube.rolling_window("val", self.mock_agg, 3)
        self.assertEqual(
            res_cube.ancillary_variables(), [self.ancillary_variable]
        )
        self.assertEqual(res_cube.cell_measures(), [self.cell_measure])

    def test_ancillary_variables_and_cell_measures_removed(self):
        res_cube = self.multi_dim_cube.rolling_window(
            "extra", self.mock_agg, 3
        )
        self.assertEqual(res_cube.ancillary_variables(), [])
        self.assertEqual(res_cube.cell_measures(), [])


class Test_slices_dim_order(tests.IrisTest):
    """
    This class tests the capability of iris.cube.Cube.slices(), including its
    ability to correctly re-order the dimensions.
    """

    def setUp(self):
        """
        setup a 4D iris cube, each dimension is length 1.
        The dimensions are;
            dim1: time
            dim2: height
            dim3: latitude
            dim4: longitude
        """
        self.cube = iris.cube.Cube(np.array([[[[8.0]]]]))
        self.cube.add_dim_coord(iris.coords.DimCoord([0], "time"), [0])
        self.cube.add_dim_coord(iris.coords.DimCoord([0], "height"), [1])
        self.cube.add_dim_coord(iris.coords.DimCoord([0], "latitude"), [2])
        self.cube.add_dim_coord(iris.coords.DimCoord([0], "longitude"), [3])

    @staticmethod
    def expected_cube_setup(dim1name, dim2name, dim3name):
        """
        input:
        ------
            dim1name: str
                name of the first dimension coordinate
            dim2name: str
                name of the second dimension coordinate
            dim3name: str
                name of the third dimension coordinate
        output:
        ------
            cube: iris cube
                iris cube with the specified axis holding the data 8
        """
        cube = iris.cube.Cube(np.array([[[8.0]]]))
        cube.add_dim_coord(iris.coords.DimCoord([0], dim1name), [0])
        cube.add_dim_coord(iris.coords.DimCoord([0], dim2name), [1])
        cube.add_dim_coord(iris.coords.DimCoord([0], dim3name), [2])
        return cube

    def check_order(self, dim1, dim2, dim3, dim_to_remove):
        """
        does two things:
        (1) slices the 4D cube in dim1, dim2, dim3 (and removes the scalar
        coordinate) and
        (2) sets up a 3D cube with dim1, dim2, dim3.
        input:
        -----
            dim1: str
                name of first dimension
            dim2: str
                name of second dimension
            dim3: str
                name of third dimension
            dim_to_remove: str
                name of the dimension that transforms into a scalar coordinate
                when slicing the cube.
        output:
        ------
            sliced_cube: 3D cube
                the cube that results if slicing the original cube
            expected_cube: 3D cube
                a cube set up with the axis corresponding to the dims
        """
        sliced_cube = next(self.cube.slices([dim1, dim2, dim3]))
        sliced_cube.remove_coord(dim_to_remove)
        expected_cube = self.expected_cube_setup(dim1, dim2, dim3)
        self.assertEqual(sliced_cube, expected_cube)

    def test_all_permutations(self):
        for perm in permutations(["time", "height", "latitude", "longitude"]):
            self.check_order(*perm)


@tests.skip_data
class Test_slices_over(tests.IrisTest):
    def setUp(self):
        self.cube = stock.realistic_4d()
        # Define expected iterators for 1D and 2D test cases.
        self.exp_iter_1d = range(
            len(self.cube.coord("model_level_number").points)
        )
        self.exp_iter_2d = np.ndindex(6, 70, 1, 1)
        # Define maximum number of interations for particularly long
        # (and so time-consuming) iterators.
        self.long_iterator_max = 5

    def test_1d_slice_coord_given(self):
        res = self.cube.slices_over(self.cube.coord("model_level_number"))
        for i, res_cube in zip(self.exp_iter_1d, res):
            expected = self.cube[:, i]
            self.assertEqual(res_cube, expected)

    def test_1d_slice_nonexistent_coord_given(self):
        with self.assertRaises(CoordinateNotFoundError):
            _ = self.cube.slices_over(self.cube.coord("wibble"))

    def test_1d_slice_coord_name_given(self):
        res = self.cube.slices_over("model_level_number")
        for i, res_cube in zip(self.exp_iter_1d, res):
            expected = self.cube[:, i]
            self.assertEqual(res_cube, expected)

    def test_1d_slice_nonexistent_coord_name_given(self):
        with self.assertRaises(CoordinateNotFoundError):
            _ = self.cube.slices_over("wibble")

    def test_1d_slice_dimension_given(self):
        res = self.cube.slices_over(1)
        for i, res_cube in zip(self.exp_iter_1d, res):
            expected = self.cube[:, i]
            self.assertEqual(res_cube, expected)

    def test_1d_slice_nonexistent_dimension_given(self):
        with self.assertRaisesRegex(ValueError, "iterator over a dimension"):
            _ = self.cube.slices_over(self.cube.ndim + 1)

    def test_2d_slice_coord_given(self):
        # Slicing over these two dimensions returns 420 2D cubes, so only check
        # cubes up to `self.long_iterator_max` to keep test runtime sensible.
        res = self.cube.slices_over(
            [self.cube.coord("time"), self.cube.coord("model_level_number")]
        )
        for ct in range(self.long_iterator_max):
            indices = list(next(self.exp_iter_2d))
            # Replace the dimensions not iterated over with spanning slices.
            indices[2] = indices[3] = slice(None)
            expected = self.cube[tuple(indices)]
            self.assertEqual(next(res), expected)

    def test_2d_slice_nonexistent_coord_given(self):
        with self.assertRaises(CoordinateNotFoundError):
            _ = self.cube.slices_over(
                [self.cube.coord("time"), self.cube.coord("wibble")]
            )

    def test_2d_slice_coord_name_given(self):
        # Slicing over these two dimensions returns 420 2D cubes, so only check
        # cubes up to `self.long_iterator_max` to keep test runtime sensible.
        res = self.cube.slices_over(["time", "model_level_number"])
        for ct in range(self.long_iterator_max):
            indices = list(next(self.exp_iter_2d))
            # Replace the dimensions not iterated over with spanning slices.
            indices[2] = indices[3] = slice(None)
            expected = self.cube[tuple(indices)]
            self.assertEqual(next(res), expected)

    def test_2d_slice_nonexistent_coord_name_given(self):
        with self.assertRaises(CoordinateNotFoundError):
            _ = self.cube.slices_over(["time", "wibble"])

    def test_2d_slice_dimension_given(self):
        # Slicing over these two dimensions returns 420 2D cubes, so only check
        # cubes up to `self.long_iterator_max` to keep test runtime sensible.
        res = self.cube.slices_over([0, 1])
        for ct in range(self.long_iterator_max):
            indices = list(next(self.exp_iter_2d))
            # Replace the dimensions not iterated over with spanning slices.
            indices[2] = indices[3] = slice(None)
            expected = self.cube[tuple(indices)]
            self.assertEqual(next(res), expected)

    def test_2d_slice_reversed_dimension_given(self):
        # Confirm that reversing the order of the dimensions returns the same
        # results as the above test.
        res = self.cube.slices_over([1, 0])
        for ct in range(self.long_iterator_max):
            indices = list(next(self.exp_iter_2d))
            # Replace the dimensions not iterated over with spanning slices.
            indices[2] = indices[3] = slice(None)
            expected = self.cube[tuple(indices)]
            self.assertEqual(next(res), expected)

    def test_2d_slice_nonexistent_dimension_given(self):
        with self.assertRaisesRegex(ValueError, "iterator over a dimension"):
            _ = self.cube.slices_over([0, self.cube.ndim + 1])

    def test_multidim_slice_coord_given(self):
        # Slicing over surface altitude returns 100x100 2D cubes, so only check
        # cubes up to `self.long_iterator_max` to keep test runtime sensible.
        res = self.cube.slices_over("surface_altitude")
        # Define special ndindex iterator for the different dims sliced over.
        nditer = np.ndindex(1, 1, 100, 100)
        for ct in range(self.long_iterator_max):
            indices = list(next(nditer))
            # Replace the dimensions not iterated over with spanning slices.
            indices[0] = indices[1] = slice(None)
            expected = self.cube[tuple(indices)]
            self.assertEqual(next(res), expected)

    def test_duplicate_coordinate_given(self):
        res = self.cube.slices_over([1, 1])
        for i, res_cube in zip(self.exp_iter_1d, res):
            expected = self.cube[:, i]
            self.assertEqual(res_cube, expected)

    def test_non_orthogonal_coordinates_given(self):
        res = self.cube.slices_over(["model_level_number", "sigma"])
        for i, res_cube in zip(self.exp_iter_1d, res):
            expected = self.cube[:, i]
            self.assertEqual(res_cube, expected)

    def test_nodimension(self):
        # Slicing over no dimension should return the whole cube.
        res = self.cube.slices_over([])
        self.assertEqual(next(res), self.cube)


def create_cube(lon_min, lon_max, bounds=False):
    n_lons = max(lon_min, lon_max) - min(lon_max, lon_min)
    data = np.arange(4 * 3 * n_lons, dtype="f4").reshape(4, 3, -1)
    data = as_lazy_data(data)
    cube = Cube(data, standard_name="x_wind", units="ms-1")
    cube.add_dim_coord(
        iris.coords.DimCoord(
            [0, 20, 40, 80], long_name="level_height", units="m"
        ),
        0,
    )
    cube.add_aux_coord(
        iris.coords.AuxCoord(
            [1.0, 0.9, 0.8, 0.6], long_name="sigma", units="1"
        ),
        0,
    )
    cube.add_dim_coord(
        iris.coords.DimCoord([-45, 0, 45], "latitude", units="degrees"), 1
    )
    step = 1 if lon_max > lon_min else -1
    circular = abs(lon_max - lon_min) == 360
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(lon_min, lon_max, step),
            "longitude",
            units="degrees",
            circular=circular,
        ),
        2,
    )
    if bounds:
        cube.coord("longitude").guess_bounds()
    cube.add_aux_coord(
        iris.coords.AuxCoord(
            np.arange(3 * n_lons).reshape(3, -1) * 10,
            "surface_altitude",
            units="m",
        ),
        [1, 2],
    )
    cube.add_aux_factory(
        iris.aux_factory.HybridHeightFactory(
            cube.coord("level_height"),
            cube.coord("sigma"),
            cube.coord("surface_altitude"),
        )
    )
    return cube


# Ensure all the other coordinates and factories are correctly preserved.
class Test_intersection__Metadata(tests.IrisTest):
    def test_metadata(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190))
        self.assertCMLApproxData(result)

    def test_metadata_wrapped(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(170, 190))
        self.assertCMLApproxData(result)


# Explicitly check the handling of `circular` on the result.
class Test_intersection__Circular(tests.IrisTest):
    def test_regional(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190))
        self.assertFalse(result.coord("longitude").circular)

    def test_regional_wrapped(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(170, 190))
        self.assertFalse(result.coord("longitude").circular)

    def test_global(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(-180, 180))
        self.assertTrue(result.coord("longitude").circular)

    def test_global_wrapped(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(10, 370))
        self.assertTrue(result.coord("longitude").circular)


# Check the various error conditions.
class Test_intersection__Invalid(tests.IrisTest):
    def test_reversed_min_max(self):
        cube = create_cube(0, 360)
        with self.assertRaises(ValueError):
            cube.intersection(longitude=(30, 10))

    def test_dest_too_large(self):
        cube = create_cube(0, 360)
        with self.assertRaises(ValueError):
            cube.intersection(longitude=(30, 500))

    def test_src_too_large(self):
        cube = create_cube(0, 400)
        with self.assertRaises(ValueError):
            cube.intersection(longitude=(10, 30))

    def test_missing_coord(self):
        cube = create_cube(0, 360)
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            cube.intersection(parrots=(10, 30))

    def test_multi_dim_coord(self):
        cube = create_cube(0, 360)
        with self.assertRaises(iris.exceptions.CoordinateMultiDimError):
            cube.intersection(surface_altitude=(10, 30))

    def test_null_region(self):
        # 10 <= v < 10
        cube = create_cube(0, 360)
        with self.assertRaises(IndexError):
            cube.intersection(longitude=(10, 10, False, False))


class Test_intersection__Lazy(tests.IrisTest):
    def test_real_data(self):
        cube = create_cube(0, 360)
        cube.data
        result = cube.intersection(longitude=(170, 190))
        self.assertFalse(result.has_lazy_data())
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(170, 191)
        )
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_real_data_wrapped(self):
        cube = create_cube(-180, 180)
        cube.data
        result = cube.intersection(longitude=(170, 190))
        self.assertFalse(result.has_lazy_data())
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(170, 191)
        )
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_lazy_data(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190))
        self.assertTrue(result.has_lazy_data())
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(170, 191)
        )
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_lazy_data_wrapped(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(170, 190))
        self.assertTrue(result.has_lazy_data())
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(170, 191)
        )
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)


class Test_intersection_Points(tests.IrisTest):
    def test_ignore_bounds(self):
        cube = create_cube(0, 30, bounds=True)
        result = cube.intersection(longitude=(9.5, 12.5), ignore_bounds=True)
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(10, 13)
        )
        self.assertArrayEqual(result.coord("longitude").bounds[0], [9.5, 10.5])
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [11.5, 12.5]
        )


# Check what happens with a regional, points-only circular intersection
# coordinate.
class Test_intersection__RegionalSrcModulus(tests.IrisTest):
    def test_request_subset(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(45, 50))
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(45, 51)
        )
        self.assertArrayEqual(result.data[0, 0], np.arange(5, 11))

    def test_request_left(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35, 45))
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(40, 46)
        )
        self.assertArrayEqual(result.data[0, 0], np.arange(0, 6))

    def test_request_right(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(55, 65))
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(55, 60)
        )
        self.assertArrayEqual(result.data[0, 0], np.arange(15, 20))

    def test_request_superset(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35, 65))
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(40, 60)
        )
        self.assertArrayEqual(result.data[0, 0], np.arange(0, 20))

    def test_request_subset_modulus(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(45 + 360, 50 + 360))
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(45 + 360, 51 + 360)
        )
        self.assertArrayEqual(result.data[0, 0], np.arange(5, 11))

    def test_request_left_modulus(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35 + 360, 45 + 360))
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(40 + 360, 46 + 360)
        )
        self.assertArrayEqual(result.data[0, 0], np.arange(0, 6))

    def test_request_right_modulus(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(55 + 360, 65 + 360))
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(55 + 360, 60 + 360)
        )
        self.assertArrayEqual(result.data[0, 0], np.arange(15, 20))

    def test_request_superset_modulus(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35 + 360, 65 + 360))
        self.assertArrayEqual(
            result.coord("longitude").points, np.arange(40 + 360, 60 + 360)
        )
        self.assertArrayEqual(result.data[0, 0], np.arange(0, 20))

    def test_tolerance_f4(self):
        cube = create_cube(0, 5)
        cube.coord("longitude").points = np.array(
            [0.0, 3.74999905, 7.49999809, 11.24999714, 14.99999619], dtype="f4"
        )
        result = cube.intersection(longitude=(0, 5))
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, np.array([0.0, 3.74999905])
        )
        self.assertArrayEqual(result.data[0, 0], np.array([0, 1]))

    def test_tolerance_f8(self):
        cube = create_cube(0, 5)
        cube.coord("longitude").points = np.array(
            [0.0, 3.74999905, 7.49999809, 11.24999714, 14.99999619], dtype="f8"
        )
        result = cube.intersection(longitude=(0, 5))
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, np.array([0.0, 3.74999905])
        )
        self.assertArrayEqual(result.data[0, 0], np.array([0, 1]))


# Check what happens with a global, points-only circular intersection
# coordinate.
class Test_intersection__GlobalSrcModulus(tests.IrisTest):
    def test_global_wrapped_extreme_increasing_base_period(self):
        # Ensure that we can correctly handle points defined at (base + period)
        cube = create_cube(-180.0, 180.0)
        lons = cube.coord("longitude")
        # Redefine longitude so that points at (base + period)
        lons.points = np.linspace(-180.0, 180, lons.points.size)
        result = cube.intersection(
            longitude=(lons.points.min(), lons.points.max())
        )
        self.assertArrayEqual(result.data, cube.data)

    def test_global_wrapped_extreme_decreasing_base_period(self):
        # Ensure that we can correctly handle points defined at (base + period)
        cube = create_cube(180.0, -180.0)
        lons = cube.coord("longitude")
        # Redefine longitude so that points at (base + period)
        lons.points = np.linspace(180.0, -180.0, lons.points.size)
        result = cube.intersection(
            longitude=(lons.points.min(), lons.points.max())
        )
        self.assertArrayEqual(result.data, cube.data)

    def test_global(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(0, 360))
        self.assertEqual(result.coord("longitude").points[0], 0)
        self.assertEqual(result.coord("longitude").points[-1], 359)
        self.assertEqual(result.data[0, 0, 0], 0)
        self.assertEqual(result.data[0, 0, -1], 359)

    def test_global_wrapped(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(-180, 180))
        self.assertEqual(result.coord("longitude").points[0], -180)
        self.assertEqual(result.coord("longitude").points[-1], 179)
        self.assertEqual(result.data[0, 0, 0], 180)
        self.assertEqual(result.data[0, 0, -1], 179)

    def test_aux_coord(self):
        cube = create_cube(0, 360)
        cube.replace_coord(
            iris.coords.AuxCoord.from_coord(cube.coord("longitude"))
        )
        result = cube.intersection(longitude=(0, 360))
        self.assertEqual(result.coord("longitude").points[0], 0)
        self.assertEqual(result.coord("longitude").points[-1], 359)
        self.assertEqual(result.data[0, 0, 0], 0)
        self.assertEqual(result.data[0, 0, -1], 359)

    def test_aux_coord_wrapped(self):
        cube = create_cube(0, 360)
        cube.replace_coord(
            iris.coords.AuxCoord.from_coord(cube.coord("longitude"))
        )
        result = cube.intersection(longitude=(-180, 180))
        self.assertEqual(result.coord("longitude").points[0], 0)
        self.assertEqual(result.coord("longitude").points[-1], -1)
        self.assertEqual(result.data[0, 0, 0], 0)
        self.assertEqual(result.data[0, 0, -1], 359)

    def test_aux_coord_non_contiguous_wrapped(self):
        cube = create_cube(0, 360)
        coord = iris.coords.AuxCoord.from_coord(cube.coord("longitude"))
        coord.points = (coord.points * 1.5) % 360
        cube.replace_coord(coord)
        result = cube.intersection(longitude=(-90, 90))
        self.assertEqual(result.coord("longitude").points[0], 0)
        self.assertEqual(result.coord("longitude").points[-1], 90)
        self.assertEqual(result.data[0, 0, 0], 0)
        self.assertEqual(result.data[0, 0, -1], 300)

    def test_decrementing(self):
        cube = create_cube(360, 0)
        result = cube.intersection(longitude=(40, 60))
        self.assertEqual(result.coord("longitude").points[0], 60)
        self.assertEqual(result.coord("longitude").points[-1], 40)
        self.assertEqual(result.data[0, 0, 0], 300)
        self.assertEqual(result.data[0, 0, -1], 320)

    def test_decrementing_wrapped(self):
        cube = create_cube(360, 0)
        result = cube.intersection(longitude=(-10, 10))
        self.assertEqual(result.coord("longitude").points[0], 10)
        self.assertEqual(result.coord("longitude").points[-1], -10)
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_no_wrap_after_modulus(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170 + 360, 190 + 360))
        self.assertEqual(result.coord("longitude").points[0], 170 + 360)
        self.assertEqual(result.coord("longitude").points[-1], 190 + 360)
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_wrap_after_modulus(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(170 + 360, 190 + 360))
        self.assertEqual(result.coord("longitude").points[0], 170 + 360)
        self.assertEqual(result.coord("longitude").points[-1], 190 + 360)
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_select_by_coord(self):
        cube = create_cube(0, 360)
        coord = iris.coords.DimCoord(0, "longitude", units="degrees")
        result = cube.intersection(iris.coords.CoordExtent(coord, 10, 30))
        self.assertEqual(result.coord("longitude").points[0], 10)
        self.assertEqual(result.coord("longitude").points[-1], 30)
        self.assertEqual(result.data[0, 0, 0], 10)
        self.assertEqual(result.data[0, 0, -1], 30)

    def test_inclusive_exclusive(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190, True, False))
        self.assertEqual(result.coord("longitude").points[0], 170)
        self.assertEqual(result.coord("longitude").points[-1], 189)
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 189)

    def test_exclusive_inclusive(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190, False))
        self.assertEqual(result.coord("longitude").points[0], 171)
        self.assertEqual(result.coord("longitude").points[-1], 190)
        self.assertEqual(result.data[0, 0, 0], 171)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_exclusive_exclusive(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190, False, False))
        self.assertEqual(result.coord("longitude").points[0], 171)
        self.assertEqual(result.coord("longitude").points[-1], 189)
        self.assertEqual(result.data[0, 0, 0], 171)
        self.assertEqual(result.data[0, 0, -1], 189)

    def test_single_point(self):
        # 10 <= v <= 10
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(10, 10))
        self.assertEqual(result.coord("longitude").points[0], 10)
        self.assertEqual(result.coord("longitude").points[-1], 10)
        self.assertEqual(result.data[0, 0, 0], 10)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_two_points(self):
        # -1.5 <= v <= 0.5
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(-1.5, 0.5))
        self.assertEqual(result.coord("longitude").points[0], -1)
        self.assertEqual(result.coord("longitude").points[-1], 0)
        self.assertEqual(result.data[0, 0, 0], 359)
        self.assertEqual(result.data[0, 0, -1], 0)

    def test_wrap_radians(self):
        cube = create_cube(0, 360)
        cube.coord("longitude").convert_units("radians")
        result = cube.intersection(longitude=(-1, 0.5))
        self.assertEqual(
            result.coord("longitude").points[0], -0.99483767363676634
        )
        self.assertEqual(
            result.coord("longitude").points[-1], 0.48869219055841207
        )
        self.assertEqual(result.data[0, 0, 0], 303)
        self.assertEqual(result.data[0, 0, -1], 28)

    def test_tolerance_bug(self):
        # Floating point changes introduced by wrapping mean
        # the resulting coordinate values are not equal to their
        # equivalents. This led to a bug that this test checks.
        cube = create_cube(0, 400)
        cube.coord("longitude").points = np.linspace(-179.55, 179.55, 400)
        result = cube.intersection(longitude=(125, 145))
        self.assertArrayAlmostEqual(
            result.coord("longitude").points,
            cube.coord("longitude").points[339:361],
        )

    def test_tolerance_bug_wrapped(self):
        cube = create_cube(0, 400)
        cube.coord("longitude").points = np.linspace(-179.55, 179.55, 400)
        result = cube.intersection(longitude=(-190, -170))
        # Expected result is the last 11 and first 11 points.
        expected = np.append(
            cube.coord("longitude").points[389:] - 360.0,
            cube.coord("longitude").points[:11],
        )
        self.assertArrayAlmostEqual(result.coord("longitude").points, expected)


# Check what happens with a global, points-and-bounds circular
# intersection coordinate.
class Test_intersection__ModulusBounds(tests.IrisTest):
    def test_global_wrapped_extreme_increasing_base_period(self):
        # Ensure that we can correctly handle bounds defined at (base + period)
        cube = create_cube(-180.0, 180.0, bounds=True)
        lons = cube.coord("longitude")
        result = cube.intersection(
            longitude=(lons.bounds.min(), lons.bounds.max())
        )
        self.assertArrayEqual(result.data, cube.data)

    def test_global_wrapped_extreme_decreasing_base_period(self):
        # Ensure that we can correctly handle bounds defined at (base + period)
        cube = create_cube(180.0, -180.0, bounds=True)
        lons = cube.coord("longitude")
        result = cube.intersection(
            longitude=(lons.bounds.min(), lons.bounds.max())
        )
        self.assertArrayEqual(result.data, cube.data)

    def test_misaligned_points_inside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(169.75, 190.25))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [169.5, 170.5]
        )
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [189.5, 190.5]
        )
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_misaligned_points_outside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(170.25, 189.75))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [169.5, 170.5]
        )
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [189.5, 190.5]
        )
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_misaligned_bounds(self):
        cube = create_cube(-180, 180, bounds=True)
        result = cube.intersection(longitude=(0, 360))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [-0.5, 0.5])
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [358.5, 359.5]
        )
        self.assertEqual(result.data[0, 0, 0], 180)
        self.assertEqual(result.data[0, 0, -1], 179)

    def test_misaligned_bounds_decreasing(self):
        cube = create_cube(180, -180, bounds=True)
        result = cube.intersection(longitude=(0, 360))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [359.5, 358.5]
        )
        self.assertArrayEqual(result.coord("longitude").points[-1], 0)
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [0.5, -0.5]
        )
        self.assertEqual(result.data[0, 0, 0], 181)
        self.assertEqual(result.data[0, 0, -1], 180)

    def test_aligned_inclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(170.5, 189.5))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [169.5, 170.5]
        )
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [189.5, 190.5]
        )
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_aligned_exclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(170.5, 189.5, False, False))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [170.5, 171.5]
        )
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [188.5, 189.5]
        )
        self.assertEqual(result.data[0, 0, 0], 171)
        self.assertEqual(result.data[0, 0, -1], 189)

    def test_negative_misaligned_points_inside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(-10.25, 10.25))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [-10.5, -9.5]
        )
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [9.5, 10.5]
        )
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_negative_misaligned_points_outside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(-9.75, 9.75))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [-10.5, -9.5]
        )
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [9.5, 10.5]
        )
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_negative_aligned_inclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(-10.5, 10.5))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [-11.5, -10.5]
        )
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [10.5, 11.5]
        )
        self.assertEqual(result.data[0, 0, 0], 349)
        self.assertEqual(result.data[0, 0, -1], 11)

    def test_negative_aligned_exclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(-10.5, 10.5, False, False))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [-10.5, -9.5]
        )
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [9.5, 10.5]
        )
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_decrementing(self):
        cube = create_cube(360, 0, bounds=True)
        result = cube.intersection(longitude=(40, 60))
        self.assertArrayEqual(
            result.coord("longitude").bounds[0], [60.5, 59.5]
        )
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [40.5, 39.5]
        )
        self.assertEqual(result.data[0, 0, 0], 300)
        self.assertEqual(result.data[0, 0, -1], 320)

    def test_decrementing_wrapped(self):
        cube = create_cube(360, 0, bounds=True)
        result = cube.intersection(longitude=(-10, 10))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [10.5, 9.5])
        self.assertArrayEqual(
            result.coord("longitude").bounds[-1], [-9.5, -10.5]
        )
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_numerical_tolerance(self):
        # test the tolerance on the coordinate value is not causing a
        # modulus wrapping
        cube = create_cube(28.5, 68.5, bounds=True)
        result = cube.intersection(longitude=(27.74, 68.61))
        self.assertAlmostEqual(result.coord("longitude").points[0], 28.5)
        self.assertAlmostEqual(result.coord("longitude").points[-1], 67.5)


def unrolled_cube():
    data = np.arange(5, dtype="f4")
    cube = Cube(data)
    cube.add_aux_coord(
        iris.coords.AuxCoord(
            [5.0, 10.0, 8.0, 5.0, 3.0], "longitude", units="degrees"
        ),
        0,
    )
    cube.add_aux_coord(
        iris.coords.AuxCoord([1.0, 3.0, -2.0, -1.0, -4.0], "latitude"), 0
    )
    return cube


# Check what happens with a "unrolled" scatter-point data with a circular
# intersection coordinate.
class Test_intersection__ScatterModulus(tests.IrisTest):
    def test_subset(self):
        cube = unrolled_cube()
        result = cube.intersection(longitude=(5, 8))
        self.assertArrayEqual(result.coord("longitude").points, [5, 8, 5])
        self.assertArrayEqual(result.data, [0, 2, 3])

    def test_subset_wrapped(self):
        cube = unrolled_cube()
        result = cube.intersection(longitude=(5 + 360, 8 + 360))
        self.assertArrayEqual(
            result.coord("longitude").points, [365, 368, 365]
        )
        self.assertArrayEqual(result.data, [0, 2, 3])

    def test_superset(self):
        cube = unrolled_cube()
        result = cube.intersection(longitude=(0, 15))
        self.assertArrayEqual(
            result.coord("longitude").points, [5, 10, 8, 5, 3]
        )
        self.assertArrayEqual(result.data, np.arange(5))


# Test the API of the cube interpolation method.
class Test_interpolate(tests.IrisTest):
    def setUp(self):
        self.cube = stock.simple_2d()

        self.scheme = mock.Mock(name="interpolation scheme")
        self.interpolator = mock.Mock(name="interpolator")
        self.interpolator.return_value = mock.sentinel.RESULT
        self.scheme.interpolator.return_value = self.interpolator
        self.collapse_coord = True

    def test_api(self):
        sample_points = (("foo", 0.5), ("bar", 0.6))
        result = self.cube.interpolate(
            sample_points, self.scheme, self.collapse_coord
        )
        self.scheme.interpolator.assert_called_once_with(
            self.cube, ("foo", "bar")
        )
        self.interpolator.assert_called_once_with(
            (0.5, 0.6), collapse_scalar=self.collapse_coord
        )
        self.assertIs(result, mock.sentinel.RESULT)


class Test_regrid(tests.IrisTest):
    def test(self):
        # Test that Cube.regrid() just defers to the regridder of the
        # given scheme.

        # Define a fake scheme and its associated regridder which just
        # capture their arguments and return them in place of the
        # regridded cube.
        class FakeRegridder:
            def __init__(self, *args):
                self.args = args

            def __call__(self, cube):
                return self.args + (cube,)

        class FakeScheme:
            def regridder(self, src, target):
                return FakeRegridder(self, src, target)

        cube = Cube(0)
        scheme = FakeScheme()
        result = cube.regrid(mock.sentinel.TARGET, scheme)
        self.assertEqual(result, (scheme, cube, mock.sentinel.TARGET, cube))


class Test_copy(tests.IrisTest):
    def _check_copy(self, cube, cube_copy):
        self.assertIsNot(cube_copy, cube)
        self.assertEqual(cube_copy, cube)
        self.assertIsNot(cube_copy.data, cube.data)
        if ma.isMaskedArray(cube.data):
            self.assertMaskedArrayEqual(cube_copy.data, cube.data)
            if cube.data.mask is not ma.nomask:
                # "No mask" is a constant : all other cases must be distinct.
                self.assertIsNot(cube_copy.data.mask, cube.data.mask)
        else:
            self.assertArrayEqual(cube_copy.data, cube.data)

    def test(self):
        cube = stock.simple_3d()
        self._check_copy(cube, cube.copy())

    def test_copy_ancillary_variables(self):
        cube = stock.simple_3d()
        avr = AncillaryVariable([2, 3], long_name="foo")
        cube.add_ancillary_variable(avr, 0)
        self._check_copy(cube, cube.copy())

    def test_copy_cell_measures(self):
        cube = stock.simple_3d()
        cms = CellMeasure([2, 3], long_name="foo")
        cube.add_cell_measure(cms, 0)
        self._check_copy(cube, cube.copy())

    def test__masked_emptymask(self):
        cube = Cube(ma.array([0, 1]))
        self._check_copy(cube, cube.copy())

    def test__masked_arraymask(self):
        cube = Cube(ma.array([0, 1], mask=[True, False]))
        self._check_copy(cube, cube.copy())

    def test__scalar(self):
        cube = Cube(0)
        self._check_copy(cube, cube.copy())

    def test__masked_scalar_emptymask(self):
        cube = Cube(ma.array(0))
        self._check_copy(cube, cube.copy())

    def test__masked_scalar_arraymask(self):
        cube = Cube(ma.array(0, mask=False))
        self._check_copy(cube, cube.copy())

    def test__lazy(self):
        cube = Cube(as_lazy_data(np.array([1, 0])))
        self._check_copy(cube, cube.copy())


class Test_dtype(tests.IrisTest):
    def setUp(self):
        self.dtypes = (
            np.dtype("int"),
            np.dtype("uint"),
            np.dtype("bool"),
            np.dtype("float"),
        )

    def test_real_data(self):
        for dtype in self.dtypes:
            data = np.array([0, 1], dtype=dtype)
            cube = Cube(data)
            self.assertEqual(cube.dtype, dtype)

    def test_real_data_masked__mask_unset(self):
        for dtype in self.dtypes:
            data = ma.array([0, 1], dtype=dtype)
            cube = Cube(data)
            self.assertEqual(cube.dtype, dtype)

    def test_real_data_masked__mask_set(self):
        for dtype in self.dtypes:
            data = ma.array([0, 1], dtype=dtype)
            data[0] = ma.masked
            cube = Cube(data)
            self.assertEqual(cube.dtype, dtype)

    def test_lazy_data(self):
        for dtype in self.dtypes:
            data = np.array([0, 1], dtype=dtype)
            cube = Cube(as_lazy_data(data))
            self.assertEqual(cube.dtype, dtype)
            # Check that accessing the dtype does not trigger loading
            # of the data.
            self.assertTrue(cube.has_lazy_data())

    def test_lazy_data_masked__mask_unset(self):
        for dtype in self.dtypes:
            data = ma.array([0, 1], dtype=dtype)
            cube = Cube(as_lazy_data(data))
            self.assertEqual(cube.dtype, dtype)
            # Check that accessing the dtype does not trigger loading
            # of the data.
            self.assertTrue(cube.has_lazy_data())

    def test_lazy_data_masked__mask_set(self):
        for dtype in self.dtypes:
            data = ma.array([0, 1], dtype=dtype)
            data[0] = ma.masked
            cube = Cube(as_lazy_data(data))
            self.assertEqual(cube.dtype, dtype)
            # Check that accessing the dtype does not trigger loading
            # of the data.
            self.assertTrue(cube.has_lazy_data())


class TestSubset(tests.IrisTest):
    def test_scalar_coordinate(self):
        cube = Cube(0, long_name="apricot", units="1")
        cube.add_aux_coord(DimCoord([0], long_name="banana", units="1"))
        result = cube.subset(cube.coord("banana"))
        self.assertEqual(cube, result)

    def test_dimensional_coordinate(self):
        cube = Cube(np.zeros((4)), long_name="tinned_peach", units="1")
        cube.add_dim_coord(
            DimCoord([0, 1, 2, 3], long_name="sixteen_ton_weight", units="1"),
            0,
        )
        result = cube.subset(cube.coord("sixteen_ton_weight"))
        self.assertEqual(cube, result)

    def test_missing_coordinate(self):
        cube = Cube(0, long_name="raspberry", units="1")
        cube.add_aux_coord(DimCoord([0], long_name="loganberry", units="1"))
        bad_coord = DimCoord([0], long_name="tiger", units="1")
        self.assertRaises(CoordinateNotFoundError, cube.subset, bad_coord)

    def test_different_coordinate(self):
        cube = Cube(0, long_name="raspberry", units="1")
        cube.add_aux_coord(DimCoord([0], long_name="loganberry", units="1"))
        different_coord = DimCoord([2], long_name="loganberry", units="1")
        result = cube.subset(different_coord)
        self.assertEqual(result, None)

    def test_not_coordinate(self):
        cube = Cube(0, long_name="peach", units="1")
        cube.add_aux_coord(DimCoord([0], long_name="crocodile", units="1"))
        self.assertRaises(ValueError, cube.subset, "Pointed Stick")


class Test_add_metadata(tests.IrisTest):
    def test_add_dim_coord(self):
        cube = Cube(np.arange(3))
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        cube.add_dim_coord(x_coord, 0)
        self.assertEqual(cube.coord("x"), x_coord)

    def test_add_aux_coord(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        x_coord = AuxCoord(points=np.arange(6).reshape(2, 3), long_name="x")
        cube.add_aux_coord(x_coord, [0, 1])
        self.assertEqual(cube.coord("x"), x_coord)

    def test_add_cell_measure(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        a_cell_measure = CellMeasure(
            np.arange(6).reshape(2, 3), long_name="area"
        )
        cube.add_cell_measure(a_cell_measure, [0, 1])
        self.assertEqual(cube.cell_measure("area"), a_cell_measure)

    def test_add_ancillary_variable(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        ancillary_variable = AncillaryVariable(
            data=np.arange(6).reshape(2, 3), long_name="detection quality"
        )
        cube.add_ancillary_variable(ancillary_variable, [0, 1])
        self.assertEqual(
            cube.ancillary_variable("detection quality"), ancillary_variable
        )

    def test_add_valid_aux_factory(self):
        cube = Cube(np.arange(8).reshape(2, 2, 2))
        delta = AuxCoord(points=[0, 1], long_name="delta", units="m")
        sigma = AuxCoord(points=[0, 1], long_name="sigma")
        orog = AuxCoord(np.arange(4).reshape(2, 2), units="m")
        cube.add_aux_coord(delta, 0)
        cube.add_aux_coord(sigma, 0)
        cube.add_aux_coord(orog, (1, 2))
        factory = HybridHeightFactory(delta=delta, sigma=sigma, orography=orog)
        self.assertIsNone(cube.add_aux_factory(factory))

    def test_error_for_add_invalid_aux_factory(self):
        cube = Cube(np.arange(8).reshape(2, 2, 2), long_name="bar")
        delta = AuxCoord(points=[0, 1], long_name="delta", units="m")
        sigma = AuxCoord(points=[0, 1], long_name="sigma")
        orog = AuxCoord(np.arange(4).reshape(2, 2), units="m", long_name="foo")
        cube.add_aux_coord(delta, 0)
        cube.add_aux_coord(sigma, 0)
        # Note orography is not added to the cube here
        factory = HybridHeightFactory(delta=delta, sigma=sigma, orography=orog)
        expected_error = (
            "foo coordinate for factory is not present on cube " "bar"
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            cube.add_aux_factory(factory)


class Test_remove_metadata(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        cube.add_dim_coord(x_coord, 1)
        z_coord = AuxCoord(points=np.arange(6).reshape(2, 3), long_name="z")
        cube.add_aux_coord(z_coord, [0, 1])
        a_cell_measure = CellMeasure(
            np.arange(6).reshape(2, 3), long_name="area"
        )
        self.b_cell_measure = CellMeasure(
            np.arange(6).reshape(2, 3), long_name="other_area",
        )
        cube.add_cell_measure(a_cell_measure, [0, 1])
        cube.add_cell_measure(self.b_cell_measure, [0, 1])
        ancillary_variable = AncillaryVariable(
            data=np.arange(6).reshape(2, 3), long_name="Quality of Detection"
        )
        cube.add_ancillary_variable(ancillary_variable, [0, 1])
        self.cube = cube

    def test_remove_dim_coord(self):
        self.cube.remove_coord(self.cube.coord("x"))
        self.assertEqual(self.cube.coords("x"), [])

    def test_remove_aux_coord(self):
        self.cube.remove_coord(self.cube.coord("z"))
        self.assertEqual(self.cube.coords("z"), [])

    def test_remove_cell_measure(self):
        self.cube.remove_cell_measure(self.cube.cell_measure("area"))
        self.assertEqual(
            self.cube._cell_measures_and_dims, [(self.b_cell_measure, (0, 1))]
        )

    def test_remove_cell_measure_by_name(self):
        self.cube.remove_cell_measure("area")
        self.assertEqual(
            self.cube._cell_measures_and_dims, [(self.b_cell_measure, (0, 1))]
        )

    def test_fail_remove_cell_measure_by_name(self):
        with self.assertRaises(CellMeasureNotFoundError):
            self.cube.remove_cell_measure("notarea")

    def test_remove_ancilliary_variable(self):
        self.cube.remove_ancillary_variable(
            self.cube.ancillary_variable("Quality of Detection")
        )
        self.assertEqual(self.cube._ancillary_variables_and_dims, [])


class Test__getitem_CellMeasure(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        cube.add_dim_coord(x_coord, 1)
        y_coord = DimCoord(points=np.array([5, 6]), long_name="y")
        cube.add_dim_coord(y_coord, 0)
        z_coord = AuxCoord(points=np.arange(6).reshape(2, 3), long_name="z")
        cube.add_aux_coord(z_coord, [0, 1])
        a_cell_measure = CellMeasure(
            np.arange(6).reshape(2, 3), long_name="area"
        )
        cube.add_cell_measure(a_cell_measure, [0, 1])
        self.cube = cube

    def test_cell_measure_2d(self):
        result = self.cube[0:2, 0:2]
        self.assertEqual(len(result.cell_measures()), 1)
        self.assertEqual(result.shape, result.cell_measures()[0].data.shape)

    def test_cell_measure_1d(self):
        result = self.cube[0, 0:2]
        self.assertEqual(len(result.cell_measures()), 1)
        self.assertEqual(result.shape, result.cell_measures()[0].data.shape)


class Test__getitem_AncillaryVariables(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        cube.add_dim_coord(x_coord, 1)
        y_coord = DimCoord(points=np.array([5, 6]), long_name="y")
        cube.add_dim_coord(y_coord, 0)
        z_coord = AuxCoord(points=np.arange(6).reshape(2, 3), long_name="z")
        cube.add_aux_coord(z_coord, [0, 1])
        a_ancillary_variable = AncillaryVariable(
            data=np.arange(6).reshape(2, 3), long_name="foo"
        )
        cube.add_ancillary_variable(a_ancillary_variable, [0, 1])
        self.cube = cube

    def test_ancillary_variables_2d(self):
        result = self.cube[0:2, 0:2]
        self.assertEqual(len(result.ancillary_variables()), 1)
        self.assertEqual(
            result.shape, result.ancillary_variables()[0].data.shape
        )

    def test_ancillary_variables_1d(self):
        result = self.cube[0, 0:2]
        self.assertEqual(len(result.ancillary_variables()), 1)
        self.assertEqual(
            result.shape, result.ancillary_variables()[0].data.shape
        )


class TestAncillaryVariables(tests.IrisTest):
    def setUp(self):
        cube = Cube(10 * np.arange(6).reshape(2, 3))
        self.ancill_var = AncillaryVariable(
            np.arange(6).reshape(2, 3),
            standard_name="number_of_observations",
            units="1",
        )
        cube.add_ancillary_variable(self.ancill_var, [0, 1])
        self.cube = cube

    def test_get_ancillary_variable(self):
        ancill_var = self.cube.ancillary_variable("number_of_observations")
        self.assertEqual(ancill_var, self.ancill_var)

    def test_get_ancillary_variables(self):
        ancill_vars = self.cube.ancillary_variables("number_of_observations")
        self.assertEqual(len(ancill_vars), 1)
        self.assertEqual(ancill_vars[0], self.ancill_var)

    def test_get_ancillary_variable_obj(self):
        ancill_vars = self.cube.ancillary_variables(self.ancill_var)
        self.assertEqual(len(ancill_vars), 1)
        self.assertEqual(ancill_vars[0], self.ancill_var)

    def test_fail_get_ancillary_variables(self):
        with self.assertRaises(AncillaryVariableNotFoundError):
            self.cube.ancillary_variable("other_ancill_var")

    def test_fail_get_ancillary_variables_obj(self):
        ancillary_variable = self.ancill_var.copy()
        ancillary_variable.long_name = "Number of observations at site"
        with self.assertRaises(AncillaryVariableNotFoundError):
            self.cube.ancillary_variable(ancillary_variable)

    def test_ancillary_variable_dims(self):
        ancill_var_dims = self.cube.ancillary_variable_dims(self.ancill_var)
        self.assertEqual(ancill_var_dims, (0, 1))

    def test_fail_ancill_variable_dims(self):
        ancillary_variable = self.ancill_var.copy()
        ancillary_variable.long_name = "Number of observations at site"
        with self.assertRaises(AncillaryVariableNotFoundError):
            self.cube.ancillary_variable_dims(ancillary_variable)


class TestCellMeasures(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        cube.add_dim_coord(x_coord, 1)
        z_coord = AuxCoord(points=np.arange(6).reshape(2, 3), long_name="z")
        cube.add_aux_coord(z_coord, [0, 1])
        self.a_cell_measure = CellMeasure(
            np.arange(6).reshape(2, 3), long_name="area", units="m2",
        )
        cube.add_cell_measure(self.a_cell_measure, [0, 1])
        self.cube = cube

    def test_get_cell_measure(self):
        cm = self.cube.cell_measure("area")
        self.assertEqual(cm, self.a_cell_measure)

    def test_get_cell_measures(self):
        cms = self.cube.cell_measures()
        self.assertEqual(len(cms), 1)
        self.assertEqual(cms[0], self.a_cell_measure)

    def test_get_cell_measures_obj(self):
        cms = self.cube.cell_measures(self.a_cell_measure)
        self.assertEqual(len(cms), 1)
        self.assertEqual(cms[0], self.a_cell_measure)

    def test_fail_get_cell_measure(self):
        with self.assertRaises(CellMeasureNotFoundError):
            _ = self.cube.cell_measure("notarea")

    def test_fail_get_cell_measures_obj(self):
        a_cell_measure = self.a_cell_measure.copy()
        a_cell_measure.units = "km2"
        with self.assertRaises(CellMeasureNotFoundError):
            _ = self.cube.cell_measure(a_cell_measure)

    def test_cell_measure_dims(self):
        cm_dims = self.cube.cell_measure_dims(self.a_cell_measure)
        self.assertEqual(cm_dims, (0, 1))

    def test_fail_cell_measure_dims(self):
        a_cell_measure = self.a_cell_measure.copy()
        a_cell_measure.units = "km2"
        with self.assertRaises(CellMeasureNotFoundError):
            _ = self.cube.cell_measure_dims(a_cell_measure)


class Test_transpose(tests.IrisTest):
    def setUp(self):
        self.data = np.arange(24).reshape(3, 2, 4)
        self.cube = Cube(self.data)
        self.lazy_cube = Cube(as_lazy_data(self.data))

    def test_lazy_data(self):
        cube = self.lazy_cube
        cube.transpose()
        self.assertTrue(cube.has_lazy_data())
        self.assertArrayEqual(self.data.T, cube.data)

    def test_real_data(self):
        self.cube.transpose()
        self.assertFalse(self.cube.has_lazy_data())
        self.assertIs(self.data.base, self.cube.data.base)
        self.assertArrayEqual(self.data.T, self.cube.data)

    def test_real_data__new_order(self):
        new_order = [2, 0, 1]
        self.cube.transpose(new_order)
        self.assertFalse(self.cube.has_lazy_data())
        self.assertIs(self.data.base, self.cube.data.base)
        self.assertArrayEqual(self.data.transpose(new_order), self.cube.data)

    def test_lazy_data__new_order(self):
        new_order = [2, 0, 1]
        cube = self.lazy_cube
        cube.transpose(new_order)
        self.assertTrue(cube.has_lazy_data())
        self.assertArrayEqual(self.data.transpose(new_order), cube.data)

    def test_lazy_data__transpose_order_ndarray(self):
        # Check that a transpose order supplied as an array does not trip up
        # a dask transpose operation.
        new_order = np.array([2, 0, 1])
        cube = self.lazy_cube
        cube.transpose(new_order)
        self.assertTrue(cube.has_lazy_data())
        self.assertArrayEqual(self.data.transpose(new_order), cube.data)

    def test_bad_transpose_order(self):
        exp_emsg = "Incorrect number of dimensions"
        with self.assertRaisesRegex(ValueError, exp_emsg):
            self.cube.transpose([1])

    def test_dim_coords(self):
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        self.cube.add_dim_coord(x_coord, 0)
        self.cube.transpose()
        self.assertEqual(self.cube._dim_coords_and_dims, [(x_coord, 2)])

    def test_aux_coords(self):
        x_coord = AuxCoord(
            points=np.array([[2, 3], [8, 4], [7, 9]]), long_name="x"
        )
        self.cube.add_aux_coord(x_coord, (0, 1))
        self.cube.transpose()
        self.assertEqual(self.cube._aux_coords_and_dims, [(x_coord, (2, 1))])

    def test_cell_measures(self):
        area_cm = CellMeasure(
            np.arange(12).reshape(3, 4), long_name="area of cells",
        )
        self.cube.add_cell_measure(area_cm, (0, 2))
        self.cube.transpose()
        self.assertEqual(
            self.cube._cell_measures_and_dims, [(area_cm, (2, 0))]
        )

    def test_ancillary_variables(self):
        ancill_var = AncillaryVariable(
            data=np.arange(8).reshape(2, 4), long_name="instrument error"
        )
        self.cube.add_ancillary_variable(ancill_var, (1, 2))
        self.cube.transpose()
        self.assertEqual(
            self.cube._ancillary_variables_and_dims, [(ancill_var, (1, 0))]
        )


class Test_convert_units(tests.IrisTest):
    def test_convert_unknown_units(self):
        cube = iris.cube.Cube(1)
        emsg = (
            "Cannot convert from unknown units. "
            'The "cube.units" attribute may be set directly.'
        )
        with self.assertRaisesRegex(UnitConversionError, emsg):
            cube.convert_units("mm day-1")

    def test_preserves_lazy(self):
        real_data = np.arange(12.0).reshape((3, 4))
        lazy_data = as_lazy_data(real_data)
        cube = iris.cube.Cube(lazy_data, units="m")
        real_data_ft = Unit("m").convert(real_data, "ft")
        cube.convert_units("ft")
        self.assertTrue(cube.has_lazy_data())
        self.assertArrayAllClose(cube.data, real_data_ft)


class Test__eq__data(tests.IrisTest):
    """Partial cube equality testing, for data type only."""

    def test_data_float_eq(self):
        cube1 = Cube([1.0])
        cube2 = Cube([1.0])
        self.assertTrue(cube1 == cube2)

    def test_data_float_eqtol(self):
        val1 = np.array(1.0, dtype=np.float32)
        # NOTE: Since v2.3, Iris uses "allclose".  Prior to that we used
        # "rtol=1e-8", and this example would *fail*.
        val2 = np.array(1.0 + 1.0e-6, dtype=np.float32)
        cube1 = Cube([val1])
        cube2 = Cube([val2])
        self.assertNotEqual(val1, val2)
        self.assertTrue(cube1 == cube2)

    def test_data_float_not_eq(self):
        val1 = 1.0
        val2 = 1.0 + 1.0e-4
        cube1 = Cube([1.0, val1])
        cube2 = Cube([1.0, val2])
        self.assertFalse(cube1 == cube2)

    def test_data_int_eq(self):
        cube1 = Cube([1, 2, 3])
        cube2 = Cube([1, 2, 3])
        self.assertTrue(cube1 == cube2)

    def test_data_int_not_eq(self):
        cube1 = Cube([1, 2, 3])
        cube2 = Cube([1, 2, 0])
        self.assertFalse(cube1 == cube2)

    # NOTE: since numpy v1.18, boolean array subtract is deprecated.
    def test_data_bool_eq(self):
        cube1 = Cube([True, False])
        cube2 = Cube([True, False])
        self.assertTrue(cube1 == cube2)

    def test_data_bool_not_eq(self):
        cube1 = Cube([True, False])
        cube2 = Cube([True, True])
        self.assertFalse(cube1 == cube2)


class Test__eq__meta(tests.IrisTest):
    def test_ancillary_fail(self):
        cube1 = Cube([0, 1])
        cube2 = Cube([0, 1])
        avr = AncillaryVariable([2, 3], long_name="foo")
        cube2.add_ancillary_variable(avr, 0)
        self.assertFalse(cube1 == cube2)

    def test_ancillary_reorder(self):
        cube1 = Cube([0, 1])
        cube2 = Cube([0, 1])
        avr1 = AncillaryVariable([2, 3], long_name="foo")
        avr2 = AncillaryVariable([4, 5], long_name="bar")
        # Add the same ancillary variables to cube1 and cube2 in
        # opposite orders.
        cube1.add_ancillary_variable(avr1, 0)
        cube1.add_ancillary_variable(avr2, 0)
        cube2.add_ancillary_variable(avr2, 0)
        cube2.add_ancillary_variable(avr1, 0)
        self.assertTrue(cube1 == cube2)

    def test_ancillary_diff_data(self):
        cube1 = Cube([0, 1])
        cube2 = Cube([0, 1])
        avr1 = AncillaryVariable([2, 3], long_name="foo")
        avr2 = AncillaryVariable([4, 5], long_name="foo")
        cube1.add_ancillary_variable(avr1, 0)
        cube2.add_ancillary_variable(avr2, 0)
        self.assertFalse(cube1 == cube2)

    def test_cell_measure_fail(self):
        cube1 = Cube([0, 1])
        cube2 = Cube([0, 1])
        cms = CellMeasure([2, 3], long_name="foo")
        cube2.add_cell_measure(cms, 0)
        self.assertFalse(cube1 == cube2)

    def test_cell_measure_reorder(self):
        cube1 = Cube([0, 1])
        cube2 = Cube([0, 1])
        cms1 = CellMeasure([2, 3], long_name="foo")
        cms2 = CellMeasure([4, 5], long_name="bar")
        # Add the same cell measure to cube1 and cube2 in
        # opposite orders.
        cube1.add_cell_measure(cms1, 0)
        cube1.add_cell_measure(cms2, 0)
        cube2.add_cell_measure(cms2, 0)
        cube2.add_cell_measure(cms1, 0)
        self.assertTrue(cube1 == cube2)

    def test_cell_measure_diff_data(self):
        cube1 = Cube([0, 1])
        cube2 = Cube([0, 1])
        cms1 = CellMeasure([2, 3], long_name="foo")
        cms2 = CellMeasure([4, 5], long_name="foo")
        cube1.add_cell_measure(cms1, 0)
        cube2.add_cell_measure(cms2, 0)
        self.assertFalse(cube1 == cube2)

    def test_cell_method_fail(self):
        cube1 = Cube([0, 1])
        cube2 = Cube([0, 1])
        cmth = CellMethod("mean", "time", "6hr")
        cube2.add_cell_method(cmth)
        self.assertFalse(cube1 == cube2)

    # Unlike cell measures, cell methods are order sensitive.
    def test_cell_method_reorder_fail(self):
        cube1 = Cube([0, 1])
        cube2 = Cube([0, 1])
        cmth1 = CellMethod("mean", "time", "6hr")
        cmth2 = CellMethod("mean", "time", "12hr")
        # Add the same cell method to cube1 and cube2 in
        # opposite orders.
        cube1.add_cell_method(cmth1)
        cube1.add_cell_method(cmth2)
        cube2.add_cell_method(cmth2)
        cube2.add_cell_method(cmth1)
        self.assertFalse(cube1 == cube2)

    def test_cell_method_correct_order(self):
        cube1 = Cube([0, 1])
        cube2 = Cube([0, 1])
        cmth1 = CellMethod("mean", "time", "6hr")
        cmth2 = CellMethod("mean", "time", "12hr")
        # Add the same cell method to cube1 and cube2 in
        # the same order.
        cube1.add_cell_method(cmth1)
        cube1.add_cell_method(cmth2)
        cube2.add_cell_method(cmth1)
        cube2.add_cell_method(cmth2)
        self.assertTrue(cube1 == cube2)


if __name__ == "__main__":
    tests.main()
