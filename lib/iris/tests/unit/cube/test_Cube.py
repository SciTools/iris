# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.cube.Cube` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from collections import namedtuple
from itertools import permutations
from unittest import mock

from cf_units import Unit
import dask.array as da
from distributed import Client
import numpy as np
import numpy.ma as ma
import pytest

from iris._lazy_data import as_lazy_data
import iris.analysis
from iris.analysis import MEAN, SUM, Aggregator, WeightedAggregator
import iris.aux_factory
from iris.aux_factory import HybridHeightFactory
from iris.common.metadata import BaseMetadata
import iris.coords
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, CellMethod, DimCoord
from iris.cube import Cube, CubeAttrsDict
import iris.exceptions
from iris.exceptions import (
    AncillaryVariableNotFoundError,
    CellMeasureNotFoundError,
    CoordinateNotFoundError,
    IrisUserWarning,
    IrisVagueMetadataWarning,
    UnitConversionError,
)
import iris.tests.stock as stock
from iris.tests.stock.mesh import sample_mesh, sample_mesh_cube, sample_meshcoord


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
    def _sample_data(self, dtype=("f4"), masked=False, fill_value=None, lazy=False):
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

    def _sample_cube(self, dtype=("f4"), masked=False, fill_value=None, lazy=False):
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

    def test_dim0_lazy_weights_none(self):
        cube_collapsed = self.cube.collapsed("y", MEAN, weights=None)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, [1.5, 2.5, 3.5])
        self.assertFalse(cube_collapsed.has_lazy_data())

    def test_dim1_lazy(self):
        cube_collapsed = self.cube.collapsed("x", MEAN)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, [1.0, 4.0])
        self.assertFalse(cube_collapsed.has_lazy_data())

    def test_dim1_lazy_weights_none(self):
        cube_collapsed = self.cube.collapsed("x", MEAN, weights=None)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, [1.0, 4.0])
        self.assertFalse(cube_collapsed.has_lazy_data())

    def test_multidims(self):
        # Check that MEAN works with multiple dims.
        cube_collapsed = self.cube.collapsed(("x", "y"), MEAN)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAllClose(cube_collapsed.data, 2.5)

    def test_multidims_weights_none(self):
        # Check that MEAN works with multiple dims.
        cube_collapsed = self.cube.collapsed(("x", "y"), MEAN, weights=None)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAllClose(cube_collapsed.data, 2.5)

    def test_non_lazy_aggregator(self):
        # An aggregator which doesn't have a lazy function should still work.
        dummy_agg = Aggregator("custom_op", lambda x, axis=None: np.mean(x, axis=axis))
        result = self.cube.collapsed("x", dummy_agg)
        self.assertFalse(result.has_lazy_data())
        self.assertArrayEqual(result.data, np.mean(self.data, axis=1))


class Test_collapsed__multidim_weighted_with_arr(tests.IrisTest):
    def setUp(self):
        self.data = np.arange(6.0).reshape((2, 3))
        self.lazydata = as_lazy_data(self.data)
        # Test cubes with (same-valued) real and lazy data
        cube_real = Cube(self.data, units="kg m-2 s-1")
        for i_dim, name in enumerate(("y", "x")):
            npts = cube_real.shape[i_dim]
            coord = DimCoord(np.arange(npts), long_name=name)
            cube_real.add_dim_coord(coord, i_dim)
        self.cube_real = cube_real
        self.cube_lazy = cube_real.copy(data=self.lazydata)
        # Test weights and expected result for a y-collapse
        self.y_weights = np.array([0.3, 0.5])
        self.full_weights_y = np.broadcast_to(
            self.y_weights.reshape((2, 1)), cube_real.shape
        )
        self.expected_result_y = np.array([1.875, 2.875, 3.875])
        # Test weights and expected result for an x-collapse
        self.x_weights = np.array([0.7, 0.4, 0.6])
        self.full_weights_x = np.broadcast_to(
            self.x_weights.reshape((1, 3)), cube_real.shape
        )
        self.expected_result_x = np.array([0.941176, 3.941176])

    def test_weighted_fullweights_real_y(self):
        # Supplying full-shape weights for collapsing over a single dimension.
        cube_collapsed = self.cube_real.collapsed(
            "y", MEAN, weights=self.full_weights_y
        )
        self.assertArrayAlmostEqual(cube_collapsed.data, self.expected_result_y)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")
        self.assertEqual(cube_collapsed.units.origin, "kg m-2 s-1")

    def test_weighted_fullweights_lazy_y(self):
        # Full-shape weights, lazy data :  Check lazy result, same values as real calc.
        cube_collapsed = self.cube_lazy.collapsed(
            "y", MEAN, weights=self.full_weights_y
        )
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, self.expected_result_y)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")

    def test_weighted_1dweights_real_y(self):
        # 1-D weights, real data :  Check same results as full-shape.
        cube_collapsed = self.cube_real.collapsed("y", MEAN, weights=self.y_weights)
        self.assertArrayAlmostEqual(cube_collapsed.data, self.expected_result_y)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")
        self.assertEqual(cube_collapsed.units.origin, "kg m-2 s-1")

    def test_weighted_1dweights_lazy_y(self):
        # 1-D weights, lazy data :  Check lazy result, same values as real calc.
        cube_collapsed = self.cube_lazy.collapsed("y", MEAN, weights=self.y_weights)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, self.expected_result_y)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")

    def test_weighted_fullweights_real_x(self):
        # Full weights, real data, ** collapse X ** :  as for 'y' case above
        cube_collapsed = self.cube_real.collapsed(
            "x", MEAN, weights=self.full_weights_x
        )
        self.assertArrayAlmostEqual(cube_collapsed.data, self.expected_result_x)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")
        self.assertEqual(cube_collapsed.units.origin, "kg m-2 s-1")

    def test_weighted_fullweights_lazy_x(self):
        # Full weights, lazy data, ** collapse X ** :  as for 'y' case above
        cube_collapsed = self.cube_lazy.collapsed(
            "x", MEAN, weights=self.full_weights_x
        )
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, self.expected_result_x)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")
        self.assertEqual(cube_collapsed.units.origin, "kg m-2 s-1")

    def test_weighted_1dweights_real_x(self):
        # 1-D weights, real data, ** collapse X ** :  as for 'y' case above
        cube_collapsed = self.cube_real.collapsed("x", MEAN, weights=self.x_weights)
        self.assertArrayAlmostEqual(cube_collapsed.data, self.expected_result_x)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")
        self.assertEqual(cube_collapsed.units.origin, "kg m-2 s-1")

    def test_weighted_1dweights_lazy_x(self):
        # 1-D weights, lazy data, ** collapse X ** :  as for 'y' case above
        cube_collapsed = self.cube_lazy.collapsed("x", MEAN, weights=self.x_weights)
        self.assertTrue(cube_collapsed.has_lazy_data())
        self.assertArrayAlmostEqual(cube_collapsed.data, self.expected_result_x)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")
        self.assertEqual(cube_collapsed.units.origin, "kg m-2 s-1")

    def test_weighted_sum_fullweights_adapt_units_real_y(self):
        # Check that units are adapted correctly (kg m-2 s-1 * 1 = kg m-2 s-1)
        cube_collapsed = self.cube_real.collapsed("y", SUM, weights=self.full_weights_y)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")
        self.assertEqual(cube_collapsed.units.origin, "kg m-2 s-1")

    def test_weighted_sum_fullweights_adapt_units_lazy_y(self):
        # Check that units are adapted correctly (kg m-2 s-1 * 1 = kg m-2 s-1)
        cube_collapsed = self.cube_lazy.collapsed("y", SUM, weights=self.full_weights_y)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")
        self.assertEqual(cube_collapsed.units.origin, "kg m-2 s-1")

    def test_weighted_sum_1dweights_adapt_units_real_y(self):
        # Check that units are adapted correctly (kg m-2 s-1 * 1 = kg m-2 s-1)
        # Note: the same test with lazy data fails:
        # https://github.com/SciTools/iris/issues/5083
        cube_collapsed = self.cube_real.collapsed("y", SUM, weights=self.y_weights)
        self.assertEqual(cube_collapsed.units, "kg m-2 s-1")
        self.assertEqual(cube_collapsed.units.origin, "kg m-2 s-1")

    def test_weighted_sum_with_unknown_units_real_y(self):
        # Check that units are adapted correctly ('unknown' * '1' = 'unknown')
        # Note: does not need to be adapted in subclasses since 'unknown'
        # multiplied by any unit is 'unknown'
        self.cube_real.units = "unknown"
        cube_collapsed = self.cube_real.collapsed(
            "y",
            SUM,
            weights=self.full_weights_y,
        )
        self.assertEqual(cube_collapsed.units, "unknown")

    def test_weighted_sum_with_unknown_units_lazy_y(self):
        # Check that units are adapted correctly ('unknown' * '1' = 'unknown')
        # Note: does not need to be adapted in subclasses since 'unknown'
        # multiplied by any unit is 'unknown'
        self.cube_lazy.units = "unknown"
        cube_collapsed = self.cube_lazy.collapsed(
            "y",
            SUM,
            weights=self.full_weights_y,
        )
        self.assertEqual(cube_collapsed.units, "unknown")


# Simply redo the tests of Test_collapsed__multidim_weighted_with_arr with
# other allowed objects for weights


class Test_collapsed__multidim_weighted_with_cube(
    Test_collapsed__multidim_weighted_with_arr
):
    def setUp(self):
        super().setUp()

        self.y_weights_original = self.y_weights
        self.full_weights_y_original = self.full_weights_y
        self.x_weights_original = self.x_weights
        self.full_weights_x_original = self.full_weights_x

        self.y_weights = self.cube_real[:, 0].copy(self.y_weights_original)
        self.y_weights.units = "m2"
        self.full_weights_y = self.cube_real.copy(self.full_weights_y_original)
        self.full_weights_y.units = "m2"
        self.x_weights = self.cube_real[0, :].copy(self.x_weights_original)
        self.full_weights_x = self.cube_real.copy(self.full_weights_x_original)

    def test_weighted_sum_fullweights_adapt_units_real_y(self):
        # Check that units are adapted correctly (kg m-2 s-1 * m2 = kg s-1)
        cube_collapsed = self.cube_real.collapsed("y", SUM, weights=self.full_weights_y)
        self.assertEqual(cube_collapsed.units, "kg s-1")

    def test_weighted_sum_fullweights_adapt_units_lazy_y(self):
        # Check that units are adapted correctly (kg m-2 s-1 * m2 = kg s-1)
        cube_collapsed = self.cube_lazy.collapsed("y", SUM, weights=self.full_weights_y)
        self.assertEqual(cube_collapsed.units, "kg s-1")

    def test_weighted_sum_1dweights_adapt_units_real_y(self):
        # Check that units are adapted correctly (kg m-2 s-1 * m2 = kg s-1)
        # Note: the same test with lazy data fails:
        # https://github.com/SciTools/iris/issues/5083
        cube_collapsed = self.cube_real.collapsed("y", SUM, weights=self.y_weights)
        self.assertEqual(cube_collapsed.units, "kg s-1")


class Test_collapsed__multidim_weighted_with_str(
    Test_collapsed__multidim_weighted_with_cube
):
    def setUp(self):
        super().setUp()

        self.full_weights_y = "full_y"
        self.full_weights_x = "full_x"
        self.y_weights = "y"
        self.x_weights = "1d_x"

        self.dim_metadata_full_y = iris.coords.CellMeasure(
            self.full_weights_y_original,
            long_name=self.full_weights_y,
            units="m2",
        )
        self.dim_metadata_full_x = iris.coords.AuxCoord(
            self.full_weights_x_original,
            long_name=self.full_weights_x,
            units="m2",
        )
        self.dim_metadata_1d_y = iris.coords.DimCoord(
            self.y_weights_original, long_name=self.y_weights, units="m2"
        )
        self.dim_metadata_1d_x = iris.coords.AncillaryVariable(
            self.x_weights_original, long_name=self.x_weights, units="m2"
        )

        for cube in (self.cube_real, self.cube_lazy):
            cube.add_cell_measure(self.dim_metadata_full_y, (0, 1))
            cube.add_aux_coord(self.dim_metadata_full_x, (0, 1))
            cube.remove_coord("y")
            cube.add_dim_coord(self.dim_metadata_1d_y, 0)
            cube.add_ancillary_variable(self.dim_metadata_1d_x, 1)


class Test_collapsed__multidim_weighted_with_dim_metadata(
    Test_collapsed__multidim_weighted_with_str
):
    def setUp(self):
        super().setUp()

        self.full_weights_y = self.dim_metadata_full_y
        self.full_weights_x = self.dim_metadata_full_x
        self.y_weights = self.dim_metadata_1d_y
        self.x_weights = self.dim_metadata_1d_x


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
            self.assertIn(
                mock.call(msg.format(coord), category=IrisUserWarning),
                warn.call_args_list,
            )

    def _assert_nowarn_collapse_without_weight(self, coords, warn):
        # Ensure that warning is not raised.
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


class Test_collapsed_coord_with_3_bounds(tests.IrisTest):
    def setUp(self):
        self.cube = Cube([1, 2])

        bounds = [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]
        lat = AuxCoord([1.0, 2.0], bounds=bounds, standard_name="latitude")
        lon = AuxCoord([1.0, 2.0], bounds=bounds, standard_name="longitude")

        self.cube.add_aux_coord(lat, 0)
        self.cube.add_aux_coord(lon, 0)

    def _assert_warn_cannot_check_contiguity(self, warn):
        # Ensure that warning is raised.
        for coord in ["latitude", "longitude"]:
            msg = (
                f"Cannot check if coordinate is contiguous: Invalid "
                f"operation for '{coord}', with 3 bound(s). Contiguous "
                f"bounds are only defined for 1D coordinates with 2 "
                f"bounds. Metadata may not be fully descriptive for "
                f"'{coord}'. Ignoring bounds."
            )
            self.assertIn(
                mock.call(msg, category=IrisVagueMetadataWarning),
                warn.call_args_list,
            )

    def _assert_cube_as_expected(self, cube):
        """Ensure that cube data and coordinates are as expected."""
        self.assertArrayEqual(cube.data, np.array(3))

        lat = cube.coord("latitude")
        self.assertArrayAlmostEqual(lat.points, np.array([1.5]))
        self.assertArrayAlmostEqual(lat.bounds, np.array([[1.0, 2.0]]))

        lon = cube.coord("longitude")
        self.assertArrayAlmostEqual(lon.points, np.array([1.5]))
        self.assertArrayAlmostEqual(lon.bounds, np.array([[1.0, 2.0]]))

    def test_collapsed_lat_with_3_bounds(self):
        """Collapse latitude with 3 bounds."""
        with mock.patch("warnings.warn") as warn:
            collapsed_cube = self.cube.collapsed("latitude", SUM)
        self._assert_warn_cannot_check_contiguity(warn)
        self._assert_cube_as_expected(collapsed_cube)

    def test_collapsed_lon_with_3_bounds(self):
        """Collapse longitude with 3 bounds."""
        with mock.patch("warnings.warn") as warn:
            collapsed_cube = self.cube.collapsed("longitude", SUM)
        self._assert_warn_cannot_check_contiguity(warn)
        self._assert_cube_as_expected(collapsed_cube)

    def test_collapsed_lat_lon_with_3_bounds(self):
        """Collapse latitude and longitude with 3 bounds."""
        with mock.patch("warnings.warn") as warn:
            collapsed_cube = self.cube.collapsed(["latitude", "longitude"], SUM)
        self._assert_warn_cannot_check_contiguity(warn)
        self._assert_cube_as_expected(collapsed_cube)


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
            "    Ancillary variables:\n"
            "        status_flag                     x       -"
        )
        self.assertEqual(expected_summary, cube.summary())

    def test_similar_coords(self):
        coord1 = AuxCoord(42, long_name="foo", attributes=dict(bar=np.array([2, 5])))
        coord2 = coord1.copy()
        coord2.attributes = dict(bar="baz")
        for coord in [coord1, coord2]:
            self.cube.add_aux_coord(coord)
        self.assertIn("baz", self.cube.summary())

    def test_long_components(self):
        # Check that components with long names 'stretch' the printout correctly.
        cube = Cube(np.zeros((20, 20, 20)), units=1)
        dimco = DimCoord(np.arange(20), long_name="dimco")
        auxco = AuxCoord(np.zeros(20), long_name="auxco")
        ancil = AncillaryVariable(np.zeros(20), long_name="ancil")
        cellm = CellMeasure(np.zeros(20), long_name="cellm")
        cube.add_dim_coord(dimco, 0)
        cube.add_aux_coord(auxco, 0)
        cube.add_cell_measure(cellm, 1)
        cube.add_ancillary_variable(ancil, 2)

        original_summary = cube.summary()
        long_name = "long_name______________________________________"
        for component in (dimco, auxco, ancil, cellm):
            # For each (type of) component, set a long name so the header columns get shifted.
            old_name = component.name()
            component.rename(long_name)
            new_summary = cube.summary()
            component.rename(old_name)  # Put each back the way it was afterwards

            # Check that the resulting 'stretched' output has dimension columns aligned correctly.
            lines = new_summary.split("\n")
            header = lines[0]
            colon_inds = [i_char for i_char, char in enumerate(header) if char == ":"]
            for line in lines[1:]:
                # Replace all '-' with 'x' to make checking easier, and add a final buffer space.
                line = line.replace("-", "x") + " "
                if " x " in line:
                    # For lines with any columns : check that columns are where expected
                    for col_ind in colon_inds:
                        # Chop out chars before+after each expected column.
                        self.assertEqual(line[col_ind - 1 : col_ind + 2], " x ")

            # Finally also: compare old with new, but replacing new name and ignoring spacing differences
            def collapse_space(string):
                # Replace all multiple spaces with a single space.
                while "  " in string:
                    string = string.replace("  ", " ")
                return string

            self.assertEqual(
                collapse_space(new_summary).replace(long_name, old_name),
                collapse_space(original_summary),
            )


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


class Test_rolling_window(tests.IrisTest):
    def setUp(self):
        self.cube = Cube(np.arange(6), units="kg")
        self.multi_dim_cube = Cube(np.arange(36).reshape(6, 6))
        val_coord = DimCoord([0, 1, 2, 3, 4, 5], long_name="val", units="s")
        month_coord = AuxCoord(
            ["jan", "feb", "mar", "apr", "may", "jun"], long_name="month"
        )
        extra_coord = AuxCoord([0, 1, 2, 3, 4, 5], long_name="extra")
        self.cube.add_dim_coord(val_coord, 0)
        self.cube.add_aux_coord(month_coord, 0)
        self.multi_dim_cube.add_dim_coord(val_coord, 0)
        self.multi_dim_cube.add_aux_coord(extra_coord, 1)
        self.ancillary_variable = AncillaryVariable([0, 1, 2, 0, 1, 2], long_name="foo")
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
            units="s",
        )
        month_coord = AuxCoord(
            np.array(["jan|feb|mar", "feb|mar|apr", "mar|apr|may", "apr|may|jun"]),
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
        res_cube = self.cube.rolling_window("val", iris.analysis.MEAN, window, mdtol=0)
        expected_result = ma.array(
            [-99.0, 1.5, 2.5, -99.0, -99.0],
            mask=[True, False, False, True, True],
            dtype=np.float64,
        )
        self.assertMaskedArrayEqual(expected_result, res_cube.data)

    def test_ancillary_variables_and_cell_measures_kept(self):
        res_cube = self.multi_dim_cube.rolling_window("val", self.mock_agg, 3)
        self.assertEqual(res_cube.ancillary_variables(), [self.ancillary_variable])
        self.assertEqual(res_cube.cell_measures(), [self.cell_measure])

    def test_ancillary_variables_and_cell_measures_removed(self):
        res_cube = self.multi_dim_cube.rolling_window("extra", self.mock_agg, 3)
        self.assertEqual(res_cube.ancillary_variables(), [])
        self.assertEqual(res_cube.cell_measures(), [])

    def test_weights_arr(self):
        weights = np.array([0, 0, 1, 0, 2])
        res_cube = self.cube.rolling_window("val", SUM, 5, weights=weights)
        np.testing.assert_array_equal(res_cube.data, [10, 13])
        self.assertEqual(res_cube.units, "kg")

    def test_weights_cube(self):
        weights = Cube([0, 0, 1, 0, 2], units="m2")
        res_cube = self.cube.rolling_window("val", SUM, 5, weights=weights)
        np.testing.assert_array_equal(res_cube.data, [10, 13])
        self.assertEqual(res_cube.units, "kg m2")

    def test_weights_str(self):
        weights = "val"
        res_cube = self.cube.rolling_window("val", SUM, 6, weights=weights)
        np.testing.assert_array_equal(res_cube.data, [55])
        self.assertEqual(res_cube.units, "kg s")

    def test_weights_dim_coord(self):
        weights = self.cube.coord("val")
        res_cube = self.cube.rolling_window("val", SUM, 6, weights=weights)
        np.testing.assert_array_equal(res_cube.data, [55])
        self.assertEqual(res_cube.units, "kg s")


class Test_slices_dim_order(tests.IrisTest):
    """Test the capability of iris.cube.Cube.slices().

    Test the capability of iris.cube.Cube.slices(), including its
    ability to correctly re-order the dimensions.
    """

    def setUp(self):
        """Setup a 4D iris cube, each dimension is length 1.
        The dimensions are;
            dim1: time
            dim2: height
            dim3: latitude
            dim4: longitude.
        """
        self.cube = iris.cube.Cube(np.array([[[[8.0]]]]))
        self.cube.add_dim_coord(iris.coords.DimCoord([0], "time"), [0])
        self.cube.add_dim_coord(iris.coords.DimCoord([0], "height"), [1])
        self.cube.add_dim_coord(iris.coords.DimCoord([0], "latitude"), [2])
        self.cube.add_dim_coord(iris.coords.DimCoord([0], "longitude"), [3])

    @staticmethod
    def expected_cube_setup(dim1name, dim2name, dim3name):
        """expected_cube_setup.

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
        """check_order.

        Does two things:
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
        self.exp_iter_1d = range(len(self.cube.coord("model_level_number").points))
        self.exp_iter_2d = np.ndindex(6, 70, 1, 1)
        # Define maximum number of interactions for particularly long
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
        iris.coords.DimCoord([0, 20, 40, 80], long_name="level_height", units="m"),
        0,
    )
    cube.add_aux_coord(
        iris.coords.AuxCoord([1.0, 0.9, 0.8, 0.6], long_name="sigma", units="1"),
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
        self.assertArrayEqual(result.coord("longitude").points, np.arange(170, 191))
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_real_data_wrapped(self):
        cube = create_cube(-180, 180)
        cube.data
        result = cube.intersection(longitude=(170, 190))
        self.assertFalse(result.has_lazy_data())
        self.assertArrayEqual(result.coord("longitude").points, np.arange(170, 191))
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_lazy_data(self):
        cube = create_cube(0, 360)
        result = cube.intersection(longitude=(170, 190))
        self.assertTrue(result.has_lazy_data())
        self.assertArrayEqual(result.coord("longitude").points, np.arange(170, 191))
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_lazy_data_wrapped(self):
        cube = create_cube(-180, 180)
        result = cube.intersection(longitude=(170, 190))
        self.assertTrue(result.has_lazy_data())
        self.assertArrayEqual(result.coord("longitude").points, np.arange(170, 191))
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)


class Test_intersection_Points(tests.IrisTest):
    def test_ignore_bounds(self):
        cube = create_cube(0, 30, bounds=True)
        result = cube.intersection(longitude=(9.5, 12.5), ignore_bounds=True)
        self.assertArrayEqual(result.coord("longitude").points, np.arange(10, 13))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [9.5, 10.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [11.5, 12.5])


# Check what happens with a regional, points-only circular intersection
# coordinate.
class Test_intersection__RegionalSrcModulus(tests.IrisTest):
    def test_request_subset(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(45, 50))
        self.assertArrayEqual(result.coord("longitude").points, np.arange(45, 51))
        self.assertArrayEqual(result.data[0, 0], np.arange(5, 11))

    def test_request_left(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35, 45))
        self.assertArrayEqual(result.coord("longitude").points, np.arange(40, 46))
        self.assertArrayEqual(result.data[0, 0], np.arange(0, 6))

    def test_request_right(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(55, 65))
        self.assertArrayEqual(result.coord("longitude").points, np.arange(55, 60))
        self.assertArrayEqual(result.data[0, 0], np.arange(15, 20))

    def test_request_superset(self):
        cube = create_cube(40, 60)
        result = cube.intersection(longitude=(35, 65))
        self.assertArrayEqual(result.coord("longitude").points, np.arange(40, 60))
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
        result = cube.intersection(longitude=(lons.points.min(), lons.points.max()))
        self.assertArrayEqual(result.data, cube.data)

    def test_global_wrapped_extreme_decreasing_base_period(self):
        # Ensure that we can correctly handle points defined at (base + period)
        cube = create_cube(180.0, -180.0)
        lons = cube.coord("longitude")
        # Redefine longitude so that points at (base + period)
        lons.points = np.linspace(180.0, -180.0, lons.points.size)
        result = cube.intersection(longitude=(lons.points.min(), lons.points.max()))
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
        cube.replace_coord(iris.coords.AuxCoord.from_coord(cube.coord("longitude")))
        result = cube.intersection(longitude=(0, 360))
        self.assertEqual(result.coord("longitude").points[0], 0)
        self.assertEqual(result.coord("longitude").points[-1], 359)
        self.assertEqual(result.data[0, 0, 0], 0)
        self.assertEqual(result.data[0, 0, -1], 359)

    def test_aux_coord_wrapped(self):
        cube = create_cube(0, 360)
        cube.replace_coord(iris.coords.AuxCoord.from_coord(cube.coord("longitude")))
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
        self.assertArrayAllClose(
            result.coord("longitude").points, np.arange(-57, 29) * np.pi / 180
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
        result = cube.intersection(longitude=(lons.bounds.min(), lons.bounds.max()))
        self.assertArrayEqual(result.data, cube.data)

    def test_global_wrapped_extreme_decreasing_base_period(self):
        # Ensure that we can correctly handle bounds defined at (base + period)
        cube = create_cube(180.0, -180.0, bounds=True)
        lons = cube.coord("longitude")
        result = cube.intersection(longitude=(lons.bounds.min(), lons.bounds.max()))
        self.assertArrayEqual(result.data, cube.data)

    def test_misaligned_points_inside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(169.75, 190.25))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [169.5, 170.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [189.5, 190.5])
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_misaligned_points_outside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(170.25, 189.75))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [169.5, 170.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [189.5, 190.5])
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_misaligned_bounds(self):
        cube = create_cube(-180, 180, bounds=True)
        result = cube.intersection(longitude=(0, 360))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [-0.5, 0.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [358.5, 359.5])
        self.assertEqual(result.data[0, 0, 0], 180)
        self.assertEqual(result.data[0, 0, -1], 179)

    def test_misaligned_bounds_decreasing(self):
        cube = create_cube(180, -180, bounds=True)
        result = cube.intersection(longitude=(0, 360))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [359.5, 358.5])
        self.assertArrayEqual(result.coord("longitude").points[-1], 0)
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [0.5, -0.5])
        self.assertEqual(result.data[0, 0, 0], 181)
        self.assertEqual(result.data[0, 0, -1], 180)

    def test_aligned_inclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(170.5, 189.5))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [169.5, 170.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [189.5, 190.5])
        self.assertEqual(result.data[0, 0, 0], 170)
        self.assertEqual(result.data[0, 0, -1], 190)

    def test_aligned_exclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(170.5, 189.5, False, False))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [170.5, 171.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [188.5, 189.5])
        self.assertEqual(result.data[0, 0, 0], 171)
        self.assertEqual(result.data[0, 0, -1], 189)

    def test_aligned_bounds_at_modulus(self):
        cube = create_cube(-179.5, 180.5, bounds=True)
        result = cube.intersection(longitude=(0, 360))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [0, 1])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [359, 360])
        self.assertEqual(result.data[0, 0, 0], 180)
        self.assertEqual(result.data[0, 0, -1], 179)

    def test_negative_aligned_bounds_at_modulus(self):
        cube = create_cube(0.5, 360.5, bounds=True)
        result = cube.intersection(longitude=(-180, 180))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [-180, -179])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [179, 180])
        self.assertEqual(result.data[0, 0, 0], 180)
        self.assertEqual(result.data[0, 0, -1], 179)

    def test_negative_misaligned_points_inside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(-10.25, 10.25))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [-10.5, -9.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [9.5, 10.5])
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_negative_misaligned_points_outside(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(-9.75, 9.75))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [-10.5, -9.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [9.5, 10.5])
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_negative_aligned_inclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(-10.5, 10.5))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [-11.5, -10.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [10.5, 11.5])
        self.assertEqual(result.data[0, 0, 0], 349)
        self.assertEqual(result.data[0, 0, -1], 11)

    def test_negative_aligned_exclusive(self):
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(-10.5, 10.5, False, False))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [-10.5, -9.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [9.5, 10.5])
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_decrementing(self):
        cube = create_cube(360, 0, bounds=True)
        result = cube.intersection(longitude=(40, 60))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [60.5, 59.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [40.5, 39.5])
        self.assertEqual(result.data[0, 0, 0], 300)
        self.assertEqual(result.data[0, 0, -1], 320)

    def test_decrementing_wrapped(self):
        cube = create_cube(360, 0, bounds=True)
        result = cube.intersection(longitude=(-10, 10))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [10.5, 9.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [-9.5, -10.5])
        self.assertEqual(result.data[0, 0, 0], 350)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_numerical_tolerance(self):
        # test the tolerance on the coordinate value is not causing a
        # modulus wrapping
        cube = create_cube(28.5, 68.5, bounds=True)
        result = cube.intersection(longitude=(27.74, 68.61))
        result_lons = result.coord("longitude")
        self.assertAlmostEqual(result_lons.points[0], 28.5)
        self.assertAlmostEqual(result_lons.points[-1], 67.5)
        dtype = result_lons.dtype
        np.testing.assert_array_almost_equal(
            result_lons.bounds[0], np.array([28.0, 29.0], dtype=dtype)
        )
        np.testing.assert_array_almost_equal(
            result_lons.bounds[-1], np.array([67.0, 68.0], dtype=dtype)
        )

    def test_numerical_tolerance_wrapped(self):
        # test the tolerance on the coordinate value causes modulus wrapping
        # where appropriate
        cube = create_cube(0.5, 3600.5, bounds=True)
        lons = cube.coord("longitude")
        lons.points = lons.points / 10
        lons.bounds = lons.bounds / 10
        result = cube.intersection(longitude=(-60, 60))
        result_lons = result.coord("longitude")
        self.assertAlmostEqual(result_lons.points[0], -60.05)
        self.assertAlmostEqual(result_lons.points[-1], 60.05)
        dtype = result_lons.dtype
        np.testing.assert_array_almost_equal(
            result_lons.bounds[0], np.array([-60.1, -60.0], dtype=dtype)
        )
        np.testing.assert_array_almost_equal(
            result_lons.bounds[-1], np.array([60.0, 60.1], dtype=dtype)
        )

    def test_ignore_bounds_wrapped(self):
        # Test `ignore_bounds` fully ignores bounds when wrapping
        cube = create_cube(0, 360, bounds=True)
        result = cube.intersection(longitude=(10.25, 370.25), ignore_bounds=True)
        # Expect points 11..370 not bounds [9.5, 10.5] .. [368.5, 369.5]
        self.assertArrayEqual(result.coord("longitude").bounds[0], [10.5, 11.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [369.5, 370.5])
        self.assertEqual(result.data[0, 0, 0], 11)
        self.assertEqual(result.data[0, 0, -1], 10)

    def test_within_cell(self):
        # Test cell is included when it entirely contains the requested range
        cube = create_cube(0, 10, bounds=True)
        result = cube.intersection(longitude=(0.7, 0.8))
        self.assertArrayEqual(result.coord("longitude").bounds[0], [0.5, 1.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [0.5, 1.5])
        self.assertEqual(result.data[0, 0, 0], 1)
        self.assertEqual(result.data[0, 0, -1], 1)

    def test_threshold_half(self):
        cube = create_cube(0, 10, bounds=True)
        result = cube.intersection(longitude=(1, 6.999), threshold=0.5)
        self.assertArrayEqual(result.coord("longitude").bounds[0], [0.5, 1.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [5.5, 6.5])
        self.assertEqual(result.data[0, 0, 0], 1)
        self.assertEqual(result.data[0, 0, -1], 6)

    def test_threshold_full(self):
        cube = create_cube(0, 10, bounds=True)
        result = cube.intersection(longitude=(0.5, 7.499), threshold=1)
        self.assertArrayEqual(result.coord("longitude").bounds[0], [0.5, 1.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [5.5, 6.5])
        self.assertEqual(result.data[0, 0, 0], 1)
        self.assertEqual(result.data[0, 0, -1], 6)

    def test_threshold_wrapped(self):
        # Test that a cell is wrapped to `maximum` if required to exceed
        # the threshold
        cube = create_cube(-180, 180, bounds=True)
        result = cube.intersection(longitude=(0.4, 360.4), threshold=0.2)
        self.assertArrayEqual(result.coord("longitude").bounds[0], [0.5, 1.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [359.5, 360.5])
        self.assertEqual(result.data[0, 0, 0], 181)
        self.assertEqual(result.data[0, 0, -1], 180)

    def test_threshold_wrapped_gap(self):
        # Test that a cell is wrapped to `maximum` if required to exceed
        # the threshold (even with a gap in the range)
        cube = create_cube(-180, 180, bounds=True)
        result = cube.intersection(longitude=(0.4, 360.35), threshold=0.2)
        self.assertArrayEqual(result.coord("longitude").bounds[0], [0.5, 1.5])
        self.assertArrayEqual(result.coord("longitude").bounds[-1], [359.5, 360.5])
        self.assertEqual(result.data[0, 0, 0], 181)
        self.assertEqual(result.data[0, 0, -1], 180)


def unrolled_cube():
    data = np.arange(5, dtype="f4")
    cube = Cube(data)
    cube.add_aux_coord(
        iris.coords.AuxCoord([5.0, 10.0, 8.0, 5.0, 3.0], "longitude", units="degrees"),
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
        self.assertArrayEqual(result.coord("longitude").points, [365, 368, 365])
        self.assertArrayEqual(result.data, [0, 2, 3])

    def test_superset(self):
        cube = unrolled_cube()
        result = cube.intersection(longitude=(0, 15))
        self.assertArrayEqual(result.coord("longitude").points, [5, 10, 8, 5, 3])
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
        result = self.cube.interpolate(sample_points, self.scheme, self.collapse_coord)
        self.scheme.interpolator.assert_called_once_with(self.cube, ("foo", "bar"))
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
        self.assertIsNot(cube_copy.core_data(), cube.core_data())
        if ma.isMaskedArray(cube.data):
            self.assertMaskedArrayEqual(cube_copy.data, cube.data)
            if cube.data.mask is not ma.nomask:
                # "No mask" is a constant : all other cases must be distinct.
                self.assertIsNot(cube_copy.core_data().mask, cube.core_data().mask)
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
        # 2022-11-02: Dask's current behaviour is that the computed array will
        #  be the same for cube and cube.copy(), even if the Dask arrays are
        #  different.
        cube = Cube(as_lazy_data(np.array([1, 0])))
        self._check_copy(cube, cube.copy())


def _add_test_meshcube(self, nomesh=False, n_z=2, **meshcoord_kwargs):
    """Common setup action : Create a standard mesh test cube with a variety of coords, and save the cube and various of
    its components as properties of the 'self' TestCase.

    """
    nomesh_faces = 5 if nomesh else None
    cube, parts = sample_mesh_cube(
        nomesh_faces=nomesh_faces, n_z=n_z, with_parts=True, **meshcoord_kwargs
    )
    mesh, zco, mesh_dimco, auxco_x, meshx, meshy = parts
    self.mesh = mesh
    self.dimco_z = zco
    self.dimco_mesh = mesh_dimco
    if not nomesh:
        self.meshco_x = meshx
        self.meshco_y = meshy
    self.auxco_x = auxco_x
    self.allcoords = [meshx, meshy, zco, mesh_dimco, auxco_x]
    self.cube = cube


class Test_coords__mesh_coords(tests.IrisTest):
    """Checking *only* the new "mesh_coords" keyword of the coord/coords methods.

    This is *not* attached to the existing tests for this area, as they are
    very old and patchy legacy tests.  See: iris.tests.test_cdm.TestQueryCoord.

    """

    def setUp(self):
        # Create a standard test cube with a variety of types of coord.
        _add_test_meshcube(self)

    def _assert_lists_equal(self, items_a, items_b):
        """Check that two lists of coords, cubes etc contain the same things.
        Lists must contain the same items, including any repeats, but can be in
        a different order.

        """

        # Compare (and thus sort) by their *common* metadata.
        def sortkey(item):
            return BaseMetadata.from_metadata(item.metadata)

        items_a = sorted(items_a, key=sortkey)
        items_b = sorted(items_b, key=sortkey)
        self.assertEqual(items_a, items_b)

    def test_coords__all__meshcoords_default(self):
        # coords() includes mesh-coords along with the others.
        result = self.cube.coords()
        expected = self.allcoords
        self._assert_lists_equal(expected, result)

    def test_coords__all__meshcoords_only(self):
        # Coords(mesh_coords=True) returns only mesh-coords.
        result = self.cube.coords(mesh_coords=True)
        expected = [self.meshco_x, self.meshco_y]
        self._assert_lists_equal(expected, result)

    def test_coords__all__meshcoords_omitted(self):
        # Coords(mesh_coords=False) omits the mesh-coords.
        result = self.cube.coords(mesh_coords=False)
        expected = set(self.allcoords) - set([self.meshco_x, self.meshco_y])
        self._assert_lists_equal(expected, result)

    def test_coords__axis__meshcoords(self):
        # Coord (singular) with axis + mesh_coords=True
        result = self.cube.coord(axis="x", mesh_coords=True)
        self.assertIs(result, self.meshco_x)

    def test_coords__dimcoords__meshcoords(self):
        # dim_coords and mesh_coords should be mutually exclusive.
        result = self.cube.coords(dim_coords=True, mesh_coords=True)
        self.assertEqual(result, [])

    def test_coords__nodimcoords__meshcoords(self):
        # When mesh_coords=True, dim_coords=False should have no effect.
        result = self.cube.coords(dim_coords=False, mesh_coords=True)
        expected = [self.meshco_x, self.meshco_y]
        self._assert_lists_equal(expected, result)


class Test_mesh(tests.IrisTest):
    def setUp(self):
        # Create a standard test cube with a variety of types of coord.
        _add_test_meshcube(self)

    def test_mesh(self):
        result = self.cube.mesh
        self.assertIs(result, self.mesh)

    def test_no_mesh(self):
        # Replace standard setUp cube with a no-mesh version.
        _add_test_meshcube(self, nomesh=True)
        result = self.cube.mesh
        self.assertIsNone(result)


class Test_location(tests.IrisTest):
    def setUp(self):
        # Create a standard test cube with a variety of types of coord.
        _add_test_meshcube(self)

    def test_no_mesh(self):
        # Replace standard setUp cube with a no-mesh version.
        _add_test_meshcube(self, nomesh=True)
        result = self.cube.location
        self.assertIsNone(result)

    def test_mesh(self):
        cube = self.cube
        result = cube.location
        self.assertEqual(result, self.meshco_x.location)

    def test_alternate_location(self):
        # Replace standard setUp cube with an edge-based version.
        _add_test_meshcube(self, location="edge")
        cube = self.cube
        result = cube.location
        self.assertEqual(result, "edge")


class Test_mesh_dim(tests.IrisTest):
    def setUp(self):
        # Create a standard test cube with a variety of types of coord.
        _add_test_meshcube(self)

    def test_no_mesh(self):
        # Replace standard setUp cube with a no-mesh version.
        _add_test_meshcube(self, nomesh=True)
        result = self.cube.mesh_dim()
        self.assertIsNone(result)

    def test_mesh(self):
        cube = self.cube
        result = cube.mesh_dim()
        self.assertEqual(result, 1)

    def test_alternate(self):
        # Replace standard setUp cube with an edge-based version.
        _add_test_meshcube(self, location="edge")
        cube = self.cube
        # Transpose the cube : the mesh dim is then 0
        cube.transpose()
        result = cube.mesh_dim()
        self.assertEqual(result, 0)


class Test__init__mesh(tests.IrisTest):
    """Test that creation with mesh-coords functions, and prevents a cube having
    incompatible mesh-coords.

    """

    def setUp(self):
        # Create a standard test mesh and other useful components.
        mesh = sample_mesh()
        meshco = sample_meshcoord(mesh=mesh)
        self.mesh = mesh
        self.meshco = meshco
        self.nz = 2
        self.n_faces = meshco.shape[0]

    def test_mesh(self):
        # Create a new cube from some of the parts.
        nz, n_faces = self.nz, self.n_faces
        dimco_z = DimCoord(np.arange(nz), long_name="z")
        dimco_mesh = DimCoord(np.arange(n_faces), long_name="x")
        meshco = self.meshco
        cube = Cube(
            np.zeros((nz, n_faces)),
            dim_coords_and_dims=[(dimco_z, 0), (dimco_mesh, 1)],
            aux_coords_and_dims=[(meshco, 1)],
        )
        self.assertEqual(cube.mesh, meshco.mesh)

    def test_fail_dim_meshcoord(self):
        # As "test_mesh", but attempt to use the meshcoord as a dim-coord.
        # This should not be allowed.
        nz, n_faces = self.nz, self.n_faces
        dimco_z = DimCoord(np.arange(nz), long_name="z")
        meshco = self.meshco
        with self.assertRaisesRegex(ValueError, "may not be an AuxCoord"):
            Cube(
                np.zeros((nz, n_faces)),
                dim_coords_and_dims=[(dimco_z, 0), (meshco, 1)],
            )

    def test_multi_meshcoords(self):
        meshco_x = sample_meshcoord(axis="x", mesh=self.mesh)
        meshco_y = sample_meshcoord(axis="y", mesh=self.mesh)
        n_faces = meshco_x.shape[0]
        cube = Cube(
            np.zeros(n_faces),
            aux_coords_and_dims=[(meshco_x, 0), (meshco_y, 0)],
        )
        self.assertEqual(cube.mesh, meshco_x.mesh)

    def test_multi_meshcoords_same_axis(self):
        # *Not* an error, as long as the coords are distinguishable.
        meshco_1 = sample_meshcoord(axis="x", mesh=self.mesh)
        meshco_2 = sample_meshcoord(axis="x", mesh=self.mesh)
        # Can't make these different at creation, owing to the limited
        # constructor args, but we can adjust common metadata afterwards.
        meshco_2.rename("junk_name")

        n_faces = meshco_1.shape[0]
        cube = Cube(
            np.zeros(n_faces),
            aux_coords_and_dims=[(meshco_1, 0), (meshco_2, 0)],
        )
        self.assertEqual(cube.mesh, meshco_1.mesh)

    def test_fail_meshcoords_different_locations(self):
        # Same as successful 'multi_mesh', but different locations.
        # N.B. must have a mesh with n-faces == n-edges to test this
        mesh = sample_mesh(n_faces=7, n_edges=7)
        meshco_1 = sample_meshcoord(axis="x", mesh=mesh, location="face")
        meshco_2 = sample_meshcoord(axis="y", mesh=mesh, location="edge")
        # They should still have the same *shape* (or would fail anyway)
        self.assertEqual(meshco_1.shape, meshco_2.shape)
        n_faces = meshco_1.shape[0]
        msg = "does not match existing cube location"
        with self.assertRaisesRegex(ValueError, msg):
            Cube(
                np.zeros(n_faces),
                aux_coords_and_dims=[(meshco_1, 0), (meshco_2, 0)],
            )

    def test_meshcoords_equal_meshes(self):
        meshco_x = sample_meshcoord(axis="x")
        meshco_y = sample_meshcoord(axis="y")
        n_faces = meshco_x.shape[0]
        Cube(
            np.zeros(n_faces),
            aux_coords_and_dims=[(meshco_x, 0), (meshco_y, 0)],
        )

    def test_fail_meshcoords_different_meshes(self):
        meshco_x = sample_meshcoord(axis="x")
        meshco_y = sample_meshcoord(axis="y")  # Own (different) mesh
        meshco_y.mesh.long_name = "new_name"
        n_faces = meshco_x.shape[0]
        with self.assertRaisesRegex(ValueError, "Mesh.* does not match"):
            Cube(
                np.zeros(n_faces),
                aux_coords_and_dims=[(meshco_x, 0), (meshco_y, 0)],
            )

    def test_fail_meshcoords_different_dims(self):
        # Same as 'test_mesh', but meshcoords on different dimensions.
        # Replace standard setup with one where n_z == n_faces.
        n_z, n_faces = 4, 4
        mesh = sample_mesh(n_faces=n_faces)
        meshco_x = sample_meshcoord(mesh=mesh, axis="x")
        meshco_y = sample_meshcoord(mesh=mesh, axis="y")
        msg = "does not match existing cube mesh dimension"
        with self.assertRaisesRegex(ValueError, msg):
            Cube(
                np.zeros((n_z, n_faces)),
                aux_coords_and_dims=[(meshco_x, 1), (meshco_y, 0)],
            )


class Test__add_aux_coord__mesh(tests.IrisTest):
    """Test that "Cube.add_aux_coord" functions with a mesh-coord, and prevents a
    cube having incompatible mesh-coords.

    """

    def setUp(self):
        _add_test_meshcube(self)
        # Remove the existing "meshco_y", so we can add similar ones without
        # needing to distinguish from the existing.
        self.cube.remove_coord(self.meshco_y)

    def test_add_compatible(self):
        cube = self.cube
        meshco_y = self.meshco_y
        # Add the y-meshco back into the cube.
        cube.add_aux_coord(meshco_y, 1)
        self.assertIn(meshco_y, cube.coords(mesh_coords=True))

    def test_add_multiple(self):
        # Show that we can add extra mesh coords.
        cube = self.cube
        meshco_y = self.meshco_y
        # Add the y-meshco back into the cube.
        cube.add_aux_coord(meshco_y, 1)
        # Make a duplicate y-meshco, renamed so it can add into the cube.
        new_meshco_y = meshco_y.copy()
        new_meshco_y.rename("alternative")
        cube.add_aux_coord(new_meshco_y, 1)
        self.assertEqual(len(cube.coords(mesh_coords=True)), 3)

    def test_add_equal_mesh(self):
        # Make a duplicate y-meshco, and rename so it can add into the cube.
        cube = self.cube
        # Create 'meshco_y' duplicate, but a new mesh
        meshco_y = sample_meshcoord(axis="y")
        cube.add_aux_coord(meshco_y, 1)
        self.assertIn(meshco_y, cube.coords(mesh_coords=True))

    def test_fail_different_mesh(self):
        # Make a duplicate y-meshco, and rename so it can add into the cube.
        cube = self.cube
        # Create 'meshco_y' duplicate, but a new mesh
        meshco_y = sample_meshcoord(axis="y")
        meshco_y.mesh.long_name = "new_name"
        msg = "does not match existing cube mesh"
        with self.assertRaisesRegex(ValueError, msg):
            cube.add_aux_coord(meshco_y, 1)

    def test_fail_different_location(self):
        # Make a new mesh with equal n_faces and n_edges
        mesh = sample_mesh(n_faces=4, n_edges=4)
        # Re-make the test objects based on that.
        _add_test_meshcube(self, mesh=mesh)
        cube = self.cube
        cube.remove_coord(self.meshco_y)  # Remove y-coord, as in setUp()
        # Create a new meshco_y, same mesh but based on edges.
        meshco_y = sample_meshcoord(axis="y", mesh=self.mesh, location="edge")
        msg = "does not match existing cube location"
        with self.assertRaisesRegex(ValueError, msg):
            cube.add_aux_coord(meshco_y, 1)

    def test_fail_different_dimension(self):
        # Re-make the test objects with the non-mesh dim equal in length.
        n_faces = self.cube.shape[1]
        _add_test_meshcube(self, n_z=n_faces)
        cube = self.cube
        meshco_y = self.meshco_y
        cube.remove_coord(meshco_y)  # Remove y-coord, as in setUp()

        # Attempt to re-attach the 'y' meshcoord, to a different cube dimension.
        msg = "does not match existing cube mesh dimension"
        with self.assertRaisesRegex(ValueError, msg):
            cube.add_aux_coord(meshco_y, 0)


class Test__add_dim_coord__mesh(tests.IrisTest):
    """Test that "Cube.add_dim_coord" cannot work with a mesh-coord."""

    def test(self):
        # Create a mesh with only 2 faces, so coord *can't* be non-monotonic.
        mesh = sample_mesh(n_faces=2)
        meshco = sample_meshcoord(mesh=mesh)
        cube = Cube([0, 1])
        with self.assertRaisesRegex(ValueError, "may not be an AuxCoord"):
            cube.add_dim_coord(meshco, 0)


class Test__eq__mesh(tests.IrisTest):
    """Check that cubes with meshes support == as expected.

    Note: there is no special code for this in iris.cube.Cube : it is
    provided by the coord comparisons.

    """

    def setUp(self):
        # Create a 'standard' test cube.
        _add_test_meshcube(self)

    def test_copied_cube_match(self):
        cube = self.cube
        cube2 = cube.copy()
        self.assertEqual(cube, cube2)

    def test_equal_mesh_match(self):
        cube1 = self.cube
        # re-create an identical cube, using the same mesh.
        _add_test_meshcube(self)
        cube2 = self.cube
        self.assertEqual(cube1, cube2)

    def test_new_mesh_different(self):
        cube1 = self.cube
        # re-create an identical cube, using a different mesh.
        _add_test_meshcube(self)
        self.cube.mesh.long_name = "new_name"
        cube2 = self.cube
        self.assertNotEqual(cube1, cube2)


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

    def test_different_coordinate_vector(self):
        cube = Cube([0, 1], long_name="raspberry", units="1")
        cube.add_dim_coord(DimCoord([0, 1], long_name="loganberry", units="1"), 0)
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
        a_cell_measure = CellMeasure(np.arange(6).reshape(2, 3), long_name="area")
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
        expected_error = "foo coordinate for factory is not present on cube bar"
        with self.assertRaisesRegex(ValueError, expected_error):
            cube.add_aux_factory(factory)


class Test_remove_metadata(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        cube.add_dim_coord(x_coord, 1)
        z_coord = AuxCoord(points=np.arange(6).reshape(2, 3), long_name="z")
        cube.add_aux_coord(z_coord, [0, 1])
        a_cell_measure = CellMeasure(np.arange(6).reshape(2, 3), long_name="area")
        self.b_cell_measure = CellMeasure(
            np.arange(6).reshape(2, 3), long_name="other_area"
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

    def test_remove_ancilliary_variable_by_name(self):
        self.cube.remove_ancillary_variable("Quality of Detection")
        self.assertEqual(self.cube._ancillary_variables_and_dims, [])

    def test_fail_remove_ancilliary_variable_by_name(self):
        with self.assertRaises(AncillaryVariableNotFoundError):
            self.cube.remove_ancillary_variable("notname")


class TestCoords(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        cube.add_dim_coord(x_coord, 1)
        self.x_coord = x_coord
        self.cube = cube

    def test_bad_coord(self):
        bad_coord = self.x_coord.copy()
        bad_coord.attributes = {"bad": "attribute"}
        re = (
            "Expected to find exactly 1 coordinate matching the given "
            "'x' coordinate's metadata, but found none."
        )
        with self.assertRaisesRegex(CoordinateNotFoundError, re):
            _ = self.cube.coord(bad_coord)


class Test_coord_division_units(tests.IrisTest):
    def test(self):
        aux = AuxCoord(1, long_name="length", units="metres")
        cube = Cube(1, units="seconds")
        self.assertEqual((aux / cube).units, "m.s-1")


class Test__getitem_CellMeasure(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        cube.add_dim_coord(x_coord, 1)
        y_coord = DimCoord(points=np.array([5, 6]), long_name="y")
        cube.add_dim_coord(y_coord, 0)
        z_coord = AuxCoord(points=np.arange(6).reshape(2, 3), long_name="z")
        cube.add_aux_coord(z_coord, [0, 1])
        a_cell_measure = CellMeasure(np.arange(6).reshape(2, 3), long_name="area")
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
        self.assertEqual(result.shape, result.ancillary_variables()[0].data.shape)

    def test_ancillary_variables_1d(self):
        result = self.cube[0, 0:2]
        self.assertEqual(len(result.ancillary_variables()), 1)
        self.assertEqual(result.shape, result.ancillary_variables()[0].data.shape)


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

    def test_ancillary_variable_dims_by_name(self):
        ancill_var_dims = self.cube.ancillary_variable_dims("number_of_observations")
        self.assertEqual(ancill_var_dims, (0, 1))

    def test_fail_ancillary_variable_dims_by_name(self):
        with self.assertRaises(AncillaryVariableNotFoundError):
            self.cube.ancillary_variable_dims("notname")


class TestCellMeasures(tests.IrisTest):
    def setUp(self):
        cube = Cube(np.arange(6).reshape(2, 3))
        x_coord = DimCoord(points=np.array([2, 3, 4]), long_name="x")
        cube.add_dim_coord(x_coord, 1)
        z_coord = AuxCoord(points=np.arange(6).reshape(2, 3), long_name="z")
        cube.add_aux_coord(z_coord, [0, 1])
        self.a_cell_measure = CellMeasure(
            np.arange(6).reshape(2, 3), long_name="area", units="m2"
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

    def test_cell_measure_dims_by_name(self):
        cm_dims = self.cube.cell_measure_dims("area")
        self.assertEqual(cm_dims, (0, 1))

    def test_fail_cell_measure_dims_by_name(self):
        with self.assertRaises(CellMeasureNotFoundError):
            self.cube.cell_measure_dims("notname")


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
        x_coord = AuxCoord(points=np.array([[2, 3], [8, 4], [7, 9]]), long_name="x")
        self.cube.add_aux_coord(x_coord, (0, 1))
        self.cube.transpose()
        self.assertEqual(self.cube._aux_coords_and_dims, [(x_coord, (2, 1))])

    def test_cell_measures(self):
        area_cm = CellMeasure(np.arange(12).reshape(3, 4), long_name="area of cells")
        self.cube.add_cell_measure(area_cm, (0, 2))
        self.cube.transpose()
        self.assertEqual(self.cube._cell_measures_and_dims, [(area_cm, (2, 0))])

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

    def test_unit_multiply(self):
        _client = Client()
        cube = iris.cube.Cube(da.arange(1), units="m")
        cube.units *= "s-1"
        cube.convert_units("m s-1")
        cube.data
        _client.close()


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


@pytest.fixture
def simplecube():
    return stock.simple_2d_w_cell_measure_ancil_var()


class Test__dimensional_metadata:
    """Tests for the "Cube._dimensional_data" method.

    NOTE: test could all be static methods, but that adds a line to each definition.
    """

    def test_not_found(self, simplecube):
        with pytest.raises(KeyError, match="was not found in"):
            simplecube._dimensional_metadata("grid_latitude")

    def test_dim_coord_name_found(self, simplecube):
        res = simplecube._dimensional_metadata("bar")
        assert res == simplecube.coord("bar")

    def test_dim_coord_instance_found(self, simplecube):
        res = simplecube._dimensional_metadata(simplecube.coord("bar"))
        assert res == simplecube.coord("bar")

    def test_aux_coord_name_found(self, simplecube):
        res = simplecube._dimensional_metadata("wibble")
        assert res == simplecube.coord("wibble")

    def test_aux_coord_instance_found(self, simplecube):
        res = simplecube._dimensional_metadata(simplecube.coord("wibble"))
        assert res == simplecube.coord("wibble")

    def test_cell_measure_name_found(self, simplecube):
        res = simplecube._dimensional_metadata("cell_area")
        assert res == simplecube.cell_measure("cell_area")

    def test_cell_measure_instance_found(self, simplecube):
        res = simplecube._dimensional_metadata(simplecube.cell_measure("cell_area"))
        assert res == simplecube.cell_measure("cell_area")

    def test_ancillary_var_name_found(self, simplecube):
        res = simplecube._dimensional_metadata("quality_flag")
        assert res == simplecube.ancillary_variable("quality_flag")

    def test_ancillary_var_instance_found(self, simplecube):
        res = simplecube._dimensional_metadata(
            simplecube.ancillary_variable("quality_flag")
        )
        assert res == simplecube.ancillary_variable("quality_flag")

    def test_two_with_same_name(self, simplecube):
        # If a cube has two _DimensionalMetadata objects with the same name, the
        # current behaviour results in _dimensional_metadata returning the first
        # one it finds.
        simplecube.cell_measure("cell_area").rename("wibble")
        res = simplecube._dimensional_metadata("wibble")
        assert res == simplecube.coord("wibble")

    def test_two_with_same_name_specify_instance(self, simplecube):
        # The cube has two _DimensionalMetadata objects with the same name so
        # we specify the _DimensionalMetadata instance to ensure it returns the
        # correct one.
        simplecube.cell_measure("cell_area").rename("wibble")
        res = simplecube._dimensional_metadata(simplecube.cell_measure("wibble"))
        assert res == simplecube.cell_measure("wibble")


class TestReprs:
    """Confirm that str(cube), repr(cube) and cube.summary() work by creating a fresh
    :class:`iris._representation.cube_printout.CubePrinter` object, and using it
    in the expected ways.

    Notes
    -----
    This only tests code connectivity.  The functionality is tested elsewhere, in
    `iris.tests.unit._representation.cube_printout.test_CubePrintout`.
    """

    # Note: logically this could be a staticmethod, but that seems to upset Pytest
    @pytest.fixture
    def patched_cubeprinter(self):
        target = "iris._representation.cube_printout.CubePrinter"
        instance_mock = mock.MagicMock(
            to_string=mock.MagicMock(return_value="")  # NB this must return a string
        )
        with mock.patch(target, return_value=instance_mock) as class_mock:
            yield class_mock, instance_mock

    @staticmethod
    def _check_expected_effects(simplecube, patched_cubeprinter, oneline, padding):
        class_mock, instance_mock = patched_cubeprinter
        assert class_mock.call_args_list == [
            # "CubePrinter()" was called exactly once, with the cube as arg
            mock.call(simplecube)
        ]
        assert instance_mock.to_string.call_args_list == [
            # "CubePrinter(cube).to_string()" was called exactly once, with these args
            mock.call(oneline=oneline, name_padding=padding)
        ]

    def test_str_effects(self, simplecube, patched_cubeprinter):
        str(simplecube)
        self._check_expected_effects(
            simplecube, patched_cubeprinter, oneline=False, padding=35
        )

    def test_repr_effects(self, simplecube, patched_cubeprinter):
        repr(simplecube)
        self._check_expected_effects(
            simplecube, patched_cubeprinter, oneline=True, padding=1
        )

    def test_summary_effects(self, simplecube, patched_cubeprinter):
        simplecube.summary(
            shorten=mock.sentinel.oneliner, name_padding=mock.sentinel.padding
        )
        self._check_expected_effects(
            simplecube,
            patched_cubeprinter,
            oneline=mock.sentinel.oneliner,
            padding=mock.sentinel.padding,
        )


class TestHtmlRepr:
    """Confirm that Cube._repr_html_() creates a fresh
    :class:`iris.experimental.representation.CubeRepresentation` object, and uses it
    in the expected way.

    Notes
    -----
    This only tests code connectivity.  The functionality is tested elsewhere, in
    `iris.tests.unit.experimental.representation.test_CubeRepresentation`.
    """

    # Note: logically this could be a staticmethod, but that seems to upset Pytest
    @pytest.fixture
    def patched_cubehtml(self):
        target = "iris.experimental.representation.CubeRepresentation"
        instance_mock = mock.MagicMock(
            repr_html=mock.MagicMock(return_value="")  # NB this must return a string
        )
        with mock.patch(target, return_value=instance_mock) as class_mock:
            yield class_mock, instance_mock

    @staticmethod
    def test__repr_html__effects(simplecube, patched_cubehtml):
        simplecube._repr_html_()

        class_mock, instance_mock = patched_cubehtml
        assert class_mock.call_args_list == [
            # "CubeRepresentation()" was called exactly once, with the cube as arg
            mock.call(simplecube)
        ]
        assert instance_mock.repr_html.call_args_list == [
            # "CubeRepresentation(cube).repr_html()" was called exactly once, with no args
            mock.call()
        ]


class Test__cell_methods:
    @pytest.fixture(autouse=True)
    def cell_measures_testdata(self):
        self.cube = Cube([0])
        self.cm = CellMethod("mean", "time", "6hr")
        self.cm2 = CellMethod("max", "latitude", "4hr")

    def test_none(self):
        assert self.cube.cell_methods == ()

    def test_one(self):
        cube = Cube([0], cell_methods=[self.cm])
        expected = (self.cm,)
        assert expected == cube.cell_methods

    def test_empty_assigns(self):
        testargs = [(), [], {}, 0, 0.0, False, None]
        results = []
        for arg in testargs:
            cube = self.cube.copy()
            cube.cell_methods = arg  # assign test object
            results.append(cube.cell_methods)  # capture what is read back
        expected_results = [()] * len(testargs)
        assert expected_results == results

    def test_single_assigns(self):
        cms = (self.cm, self.cm2)
        # Any type of iterable ought to work
        # But N.B. *not* testing sets, as order is not stable
        testargs = [cms, list(cms), {cm: 1 for cm in cms}]
        results = []
        for arg in testargs:
            cube = self.cube.copy()
            cube.cell_methods = arg  # assign test object
            results.append(cube.cell_methods)  # capture what is read back
        expected_results = [cms] * len(testargs)
        assert expected_results == results

    def test_fail_assign_noniterable(self):
        test_object = object()
        with pytest.raises(TypeError, match="not iterable"):
            self.cube.cell_methods = test_object

    def test_fail_create_noniterable(self):
        test_object = object()
        with pytest.raises(TypeError, match="not iterable"):
            Cube([0], cell_methods=test_object)

    def test_fail_assign_noncellmethod(self):
        test_object = object()
        with pytest.raises(ValueError, match="not an iris.coords.CellMethod"):
            self.cube.cell_methods = (test_object,)

    def test_fail_create_noncellmethod(self):
        test_object = object()
        with pytest.raises(ValueError, match="not an iris.coords.CellMethod"):
            Cube([0], cell_methods=[test_object])

    def test_assign_derivedcellmethod(self):
        class DerivedCellMethod(CellMethod):
            pass

        test_object = DerivedCellMethod("mean", "time", "6hr")
        cms = (test_object,)
        self.cube.cell_methods = (test_object,)
        assert cms == self.cube.cell_methods

    def test_fail_assign_duckcellmethod(self):
        # Can't currently assign a "duck-typed" CellMethod replacement, since
        # implementation requires class membership (boo!)
        DuckCellMethod = namedtuple("DuckCellMethod", CellMethod._names)
        test_object = DuckCellMethod(*CellMethod._names)  # fill props with value==name
        with pytest.raises(ValueError, match="not an iris.coords.CellMethod"):
            self.cube.cell_methods = (test_object,)


class TestAttributesProperty:
    def test_attrs_type(self):
        # Cube attributes are always of a special dictionary type.
        cube = Cube([0], attributes={"a": 1})
        assert type(cube.attributes) is CubeAttrsDict
        assert cube.attributes == {"a": 1}

    def test_attrs_remove(self):
        # Wiping attributes replaces the stored object
        cube = Cube([0], attributes={"a": 1})
        attrs = cube.attributes
        cube.attributes = None
        assert cube.attributes is not attrs
        assert type(cube.attributes) is CubeAttrsDict
        assert cube.attributes == {}

    def test_attrs_clear(self):
        # Clearing attributes leaves the same object
        cube = Cube([0], attributes={"a": 1})
        attrs = cube.attributes
        cube.attributes.clear()
        assert cube.attributes is attrs
        assert type(cube.attributes) is CubeAttrsDict
        assert cube.attributes == {}


if __name__ == "__main__":
    tests.main()
