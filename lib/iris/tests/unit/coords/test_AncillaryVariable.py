# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coords.AncillaryVariable` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from cf_units import Unit
import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_lazy_data
from iris.coords import AncillaryVariable
from iris.cube import Cube
from iris.tests.unit.coords import CoordTestMixin, lazyness_string


def data_all_dtypes_and_lazynesses(self):
    # Generate ancillary variables with real and lazy data, and a few different
    # dtypes.
    data_types = ["real", "lazy"]
    dtypes = [np.int16, np.int32, np.float32, np.float64]
    for dtype in dtypes:
        for data_type_name in data_types:
            data = np.asarray(self.data_real, dtype=dtype)
            if data_type_name == "lazy":
                data = da.from_array(data, data.shape)
            ancill_var = AncillaryVariable(data)
            result = (ancill_var, data_type_name)
            yield result


class AncillaryVariableTestMixin(CoordTestMixin):
    # Define a 2-D default array shape.
    def setupTestArrays(self, shape=(2, 3), masked=False):
        # Create concrete and lazy data test arrays, given a desired shape.
        # If masked=True, also add masked arrays with some or no masked data.
        n_vals = np.prod(shape)
        # Note: the values must be integral for testing integer dtypes.
        values = 100.0 + 10.0 * np.arange(n_vals, dtype=float).reshape(shape)
        self.data_real = values
        self.data_lazy = da.from_array(values, values.shape)

        if masked:
            mvalues = ma.array(values)
            self.no_masked_data_real = mvalues
            self.no_masked_data_lazy = da.from_array(
                mvalues, mvalues.shape, asarray=False
            )
            mvalues = ma.array(mvalues, copy=True)
            mvalues[0] = ma.masked
            self.masked_data_real = mvalues
            self.masked_data_lazy = da.from_array(
                mvalues, mvalues.shape, asarray=False
            )


class Test__init__(tests.IrisTest, AncillaryVariableTestMixin):
    # Test for AncillaryVariable creation, with real / lazy data
    def setUp(self):
        self.setupTestArrays(masked=True)

    def test_lazyness_and_dtype_combinations(self):
        for (ancill_var, data_lazyness) in data_all_dtypes_and_lazynesses(
            self,
        ):
            data = ancill_var.core_data()
            # Check properties of data.
            if data_lazyness == "real":
                # Real data.
                if ancill_var.dtype == self.data_real.dtype:
                    self.assertArraysShareData(
                        data,
                        self.data_real,
                        "Data values are not the same "
                        "data as the provided array.",
                    )
                    self.assertIsNot(
                        data,
                        self.data_real,
                        "Data array is the same instance as the provided "
                        "array.",
                    )
                else:
                    # the original data values were cast to a test dtype.
                    check_data = self.data_real.astype(ancill_var.dtype)
                    self.assertEqualRealArraysAndDtypes(data, check_data)
            else:
                # Lazy data : the core data may be promoted to float.
                check_data = self.data_lazy.astype(data.dtype)
                self.assertEqualLazyArraysAndDtypes(data, check_data)
                # The realisation type should be correct, though.
                target_dtype = ancill_var.dtype
                self.assertEqual(ancill_var.data.dtype, target_dtype)

    def test_no_masked_data_real(self):
        data = self.no_masked_data_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertEqual(ma.count_masked(data), 0)
        ancill_var = AncillaryVariable(data)
        self.assertFalse(ancill_var.has_lazy_data())
        self.assertTrue(ma.isMaskedArray(ancill_var.data))
        self.assertEqual(ma.count_masked(ancill_var.data), 0)

    def test_no_masked_data_lazy(self):
        data = self.no_masked_data_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertEqual(ma.count_masked(computed), 0)
        ancill_var = AncillaryVariable(data)
        self.assertTrue(ancill_var.has_lazy_data())
        self.assertTrue(ma.isMaskedArray(ancill_var.data))
        self.assertEqual(ma.count_masked(ancill_var.data), 0)

    def test_masked_data_real(self):
        data = self.masked_data_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertTrue(ma.count_masked(data))
        ancill_var = AncillaryVariable(data)
        self.assertFalse(ancill_var.has_lazy_data())
        self.assertTrue(ma.isMaskedArray(ancill_var.data))
        self.assertTrue(ma.count_masked(ancill_var.data))

    def test_masked_data_lazy(self):
        data = self.masked_data_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertTrue(ma.count_masked(computed))
        ancill_var = AncillaryVariable(data)
        self.assertTrue(ancill_var.has_lazy_data())
        self.assertTrue(ma.isMaskedArray(ancill_var.data))
        self.assertTrue(ma.count_masked(ancill_var.data))


class Test_core_data(tests.IrisTest, AncillaryVariableTestMixin):
    # Test for AncillaryVariable.core_data() with various lazy/real data.
    def setUp(self):
        self.setupTestArrays()

    def test_real_data(self):
        ancill_var = AncillaryVariable(self.data_real)
        result = ancill_var.core_data()
        self.assertArraysShareData(
            result,
            self.data_real,
            "core_data() do not share data with the internal array.",
        )

    def test_lazy_data(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        result = ancill_var.core_data()
        self.assertEqualLazyArraysAndDtypes(result, self.data_lazy)

    def test_lazy_points_realise(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        real_data = ancill_var.data
        result = ancill_var.core_data()
        self.assertEqualRealArraysAndDtypes(result, real_data)


class Test_lazy_data(tests.IrisTest, AncillaryVariableTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_core(self):
        ancill_var = AncillaryVariable(self.data_real)
        result = ancill_var.lazy_data()
        self.assertEqualLazyArraysAndDtypes(result, self.data_lazy)

    def test_lazy_core(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        result = ancill_var.lazy_data()
        self.assertIs(result, self.data_lazy)


class Test_has_lazy_data(tests.IrisTest, AncillaryVariableTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_core(self):
        ancill_var = AncillaryVariable(self.data_real)
        result = ancill_var.has_lazy_data()
        self.assertFalse(result)

    def test_lazy_core(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        result = ancill_var.has_lazy_data()
        self.assertTrue(result)

    def test_lazy_core_realise(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        ancill_var.data
        result = ancill_var.has_lazy_data()
        self.assertFalse(result)


class Test__getitem__(tests.IrisTest, AncillaryVariableTestMixin):
    # Test for AncillaryVariable indexing with various types of data.
    def setUp(self):
        self.setupTestArrays()

    def test_partial_slice_data_copy(self):
        parent_ancill_var = AncillaryVariable([1.0, 2.0, 3.0])
        sub_ancill_var = parent_ancill_var[:1]
        values_before_change = sub_ancill_var.data.copy()
        parent_ancill_var.data[:] = -999.9
        self.assertArrayEqual(sub_ancill_var.data, values_before_change)

    def test_full_slice_data_copy(self):
        parent_ancill_var = AncillaryVariable([1.0, 2.0, 3.0])
        sub_ancill_var = parent_ancill_var[:]
        values_before_change = sub_ancill_var.data.copy()
        parent_ancill_var.data[:] = -999.9
        self.assertArrayEqual(sub_ancill_var.data, values_before_change)

    def test_dtypes(self):
        # Index ancillary variables with real+lazy data, and either an int or
        # floating dtype.
        # Check that dtypes remain the same in all cases, taking the dtypes
        # directly from the core data as we have no masking).
        for (main_ancill_var, data_lazyness) in data_all_dtypes_and_lazynesses(
            self
        ):

            sub_ancill_var = main_ancill_var[:2, 1]

            ancill_var_dtype = main_ancill_var.dtype
            msg = (
                "Indexing main_ancill_var of dtype {} with {} data changed"
                "dtype of {} to {}."
            )

            sub_data = sub_ancill_var.core_data()
            self.assertEqual(
                sub_data.dtype,
                ancill_var_dtype,
                msg.format(
                    ancill_var_dtype, data_lazyness, "data", sub_data.dtype
                ),
            )

    def test_lazyness(self):
        # Index ancillary variables with real+lazy data, and either an int or
        # floating dtype.
        # Check that lazy data stays lazy and real stays real, in all cases.
        for (main_ancill_var, data_lazyness) in data_all_dtypes_and_lazynesses(
            self
        ):

            sub_ancill_var = main_ancill_var[:2, 1]

            msg = (
                "Indexing main_ancill_var of dtype {} with {} data "
                "changed laziness of {} from {!r} to {!r}."
            )
            ancill_var_dtype = main_ancill_var.dtype
            sub_data_lazyness = lazyness_string(sub_ancill_var.core_data())
            self.assertEqual(
                sub_data_lazyness,
                data_lazyness,
                msg.format(
                    ancill_var_dtype,
                    data_lazyness,
                    "data",
                    data_lazyness,
                    sub_data_lazyness,
                ),
            )

    def test_real_data_copies(self):
        # Index ancillary variables with real+lazy data.
        # In all cases, check that any real arrays are copied by the indexing.
        for (main_ancill_var, data_lazyness) in data_all_dtypes_and_lazynesses(
            self
        ):

            sub_ancill_var = main_ancill_var[:2, 1]

            msg = (
                "Indexed ancillary variable with {} data "
                "does not have its own separate {} array."
            )
            if data_lazyness == "real":
                main_data = main_ancill_var.core_data()
                sub_data = sub_ancill_var.core_data()
                sub_main_data = main_data[:2, 1]
                self.assertEqualRealArraysAndDtypes(sub_data, sub_main_data)
                self.assertArraysDoNotShareData(
                    sub_data,
                    sub_main_data,
                    msg.format(data_lazyness, "points"),
                )


class Test_copy(tests.IrisTest, AncillaryVariableTestMixin):
    # Test for AncillaryVariable.copy() with various types of data.
    def setUp(self):
        self.setupTestArrays()

    def test_lazyness(self):
        # Copy ancillary variables with real+lazy data, and either an int or
        # floating dtype.
        # Check that lazy data stays lazy and real stays real, in all cases.
        for (main_ancill_var, data_lazyness) in data_all_dtypes_and_lazynesses(
            self
        ):

            ancill_var_dtype = main_ancill_var.dtype
            copied_ancill_var = main_ancill_var.copy()

            msg = (
                "Copying main_ancill_var of dtype {} with {} data "
                "changed lazyness of {} from {!r} to {!r}."
            )

            copied_data_lazyness = lazyness_string(
                copied_ancill_var.core_data()
            )
            self.assertEqual(
                copied_data_lazyness,
                data_lazyness,
                msg.format(
                    ancill_var_dtype,
                    data_lazyness,
                    "points",
                    data_lazyness,
                    copied_data_lazyness,
                ),
            )

    def test_realdata_copies(self):
        # Copy ancillary variables with real+lazy data.
        # In all cases, check that any real arrays are copies, not views.
        for (main_ancill_var, data_lazyness) in data_all_dtypes_and_lazynesses(
            self
        ):

            copied_ancill_var = main_ancill_var.copy()

            msg = (
                "Copied ancillary variable with {} data "
                "does not have its own separate {} array."
            )

            if data_lazyness == "real":
                main_data = main_ancill_var.core_data()
                copied_data = copied_ancill_var.core_data()
                self.assertEqualRealArraysAndDtypes(main_data, copied_data)
                self.assertArraysDoNotShareData(
                    main_data, copied_data, msg.format(data_lazyness, "points")
                )


class Test_data__getter(tests.IrisTest, AncillaryVariableTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_mutable_real_data(self):
        # Check that ancill_var.data returns a modifiable array, and changes
        # to it are reflected to the ancillary_var.
        data = np.array([1.0, 2.0, 3.0, 4.0])
        ancill_var = AncillaryVariable(data)
        initial_values = data.copy()
        ancill_var.data[1:2] += 33.1
        result = ancill_var.data
        self.assertFalse(np.all(result == initial_values))

    def test_real_data(self):
        # Getting real data does not change or copy them.
        ancill_var = AncillaryVariable(self.data_real)
        result = ancill_var.data
        self.assertArraysShareData(
            result,
            self.data_real,
            "Data values do not share data with the provided array.",
        )

    def test_lazy_data(self):
        # Getting lazy data realises them.
        ancill_var = AncillaryVariable(self.data_lazy)
        self.assertTrue(ancill_var.has_lazy_data())
        result = ancill_var.data
        self.assertFalse(ancill_var.has_lazy_data())
        self.assertEqualRealArraysAndDtypes(result, self.data_real)


class Test_data__setter(tests.IrisTest, AncillaryVariableTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_set_real(self):
        # Setting new real data does not make a copy.
        ancill_var = AncillaryVariable(self.data_real)
        new_data = self.data_real + 102.3
        ancill_var.data = new_data
        result = ancill_var.core_data()
        self.assertArraysShareData(
            result,
            new_data,
            "Data values do not share data with the assigned array.",
        )

    def test_fail_bad_shape(self):
        # Setting real data requires matching shape.
        ancill_var = AncillaryVariable([1.0, 2.0])
        msg = r"Require data with shape \(2,\), got \(3,\)"
        with self.assertRaisesRegex(ValueError, msg):
            ancill_var.data = np.array([1.0, 2.0, 3.0])

    def test_real_set_lazy(self):
        # Setting new lazy data does not make a copy.
        ancill_var = AncillaryVariable(self.data_real)
        new_data = self.data_lazy + 102.3
        ancill_var.data = new_data
        result = ancill_var.core_data()
        self.assertEqualLazyArraysAndDtypes(result, new_data)


class Test__str__(tests.IrisTest):
    def test_non_time_values(self):
        ancillary_var = AncillaryVariable(
            np.array([2, 5, 9]),
            standard_name="height",
            long_name="height of detector",
            var_name="height",
            units="m",
            attributes={"notes": "Measured from sea level"},
        )
        expected = "\n".join(
            [
                "AncillaryVariable :  height / (m)",
                "    data: [2, 5, 9]",
                "    shape: (3,)",
                "    dtype: int64",
                "    standard_name: 'height'",
                "    long_name: 'height of detector'",
                "    var_name: 'height'",
                "    attributes:",
                "        notes  'Measured from sea level'",
            ]
        )
        self.assertEqual(expected, ancillary_var.__str__())

    def test_time_values(self):
        ancillary_var = AncillaryVariable(
            np.array([2, 5, 9]),
            units="hours since 1970-01-01 01:00",
            long_name="time of previous valid detection",
        )
        expected = "\n".join(
            [
                (
                    "AncillaryVariable :  time of previous valid detection / "
                    "(hours since 1970-01-01 01:00, gregorian calendar)"
                ),
                (
                    "    data: [1970-01-01 03:00:00, 1970-01-01 06:00:00, "
                    "1970-01-01 10:00:00]"
                ),
                "    shape: (3,)",
                "    dtype: int64",
                "    long_name: 'time of previous valid detection'",
            ]
        )
        self.assertEqual(expected, ancillary_var.__str__())


class Test__repr__(tests.IrisTest):
    def test_non_time_values(self):
        ancillary_var = AncillaryVariable(
            np.array([2, 5, 9]),
            standard_name="height",
            long_name="height of detector",
            var_name="height",
            units="m",
            attributes={"notes": "Measured from sea level"},
        )
        expected = "<AncillaryVariable: height / (m)  [2, 5, 9]  shape(3,)>"
        self.assertEqual(expected, ancillary_var.__repr__())

    def test_time_values(self):
        ancillary_var = AncillaryVariable(
            np.array([2, 5, 9]),
            units="hours since 1970-01-01 01:00",
            long_name="time of previous valid detection",
        )
        expected = (
            "<AncillaryVariable: time of previous valid detection / (hours since 1970-01-01 01:00)  "
            "[...]  shape(3,)>"
        )
        self.assertEqual(expected, ancillary_var.__repr__())


class Test___binary_operator__(tests.IrisTest, AncillaryVariableTestMixin):
    # Test maths operations on on real+lazy data.
    def setUp(self):
        self.setupTestArrays()

        self.real_ancill_var = AncillaryVariable(self.data_real)
        self.lazy_ancill_var = AncillaryVariable(self.data_lazy)

        self.test_combinations = [
            (self.real_ancill_var, self.data_real, "real"),
            (self.lazy_ancill_var, self.data_lazy, "lazy"),
        ]

    def _check(self, result_ancill_var, expected_data, lazyness):
        # Test each operation on
        data = result_ancill_var.core_data()
        if lazyness == "real":
            self.assertEqualRealArraysAndDtypes(expected_data, data)
        else:
            self.assertEqualLazyArraysAndDtypes(expected_data, data)

    def test_add(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            result = ancill_var + 10
            expected_data = orig_data + 10
            self._check(result, expected_data, data_lazyness)

    def test_add_inplace(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            ancill_var += 10
            expected_data = orig_data + 10
            self._check(ancill_var, expected_data, data_lazyness)

    def test_right_add(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            result = 10 + ancill_var
            expected_data = 10 + orig_data
            self._check(result, expected_data, data_lazyness)

    def test_subtract(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            result = ancill_var - 10
            expected_data = orig_data - 10
            self._check(result, expected_data, data_lazyness)

    def test_subtract_inplace(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            ancill_var -= 10
            expected_data = orig_data - 10
            self._check(ancill_var, expected_data, data_lazyness)

    def test_right_subtract(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            result = 10 - ancill_var
            expected_data = 10 - orig_data
            self._check(result, expected_data, data_lazyness)

    def test_multiply(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            result = ancill_var * 10
            expected_data = orig_data * 10
            self._check(result, expected_data, data_lazyness)

    def test_multiply_inplace(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            ancill_var *= 10
            expected_data = orig_data * 10
            self._check(ancill_var, expected_data, data_lazyness)

    def test_right_multiply(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            result = 10 * ancill_var
            expected_data = 10 * orig_data
            self._check(result, expected_data, data_lazyness)

    def test_divide(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            result = ancill_var / 10
            expected_data = orig_data / 10
            self._check(result, expected_data, data_lazyness)

    def test_divide_inplace(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            ancill_var /= 10
            expected_data = orig_data / 10
            self._check(ancill_var, expected_data, data_lazyness)

    def test_right_divide(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            result = 10 / ancill_var
            expected_data = 10 / orig_data
            self._check(result, expected_data, data_lazyness)

    def test_negative(self):
        for (ancill_var, orig_data, data_lazyness) in self.test_combinations:
            result = -ancill_var
            expected_data = -orig_data
            self._check(result, expected_data, data_lazyness)


class Test_has_bounds(tests.IrisTest):
    def test(self):
        ancillary_var = AncillaryVariable(np.array([2, 9, 5]))
        self.assertFalse(ancillary_var.has_bounds())


class Test_convert_units(tests.IrisTest):
    def test_preserves_lazy(self):
        test_data = np.array([[11.1, 12.2, 13.3], [21.4, 22.5, 23.6]])
        lazy_data = as_lazy_data(test_data)
        ancill_var = AncillaryVariable(data=lazy_data, units="m")
        ancill_var.convert_units("ft")
        self.assertTrue(ancill_var.has_lazy_data())
        test_data_ft = Unit("m").convert(test_data, "ft")
        self.assertArrayAllClose(ancill_var.data, test_data_ft)


class Test_is_compatible(tests.IrisTest):
    def setUp(self):
        self.ancill_var = AncillaryVariable(
            [1.0, 8.0, 22.0], standard_name="number_of_observations", units="1"
        )
        self.modified_ancill_var = self.ancill_var.copy()

    def test_not_compatible_diff_name(self):
        # Different name() - not compatible
        self.modified_ancill_var.rename("air_temperature")
        self.assertFalse(
            self.ancill_var.is_compatible(self.modified_ancill_var)
        )

    def test_not_compatible_diff_units(self):
        # Different units- not compatible
        self.modified_ancill_var.units = "m"
        self.assertFalse(
            self.ancill_var.is_compatible(self.modified_ancill_var)
        )

    def test_not_compatible_diff_common_attrs(self):
        # Different common attributes - not compatible.
        self.ancill_var.attributes["source"] = "A"
        self.modified_ancill_var.attributes["source"] = "B"
        self.assertFalse(
            self.ancill_var.is_compatible(self.modified_ancill_var)
        )

    def test_compatible_diff_data(self):
        # Different data values - compatible.
        self.modified_ancill_var.data = [10.0, 20.0, 100.0]
        self.assertTrue(
            self.ancill_var.is_compatible(self.modified_ancill_var)
        )

    def test_compatible_diff_var_name(self):
        # Different var_name (but same name()) - compatible.
        self.modified_ancill_var.var_name = "obs_num"
        self.assertTrue(
            self.ancill_var.is_compatible(self.modified_ancill_var)
        )

    def test_compatible_diff_non_common_attributes(self):
        # Different non-common attributes - compatible.
        self.ancill_var.attributes["source"] = "A"
        self.modified_ancill_var.attributes["origin"] = "B"
        self.assertTrue(
            self.ancill_var.is_compatible(self.modified_ancill_var)
        )

    def test_compatible_ignore_common_attribute(self):
        # ignore different common attributes - compatible.
        self.ancill_var.attributes["source"] = "A"
        self.modified_ancill_var.attributes["source"] = "B"
        self.assertTrue(
            self.ancill_var.is_compatible(
                self.modified_ancill_var, ignore="source"
            )
        )


class TestEquality(tests.IrisTest):
    def test_nanpoints_eq_self(self):
        av1 = AncillaryVariable([1.0, np.nan, 2.0])
        self.assertEqual(av1, av1)

    def test_nanpoints_eq_copy(self):
        av1 = AncillaryVariable([1.0, np.nan, 2.0])
        av2 = av1.copy()
        self.assertEqual(av1, av2)


class Test_cube_dims(tests.IrisTest):
    def test(self):
        # Check that "coord.cube_dims(cube)" calls "cube.coord_dims(coord)".
        mock_dims_result = mock.sentinel.AV_DIMS
        mock_dims_call = mock.Mock(return_value=mock_dims_result)
        mock_cube = mock.Mock(Cube, ancillary_variable_dims=mock_dims_call)
        test_var = AncillaryVariable([1], long_name="test_name")

        result = test_var.cube_dims(mock_cube)
        self.assertEqual(result, mock_dims_result)
        self.assertEqual(mock_dims_call.call_args_list, [mock.call(test_var)])


if __name__ == "__main__":
    tests.main()
