# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coords.AncillaryVariable` class."""

from cf_units import Unit
import dask.array as da
import numpy as np
import numpy.ma as ma
import pytest

from iris._lazy_data import as_lazy_data
from iris.coords import AncillaryVariable
from iris.cube import Cube
from iris.tests import _shared_utils
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
    def setup_test_arrays(self, shape=(2, 3), masked=False):
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
            self.masked_data_lazy = da.from_array(mvalues, mvalues.shape, asarray=False)


class Test__init__(AncillaryVariableTestMixin):
    # Test for AncillaryVariable creation, with real / lazy data
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.setup_test_arrays(masked=True)

    def test_lazyness_and_dtype_combinations(self):
        for ancill_var, data_lazyness in data_all_dtypes_and_lazynesses(
            self,
        ):
            data = ancill_var.core_data()
            # Check properties of data.
            if data_lazyness == "real":
                # Real data.
                if ancill_var.dtype == self.data_real.dtype:
                    self.assert_arrays_share_data(
                        data,
                        self.data_real,
                        "Data values are not the same data as the provided array.",
                    )
                    assert data is not self.data_real, (
                        "Data array is the same instance as the provided array."
                    )
                else:
                    # the original data values were cast to a test dtype.
                    check_data = self.data_real.astype(ancill_var.dtype)
                    self.assert_equal_real_arrays_and_dtypes(data, check_data)
            else:
                # Lazy data : the core data may be promoted to float.
                check_data = self.data_lazy.astype(data.dtype)
                self.assert_equal_lazy_arrays_and_dtypes(data, check_data)
                # The realisation type should be correct, though.
                target_dtype = ancill_var.dtype
                assert ancill_var.data.dtype == target_dtype

    def test_no_masked_data_real(self):
        data = self.no_masked_data_real
        assert ma.isMaskedArray(data)
        assert ma.count_masked(data) == 0
        ancill_var = AncillaryVariable(data)
        assert not ancill_var.has_lazy_data()
        assert ma.isMaskedArray(ancill_var.data)
        assert ma.count_masked(ancill_var.data) == 0

    def test_no_masked_data_lazy(self):
        data = self.no_masked_data_lazy
        computed = data.compute()
        assert ma.isMaskedArray(computed)
        assert ma.count_masked(computed) == 0
        ancill_var = AncillaryVariable(data)
        assert ancill_var.has_lazy_data()
        assert ma.isMaskedArray(ancill_var.data)
        assert ma.count_masked(ancill_var.data) == 0

    def test_masked_data_real(self):
        data = self.masked_data_real
        assert ma.isMaskedArray(data)
        assert ma.count_masked(data)
        ancill_var = AncillaryVariable(data)
        assert not ancill_var.has_lazy_data()
        assert ma.isMaskedArray(ancill_var.data)
        assert ma.count_masked(ancill_var.data)

    def test_masked_data_lazy(self):
        data = self.masked_data_lazy
        computed = data.compute()
        assert ma.isMaskedArray(computed)
        assert ma.count_masked(computed)
        ancill_var = AncillaryVariable(data)
        assert ancill_var.has_lazy_data()
        assert ma.isMaskedArray(ancill_var.data)
        assert ma.count_masked(ancill_var.data)


class Test_core_data(AncillaryVariableTestMixin):
    # Test for AncillaryVariable.core_data() with various lazy/real data.
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.setup_test_arrays()

    def test_real_data(self):
        ancill_var = AncillaryVariable(self.data_real)
        result = ancill_var.core_data()
        self.assert_arrays_share_data(
            result,
            self.data_real,
            "core_data() do not share data with the internal array.",
        )

    def test_lazy_data(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        result = ancill_var.core_data()
        self.assert_equal_lazy_arrays_and_dtypes(result, self.data_lazy)

    def test_lazy_points_realise(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        real_data = ancill_var.data
        result = ancill_var.core_data()
        self.assert_equal_real_arrays_and_dtypes(result, real_data)


class Test_lazy_data(AncillaryVariableTestMixin):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.setup_test_arrays()

    def test_real_core(self):
        ancill_var = AncillaryVariable(self.data_real)
        result = ancill_var.lazy_data()
        self.assert_equal_lazy_arrays_and_dtypes(result, self.data_lazy)

    def test_lazy_core(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        result = ancill_var.lazy_data()
        assert result is self.data_lazy


class Test_has_lazy_data(AncillaryVariableTestMixin):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.setup_test_arrays()

    def test_real_core(self):
        ancill_var = AncillaryVariable(self.data_real)
        result = ancill_var.has_lazy_data()
        assert not result

    def test_lazy_core(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        result = ancill_var.has_lazy_data()
        assert result

    def test_lazy_core_realise(self):
        ancill_var = AncillaryVariable(self.data_lazy)
        ancill_var.data
        result = ancill_var.has_lazy_data()
        assert not result


class Test__getitem__(AncillaryVariableTestMixin):
    # Test for AncillaryVariable indexing with various types of data.
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.setup_test_arrays()

    def test_partial_slice_data_copy(self):
        parent_ancill_var = AncillaryVariable([1.0, 2.0, 3.0])
        sub_ancill_var = parent_ancill_var[:1]
        values_before_change = sub_ancill_var.data.copy()
        parent_ancill_var.data[:] = -999.9
        _shared_utils.assert_array_equal(sub_ancill_var.data, values_before_change)

    def test_full_slice_data_copy(self):
        parent_ancill_var = AncillaryVariable([1.0, 2.0, 3.0])
        sub_ancill_var = parent_ancill_var[:]
        values_before_change = sub_ancill_var.data.copy()
        parent_ancill_var.data[:] = -999.9
        _shared_utils.assert_array_equal(sub_ancill_var.data, values_before_change)

    def test_dtypes(self):
        # Index ancillary variables with real+lazy data, and either an int or
        # floating dtype.
        # Check that dtypes remain the same in all cases, taking the dtypes
        # directly from the core data as we have no masking).
        for main_ancill_var, data_lazyness in data_all_dtypes_and_lazynesses(self):
            sub_ancill_var = main_ancill_var[:2, 1]

            ancill_var_dtype = main_ancill_var.dtype
            msg = (
                "Indexing main_ancill_var of dtype {} with {} data changed"
                "dtype of {} to {}."
            )

            sub_data = sub_ancill_var.core_data()
            assert sub_data.dtype == ancill_var_dtype, msg.format(
                ancill_var_dtype, data_lazyness, "data", sub_data.dtype
            )

    def test_lazyness(self):
        # Index ancillary variables with real+lazy data, and either an int or
        # floating dtype.
        # Check that lazy data stays lazy and real stays real, in all cases.
        for main_ancill_var, data_lazyness in data_all_dtypes_and_lazynesses(self):
            sub_ancill_var = main_ancill_var[:2, 1]

            msg = (
                "Indexing main_ancill_var of dtype {} with {} data "
                "changed laziness of {} from {!r} to {!r}."
            )
            ancill_var_dtype = main_ancill_var.dtype
            sub_data_lazyness = lazyness_string(sub_ancill_var.core_data())
            assert sub_data_lazyness == data_lazyness, msg.format(
                ancill_var_dtype,
                data_lazyness,
                "data",
                data_lazyness,
                sub_data_lazyness,
            )

    def test_real_data_copies(self):
        # Index ancillary variables with real+lazy data.
        # In all cases, check that any real arrays are copied by the indexing.
        for main_ancill_var, data_lazyness in data_all_dtypes_and_lazynesses(self):
            sub_ancill_var = main_ancill_var[:2, 1]

            msg = (
                "Indexed ancillary variable with {} data "
                "does not have its own separate {} array."
            )
            if data_lazyness == "real":
                main_data = main_ancill_var.core_data()
                sub_data = sub_ancill_var.core_data()
                sub_main_data = main_data[:2, 1]
                self.assert_equal_real_arrays_and_dtypes(sub_data, sub_main_data)
                self.assert_arrays_do_not_share_data(
                    sub_data,
                    sub_main_data,
                    msg.format(data_lazyness, "points"),
                )


class Test_copy(AncillaryVariableTestMixin):
    # Test for AncillaryVariable.copy() with various types of data.
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.setup_test_arrays()

    def test_lazyness(self):
        # Copy ancillary variables with real+lazy data, and either an int or
        # floating dtype.
        # Check that lazy data stays lazy and real stays real, in all cases.
        for main_ancill_var, data_lazyness in data_all_dtypes_and_lazynesses(self):
            ancill_var_dtype = main_ancill_var.dtype
            copied_ancill_var = main_ancill_var.copy()

            msg = (
                "Copying main_ancill_var of dtype {} with {} data "
                "changed lazyness of {} from {!r} to {!r}."
            )

            copied_data_lazyness = lazyness_string(copied_ancill_var.core_data())
            assert copied_data_lazyness == data_lazyness, msg.format(
                ancill_var_dtype,
                data_lazyness,
                "points",
                data_lazyness,
                copied_data_lazyness,
            )

    def test_realdata_copies(self):
        # Copy ancillary variables with real+lazy data.
        # In all cases, check that any real arrays are copies, not views.
        for main_ancill_var, data_lazyness in data_all_dtypes_and_lazynesses(self):
            copied_ancill_var = main_ancill_var.copy()

            msg = (
                "Copied ancillary variable with {} data "
                "does not have its own separate {} array."
            )

            if data_lazyness == "real":
                main_data = main_ancill_var.core_data()
                copied_data = copied_ancill_var.core_data()
                self.assert_equal_real_arrays_and_dtypes(main_data, copied_data)
                self.assert_arrays_do_not_share_data(
                    main_data, copied_data, msg.format(data_lazyness, "points")
                )


class Test_data__getter(AncillaryVariableTestMixin):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.setup_test_arrays()

    def test_mutable_real_data(self):
        # Check that ancill_var.data returns a modifiable array, and changes
        # to it are reflected to the ancillary_var.
        data = np.array([1.0, 2.0, 3.0, 4.0])
        ancill_var = AncillaryVariable(data)
        initial_values = data.copy()
        ancill_var.data[1:2] += 33.1
        result = ancill_var.data
        assert not np.all(result == initial_values)

    def test_real_data(self):
        # Getting real data does not change or copy them.
        ancill_var = AncillaryVariable(self.data_real)
        result = ancill_var.data
        self.assert_arrays_share_data(
            result,
            self.data_real,
            "Data values do not share data with the provided array.",
        )

    def test_lazy_data(self):
        # Getting lazy data realises them.
        ancill_var = AncillaryVariable(self.data_lazy)
        assert ancill_var.has_lazy_data()
        result = ancill_var.data
        assert not ancill_var.has_lazy_data()
        self.assert_equal_real_arrays_and_dtypes(result, self.data_real)


class Test_data__setter(AncillaryVariableTestMixin):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.setup_test_arrays()

    def test_real_set_real(self):
        # Setting new real data does not make a copy.
        ancill_var = AncillaryVariable(self.data_real)
        new_data = self.data_real + 102.3
        ancill_var.data = new_data
        result = ancill_var.core_data()
        self.assert_arrays_share_data(
            result,
            new_data,
            "Data values do not share data with the assigned array.",
        )

    def test_fail_bad_shape(self):
        # Setting real data requires matching shape.
        ancill_var = AncillaryVariable([1.0, 2.0])
        msg = r"Require data with shape \(2,\), got \(3,\)"
        with pytest.raises(ValueError, match=msg):
            ancill_var.data = np.array([1.0, 2.0, 3.0])

    def test_real_set_lazy(self):
        # Setting new lazy data does not make a copy.
        ancill_var = AncillaryVariable(self.data_real)
        new_data = self.data_lazy + 102.3
        ancill_var.data = new_data
        result = ancill_var.core_data()
        self.assert_equal_lazy_arrays_and_dtypes(result, new_data)


class Test__str__:
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
        assert expected == ancillary_var.__str__()

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
                    "(hours since 1970-01-01 01:00, standard calendar)"
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
        assert expected == ancillary_var.__str__()


class Test__repr__:
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
        assert expected == ancillary_var.__repr__()

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
        assert expected == ancillary_var.__repr__()


class Test___binary_operator__(AncillaryVariableTestMixin):
    # Test maths operations on on real+lazy data.
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.setup_test_arrays()

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
            self.assert_equal_real_arrays_and_dtypes(expected_data, data)
        else:
            self.assert_equal_lazy_arrays_and_dtypes(expected_data, data)

    def test_add(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            result = ancill_var + 10
            expected_data = orig_data + 10
            self._check(result, expected_data, data_lazyness)

    def test_add_inplace(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            ancill_var += 10
            expected_data = orig_data + 10
            self._check(ancill_var, expected_data, data_lazyness)

    def test_right_add(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            result = 10 + ancill_var
            expected_data = 10 + orig_data
            self._check(result, expected_data, data_lazyness)

    def test_subtract(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            result = ancill_var - 10
            expected_data = orig_data - 10
            self._check(result, expected_data, data_lazyness)

    def test_subtract_inplace(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            ancill_var -= 10
            expected_data = orig_data - 10
            self._check(ancill_var, expected_data, data_lazyness)

    def test_right_subtract(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            result = 10 - ancill_var
            expected_data = 10 - orig_data
            self._check(result, expected_data, data_lazyness)

    def test_multiply(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            result = ancill_var * 10
            expected_data = orig_data * 10
            self._check(result, expected_data, data_lazyness)

    def test_multiply_inplace(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            ancill_var *= 10
            expected_data = orig_data * 10
            self._check(ancill_var, expected_data, data_lazyness)

    def test_right_multiply(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            result = 10 * ancill_var
            expected_data = 10 * orig_data
            self._check(result, expected_data, data_lazyness)

    def test_divide(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            result = ancill_var / 10
            expected_data = orig_data / 10
            self._check(result, expected_data, data_lazyness)

    def test_divide_inplace(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            ancill_var /= 10
            expected_data = orig_data / 10
            self._check(ancill_var, expected_data, data_lazyness)

    def test_right_divide(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            result = 10 / ancill_var
            expected_data = 10 / orig_data
            self._check(result, expected_data, data_lazyness)

    def test_negative(self):
        for ancill_var, orig_data, data_lazyness in self.test_combinations:
            result = -ancill_var
            expected_data = -orig_data
            self._check(result, expected_data, data_lazyness)


class Test_has_bounds:
    def test(self):
        ancillary_var = AncillaryVariable(np.array([2, 9, 5]))
        assert not ancillary_var.has_bounds()


class Test_convert_units:
    def test_preserves_lazy(self):
        test_data = np.array([[11.1, 12.2, 13.3], [21.4, 22.5, 23.6]])
        lazy_data = as_lazy_data(test_data)
        ancill_var = AncillaryVariable(data=lazy_data, units="m")
        ancill_var.convert_units("ft")
        assert ancill_var.has_lazy_data()
        test_data_ft = Unit("m").convert(test_data, "ft")
        _shared_utils.assert_array_all_close(ancill_var.data, test_data_ft)


class Test_is_compatible:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.ancill_var = AncillaryVariable(
            [1.0, 8.0, 22.0], standard_name="number_of_observations", units="1"
        )
        self.modified_ancill_var = self.ancill_var.copy()

    def test_not_compatible_diff_name(self):
        # Different name() - not compatible
        self.modified_ancill_var.rename("air_temperature")
        assert not self.ancill_var.is_compatible(self.modified_ancill_var)

    def test_not_compatible_diff_units(self):
        # Different units- not compatible
        self.modified_ancill_var.units = "m"
        assert not self.ancill_var.is_compatible(self.modified_ancill_var)

    def test_not_compatible_diff_common_attrs(self):
        # Different common attributes - not compatible.
        self.ancill_var.attributes["source"] = "A"
        self.modified_ancill_var.attributes["source"] = "B"
        assert not self.ancill_var.is_compatible(self.modified_ancill_var)

    def test_compatible_diff_data(self):
        # Different data values - compatible.
        self.modified_ancill_var.data = [10.0, 20.0, 100.0]
        assert self.ancill_var.is_compatible(self.modified_ancill_var)

    def test_compatible_diff_var_name(self):
        # Different var_name (but same name()) - compatible.
        self.modified_ancill_var.var_name = "obs_num"
        assert self.ancill_var.is_compatible(self.modified_ancill_var)

    def test_compatible_diff_non_common_attributes(self):
        # Different non-common attributes - compatible.
        self.ancill_var.attributes["source"] = "A"
        self.modified_ancill_var.attributes["origin"] = "B"
        assert self.ancill_var.is_compatible(self.modified_ancill_var)

    def test_compatible_ignore_common_attribute(self):
        # ignore different common attributes - compatible.
        self.ancill_var.attributes["source"] = "A"
        self.modified_ancill_var.attributes["source"] = "B"
        assert self.ancill_var.is_compatible(self.modified_ancill_var, ignore="source")


class TestEquality:
    def test_nanpoints_eq_self(self):
        av1 = AncillaryVariable([1.0, np.nan, 2.0])
        assert av1 == av1

    def test_nanpoints_eq_copy(self):
        av1 = AncillaryVariable([1.0, np.nan, 2.0])
        av2 = av1.copy()
        assert av1 == av2


class Test_cube_dims:
    def test_cube_dims(self, mocker):
        # Check that "coord.cube_dims(cube)" calls "cube.coord_dims(coord)".
        mock_dims_result = mocker.sentinel.AV_DIMS
        mock_dims_call = mocker.Mock(return_value=mock_dims_result)
        mock_cube = mocker.Mock(Cube, ancillary_variable_dims=mock_dims_call)
        test_var = AncillaryVariable([1], long_name="test_name")

        result = test_var.cube_dims(mock_cube)
        assert result == mock_dims_result
        assert mock_dims_call.call_args_list == [mocker.call(test_var)]
