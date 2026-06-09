# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.common.metadata.hexdigest`."""

import numpy as np
import numpy.ma as ma
import pytest
from xxhash import xxh64, xxh64_hexdigest

from iris.common.metadata import hexdigest


class TestBytesLikeObject:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.hasher = xxh64()
        self.hasher.reset()

    @staticmethod
    def _ndarray(value):
        parts = str((value.shape, xxh64_hexdigest(value)))
        return xxh64_hexdigest(parts)

    @staticmethod
    def _masked(value):
        parts = str(
            (
                value.shape,
                xxh64_hexdigest(value.data),
                xxh64_hexdigest(value.mask),
            )
        )
        return xxh64_hexdigest(parts)

    def test_string(self):
        value = "hello world"
        self.hasher.update(value)
        expected = self.hasher.hexdigest()
        assert hexdigest(value) == expected

    def test_numpy_array_int(self):
        value = np.arange(10, dtype=np.int_)
        expected = self._ndarray(value)
        assert hexdigest(value) == expected

    def test_numpy_array_float(self):
        value = np.arange(10, dtype=np.float64)
        expected = self._ndarray(value)
        assert hexdigest(value) == expected

    def test_numpy_array_float_not_int(self):
        ivalue = np.arange(10, dtype=np.int_)
        fvalue = np.arange(10, dtype=np.float64)
        expected = self._ndarray(ivalue)
        assert hexdigest(fvalue) != expected

    def test_numpy_array_reshape(self):
        value = np.arange(10).reshape(2, 5)
        expected = self._ndarray(value)
        assert hexdigest(value) == expected

    def test_numpy_array_reshape_not_flat(self):
        value = np.arange(10).reshape(2, 5)
        expected = self._ndarray(value)
        assert hexdigest(value.flatten()) != expected

    def test_masked_array_int(self):
        value = ma.arange(10, dtype=np.int_)
        expected = self._masked(value)
        assert hexdigest(value) == expected

        value[0] = ma.masked
        assert hexdigest(value) != expected
        expected = self._masked(value)
        assert hexdigest(value) == expected

    def test_masked_array_float(self):
        value = ma.arange(10, dtype=np.float64)
        expected = self._masked(value)
        assert hexdigest(value) == expected

        value[0] = ma.masked
        assert hexdigest(value) != expected
        expected = self._masked(value)
        assert hexdigest(value) == expected

    def test_masked_array_float_not_int(self):
        ivalue = ma.arange(10, dtype=np.int_)
        fvalue = ma.arange(10, dtype=np.float64)
        expected = self._masked(ivalue)
        assert hexdigest(fvalue) != expected

    def test_masked_array_not_array(self):
        value = ma.arange(10)
        expected = self._masked(value)
        assert hexdigest(value.data) != expected

    def test_masked_array_reshape(self):
        value = ma.arange(10).reshape(2, 5)
        expected = self._masked(value)
        assert hexdigest(value) == expected

    def test_masked_array_reshape_not_flat(self):
        value = ma.arange(10).reshape(2, 5)
        expected = self._masked(value)
        assert hexdigest(value.flatten()) != expected


class TestNotBytesLikeObject:
    def _expected(self, value):
        parts = str((type(value), value))
        return xxh64_hexdigest(parts)

    def test_int(self):
        value = 123
        expected = self._expected(value)
        assert hexdigest(value) == expected

    def test_numpy_int(self):
        value = int(123)
        expected = self._expected(value)
        assert hexdigest(value) == expected

    def test_float(self):
        value = 123.4
        expected = self._expected(value)
        assert hexdigest(value) == expected

    def test_numpy_float(self):
        value = float(123.4)
        expected = self._expected(value)
        assert hexdigest(value) == expected

    def test_list(self):
        value = [1, 2, 3]
        expected = self._expected(value)
        assert hexdigest(value) == expected

    def test_tuple(self):
        value = (1, 2, 3)
        expected = self._expected(value)
        assert hexdigest(value) == expected

    def test_dict(self):
        value = dict(one=1, two=2, three=3)
        expected = self._expected(value)
        assert hexdigest(value) == expected

    def test_sentinel(self, mocker):
        value = mocker.sentinel.value
        expected = self._expected(value)
        assert hexdigest(value) == expected

    def test_instance(self):
        class Dummy:
            pass

        value = Dummy()
        expected = self._expected(value)
        assert hexdigest(value) == expected

    def test_int_not_str(self):
        value = 123
        expected = self._expected(value)
        assert hexdigest(str(value)) != expected
