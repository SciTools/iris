# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Shared fixtures for CF fileformat unit tests."""

import re

import numpy as np
import pytest


class _NetCDFVariableStub:
    """Minimal netCDF-like variable object for CF identify tests."""

    def __init__(self, name, dtype):
        self.name = name
        self.dtype = np.dtype(dtype)

    def ncattrs(self):
        return [
            attr
            for attr in self.__dict__
            if not attr.startswith("_") and attr not in ["name", "dtype"]
        ]


@pytest.fixture
def named_variable():
    def _factory(name, dtype=int):
        return _NetCDFVariableStub(name=name, dtype=dtype)

    return _factory


@pytest.fixture
def assert_warning_gated():
    def _assert(operation, warning_category, warning_regex):
        with pytest.warns(warning_category, match=warning_regex):
            operation(warn=True)

        try:
            with pytest.warns(warning_category, match=warning_regex):
                operation(warn=False)
        except pytest.fail.Exception:
            pass
        else:
            pytest.fail(
                f"Operation {operation.__name__} raised {warning_category.__name__} "
                "when warn=False"
            )

    return _assert
