# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFDataVariable`."""

import pytest

from iris.fileformats.cf import CFDataVariable

from .identify_mixins import _NetCDFVar


class TestIdentify:
    def test_identify_raises_not_implemented(self):
        vars_all = {"data_var": _NetCDFVar("data_var")}
        with pytest.raises(NotImplementedError):
            CFDataVariable.identify(vars_all)


class TestConstructor:
    def test_cf_name_and_data_stored(self):
        nc_var = _NetCDFVar("data_var")
        cf_var = CFDataVariable("data_var", nc_var)
        assert cf_var.cf_name == "data_var"
        assert cf_var.cf_data is nc_var
