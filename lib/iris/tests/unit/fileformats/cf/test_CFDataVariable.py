# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFDataVariable`."""

import pytest

from iris.fileformats.cf import CFDataVariable


class TestIdentify:
    def test_identify_raises_not_implemented(self, named_variable):
        vars_all = {"data_var": named_variable("data_var")}
        with pytest.raises(NotImplementedError):
            CFDataVariable.identify(vars_all)


class TestConstructor:
    def test_cf_name_and_data_stored(self, named_variable):
        nc_var = named_variable("data_var")
        cf_var = CFDataVariable("data_var", nc_var)
        assert cf_var.cf_name == "data_var"
        assert cf_var.cf_data is nc_var
