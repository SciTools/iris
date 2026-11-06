# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for
:func:`iris.fileformats.pp_load_rules._convert_scalar_realization_coords`.

"""

from iris.coords import DimCoord
from iris.fileformats.pp_load_rules import _convert_scalar_realization_coords


class Test:
    def test_valid(self):
        coords_and_dims = _convert_scalar_realization_coords(lbrsvd4=21)
        assert coords_and_dims == [
            (DimCoord([21], standard_name="realization", units="1"), None)
        ]

    def test_missing_indicator(self):
        coords_and_dims = _convert_scalar_realization_coords(lbrsvd4=0)
        assert coords_and_dims == []
