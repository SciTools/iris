# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for
:func:`iris.fileformats.pp_load_rules._convert_pseudo_level_coords`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.coords import DimCoord
from iris.tests.unit.fileformats import TestField

from iris.fileformats.pp_load_rules import _convert_scalar_pseudo_level_coords


class Test(TestField):
    def test_valid(self):
        coords_and_dims = _convert_scalar_pseudo_level_coords(lbuser5=21)
        self.assertEqual(
            coords_and_dims,
            [(DimCoord([21], long_name="pseudo_level", units="1"), None)],
        )

    def test_missing_indicator(self):
        coords_and_dims = _convert_scalar_pseudo_level_coords(lbuser5=0)
        self.assertEqual(coords_and_dims, [])


if __name__ == "__main__":
    tests.main()
