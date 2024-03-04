# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :func:`iris.fileformats.pp_load_rules._dim_or_aux`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.coords import AuxCoord, DimCoord
from iris.fileformats.pp_load_rules import _dim_or_aux


class Test(tests.IrisTest):
    def setUp(self):
        self.mono = list(range(5))
        self.non_mono = [0, 1, 3, 2, 4]
        self.std_name = "depth"
        self.units = "m"
        self.attr = {"positive": "up", "wibble": "wobble"}

    def test_dim_monotonic(self):
        result = _dim_or_aux(
            self.mono,
            standard_name=self.std_name,
            units=self.units,
            attributes=self.attr.copy(),
        )
        expected = DimCoord(
            self.mono,
            standard_name=self.std_name,
            units=self.units,
            attributes=self.attr,
        )
        self.assertEqual(result, expected)

    def test_dim_non_monotonic(self):
        result = _dim_or_aux(
            self.non_mono,
            standard_name=self.std_name,
            units=self.units,
            attributes=self.attr.copy(),
        )
        attr = self.attr.copy()
        del attr["positive"]
        expected = AuxCoord(
            self.non_mono,
            standard_name=self.std_name,
            units=self.units,
            attributes=attr,
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    tests.main()
