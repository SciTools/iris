# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris.analysis.AreaWeighted`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.analysis import AreaWeighted


class Test(tests.IrisTest):
    def check_call(self, mdtol=None):
        # Check that `iris.analysis.AreaWeighted` correctly calls an
        # `iris.analysis._area_weighted.AreaWeightedRegridder` object.
        if mdtol is None:
            area_weighted = AreaWeighted()
            mdtol = 1
        else:
            area_weighted = AreaWeighted(mdtol=mdtol)
        self.assertEqual(area_weighted.mdtol, mdtol)

        with mock.patch(
            "iris.analysis.AreaWeightedRegridder",
            return_value=mock.sentinel.regridder,
        ) as awr:
            regridder = area_weighted.regridder(
                mock.sentinel.src, mock.sentinel.target
            )

        awr.assert_called_once_with(
            mock.sentinel.src, mock.sentinel.target, mdtol=mdtol
        )
        self.assertIs(regridder, mock.sentinel.regridder)

    def test_default(self):
        self.check_call()

    def test_specified_mdtol(self):
        self.check_call(0.5)

    def test_invalid_high_mdtol(self):
        msg = "mdtol must be in range 0 - 1"
        with self.assertRaisesRegex(ValueError, msg):
            AreaWeighted(mdtol=1.2)

    def test_invalid_low_mdtol(self):
        msg = "mdtol must be in range 0 - 1"
        with self.assertRaisesRegex(ValueError, msg):
            AreaWeighted(mdtol=-0.2)


if __name__ == "__main__":
    tests.main()
