# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.analysis.AreaWeighted`."""

import pytest

from iris.analysis import AreaWeighted


class Test:
    def check_call(self, mocker, mdtol=None):
        # Check that `iris.analysis.AreaWeighted` correctly calls an
        # `iris.analysis._area_weighted.AreaWeightedRegridder` object.
        if mdtol is None:
            area_weighted = AreaWeighted()
            mdtol = 1
        else:
            area_weighted = AreaWeighted(mdtol=mdtol)
        assert area_weighted.mdtol == mdtol

        awr = mocker.patch(
            "iris.analysis.AreaWeightedRegridder",
            return_value=mocker.sentinel.regridder,
        )
        regridder = area_weighted.regridder(mocker.sentinel.src, mocker.sentinel.target)

        awr.assert_called_once_with(
            mocker.sentinel.src, mocker.sentinel.target, mdtol=mdtol
        )
        assert regridder is mocker.sentinel.regridder

    def test_default(self, mocker):
        self.check_call(mocker)

    def test_specified_mdtol(self, mocker):
        self.check_call(mocker, 0.5)

    def test_invalid_high_mdtol(self):
        msg = "mdtol must be in range 0 - 1"
        with pytest.raises(ValueError, match=msg):
            AreaWeighted(mdtol=1.2)

    def test_invalid_low_mdtol(self):
        msg = "mdtol must be in range 0 - 1"
        with pytest.raises(ValueError, match=msg):
            AreaWeighted(mdtol=-0.2)
