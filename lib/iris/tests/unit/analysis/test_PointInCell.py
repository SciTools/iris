# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.analysis.PointInCell`."""

from iris.analysis import PointInCell


class Test_regridder:
    def test(self, mocker):
        point_in_cell = PointInCell(mocker.sentinel.weights)

        ecr = mocker.patch(
            "iris.analysis.CurvilinearRegridder",
            return_value=mocker.sentinel.regridder,
        )
        regridder = point_in_cell.regridder(mocker.sentinel.src, mocker.sentinel.target)

        ecr.assert_called_once_with(
            mocker.sentinel.src, mocker.sentinel.target, mocker.sentinel.weights
        )
        assert regridder is mocker.sentinel.regridder
