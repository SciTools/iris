# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for cube arithmetic involving derived (i.e. factory) coords."""

from iris.tests import _shared_utils
from iris.tests.unit.analysis.maths import (
    CubeArithmeticBroadcastingTestMixin,
    MathsAddOperationMixin,
)


@_shared_utils.skip_data
class TestBroadcastingDerived(
    MathsAddOperationMixin,
    CubeArithmeticBroadcastingTestMixin,
):
    """Repeat the broadcasting tests while retaining derived coordinates.

    NOTE: apart from showing that these operations do succeed, this mostly
    produces a new set of CML result files,
    in "lib/iris/tests/results/unit/analysis/maths/_arith__derived_coords" .
    See there to confirm that the results preserve the derived coordinates.

    """

    def _base_testcube(self):
        return super()._base_testcube(include_derived=True)
