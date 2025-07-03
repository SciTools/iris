# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests fixture infra-structure."""

import pytest

import iris


@pytest.fixture
def sample_coord():
    sample_coord = iris.coords.DimCoord(points=(1, 2, 3, 4, 5))
    return sample_coord
