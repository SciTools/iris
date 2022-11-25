# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""Pytest fixtures for the tests."""

import pytest

from iris.experimental.ugrid import Connectivity


# TODO: iris#5061 HANG
@pytest.fixture(scope="session", autouse=True)
def no_connectivity_validation():
    """File locking behaviour assumes no data access during Cube creation."""

    def mock__validate_indices(self, indices, shapes_only=False):
        pass

    original_validate = Connectivity._validate_indices
    Connectivity._validate_indices = mock__validate_indices
    yield
    Connectivity._validate_indices = original_validate
