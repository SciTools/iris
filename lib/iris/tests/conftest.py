# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""Pytest fixtures for the tests."""

import threading

import pytest

from iris.experimental.ugrid import Connectivity
from iris.fileformats.netcdf import loader as nc_loader


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

    # with monkeypatch.context() as m:
    #     m.setattr(
    #         Connectivity,
    #         "_validate_indices",
    #         mock__validate_indices
    #     )
    #     yield


# TODO: iris#5061 FAIL
@pytest.fixture()
def no_file_lock_reuse(monkeypatch):
    """For use in mock tests, where it is irrelevant anyway."""

    def mock_get_filepath_lock(path, already_exists=None):
        return threading.RLock()

    with monkeypatch.context() as m:
        m.setattr(nc_loader, "get_filepath_lock", mock_get_filepath_lock)
        yield
