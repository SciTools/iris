# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests fixture infra-structure."""

import dask
import pytest

import iris


@pytest.fixture
def sample_coord():
    sample_coord = iris.coords.DimCoord(points=(1, 2, 3, 4, 5))
    return sample_coord


@pytest.fixture
def mocked_compute(monkeypatch, mocker):
    """A fixture to provide a mock for `dask.compute()`."""
    m_compute = mocker.Mock(wraps=dask.base.compute)

    # The three dask compute functions are all the same function but monkeypatch
    # does not automatically know that.
    # https://stackoverflow.com/questions/77820437
    monkeypatch.setattr(dask.base, dask.base.compute.__name__, m_compute)
    monkeypatch.setattr(dask, dask.compute.__name__, m_compute)
    monkeypatch.setattr(dask.array, dask.array.compute.__name__, m_compute)

    return m_compute
