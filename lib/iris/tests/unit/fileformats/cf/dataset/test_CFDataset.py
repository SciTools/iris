# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.dataset.CFDataset` and friends."""

import numpy as np
import pytest

from iris.fileformats.cf.dataset import CFDataset, CFDatasetVariable


class MinimalVariable(CFDatasetVariable):
    """The smallest thing that satisfies CFDatasetVariable."""

    name = "air_temperature"
    location = "<memory>"
    dimensions = ("time", "lat")
    shape = (3, 4)
    dtype = np.dtype("f4")
    size = 12
    fill_value = None
    chunking = None
    attributes: dict = {}

    def __getitem__(self, keys):
        return np.zeros(self.shape, dtype=self.dtype)[keys]

    def __setitem__(self, keys, values):
        raise NotImplementedError

    def write_handle(self):
        return self


class MinimalDataset(CFDataset):
    """The smallest thing that satisfies CFDataset."""

    location = "<memory>"
    mode = "r"
    closed = False
    variables: dict = {}
    dimensions: dict = {}
    attributes: dict = {}

    def __init__(self):
        self.closes = 0

    def create_dimension(self, name, size):
        raise NotImplementedError

    def create_variable(self, name, dtype, dimensions=(), *, fill_value=None, **kw):
        raise NotImplementedError

    def sync(self):
        pass

    def finalise(self):
        pass

    def close(self):
        self.closes += 1


class TestAbstractness:
    def test_variable_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="abstract"):
            CFDatasetVariable()

    def test_dataset_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="abstract"):
            CFDataset()

    @pytest.mark.parametrize(
        "name",
        [
            "name",
            "location",
            "dimensions",
            "shape",
            "dtype",
            "size",
            "fill_value",
            "chunking",
            "attributes",
            "ndim",
            "__getitem__",
            "__setitem__",
            "write_handle",
        ],
    )
    def test_variable_declares_member(self, name):
        assert name in CFDatasetVariable.__abstractmethods__ or hasattr(
            CFDatasetVariable, name
        )

    @pytest.mark.parametrize(
        "name",
        [
            "location",
            "mode",
            "closed",
            "variables",
            "dimensions",
            "attributes",
            "create_dimension",
            "create_variable",
            "sync",
            "finalise",
            "close",
        ],
    )
    def test_dataset_declares_member(self, name):
        assert name in CFDataset.__abstractmethods__ or hasattr(CFDataset, name)


class TestConcreteDefaults:
    def test_ndim(self):
        assert MinimalVariable().ndim == 2

    def test_len_is_the_leading_dimension(self):
        assert len(MinimalVariable()) == 3

    def test_len_of_a_scalar_matches_netcdf4(self):
        # netCDF4.Variable raises TypeError, and CFVariable.__len__ forwards
        # to it today, so anything catching that keeps working.
        class Scalar(MinimalVariable):
            shape = ()

        with pytest.raises(TypeError, match="unsized"):
            len(Scalar())

    def test_deprecated_netcdf_member_raises_attribute_error(self):
        # A store with no backing netCDF4 object - Zarr, say - has no fallback
        # to offer, so the compatibility route simply does not apply.
        with pytest.raises(AttributeError, match="getncattr"):
            MinimalVariable().deprecated_netcdf_member("getncattr")

    def test_context_manager_closes(self):
        dataset = MinimalDataset()
        with dataset as entered:
            assert entered is dataset
            assert dataset.closes == 0
        assert dataset.closes == 1

    def test_context_manager_closes_on_exception(self):
        dataset = MinimalDataset()
        with pytest.raises(ValueError, match="boom"):
            with dataset:
                raise ValueError("boom")
        assert dataset.closes == 1
