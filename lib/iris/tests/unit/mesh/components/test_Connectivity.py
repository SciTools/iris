# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.mesh.Connectivity` class."""

from platform import python_version
from xml.dom import minidom

import numpy as np
from numpy import ma
from packaging import version
import pytest

from iris._lazy_data import as_lazy_data, is_lazy_data
from iris.mesh import Connectivity
from iris.tests import _shared_utils


class TestStandard:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # Crete an instance, with non-default arguments to allow testing of
        # correct property setting.
        self.kwargs = {
            "indices": np.linspace(1, 12, 12, dtype=int).reshape((4, -1)),
            "cf_role": "face_node_connectivity",
            "long_name": "my_face_nodes",
            "var_name": "face_nodes",
            "attributes": {"notes": "this is a test"},
            "start_index": 1,
            "location_axis": 1,
        }
        self.connectivity = Connectivity(**self.kwargs)

    def test_cf_role(self):
        assert self.kwargs["cf_role"] == self.connectivity.cf_role

    def test_location(self):
        expected = self.kwargs["cf_role"].split("_")[0]
        assert expected == self.connectivity.location

    def test_connected(self):
        expected = self.kwargs["cf_role"].split("_")[1]
        assert expected == self.connectivity.connected

    def test_start_index(self):
        assert self.kwargs["start_index"] == self.connectivity.start_index

    def test_location_axis(self):
        assert self.kwargs["location_axis"] == self.connectivity.location_axis

    def test_indices(self):
        _shared_utils.assert_array_equal(
            self.kwargs["indices"], self.connectivity.indices
        )

    def test_read_only(self):
        attributes = ("indices", "cf_role", "start_index", "location_axis")
        if version.parse(python_version()) >= version.parse("3.11"):
            msg = "object has no setter"
        else:
            msg = "can't set attribute"
        for attribute in attributes:
            with pytest.raises(AttributeError, match=msg):
                setattr(self.connectivity, attribute, 1)

    def test_transpose(self):
        expected_dim = 1 - self.kwargs["location_axis"]
        expected_indices = self.kwargs["indices"].transpose()
        new_connectivity = self.connectivity.transpose()
        assert expected_dim == new_connectivity.location_axis
        _shared_utils.assert_array_equal(expected_indices, new_connectivity.indices)

    def test_lazy_indices(self):
        assert is_lazy_data(self.connectivity.lazy_indices())

    def test_core_indices(self):
        _shared_utils.assert_array_equal(
            self.kwargs["indices"], self.connectivity.core_indices()
        )

    def test_has_lazy_indices(self):
        assert not self.connectivity.has_lazy_indices()

    def test_lazy_location_lengths(self):
        assert is_lazy_data(self.connectivity.lazy_location_lengths())

    def test_location_lengths(self):
        expected = [4, 4, 4]
        _shared_utils.assert_array_equal(expected, self.connectivity.location_lengths())

    def test___str__(self):
        expected = "\n".join(
            [
                "Connectivity :  my_face_nodes / (unknown)",
                "    data: [",
                "        [ 1,  2,  3],",
                "        [ 4,  5,  6],",
                "        [ 7,  8,  9],",
                "        [10, 11, 12]]",
                "    shape: (4, 3)",
                "    dtype: int64",
                "    long_name: 'my_face_nodes'",
                "    var_name: 'face_nodes'",
                "    attributes:",
                "        notes  'this is a test'",
                "    cf_role: 'face_node_connectivity'",
                "    start_index: 1",
                "    location_axis: 1",
            ]
        )
        assert expected == self.connectivity.__str__()

    def test___repr__(self):
        expected = (
            "<Connectivity: my_face_nodes / (unknown)  [[1, 2, 3], ...]  shape(4, 3)>"
        )
        assert expected == self.connectivity.__repr__()

    def test_xml_element(self):
        doc = minidom.Document()
        connectivity_element = self.connectivity.xml_element(doc)
        assert connectivity_element.tagName == "connectivity"
        for attribute in ("cf_role", "start_index", "location_axis"):
            assert attribute in connectivity_element.attributes

    def test___eq__(self):
        equivalent_kwargs = self.kwargs
        equivalent_kwargs["indices"] = self.kwargs["indices"].transpose()
        equivalent_kwargs["location_axis"] = 1 - self.kwargs["location_axis"]
        equivalent = Connectivity(**equivalent_kwargs)
        assert not np.array_equal(equivalent.indices, self.connectivity.indices)
        assert equivalent == self.connectivity

    def test_different(self):
        different_kwargs = self.kwargs
        different_kwargs["indices"] = self.kwargs["indices"].transpose()
        different = Connectivity(**different_kwargs)
        assert different != self.connectivity

    def test_no_cube_dims(self):
        with pytest.raises(NotImplementedError):
            self.connectivity.cube_dims(1)

    def test_shape(self):
        assert self.kwargs["indices"].shape == self.connectivity.shape

    def test_ndim(self):
        assert self.kwargs["indices"].ndim == self.connectivity.ndim

    def test___getitem_(self):
        subset = self.connectivity[:, 0:1]
        _shared_utils.assert_array_equal(self.kwargs["indices"][:, 0:1], subset.indices)

    def test_copy(self):
        new_indices = np.linspace(11, 16, 6, dtype=int).reshape((3, -1))
        copy_connectivity = self.connectivity.copy(new_indices)
        _shared_utils.assert_array_equal(new_indices, copy_connectivity.indices)

    def test_indices_by_location(self):
        expected = self.kwargs["indices"].transpose()
        _shared_utils.assert_array_equal(
            expected, self.connectivity.indices_by_location()
        )

    def test_indices_by_location_input(self):
        expected = as_lazy_data(self.kwargs["indices"].transpose())
        by_location = self.connectivity.indices_by_location(
            self.connectivity.lazy_indices()
        )
        _shared_utils.assert_array_equal(expected, by_location)


class TestAltIndices:
    @pytest.fixture(autouse=True)
    def _setup(self):
        mask = ([0, 0, 0, 0, 1] * 2) + [0, 0, 0, 1, 1]
        data = np.linspace(1, 15, 15, dtype=int).reshape((-1, 5))
        self.masked_indices = ma.array(data=data, mask=mask)
        self.lazy_indices = as_lazy_data(data)

    def common(self, indices):
        connectivity = Connectivity(indices=indices, cf_role="face_node_connectivity")
        _shared_utils.assert_array_equal(indices, connectivity.indices)

    def test_int32(self):
        indices = np.linspace(1, 9, 9, dtype=np.int32).reshape((-1, 3))
        self.common(indices)

    def test_uint32(self):
        indices = np.linspace(1, 9, 9, dtype=np.uint32).reshape((-1, 3))
        self.common(indices)

    def test_lazy(self):
        self.common(self.lazy_indices)

    def test_masked(self):
        self.common(self.masked_indices)

    def test_masked_lazy(self):
        self.common(as_lazy_data(self.masked_indices))

    def test_has_lazy_indices(self):
        connectivity = Connectivity(
            indices=self.lazy_indices, cf_role="face_node_connectivity"
        )
        assert connectivity.has_lazy_indices()


class TestValidations:
    def test_start_index(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "face_node_connectivity",
            "start_index": 2,
        }
        with pytest.raises(ValueError, match="Invalid start_index ."):
            Connectivity(**kwargs)

    def test_location_axis(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "face_node_connectivity",
            "location_axis": 2,
        }
        with pytest.raises(ValueError, match="Invalid location_axis ."):
            Connectivity(**kwargs)

    def test_cf_role(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "error",
        }
        with pytest.raises(ValueError, match="Invalid cf_role ."):
            Connectivity(**kwargs)

    def test_indices_int(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9).reshape((-1, 3)),
            "cf_role": "face_node_connectivity",
        }
        with pytest.raises(ValueError, match="dtype must be numpy integer subtype"):
            Connectivity(**kwargs)

    def test_indices_start_index(self):
        kwargs = {
            "indices": np.linspace(-9, -1, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "face_node_connectivity",
        }
        with pytest.raises(ValueError, match="< start_index"):
            Connectivity(**kwargs)

    def test_indices_dims_low(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int),
            "cf_role": "face_node_connectivity",
        }
        with pytest.raises(ValueError, match="Expected 2-dimensional shape,"):
            Connectivity(**kwargs)

    def test_indices_dims_high(self):
        kwargs = {
            "indices": np.linspace(1, 12, 12, dtype=int).reshape((-1, 3, 2)),
            "cf_role": "face_node_connectivity",
        }
        with pytest.raises(ValueError, match="Expected 2-dimensional shape,"):
            Connectivity(**kwargs)

    def test_indices_locations_edge(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "edge_node_connectivity",
        }
        with pytest.raises(ValueError, match="Not all edges meet requirement: len=2"):
            Connectivity(**kwargs)

    def test_indices_locations_face(self):
        kwargs = {
            "indices": np.linspace(1, 6, 6, dtype=int).reshape((-1, 2)),
            "cf_role": "face_node_connectivity",
        }
        with pytest.raises(ValueError, match="Not all faces meet requirement: len>=3"):
            Connectivity(**kwargs)

    def test_indices_locations_volume_face(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "volume_face_connectivity",
        }
        with pytest.raises(
            ValueError, match="Not all volumes meet requirement: len>=4"
        ):
            Connectivity(**kwargs)

    def test_indices_locations_volume_edge(self):
        kwargs = {
            "indices": np.linspace(1, 12, 12, dtype=int).reshape((-1, 3)),
            "cf_role": "volume_edge_connectivity",
        }
        with pytest.raises(
            ValueError, match="Not all volumes meet requirement: len>=6"
        ):
            Connectivity(**kwargs)

    def test_indices_locations_alt_dim(self):
        """The transposed equivalent of `test_indices_locations_volume_face`."""
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((3, -1)),
            "cf_role": "volume_face_connectivity",
            "location_axis": 1,
        }
        with pytest.raises(
            ValueError, match="Not all volumes meet requirement: len>=4"
        ):
            Connectivity(**kwargs)

    def test_indices_locations_masked(self):
        mask = ([0, 0, 0] * 2) + [0, 0, 1]
        data = np.linspace(1, 9, 9, dtype=int).reshape((3, -1))
        kwargs = {
            "indices": ma.array(data=data, mask=mask),
            "cf_role": "face_node_connectivity",
        }
        # Validation of individual location sizes (denoted by masks) only
        # available through explicit call of Connectivity.validate_indices().
        connectivity = Connectivity(**kwargs)
        with pytest.raises(ValueError, match="Not all faces meet requirement: len>=3"):
            connectivity.validate_indices()
