# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coords.Connectivity` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from xml.dom import minidom

import numpy as np
from numpy import ma

from iris.coords import Connectivity
from iris._lazy_data import as_lazy_data, is_lazy_data


class TestStandard(tests.IrisTest):
    def setUp(self):
        # Crete an instance, with non-default arguments to allow testing of
        # correct property setting.
        self.kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((3, -1)),
            "cf_role": "face_node_connectivity",
            "long_name": "my_face_nodes",
            "var_name": "face_nodes",
            "attributes": {"notes": "this is a test"},
            "start_index": 1,
            "element_dim": 0,
        }
        self.connectivity = Connectivity(**self.kwargs)

    def test_cf_role(self):
        self.assertEqual(self.kwargs["cf_role"], self.connectivity.cf_role)

    def test_cf_role_element(self):
        expected = self.kwargs["cf_role"].split("_")[0]
        self.assertEqual(expected, self.connectivity.cf_role_element)

    def test_cf_role_indexed_element(self):
        expected = self.kwargs["cf_role"].split("_")[1]
        self.assertEqual(expected, self.connectivity.cf_role_indexed_element)

    def test_start_index(self):
        self.assertEqual(
            self.kwargs["start_index"], self.connectivity.start_index
        )

    def test_element_dim(self):
        self.assertEqual(
            self.kwargs["element_dim"], self.connectivity.element_dim
        )

    def test_indices(self):
        self.assertArrayEqual(
            self.kwargs["indices"], self.connectivity.indices
        )

    def test_element_lengths(self):
        expected = [3, 3, 3]
        self.assertArrayEqual(expected, self.connectivity.element_lengths)

    def test_has_equal_element_lengths(self):
        self.assertTrue(self.connectivity.has_equal_element_lengths)

    def test_read_only(self):
        attributes = ("indices", "cf_role", "start_index", "element_dim")
        for attribute in attributes:
            self.assertRaisesRegex(
                AttributeError,
                "can't set attribute",
                setattr,
                self.connectivity,
                attribute,
                1,
            )

    def test_switch_start_index(self):
        expected_index = 1 - self.kwargs["start_index"]
        expected_change = expected_index - self.kwargs["start_index"]
        expected_indices = self.kwargs["indices"] + expected_change
        self.connectivity.switch_start_index()
        self.assertEqual(expected_index, self.connectivity.start_index)
        self.assertArrayEqual(expected_indices, self.connectivity.indices)

    def test_switch_element_dim(self):
        expected_dim = 1 - self.kwargs["element_dim"]
        expected_indices = self.kwargs["indices"].swapaxes(0, 1)
        self.connectivity.switch_element_dim()
        self.assertEqual(expected_dim, self.connectivity.element_dim)
        self.assertArrayEqual(expected_indices, self.connectivity.indices)

    def test_lazy_indices(self):
        self.assertTrue(is_lazy_data(self.connectivity.lazy_indices()))

    def test_core_indices(self):
        self.assertArrayEqual(
            self.kwargs["indices"], self.connectivity.core_indices()
        )

    def test_has_lazy_indices(self):
        self.assertFalse(self.connectivity.has_lazy_indices())

    def test___str__(self):
        expected = (
            "Connectivity([[1 2 3]\n [4 5 6]\n [7 8 9]]), "
            "cf_role='face_node_connectivity', start_index=1, element_dim=0, "
            "long_name='my_face_nodes', var_name='face_nodes', "
            "attributes={'notes': 'this is a test'})"
        )
        self.assertEqual(expected, self.connectivity.__str__())

    def test___repr__(self):
        expected = (
            "Connectivity([[1 2 3]\n [4 5 6]\n [7 8 9]]), "
            "cf_role='face_node_connectivity', start_index=1, element_dim=0, "
            "long_name='my_face_nodes', var_name='face_nodes', "
            "attributes={'notes': 'this is a test'})"
        )
        self.assertEqual(expected, self.connectivity.__repr__())

    def test_xml_element(self):
        doc = minidom.Document()
        connectivity_element = self.connectivity.xml_element(doc)
        self.assertEqual(connectivity_element.tagName, "connectivity")
        for attribute in ("cf_role", "start_index", "element_dim"):
            self.assertIn(attribute, connectivity_element.attributes)

    def test___eq__(self):
        self.assertEqual(self.connectivity, self.connectivity)

    def test_no_cube_dims(self):
        self.assertRaises(NotImplementedError, self.connectivity.cube_dims, 1)

    def test_shape(self):
        self.assertEqual(self.kwargs["indices"].shape, self.connectivity.shape)

    def test_ndim(self):
        self.assertEqual(self.kwargs["indices"].ndim, self.connectivity.ndim)

    def test___getitem_(self):
        subset = self.connectivity[:, 0:1]
        self.assertArrayEqual(self.kwargs["indices"][:, 0:1], subset.indices)

    def test___getitem__data_copy(self):
        # Check that a sliced connectivity has independent data.
        subset = self.connectivity[:, 1:2]
        old_indices = subset.indices.copy()
        # Change the original one.
        self.connectivity.switch_start_index()
        # Check the new one has not changed.
        self.assertArrayEqual(old_indices, subset.indices)

    def test_copy(self):
        new_indices = np.linspace(11, 16, 6, dtype=int).reshape((3, -1))
        copy_connectivity = self.connectivity.copy(new_indices)
        self.assertArrayEqual(new_indices, copy_connectivity.indices)


class TestAltIndices(tests.IrisTest):
    def common(self, indices):
        connectivity = Connectivity(
            indices=indices, cf_role="face_node_connectivity"
        )
        self.assertArrayEqual(indices, connectivity.indices)

    def test_int32(self):
        indices = np.linspace(1, 9, 9, dtype=np.int32).reshape((-1, 3))
        self.common(indices)

    def test_uint32(self):
        indices = np.linspace(1, 9, 9, dtype=np.uint32).reshape((-1, 3))
        self.common(indices)

    def test_lazy(self):
        indices = as_lazy_data(
            np.linspace(1, 9, 9, dtype=int).reshape((-1, 3))
        )
        self.common(indices)

    def test_masked(self):
        mask = [0, 0, 0, 1] * 3
        data = np.linspace(1, 12, 12, dtype=int).reshape((-1, 4))
        indices = ma.array(data=data, mask=mask)
        self.common(indices)

    def test_masked_lazy(self):
        mask = [0, 0, 0, 1] * 3
        data = np.linspace(1, 12, 12, dtype=int).reshape((-1, 4))
        indices = as_lazy_data(ma.array(data=data, mask=mask))
        self.common(indices)


class TestValidations(tests.IrisTest):
    def test_start_index(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "face_node_connectivity",
            "start_index": 2,
        }
        self.assertRaisesRegex(
            ValueError, "Invalid start_index .", Connectivity, **kwargs
        )

    def test_element_dim(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "face_node_connectivity",
            "element_dim": 2,
        }
        self.assertRaisesRegex(
            ValueError, "Invalid element_dim .", Connectivity, **kwargs
        )

    def test_cf_role(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "error",
        }
        self.assertRaisesRegex(
            ValueError, "Invalid cf_role .", Connectivity, **kwargs
        )

    def test_indices_int(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9).reshape((-1, 3)),
            "cf_role": "face_node_connectivity",
        }
        self.assertRaisesRegex(
            ValueError,
            "dtype must be numpy integer subtype",
            Connectivity,
            **kwargs,
        )

    def test_indices_start_index(self):
        kwargs = {
            "indices": np.linspace(-9, -1, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "face_node_connectivity",
        }
        self.assertRaisesRegex(
            ValueError, " < start_index", Connectivity, **kwargs
        )

    def test_indices_dims_low(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int),
            "cf_role": "face_node_connectivity",
        }
        self.assertRaisesRegex(
            ValueError, "Expected 2-dimensional shape,", Connectivity, **kwargs
        )

    def test_indices_dims_high(self):
        kwargs = {
            "indices": np.linspace(1, 12, 12, dtype=int).reshape((-1, 3, 2)),
            "cf_role": "face_node_connectivity",
        }
        self.assertRaisesRegex(
            ValueError, "Expected 2-dimensional shape,", Connectivity, **kwargs
        )

    def test_indices_elements_edge(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "edge_node_connectivity",
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all elements meet requirement: len=2",
            Connectivity,
            **kwargs,
        )

    def test_indices_elements_face(self):
        kwargs = {
            "indices": np.linspace(1, 6, 6, dtype=int).reshape((-1, 2)),
            "cf_role": "face_node_connectivity",
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all elements meet requirement: len>=3",
            Connectivity,
            **kwargs,
        )

    def test_indices_elements_volume_face(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "volume_face_connectivity",
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all elements meet requirement: len>=4",
            Connectivity,
            **kwargs,
        )

    def test_indices_elements_volume_edge(self):
        kwargs = {
            "indices": np.linspace(1, 12, 12, dtype=int).reshape((-1, 3)),
            "cf_role": "volume_edge_connectivity",
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all elements meet requirement: len>=6",
            Connectivity,
            **kwargs,
        )

    def test_indices_elements_alt_dim(self):
        """The transposed equivalent of `test_indices_elements_volume_face`."""
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((3, -1)),
            "cf_role": "volume_face_connectivity",
            "element_dim": 1,
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all elements meet requirement: len>=4",
            Connectivity,
            **kwargs,
        )
