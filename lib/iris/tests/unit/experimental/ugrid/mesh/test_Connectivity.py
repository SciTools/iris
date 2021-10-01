# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.experimental.ugrid.mesh.Connectivity` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from xml.dom import minidom

import numpy as np
from numpy import ma

from iris._lazy_data import as_lazy_data, is_lazy_data
from iris.experimental.ugrid.mesh import Connectivity


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
            "src_dim": 1,
        }
        self.connectivity = Connectivity(**self.kwargs)

    def test_cf_role(self):
        self.assertEqual(self.kwargs["cf_role"], self.connectivity.cf_role)

    def test_src_location(self):
        expected = self.kwargs["cf_role"].split("_")[0]
        self.assertEqual(expected, self.connectivity.src_location)

    def test_tgt_location(self):
        expected = self.kwargs["cf_role"].split("_")[1]
        self.assertEqual(expected, self.connectivity.tgt_location)

    def test_start_index(self):
        self.assertEqual(
            self.kwargs["start_index"], self.connectivity.start_index
        )

    def test_src_dim(self):
        self.assertEqual(self.kwargs["src_dim"], self.connectivity.src_dim)

    def test_indices(self):
        self.assertArrayEqual(
            self.kwargs["indices"], self.connectivity.indices
        )

    def test_read_only(self):
        attributes = ("indices", "cf_role", "start_index", "src_dim")
        for attribute in attributes:
            self.assertRaisesRegex(
                AttributeError,
                "can't set attribute",
                setattr,
                self.connectivity,
                attribute,
                1,
            )

    def test_transpose(self):
        expected_dim = 1 - self.kwargs["src_dim"]
        expected_indices = self.kwargs["indices"].transpose()
        new_connectivity = self.connectivity.transpose()
        self.assertEqual(expected_dim, new_connectivity.src_dim)
        self.assertArrayEqual(expected_indices, new_connectivity.indices)

    def test_lazy_indices(self):
        self.assertTrue(is_lazy_data(self.connectivity.lazy_indices()))

    def test_core_indices(self):
        self.assertArrayEqual(
            self.kwargs["indices"], self.connectivity.core_indices()
        )

    def test_has_lazy_indices(self):
        self.assertFalse(self.connectivity.has_lazy_indices())

    def test_lazy_src_lengths(self):
        self.assertTrue(is_lazy_data(self.connectivity.lazy_src_lengths()))

    def test_src_lengths(self):
        expected = [3, 3, 3]
        self.assertArrayEqual(expected, self.connectivity.src_lengths())

    def test___str__(self):
        expected = (
            "Connectivity(cf_role='face_node_connectivity', start_index=1)"
        )
        self.assertEqual(expected, self.connectivity.__str__())

    def test___repr__(self):
        expected = (
            "Connectivity(array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "
            "cf_role='face_node_connectivity', long_name='my_face_nodes', "
            "var_name='face_nodes', attributes={'notes': 'this is a test'}, "
            "start_index=1, src_dim=1)"
        )
        self.assertEqual(expected, self.connectivity.__repr__())

    def test_xml_element(self):
        doc = minidom.Document()
        connectivity_element = self.connectivity.xml_element(doc)
        self.assertEqual(connectivity_element.tagName, "connectivity")
        for attribute in ("cf_role", "start_index", "src_dim"):
            self.assertIn(attribute, connectivity_element.attributes)

    def test___eq__(self):
        equivalent_kwargs = self.kwargs
        equivalent_kwargs["indices"] = self.kwargs["indices"].transpose()
        equivalent_kwargs["src_dim"] = 1 - self.kwargs["src_dim"]
        equivalent = Connectivity(**equivalent_kwargs)
        self.assertFalse(
            (equivalent.indices == self.connectivity.indices).all()
        )
        self.assertEqual(equivalent, self.connectivity)

    def test_different(self):
        different_kwargs = self.kwargs
        different_kwargs["indices"] = self.kwargs["indices"].transpose()
        different = Connectivity(**different_kwargs)
        self.assertNotEqual(different, self.connectivity)

    def test_no_cube_dims(self):
        self.assertRaises(NotImplementedError, self.connectivity.cube_dims, 1)

    def test_shape(self):
        self.assertEqual(self.kwargs["indices"].shape, self.connectivity.shape)

    def test_ndim(self):
        self.assertEqual(self.kwargs["indices"].ndim, self.connectivity.ndim)

    def test___getitem_(self):
        subset = self.connectivity[:, 0:1]
        self.assertArrayEqual(self.kwargs["indices"][:, 0:1], subset.indices)

    def test_copy(self):
        new_indices = np.linspace(11, 16, 6, dtype=int).reshape((3, -1))
        copy_connectivity = self.connectivity.copy(new_indices)
        self.assertArrayEqual(new_indices, copy_connectivity.indices)

    def test_indices_by_src(self):
        expected = self.kwargs["indices"].transpose()
        self.assertArrayEqual(expected, self.connectivity.indices_by_src())

    def test_indices_by_src_input(self):
        expected = as_lazy_data(self.kwargs["indices"].transpose())
        by_src = self.connectivity.indices_by_src(
            self.connectivity.lazy_indices()
        )
        self.assertArrayEqual(expected, by_src)


class TestAltIndices(tests.IrisTest):
    def setUp(self):
        mask = ([0, 0, 0, 0, 1] * 2) + [0, 0, 0, 1, 1]
        data = np.linspace(1, 15, 15, dtype=int).reshape((-1, 5))
        self.masked_indices = ma.array(data=data, mask=mask)
        self.lazy_indices = as_lazy_data(data)

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
        self.common(self.lazy_indices)

    def test_masked(self):
        self.common(self.masked_indices)

    def test_masked_lazy(self):
        self.common(as_lazy_data(self.masked_indices))

    def test_has_lazy_indices(self):
        connectivity = Connectivity(
            indices=self.lazy_indices, cf_role="face_node_connectivity"
        )
        self.assertTrue(connectivity.has_lazy_indices())


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

    def test_src_dim(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "face_node_connectivity",
            "src_dim": 2,
        }
        self.assertRaisesRegex(
            ValueError, "Invalid src_dim .", Connectivity, **kwargs
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

    def test_indices_locations_edge(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "edge_node_connectivity",
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all src_locations meet requirement: len=2",
            Connectivity,
            **kwargs,
        )

    def test_indices_locations_face(self):
        kwargs = {
            "indices": np.linspace(1, 6, 6, dtype=int).reshape((-1, 2)),
            "cf_role": "face_node_connectivity",
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all src_locations meet requirement: len>=3",
            Connectivity,
            **kwargs,
        )

    def test_indices_locations_volume_face(self):
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((-1, 3)),
            "cf_role": "volume_face_connectivity",
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all src_locations meet requirement: len>=4",
            Connectivity,
            **kwargs,
        )

    def test_indices_locations_volume_edge(self):
        kwargs = {
            "indices": np.linspace(1, 12, 12, dtype=int).reshape((-1, 3)),
            "cf_role": "volume_edge_connectivity",
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all src_locations meet requirement: len>=6",
            Connectivity,
            **kwargs,
        )

    def test_indices_locations_alt_dim(self):
        """The transposed equivalent of `test_indices_locations_volume_face`."""
        kwargs = {
            "indices": np.linspace(1, 9, 9, dtype=int).reshape((3, -1)),
            "cf_role": "volume_face_connectivity",
            "src_dim": 1,
        }
        self.assertRaisesRegex(
            ValueError,
            "Not all src_locations meet requirement: len>=4",
            Connectivity,
            **kwargs,
        )

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
        self.assertRaisesRegex(
            ValueError,
            "Not all src_locations meet requirement: len>=3",
            connectivity.validate_indices,
        )
