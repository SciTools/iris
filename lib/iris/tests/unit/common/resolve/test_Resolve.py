# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.resolve.Resolve`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from collections import namedtuple
from copy import deepcopy
import unittest.mock as mock
from unittest.mock import sentinel

from cf_units import Unit
import numpy as np

from iris.common.lenient import LENIENT
from iris.common.metadata import CubeMetadata
from iris.common.resolve import (
    Resolve,
    _AuxCoverage,
    _CategoryItems,
    _DimCoverage,
    _Item,
    _PreparedFactory,
    _PreparedItem,
    _PreparedMetadata,
)
from iris.coords import DimCoord
from iris.cube import Cube


class Test___init__(tests.IrisTest):
    def setUp(self):
        target = "iris.common.resolve.Resolve.__call__"
        self.m_call = mock.MagicMock(return_value=sentinel.return_value)
        _ = self.patch(target, new=self.m_call)

    def _assert_members_none(self, resolve):
        self.assertIsNone(resolve.lhs_cube_resolved)
        self.assertIsNone(resolve.rhs_cube_resolved)
        self.assertIsNone(resolve.lhs_cube_category)
        self.assertIsNone(resolve.rhs_cube_category)
        self.assertIsNone(resolve.lhs_cube_category_local)
        self.assertIsNone(resolve.rhs_cube_category_local)
        self.assertIsNone(resolve.category_common)
        self.assertIsNone(resolve.lhs_cube_dim_coverage)
        self.assertIsNone(resolve.lhs_cube_aux_coverage)
        self.assertIsNone(resolve.rhs_cube_dim_coverage)
        self.assertIsNone(resolve.rhs_cube_aux_coverage)
        self.assertIsNone(resolve.map_rhs_to_lhs)
        self.assertIsNone(resolve.mapping)
        self.assertIsNone(resolve.prepared_category)
        self.assertIsNone(resolve.prepared_factories)
        self.assertIsNone(resolve._broadcast_shape)

    def test_lhs_rhs_default(self):
        resolve = Resolve()
        self.assertIsNone(resolve.lhs_cube)
        self.assertIsNone(resolve.rhs_cube)
        self._assert_members_none(resolve)
        self.assertEqual(0, self.m_call.call_count)

    def test_lhs_rhs_provided(self):
        m_lhs = sentinel.lhs
        m_rhs = sentinel.rhs
        resolve = Resolve(lhs=m_lhs, rhs=m_rhs)
        # The lhs_cube and rhs_cube are only None due
        # to __call__ being mocked. See Test___call__
        # for appropriate test coverage.
        self.assertIsNone(resolve.lhs_cube)
        self.assertIsNone(resolve.rhs_cube)
        self._assert_members_none(resolve)
        self.assertEqual(1, self.m_call.call_count)
        call_args = mock.call(m_lhs, m_rhs)
        self.assertEqual(call_args, self.m_call.call_args)


class Test___call__(tests.IrisTest):
    def setUp(self):
        self.m_lhs = mock.MagicMock(spec=Cube)
        self.m_rhs = mock.MagicMock(spec=Cube)
        target = "iris.common.resolve.Resolve.{method}"
        method = target.format(method="_metadata_resolve")
        self.m_metadata_resolve = self.patch(method)
        method = target.format(method="_metadata_coverage")
        self.m_metadata_coverage = self.patch(method)
        method = target.format(method="_metadata_mapping")
        self.m_metadata_mapping = self.patch(method)
        method = target.format(method="_metadata_prepare")
        self.m_metadata_prepare = self.patch(method)

    def test_lhs_not_cube(self):
        emsg = "'LHS' argument to be a 'Cube'"
        with self.assertRaisesRegex(TypeError, emsg):
            _ = Resolve(rhs=self.m_rhs)

    def test_rhs_not_cube(self):
        emsg = "'RHS' argument to be a 'Cube'"
        with self.assertRaisesRegex(TypeError, emsg):
            _ = Resolve(lhs=self.m_lhs)

    def _assert_called_metadata_methods(self):
        call_args = mock.call()
        self.assertEqual(1, self.m_metadata_resolve.call_count)
        self.assertEqual(call_args, self.m_metadata_resolve.call_args)
        self.assertEqual(1, self.m_metadata_coverage.call_count)
        self.assertEqual(call_args, self.m_metadata_coverage.call_args)
        self.assertEqual(1, self.m_metadata_mapping.call_count)
        self.assertEqual(call_args, self.m_metadata_mapping.call_args)
        self.assertEqual(1, self.m_metadata_prepare.call_count)
        self.assertEqual(call_args, self.m_metadata_prepare.call_args)

    def test_map_rhs_to_lhs__less_than(self):
        self.m_lhs.ndim = 2
        self.m_rhs.ndim = 1
        resolve = Resolve(lhs=self.m_lhs, rhs=self.m_rhs)
        self.assertEqual(self.m_lhs, resolve.lhs_cube)
        self.assertEqual(self.m_rhs, resolve.rhs_cube)
        self.assertTrue(resolve.map_rhs_to_lhs)
        self._assert_called_metadata_methods()

    def test_map_rhs_to_lhs__equal(self):
        self.m_lhs.ndim = 2
        self.m_rhs.ndim = 2
        resolve = Resolve(lhs=self.m_lhs, rhs=self.m_rhs)
        self.assertEqual(self.m_lhs, resolve.lhs_cube)
        self.assertEqual(self.m_rhs, resolve.rhs_cube)
        self.assertTrue(resolve.map_rhs_to_lhs)
        self._assert_called_metadata_methods()

    def test_map_lhs_to_rhs(self):
        self.m_lhs.ndim = 2
        self.m_rhs.ndim = 3
        resolve = Resolve(lhs=self.m_lhs, rhs=self.m_rhs)
        self.assertEqual(self.m_lhs, resolve.lhs_cube)
        self.assertEqual(self.m_rhs, resolve.rhs_cube)
        self.assertFalse(resolve.map_rhs_to_lhs)
        self._assert_called_metadata_methods()


class Test__categorise_items(tests.IrisTest):
    def setUp(self):
        self.coord_dims = {}
        # configure dim coords
        coord = mock.Mock(metadata=sentinel.dim_metadata1)
        self.dim_coords = [coord]
        self.coord_dims[coord] = sentinel.dims1
        # configure aux and scalar coords
        self.aux_coords = []
        pairs = [
            (sentinel.aux_metadata2, sentinel.dims2),
            (sentinel.aux_metadata3, sentinel.dims3),
            (sentinel.scalar_metadata4, None),
            (sentinel.scalar_metadata5, None),
            (sentinel.scalar_metadata6, None),
        ]
        for metadata, dims in pairs:
            coord = mock.Mock(metadata=metadata)
            self.aux_coords.append(coord)
            self.coord_dims[coord] = dims
        func = lambda coord: self.coord_dims[coord]
        self.cube = mock.Mock(
            aux_coords=self.aux_coords,
            dim_coords=self.dim_coords,
            coord_dims=func,
        )

    def test(self):
        result = Resolve._categorise_items(self.cube)
        self.assertIsInstance(result, _CategoryItems)
        self.assertEqual(1, len(result.items_dim))
        # check dim coords
        for item in result.items_dim:
            self.assertIsInstance(item, _Item)
        (coord,) = self.dim_coords
        dims = self.coord_dims[coord]
        expected = [_Item(metadata=coord.metadata, coord=coord, dims=dims)]
        self.assertEqual(expected, result.items_dim)
        # check aux coords
        self.assertEqual(2, len(result.items_aux))
        for item in result.items_aux:
            self.assertIsInstance(item, _Item)
        expected_aux, expected_scalar = [], []
        for coord in self.aux_coords:
            dims = self.coord_dims[coord]
            item = _Item(metadata=coord.metadata, coord=coord, dims=dims)
            if dims:
                expected_aux.append(item)
            else:
                expected_scalar.append(item)
        self.assertEqual(expected_aux, result.items_aux)
        # check scalar coords
        self.assertEqual(3, len(result.items_scalar))
        for item in result.items_scalar:
            self.assertIsInstance(item, _Item)
        self.assertEqual(expected_scalar, result.items_scalar)


class Test__metadata_resolve(tests.IrisTest):
    def setUp(self):
        self.target = "iris.common.resolve.Resolve._categorise_items"
        self.m_lhs_cube = sentinel.lhs_cube
        self.m_rhs_cube = sentinel.rhs_cube

    @staticmethod
    def _create_items(pairs):
        # this wrapper (hack) is necessary in order to support mocking
        # the "name" method (callable) of the metadata, as "name" is already
        # part of the mock API - this is always troublesome in mock-world.
        Wrapper = namedtuple("Wrapper", ("name", "value"))
        result = []
        for name, dims in pairs:
            metadata = Wrapper(name=lambda: str(name), value=name)
            coord = mock.Mock(metadata=metadata)
            item = _Item(metadata=metadata, coord=coord, dims=dims)
            result.append(item)
        return result

    def test_metadata_same(self):
        category = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        # configure dim coords
        pairs = [(sentinel.dim_metadata1, sentinel.dims1)]
        category.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        pairs = [
            (sentinel.aux_metadata1, sentinel.dims2),
            (sentinel.aux_metadata2, sentinel.dims3),
        ]
        category.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        pairs = [
            (sentinel.scalar_metadata1, None),
            (sentinel.scalar_metadata2, None),
            (sentinel.scalar_metadata3, None),
        ]
        category.items_scalar.extend(self._create_items(pairs))

        side_effect = (category, category)
        mocker = self.patch(self.target, side_effect=side_effect)

        resolve = Resolve()
        self.assertIsNone(resolve.lhs_cube)
        self.assertIsNone(resolve.rhs_cube)
        self.assertIsNone(resolve.lhs_cube_category)
        self.assertIsNone(resolve.rhs_cube_category)
        self.assertIsNone(resolve.lhs_cube_category_local)
        self.assertIsNone(resolve.rhs_cube_category_local)
        self.assertIsNone(resolve.category_common)

        # require to explicitly configure cubes
        resolve.lhs_cube = self.m_lhs_cube
        resolve.rhs_cube = self.m_rhs_cube
        resolve._metadata_resolve()

        self.assertEqual(mocker.call_count, 2)
        calls = [mock.call(self.m_lhs_cube), mock.call(self.m_rhs_cube)]
        self.assertEqual(calls, mocker.call_args_list)

        self.assertEqual(category, resolve.lhs_cube_category)
        self.assertEqual(category, resolve.rhs_cube_category)
        expected = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        self.assertEqual(expected, resolve.lhs_cube_category_local)
        self.assertEqual(expected, resolve.rhs_cube_category_local)
        self.assertEqual(category, resolve.category_common)

    def test_metadata_overlap(self):
        # configure the lhs cube category
        category_lhs = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        # configure dim coords
        pairs = [
            (sentinel.dim_metadata1, sentinel.dims1),
            (sentinel.dim_metadata2, sentinel.dims2),
        ]
        category_lhs.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        pairs = [
            (sentinel.aux_metadata1, sentinel.dims3),
            (sentinel.aux_metadata2, sentinel.dims4),
        ]
        category_lhs.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        pairs = [
            (sentinel.scalar_metadata1, None),
            (sentinel.scalar_metadata2, None),
        ]
        category_lhs.items_scalar.extend(self._create_items(pairs))

        # configure the rhs cube category
        category_rhs = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        # configure dim coords
        category_rhs.items_dim.append(category_lhs.items_dim[0])
        pairs = [(sentinel.dim_metadata200, sentinel.dims2)]
        category_rhs.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        category_rhs.items_aux.append(category_lhs.items_aux[0])
        pairs = [(sentinel.aux_metadata200, sentinel.dims4)]
        category_rhs.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        category_rhs.items_scalar.append(category_lhs.items_scalar[0])
        pairs = [(sentinel.scalar_metadata200, None)]
        category_rhs.items_scalar.extend(self._create_items(pairs))

        side_effect = (category_lhs, category_rhs)
        mocker = self.patch(self.target, side_effect=side_effect)

        resolve = Resolve()
        self.assertIsNone(resolve.lhs_cube)
        self.assertIsNone(resolve.rhs_cube)
        self.assertIsNone(resolve.lhs_cube_category)
        self.assertIsNone(resolve.rhs_cube_category)
        self.assertIsNone(resolve.lhs_cube_category_local)
        self.assertIsNone(resolve.rhs_cube_category_local)
        self.assertIsNone(resolve.category_common)

        # require to explicitly configure cubes
        resolve.lhs_cube = self.m_lhs_cube
        resolve.rhs_cube = self.m_rhs_cube
        resolve._metadata_resolve()

        self.assertEqual(2, mocker.call_count)
        calls = [mock.call(self.m_lhs_cube), mock.call(self.m_rhs_cube)]
        self.assertEqual(calls, mocker.call_args_list)

        self.assertEqual(category_lhs, resolve.lhs_cube_category)
        self.assertEqual(category_rhs, resolve.rhs_cube_category)

        items_dim = [category_lhs.items_dim[1]]
        items_aux = [category_lhs.items_aux[1]]
        items_scalar = [category_lhs.items_scalar[1]]
        expected = _CategoryItems(
            items_dim=items_dim, items_aux=items_aux, items_scalar=items_scalar
        )
        self.assertEqual(expected, resolve.lhs_cube_category_local)

        items_dim = [category_rhs.items_dim[1]]
        items_aux = [category_rhs.items_aux[1]]
        items_scalar = [category_rhs.items_scalar[1]]
        expected = _CategoryItems(
            items_dim=items_dim, items_aux=items_aux, items_scalar=items_scalar
        )
        self.assertEqual(expected, resolve.rhs_cube_category_local)

        items_dim = [category_lhs.items_dim[0]]
        items_aux = [category_lhs.items_aux[0]]
        items_scalar = [category_lhs.items_scalar[0]]
        expected = _CategoryItems(
            items_dim=items_dim, items_aux=items_aux, items_scalar=items_scalar
        )
        self.assertEqual(expected, resolve.category_common)

    def test_metadata_different(self):
        # configure the lhs cube category
        category_lhs = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        # configure dim coords
        pairs = [
            (sentinel.dim_metadata1, sentinel.dims1),
            (sentinel.dim_metadata2, sentinel.dims2),
        ]
        category_lhs.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        pairs = [
            (sentinel.aux_metadata1, sentinel.dims3),
            (sentinel.aux_metadata2, sentinel.dims4),
        ]
        category_lhs.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        pairs = [
            (sentinel.scalar_metadata1, None),
            (sentinel.scalar_metadata2, None),
        ]
        category_lhs.items_scalar.extend(self._create_items(pairs))

        # configure the rhs cube category
        category_rhs = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        # configure dim coords
        pairs = [
            (sentinel.dim_metadata100, sentinel.dims1),
            (sentinel.dim_metadata200, sentinel.dims2),
        ]
        category_rhs.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        pairs = [
            (sentinel.aux_metadata100, sentinel.dims3),
            (sentinel.aux_metadata200, sentinel.dims4),
        ]
        category_rhs.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        pairs = [
            (sentinel.scalar_metadata100, None),
            (sentinel.scalar_metadata200, None),
        ]
        category_rhs.items_scalar.extend(self._create_items(pairs))

        side_effect = (category_lhs, category_rhs)
        mocker = self.patch(self.target, side_effect=side_effect)

        resolve = Resolve()
        self.assertIsNone(resolve.lhs_cube)
        self.assertIsNone(resolve.rhs_cube)
        self.assertIsNone(resolve.lhs_cube_category)
        self.assertIsNone(resolve.rhs_cube_category)
        self.assertIsNone(resolve.lhs_cube_category_local)
        self.assertIsNone(resolve.rhs_cube_category_local)
        self.assertIsNone(resolve.category_common)

        # first require to explicitly lhs/rhs configure cubes
        resolve.lhs_cube = self.m_lhs_cube
        resolve.rhs_cube = self.m_rhs_cube
        resolve._metadata_resolve()

        self.assertEqual(2, mocker.call_count)
        calls = [mock.call(self.m_lhs_cube), mock.call(self.m_rhs_cube)]
        self.assertEqual(calls, mocker.call_args_list)

        self.assertEqual(category_lhs, resolve.lhs_cube_category)
        self.assertEqual(category_rhs, resolve.rhs_cube_category)
        self.assertEqual(category_lhs, resolve.lhs_cube_category_local)
        self.assertEqual(category_rhs, resolve.rhs_cube_category_local)
        expected = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        self.assertEqual(expected, resolve.category_common)


class Test__dim_coverage(tests.IrisTest):
    def setUp(self):
        self.ndim = 4
        self.cube = mock.Mock(ndim=self.ndim)
        self.items = []
        parts = [
            (sentinel.metadata0, sentinel.coord0, (0,)),
            (sentinel.metadata1, sentinel.coord1, (1,)),
            (sentinel.metadata2, sentinel.coord2, (2,)),
            (sentinel.metadata3, sentinel.coord3, (3,)),
        ]
        column_parts = [x for x in zip(*parts)]
        self.metadata, self.coords, self.dims = [list(x) for x in column_parts]
        self.dims = [dim for dim, in self.dims]
        for metadata, coord, dims in parts:
            item = _Item(metadata=metadata, coord=coord, dims=dims)
            self.items.append(item)

    def test_coverage_no_local_no_common_all_free(self):
        items = []
        common = []
        result = Resolve._dim_coverage(self.cube, items, common)
        self.assertIsInstance(result, _DimCoverage)
        self.assertEqual(self.cube, result.cube)
        expected = [None] * self.ndim
        self.assertEqual(expected, result.metadata)
        self.assertEqual(expected, result.coords)
        self.assertEqual([], result.dims_common)
        self.assertEqual([], result.dims_local)
        expected = list(range(self.ndim))
        self.assertEqual(expected, result.dims_free)

    def test_coverage_all_local_no_common_no_free(self):
        common = []
        result = Resolve._dim_coverage(self.cube, self.items, common)
        self.assertIsInstance(result, _DimCoverage)
        self.assertEqual(self.cube, result.cube)
        self.assertEqual(self.metadata, result.metadata)
        self.assertEqual(self.coords, result.coords)
        self.assertEqual([], result.dims_common)
        self.assertEqual(self.dims, result.dims_local)
        self.assertEqual([], result.dims_free)

    def test_coverage_no_local_all_common_no_free(self):
        result = Resolve._dim_coverage(self.cube, self.items, self.metadata)
        self.assertIsInstance(result, _DimCoverage)
        self.assertEqual(self.cube, result.cube)
        self.assertEqual(self.metadata, result.metadata)
        self.assertEqual(self.coords, result.coords)
        self.assertEqual(self.dims, result.dims_common)
        self.assertEqual([], result.dims_local)
        self.assertEqual([], result.dims_free)

    def test_coverage_mixed(self):
        common = [self.items[1].metadata, self.items[2].metadata]
        self.items.pop(0)
        self.items.pop(-1)
        metadata, coord, dims = sentinel.metadata100, sentinel.coord100, (0,)
        self.items.append(_Item(metadata=metadata, coord=coord, dims=dims))
        result = Resolve._dim_coverage(self.cube, self.items, common)
        self.assertIsInstance(result, _DimCoverage)
        self.assertEqual(self.cube, result.cube)
        expected = [
            metadata,
            self.items[0].metadata,
            self.items[1].metadata,
            None,
        ]
        self.assertEqual(expected, result.metadata)
        expected = [coord, self.items[0].coord, self.items[1].coord, None]
        self.assertEqual(expected, result.coords)
        self.assertEqual([1, 2], result.dims_common)
        self.assertEqual([0], result.dims_local)
        self.assertEqual([3], result.dims_free)


class Test__aux_coverage(tests.IrisTest):
    def setUp(self):
        self.ndim = 4
        self.cube = mock.Mock(ndim=self.ndim)
        # configure aux coords
        self.items_aux = []
        aux_parts = [
            (sentinel.aux_metadata0, sentinel.aux_coord0, (0,)),
            (sentinel.aux_metadata1, sentinel.aux_coord1, (1,)),
            (sentinel.aux_metadata23, sentinel.aux_coord23, (2, 3)),
        ]
        column_aux_parts = [x for x in zip(*aux_parts)]
        self.aux_metadata, self.aux_coords, self.aux_dims = [
            list(x) for x in column_aux_parts
        ]
        for metadata, coord, dims in aux_parts:
            item = _Item(metadata=metadata, coord=coord, dims=dims)
            self.items_aux.append(item)
        # configure scalar coords
        self.items_scalar = []
        scalar_parts = [
            (sentinel.scalar_metadata0, sentinel.scalar_coord0, ()),
            (sentinel.scalar_metadata1, sentinel.scalar_coord1, ()),
            (sentinel.scalar_metadata2, sentinel.scalar_coord2, ()),
        ]
        column_scalar_parts = [x for x in zip(*scalar_parts)]
        self.scalar_metadata, self.scalar_coords, self.scalar_dims = [
            list(x) for x in column_scalar_parts
        ]
        for metadata, coord, dims in scalar_parts:
            item = _Item(metadata=metadata, coord=coord, dims=dims)
            self.items_scalar.append(item)

    def test_coverage_no_local_no_common_all_free(self):
        items_aux, items_scalar = [], []
        common_aux, common_scalar = [], []
        result = Resolve._aux_coverage(
            self.cube, items_aux, items_scalar, common_aux, common_scalar
        )
        self.assertIsInstance(result, _AuxCoverage)
        self.assertEqual(self.cube, result.cube)
        self.assertEqual([], result.common_items_aux)
        self.assertEqual([], result.common_items_scalar)
        self.assertEqual([], result.local_items_aux)
        self.assertEqual([], result.local_items_scalar)
        self.assertEqual([], result.dims_common)
        self.assertEqual([], result.dims_local)
        expected = list(range(self.ndim))
        self.assertEqual(expected, result.dims_free)

    def test_coverage_all_local_no_common_no_free(self):
        common_aux, common_scalar = [], []
        result = Resolve._aux_coverage(
            self.cube,
            self.items_aux,
            self.items_scalar,
            common_aux,
            common_scalar,
        )
        self.assertIsInstance(result, _AuxCoverage)
        self.assertEqual(self.cube, result.cube)
        expected = []
        self.assertEqual(expected, result.common_items_aux)
        self.assertEqual(expected, result.common_items_scalar)
        self.assertEqual(self.items_aux, result.local_items_aux)
        self.assertEqual(self.items_scalar, result.local_items_scalar)
        self.assertEqual([], result.dims_common)
        expected = list(range(self.ndim))
        self.assertEqual(expected, result.dims_local)
        self.assertEqual([], result.dims_free)

    def test_coverage_no_local_all_common_no_free(self):
        result = Resolve._aux_coverage(
            self.cube,
            self.items_aux,
            self.items_scalar,
            self.aux_metadata,
            self.scalar_metadata,
        )
        self.assertIsInstance(result, _AuxCoverage)
        self.assertEqual(self.cube, result.cube)
        self.assertEqual(self.items_aux, result.common_items_aux)
        self.assertEqual(self.items_scalar, result.common_items_scalar)
        self.assertEqual([], result.local_items_aux)
        self.assertEqual([], result.local_items_scalar)
        expected = list(range(self.ndim))
        self.assertEqual(expected, result.dims_common)
        self.assertEqual([], result.dims_local)
        self.assertEqual([], result.dims_free)

    def test_coverage_mixed(self):
        common_aux = [self.items_aux[-1].metadata]
        common_scalar = [self.items_scalar[1].metadata]
        self.items_aux.pop(1)
        result = Resolve._aux_coverage(
            self.cube,
            self.items_aux,
            self.items_scalar,
            common_aux,
            common_scalar,
        )
        self.assertIsInstance(result, _AuxCoverage)
        self.assertEqual(self.cube, result.cube)
        expected = [self.items_aux[-1]]
        self.assertEqual(expected, result.common_items_aux)
        expected = [self.items_scalar[1]]
        self.assertEqual(expected, result.common_items_scalar)
        expected = [self.items_aux[0]]
        self.assertEqual(expected, result.local_items_aux)
        expected = [self.items_scalar[0], self.items_scalar[2]]
        self.assertEqual(expected, result.local_items_scalar)
        self.assertEqual([2, 3], result.dims_common)
        self.assertEqual([0], result.dims_local)
        self.assertEqual([1], result.dims_free)


class Test__metadata_coverage(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()
        self.m_lhs_cube = sentinel.lhs_cube
        self.resolve.lhs_cube = self.m_lhs_cube
        self.m_rhs_cube = sentinel.rhs_cube
        self.resolve.rhs_cube = self.m_rhs_cube
        self.m_items_dim_metadata = sentinel.items_dim_metadata
        self.m_items_aux_metadata = sentinel.items_aux_metadata
        self.m_items_scalar_metadata = sentinel.items_scalar_metadata
        items_dim = [mock.Mock(metadata=self.m_items_dim_metadata)]
        items_aux = [mock.Mock(metadata=self.m_items_aux_metadata)]
        items_scalar = [mock.Mock(metadata=self.m_items_scalar_metadata)]
        category = _CategoryItems(
            items_dim=items_dim, items_aux=items_aux, items_scalar=items_scalar
        )
        self.resolve.category_common = category
        self.m_items_dim = sentinel.items_dim
        self.m_items_aux = sentinel.items_aux
        self.m_items_scalar = sentinel.items_scalar
        category = _CategoryItems(
            items_dim=self.m_items_dim,
            items_aux=self.m_items_aux,
            items_scalar=self.m_items_scalar,
        )
        self.resolve.lhs_cube_category = category
        self.resolve.rhs_cube_category = category
        target = "iris.common.resolve.Resolve._dim_coverage"
        self.m_lhs_cube_dim_coverage = sentinel.lhs_cube_dim_coverage
        self.m_rhs_cube_dim_coverage = sentinel.rhs_cube_dim_coverage
        side_effect = (
            self.m_lhs_cube_dim_coverage,
            self.m_rhs_cube_dim_coverage,
        )
        self.mocker_dim_coverage = self.patch(target, side_effect=side_effect)
        target = "iris.common.resolve.Resolve._aux_coverage"
        self.m_lhs_cube_aux_coverage = sentinel.lhs_cube_aux_coverage
        self.m_rhs_cube_aux_coverage = sentinel.rhs_cube_aux_coverage
        side_effect = (
            self.m_lhs_cube_aux_coverage,
            self.m_rhs_cube_aux_coverage,
        )
        self.mocker_aux_coverage = self.patch(target, side_effect=side_effect)

    def test(self):
        self.resolve._metadata_coverage()
        self.assertEqual(2, self.mocker_dim_coverage.call_count)
        calls = [
            mock.call(
                self.m_lhs_cube, self.m_items_dim, [self.m_items_dim_metadata]
            ),
            mock.call(
                self.m_rhs_cube, self.m_items_dim, [self.m_items_dim_metadata]
            ),
        ]
        self.assertEqual(calls, self.mocker_dim_coverage.call_args_list)
        self.assertEqual(2, self.mocker_aux_coverage.call_count)
        calls = [
            mock.call(
                self.m_lhs_cube,
                self.m_items_aux,
                self.m_items_scalar,
                [self.m_items_aux_metadata],
                [self.m_items_scalar_metadata],
            ),
            mock.call(
                self.m_rhs_cube,
                self.m_items_aux,
                self.m_items_scalar,
                [self.m_items_aux_metadata],
                [self.m_items_scalar_metadata],
            ),
        ]
        self.assertEqual(calls, self.mocker_aux_coverage.call_args_list)
        self.assertEqual(
            self.m_lhs_cube_dim_coverage, self.resolve.lhs_cube_dim_coverage
        )
        self.assertEqual(
            self.m_rhs_cube_dim_coverage, self.resolve.rhs_cube_dim_coverage
        )
        self.assertEqual(
            self.m_lhs_cube_aux_coverage, self.resolve.lhs_cube_aux_coverage
        )
        self.assertEqual(
            self.m_rhs_cube_aux_coverage, self.resolve.rhs_cube_aux_coverage
        )


class Test__dim_mapping(tests.IrisTest):
    def setUp(self):
        self.ndim = 3
        Wrapper = namedtuple("Wrapper", ("name",))
        cube = Wrapper(name=lambda: sentinel.name)
        self.src_coverage = _DimCoverage(
            cube=cube,
            metadata=[],
            coords=None,
            dims_common=None,
            dims_local=None,
            dims_free=None,
        )
        self.tgt_coverage = _DimCoverage(
            cube=cube,
            metadata=[],
            coords=None,
            dims_common=[],
            dims_local=None,
            dims_free=None,
        )
        self.metadata = [
            sentinel.metadata_0,
            sentinel.metadata_1,
            sentinel.metadata_2,
        ]
        self.dummy = [sentinel.dummy_0, sentinel.dummy_1, sentinel.dummy_2]

    def test_no_mapping(self):
        self.src_coverage.metadata.extend(self.metadata)
        self.tgt_coverage.metadata.extend(self.dummy)
        result = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)
        self.assertEqual(dict(), result)

    def test_full_mapping(self):
        self.src_coverage.metadata.extend(self.metadata)
        self.tgt_coverage.metadata.extend(self.metadata)
        dims_common = list(range(self.ndim))
        self.tgt_coverage.dims_common.extend(dims_common)
        result = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 0, 1: 1, 2: 2}
        self.assertEqual(expected, result)

    def test_transpose_mapping(self):
        self.src_coverage.metadata.extend(self.metadata[::-1])
        self.tgt_coverage.metadata.extend(self.metadata)
        dims_common = list(range(self.ndim))
        self.tgt_coverage.dims_common.extend(dims_common)
        result = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 2, 1: 1, 2: 0}
        self.assertEqual(expected, result)

    def test_partial_mapping__transposed(self):
        self.src_coverage.metadata.extend(self.metadata)
        self.metadata[1] = sentinel.nope
        self.tgt_coverage.metadata.extend(self.metadata[::-1])
        dims_common = [0, 2]
        self.tgt_coverage.dims_common.extend(dims_common)
        result = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 2, 2: 0}
        self.assertEqual(expected, result)

    def test_bad_metadata_mapping(self):
        self.src_coverage.metadata.extend(self.metadata)
        self.metadata[0] = sentinel.bad
        self.tgt_coverage.metadata.extend(self.metadata)
        dims_common = [0]
        self.tgt_coverage.dims_common.extend(dims_common)
        emsg = "Failed to map common dim coordinate metadata"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)


class Test__aux_mapping(tests.IrisTest):
    def setUp(self):
        self.ndim = 3
        Wrapper = namedtuple("Wrapper", ("name",))
        cube = Wrapper(name=lambda: sentinel.name)
        self.src_coverage = _AuxCoverage(
            cube=cube,
            common_items_aux=[],
            common_items_scalar=None,
            local_items_aux=None,
            local_items_scalar=None,
            dims_common=None,
            dims_local=None,
            dims_free=None,
        )
        self.tgt_coverage = _AuxCoverage(
            cube=cube,
            common_items_aux=[],
            common_items_scalar=None,
            local_items_aux=None,
            local_items_scalar=None,
            dims_common=None,
            dims_local=None,
            dims_free=None,
        )
        self.items = [
            _Item(
                metadata=sentinel.metadata0, coord=sentinel.coord0, dims=[0]
            ),
            _Item(
                metadata=sentinel.metadata1, coord=sentinel.coord1, dims=[1]
            ),
            _Item(
                metadata=sentinel.metadata2, coord=sentinel.coord2, dims=[2]
            ),
        ]

    def _copy(self, items):
        # Due to a bug in python 3.6.x, performing a deepcopy of a mock.sentinel
        # will yield an object that is not equivalent to its parent, so this
        # is a work-around until we drop support for python 3.6.x.
        import sys

        version = sys.version_info
        major, minor = version.major, version.minor
        result = deepcopy(items)
        if major == 3 and minor <= 6:
            for i, item in enumerate(items):
                result[i] = result[i]._replace(metadata=item.metadata)
        return result

    def test_no_mapping(self):
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        self.assertEqual(dict(), result)

    def test_full_mapping(self):
        self.src_coverage.common_items_aux.extend(self.items)
        self.tgt_coverage.common_items_aux.extend(self.items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 0, 1: 1, 2: 2}
        self.assertEqual(expected, result)

    def test_transpose_mapping(self):
        self.src_coverage.common_items_aux.extend(self.items)
        items = self._copy(self.items)
        items[0].dims[0] = 2
        items[2].dims[0] = 0
        self.tgt_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 2, 1: 1, 2: 0}
        self.assertEqual(expected, result)

    def test_partial_mapping__transposed(self):
        _ = self.items.pop(1)
        self.src_coverage.common_items_aux.extend(self.items)
        items = self._copy(self.items)
        items[0].dims[0] = 2
        items[1].dims[0] = 0
        self.tgt_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 2, 2: 0}
        self.assertEqual(expected, result)

    def test_mapping__match_multiple_src_metadata(self):
        items = self._copy(self.items)
        _ = self.items.pop(1)
        self.src_coverage.common_items_aux.extend(self.items)
        items[1] = items[0]
        self.tgt_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 0, 2: 2}
        self.assertEqual(expected, result)

    def test_mapping__skip_match_multiple_src_metadata(self):
        items = self._copy(self.items)
        _ = self.items.pop(1)
        self.tgt_coverage.common_items_aux.extend(self.items)
        items[1] = items[0]._replace(dims=[1])
        self.src_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {2: 2}
        self.assertEqual(expected, result)

    def test_mapping__skip_different_rank(self):
        items = self._copy(self.items)
        self.src_coverage.common_items_aux.extend(self.items)
        items[2] = items[2]._replace(dims=[1, 2])
        self.tgt_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 0, 1: 1}
        self.assertEqual(expected, result)

    def test_bad_metadata_mapping(self):
        self.src_coverage.common_items_aux.extend(self.items)
        items = self._copy(self.items)
        items[0] = items[0]._replace(metadata=sentinel.bad)
        self.tgt_coverage.common_items_aux.extend(items)
        emsg = "Failed to map common aux coordinate metadata"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)


class Test_mapped(tests.IrisTest):
    def test_mapping_none(self):
        resolve = Resolve()
        self.assertIsNone(resolve.mapping)
        self.assertIsNone(resolve.mapped)

    def test_mapped__src_cube_lhs(self):
        resolve = Resolve()
        lhs = mock.Mock(ndim=2)
        rhs = mock.Mock(ndim=3)
        resolve.lhs_cube = lhs
        resolve.rhs_cube = rhs
        resolve.map_rhs_to_lhs = False
        resolve.mapping = {0: 0, 1: 1}
        self.assertTrue(resolve.mapped)

    def test_mapped__src_cube_rhs(self):
        resolve = Resolve()
        lhs = mock.Mock(ndim=3)
        rhs = mock.Mock(ndim=2)
        resolve.lhs_cube = lhs
        resolve.rhs_cube = rhs
        resolve.map_rhs_to_lhs = True
        resolve.mapping = {0: 0, 1: 1}
        self.assertTrue(resolve.mapped)

    def test_partial_mapping(self):
        resolve = Resolve()
        lhs = mock.Mock(ndim=3)
        rhs = mock.Mock(ndim=2)
        resolve.lhs_cube = lhs
        resolve.rhs_cube = rhs
        resolve.map_rhs_to_lhs = True
        resolve.mapping = {0: 0}
        self.assertFalse(resolve.mapped)


class Test__free_mapping(tests.IrisTest):
    def setUp(self):
        self.Cube = namedtuple("Wrapper", ("name", "ndim", "shape"))
        self.src_dim_coverage = dict(
            cube=None,
            metadata=None,
            coords=None,
            dims_common=None,
            dims_local=None,
            dims_free=[],
        )
        self.tgt_dim_coverage = deepcopy(self.src_dim_coverage)
        self.src_aux_coverage = dict(
            cube=None,
            common_items_aux=None,
            common_items_scalar=None,
            local_items_aux=None,
            local_items_scalar=None,
            dims_common=None,
            dims_local=None,
            dims_free=[],
        )
        self.tgt_aux_coverage = deepcopy(self.src_aux_coverage)
        self.resolve = Resolve()
        self.resolve.map_rhs_to_lhs = True
        self.resolve.mapping = {}

    def _make_args(self):
        args = dict(
            src_dim_coverage=_DimCoverage(**self.src_dim_coverage),
            tgt_dim_coverage=_DimCoverage(**self.tgt_dim_coverage),
            src_aux_coverage=_AuxCoverage(**self.src_aux_coverage),
            tgt_aux_coverage=_AuxCoverage(**self.tgt_aux_coverage),
        )
        return args

    def test_mapping_no_dims_free(self):
        ndim = 4
        shape = tuple(range(ndim))
        cube = self.Cube(name=lambda: "name", ndim=ndim, shape=shape)
        self.src_dim_coverage["cube"] = cube
        self.tgt_dim_coverage["cube"] = cube
        args = self._make_args()
        emsg = "Insufficient matching coordinate metadata"
        with self.assertRaisesRegex(ValueError, emsg):
            self.resolve._free_mapping(**args)

    def _make_coverage(self, name, shape, dims_free):
        if name == "src":
            dim_coverage = self.src_dim_coverage
            aux_coverage = self.src_aux_coverage
        else:
            dim_coverage = self.tgt_dim_coverage
            aux_coverage = self.tgt_aux_coverage
        ndim = len(shape)
        cube = self.Cube(name=lambda: name, ndim=ndim, shape=shape)
        dim_coverage["cube"] = cube
        dim_coverage["dims_free"].extend(dims_free)
        aux_coverage["cube"] = cube
        aux_coverage["dims_free"].extend(dims_free)

    def test_mapping_src_free_to_tgt_local(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 2 3 4
        #   state f l c l      state f c f
        #   coord d d d a      coord a d d
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->3 1->2 2->1
        src_shape = (2, 3, 4)
        src_free = [0, 2]
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [0]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 3, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_src_free_to_tgt_local__broadcast_src_first(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 1 3 4
        #   state f l c l      state f c f
        #   coord d d d a      coord a d d
        #                      bcast ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->3 1->2 2->1
        src_shape = (1, 3, 4)
        src_free = [0, 2]
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [0]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 3, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_src_free_to_tgt_local__broadcast_src_last(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 2 3 1
        #   state f l c l      state f c f
        #   coord d d d a      coord a d d
        #                      bcast     ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->3 1->2 2->1
        src_shape = (2, 3, 1)
        src_free = [0, 2]
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [0]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 3, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_src_free_to_tgt_local__broadcast_src_both(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 1 3 1
        #   state f l c l      state f c f
        #   coord d d d a      coord a d d
        #                      bcast ^   ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->1 1->2 2->3
        src_shape = (1, 3, 1)
        src_free = [0, 2]
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [0]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 1, 1: 2, 2: 3}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_src_free_to_tgt_free(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 2 3 4
        #   state f f c f      state f c f
        #   coord d d d a      coord a d d
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->0 1->2 2->1
        src_shape = (2, 3, 4)
        src_free = [0, 2]
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [0, 1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 0, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_src_free_to_tgt_free__broadcast_src_first(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 1 3 4
        #   state f f c f      state f c f
        #   coord d d d a      coord a d d
        #                      bcast ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->0 1->2 2->1
        src_shape = (1, 3, 4)
        src_free = [0, 2]
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [0, 1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 0, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_src_free_to_tgt_free__broadcast_src_last(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 2 3 1
        #   state f f c f      state f c f
        #   coord d d d a      coord a d d
        #                      bcast     ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->0 1->2 2->1
        src_shape = (2, 3, 1)
        src_free = [0, 2]
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [0, 1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 0, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_src_free_to_tgt_free__broadcast_src_both(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 1 3 1
        #   state f f c f      state f c f
        #   coord d d d a      coord a d d
        #                      bcast ^   ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->0 1->2 2->1
        src_shape = (1, 3, 1)
        src_free = [0, 2]
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [0, 1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 0, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_src_free_to_tgt__fail(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 2 3 5
        #   state f f c f      state f c f
        #   coord d d d a      coord a d d
        #                      fail      ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->0 1->2 2->?
        src_shape = (2, 3, 5)
        src_free = [0, 2]
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [0, 1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        emsg = "Insufficient matching coordinate metadata to resolve cubes"
        with self.assertRaisesRegex(ValueError, emsg):
            self.resolve._free_mapping(**args)

    def test_mapping_tgt_free_to_src_local(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            -> src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 2 3 4
        #   state l f c f      state l c l
        #   coord d d d a      coord a d d
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->3 1->2 2->1
        src_shape = (2, 3, 4)
        src_free = []
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 2)
        tgt_free = [1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 3, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_tgt_free_to_src_local__broadcast_tgt_first(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            -> src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 1 3 2      shape 2 3 4
        #   state l f c f      state l c l
        #   coord d d d a      coord a d d
        #   bcast   ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->3 1->2 2->1
        src_shape = (2, 3, 4)
        src_free = []
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 1, 3, 2)
        tgt_free = [1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 3, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_tgt_free_to_src_local__broadcast_tgt_last(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            -> src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 1      shape 2 3 4
        #   state l f c f      state l c l
        #   coord d d d a      coord a d d
        #   bcast       ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->3 1->2 2->1
        src_shape = (2, 3, 4)
        src_free = []
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 1)
        tgt_free = [1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 3, 1: 2, 2: 1}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_tgt_free_to_src_local__broadcast_tgt_both(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            -> src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 1 3 1      shape 2 3 4
        #   state l f c f      state l c l
        #   coord d d d a      coord a d d
        #   bcast   ^   ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->1 1->2 2->3
        src_shape = (2, 3, 4)
        src_free = []
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 1, 3, 1)
        tgt_free = [1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        self.resolve._free_mapping(**args)
        expected = {0: 1, 1: 2, 2: 3}
        self.assertEqual(expected, self.resolve.mapping)

    def test_mapping_tgt_free_to_src_no_free__fail(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            -> src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 5      shape 2 3 4
        #   state l f c f      state l c l
        #   coord d d d a      coord a d d
        #   fail        ^
        #
        # src-to-tgt mapping:
        #   before      1->2
        #   after  0->0 1->2 2->?
        src_shape = (2, 3, 4)
        src_free = []
        self._make_coverage("src", src_shape, src_free)
        tgt_shape = (2, 4, 3, 5)
        tgt_free = [1, 3]
        self._make_coverage("tgt", tgt_shape, tgt_free)
        self.resolve.mapping = {1: 2}
        args = self._make_args()
        emsg = "Insufficient matching coordinate metadata to resolve cubes"
        with self.assertRaisesRegex(ValueError, emsg):
            self.resolve._free_mapping(**args)


class Test__src_cube(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()
        self.expected = sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve.rhs_cube = self.expected
        self.assertEqual(self.expected, self.resolve._src_cube)

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve.lhs_cube = self.expected
        self.assertEqual(self.expected, self.resolve._src_cube)

    def test_fail__no_map_rhs_to_lhs(self):
        with self.assertRaises(AssertionError):
            self.resolve._src_cube


class Test__src_cube_position(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.assertEqual("RHS", self.resolve._src_cube_position)

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.assertEqual("LHS", self.resolve._src_cube_position)

    def test_fail__no_map_rhs_to_lhs(self):
        with self.assertRaises(AssertionError):
            self.resolve._src_cube_position


class Test__src_cube_resolved__getter(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()
        self.expected = sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve.rhs_cube_resolved = self.expected
        self.assertEqual(self.expected, self.resolve._src_cube_resolved)

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve.lhs_cube_resolved = self.expected
        self.assertEqual(self.expected, self.resolve._src_cube_resolved)

    def test_fail__no_map_rhs_to_lhs(self):
        with self.assertRaises(AssertionError):
            self.resolve._src_cube_resolved


class Test__src_cube_resolved__setter(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()
        self.expected = sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve._src_cube_resolved = self.expected
        self.assertEqual(self.expected, self.resolve.rhs_cube_resolved)

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve._src_cube_resolved = self.expected
        self.assertEqual(self.expected, self.resolve.lhs_cube_resolved)

    def test_fail__no_map_rhs_to_lhs(self):
        with self.assertRaises(AssertionError):
            self.resolve._src_cube_resolved = self.expected


class Test__tgt_cube(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()
        self.expected = sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve.rhs_cube = self.expected
        self.assertEqual(self.expected, self.resolve._tgt_cube)

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve.lhs_cube = self.expected
        self.assertEqual(self.expected, self.resolve._tgt_cube)

    def test_fail__no_map_rhs_to_lhs(self):
        with self.assertRaises(AssertionError):
            self.resolve._tgt_cube


class Test__tgt_cube_position(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.assertEqual("RHS", self.resolve._tgt_cube_position)

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.assertEqual("LHS", self.resolve._tgt_cube_position)

    def test_fail__no_map_rhs_to_lhs(self):
        with self.assertRaises(AssertionError):
            self.resolve._tgt_cube_position


class Test__tgt_cube_resolved__getter(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()
        self.expected = sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve.rhs_cube_resolved = self.expected
        self.assertEqual(self.expected, self.resolve._tgt_cube_resolved)

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve.lhs_cube_resolved = self.expected
        self.assertEqual(self.expected, self.resolve._tgt_cube_resolved)

    def test_fail__no_map_rhs_to_lhs(self):
        with self.assertRaises(AssertionError):
            self.resolve._tgt_cube_resolved


class Test__tgt_cube_resolved__setter(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()
        self.expected = sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve._tgt_cube_resolved = self.expected
        self.assertEqual(self.expected, self.resolve.rhs_cube_resolved)

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve._tgt_cube_resolved = self.expected
        self.assertEqual(self.expected, self.resolve.lhs_cube_resolved)

    def test_fail__no_map_rhs_to_lhs(self):
        with self.assertRaises(AssertionError):
            self.resolve._tgt_cube_resolved = self.expected


class Test_shape(tests.IrisTest):
    def setUp(self):
        self.resolve = Resolve()

    def test_no_shape(self):
        self.assertIsNone(self.resolve.shape)

    def test_shape(self):
        expected = sentinel.shape
        self.resolve._broadcast_shape = expected
        self.assertEqual(expected, self.resolve.shape)


class Test__as_compatible_cubes(tests.IrisTest):
    def setUp(self):
        self.Cube = namedtuple(
            "Wrapper",
            (
                "name",
                "ndim",
                "shape",
                "metadata",
                "core_data",
                "coord_dims",
                "dim_coords",
                "aux_coords",
                "aux_factories",
            ),
        )
        self.resolve = Resolve()
        self.resolve.map_rhs_to_lhs = True
        self.resolve.mapping = {}
        self.mocker = self.patch("iris.cube.Cube")
        self.args = dict(
            name=None,
            ndim=None,
            shape=None,
            metadata=None,
            core_data=None,
            coord_dims=None,
            dim_coords=None,
            aux_coords=None,
            aux_factories=None,
        )

    def _make_cube(self, name, shape, transpose_shape=None):
        self.args["name"] = lambda: name
        ndim = len(shape)
        self.args["ndim"] = ndim
        self.args["shape"] = shape
        if name == "src":
            self.args["metadata"] = sentinel.metadata
            self.reshape = sentinel.reshape
            m_reshape = mock.Mock(return_value=self.reshape)
            self.transpose = mock.Mock(
                shape=transpose_shape, reshape=m_reshape
            )
            m_transpose = mock.Mock(return_value=self.transpose)
            self.data = mock.Mock(
                shape=shape, transpose=m_transpose, reshape=m_reshape
            )
            m_copy = mock.Mock(return_value=self.data)
            m_core_data = mock.Mock(copy=m_copy)
            self.args["core_data"] = mock.Mock(return_value=m_core_data)
            self.args["coord_dims"] = mock.Mock(side_effect=([0], [ndim - 1]))
            self.dim_coord = sentinel.dim_coord
            self.aux_coord = sentinel.aux_coord
            self.aux_factory = sentinel.aux_factory
            self.args["dim_coords"] = [self.dim_coord]
            self.args["aux_coords"] = [self.aux_coord]
            self.args["aux_factories"] = [self.aux_factory]
            cube = self.Cube(**self.args)
            self.resolve.rhs_cube = cube
            self.cube = mock.Mock()
            self.mocker.return_value = self.cube
        else:
            cube = self.Cube(**self.args)
            self.resolve.lhs_cube = cube

    def test_incomplete_src_to_tgt_mapping__fail(self):
        src_shape = (1, 2)
        self._make_cube("src", src_shape)
        tgt_shape = (3, 4)
        self._make_cube("tgt", tgt_shape)
        with self.assertRaises(AssertionError):
            self.resolve._as_compatible_cubes()

    def test_incompatible_shapes__fail(self):
        # key: (state) c=common, f=free
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 2 3 4      shape 2 3 5
        #   state f c c c      state c c c
        #   fail        ^      fail      ^
        #
        # src-to-tgt mapping:
        #   0->1, 1->2, 2->3
        src_shape = (2, 3, 5)
        self._make_cube("src", src_shape)
        tgt_shape = (2, 2, 3, 4)
        self._make_cube("tgt", tgt_shape)
        self.resolve.mapping = {0: 1, 1: 2, 2: 3}
        emsg = "Cannot resolve cubes"
        with self.assertRaisesRegex(ValueError, emsg):
            self.resolve._as_compatible_cubes()

    def test_incompatible_shapes__fail_broadcast(self):
        # key: (state) c=common, f=free
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 2 4 3 2      shape 2 3 5
        #   state f c c c      state c c c
        #   fail    ^          fail      ^
        #
        # src-to-tgt mapping:
        #   0->3, 1->2, 2->1
        src_shape = (2, 3, 5)
        self._make_cube("src", src_shape)
        tgt_shape = (2, 4, 3, 2)
        self._make_cube("tgt", tgt_shape)
        self.resolve.mapping = {0: 3, 1: 2, 2: 1}
        emsg = "Cannot resolve cubes"
        with self.assertRaisesRegex(ValueError, emsg):
            self.resolve._as_compatible_cubes()

    def _check_compatible(self, broadcast_shape):
        self.assertEqual(
            self.resolve.lhs_cube, self.resolve._tgt_cube_resolved
        )
        self.assertEqual(self.cube, self.resolve._src_cube_resolved)
        self.assertEqual(broadcast_shape, self.resolve._broadcast_shape)
        self.assertEqual(1, self.mocker.call_count)
        self.assertEqual(self.args["metadata"], self.cube.metadata)
        self.assertEqual(2, self.resolve.rhs_cube.coord_dims.call_count)
        self.assertEqual(
            [mock.call(self.dim_coord), mock.call(self.aux_coord)],
            self.resolve.rhs_cube.coord_dims.call_args_list,
        )
        self.assertEqual(1, self.cube.add_dim_coord.call_count)
        self.assertEqual(
            [mock.call(self.dim_coord, [self.resolve.mapping[0]])],
            self.cube.add_dim_coord.call_args_list,
        )
        self.assertEqual(1, self.cube.add_aux_coord.call_count)
        self.assertEqual(
            [mock.call(self.aux_coord, [self.resolve.mapping[2]])],
            self.cube.add_aux_coord.call_args_list,
        )
        self.assertEqual(1, self.cube.add_aux_factory.call_count)
        self.assertEqual(
            [mock.call(self.aux_factory)],
            self.cube.add_aux_factory.call_args_list,
        )

    def test_compatible(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:          <- src:
        #   dims  0 1 2      dims  0 1 2
        #   shape 4 3 2      shape 4 3 2
        #   state c c c      state c c c
        #                    coord d   a
        #
        # src-to-tgt mapping:
        #   0->0, 1->1, 2->2
        src_shape = (4, 3, 2)
        self._make_cube("src", src_shape)
        tgt_shape = (4, 3, 2)
        self._make_cube("tgt", tgt_shape)
        mapping = {0: 0, 1: 1, 2: 2}
        self.resolve.mapping = mapping
        self.resolve._as_compatible_cubes()
        self._check_compatible(broadcast_shape=tgt_shape)
        self.assertEqual([mock.call(self.data)], self.mocker.call_args_list)

    def test_compatible__transpose(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:          <- src:
        #   dims  0 1 2      dims  0 1 2
        #   shape 4 3 2      shape 2 3 4
        #   state c c c      state c c c
        #                    coord d   a
        #
        # src-to-tgt mapping:
        #   0->2, 1->1, 2->0
        src_shape = (2, 3, 4)
        self._make_cube("src", src_shape, transpose_shape=(4, 3, 2))
        tgt_shape = (4, 3, 2)
        self._make_cube("tgt", tgt_shape)
        mapping = {0: 2, 1: 1, 2: 0}
        self.resolve.mapping = mapping
        self.resolve._as_compatible_cubes()
        self._check_compatible(broadcast_shape=tgt_shape)
        self.assertEqual(1, self.data.transpose.call_count)
        self.assertEqual(
            [mock.call([2, 1, 0])], self.data.transpose.call_args_list
        )
        self.assertEqual(
            [mock.call(self.transpose)], self.mocker.call_args_list
        )

    def test_compatible__reshape(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 5 4 3 2      shape 4 3 2
        #   state f c c c      state c c c
        #                      coord d   a
        #
        # src-to-tgt mapping:
        #   0->1, 1->2, 2->3
        src_shape = (4, 3, 2)
        self._make_cube("src", src_shape)
        tgt_shape = (5, 4, 3, 2)
        self._make_cube("tgt", tgt_shape)
        mapping = {0: 1, 1: 2, 2: 3}
        self.resolve.mapping = mapping
        self.resolve._as_compatible_cubes()
        self._check_compatible(broadcast_shape=tgt_shape)
        self.assertEqual(1, self.data.reshape.call_count)
        self.assertEqual(
            [mock.call((1,) + src_shape)], self.data.reshape.call_args_list
        )
        self.assertEqual([mock.call(self.reshape)], self.mocker.call_args_list)

    def test_compatible__transpose_reshape(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 5 4 3 2      shape 2 3 4
        #   state f c c c      state c c c
        #                      coord d   a
        #
        # src-to-tgt mapping:
        #   0->3, 1->2, 2->1
        src_shape = (2, 3, 4)
        transpose_shape = (4, 3, 2)
        self._make_cube("src", src_shape, transpose_shape=transpose_shape)
        tgt_shape = (5, 4, 3, 2)
        self._make_cube("tgt", tgt_shape)
        mapping = {0: 3, 1: 2, 2: 1}
        self.resolve.mapping = mapping
        self.resolve._as_compatible_cubes()
        self._check_compatible(broadcast_shape=tgt_shape)
        self.assertEqual(1, self.data.transpose.call_count)
        self.assertEqual(
            [mock.call([2, 1, 0])], self.data.transpose.call_args_list
        )
        self.assertEqual(1, self.data.reshape.call_count)
        self.assertEqual(
            [mock.call((1,) + transpose_shape)],
            self.data.reshape.call_args_list,
        )
        self.assertEqual([mock.call(self.reshape)], self.mocker.call_args_list)

    def test_compatible__broadcast(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:          <- src:
        #   dims  0 1 2      dims  0 1 2
        #   shape 1 3 2      shape 4 1 2
        #   state c c c      state c c c
        #                    coord d   a
        #   bcast ^          bcast   ^
        #
        # src-to-tgt mapping:
        #   0->0, 1->1, 2->2
        src_shape = (4, 1, 2)
        self._make_cube("src", src_shape)
        tgt_shape = (1, 3, 2)
        self._make_cube("tgt", tgt_shape)
        mapping = {0: 0, 1: 1, 2: 2}
        self.resolve.mapping = mapping
        self.resolve._as_compatible_cubes()
        self._check_compatible(broadcast_shape=(4, 3, 2))
        self.assertEqual([mock.call(self.data)], self.mocker.call_args_list)

    def test_compatible__broadcast_transpose_reshape(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 5 1 3 2      shape 2 1 4
        #   state f c c c      state c c c
        #                      coord d   a
        #   bcast   ^          bcast   ^
        #
        # src-to-tgt mapping:
        #   0->3, 1->2, 2->1
        src_shape = (2, 1, 4)
        transpose_shape = (4, 1, 2)
        self._make_cube("src", src_shape)
        tgt_shape = (5, 1, 3, 2)
        self._make_cube("tgt", tgt_shape)
        mapping = {0: 3, 1: 2, 2: 1}
        self.resolve.mapping = mapping
        self.resolve._as_compatible_cubes()
        self._check_compatible(broadcast_shape=(5, 4, 3, 2))
        self.assertEqual(1, self.data.transpose.call_count)
        self.assertEqual(
            [mock.call([2, 1, 0])], self.data.transpose.call_args_list
        )
        self.assertEqual(1, self.data.reshape.call_count)
        self.assertEqual(
            [mock.call((1,) + transpose_shape)],
            self.data.reshape.call_args_list,
        )
        self.assertEqual([mock.call(self.reshape)], self.mocker.call_args_list)


class Test__metadata_mapping(tests.IrisTest):
    def setUp(self):
        self.ndim = sentinel.ndim
        self.src_cube = mock.Mock(ndim=self.ndim)
        self.src_dim_coverage = mock.Mock(dims_free=[])
        self.src_aux_coverage = mock.Mock(dims_free=[])
        self.tgt_cube = mock.Mock(ndim=self.ndim)
        self.tgt_dim_coverage = mock.Mock(dims_free=[])
        self.tgt_aux_coverage = mock.Mock(dims_free=[])
        self.resolve = Resolve()
        self.map_rhs_to_lhs = True
        self.resolve.map_rhs_to_lhs = self.map_rhs_to_lhs
        self.resolve.rhs_cube = self.src_cube
        self.resolve.rhs_cube_dim_coverage = self.src_dim_coverage
        self.resolve.rhs_cube_aux_coverage = self.src_aux_coverage
        self.resolve.lhs_cube = self.tgt_cube
        self.resolve.lhs_cube_dim_coverage = self.tgt_dim_coverage
        self.resolve.lhs_cube_aux_coverage = self.tgt_aux_coverage
        self.resolve.mapping = {}
        self.shape = sentinel.shape
        self.resolve._broadcast_shape = self.shape
        self.resolve._src_cube_resolved = mock.Mock(shape=self.shape)
        self.resolve._tgt_cube_resolved = mock.Mock(shape=self.shape)
        self.m_dim_mapping = self.patch(
            "iris.common.resolve.Resolve._dim_mapping", return_value={}
        )
        self.m_aux_mapping = self.patch(
            "iris.common.resolve.Resolve._aux_mapping", return_value={}
        )
        self.m_free_mapping = self.patch(
            "iris.common.resolve.Resolve._free_mapping"
        )
        self.m_as_compatible_cubes = self.patch(
            "iris.common.resolve.Resolve._as_compatible_cubes"
        )
        self.mapping = {0: 1, 1: 2, 2: 3}

    def test_mapped__dim_coords(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 5 4 3 2      shape 4 3 2
        #   state f c c c      state c c c
        #   coord   d d d      coord d d d
        #
        # src-to-tgt mapping:
        #   0->1, 1->2, 2->3
        self.src_cube.ndim = 3
        self.m_dim_mapping.return_value = self.mapping
        self.resolve._metadata_mapping()
        self.assertEqual(self.mapping, self.resolve.mapping)
        self.assertEqual(1, self.m_dim_mapping.call_count)
        expected = [mock.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        self.assertEqual(expected, self.m_dim_mapping.call_args_list)
        self.assertEqual(0, self.m_aux_mapping.call_count)
        self.assertEqual(0, self.m_free_mapping.call_count)
        self.assertEqual(1, self.m_as_compatible_cubes.call_count)

    def test_mapped__aux_coords(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 5 4 3 2      shape 4 3 2
        #   state f c c c      state c c c
        #   coord   a a a      coord a a a
        #
        # src-to-tgt mapping:
        #   0->1, 1->2, 2->3
        self.src_cube.ndim = 3
        self.m_aux_mapping.return_value = self.mapping
        self.resolve._metadata_mapping()
        self.assertEqual(self.mapping, self.resolve.mapping)
        self.assertEqual(1, self.m_dim_mapping.call_count)
        expected = [mock.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        self.assertEqual(expected, self.m_dim_mapping.call_args_list)
        self.assertEqual(1, self.m_aux_mapping.call_count)
        expected = [mock.call(self.src_aux_coverage, self.tgt_aux_coverage)]
        self.assertEqual(expected, self.m_aux_mapping.call_args_list)
        self.assertEqual(0, self.m_free_mapping.call_count)
        self.assertEqual(1, self.m_as_compatible_cubes.call_count)

    def test_mapped__dim_and_aux_coords(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 5 4 3 2      shape 4 3 2
        #   state f c c c      state c c c
        #   coord   d a d      coord d a d
        #
        # src-to-tgt mapping:
        #   0->1, 1->2, 2->3
        dim_mapping = {0: 1, 2: 3}
        aux_mapping = {1: 2}
        self.src_cube.ndim = 3
        self.m_dim_mapping.return_value = dim_mapping
        self.m_aux_mapping.return_value = aux_mapping
        self.resolve._metadata_mapping()
        self.assertEqual(self.mapping, self.resolve.mapping)
        self.assertEqual(1, self.m_dim_mapping.call_count)
        expected = [mock.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        self.assertEqual(expected, self.m_dim_mapping.call_args_list)
        self.assertEqual(1, self.m_aux_mapping.call_count)
        expected = [mock.call(self.src_aux_coverage, self.tgt_aux_coverage)]
        self.assertEqual(expected, self.m_aux_mapping.call_args_list)
        self.assertEqual(0, self.m_free_mapping.call_count)
        self.assertEqual(1, self.m_as_compatible_cubes.call_count)

    def test_mapped__dim_coords_and_free_dims(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 5 4 3 2      shape 4 3 2
        #   state l f c c      state f c c
        #   coord d   d d      coord   d d
        #
        # src-to-tgt mapping:
        #   0->1, 1->2, 2->3
        dim_mapping = {1: 2, 2: 3}
        free_mapping = {0: 1}
        self.src_cube.ndim = 3
        self.m_dim_mapping.return_value = dim_mapping
        side_effect = lambda a, b, c, d: self.resolve.mapping.update(
            free_mapping
        )
        self.m_free_mapping.side_effect = side_effect
        self.resolve._metadata_mapping()
        self.assertEqual(self.mapping, self.resolve.mapping)
        self.assertEqual(1, self.m_dim_mapping.call_count)
        expected = [mock.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        self.assertEqual(expected, self.m_dim_mapping.call_args_list)
        self.assertEqual(1, self.m_aux_mapping.call_count)
        expected = [mock.call(self.src_aux_coverage, self.tgt_aux_coverage)]
        self.assertEqual(expected, self.m_aux_mapping.call_args_list)
        self.assertEqual(1, self.m_free_mapping.call_count)
        expected = [
            mock.call(
                self.src_dim_coverage,
                self.tgt_dim_coverage,
                self.src_aux_coverage,
                self.tgt_aux_coverage,
            )
        ]
        self.assertEqual(expected, self.m_free_mapping.call_args_list)
        self.assertEqual(1, self.m_as_compatible_cubes.call_count)

    def test_mapped__dim_coords_with_broadcast_flip(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 4      dims  0 1 2 4
        #   shape 1 4 3 2      shape 5 4 3 2
        #   state c c c c      state c c c c
        #   coord d d d d      coord d d d d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1, 2->2, 3->3
        mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        self.src_cube.ndim = 4
        self.tgt_cube.ndim = 4
        self.m_dim_mapping.return_value = mapping
        broadcast_shape = (5, 4, 3, 2)
        self.resolve._broadcast_shape = broadcast_shape
        self.resolve._src_cube_resolved.shape = broadcast_shape
        self.resolve._tgt_cube_resolved.shape = (1, 4, 3, 2)
        self.resolve._metadata_mapping()
        self.assertEqual(mapping, self.resolve.mapping)
        self.assertEqual(1, self.m_dim_mapping.call_count)
        expected = [mock.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        self.assertEqual(expected, self.m_dim_mapping.call_args_list)
        self.assertEqual(0, self.m_aux_mapping.call_count)
        self.assertEqual(0, self.m_free_mapping.call_count)
        self.assertEqual(2, self.m_as_compatible_cubes.call_count)
        self.assertEqual(not self.map_rhs_to_lhs, self.resolve.map_rhs_to_lhs)

    def test_mapped__dim_coords_free_flip_with_free_flip(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) a=aux, d=dim
        #
        # tgt:          <- src:
        #   dims  0 1 2    dims  0 1 2
        #   shape 4 3 2    shape 4 3 2
        #   state f f c    state l l c
        #   coord     d    coord d d d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1, 2->2
        dim_mapping = {2: 2}
        free_mapping = {0: 0, 1: 1}
        mapping = {0: 0, 1: 1, 2: 2}
        self.src_cube.ndim = 3
        self.tgt_cube.ndim = 3
        self.m_dim_mapping.return_value = dim_mapping
        side_effect = lambda a, b, c, d: self.resolve.mapping.update(
            free_mapping
        )
        self.m_free_mapping.side_effect = side_effect
        self.tgt_dim_coverage.dims_free = [0, 1]
        self.tgt_aux_coverage.dims_free = [0, 1]
        self.resolve._metadata_mapping()
        self.assertEqual(mapping, self.resolve.mapping)
        self.assertEqual(1, self.m_dim_mapping.call_count)
        expected = [mock.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        self.assertEqual(expected, self.m_dim_mapping.call_args_list)
        self.assertEqual(1, self.m_aux_mapping.call_count)
        expected = [mock.call(self.src_aux_coverage, self.tgt_aux_coverage)]
        self.assertEqual(expected, self.m_aux_mapping.call_args_list)
        self.assertEqual(1, self.m_free_mapping.call_count)
        expected = [
            mock.call(
                self.src_dim_coverage,
                self.tgt_dim_coverage,
                self.src_aux_coverage,
                self.tgt_aux_coverage,
            )
        ]
        self.assertEqual(expected, self.m_free_mapping.call_args_list)
        self.assertEqual(2, self.m_as_compatible_cubes.call_count)


class Test__prepare_common_dim_payload(tests.IrisTest):
    def setUp(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 5 4 3 2      shape 4 3 2
        #   state l c c c      state c c c
        #   coord   d d d      coord d d d
        #
        # src-to-tgt mapping:
        #   0->1, 1->2, 2->3
        self.points = (sentinel.points_0, sentinel.points_1, sentinel.points_2)
        self.bounds = (sentinel.bounds_0, sentinel.bounds_1, sentinel.bounds_2)
        self.pb_0 = (
            mock.Mock(copy=mock.Mock(return_value=self.points[0])),
            mock.Mock(copy=mock.Mock(return_value=self.bounds[0])),
        )
        self.pb_1 = (
            mock.Mock(copy=mock.Mock(return_value=self.points[1])),
            None,
        )
        self.pb_2 = (
            mock.Mock(copy=mock.Mock(return_value=self.points[2])),
            mock.Mock(copy=mock.Mock(return_value=self.bounds[2])),
        )
        side_effect = (self.pb_0, self.pb_1, self.pb_2)
        self.m_prepare_points_and_bounds = self.patch(
            "iris.common.resolve.Resolve._prepare_points_and_bounds",
            side_effect=side_effect,
        )
        self.resolve = Resolve()
        self.resolve.prepared_category = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        self.mapping = {0: 1, 1: 2, 2: 3}
        self.resolve.mapping = self.mapping
        self.metadata_combined = (
            sentinel.combined_0,
            sentinel.combined_1,
            sentinel.combined_2,
        )
        self.src_metadata = mock.Mock(
            combine=mock.Mock(side_effect=self.metadata_combined)
        )
        metadata = [self.src_metadata] * len(self.mapping)
        self.src_coords = [
            sentinel.src_coord_0,
            sentinel.src_coord_1,
            sentinel.src_coord_2,
        ]
        self.src_dims_common = [0, 1, 2]
        self.container = DimCoord
        self.src_dim_coverage = _DimCoverage(
            cube=None,
            metadata=metadata,
            coords=self.src_coords,
            dims_common=self.src_dims_common,
            dims_local=[],
            dims_free=[],
        )
        self.tgt_metadata = [
            sentinel.tgt_metadata_0,
            sentinel.tgt_metadata_1,
            sentinel.tgt_metadata_2,
            sentinel.tgt_metadata_3,
        ]
        self.tgt_coords = [
            sentinel.tgt_coord_0,
            sentinel.tgt_coord_1,
            sentinel.tgt_coord_2,
            sentinel.tgt_coord_3,
        ]
        self.tgt_dims_common = [1, 2, 3]
        self.tgt_dim_coverage = _DimCoverage(
            cube=None,
            metadata=self.tgt_metadata,
            coords=self.tgt_coords,
            dims_common=self.tgt_dims_common,
            dims_local=[],
            dims_free=[],
        )

    def _check(self, ignore_mismatch=None, bad_points=None):
        if bad_points is None:
            bad_points = False
        self.resolve._prepare_common_dim_payload(
            self.src_dim_coverage,
            self.tgt_dim_coverage,
            ignore_mismatch=ignore_mismatch,
        )
        self.assertEqual(0, len(self.resolve.prepared_category.items_aux))
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))
        if not bad_points:
            self.assertEqual(3, len(self.resolve.prepared_category.items_dim))
            expected = [
                _PreparedItem(
                    metadata=_PreparedMetadata(
                        combined=self.metadata_combined[0],
                        src=self.src_metadata,
                        tgt=self.tgt_metadata[self.mapping[0]],
                    ),
                    points=self.points[0],
                    bounds=self.bounds[0],
                    dims=(self.mapping[0],),
                    container=self.container,
                ),
                _PreparedItem(
                    metadata=_PreparedMetadata(
                        combined=self.metadata_combined[1],
                        src=self.src_metadata,
                        tgt=self.tgt_metadata[self.mapping[1]],
                    ),
                    points=self.points[1],
                    bounds=None,
                    dims=(self.mapping[1],),
                    container=self.container,
                ),
                _PreparedItem(
                    metadata=_PreparedMetadata(
                        combined=self.metadata_combined[2],
                        src=self.src_metadata,
                        tgt=self.tgt_metadata[self.mapping[2]],
                    ),
                    points=self.points[2],
                    bounds=self.bounds[2],
                    dims=(self.mapping[2],),
                    container=self.container,
                ),
            ]
            self.assertEqual(
                expected, self.resolve.prepared_category.items_dim
            )
        else:
            self.assertEqual(0, len(self.resolve.prepared_category.items_dim))
        self.assertEqual(3, self.m_prepare_points_and_bounds.call_count)
        if ignore_mismatch is None:
            ignore_mismatch = False
        expected = [
            mock.call(
                self.src_coords[0],
                self.tgt_coords[self.mapping[0]],
                0,
                1,
                ignore_mismatch=ignore_mismatch,
            ),
            mock.call(
                self.src_coords[1],
                self.tgt_coords[self.mapping[1]],
                1,
                2,
                ignore_mismatch=ignore_mismatch,
            ),
            mock.call(
                self.src_coords[2],
                self.tgt_coords[self.mapping[2]],
                2,
                3,
                ignore_mismatch=ignore_mismatch,
            ),
        ]
        self.assertEqual(
            expected, self.m_prepare_points_and_bounds.call_args_list
        )
        if not bad_points:
            self.assertEqual(3, self.src_metadata.combine.call_count)
            expected = [
                mock.call(metadata) for metadata in self.tgt_metadata[1:]
            ]
            self.assertEqual(
                expected, self.src_metadata.combine.call_args_list
            )

    def test__default_ignore_mismatch(self):
        self._check()

    def test__not_ignore_mismatch(self):
        self._check(ignore_mismatch=False)

    def test__ignore_mismatch(self):
        self._check(ignore_mismatch=True)

    def test__bad_points(self):
        side_effect = [(None, None)] * len(self.mapping)
        self.m_prepare_points_and_bounds.side_effect = side_effect
        self._check(bad_points=True)


class Test__prepare_common_aux_payload(tests.IrisTest):
    def setUp(self):
        # key: (state) c=common, f=free
        #      (coord) a=aux, d=dim
        #
        # tgt:            <- src:
        #   dims  0 1 2 3      dims  0 1 2
        #   shape 5 4 3 2      shape 4 3 2
        #   state l c c c      state c c c
        #   coord   a a a      coord a a a
        #
        # src-to-tgt mapping:
        #   0->1, 1->2, 2->3
        self.points = (sentinel.points_0, sentinel.points_1, sentinel.points_2)
        self.bounds = (sentinel.bounds_0, sentinel.bounds_1, sentinel.bounds_2)
        self.pb_0 = (
            mock.Mock(copy=mock.Mock(return_value=self.points[0])),
            mock.Mock(copy=mock.Mock(return_value=self.bounds[0])),
        )
        self.pb_1 = (
            mock.Mock(copy=mock.Mock(return_value=self.points[1])),
            None,
        )
        self.pb_2 = (
            mock.Mock(copy=mock.Mock(return_value=self.points[2])),
            mock.Mock(copy=mock.Mock(return_value=self.bounds[2])),
        )
        side_effect = (self.pb_0, self.pb_1, self.pb_2)
        self.m_prepare_points_and_bounds = self.patch(
            "iris.common.resolve.Resolve._prepare_points_and_bounds",
            side_effect=side_effect,
        )
        self.resolve = Resolve()
        self.resolve.prepared_category = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        self.mapping = {0: 1, 1: 2, 2: 3}
        self.resolve.mapping = self.mapping
        self.resolve.map_rhs_to_lhs = True
        self.metadata_combined = (
            sentinel.combined_0,
            sentinel.combined_1,
            sentinel.combined_2,
        )
        self.src_metadata = [
            mock.Mock(
                combine=mock.Mock(return_value=self.metadata_combined[0])
            ),
            mock.Mock(
                combine=mock.Mock(return_value=self.metadata_combined[1])
            ),
            mock.Mock(
                combine=mock.Mock(return_value=self.metadata_combined[2])
            ),
        ]
        self.src_coords = [
            sentinel.src_coord_0,
            sentinel.src_coord_1,
            sentinel.src_coord_2,
        ]
        self.src_dims = [(dim,) for dim in self.mapping.keys()]
        self.src_common_items = [
            _Item(*item)
            for item in zip(self.src_metadata, self.src_coords, self.src_dims)
        ]
        self.tgt_metadata = [sentinel.tgt_metadata_0] + self.src_metadata
        self.tgt_coords = [
            sentinel.tgt_coord_0,
            sentinel.tgt_coord_1,
            sentinel.tgt_coord_2,
            sentinel.tgt_coord_3,
        ]
        self.tgt_dims = [None] + [(dim,) for dim in self.mapping.values()]
        self.tgt_common_items = [
            _Item(*item)
            for item in zip(self.tgt_metadata, self.tgt_coords, self.tgt_dims)
        ]
        self.container = type(self.src_coords[0])

    def _check(self, ignore_mismatch=None, bad_points=None):
        if bad_points is None:
            bad_points = False
        prepared_items = []
        self.resolve._prepare_common_aux_payload(
            self.src_common_items,
            self.tgt_common_items,
            prepared_items,
            ignore_mismatch=ignore_mismatch,
        )
        if not bad_points:
            self.assertEqual(3, len(prepared_items))
            expected = [
                _PreparedItem(
                    metadata=_PreparedMetadata(
                        combined=self.metadata_combined[0],
                        src=self.src_metadata[0],
                        tgt=self.tgt_metadata[self.mapping[0]],
                    ),
                    points=self.points[0],
                    bounds=self.bounds[0],
                    dims=self.tgt_dims[self.mapping[0]],
                    container=self.container,
                ),
                _PreparedItem(
                    metadata=_PreparedMetadata(
                        combined=self.metadata_combined[1],
                        src=self.src_metadata[1],
                        tgt=self.tgt_metadata[self.mapping[1]],
                    ),
                    points=self.points[1],
                    bounds=None,
                    dims=self.tgt_dims[self.mapping[1]],
                    container=self.container,
                ),
                _PreparedItem(
                    metadata=_PreparedMetadata(
                        combined=self.metadata_combined[2],
                        src=self.src_metadata[2],
                        tgt=self.tgt_metadata[self.mapping[2]],
                    ),
                    points=self.points[2],
                    bounds=self.bounds[2],
                    dims=self.tgt_dims[self.mapping[2]],
                    container=self.container,
                ),
            ]
            self.assertEqual(expected, prepared_items)
        else:
            self.assertEqual(0, len(prepared_items))
        self.assertEqual(3, self.m_prepare_points_and_bounds.call_count)
        if ignore_mismatch is None:
            ignore_mismatch = False
        expected = [
            mock.call(
                self.src_coords[0],
                self.tgt_coords[self.mapping[0]],
                self.src_dims[0],
                self.tgt_dims[self.mapping[0]],
                ignore_mismatch=ignore_mismatch,
            ),
            mock.call(
                self.src_coords[1],
                self.tgt_coords[self.mapping[1]],
                self.src_dims[1],
                self.tgt_dims[self.mapping[1]],
                ignore_mismatch=ignore_mismatch,
            ),
            mock.call(
                self.src_coords[2],
                self.tgt_coords[self.mapping[2]],
                self.src_dims[2],
                self.tgt_dims[self.mapping[2]],
                ignore_mismatch=ignore_mismatch,
            ),
        ]
        self.assertEqual(
            expected, self.m_prepare_points_and_bounds.call_args_list
        )
        if not bad_points:
            for src_metadata, tgt_metadata in zip(
                self.src_metadata, self.tgt_metadata[1:]
            ):
                self.assertEqual(1, src_metadata.combine.call_count)
                expected = [mock.call(tgt_metadata)]
                self.assertEqual(expected, src_metadata.combine.call_args_list)

    def test__default_ignore_mismatch(self):
        self._check()

    def test__not_ignore_mismatch(self):
        self._check(ignore_mismatch=False)

    def test__ignore_mismatch(self):
        self._check(ignore_mismatch=True)

    def test__bad_points(self):
        side_effect = [(None, None)] * len(self.mapping)
        self.m_prepare_points_and_bounds.side_effect = side_effect
        self._check(bad_points=True)

    def test__no_tgt_metadata_match(self):
        item = self.tgt_common_items[0]
        tgt_common_items = [item] * len(self.tgt_common_items)
        prepared_items = []
        self.resolve._prepare_common_aux_payload(
            self.src_common_items, tgt_common_items, prepared_items
        )
        self.assertEqual(0, len(prepared_items))

    def test__multi_tgt_metadata_match(self):
        item = self.tgt_common_items[1]
        tgt_common_items = [item] * len(self.tgt_common_items)
        prepared_items = []
        self.resolve._prepare_common_aux_payload(
            self.src_common_items, tgt_common_items, prepared_items
        )
        self.assertEqual(0, len(prepared_items))


class Test__prepare_points_and_bounds(tests.IrisTest):
    def setUp(self):
        self.Coord = namedtuple(
            "Coord",
            [
                "name",
                "points",
                "bounds",
                "metadata",
                "ndim",
                "shape",
                "has_bounds",
            ],
        )
        self.Cube = namedtuple("Cube", ["name", "shape"])
        self.resolve = Resolve()
        self.resolve.map_rhs_to_lhs = True
        self.src_name = sentinel.src_name
        self.src_points = sentinel.src_points
        self.src_bounds = sentinel.src_bounds
        self.src_metadata = sentinel.src_metadata
        self.src_items = dict(
            name=lambda: self.src_name,
            points=self.src_points,
            bounds=self.src_bounds,
            metadata=self.src_metadata,
            ndim=None,
            shape=None,
            has_bounds=None,
        )
        self.tgt_name = sentinel.tgt_name
        self.tgt_points = sentinel.tgt_points
        self.tgt_bounds = sentinel.tgt_bounds
        self.tgt_metadata = sentinel.tgt_metadata
        self.tgt_items = dict(
            name=lambda: self.tgt_name,
            points=self.tgt_points,
            bounds=self.tgt_bounds,
            metadata=self.tgt_metadata,
            ndim=None,
            shape=None,
            has_bounds=None,
        )
        self.m_array_equal = self.patch(
            "iris.util.array_equal", side_effect=(True, True)
        )

    def test_coord_ndim_unequal__tgt_ndim_greater(self):
        self.src_items["ndim"] = 1
        src_coord = self.Coord(**self.src_items)
        self.tgt_items["ndim"] = 10
        tgt_coord = self.Coord(**self.tgt_items)
        points, bounds = self.resolve._prepare_points_and_bounds(
            src_coord, tgt_coord, src_dims=None, tgt_dims=None
        )
        self.assertEqual(self.tgt_points, points)
        self.assertEqual(self.tgt_bounds, bounds)

    def test_coord_ndim_unequal__src_ndim_greater(self):
        self.src_items["ndim"] = 10
        src_coord = self.Coord(**self.src_items)
        self.tgt_items["ndim"] = 1
        tgt_coord = self.Coord(**self.tgt_items)
        points, bounds = self.resolve._prepare_points_and_bounds(
            src_coord, tgt_coord, src_dims=None, tgt_dims=None
        )
        self.assertEqual(self.src_points, points)
        self.assertEqual(self.src_bounds, bounds)

    def test_coord_ndim_equal__shape_unequal_with_src_broadcasting(self):
        # key: (state) c=common, f=free
        #      (coord) x=coord
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 9 9      shape 1 9
        #   state c c      state c c
        #   coord x-x      coord x-x
        #                  bcast ^
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        broadcast_shape = (9, 9)
        ndim = len(broadcast_shape)
        self.resolve.mapping = mapping
        self.resolve._broadcast_shape = broadcast_shape
        src_shape = (1, 9)
        src_dims = tuple(mapping.keys())
        self.resolve.rhs_cube = self.Cube(name=None, shape=src_shape)
        self.src_items["ndim"] = ndim
        self.src_items["shape"] = src_shape
        src_coord = self.Coord(**self.src_items)
        tgt_shape = broadcast_shape
        tgt_dims = tuple(mapping.values())
        self.resolve.lhs_cube = self.Cube(name=None, shape=tgt_shape)
        self.tgt_items["ndim"] = ndim
        self.tgt_items["shape"] = tgt_shape
        tgt_coord = self.Coord(**self.tgt_items)
        points, bounds = self.resolve._prepare_points_and_bounds(
            src_coord, tgt_coord, src_dims, tgt_dims
        )
        self.assertEqual(self.tgt_points, points)
        self.assertEqual(self.tgt_bounds, bounds)

    def test_coord_ndim_equal__shape_unequal_with_tgt_broadcasting(self):
        # key: (state) c=common, f=free
        #      (coord) x=coord
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 1 9      shape 9 9
        #   state c c      state c c
        #   coord x-x      coord x-x
        #   bcast ^
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        broadcast_shape = (9, 9)
        ndim = len(broadcast_shape)
        self.resolve.mapping = mapping
        self.resolve._broadcast_shape = broadcast_shape
        src_shape = broadcast_shape
        src_dims = tuple(mapping.keys())
        self.resolve.rhs_cube = self.Cube(name=None, shape=src_shape)
        self.src_items["ndim"] = ndim
        self.src_items["shape"] = src_shape
        src_coord = self.Coord(**self.src_items)
        tgt_shape = (1, 9)
        tgt_dims = tuple(mapping.values())
        self.resolve.lhs_cube = self.Cube(name=None, shape=tgt_shape)
        self.tgt_items["ndim"] = ndim
        self.tgt_items["shape"] = tgt_shape
        tgt_coord = self.Coord(**self.tgt_items)
        points, bounds = self.resolve._prepare_points_and_bounds(
            src_coord, tgt_coord, src_dims, tgt_dims
        )
        self.assertEqual(self.src_points, points)
        self.assertEqual(self.src_bounds, bounds)

    def test_coord_ndim_equal__shape_unequal_with_unsupported_broadcasting(
        self,
    ):
        # key: (state) c=common, f=free
        #      (coord) x=coord
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 1 9      shape 9 1
        #   state c c      state c c
        #   coord x-x      coord x-x
        #   bcast ^        bcast   ^
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        broadcast_shape = (9, 9)
        ndim = len(broadcast_shape)
        self.resolve.mapping = mapping
        self.resolve._broadcast_shape = broadcast_shape
        src_shape = (9, 1)
        src_dims = tuple(mapping.keys())
        self.resolve.rhs_cube = self.Cube(
            name=lambda: sentinel.src_cube, shape=src_shape
        )
        self.src_items["ndim"] = ndim
        self.src_items["shape"] = src_shape
        src_coord = self.Coord(**self.src_items)
        tgt_shape = (1, 9)
        tgt_dims = tuple(mapping.values())
        self.resolve.lhs_cube = self.Cube(
            name=lambda: sentinel.tgt_cube, shape=tgt_shape
        )
        self.tgt_items["ndim"] = ndim
        self.tgt_items["shape"] = tgt_shape
        tgt_coord = self.Coord(**self.tgt_items)
        emsg = "Cannot broadcast"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = self.resolve._prepare_points_and_bounds(
                src_coord, tgt_coord, src_dims, tgt_dims
            )

    def _populate(
        self, src_points, tgt_points, src_bounds=None, tgt_bounds=None
    ):
        # key: (state) c=common, f=free
        #      (coord) x=coord
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state f c      state f c
        #   coord   x      coord   x
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        shape = (2, 3)
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        self.resolve.map_rhs_to_lhs = True
        self.resolve.rhs_cube = self.Cube(
            name=lambda: sentinel.src_cube, shape=None
        )
        self.resolve.lhs_cube = self.Cube(
            name=lambda: sentinel.tgt_cube, shape=None
        )
        ndim = 1
        src_dims = 1
        self.src_items["ndim"] = ndim
        self.src_items["shape"] = (shape[src_dims],)
        self.src_items["points"] = src_points
        self.src_items["bounds"] = src_bounds
        self.src_items["has_bounds"] = lambda: src_bounds is not None
        src_coord = self.Coord(**self.src_items)
        tgt_dims = 1
        self.tgt_items["ndim"] = ndim
        self.tgt_items["shape"] = (shape[mapping[tgt_dims]],)
        self.tgt_items["points"] = tgt_points
        self.tgt_items["bounds"] = tgt_bounds
        self.tgt_items["has_bounds"] = lambda: tgt_bounds is not None
        tgt_coord = self.Coord(**self.tgt_items)
        args = dict(
            src_coord=src_coord,
            tgt_coord=tgt_coord,
            src_dims=src_dims,
            tgt_dims=tgt_dims,
        )
        return args

    def test_coord_ndim_and_shape_equal__points_equal_with_no_bounds(self):
        args = self._populate(self.src_points, self.src_points)
        points, bounds = self.resolve._prepare_points_and_bounds(**args)
        self.assertEqual(self.src_points, points)
        self.assertIsNone(bounds)
        self.assertEqual(1, self.m_array_equal.call_count)
        expected = [mock.call(self.src_points, self.src_points, withnans=True)]
        self.assertEqual(expected, self.m_array_equal.call_args_list)

    def test_coord_ndim_and_shape_equal__points_equal_with_src_bounds_only(
        self,
    ):
        args = self._populate(
            self.src_points, self.src_points, src_bounds=self.src_bounds
        )
        points, bounds = self.resolve._prepare_points_and_bounds(**args)
        self.assertEqual(self.src_points, points)
        self.assertEqual(self.src_bounds, bounds)
        self.assertEqual(1, self.m_array_equal.call_count)
        expected = [mock.call(self.src_points, self.src_points, withnans=True)]
        self.assertEqual(expected, self.m_array_equal.call_args_list)

    def test_coord_ndim_and_shape_equal__points_equal_with_tgt_bounds_only(
        self,
    ):
        args = self._populate(
            self.src_points, self.src_points, tgt_bounds=self.tgt_bounds
        )
        points, bounds = self.resolve._prepare_points_and_bounds(**args)
        self.assertEqual(self.src_points, points)
        self.assertEqual(self.tgt_bounds, bounds)
        self.assertEqual(1, self.m_array_equal.call_count)
        expected = [mock.call(self.src_points, self.src_points, withnans=True)]
        self.assertEqual(expected, self.m_array_equal.call_args_list)

    def test_coord_ndim_and_shape_equal__points_equal_with_src_bounds_only_strict(
        self,
    ):
        args = self._populate(
            self.src_points, self.src_points, src_bounds=self.src_bounds
        )
        with LENIENT.context(maths=False):
            emsg = f"Coordinate {self.src_name} has bounds"
            with self.assertRaisesRegex(ValueError, emsg):
                _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_equal_with_tgt_bounds_only_strict(
        self,
    ):
        args = self._populate(
            self.src_points, self.src_points, tgt_bounds=self.tgt_bounds
        )
        with LENIENT.context(maths=False):
            emsg = f"Coordinate {self.tgt_name} has bounds"
            with self.assertRaisesRegex(ValueError, emsg):
                _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_equal_with_bounds_equal(self):
        args = self._populate(
            self.src_points,
            self.src_points,
            src_bounds=self.src_bounds,
            tgt_bounds=self.src_bounds,
        )
        points, bounds = self.resolve._prepare_points_and_bounds(**args)
        self.assertEqual(self.src_points, points)
        self.assertEqual(self.src_bounds, bounds)
        self.assertEqual(2, self.m_array_equal.call_count)
        expected = [
            mock.call(self.src_points, self.src_points, withnans=True),
            mock.call(self.src_bounds, self.src_bounds, withnans=True),
        ]
        self.assertEqual(expected, self.m_array_equal.call_args_list)

    def test_coord_ndim_and_shape_equal__points_equal_with_bounds_different(
        self,
    ):
        self.m_array_equal.side_effect = (True, False)
        args = self._populate(
            self.src_points,
            self.src_points,
            src_bounds=self.src_bounds,
            tgt_bounds=self.tgt_bounds,
        )
        emsg = f"Coordinate {self.src_name} has different bounds"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_equal_with_bounds_different_ignore_mismatch(
        self,
    ):
        self.m_array_equal.side_effect = (True, False)
        args = self._populate(
            self.src_points,
            self.src_points,
            src_bounds=self.src_bounds,
            tgt_bounds=self.tgt_bounds,
        )
        points, bounds = self.resolve._prepare_points_and_bounds(
            **args, ignore_mismatch=True
        )
        self.assertEqual(self.src_points, points)
        self.assertIsNone(bounds)
        self.assertEqual(2, self.m_array_equal.call_count)
        expected = [
            mock.call(self.src_points, self.src_points, withnans=True),
            mock.call(self.src_bounds, self.tgt_bounds, withnans=True),
        ]
        self.assertEqual(expected, self.m_array_equal.call_args_list)

    def test_coord_ndim_and_shape_equal__points_equal_with_bounds_different_strict(
        self,
    ):
        self.m_array_equal.side_effect = (True, False)
        args = self._populate(
            self.src_points,
            self.src_points,
            src_bounds=self.src_bounds,
            tgt_bounds=self.tgt_bounds,
        )
        with LENIENT.context(maths=False):
            emsg = f"Coordinate {self.src_name} has different bounds"
            with self.assertRaisesRegex(ValueError, emsg):
                _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_different(self):
        self.m_array_equal.side_effect = (False,)
        args = self._populate(self.src_points, self.tgt_points)
        emsg = f"Coordinate {self.src_name} has different points"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_different_ignore_mismatch(
        self,
    ):
        self.m_array_equal.side_effect = (False,)
        args = self._populate(self.src_points, self.tgt_points)
        points, bounds = self.resolve._prepare_points_and_bounds(
            **args, ignore_mismatch=True
        )
        self.assertIsNone(points)
        self.assertIsNone(bounds)

    def test_coord_ndim_and_shape_equal__points_different_strict(self):
        self.m_array_equal.side_effect = (False,)
        args = self._populate(self.src_points, self.tgt_points)
        with LENIENT.context(maths=False):
            emsg = f"Coordinate {self.src_name} has different points"
            with self.assertRaisesRegex(ValueError, emsg):
                _ = self.resolve._prepare_points_and_bounds(**args)


class Test__create_prepared_item(tests.IrisTest):
    def setUp(self):
        Coord = namedtuple("Coord", ["points", "bounds"])
        self.points_value = sentinel.points
        self.points = mock.Mock(copy=mock.Mock(return_value=self.points_value))
        self.bounds_value = sentinel.bounds
        self.bounds = mock.Mock(copy=mock.Mock(return_value=self.bounds_value))
        self.coord = Coord(points=self.points, bounds=self.bounds)
        self.container = type(self.coord)
        self.combined = sentinel.combined
        self.src = mock.Mock(combine=mock.Mock(return_value=self.combined))
        self.tgt = sentinel.tgt

    def _check(self, src=None, tgt=None):
        dims = 0
        if src is not None and tgt is not None:
            combined = self.combined
        else:
            combined = src or tgt
        result = Resolve._create_prepared_item(
            self.coord, dims, src_metadata=src, tgt_metadata=tgt
        )
        self.assertIsInstance(result, _PreparedItem)
        self.assertIsInstance(result.metadata, _PreparedMetadata)
        expected = _PreparedMetadata(combined=combined, src=src, tgt=tgt)
        self.assertEqual(expected, result.metadata)
        self.assertEqual(self.points_value, result.points)
        self.assertEqual(1, self.points.copy.call_count)
        self.assertEqual([mock.call()], self.points.copy.call_args_list)
        self.assertEqual(self.bounds_value, result.bounds)
        self.assertEqual(1, self.bounds.copy.call_count)
        self.assertEqual([mock.call()], self.bounds.copy.call_args_list)
        self.assertEqual((dims,), result.dims)
        self.assertEqual(self.container, result.container)

    def test__no_metadata(self):
        self._check()

    def test__src_metadata_only(self):
        self._check(src=self.src)

    def test__tgt_metadata_only(self):
        self._check(tgt=self.tgt)

    def test__combine_metadata(self):
        self._check(src=self.src, tgt=self.tgt)


class Test__prepare_local_payload_dim(tests.IrisTest):
    def setUp(self):
        self.Cube = namedtuple("Cube", ["ndim"])
        self.resolve = Resolve()
        self.resolve.prepared_category = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        self.resolve.map_rhs_to_lhs = True
        self.src_coverage = dict(
            cube=None,
            metadata=[],
            coords=[],
            dims_common=None,
            dims_local=[],
            dims_free=None,
        )
        self.tgt_coverage = deepcopy(self.src_coverage)
        self.prepared_item = sentinel.prepared_item
        self.m_create_prepared_item = self.patch(
            "iris.common.resolve.Resolve._create_prepared_item",
            return_value=self.prepared_item,
        )

    def test_src_no_local_with_tgt_no_local(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c c      state c c
        #   coord d d      coord d d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_dim))

    def test_src_no_local_with_tgt_no_local__strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c c      state c c
        #   coord d d      coord d d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_dim))

    def test_src_local_with_tgt_local(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c l      state c l
        #   coord d d      coord d d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        self.src_coverage["dims_local"] = (1,)
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["dims_local"] = (1,)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_dim))

    def test_src_local_with_tgt_local__strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c l      state c l
        #   coord d d      coord d d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        self.src_coverage["dims_local"] = (1,)
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["dims_local"] = (1,)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_dim))

    def test_src_local_with_tgt_free(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c f      state c l
        #   coord d        coord d d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_dim = 1
        self.src_coverage["dims_local"] = (src_dim,)
        src_metadata = sentinel.src_metadata
        self.src_coverage["metadata"] = [None, src_metadata]
        src_coord = sentinel.src_coord
        self.src_coverage["coords"] = [None, src_coord]
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_dim))
        self.assertEqual(
            self.prepared_item, self.resolve.prepared_category.items_dim[0]
        )
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        expected = [
            mock.call(src_coord, mapping[src_dim], src_metadata=src_metadata)
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_local_with_tgt_free__strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c f      state c l
        #   coord d        coord d d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_dim = 1
        self.src_coverage["dims_local"] = (src_dim,)
        src_metadata = sentinel.src_metadata
        self.src_coverage["metadata"] = [None, src_metadata]
        src_coord = sentinel.src_coord
        self.src_coverage["coords"] = [None, src_coord]
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_dim))

    def test_src_free_with_tgt_local(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c l      state c f
        #   coord d d      coord d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_dim = 1
        self.tgt_coverage["dims_local"] = (tgt_dim,)
        tgt_metadata = sentinel.tgt_metadata
        self.tgt_coverage["metadata"] = [None, tgt_metadata]
        tgt_coord = sentinel.tgt_coord
        self.tgt_coverage["coords"] = [None, tgt_coord]
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_dim))
        self.assertEqual(
            self.prepared_item, self.resolve.prepared_category.items_dim[0]
        )
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        expected = [mock.call(tgt_coord, tgt_dim, tgt_metadata=tgt_metadata)]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_free_with_tgt_local__strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c l      state c f
        #   coord d d      coord d
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_dim = 1
        self.tgt_coverage["dims_local"] = (tgt_dim,)
        tgt_metadata = sentinel.tgt_metadata
        self.tgt_coverage["metadata"] = [None, tgt_metadata]
        tgt_coord = sentinel.tgt_coord
        self.tgt_coverage["coords"] = [None, tgt_coord]
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_dim))

    def test_src_no_local_with_tgt_local__extra_dims(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:           <- src:
        #   dims  0 1 2      dims  0 1
        #   shape 4 2 3      shape 2 3
        #   state l c c      state c c
        #   coord d d d      coord d d
        #
        # src-to-tgt mapping:
        #   0->1, 1->2
        mapping = {0: 1, 1: 2}
        self.resolve.mapping = mapping
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=3)
        tgt_dim = 0
        self.tgt_coverage["dims_local"] = (tgt_dim,)
        tgt_metadata = sentinel.tgt_metadata
        self.tgt_coverage["metadata"] = [tgt_metadata, None, None]
        tgt_coord = sentinel.tgt_coord
        self.tgt_coverage["coords"] = [tgt_coord, None, None]
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_dim))
        self.assertEqual(
            self.prepared_item, self.resolve.prepared_category.items_dim[0]
        )
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        expected = [mock.call(tgt_coord, tgt_dim, tgt_metadata=tgt_metadata)]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_no_local_with_tgt_local__extra_dims_strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:           <- src:
        #   dims  0 1 2      dims  0 1
        #   shape 4 2 3      shape 2 3
        #   state l c c      state c c
        #   coord d d d      coord d d
        #
        # src-to-tgt mapping:
        #   0->1, 1->2
        mapping = {0: 1, 1: 2}
        self.resolve.mapping = mapping
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=3)
        tgt_dim = 0
        self.tgt_coverage["dims_local"] = (tgt_dim,)
        tgt_metadata = sentinel.tgt_metadata
        self.tgt_coverage["metadata"] = [tgt_metadata, None, None]
        tgt_coord = sentinel.tgt_coord
        self.tgt_coverage["coords"] = [tgt_coord, None, None]
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_dim))
        self.assertEqual(
            self.prepared_item, self.resolve.prepared_category.items_dim[0]
        )
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        expected = [mock.call(tgt_coord, tgt_dim, tgt_metadata=tgt_metadata)]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)


class Test__prepare_local_payload_aux(tests.IrisTest):
    def setUp(self):
        self.Cube = namedtuple("Cube", ["ndim"])
        self.resolve = Resolve()
        self.resolve.prepared_category = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        self.resolve.map_rhs_to_lhs = True
        self.src_coverage = dict(
            cube=None,
            common_items_aux=None,
            common_items_scalar=None,
            local_items_aux=[],
            local_items_scalar=None,
            dims_common=None,
            dims_local=[],
            dims_free=None,
        )
        self.tgt_coverage = deepcopy(self.src_coverage)
        self.src_prepared_item = sentinel.src_prepared_item
        self.tgt_prepared_item = sentinel.tgt_prepared_item
        self.m_create_prepared_item = self.patch(
            "iris.common.resolve.Resolve._create_prepared_item",
            side_effect=(self.src_prepared_item, self.tgt_prepared_item),
        )

    def test_src_no_local_with_tgt_no_local(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c c      state c c
        #   coord a a      coord a a
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_aux))

    def test_src_no_local_with_tgt_no_local__strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c c      state c c
        #   coord a a      coord a a
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_aux))

    def test_src_local_with_tgt_local(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c l      state c l
        #   coord a a      coord a a
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_dims = (1,)
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=src_dims)
        self.src_coverage["local_items_aux"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_dims = (1,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(2, len(self.resolve.prepared_category.items_aux))
        expected = [self.src_prepared_item, self.tgt_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_aux)
        expected = [
            mock.call(src_coord, tgt_dims, src_metadata=src_metadata),
            mock.call(tgt_coord, tgt_dims, tgt_metadata=tgt_metadata),
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_local_with_tgt_local__strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c l      state c l
        #   coord a a      coord a a
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_dims = (1,)
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=src_dims)
        self.src_coverage["local_items_aux"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_dims = (1,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_aux))

    def test_src_local_with_tgt_free(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c f      state c l
        #   coord a        coord a a
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_dims = (1,)
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=src_dims)
        self.src_coverage["local_items_aux"].append(src_item)
        self.src_coverage["dims_local"].extend(src_dims)
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_aux))
        expected = [self.src_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_aux)
        expected = [mock.call(src_coord, src_dims, src_metadata=src_metadata)]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_local_with_tgt_free__strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c f      state c l
        #   coord a        coord a a
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_dims = (1,)
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=src_dims)
        self.src_coverage["local_items_aux"].append(src_item)
        self.src_coverage["dims_local"].extend(src_dims)
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_aux))

    def test_src_free_with_tgt_local(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c l      state c f
        #   coord a a      coord a
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_dims = (1,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        self.tgt_coverage["dims_local"].extend(tgt_dims)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_aux))
        expected = [self.tgt_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_aux)
        expected = [mock.call(tgt_coord, tgt_dims, tgt_metadata=tgt_metadata)]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_free_with_tgt_local__strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:         <- src:
        #   dims  0 1      dims  0 1
        #   shape 2 3      shape 2 3
        #   state c l      state c f
        #   coord a a      coord a
        #
        # src-to-tgt mapping:
        #   0->0, 1->1
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        mapping = {0: 0, 1: 1}
        self.resolve.mapping = mapping
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_dims = (1,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        self.tgt_coverage["dims_local"].extend(tgt_dims)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_aux))

    def test_src_no_local_with_tgt_local__extra_dims(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:           <- src:
        #   dims  0 1 2      dims  0 1
        #   shape 4 2 3      shape 2 3
        #   state l c c      state c c
        #   coord a a a      coord a a
        #
        # src-to-tgt mapping:
        #   0->1, 1->2
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        mapping = {0: 1, 1: 2}
        self.resolve.mapping = mapping
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=3)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_dims = (0,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        self.tgt_coverage["dims_local"].extend(tgt_dims)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_aux))
        expected = [self.tgt_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_aux)
        expected = [mock.call(tgt_coord, tgt_dims, tgt_metadata=tgt_metadata)]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_no_local_with_tgt_local__extra_dims_strict(self):
        # key: (state) c=common, f=free, l=local
        #      (coord) d=dim
        #
        # tgt:           <- src:
        #   dims  0 1 2      dims  0 1
        #   shape 4 2 3      shape 2 3
        #   state l c c      state c c
        #   coord a a a      coord a a
        #
        # src-to-tgt mapping:
        #   0->1, 1->2
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        mapping = {0: 1, 1: 2}
        self.resolve.mapping = mapping
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=3)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_dims = (0,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        self.tgt_coverage["dims_local"].extend(tgt_dims)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=True):
            self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_aux))
        expected = [self.tgt_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_aux)
        expected = [mock.call(tgt_coord, tgt_dims, tgt_metadata=tgt_metadata)]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)


class Test__prepare_local_payload_scalar(tests.IrisTest):
    def setUp(self):
        self.Cube = namedtuple("Cube", ["ndim"])
        self.resolve = Resolve()
        self.resolve.prepared_category = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        self.src_coverage = dict(
            cube=None,
            common_items_aux=None,
            common_items_scalar=None,
            local_items_aux=None,
            local_items_scalar=[],
            dims_common=None,
            dims_local=[],
            dims_free=None,
        )
        self.tgt_coverage = deepcopy(self.src_coverage)
        self.src_prepared_item = sentinel.src_prepared_item
        self.tgt_prepared_item = sentinel.tgt_prepared_item
        self.m_create_prepared_item = self.patch(
            "iris.common.resolve.Resolve._create_prepared_item",
            side_effect=(self.src_prepared_item, self.tgt_prepared_item),
        )
        self.src_dims = ()
        self.tgt_dims = ()

    def test_src_no_local_with_tgt_no_local(self):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))

    def test_src_no_local_with_tgt_no_local__strict(self):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(
                src_coverage, tgt_coverage
            )
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))

    def test_src_no_local_with_tgt_no_local__src_scalar_cube(self):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))

    def test_src_no_local_with_tgt_no_local__src_scalar_cube_strict(self):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(
                src_coverage, tgt_coverage
            )
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))

    def test_src_local_with_tgt_no_local(self):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_item = _Item(
            metadata=src_metadata, coord=src_coord, dims=self.src_dims
        )
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_scalar))
        expected = [self.src_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_scalar)
        expected = [
            mock.call(src_coord, self.src_dims, src_metadata=src_metadata)
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_local_with_tgt_no_local__strict(self):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_item = _Item(
            metadata=src_metadata, coord=src_coord, dims=self.src_dims
        )
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(
                src_coverage, tgt_coverage
            )
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))

    def test_src_local_with_tgt_no_local__src_scalar_cube(self):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_item = _Item(
            metadata=src_metadata, coord=src_coord, dims=self.src_dims
        )
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_scalar))
        expected = [self.src_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_scalar)
        expected = [
            mock.call(src_coord, self.src_dims, src_metadata=src_metadata)
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_local_with_tgt_no_local__src_scalar_cube_strict(self):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_item = _Item(
            metadata=src_metadata, coord=src_coord, dims=self.src_dims
        )
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(
                src_coverage, tgt_coverage
            )
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))

    def test_src_no_local_with_tgt_local(self):
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_item = _Item(
            metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims
        )
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_scalar))
        expected = [self.tgt_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_scalar)
        expected = [
            mock.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata)
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_no_local_with_tgt_local__strict(self):
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_item = _Item(
            metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims
        )
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(
                src_coverage, tgt_coverage
            )
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))

    def test_src_no_local_with_tgt_local__src_scalar_cube(self):
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_item = _Item(
            metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims
        )
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        self.assertEqual(1, len(self.resolve.prepared_category.items_scalar))
        expected = [self.tgt_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_scalar)
        expected = [
            mock.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata)
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_no_local_with_tgt_local__src_scalar_cube_strict(self):
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_item = _Item(
            metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims
        )
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(
                src_coverage, tgt_coverage
            )
        self.assertEqual(1, len(self.resolve.prepared_category.items_scalar))
        expected = [self.tgt_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_scalar)
        expected = [
            mock.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata)
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_local_with_tgt_local(self):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_item = _Item(
            metadata=src_metadata, coord=src_coord, dims=self.src_dims
        )
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_item = _Item(
            metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims
        )
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        self.assertEqual(2, len(self.resolve.prepared_category.items_scalar))
        expected = [self.src_prepared_item, self.tgt_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_scalar)
        expected = [
            mock.call(src_coord, self.src_dims, src_metadata=src_metadata),
            mock.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata),
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_local_with_tgt_local__strict(self):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_item = _Item(
            metadata=src_metadata, coord=src_coord, dims=self.src_dims
        )
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_item = _Item(
            metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims
        )
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(
                src_coverage, tgt_coverage
            )
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))

    def test_src_local_with_tgt_local__src_scalar_cube(self):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_item = _Item(
            metadata=src_metadata, coord=src_coord, dims=self.src_dims
        )
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_item = _Item(
            metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims
        )
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        self.assertEqual(2, len(self.resolve.prepared_category.items_scalar))
        expected = [self.src_prepared_item, self.tgt_prepared_item]
        self.assertEqual(expected, self.resolve.prepared_category.items_scalar)
        expected = [
            mock.call(src_coord, self.src_dims, src_metadata=src_metadata),
            mock.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata),
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_src_local_with_tgt_local__src_scalar_cube_strict(self):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = sentinel.src_metadata
        src_coord = sentinel.src_coord
        src_item = _Item(
            metadata=src_metadata, coord=src_coord, dims=self.src_dims
        )
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = sentinel.tgt_metadata
        tgt_coord = sentinel.tgt_coord
        tgt_item = _Item(
            metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims
        )
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(
                src_coverage, tgt_coverage
            )
        self.assertEqual(0, len(self.resolve.prepared_category.items_scalar))


class Test__prepare_local_payload(tests.IrisTest):
    def test(self):
        src_dim_coverage = sentinel.src_dim_coverage
        src_aux_coverage = sentinel.src_aux_coverage
        tgt_dim_coverage = sentinel.tgt_dim_coverage
        tgt_aux_coverage = sentinel.tgt_aux_coverage
        root = "iris.common.resolve.Resolve"
        m_prepare_dim = self.patch(f"{root}._prepare_local_payload_dim")
        m_prepare_aux = self.patch(f"{root}._prepare_local_payload_aux")
        m_prepare_scalar = self.patch(f"{root}._prepare_local_payload_scalar")
        resolve = Resolve()
        resolve._prepare_local_payload(
            src_dim_coverage,
            src_aux_coverage,
            tgt_dim_coverage,
            tgt_aux_coverage,
        )
        self.assertEqual(1, m_prepare_dim.call_count)
        expected = [mock.call(src_dim_coverage, tgt_dim_coverage)]
        self.assertEqual(expected, m_prepare_dim.call_args_list)
        self.assertEqual(1, m_prepare_aux.call_count)
        expected = [mock.call(src_aux_coverage, tgt_aux_coverage)]
        self.assertEqual(expected, m_prepare_aux.call_args_list)
        self.assertEqual(1, m_prepare_scalar.call_count)
        expected = [mock.call(src_aux_coverage, tgt_aux_coverage)]
        self.assertEqual(expected, m_prepare_scalar.call_args_list)


class Test__metadata_prepare(tests.IrisTest):
    def setUp(self):
        self.src_cube = sentinel.src_cube
        self.src_category_local = sentinel.src_category_local
        self.src_dim_coverage = sentinel.src_dim_coverage
        self.src_aux_coverage = mock.Mock(
            common_items_aux=sentinel.src_aux_coverage_common_items_aux,
            common_items_scalar=sentinel.src_aux_coverage_common_items_scalar,
        )
        self.tgt_cube = sentinel.tgt_cube
        self.tgt_category_local = sentinel.tgt_category_local
        self.tgt_dim_coverage = sentinel.tgt_dim_coverage
        self.tgt_aux_coverage = mock.Mock(
            common_items_aux=sentinel.tgt_aux_coverage_common_items_aux,
            common_items_scalar=sentinel.tgt_aux_coverage_common_items_scalar,
        )
        self.resolve = Resolve()
        root = "iris.common.resolve.Resolve"
        self.m_prepare_common_dim_payload = self.patch(
            f"{root}._prepare_common_dim_payload"
        )
        self.m_prepare_common_aux_payload = self.patch(
            f"{root}._prepare_common_aux_payload"
        )
        self.m_prepare_local_payload = self.patch(
            f"{root}._prepare_local_payload"
        )
        self.m_prepare_factory_payload = self.patch(
            f"{root}._prepare_factory_payload"
        )

    def _check(self):
        self.assertIsNone(self.resolve.prepared_category)
        self.assertIsNone(self.resolve.prepared_factories)
        self.resolve._metadata_prepare()
        expected = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        self.assertEqual(expected, self.resolve.prepared_category)
        self.assertEqual([], self.resolve.prepared_factories)
        self.assertEqual(1, self.m_prepare_common_dim_payload.call_count)
        expected = [mock.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        self.assertEqual(
            expected, self.m_prepare_common_dim_payload.call_args_list
        )
        self.assertEqual(2, self.m_prepare_common_aux_payload.call_count)
        expected = [
            mock.call(
                self.src_aux_coverage.common_items_aux,
                self.tgt_aux_coverage.common_items_aux,
                [],
            ),
            mock.call(
                self.src_aux_coverage.common_items_scalar,
                self.tgt_aux_coverage.common_items_scalar,
                [],
                ignore_mismatch=True,
            ),
        ]
        self.assertEqual(
            expected, self.m_prepare_common_aux_payload.call_args_list
        )
        self.assertEqual(1, self.m_prepare_local_payload.call_count)
        expected = [
            mock.call(
                self.src_dim_coverage,
                self.src_aux_coverage,
                self.tgt_dim_coverage,
                self.tgt_aux_coverage,
            )
        ]
        self.assertEqual(expected, self.m_prepare_local_payload.call_args_list)
        self.assertEqual(2, self.m_prepare_factory_payload.call_count)
        expected = [
            mock.call(self.tgt_cube, self.tgt_category_local, from_src=False),
            mock.call(self.src_cube, self.src_category_local),
        ]
        self.assertEqual(
            expected, self.m_prepare_factory_payload.call_args_list
        )

    def test_map_rhs_to_lhs__true(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve.rhs_cube = self.src_cube
        self.resolve.rhs_cube_category_local = self.src_category_local
        self.resolve.rhs_cube_dim_coverage = self.src_dim_coverage
        self.resolve.rhs_cube_aux_coverage = self.src_aux_coverage
        self.resolve.lhs_cube = self.tgt_cube
        self.resolve.lhs_cube_category_local = self.tgt_category_local
        self.resolve.lhs_cube_dim_coverage = self.tgt_dim_coverage
        self.resolve.lhs_cube_aux_coverage = self.tgt_aux_coverage
        self._check()

    def test_map_rhs_to_lhs__false(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve.lhs_cube = self.src_cube
        self.resolve.lhs_cube_category_local = self.src_category_local
        self.resolve.lhs_cube_dim_coverage = self.src_dim_coverage
        self.resolve.lhs_cube_aux_coverage = self.src_aux_coverage
        self.resolve.rhs_cube = self.tgt_cube
        self.resolve.rhs_cube_category_local = self.tgt_category_local
        self.resolve.rhs_cube_dim_coverage = self.tgt_dim_coverage
        self.resolve.rhs_cube_aux_coverage = self.tgt_aux_coverage
        self._check()


class Test__prepare_factory_payload(tests.IrisTest):
    def setUp(self):
        self.Cube = namedtuple("Cube", ["aux_factories"])
        self.Coord = namedtuple("Coord", ["metadata"])
        self.Factory_T1 = namedtuple(
            "Factory_T1", ["dependencies"]
        )  # dummy factory type
        self.container_T1 = type(self.Factory_T1(None))
        self.Factory_T2 = namedtuple(
            "Factory_T2", ["dependencies"]
        )  # dummy factory type
        self.container_T2 = type(self.Factory_T2(None))
        self.resolve = Resolve()
        self.resolve.map_rhs_to_lhs = True
        self.resolve.prepared_factories = []
        self.m_get_prepared_item = self.patch(
            "iris.common.resolve.Resolve._get_prepared_item"
        )
        self.category_local = sentinel.category_local
        self.from_src = sentinel.from_src

    def test_no_factory(self):
        cube = self.Cube(aux_factories=[])
        self.resolve._prepare_factory_payload(cube, self.category_local)
        self.assertEqual(0, len(self.resolve.prepared_factories))

    def test_skip_factory__already_prepared(self):
        aux_factory = self.Factory_T1(dependencies=None)
        aux_factories = [aux_factory]
        cube = self.Cube(aux_factories=aux_factories)
        prepared_factories = [
            _PreparedFactory(container=self.container_T1, dependencies=None),
            _PreparedFactory(container=self.container_T2, dependencies=None),
        ]
        self.resolve.prepared_factories.extend(prepared_factories)
        self.resolve._prepare_factory_payload(cube, self.category_local)
        self.assertEqual(prepared_factories, self.resolve.prepared_factories)

    def test_factory__dependency_already_prepared(self):
        coord_a = self.Coord(metadata=sentinel.coord_a_metadata)
        coord_b = self.Coord(metadata=sentinel.coord_b_metadata)
        coord_c = self.Coord(metadata=sentinel.coord_c_metadata)
        side_effect = (coord_a, coord_b, coord_c)
        self.m_get_prepared_item.side_effect = side_effect
        dependencies = dict(name_a=coord_a, name_b=coord_b, name_c=coord_c)
        aux_factory = self.Factory_T1(dependencies=dependencies)
        aux_factories = [aux_factory]
        cube = self.Cube(aux_factories=aux_factories)
        self.resolve._prepare_factory_payload(
            cube, self.category_local, from_src=self.from_src
        )
        self.assertEqual(1, len(self.resolve.prepared_factories))
        prepared_dependencies = {
            name: coord.metadata for name, coord in dependencies.items()
        }
        expected = [
            _PreparedFactory(
                container=self.container_T1, dependencies=prepared_dependencies
            )
        ]
        self.assertEqual(expected, self.resolve.prepared_factories)
        self.assertEqual(len(side_effect), self.m_get_prepared_item.call_count)
        expected = [
            mock.call(
                coord_a.metadata, self.category_local, from_src=self.from_src
            ),
            mock.call(
                coord_b.metadata, self.category_local, from_src=self.from_src
            ),
            mock.call(
                coord_c.metadata, self.category_local, from_src=self.from_src
            ),
        ]
        actual = self.m_get_prepared_item.call_args_list
        for call in expected:
            self.assertIn(call, actual)

    def test_factory__dependency_local_not_prepared(self):
        coord_a = self.Coord(metadata=sentinel.coord_a_metadata)
        coord_b = self.Coord(metadata=sentinel.coord_b_metadata)
        coord_c = self.Coord(metadata=sentinel.coord_c_metadata)
        side_effect = (None, coord_a, None, coord_b, None, coord_c)
        self.m_get_prepared_item.side_effect = side_effect
        dependencies = dict(name_a=coord_a, name_b=coord_b, name_c=coord_c)
        aux_factory = self.Factory_T1(dependencies=dependencies)
        aux_factories = [aux_factory]
        cube = self.Cube(aux_factories=aux_factories)
        self.resolve._prepare_factory_payload(
            cube, self.category_local, from_src=self.from_src
        )
        self.assertEqual(1, len(self.resolve.prepared_factories))
        prepared_dependencies = {
            name: coord.metadata for name, coord in dependencies.items()
        }
        expected = [
            _PreparedFactory(
                container=self.container_T1, dependencies=prepared_dependencies
            )
        ]
        self.assertEqual(expected, self.resolve.prepared_factories)
        self.assertEqual(len(side_effect), self.m_get_prepared_item.call_count)
        expected = [
            mock.call(
                coord_a.metadata, self.category_local, from_src=self.from_src
            ),
            mock.call(
                coord_b.metadata, self.category_local, from_src=self.from_src
            ),
            mock.call(
                coord_c.metadata, self.category_local, from_src=self.from_src
            ),
            mock.call(
                coord_a.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
            mock.call(
                coord_b.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
            mock.call(
                coord_c.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
        ]
        actual = self.m_get_prepared_item.call_args_list
        for call in expected:
            self.assertIn(call, actual)

    def test_factory__dependency_not_found(self):
        coord_a = self.Coord(metadata=sentinel.coord_a_metadata)
        coord_b = self.Coord(metadata=sentinel.coord_b_metadata)
        coord_c = self.Coord(metadata=sentinel.coord_c_metadata)
        side_effect = (None, None)
        self.m_get_prepared_item.side_effect = side_effect
        dependencies = dict(name_a=coord_a, name_b=coord_b, name_c=coord_c)
        aux_factory = self.Factory_T1(dependencies=dependencies)
        aux_factories = [aux_factory]
        cube = self.Cube(aux_factories=aux_factories)
        self.resolve._prepare_factory_payload(
            cube, self.category_local, from_src=self.from_src
        )
        self.assertEqual(0, len(self.resolve.prepared_factories))
        self.assertEqual(len(side_effect), self.m_get_prepared_item.call_count)
        expected = [
            mock.call(
                coord_a.metadata, self.category_local, from_src=self.from_src
            ),
            mock.call(
                coord_b.metadata, self.category_local, from_src=self.from_src
            ),
            mock.call(
                coord_c.metadata, self.category_local, from_src=self.from_src
            ),
            mock.call(
                coord_a.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
            mock.call(
                coord_b.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
            mock.call(
                coord_c.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
        ]
        actual = self.m_get_prepared_item.call_args_list
        for call in actual:
            self.assertIn(call, expected)


class Test__get_prepared_item(tests.IrisTest):
    def setUp(self):
        PreparedItem = namedtuple("PreparedItem", ["metadata"])
        self.resolve = Resolve()
        self.prepared_dim_metadata_src = sentinel.prepared_dim_metadata_src
        self.prepared_dim_metadata_tgt = sentinel.prepared_dim_metadata_tgt
        self.prepared_items_dim = PreparedItem(
            metadata=_PreparedMetadata(
                combined=None,
                src=self.prepared_dim_metadata_src,
                tgt=self.prepared_dim_metadata_tgt,
            )
        )
        self.prepared_aux_metadata_src = sentinel.prepared_aux_metadata_src
        self.prepared_aux_metadata_tgt = sentinel.prepared_aux_metadata_tgt
        self.prepared_items_aux = PreparedItem(
            metadata=_PreparedMetadata(
                combined=None,
                src=self.prepared_aux_metadata_src,
                tgt=self.prepared_aux_metadata_tgt,
            )
        )
        self.prepared_scalar_metadata_src = (
            sentinel.prepared_scalar_metadata_src
        )
        self.prepared_scalar_metadata_tgt = (
            sentinel.prepared_scalar_metadata_tgt
        )
        self.prepared_items_scalar = PreparedItem(
            metadata=_PreparedMetadata(
                combined=None,
                src=self.prepared_scalar_metadata_src,
                tgt=self.prepared_scalar_metadata_tgt,
            )
        )
        self.resolve.prepared_category = _CategoryItems(
            items_dim=[self.prepared_items_dim],
            items_aux=[self.prepared_items_aux],
            items_scalar=[self.prepared_items_scalar],
        )
        self.resolve.mapping = {0: 10}
        self.m_create_prepared_item = self.patch(
            "iris.common.resolve.Resolve._create_prepared_item"
        )
        self.local_dim_metadata = sentinel.local_dim_metadata
        self.local_aux_metadata = sentinel.local_aux_metadata
        self.local_scalar_metadata = sentinel.local_scalar_metadata
        self.local_coord = sentinel.local_coord
        self.local_coord_dims = (0,)
        self.local_items_dim = _Item(
            metadata=self.local_dim_metadata,
            coord=self.local_coord,
            dims=self.local_coord_dims,
        )
        self.local_items_aux = _Item(
            metadata=self.local_aux_metadata,
            coord=self.local_coord,
            dims=self.local_coord_dims,
        )
        self.local_items_scalar = _Item(
            metadata=self.local_scalar_metadata,
            coord=self.local_coord,
            dims=self.local_coord_dims,
        )
        self.category_local = _CategoryItems(
            items_dim=[self.local_items_dim],
            items_aux=[self.local_items_aux],
            items_scalar=[self.local_items_scalar],
        )

    def test_missing_prepared_coord__from_src(self):
        metadata = sentinel.missing
        category_local = None
        result = self.resolve._get_prepared_item(metadata, category_local)
        self.assertIsNone(result)

    def test_missing_prepared_coord__from_tgt(self):
        metadata = sentinel.missing
        category_local = None
        result = self.resolve._get_prepared_item(
            metadata, category_local, from_src=False
        )
        self.assertIsNone(result)

    def test_get_prepared_dim_coord__from_src(self):
        metadata = self.prepared_dim_metadata_src
        category_local = None
        result = self.resolve._get_prepared_item(metadata, category_local)
        self.assertEqual(self.prepared_items_dim, result)

    def test_get_prepared_dim_coord__from_tgt(self):
        metadata = self.prepared_dim_metadata_tgt
        category_local = None
        result = self.resolve._get_prepared_item(
            metadata, category_local, from_src=False
        )
        self.assertEqual(self.prepared_items_dim, result)

    def test_get_prepared_aux_coord__from_src(self):
        metadata = self.prepared_aux_metadata_src
        category_local = None
        result = self.resolve._get_prepared_item(metadata, category_local)
        self.assertEqual(self.prepared_items_aux, result)

    def test_get_prepared_aux_coord__from_tgt(self):
        metadata = self.prepared_aux_metadata_tgt
        category_local = None
        result = self.resolve._get_prepared_item(
            metadata, category_local, from_src=False
        )
        self.assertEqual(self.prepared_items_aux, result)

    def test_get_prepared_scalar_coord__from_src(self):
        metadata = self.prepared_scalar_metadata_src
        category_local = None
        result = self.resolve._get_prepared_item(metadata, category_local)
        self.assertEqual(self.prepared_items_scalar, result)

    def test_get_prepared_scalar_coord__from_tgt(self):
        metadata = self.prepared_scalar_metadata_tgt
        category_local = None
        result = self.resolve._get_prepared_item(
            metadata, category_local, from_src=False
        )
        self.assertEqual(self.prepared_items_scalar, result)

    def test_missing_local_coord__from_src(self):
        metadata = sentinel.missing
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_local=True
        )
        self.assertIsNone(result)

    def test_missing_local_coord__from_tgt(self):
        metadata = sentinel.missing
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_src=False, from_local=True
        )
        self.assertIsNone(result)

    def test_get_local_dim_coord__from_src(self):
        created_local_item = sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_dim_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_local=True
        )
        expected = created_local_item
        self.assertEqual(expected, result)
        self.assertEqual(2, len(self.resolve.prepared_category.items_dim))
        self.assertEqual(expected, self.resolve.prepared_category.items_dim[1])
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        dims = (self.resolve.mapping[self.local_coord_dims[0]],)
        expected = [
            mock.call(
                self.local_coord,
                dims,
                src_metadata=metadata,
                tgt_metadata=None,
            )
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_get_local_dim_coord__from_tgt(self):
        created_local_item = sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_dim_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_src=False, from_local=True
        )
        expected = created_local_item
        self.assertEqual(expected, result)
        self.assertEqual(2, len(self.resolve.prepared_category.items_dim))
        self.assertEqual(expected, self.resolve.prepared_category.items_dim[1])
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        dims = self.local_coord_dims
        expected = [
            mock.call(
                self.local_coord,
                dims,
                src_metadata=None,
                tgt_metadata=metadata,
            )
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_get_local_aux_coord__from_src(self):
        created_local_item = sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_aux_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_local=True
        )
        expected = created_local_item
        self.assertEqual(expected, result)
        self.assertEqual(2, len(self.resolve.prepared_category.items_aux))
        self.assertEqual(expected, self.resolve.prepared_category.items_aux[1])
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        dims = (self.resolve.mapping[self.local_coord_dims[0]],)
        expected = [
            mock.call(
                self.local_coord,
                dims,
                src_metadata=metadata,
                tgt_metadata=None,
            )
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_get_local_aux_coord__from_tgt(self):
        created_local_item = sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_aux_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_src=False, from_local=True
        )
        expected = created_local_item
        self.assertEqual(expected, result)
        self.assertEqual(2, len(self.resolve.prepared_category.items_aux))
        self.assertEqual(expected, self.resolve.prepared_category.items_aux[1])
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        dims = self.local_coord_dims
        expected = [
            mock.call(
                self.local_coord,
                dims,
                src_metadata=None,
                tgt_metadata=metadata,
            )
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_get_local_scalar_coord__from_src(self):
        created_local_item = sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_scalar_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_local=True
        )
        expected = created_local_item
        self.assertEqual(expected, result)
        self.assertEqual(2, len(self.resolve.prepared_category.items_scalar))
        self.assertEqual(
            expected, self.resolve.prepared_category.items_scalar[1]
        )
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        dims = (self.resolve.mapping[self.local_coord_dims[0]],)
        expected = [
            mock.call(
                self.local_coord,
                dims,
                src_metadata=metadata,
                tgt_metadata=None,
            )
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)

    def test_get_local_scalar_coord__from_tgt(self):
        created_local_item = sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_scalar_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_src=False, from_local=True
        )
        expected = created_local_item
        self.assertEqual(expected, result)
        self.assertEqual(2, len(self.resolve.prepared_category.items_scalar))
        self.assertEqual(
            expected, self.resolve.prepared_category.items_scalar[1]
        )
        self.assertEqual(1, self.m_create_prepared_item.call_count)
        dims = self.local_coord_dims
        expected = [
            mock.call(
                self.local_coord,
                dims,
                src_metadata=None,
                tgt_metadata=metadata,
            )
        ]
        self.assertEqual(expected, self.m_create_prepared_item.call_args_list)


class Test_cube(tests.IrisTest):
    def setUp(self):
        self.shape = (2, 3)
        self.data = np.zeros(np.multiply(*self.shape), dtype=np.int8).reshape(
            self.shape
        )
        self.bad_data = np.zeros(np.multiply(*self.shape), dtype=np.int8)
        self.resolve = Resolve()
        self.resolve.map_rhs_to_lhs = True
        self.resolve._broadcast_shape = self.shape
        self.cube_metadata = CubeMetadata(
            standard_name="air_temperature",
            long_name="air temp",
            var_name="airT",
            units=Unit("K"),
            attributes={},
            cell_methods=(),
        )
        lhs_cube = Cube(self.data)
        lhs_cube.metadata = self.cube_metadata
        self.resolve.lhs_cube = lhs_cube
        rhs_cube = Cube(self.data)
        rhs_cube.metadata = self.cube_metadata
        self.resolve.rhs_cube = rhs_cube
        self.m_add_dim_coord = self.patch("iris.cube.Cube.add_dim_coord")
        self.m_add_aux_coord = self.patch("iris.cube.Cube.add_aux_coord")
        self.m_add_aux_factory = self.patch("iris.cube.Cube.add_aux_factory")
        self.m_coord = self.patch("iris.cube.Cube.coord")
        #
        # prepared coordinates
        #
        prepared_category = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        # prepared dim coordinates
        self.prepared_dim_0_metadata = _PreparedMetadata(
            combined=sentinel.prepared_dim_0_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_dim_0_points = sentinel.prepared_dim_0_points
        self.prepared_dim_0_bounds = sentinel.prepared_dim_0_bounds
        self.prepared_dim_0_dims = (0,)
        self.prepared_dim_0_coord = mock.Mock(metadata=None)
        self.prepared_dim_0_container = mock.Mock(
            return_value=self.prepared_dim_0_coord
        )
        self.prepared_dim_0 = _PreparedItem(
            metadata=self.prepared_dim_0_metadata,
            points=self.prepared_dim_0_points,
            bounds=self.prepared_dim_0_bounds,
            dims=self.prepared_dim_0_dims,
            container=self.prepared_dim_0_container,
        )
        prepared_category.items_dim.append(self.prepared_dim_0)
        self.prepared_dim_1_metadata = _PreparedMetadata(
            combined=sentinel.prepared_dim_1_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_dim_1_points = sentinel.prepared_dim_1_points
        self.prepared_dim_1_bounds = sentinel.prepared_dim_1_bounds
        self.prepared_dim_1_dims = (1,)
        self.prepared_dim_1_coord = mock.Mock(metadata=None)
        self.prepared_dim_1_container = mock.Mock(
            return_value=self.prepared_dim_1_coord
        )
        self.prepared_dim_1 = _PreparedItem(
            metadata=self.prepared_dim_1_metadata,
            points=self.prepared_dim_1_points,
            bounds=self.prepared_dim_1_bounds,
            dims=self.prepared_dim_1_dims,
            container=self.prepared_dim_1_container,
        )
        prepared_category.items_dim.append(self.prepared_dim_1)

        # prepared auxiliary coordinates
        self.prepared_aux_0_metadata = _PreparedMetadata(
            combined=sentinel.prepared_aux_0_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_aux_0_points = sentinel.prepared_aux_0_points
        self.prepared_aux_0_bounds = sentinel.prepared_aux_0_bounds
        self.prepared_aux_0_dims = (0,)
        self.prepared_aux_0_coord = mock.Mock(metadata=None)
        self.prepared_aux_0_container = mock.Mock(
            return_value=self.prepared_aux_0_coord
        )
        self.prepared_aux_0 = _PreparedItem(
            metadata=self.prepared_aux_0_metadata,
            points=self.prepared_aux_0_points,
            bounds=self.prepared_aux_0_bounds,
            dims=self.prepared_aux_0_dims,
            container=self.prepared_aux_0_container,
        )
        prepared_category.items_aux.append(self.prepared_aux_0)
        self.prepared_aux_1_metadata = _PreparedMetadata(
            combined=sentinel.prepared_aux_1_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_aux_1_points = sentinel.prepared_aux_1_points
        self.prepared_aux_1_bounds = sentinel.prepared_aux_1_bounds
        self.prepared_aux_1_dims = (1,)
        self.prepared_aux_1_coord = mock.Mock(metadata=None)
        self.prepared_aux_1_container = mock.Mock(
            return_value=self.prepared_aux_1_coord
        )
        self.prepared_aux_1 = _PreparedItem(
            metadata=self.prepared_aux_1_metadata,
            points=self.prepared_aux_1_points,
            bounds=self.prepared_aux_1_bounds,
            dims=self.prepared_aux_1_dims,
            container=self.prepared_aux_1_container,
        )
        prepared_category.items_aux.append(self.prepared_aux_1)

        # prepare scalar coordinates
        self.prepared_scalar_0_metadata = _PreparedMetadata(
            combined=sentinel.prepared_scalar_0_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_scalar_0_points = sentinel.prepared_scalar_0_points
        self.prepared_scalar_0_bounds = sentinel.prepared_scalar_0_bounds
        self.prepared_scalar_0_dims = ()
        self.prepared_scalar_0_coord = mock.Mock(metadata=None)
        self.prepared_scalar_0_container = mock.Mock(
            return_value=self.prepared_scalar_0_coord
        )
        self.prepared_scalar_0 = _PreparedItem(
            metadata=self.prepared_scalar_0_metadata,
            points=self.prepared_scalar_0_points,
            bounds=self.prepared_scalar_0_bounds,
            dims=self.prepared_scalar_0_dims,
            container=self.prepared_scalar_0_container,
        )
        prepared_category.items_scalar.append(self.prepared_scalar_0)
        self.prepared_scalar_1_metadata = _PreparedMetadata(
            combined=sentinel.prepared_scalar_1_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_scalar_1_points = sentinel.prepared_scalar_1_points
        self.prepared_scalar_1_bounds = sentinel.prepared_scalar_1_bounds
        self.prepared_scalar_1_dims = ()
        self.prepared_scalar_1_coord = mock.Mock(metadata=None)
        self.prepared_scalar_1_container = mock.Mock(
            return_value=self.prepared_scalar_1_coord
        )
        self.prepared_scalar_1 = _PreparedItem(
            metadata=self.prepared_scalar_1_metadata,
            points=self.prepared_scalar_1_points,
            bounds=self.prepared_scalar_1_bounds,
            dims=self.prepared_scalar_1_dims,
            container=self.prepared_scalar_1_container,
        )
        prepared_category.items_scalar.append(self.prepared_scalar_1)
        #
        # prepared factories
        #
        prepared_factories = []
        self.aux_factory = sentinel.aux_factory
        self.prepared_factory_container = mock.Mock(
            return_value=self.aux_factory
        )
        self.prepared_factory_metadata_a = _PreparedMetadata(
            combined=sentinel.prepared_factory_metadata_a_combined,
            src=None,
            tgt=None,
        )
        self.prepared_factory_metadata_b = _PreparedMetadata(
            combined=sentinel.prepared_factory_metadata_b_combined,
            src=None,
            tgt=None,
        )
        self.prepared_factory_metadata_c = _PreparedMetadata(
            combined=sentinel.prepared_factory_metadata_c_combined,
            src=None,
            tgt=None,
        )
        self.prepared_factory_dependencies = dict(
            name_a=self.prepared_factory_metadata_a,
            name_b=self.prepared_factory_metadata_b,
            name_c=self.prepared_factory_metadata_c,
        )
        self.prepared_factory = _PreparedFactory(
            container=self.prepared_factory_container,
            dependencies=self.prepared_factory_dependencies,
        )
        prepared_factories.append(self.prepared_factory)
        self.prepared_factory_side_effect = (
            sentinel.prepared_factory_coord_a,
            sentinel.prepared_factory_coord_b,
            sentinel.prepared_factory_coord_c,
        )
        self.m_coord.side_effect = self.prepared_factory_side_effect
        self.resolve.prepared_category = prepared_category
        self.resolve.prepared_factories = prepared_factories

    def test_no_resolved_shape(self):
        self.resolve._broadcast_shape = None
        data = None
        emsg = "Cannot resolve resultant cube, as no candidate cubes have been provided"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = self.resolve.cube(data)

    def test_bad_data_shape(self):
        emsg = "Cannot resolve resultant cube, as the provided data must have shape"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = self.resolve.cube(self.bad_data)

    def test_bad_data_shape__inplace(self):
        self.resolve.lhs_cube = Cube(self.bad_data)
        emsg = "Cannot resolve resultant cube in-place"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = self.resolve.cube(self.data, in_place=True)

    def _check(self):
        # check dim coordinate 0
        self.assertEqual(1, self.prepared_dim_0.container.call_count)
        expected = [
            mock.call(
                self.prepared_dim_0_points, bounds=self.prepared_dim_0_bounds
            )
        ]
        self.assertEqual(
            expected, self.prepared_dim_0.container.call_args_list
        )
        self.assertEqual(
            self.prepared_dim_0_coord.metadata,
            self.prepared_dim_0_metadata.combined,
        )
        # check dim coordinate 1
        self.assertEqual(1, self.prepared_dim_1.container.call_count)
        expected = [
            mock.call(
                self.prepared_dim_1_points, bounds=self.prepared_dim_1_bounds
            )
        ]
        self.assertEqual(
            expected, self.prepared_dim_1.container.call_args_list
        )
        self.assertEqual(
            self.prepared_dim_1_coord.metadata,
            self.prepared_dim_1_metadata.combined,
        )
        # check add_dim_coord
        self.assertEqual(2, self.m_add_dim_coord.call_count)
        expected = [
            mock.call(self.prepared_dim_0_coord, self.prepared_dim_0_dims),
            mock.call(self.prepared_dim_1_coord, self.prepared_dim_1_dims),
        ]
        self.assertEqual(expected, self.m_add_dim_coord.call_args_list)

        # check aux coordinate 0
        self.assertEqual(1, self.prepared_aux_0.container.call_count)
        expected = [
            mock.call(
                self.prepared_aux_0_points, bounds=self.prepared_aux_0_bounds
            )
        ]
        self.assertEqual(
            expected, self.prepared_aux_0.container.call_args_list
        )
        self.assertEqual(
            self.prepared_aux_0_coord.metadata,
            self.prepared_aux_0_metadata.combined,
        )
        # check aux coordinate 1
        self.assertEqual(1, self.prepared_aux_1.container.call_count)
        expected = [
            mock.call(
                self.prepared_aux_1_points, bounds=self.prepared_aux_1_bounds
            )
        ]
        self.assertEqual(
            expected, self.prepared_aux_1.container.call_args_list
        )
        self.assertEqual(
            self.prepared_aux_1_coord.metadata,
            self.prepared_aux_1_metadata.combined,
        )
        # check scalar coordinate 0
        self.assertEqual(1, self.prepared_scalar_0.container.call_count)
        expected = [
            mock.call(
                self.prepared_scalar_0_points,
                bounds=self.prepared_scalar_0_bounds,
            )
        ]
        self.assertEqual(
            expected, self.prepared_scalar_0.container.call_args_list
        )
        self.assertEqual(
            self.prepared_scalar_0_coord.metadata,
            self.prepared_scalar_0_metadata.combined,
        )
        # check scalar coordinate 1
        self.assertEqual(1, self.prepared_scalar_1.container.call_count)
        expected = [
            mock.call(
                self.prepared_scalar_1_points,
                bounds=self.prepared_scalar_1_bounds,
            )
        ]
        self.assertEqual(
            expected, self.prepared_scalar_1.container.call_args_list
        )
        self.assertEqual(
            self.prepared_scalar_1_coord.metadata,
            self.prepared_scalar_1_metadata.combined,
        )
        # check add_aux_coord
        self.assertEqual(4, self.m_add_aux_coord.call_count)
        expected = [
            mock.call(self.prepared_aux_0_coord, self.prepared_aux_0_dims),
            mock.call(self.prepared_aux_1_coord, self.prepared_aux_1_dims),
            mock.call(
                self.prepared_scalar_0_coord, self.prepared_scalar_0_dims
            ),
            mock.call(
                self.prepared_scalar_1_coord, self.prepared_scalar_1_dims
            ),
        ]
        self.assertEqual(expected, self.m_add_aux_coord.call_args_list)

        # check auxiliary factories
        self.assertEqual(1, self.m_add_aux_factory.call_count)
        expected = [mock.call(self.aux_factory)]
        self.assertEqual(expected, self.m_add_aux_factory.call_args_list)
        self.assertEqual(1, self.prepared_factory_container.call_count)
        expected = [
            mock.call(
                **{
                    name: value
                    for name, value in zip(
                        sorted(self.prepared_factory_dependencies.keys()),
                        self.prepared_factory_side_effect,
                    )
                }
            )
        ]
        self.assertEqual(
            expected, self.prepared_factory_container.call_args_list
        )
        self.assertEqual(3, self.m_coord.call_count)
        expected = [
            mock.call(self.prepared_factory_metadata_a.combined),
            mock.call(self.prepared_factory_metadata_b.combined),
            mock.call(self.prepared_factory_metadata_c.combined),
        ]
        self.assertEqual(expected, self.m_coord.call_args_list)

    def test_resolve(self):
        result = self.resolve.cube(self.data)
        self.assertEqual(self.cube_metadata, result.metadata)
        self._check()
        self.assertIsNot(self.resolve.lhs_cube, result)

    def test_resolve__inplace(self):
        result = self.resolve.cube(self.data, in_place=True)
        self.assertEqual(self.cube_metadata, result.metadata)
        self._check()
        self.assertIs(self.resolve.lhs_cube, result)


if __name__ == "__main__":
    tests.main()
