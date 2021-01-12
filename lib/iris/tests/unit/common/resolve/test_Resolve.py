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
import iris.tests as tests

from collections import namedtuple
from copy import deepcopy

import unittest.mock as mock
from unittest.mock import sentinel

from iris.common.resolve import (
    Resolve,
    _AuxCoverage,
    _CategoryItems,
    _DimCoverage,
    _Item,
)
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
    pass


class Test__metadata_mapping(tests.IrisTest):
    pass


if __name__ == "__main__":
    tests.main()
