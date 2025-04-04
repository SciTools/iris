# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.resolve.Resolve`."""

from collections import namedtuple
from copy import deepcopy
from unittest import mock

from cf_units import Unit
import numpy as np
import pytest

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


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        target = "iris.common.resolve.Resolve.__call__"
        self.m_call = mocker.MagicMock(return_value=mocker.sentinel.return_value)
        mocker.patch(target, new=self.m_call)

    def _assert_members_none(self, resolve):
        assert resolve.lhs_cube_resolved is None
        assert resolve.rhs_cube_resolved is None
        assert resolve.lhs_cube_category is None
        assert resolve.rhs_cube_category is None
        assert resolve.lhs_cube_category_local is None
        assert resolve.rhs_cube_category_local is None
        assert resolve.category_common is None
        assert resolve.lhs_cube_dim_coverage is None
        assert resolve.lhs_cube_aux_coverage is None
        assert resolve.rhs_cube_dim_coverage is None
        assert resolve.rhs_cube_aux_coverage is None
        assert resolve.map_rhs_to_lhs is None
        assert resolve.mapping is None
        assert resolve.prepared_category is None
        assert resolve.prepared_factories is None
        assert resolve._broadcast_shape is None

    def test_lhs_rhs_default(self):
        resolve = Resolve()
        assert resolve.lhs_cube is None
        assert resolve.rhs_cube is None
        self._assert_members_none(resolve)
        assert self.m_call.call_count == 0

    def test_lhs_rhs_provided(self, mocker):
        m_lhs = mocker.sentinel.lhs
        m_rhs = mocker.sentinel.rhs
        resolve = Resolve(lhs=m_lhs, rhs=m_rhs)
        # The lhs_cube and rhs_cube are only None due
        # to __call__ being mocked. See Test___call__
        # for appropriate test coverage.
        assert resolve.lhs_cube is None
        assert resolve.rhs_cube is None
        self._assert_members_none(resolve)
        assert self.m_call.call_count == 1
        call_args = mocker.call(m_lhs, m_rhs)
        assert self.m_call.call_args == call_args


class Test___call__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.m_lhs = mocker.MagicMock(spec=Cube)
        self.m_rhs = mocker.MagicMock(spec=Cube)
        target = "iris.common.resolve.Resolve.{method}"
        method = target.format(method="_metadata_resolve")
        self.m_metadata_resolve = mocker.patch(method)
        method = target.format(method="_metadata_coverage")
        self.m_metadata_coverage = mocker.patch(method)
        method = target.format(method="_metadata_mapping")
        self.m_metadata_mapping = mocker.patch(method)
        method = target.format(method="_metadata_prepare")
        self.m_metadata_prepare = mocker.patch(method)

    def test_lhs_not_cube(self):
        emsg = "'LHS' argument to be a 'Cube'"
        with pytest.raises(TypeError, match=emsg):
            _ = Resolve(rhs=self.m_rhs)

    def test_rhs_not_cube(self):
        emsg = "'RHS' argument to be a 'Cube'"
        with pytest.raises(TypeError, match=emsg):
            _ = Resolve(lhs=self.m_lhs)

    def _assert_called_metadata_methods(self):
        call_args = mock.call()
        assert self.m_metadata_resolve.call_count == 1
        assert self.m_metadata_resolve.call_args == call_args
        assert self.m_metadata_coverage.call_count == 1
        assert self.m_metadata_coverage.call_args == call_args
        assert self.m_metadata_mapping.call_count == 1
        assert self.m_metadata_mapping.call_args == call_args
        assert self.m_metadata_prepare.call_count == 1
        assert self.m_metadata_prepare.call_args == call_args

    def test_map_rhs_to_lhs__less_than(self):
        self.m_lhs.ndim = 2
        self.m_rhs.ndim = 1
        resolve = Resolve(lhs=self.m_lhs, rhs=self.m_rhs)
        assert resolve.lhs_cube == self.m_lhs
        assert resolve.rhs_cube == self.m_rhs
        assert resolve.map_rhs_to_lhs
        self._assert_called_metadata_methods()

    def test_map_rhs_to_lhs__equal(self):
        self.m_lhs.ndim = 2
        self.m_rhs.ndim = 2
        resolve = Resolve(lhs=self.m_lhs, rhs=self.m_rhs)
        assert resolve.lhs_cube == self.m_lhs
        assert resolve.rhs_cube == self.m_rhs
        assert resolve.map_rhs_to_lhs
        self._assert_called_metadata_methods()

    def test_map_lhs_to_rhs(self):
        self.m_lhs.ndim = 2
        self.m_rhs.ndim = 3
        resolve = Resolve(lhs=self.m_lhs, rhs=self.m_rhs)
        assert resolve.lhs_cube == self.m_lhs
        assert resolve.rhs_cube == self.m_rhs
        assert not resolve.map_rhs_to_lhs
        self._assert_called_metadata_methods()


class Test__categorise_items:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.coord_dims = {}
        # configure dim coords
        coord = mocker.Mock(metadata=mocker.sentinel.dim_metadata1)
        self.dim_coords = [coord]
        self.coord_dims[coord] = mocker.sentinel.dims1
        # configure aux and scalar coords
        self.aux_coords = []
        pairs = [
            (mocker.sentinel.aux_metadata2, mocker.sentinel.dims2),
            (mocker.sentinel.aux_metadata3, mocker.sentinel.dims3),
            (mocker.sentinel.scalar_metadata4, None),
            (mocker.sentinel.scalar_metadata5, None),
            (mocker.sentinel.scalar_metadata6, None),
        ]
        for metadata, dims in pairs:
            coord = mocker.Mock(metadata=metadata)
            self.aux_coords.append(coord)
            self.coord_dims[coord] = dims
        func = lambda coord: self.coord_dims[coord]
        self.cube = mocker.Mock(
            aux_coords=self.aux_coords,
            dim_coords=self.dim_coords,
            coord_dims=func,
        )

    def test(self):
        result = Resolve._categorise_items(self.cube)
        assert isinstance(result, _CategoryItems)
        assert len(result.items_dim) == 1
        # check dim coords
        for item in result.items_dim:
            assert isinstance(item, _Item)
        (coord,) = self.dim_coords
        dims = self.coord_dims[coord]
        expected = [_Item(metadata=coord.metadata, coord=coord, dims=dims)]
        assert result.items_dim == expected
        # check aux coords
        assert len(result.items_aux) == 2
        for item in result.items_aux:
            assert isinstance(item, _Item)
        expected_aux, expected_scalar = [], []
        for coord in self.aux_coords:
            dims = self.coord_dims[coord]
            item = _Item(metadata=coord.metadata, coord=coord, dims=dims)
            if dims:
                expected_aux.append(item)
            else:
                expected_scalar.append(item)
        assert result.items_aux == expected_aux
        # check scalar coords
        assert len(result.items_scalar) == 3
        for item in result.items_scalar:
            assert isinstance(item, _Item)
        assert result.items_scalar == expected_scalar


class Test__metadata_resolve:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.target = "iris.common.resolve.Resolve._categorise_items"
        self.m_lhs_cube = mocker.sentinel.lhs_cube
        self.m_rhs_cube = mocker.sentinel.rhs_cube

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

    def test_metadata_same(self, mocker):
        category = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        # configure dim coords
        pairs = [(mocker.sentinel.dim_metadata1, mocker.sentinel.dims1)]
        category.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        pairs = [
            (mocker.sentinel.aux_metadata1, mocker.sentinel.dims2),
            (mocker.sentinel.aux_metadata2, mocker.sentinel.dims3),
        ]
        category.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        pairs = [
            (mocker.sentinel.scalar_metadata1, None),
            (mocker.sentinel.scalar_metadata2, None),
            (mocker.sentinel.scalar_metadata3, None),
        ]
        category.items_scalar.extend(self._create_items(pairs))

        side_effect = (category, category)
        patcher = mocker.patch(self.target, side_effect=side_effect)

        resolve = Resolve()
        assert resolve.lhs_cube is None
        assert resolve.rhs_cube is None
        assert resolve.lhs_cube_category is None
        assert resolve.rhs_cube_category is None
        assert resolve.lhs_cube_category_local is None
        assert resolve.rhs_cube_category_local is None
        assert resolve.category_common is None

        # require to explicitly configure cubes
        resolve.lhs_cube = self.m_lhs_cube
        resolve.rhs_cube = self.m_rhs_cube
        resolve._metadata_resolve()

        assert patcher.call_count == 2
        calls = [mocker.call(self.m_lhs_cube), mocker.call(self.m_rhs_cube)]
        assert patcher.call_args_list == calls

        assert resolve.lhs_cube_category == category
        assert resolve.rhs_cube_category == category
        expected = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        assert resolve.lhs_cube_category_local == expected
        assert resolve.rhs_cube_category_local == expected
        assert resolve.category_common == category

    def test_metadata_overlap(self, mocker):
        # configure the lhs cube category
        category_lhs = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        # configure dim coords
        pairs = [
            (mocker.sentinel.dim_metadata1, mocker.sentinel.dims1),
            (mocker.sentinel.dim_metadata2, mocker.sentinel.dims2),
        ]
        category_lhs.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        pairs = [
            (mocker.sentinel.aux_metadata1, mocker.sentinel.dims3),
            (mocker.sentinel.aux_metadata2, mocker.sentinel.dims4),
        ]
        category_lhs.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        pairs = [
            (mocker.sentinel.scalar_metadata1, None),
            (mocker.sentinel.scalar_metadata2, None),
        ]
        category_lhs.items_scalar.extend(self._create_items(pairs))

        # configure the rhs cube category
        category_rhs = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        # configure dim coords
        category_rhs.items_dim.append(category_lhs.items_dim[0])
        pairs = [(mocker.sentinel.dim_metadata200, mocker.sentinel.dims2)]
        category_rhs.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        category_rhs.items_aux.append(category_lhs.items_aux[0])
        pairs = [(mocker.sentinel.aux_metadata200, mocker.sentinel.dims4)]
        category_rhs.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        category_rhs.items_scalar.append(category_lhs.items_scalar[0])
        pairs = [(mocker.sentinel.scalar_metadata200, None)]
        category_rhs.items_scalar.extend(self._create_items(pairs))

        side_effect = (category_lhs, category_rhs)
        patcher = mocker.patch(self.target, side_effect=side_effect)

        resolve = Resolve()
        assert resolve.lhs_cube is None
        assert resolve.rhs_cube is None
        assert resolve.lhs_cube_category is None
        assert resolve.rhs_cube_category is None
        assert resolve.lhs_cube_category_local is None
        assert resolve.rhs_cube_category_local is None
        assert resolve.category_common is None

        # require to explicitly configure cubes
        resolve.lhs_cube = self.m_lhs_cube
        resolve.rhs_cube = self.m_rhs_cube
        resolve._metadata_resolve()

        assert patcher.call_count == 2
        calls = [mocker.call(self.m_lhs_cube), mocker.call(self.m_rhs_cube)]
        assert patcher.call_args_list == calls

        assert resolve.lhs_cube_category == category_lhs
        assert resolve.rhs_cube_category == category_rhs

        items_dim = [category_lhs.items_dim[1]]
        items_aux = [category_lhs.items_aux[1]]
        items_scalar = [category_lhs.items_scalar[1]]
        expected = _CategoryItems(
            items_dim=items_dim, items_aux=items_aux, items_scalar=items_scalar
        )
        assert resolve.lhs_cube_category_local == expected

        items_dim = [category_rhs.items_dim[1]]
        items_aux = [category_rhs.items_aux[1]]
        items_scalar = [category_rhs.items_scalar[1]]
        expected = _CategoryItems(
            items_dim=items_dim, items_aux=items_aux, items_scalar=items_scalar
        )
        assert resolve.rhs_cube_category_local == expected

        items_dim = [category_lhs.items_dim[0]]
        items_aux = [category_lhs.items_aux[0]]
        items_scalar = [category_lhs.items_scalar[0]]
        expected = _CategoryItems(
            items_dim=items_dim, items_aux=items_aux, items_scalar=items_scalar
        )
        assert resolve.category_common == expected

    def test_metadata_different(self, mocker):
        # configure the lhs cube category
        category_lhs = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        # configure dim coords
        pairs = [
            (mocker.sentinel.dim_metadata1, mocker.sentinel.dims1),
            (mocker.sentinel.dim_metadata2, mocker.sentinel.dims2),
        ]
        category_lhs.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        pairs = [
            (mocker.sentinel.aux_metadata1, mocker.sentinel.dims3),
            (mocker.sentinel.aux_metadata2, mocker.sentinel.dims4),
        ]
        category_lhs.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        pairs = [
            (mocker.sentinel.scalar_metadata1, None),
            (mocker.sentinel.scalar_metadata2, None),
        ]
        category_lhs.items_scalar.extend(self._create_items(pairs))

        # configure the rhs cube category
        category_rhs = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        # configure dim coords
        pairs = [
            (mocker.sentinel.dim_metadata100, mocker.sentinel.dims1),
            (mocker.sentinel.dim_metadata200, mocker.sentinel.dims2),
        ]
        category_rhs.items_dim.extend(self._create_items(pairs))
        # configure aux coords
        pairs = [
            (mocker.sentinel.aux_metadata100, mocker.sentinel.dims3),
            (mocker.sentinel.aux_metadata200, mocker.sentinel.dims4),
        ]
        category_rhs.items_aux.extend(self._create_items(pairs))
        # configure scalar coords
        pairs = [
            (mocker.sentinel.scalar_metadata100, None),
            (mocker.sentinel.scalar_metadata200, None),
        ]
        category_rhs.items_scalar.extend(self._create_items(pairs))

        side_effect = (category_lhs, category_rhs)
        patcher = mocker.patch(self.target, side_effect=side_effect)

        resolve = Resolve()
        assert resolve.lhs_cube is None
        assert resolve.rhs_cube is None
        assert resolve.lhs_cube_category is None
        assert resolve.rhs_cube_category is None
        assert resolve.lhs_cube_category_local is None
        assert resolve.rhs_cube_category_local is None
        assert resolve.category_common is None

        # first require to explicitly lhs/rhs configure cubes
        resolve.lhs_cube = self.m_lhs_cube
        resolve.rhs_cube = self.m_rhs_cube
        resolve._metadata_resolve()

        assert patcher.call_count == 2
        calls = [mocker.call(self.m_lhs_cube), mocker.call(self.m_rhs_cube)]
        assert patcher.call_args_list == calls

        assert resolve.lhs_cube_category == category_lhs
        assert resolve.rhs_cube_category == category_rhs
        assert resolve.lhs_cube_category_local == category_lhs
        assert resolve.rhs_cube_category_local == category_rhs
        expected = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        assert resolve.category_common == expected


class Test__dim_coverage:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.ndim = 4
        self.cube = mocker.Mock(ndim=self.ndim)
        self.items = []
        parts = [
            (mocker.sentinel.metadata0, mocker.sentinel.coord0, (0,)),
            (mocker.sentinel.metadata1, mocker.sentinel.coord1, (1,)),
            (mocker.sentinel.metadata2, mocker.sentinel.coord2, (2,)),
            (mocker.sentinel.metadata3, mocker.sentinel.coord3, (3,)),
        ]
        column_parts = [x for x in zip(*parts)]
        self.metadata, self.coords, self.dims = [list(x) for x in column_parts]
        self.dims = [dim for (dim,) in self.dims]
        for metadata, coord, dims in parts:
            item = _Item(metadata=metadata, coord=coord, dims=dims)
            self.items.append(item)

    def test_coverage_no_local_no_common_all_free(self):
        items = []
        common = []
        result = Resolve._dim_coverage(self.cube, items, common)
        assert isinstance(result, _DimCoverage)
        assert result.cube == self.cube
        expected = [None] * self.ndim
        assert result.metadata == expected
        assert result.coords == expected
        assert result.dims_common == []
        assert result.dims_local == []
        expected = list(range(self.ndim))
        assert result.dims_free == expected

    def test_coverage_all_local_no_common_no_free(self):
        common = []
        result = Resolve._dim_coverage(self.cube, self.items, common)
        assert isinstance(result, _DimCoverage)
        assert result.cube == self.cube
        assert result.metadata == self.metadata
        assert result.coords == self.coords
        assert result.dims_common == []
        assert result.dims_local == self.dims
        assert result.dims_free == []

    def test_coverage_no_local_all_common_no_free(self):
        result = Resolve._dim_coverage(self.cube, self.items, self.metadata)
        assert isinstance(result, _DimCoverage)
        assert result.cube == self.cube
        assert result.metadata == self.metadata
        assert result.coords == self.coords
        assert result.dims_common == self.dims
        assert result.dims_local == []
        assert result.dims_free == []

    def test_coverage_mixed(self, mocker):
        common = [self.items[1].metadata, self.items[2].metadata]
        self.items.pop(0)
        self.items.pop(-1)
        metadata, coord, dims = (
            mocker.sentinel.metadata100,
            mocker.sentinel.coord100,
            (0,),
        )
        self.items.append(_Item(metadata=metadata, coord=coord, dims=dims))
        result = Resolve._dim_coverage(self.cube, self.items, common)
        assert isinstance(result, _DimCoverage)
        assert result.cube == self.cube
        expected = [
            metadata,
            self.items[0].metadata,
            self.items[1].metadata,
            None,
        ]
        assert result.metadata == expected
        expected = [coord, self.items[0].coord, self.items[1].coord, None]
        assert result.coords == expected
        assert result.dims_common == [1, 2]
        assert result.dims_local == [0]
        assert result.dims_free == [3]


class Test__aux_coverage:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.ndim = 4
        self.cube = mocker.Mock(ndim=self.ndim)
        # configure aux coords
        self.items_aux = []
        aux_parts = [
            (mocker.sentinel.aux_metadata0, mocker.sentinel.aux_coord0, (0,)),
            (mocker.sentinel.aux_metadata1, mocker.sentinel.aux_coord1, (1,)),
            (mocker.sentinel.aux_metadata23, mocker.sentinel.aux_coord23, (2, 3)),
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
            (mocker.sentinel.scalar_metadata0, mocker.sentinel.scalar_coord0, ()),
            (mocker.sentinel.scalar_metadata1, mocker.sentinel.scalar_coord1, ()),
            (mocker.sentinel.scalar_metadata2, mocker.sentinel.scalar_coord2, ()),
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
        assert isinstance(result, _AuxCoverage)
        assert result.cube == self.cube
        assert result.common_items_aux == []
        assert result.common_items_scalar == []
        assert result.local_items_aux == []
        assert result.local_items_scalar == []
        assert result.dims_common == []
        assert result.dims_local == []
        expected = list(range(self.ndim))
        assert result.dims_free == expected

    def test_coverage_all_local_no_common_no_free(self):
        common_aux, common_scalar = [], []
        result = Resolve._aux_coverage(
            self.cube,
            self.items_aux,
            self.items_scalar,
            common_aux,
            common_scalar,
        )
        assert isinstance(result, _AuxCoverage)
        assert result.cube == self.cube
        assert result.common_items_aux == []
        assert result.common_items_scalar == []
        assert result.local_items_aux == self.items_aux
        assert result.local_items_scalar == self.items_scalar
        assert result.dims_common == []
        expected = list(range(self.ndim))
        assert result.dims_local == expected
        assert result.dims_free == []

    def test_coverage_no_local_all_common_no_free(self):
        result = Resolve._aux_coverage(
            self.cube,
            self.items_aux,
            self.items_scalar,
            self.aux_metadata,
            self.scalar_metadata,
        )
        assert isinstance(result, _AuxCoverage)
        assert result.cube == self.cube
        assert result.common_items_aux == self.items_aux
        assert result.common_items_scalar == self.items_scalar
        assert result.local_items_aux == []
        assert result.local_items_scalar == []
        expected = list(range(self.ndim))
        assert result.dims_common == expected
        assert result.dims_local == []
        assert result.dims_free == []

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
        assert isinstance(result, _AuxCoverage)
        assert result.cube == self.cube
        expected = [self.items_aux[-1]]
        assert result.common_items_aux == expected
        expected = [self.items_scalar[1]]
        assert result.common_items_scalar == expected
        expected = [self.items_aux[0]]
        assert result.local_items_aux == expected
        expected = [self.items_scalar[0], self.items_scalar[2]]
        assert result.local_items_scalar == expected
        assert result.dims_common == [2, 3]
        assert result.dims_local == [0]
        assert result.dims_free == [1]


class Test__metadata_coverage:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.resolve = Resolve()
        self.m_lhs_cube = mocker.sentinel.lhs_cube
        self.resolve.lhs_cube = self.m_lhs_cube
        self.m_rhs_cube = mocker.sentinel.rhs_cube
        self.resolve.rhs_cube = self.m_rhs_cube
        self.m_items_dim_metadata = mocker.sentinel.items_dim_metadata
        self.m_items_aux_metadata = mocker.sentinel.items_aux_metadata
        self.m_items_scalar_metadata = mocker.sentinel.items_scalar_metadata
        items_dim = [mocker.Mock(metadata=self.m_items_dim_metadata)]
        items_aux = [mocker.Mock(metadata=self.m_items_aux_metadata)]
        items_scalar = [mocker.Mock(metadata=self.m_items_scalar_metadata)]
        category = _CategoryItems(
            items_dim=items_dim, items_aux=items_aux, items_scalar=items_scalar
        )
        self.resolve.category_common = category
        self.m_items_dim = mocker.sentinel.items_dim
        self.m_items_aux = mocker.sentinel.items_aux
        self.m_items_scalar = mocker.sentinel.items_scalar
        category = _CategoryItems(
            items_dim=self.m_items_dim,
            items_aux=self.m_items_aux,
            items_scalar=self.m_items_scalar,
        )
        self.resolve.lhs_cube_category = category
        self.resolve.rhs_cube_category = category
        target = "iris.common.resolve.Resolve._dim_coverage"
        self.m_lhs_cube_dim_coverage = mocker.sentinel.lhs_cube_dim_coverage
        self.m_rhs_cube_dim_coverage = mocker.sentinel.rhs_cube_dim_coverage
        side_effect = (
            self.m_lhs_cube_dim_coverage,
            self.m_rhs_cube_dim_coverage,
        )
        self.mocker_dim_coverage = mocker.patch(target, side_effect=side_effect)
        target = "iris.common.resolve.Resolve._aux_coverage"
        self.m_lhs_cube_aux_coverage = mocker.sentinel.lhs_cube_aux_coverage
        self.m_rhs_cube_aux_coverage = mocker.sentinel.rhs_cube_aux_coverage
        side_effect = (
            self.m_lhs_cube_aux_coverage,
            self.m_rhs_cube_aux_coverage,
        )
        self.mocker_aux_coverage = mocker.patch(target, side_effect=side_effect)

    def test(self, mocker):
        self.resolve._metadata_coverage()
        assert self.mocker_dim_coverage.call_count == 2
        calls = [
            mocker.call(self.m_lhs_cube, self.m_items_dim, [self.m_items_dim_metadata]),
            mocker.call(self.m_rhs_cube, self.m_items_dim, [self.m_items_dim_metadata]),
        ]
        assert self.mocker_dim_coverage.call_args_list == calls
        assert self.mocker_aux_coverage.call_count == 2
        calls = [
            mocker.call(
                self.m_lhs_cube,
                self.m_items_aux,
                self.m_items_scalar,
                [self.m_items_aux_metadata],
                [self.m_items_scalar_metadata],
            ),
            mocker.call(
                self.m_rhs_cube,
                self.m_items_aux,
                self.m_items_scalar,
                [self.m_items_aux_metadata],
                [self.m_items_scalar_metadata],
            ),
        ]
        assert self.mocker_aux_coverage.call_args_list == calls
        assert self.resolve.lhs_cube_dim_coverage == self.m_lhs_cube_dim_coverage
        assert self.resolve.rhs_cube_dim_coverage == self.m_rhs_cube_dim_coverage
        assert self.resolve.lhs_cube_aux_coverage == self.m_lhs_cube_aux_coverage
        assert self.resolve.rhs_cube_aux_coverage == self.m_rhs_cube_aux_coverage


class Test__dim_mapping:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.ndim = 3
        Wrapper = namedtuple("Wrapper", ("name",))
        cube = Wrapper(name=lambda: mocker.sentinel.name)
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
            mocker.sentinel.metadata_0,
            mocker.sentinel.metadata_1,
            mocker.sentinel.metadata_2,
        ]
        self.dummy = [
            mocker.sentinel.dummy_0,
            mocker.sentinel.dummy_1,
            mocker.sentinel.dummy_2,
        ]

    def test_no_mapping(self):
        self.src_coverage.metadata.extend(self.metadata)
        self.tgt_coverage.metadata.extend(self.dummy)
        result = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)
        assert result == {}

    def test_full_mapping(self):
        self.src_coverage.metadata.extend(self.metadata)
        self.tgt_coverage.metadata.extend(self.metadata)
        dims_common = list(range(self.ndim))
        self.tgt_coverage.dims_common.extend(dims_common)
        result = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 0, 1: 1, 2: 2}
        assert result == expected

    def test_transpose_mapping(self):
        self.src_coverage.metadata.extend(self.metadata[::-1])
        self.tgt_coverage.metadata.extend(self.metadata)
        dims_common = list(range(self.ndim))
        self.tgt_coverage.dims_common.extend(dims_common)
        result = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 2, 1: 1, 2: 0}
        assert result == expected

    def test_partial_mapping__transposed(self, mocker):
        self.src_coverage.metadata.extend(self.metadata)
        self.metadata[1] = mocker.sentinel.nope
        self.tgt_coverage.metadata.extend(self.metadata[::-1])
        dims_common = [0, 2]
        self.tgt_coverage.dims_common.extend(dims_common)
        result = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 2, 2: 0}
        assert result == expected

    def test_bad_metadata_mapping(self, mocker):
        self.src_coverage.metadata.extend(self.metadata)
        self.metadata[0] = mocker.sentinel.bad
        self.tgt_coverage.metadata.extend(self.metadata)
        dims_common = [0]
        self.tgt_coverage.dims_common.extend(dims_common)
        emsg = "Failed to map common dim coordinate metadata"
        with pytest.raises(ValueError, match=emsg):
            _ = Resolve._dim_mapping(self.src_coverage, self.tgt_coverage)


class Test__aux_mapping:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.ndim = 3
        Wrapper = namedtuple("Wrapper", ("name",))
        cube = Wrapper(name=lambda: mocker.sentinel.name)
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
                metadata=mocker.sentinel.metadata0,
                coord=mocker.sentinel.coord0,
                dims=[0],
            ),
            _Item(
                metadata=mocker.sentinel.metadata1,
                coord=mocker.sentinel.coord1,
                dims=[1],
            ),
            _Item(
                metadata=mocker.sentinel.metadata2,
                coord=mocker.sentinel.coord2,
                dims=[2],
            ),
        ]

    def test_no_mapping(self):
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        assert result == {}

    def test_full_mapping(self):
        self.src_coverage.common_items_aux.extend(self.items)
        self.tgt_coverage.common_items_aux.extend(self.items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 0, 1: 1, 2: 2}
        assert result == expected

    def test_transpose_mapping(self):
        self.src_coverage.common_items_aux.extend(self.items)
        items = deepcopy(self.items)
        items[0].dims[0] = 2
        items[2].dims[0] = 0
        self.tgt_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 2, 1: 1, 2: 0}
        assert result == expected

    def test_partial_mapping__transposed(self):
        _ = self.items.pop(1)
        self.src_coverage.common_items_aux.extend(self.items)
        items = deepcopy(self.items)
        items[0].dims[0] = 2
        items[1].dims[0] = 0
        self.tgt_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 2, 2: 0}
        assert result == expected

    def test_mapping__match_multiple_src_metadata(self):
        items = deepcopy(self.items)
        _ = self.items.pop(1)
        self.src_coverage.common_items_aux.extend(self.items)
        items[1] = items[0]
        self.tgt_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 0, 2: 2}
        assert result == expected

    def test_mapping__skip_match_multiple_src_metadata(self):
        items = deepcopy(self.items)
        _ = self.items.pop(1)
        self.tgt_coverage.common_items_aux.extend(self.items)
        items[1] = items[0]._replace(dims=[1])
        self.src_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {2: 2}
        assert result == expected

    def test_mapping__skip_different_rank(self):
        items = deepcopy(self.items)
        self.src_coverage.common_items_aux.extend(self.items)
        items[2] = items[2]._replace(dims=[1, 2])
        self.tgt_coverage.common_items_aux.extend(items)
        result = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)
        expected = {0: 0, 1: 1}
        assert result == expected

    def test_bad_metadata_mapping(self, mocker):
        self.src_coverage.common_items_aux.extend(self.items)
        items = deepcopy(self.items)
        items[0] = items[0]._replace(metadata=mocker.sentinel.bad)
        self.tgt_coverage.common_items_aux.extend(items)
        emsg = "Failed to map common aux coordinate metadata"
        with pytest.raises(ValueError, match=emsg):
            _ = Resolve._aux_mapping(self.src_coverage, self.tgt_coverage)


class Test_mapped:
    def test_mapping_none(self):
        resolve = Resolve()
        assert resolve.mapping is None
        assert resolve.mapped is None

    def test_mapped__src_cube_lhs(self, mocker):
        resolve = Resolve()
        lhs = mocker.Mock(ndim=2)
        rhs = mocker.Mock(ndim=3)
        resolve.lhs_cube = lhs
        resolve.rhs_cube = rhs
        resolve.map_rhs_to_lhs = False
        resolve.mapping = {0: 0, 1: 1}
        assert resolve.mapped

    def test_mapped__src_cube_rhs(self, mocker):
        resolve = Resolve()
        lhs = mocker.Mock(ndim=3)
        rhs = mocker.Mock(ndim=2)
        resolve.lhs_cube = lhs
        resolve.rhs_cube = rhs
        resolve.map_rhs_to_lhs = True
        resolve.mapping = {0: 0, 1: 1}
        assert resolve.mapped

    def test_partial_mapping(self, mocker):
        resolve = Resolve()
        lhs = mocker.Mock(ndim=3)
        rhs = mocker.Mock(ndim=2)
        resolve.lhs_cube = lhs
        resolve.rhs_cube = rhs
        resolve.map_rhs_to_lhs = True
        resolve.mapping = {0: 0}
        assert not resolve.mapped


class Test__free_mapping:
    @pytest.fixture(autouse=True)
    def _setup(self):
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
        with pytest.raises(ValueError, match=emsg):
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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        with pytest.raises(ValueError, match=emsg):
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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        assert self.resolve.mapping == expected

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
        with pytest.raises(ValueError, match=emsg):
            self.resolve._free_mapping(**args)


class Test__src_cube:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.resolve = Resolve()
        self.expected = mocker.sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve.rhs_cube = self.expected
        assert self.resolve._src_cube == self.expected

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve.lhs_cube = self.expected
        assert self.resolve._src_cube == self.expected

    def test_fail__no_map_rhs_to_lhs(self):
        with pytest.raises(AssertionError):
            self.resolve._src_cube


class Test__src_cube_position:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.resolve = Resolve()

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        assert self.resolve._src_cube_position == "RHS"

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        assert self.resolve._src_cube_position == "LHS"

    def test_fail__no_map_rhs_to_lhs(self):
        with pytest.raises(AssertionError):
            self.resolve._src_cube_position


class Test__src_cube_resolved__getter:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.resolve = Resolve()
        self.expected = mocker.sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve.rhs_cube_resolved = self.expected
        assert self.resolve._src_cube_resolved == self.expected

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve.lhs_cube_resolved = self.expected
        assert self.resolve._src_cube_resolved == self.expected

    def test_fail__no_map_rhs_to_lhs(self):
        with pytest.raises(AssertionError):
            self.resolve._src_cube_resolved


class Test__src_cube_resolved__setter:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.resolve = Resolve()
        self.expected = mocker.sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve._src_cube_resolved = self.expected
        assert self.resolve.rhs_cube_resolved == self.expected

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve._src_cube_resolved = self.expected
        assert self.resolve.lhs_cube_resolved == self.expected

    def test_fail__no_map_rhs_to_lhs(self):
        with pytest.raises(AssertionError):
            self.resolve._src_cube_resolved = self.expected


class Test__tgt_cube:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.resolve = Resolve()
        self.expected = mocker.sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve.rhs_cube = self.expected
        assert self.resolve._tgt_cube == self.expected

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve.lhs_cube = self.expected
        assert self.resolve._tgt_cube == self.expected

    def test_fail__no_map_rhs_to_lhs(self):
        with pytest.raises(AssertionError):
            self.resolve._tgt_cube


class Test__tgt_cube_position:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.resolve = Resolve()

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        assert self.resolve._tgt_cube_position == "RHS"

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        assert self.resolve._tgt_cube_position == "LHS"

    def test_fail__no_map_rhs_to_lhs(self):
        with pytest.raises(AssertionError):
            self.resolve._tgt_cube_position


class Test__tgt_cube_resolved__getter:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.resolve = Resolve()
        self.expected = mocker.sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve.rhs_cube_resolved = self.expected
        assert self.resolve._tgt_cube_resolved == self.expected

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve.lhs_cube_resolved = self.expected
        assert self.resolve._tgt_cube_resolved == self.expected

    def test_fail__no_map_rhs_to_lhs(self):
        with pytest.raises(AssertionError):
            self.resolve._tgt_cube_resolved


class Test__tgt_cube_resolved__setter:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.resolve = Resolve()
        self.expected = mocker.sentinel.cube

    def test_rhs_cube(self):
        self.resolve.map_rhs_to_lhs = False
        self.resolve._tgt_cube_resolved = self.expected
        assert self.resolve.rhs_cube_resolved == self.expected

    def test_lhs_cube(self):
        self.resolve.map_rhs_to_lhs = True
        self.resolve._tgt_cube_resolved = self.expected
        assert self.resolve.lhs_cube_resolved == self.expected

    def test_fail__no_map_rhs_to_lhs(self):
        with pytest.raises(AssertionError):
            self.resolve._tgt_cube_resolved = self.expected


class Test_shape:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.resolve = Resolve()

    def test_no_shape(self):
        assert self.resolve.shape is None

    def test_shape(self, mocker):
        expected = mocker.sentinel.shape
        self.resolve._broadcast_shape = expected
        assert self.resolve.shape == expected


class Test__as_compatible_cubes:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
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
        self.mocker = mocker.patch("iris.cube.Cube")
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
            self.args["metadata"] = mock.sentinel.metadata
            self.reshape = mock.sentinel.reshape
            m_reshape = mock.Mock(return_value=self.reshape)
            self.transpose = mock.Mock(shape=transpose_shape, reshape=m_reshape)
            m_transpose = mock.Mock(return_value=self.transpose)
            self.data = mock.Mock(shape=shape, transpose=m_transpose, reshape=m_reshape)
            m_copy = mock.Mock(return_value=self.data)
            m_core_data = mock.Mock(copy=m_copy)
            self.args["core_data"] = mock.Mock(return_value=m_core_data)
            self.args["coord_dims"] = mock.Mock(side_effect=([0], [ndim - 1]))
            self.dim_coord = mock.sentinel.dim_coord
            self.aux_coord = mock.sentinel.aux_coord
            self.aux_factory = mock.sentinel.aux_factory
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
        with pytest.raises(AssertionError):
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
        with pytest.raises(ValueError, match=emsg):
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
        with pytest.raises(ValueError, match=emsg):
            self.resolve._as_compatible_cubes()

    def _check_compatible(self, broadcast_shape):
        assert self.resolve.lhs_cube == self.resolve._tgt_cube_resolved
        assert self.resolve._src_cube_resolved == self.cube
        assert self.resolve._broadcast_shape == broadcast_shape
        assert self.mocker.call_count == 1
        assert self.args["metadata"] == self.cube.metadata
        assert self.resolve.rhs_cube.coord_dims.call_count == 2
        assert self.resolve.rhs_cube.coord_dims.call_args_list == [
            mock.call(self.dim_coord),
            mock.call(self.aux_coord),
        ]
        assert self.cube.add_dim_coord.call_count == 1
        assert self.cube.add_dim_coord.call_args_list == [
            mock.call(self.dim_coord, [self.resolve.mapping[0]])
        ]
        assert self.cube.add_aux_coord.call_count == 1
        assert self.cube.add_aux_coord.call_args_list == [
            mock.call(self.aux_coord, [self.resolve.mapping[2]])
        ]
        assert self.cube.add_aux_factory.call_count == 1
        assert self.cube.add_aux_factory.call_args_list == [mock.call(self.aux_factory)]

    def test_compatible(self, mocker):
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
        assert self.mocker.call_args_list == [mocker.call(self.data)]

    def test_compatible__transpose(self, mocker):
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
        assert self.data.transpose.call_count == 1
        assert self.data.transpose.call_args_list == [mocker.call([2, 1, 0])]
        assert self.mocker.call_args_list == [mocker.call(self.transpose)]

    def test_compatible__reshape(self, mocker):
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
        assert self.data.reshape.call_count == 1
        assert self.data.reshape.call_args_list == [mocker.call((1,) + src_shape)]
        assert self.mocker.call_args_list == [mocker.call(self.reshape)]

    def test_compatible__transpose_reshape(self, mocker):
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
        assert self.data.transpose.call_count == 1
        assert self.data.transpose.call_args_list == [mocker.call([2, 1, 0])]
        assert self.data.reshape.call_count == 1
        assert self.data.reshape.call_args_list == [mocker.call((1,) + transpose_shape)]
        assert self.mocker.call_args_list == [mocker.call(self.reshape)]

    def test_compatible__broadcast(self, mocker):
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
        assert self.mocker.call_args_list == [mocker.call(self.data)]

    def test_compatible__broadcast_transpose_reshape(self, mocker):
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
        assert self.data.transpose.call_count == 1
        assert self.data.transpose.call_args_list == [mocker.call([2, 1, 0])]
        assert self.data.reshape.call_count == 1
        assert self.data.reshape.call_args_list == [mocker.call((1,) + transpose_shape)]
        assert self.mocker.call_args_list == [mocker.call(self.reshape)]


class Test__metadata_mapping:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.ndim = mocker.sentinel.ndim
        self.src_cube = mocker.Mock(ndim=self.ndim)
        self.src_dim_coverage = mocker.Mock(dims_free=[])
        self.src_aux_coverage = mocker.Mock(dims_free=[])
        self.tgt_cube = mocker.Mock(ndim=self.ndim)
        self.tgt_dim_coverage = mocker.Mock(dims_free=[])
        self.tgt_aux_coverage = mocker.Mock(dims_free=[])
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
        self.shape = mocker.sentinel.shape
        self.resolve._broadcast_shape = self.shape
        self.resolve._src_cube_resolved = mocker.Mock(shape=self.shape)
        self.resolve._tgt_cube_resolved = mocker.Mock(shape=self.shape)
        self.m_dim_mapping = mocker.patch(
            "iris.common.resolve.Resolve._dim_mapping", return_value={}
        )
        self.m_aux_mapping = mocker.patch(
            "iris.common.resolve.Resolve._aux_mapping", return_value={}
        )
        self.m_free_mapping = mocker.patch("iris.common.resolve.Resolve._free_mapping")
        self.m_as_compatible_cubes = mocker.patch(
            "iris.common.resolve.Resolve._as_compatible_cubes"
        )
        self.mapping = {0: 1, 1: 2, 2: 3}

    def test_mapped__dim_coords(self, mocker):
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
        assert self.resolve.mapping == self.mapping
        assert self.m_dim_mapping.call_count == 1
        expected = [mocker.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        assert self.m_dim_mapping.call_args_list == expected
        assert self.m_aux_mapping.call_count == 0
        assert self.m_free_mapping.call_count == 0
        assert self.m_as_compatible_cubes.call_count == 1

    def test_mapped__aux_coords(self, mocker):
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
        assert self.resolve.mapping == self.mapping
        assert self.m_dim_mapping.call_count == 1
        expected = [mocker.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        assert self.m_dim_mapping.call_args_list == expected
        assert self.m_aux_mapping.call_count == 1
        expected = [mocker.call(self.src_aux_coverage, self.tgt_aux_coverage)]
        assert self.m_aux_mapping.call_args_list == expected
        assert self.m_free_mapping.call_count == 0
        assert self.m_as_compatible_cubes.call_count == 1

    def test_mapped__dim_and_aux_coords(self, mocker):
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
        assert self.resolve.mapping == self.mapping
        assert self.m_dim_mapping.call_count == 1
        expected = [mocker.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        assert self.m_dim_mapping.call_args_list == expected
        assert self.m_aux_mapping.call_count == 1
        expected = [mocker.call(self.src_aux_coverage, self.tgt_aux_coverage)]
        assert self.m_aux_mapping.call_args_list == expected
        assert self.m_free_mapping.call_count == 0
        assert self.m_as_compatible_cubes.call_count == 1

    def test_mapped__dim_coords_and_free_dims(self, mocker):
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
        side_effect = lambda a, b, c, d: self.resolve.mapping.update(free_mapping)
        self.m_free_mapping.side_effect = side_effect
        self.resolve._metadata_mapping()
        assert self.resolve.mapping == self.mapping
        assert self.m_dim_mapping.call_count == 1
        expected = [mocker.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        assert self.m_dim_mapping.call_args_list == expected
        assert self.m_aux_mapping.call_count == 1
        expected = [mocker.call(self.src_aux_coverage, self.tgt_aux_coverage)]
        assert self.m_aux_mapping.call_args_list == expected
        assert self.m_free_mapping.call_count == 1
        expected = [
            mocker.call(
                self.src_dim_coverage,
                self.tgt_dim_coverage,
                self.src_aux_coverage,
                self.tgt_aux_coverage,
            )
        ]
        assert self.m_free_mapping.call_args_list == expected
        assert self.m_as_compatible_cubes.call_count == 1

    def test_mapped__dim_coords_with_broadcast_flip(self, mocker):
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
        assert self.resolve.mapping == mapping
        assert self.m_dim_mapping.call_count == 1
        expected = [mocker.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        assert self.m_dim_mapping.call_args_list == expected
        assert self.m_aux_mapping.call_count == 0
        assert self.m_free_mapping.call_count == 0
        assert self.m_as_compatible_cubes.call_count == 2
        assert self.resolve.map_rhs_to_lhs != self.map_rhs_to_lhs

    def test_mapped__dim_coords_free_flip_with_free_flip(self, mocker):
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
        side_effect = lambda a, b, c, d: self.resolve.mapping.update(free_mapping)
        self.m_free_mapping.side_effect = side_effect
        self.tgt_dim_coverage.dims_free = [0, 1]
        self.tgt_aux_coverage.dims_free = [0, 1]
        self.resolve._metadata_mapping()
        assert self.resolve.mapping == mapping
        assert self.m_dim_mapping.call_count == 1
        expected = [mocker.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        assert self.m_dim_mapping.call_args_list == expected
        assert self.m_aux_mapping.call_count == 1
        expected = [mocker.call(self.src_aux_coverage, self.tgt_aux_coverage)]
        assert self.m_aux_mapping.call_args_list == expected
        assert self.m_free_mapping.call_count == 1
        expected = [
            mocker.call(
                self.src_dim_coverage,
                self.tgt_dim_coverage,
                self.src_aux_coverage,
                self.tgt_aux_coverage,
            )
        ]
        assert self.m_free_mapping.call_args_list == expected
        assert self.m_as_compatible_cubes.call_count == 2


class Test__prepare_common_dim_payload:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
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
        self.points = (
            mocker.sentinel.points_0,
            mocker.sentinel.points_1,
            mocker.sentinel.points_2,
            mocker.sentinel.points_3,
        )
        self.bounds = (
            mocker.sentinel.bounds_0,
            mocker.sentinel.bounds_1,
            mocker.sentinel.bounds_2,
        )
        self.pb_0 = (
            mocker.Mock(copy=mocker.Mock(return_value=self.points[0])),
            mocker.Mock(copy=mocker.Mock(return_value=self.bounds[0])),
        )
        self.pb_1 = (
            mocker.Mock(copy=mocker.Mock(return_value=self.points[1])),
            None,
        )
        self.pb_2 = (
            mocker.Mock(copy=mocker.Mock(return_value=self.points[2])),
            mocker.Mock(copy=mocker.Mock(return_value=self.bounds[2])),
        )
        side_effect = (self.pb_0, self.pb_1, self.pb_2)
        self.m_prepare_points_and_bounds = mocker.patch(
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
            mocker.sentinel.combined_0,
            mocker.sentinel.combined_1,
            mocker.sentinel.combined_2,
        )
        self.src_metadata = mocker.Mock(
            combine=mocker.Mock(side_effect=self.metadata_combined)
        )
        metadata = [self.src_metadata] * len(self.mapping)
        self.src_coords = [
            # N.B. these need to mimic a Coord with points and bounds, and
            # be of a class which is not-a-MeshCoord.
            # NOTE: strictly, bounds should =above values, and support .copy().
            # For these tests, just omitting them works + is simpler.
            mocker.Mock(spec=DimCoord, points=self.points[0], bounds=None),
            mocker.Mock(spec=DimCoord, points=self.points[1], bounds=None),
            mocker.Mock(spec=DimCoord, points=self.points[2], bounds=None),
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
            mocker.sentinel.tgt_metadata_0,
            mocker.sentinel.tgt_metadata_1,
            mocker.sentinel.tgt_metadata_2,
            mocker.sentinel.tgt_metadata_3,
        ]
        self.tgt_coords = [
            # N.B. these need to mimic a Coord with points and bounds, and
            # be of a class which is not-a-MeshCoord.
            # NOTE: strictly, bounds should =above values, and support .copy().
            # For these tests, just omitting them works + is simpler.
            mocker.Mock(spec=DimCoord, points=self.points[0], bounds=None),
            mocker.Mock(spec=DimCoord, points=self.points[1], bounds=None),
            mocker.Mock(spec=DimCoord, points=self.points[2], bounds=None),
            mocker.Mock(spec=DimCoord, points=self.points[3], bounds=None),
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
        assert len(self.resolve.prepared_category.items_aux) == 0
        assert len(self.resolve.prepared_category.items_scalar) == 0
        if not bad_points:
            assert len(self.resolve.prepared_category.items_dim) == 3
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
            assert self.resolve.prepared_category.items_dim == expected
        else:
            assert len(self.resolve.prepared_category.items_dim) == 0
        assert self.m_prepare_points_and_bounds.call_count == 3
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
        assert self.m_prepare_points_and_bounds.call_args_list == expected
        if not bad_points:
            assert self.src_metadata.combine.call_count == 3
            expected = [mock.call(metadata) for metadata in self.tgt_metadata[1:]]
            assert self.src_metadata.combine.call_args_list == expected

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


class Test__prepare_common_aux_payload:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
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
        self.points = (
            mocker.sentinel.points_0,
            mocker.sentinel.points_1,
            mocker.sentinel.points_2,
            mocker.sentinel.points_3,
        )
        self.bounds = (
            mocker.sentinel.bounds_0,
            mocker.sentinel.bounds_1,
            mocker.sentinel.bounds_2,
        )
        self.pb_0 = (
            mocker.Mock(copy=mocker.Mock(return_value=self.points[0])),
            mocker.Mock(copy=mocker.Mock(return_value=self.bounds[0])),
        )
        self.pb_1 = (
            mocker.Mock(copy=mocker.Mock(return_value=self.points[1])),
            None,
        )
        self.pb_2 = (
            mocker.Mock(copy=mocker.Mock(return_value=self.points[2])),
            mocker.Mock(copy=mocker.Mock(return_value=self.bounds[2])),
        )
        side_effect = (self.pb_0, self.pb_1, self.pb_2)
        self.m_prepare_points_and_bounds = mocker.patch(
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
            mocker.sentinel.combined_0,
            mocker.sentinel.combined_1,
            mocker.sentinel.combined_2,
        )
        self.src_metadata = [
            mocker.Mock(combine=mocker.Mock(return_value=self.metadata_combined[0])),
            mocker.Mock(combine=mocker.Mock(return_value=self.metadata_combined[1])),
            mocker.Mock(combine=mocker.Mock(return_value=self.metadata_combined[2])),
        ]
        self.src_coords = [
            # N.B. these need to mimic a Coord with points and bounds, but also
            # the type() defines the 'container' property of a prepared item.
            # It seems that 'type()' is not fake-able in Python, so we need to
            # provide *real* DimCoords, to match "self.container" below.
            DimCoord(points=[0], bounds=None),
            DimCoord(points=[1], bounds=None),
            DimCoord(points=[2], bounds=None),
        ]
        self.src_dims = [(dim,) for dim in self.mapping.keys()]
        self.src_common_items = [
            _Item(*item)
            for item in zip(self.src_metadata, self.src_coords, self.src_dims)
        ]
        self.tgt_metadata = [mocker.sentinel.tgt_metadata_0] + self.src_metadata
        self.tgt_coords = [
            # N.B. these need to mimic a Coord with points and bounds, but also
            # the type() defines the 'container' property of a prepared item.
            # It seems that 'type()' is not fake-able in Python, so we need to
            # provide *real* DimCoords, to match "self.container" below.
            DimCoord(points=[0], bounds=None),
            DimCoord(points=[1], bounds=None),
            DimCoord(points=[2], bounds=None),
            DimCoord(points=[3], bounds=None),
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
            assert len(prepared_items) == 3
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
            assert prepared_items == expected
        else:
            assert len(prepared_items) == 0
        assert self.m_prepare_points_and_bounds.call_count == 3
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
        assert self.m_prepare_points_and_bounds.call_args_list == expected
        if not bad_points:
            for src_metadata, tgt_metadata in zip(
                self.src_metadata, self.tgt_metadata[1:]
            ):
                assert src_metadata.combine.call_count == 1
                expected = [mock.call(tgt_metadata)]
                assert src_metadata.combine.call_args_list == expected

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
        assert len(prepared_items) == 0

    def test__multi_tgt_metadata_match(self):
        item = self.tgt_common_items[1]
        tgt_common_items = [item] * len(self.tgt_common_items)
        prepared_items = []
        self.resolve._prepare_common_aux_payload(
            self.src_common_items, tgt_common_items, prepared_items
        )
        assert len(prepared_items) == 0


class Test__prepare_points_and_bounds:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
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
        self.src_name = mocker.sentinel.src_name
        self.src_points = mocker.sentinel.src_points
        self.src_bounds = mocker.sentinel.src_bounds
        self.src_metadata = mocker.sentinel.src_metadata
        self.src_items = dict(
            name=lambda: self.src_name,
            points=self.src_points,
            bounds=self.src_bounds,
            metadata=self.src_metadata,
            ndim=None,
            shape=None,
            has_bounds=None,
        )
        self.tgt_name = mocker.sentinel.tgt_name
        self.tgt_points = mocker.sentinel.tgt_points
        self.tgt_bounds = mocker.sentinel.tgt_bounds
        self.tgt_metadata = mocker.sentinel.tgt_metadata
        self.tgt_items = dict(
            name=lambda: self.tgt_name,
            points=self.tgt_points,
            bounds=self.tgt_bounds,
            metadata=self.tgt_metadata,
            ndim=None,
            shape=None,
            has_bounds=None,
        )
        self.m_array_equal = mocker.patch(
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
        assert points == self.tgt_points
        assert bounds == self.tgt_bounds

    def test_coord_ndim_unequal__src_ndim_greater(self):
        self.src_items["ndim"] = 10
        src_coord = self.Coord(**self.src_items)
        self.tgt_items["ndim"] = 1
        tgt_coord = self.Coord(**self.tgt_items)
        points, bounds = self.resolve._prepare_points_and_bounds(
            src_coord, tgt_coord, src_dims=None, tgt_dims=None
        )
        assert points == self.src_points
        assert bounds == self.src_bounds

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
        assert points == self.tgt_points
        assert bounds == self.tgt_bounds

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
        assert points == self.src_points
        assert bounds == self.src_bounds

    def test_coord_ndim_equal__shape_unequal_with_unsupported_broadcasting(
        self,
        mocker,
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
            name=lambda: mocker.sentinel.src_cube, shape=src_shape
        )
        self.src_items["ndim"] = ndim
        self.src_items["shape"] = src_shape
        src_coord = self.Coord(**self.src_items)
        tgt_shape = (1, 9)
        tgt_dims = tuple(mapping.values())
        self.resolve.lhs_cube = self.Cube(
            name=lambda: mocker.sentinel.tgt_cube, shape=tgt_shape
        )
        self.tgt_items["ndim"] = ndim
        self.tgt_items["shape"] = tgt_shape
        tgt_coord = self.Coord(**self.tgt_items)
        emsg = "Cannot broadcast"
        with pytest.raises(ValueError, match=emsg):
            _ = self.resolve._prepare_points_and_bounds(
                src_coord, tgt_coord, src_dims, tgt_dims
            )

    def _populate(self, src_points, tgt_points, src_bounds=None, tgt_bounds=None):
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
            name=lambda: mock.sentinel.src_cube, shape=None
        )
        self.resolve.lhs_cube = self.Cube(
            name=lambda: mock.sentinel.tgt_cube, shape=None
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

    def test_coord_ndim_and_shape_equal__points_equal_with_no_bounds(self, mocker):
        args = self._populate(self.src_points, self.src_points)
        points, bounds = self.resolve._prepare_points_and_bounds(**args)
        assert points == self.src_points
        assert bounds is None
        assert self.m_array_equal.call_count == 1
        expected = [mocker.call(self.src_points, self.src_points, withnans=True)]
        assert self.m_array_equal.call_args_list == expected

    def test_coord_ndim_and_shape_equal__points_equal_with_src_bounds_only(
        self,
        mocker,
    ):
        args = self._populate(
            self.src_points, self.src_points, src_bounds=self.src_bounds
        )
        points, bounds = self.resolve._prepare_points_and_bounds(**args)
        assert points == self.src_points
        assert bounds == self.src_bounds
        assert self.m_array_equal.call_count == 1
        expected = [mocker.call(self.src_points, self.src_points, withnans=True)]
        assert self.m_array_equal.call_args_list == expected

    def test_coord_ndim_and_shape_equal__points_equal_with_tgt_bounds_only(
        self,
        mocker,
    ):
        args = self._populate(
            self.src_points, self.src_points, tgt_bounds=self.tgt_bounds
        )
        points, bounds = self.resolve._prepare_points_and_bounds(**args)
        assert points == self.src_points
        assert bounds == self.tgt_bounds
        assert self.m_array_equal.call_count == 1
        expected = [mocker.call(self.src_points, self.src_points, withnans=True)]
        assert self.m_array_equal.call_args_list == expected

    def test_coord_ndim_and_shape_equal__points_equal_with_src_bounds_only_strict(
        self,
    ):
        args = self._populate(
            self.src_points, self.src_points, src_bounds=self.src_bounds
        )
        with LENIENT.context(maths=False):
            emsg = f"Coordinate {self.src_name} has bounds"
            with pytest.raises(ValueError, match=emsg):
                _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_equal_with_tgt_bounds_only_strict(
        self,
    ):
        args = self._populate(
            self.src_points, self.src_points, tgt_bounds=self.tgt_bounds
        )
        with LENIENT.context(maths=False):
            emsg = f"Coordinate {self.tgt_name} has bounds"
            with pytest.raises(ValueError, match=emsg):
                _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_equal_with_bounds_equal(self, mocker):
        args = self._populate(
            self.src_points,
            self.src_points,
            src_bounds=self.src_bounds,
            tgt_bounds=self.src_bounds,
        )
        points, bounds = self.resolve._prepare_points_and_bounds(**args)
        assert points == self.src_points
        assert bounds == self.src_bounds
        assert self.m_array_equal.call_count == 2
        expected = [
            mocker.call(self.src_points, self.src_points, withnans=True),
            mocker.call(self.src_bounds, self.src_bounds, withnans=True),
        ]
        assert self.m_array_equal.call_args_list == expected

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
        with pytest.raises(ValueError, match=emsg):
            _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_equal_with_bounds_different_ignore_mismatch(
        self,
        mocker,
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
        assert points == self.src_points
        assert bounds is None
        assert self.m_array_equal.call_count == 2
        expected = [
            mocker.call(self.src_points, self.src_points, withnans=True),
            mocker.call(self.src_bounds, self.tgt_bounds, withnans=True),
        ]
        assert self.m_array_equal.call_args_list == expected

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
            with pytest.raises(ValueError, match=emsg):
                _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_different(self):
        self.m_array_equal.side_effect = (False,)
        args = self._populate(self.src_points, self.tgt_points)
        emsg = f"Coordinate {self.src_name} has different points"
        with pytest.raises(ValueError, match=emsg):
            _ = self.resolve._prepare_points_and_bounds(**args)

    def test_coord_ndim_and_shape_equal__points_different_ignore_mismatch(
        self,
    ):
        self.m_array_equal.side_effect = (False,)
        args = self._populate(self.src_points, self.tgt_points)
        points, bounds = self.resolve._prepare_points_and_bounds(
            **args, ignore_mismatch=True
        )
        assert points is None
        assert bounds is None

    def test_coord_ndim_and_shape_equal__points_different_strict(self):
        self.m_array_equal.side_effect = (False,)
        args = self._populate(self.src_points, self.tgt_points)
        with LENIENT.context(maths=False):
            emsg = f"Coordinate {self.src_name} has different points"
            with pytest.raises(ValueError, match=emsg):
                _ = self.resolve._prepare_points_and_bounds(**args)


class Test__create_prepared_item:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        Coord = namedtuple("Coord", ["points", "bounds"])
        self.points_value = mocker.sentinel.points
        self.points = mocker.Mock(copy=mocker.Mock(return_value=self.points_value))
        self.bounds_value = mocker.sentinel.bounds
        self.bounds = mocker.Mock(copy=mocker.Mock(return_value=self.bounds_value))
        self.coord = Coord(points=self.points, bounds=self.bounds)
        self.container = type(self.coord)
        self.combined = mocker.sentinel.combined
        self.src = mocker.Mock(combine=mocker.Mock(return_value=self.combined))
        self.tgt = mocker.sentinel.tgt

    def _check(self, src=None, tgt=None):
        dims = 0
        if src is not None and tgt is not None:
            combined = self.combined
        else:
            combined = src or tgt
        result = Resolve._create_prepared_item(
            self.coord, dims, src_metadata=src, tgt_metadata=tgt
        )
        assert isinstance(result, _PreparedItem)
        assert isinstance(result.metadata, _PreparedMetadata)
        expected = _PreparedMetadata(combined=combined, src=src, tgt=tgt)
        assert result.metadata == expected
        assert result.points == self.points_value
        assert self.points.copy.call_count == 1
        assert self.points.copy.call_args_list == [mock.call()]
        assert result.bounds == self.bounds_value
        assert self.bounds.copy.call_count == 1
        assert self.bounds.copy.call_args_list == [mock.call()]
        assert result.dims == (dims,)
        assert result.container == self.container

    def test__no_metadata(self):
        self._check()

    def test__src_metadata_only(self):
        self._check(src=self.src)

    def test__tgt_metadata_only(self):
        self._check(tgt=self.tgt)

    def test__combine_metadata(self):
        self._check(src=self.src, tgt=self.tgt)


class Test__prepare_local_payload_dim:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
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
        self.prepared_item = mocker.sentinel.prepared_item
        self.m_create_prepared_item = mocker.patch(
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
        assert len(self.resolve.prepared_category.items_dim) == 0

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
        assert len(self.resolve.prepared_category.items_dim) == 0

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
        assert len(self.resolve.prepared_category.items_dim) == 0

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
        assert len(self.resolve.prepared_category.items_dim) == 0

    def test_src_local_with_tgt_free(self, mocker):
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
        src_metadata = mocker.sentinel.src_metadata
        self.src_coverage["metadata"] = [None, src_metadata]
        src_coord = mocker.sentinel.src_coord
        self.src_coverage["coords"] = [None, src_coord]
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_dim) == 1
        assert self.resolve.prepared_category.items_dim[0] == self.prepared_item
        assert self.m_create_prepared_item.call_count == 1
        expected = [mocker.call(src_coord, mapping[src_dim], src_metadata=src_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_local_with_tgt_free__strict(self, mocker):
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
        src_metadata = mocker.sentinel.src_metadata
        self.src_coverage["metadata"] = [None, src_metadata]
        src_coord = mocker.sentinel.src_coord
        self.src_coverage["coords"] = [None, src_coord]
        src_coverage = _DimCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_dim) == 0

    def test_src_free_with_tgt_local(self, mocker):
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
        tgt_metadata = mocker.sentinel.tgt_metadata
        self.tgt_coverage["metadata"] = [None, tgt_metadata]
        tgt_coord = mocker.sentinel.tgt_coord
        self.tgt_coverage["coords"] = [None, tgt_coord]
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_dim) == 1
        assert self.prepared_item == self.resolve.prepared_category.items_dim[0]
        assert self.m_create_prepared_item.call_count == 1
        expected = [mocker.call(tgt_coord, tgt_dim, tgt_metadata=tgt_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_free_with_tgt_local__strict(self, mocker):
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
        tgt_metadata = mocker.sentinel.tgt_metadata
        self.tgt_coverage["metadata"] = [None, tgt_metadata]
        tgt_coord = mocker.sentinel.tgt_coord
        self.tgt_coverage["coords"] = [None, tgt_coord]
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_dim) == 0

    def test_src_no_local_with_tgt_local__extra_dims(self, mocker):
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
        tgt_metadata = mocker.sentinel.tgt_metadata
        self.tgt_coverage["metadata"] = [tgt_metadata, None, None]
        tgt_coord = mocker.sentinel.tgt_coord
        self.tgt_coverage["coords"] = [tgt_coord, None, None]
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_dim) == 1
        assert self.resolve.prepared_category.items_dim[0] == self.prepared_item
        assert self.m_create_prepared_item.call_count == 1
        expected = [mocker.call(tgt_coord, tgt_dim, tgt_metadata=tgt_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_no_local_with_tgt_local__extra_dims_strict(self, mocker):
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
        tgt_metadata = mocker.sentinel.tgt_metadata
        self.tgt_coverage["metadata"] = [tgt_metadata, None, None]
        tgt_coord = mocker.sentinel.tgt_coord
        self.tgt_coverage["coords"] = [tgt_coord, None, None]
        tgt_coverage = _DimCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_dim(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_dim) == 1
        assert self.resolve.prepared_category.items_dim[0] == self.prepared_item
        assert self.m_create_prepared_item.call_count == 1
        expected = [mocker.call(tgt_coord, tgt_dim, tgt_metadata=tgt_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected


class Test__prepare_local_payload_aux:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
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
        self.src_prepared_item = mocker.sentinel.src_prepared_item
        self.tgt_prepared_item = mocker.sentinel.tgt_prepared_item
        self.m_create_prepared_item = mocker.patch(
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
        assert len(self.resolve.prepared_category.items_aux) == 0

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
        assert len(self.resolve.prepared_category.items_aux) == 0

    def test_src_local_with_tgt_local(self, mocker):
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
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_dims = (1,)
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=src_dims)
        self.src_coverage["local_items_aux"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_dims = (1,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_aux) == 2
        expected = [self.src_prepared_item, self.tgt_prepared_item]
        assert self.resolve.prepared_category.items_aux == expected
        expected = [
            mocker.call(src_coord, tgt_dims, src_metadata=src_metadata),
            mocker.call(tgt_coord, tgt_dims, tgt_metadata=tgt_metadata),
        ]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_local_with_tgt_local__strict(self, mocker):
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
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_dims = (1,)
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=src_dims)
        self.src_coverage["local_items_aux"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_dims = (1,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_aux) == 0

    def test_src_local_with_tgt_free(self, mocker):
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
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_dims = (1,)
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=src_dims)
        self.src_coverage["local_items_aux"].append(src_item)
        self.src_coverage["dims_local"].extend(src_dims)
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_aux) == 1
        expected = [self.src_prepared_item]
        assert self.resolve.prepared_category.items_aux == expected
        expected = [mocker.call(src_coord, src_dims, src_metadata=src_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_local_with_tgt_free__strict(self, mocker):
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
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_dims = (1,)
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=src_dims)
        self.src_coverage["local_items_aux"].append(src_item)
        self.src_coverage["dims_local"].extend(src_dims)
        src_coverage = _AuxCoverage(**self.src_coverage)
        self.tgt_coverage["cube"] = self.Cube(ndim=2)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_aux) == 0

    def test_src_free_with_tgt_local(self, mocker):
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
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_dims = (1,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        self.tgt_coverage["dims_local"].extend(tgt_dims)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_aux) == 1
        expected = [self.tgt_prepared_item]
        assert self.resolve.prepared_category.items_aux == expected
        expected = [mocker.call(tgt_coord, tgt_dims, tgt_metadata=tgt_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_free_with_tgt_local__strict(self, mocker):
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
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_dims = (1,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        self.tgt_coverage["dims_local"].extend(tgt_dims)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_aux) == 0

    def test_src_no_local_with_tgt_local__extra_dims(self, mocker):
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
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_dims = (0,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        self.tgt_coverage["dims_local"].extend(tgt_dims)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_aux) == 1
        expected = [self.tgt_prepared_item]
        assert self.resolve.prepared_category.items_aux == expected
        expected = [mocker.call(tgt_coord, tgt_dims, tgt_metadata=tgt_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_no_local_with_tgt_local__extra_dims_strict(self, mocker):
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
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_dims = (0,)
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=tgt_dims)
        self.tgt_coverage["local_items_aux"].append(tgt_item)
        self.tgt_coverage["dims_local"].extend(tgt_dims)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=True):
            self.resolve._prepare_local_payload_aux(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_aux) == 1
        expected = [self.tgt_prepared_item]
        assert self.resolve.prepared_category.items_aux == expected
        expected = [mocker.call(tgt_coord, tgt_dims, tgt_metadata=tgt_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected


class Test__prepare_local_payload_scalar:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
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
        self.src_prepared_item = mocker.sentinel.src_prepared_item
        self.tgt_prepared_item = mocker.sentinel.tgt_prepared_item
        self.m_create_prepared_item = mocker.patch(
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
        assert len(self.resolve.prepared_category.items_scalar) == 0

    def test_src_no_local_with_tgt_no_local__strict(self):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 0

    def test_src_no_local_with_tgt_no_local__src_scalar_cube(self):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 0

    def test_src_no_local_with_tgt_no_local__src_scalar_cube_strict(self):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 0

    def test_src_local_with_tgt_no_local(self, mocker):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=self.src_dims)
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 1
        expected = [self.src_prepared_item]
        assert self.resolve.prepared_category.items_scalar == expected
        expected = [mocker.call(src_coord, self.src_dims, src_metadata=src_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_local_with_tgt_no_local__strict(self, mocker):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=self.src_dims)
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 0

    def test_src_local_with_tgt_no_local__src_scalar_cube(self, mocker):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=self.src_dims)
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 1
        expected = [self.src_prepared_item]
        assert self.resolve.prepared_category.items_scalar == expected
        expected = [mocker.call(src_coord, self.src_dims, src_metadata=src_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_local_with_tgt_no_local__src_scalar_cube_strict(self, mocker):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=self.src_dims)
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 0

    def test_src_no_local_with_tgt_local(self, mocker):
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims)
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 1
        expected = [self.tgt_prepared_item]
        assert self.resolve.prepared_category.items_scalar == expected
        expected = [mocker.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_no_local_with_tgt_local__strict(self, mocker):
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims)
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 0

    def test_src_no_local_with_tgt_local__src_scalar_cube(self, mocker):
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims)
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 1
        expected = [self.tgt_prepared_item]
        assert self.resolve.prepared_category.items_scalar == expected
        expected = [mocker.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_no_local_with_tgt_local__src_scalar_cube_strict(self, mocker):
        self.m_create_prepared_item.side_effect = (self.tgt_prepared_item,)
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims)
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 1
        expected = [self.tgt_prepared_item]
        assert self.resolve.prepared_category.items_scalar == expected
        expected = [mocker.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata)]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_local_with_tgt_local(self, mocker):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=self.src_dims)
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims)
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 2
        expected = [self.src_prepared_item, self.tgt_prepared_item]
        assert self.resolve.prepared_category.items_scalar == expected
        expected = [
            mocker.call(src_coord, self.src_dims, src_metadata=src_metadata),
            mocker.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata),
        ]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_local_with_tgt_local__strict(self, mocker):
        ndim = 2
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=self.src_dims)
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims)
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 0

    def test_src_local_with_tgt_local__src_scalar_cube(self, mocker):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=self.src_dims)
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims)
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 2
        expected = [self.src_prepared_item, self.tgt_prepared_item]
        assert self.resolve.prepared_category.items_scalar == expected
        expected = [
            mocker.call(src_coord, self.src_dims, src_metadata=src_metadata),
            mocker.call(tgt_coord, self.tgt_dims, tgt_metadata=tgt_metadata),
        ]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_src_local_with_tgt_local__src_scalar_cube_strict(self, mocker):
        ndim = 0
        self.src_coverage["cube"] = self.Cube(ndim=ndim)
        src_metadata = mocker.sentinel.src_metadata
        src_coord = mocker.sentinel.src_coord
        src_item = _Item(metadata=src_metadata, coord=src_coord, dims=self.src_dims)
        self.src_coverage["local_items_scalar"].append(src_item)
        src_coverage = _AuxCoverage(**self.src_coverage)
        tgt_metadata = mocker.sentinel.tgt_metadata
        tgt_coord = mocker.sentinel.tgt_coord
        tgt_item = _Item(metadata=tgt_metadata, coord=tgt_coord, dims=self.tgt_dims)
        self.tgt_coverage["local_items_scalar"].append(tgt_item)
        tgt_coverage = _AuxCoverage(**self.tgt_coverage)
        with LENIENT.context(maths=False):
            self.resolve._prepare_local_payload_scalar(src_coverage, tgt_coverage)
        assert len(self.resolve.prepared_category.items_scalar) == 0


class Test__prepare_local_payload:
    def test(self, mocker):
        src_dim_coverage = mocker.sentinel.src_dim_coverage
        src_aux_coverage = mocker.sentinel.src_aux_coverage
        tgt_dim_coverage = mocker.sentinel.tgt_dim_coverage
        tgt_aux_coverage = mocker.sentinel.tgt_aux_coverage
        root = "iris.common.resolve.Resolve"
        m_prepare_dim = mocker.patch(f"{root}._prepare_local_payload_dim")
        m_prepare_aux = mocker.patch(f"{root}._prepare_local_payload_aux")
        m_prepare_scalar = mocker.patch(f"{root}._prepare_local_payload_scalar")
        resolve = Resolve()
        resolve._prepare_local_payload(
            src_dim_coverage,
            src_aux_coverage,
            tgt_dim_coverage,
            tgt_aux_coverage,
        )
        assert m_prepare_dim.call_count == 1
        expected = [mocker.call(src_dim_coverage, tgt_dim_coverage)]
        assert m_prepare_dim.call_args_list == expected
        assert m_prepare_aux.call_count == 1
        expected = [mocker.call(src_aux_coverage, tgt_aux_coverage)]
        assert m_prepare_aux.call_args_list == expected
        assert m_prepare_scalar.call_count == 1
        expected = [mocker.call(src_aux_coverage, tgt_aux_coverage)]
        assert m_prepare_scalar.call_args_list == expected


class Test__metadata_prepare:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.src_cube = mocker.sentinel.src_cube
        self.src_category_local = mocker.sentinel.src_category_local
        self.src_dim_coverage = mocker.sentinel.src_dim_coverage
        self.src_aux_coverage = mocker.Mock(
            common_items_aux=mocker.sentinel.src_aux_coverage_common_items_aux,
            common_items_scalar=mocker.sentinel.src_aux_coverage_common_items_scalar,
        )
        self.tgt_cube = mocker.sentinel.tgt_cube
        self.tgt_category_local = mocker.sentinel.tgt_category_local
        self.tgt_dim_coverage = mocker.sentinel.tgt_dim_coverage
        self.tgt_aux_coverage = mocker.Mock(
            common_items_aux=mocker.sentinel.tgt_aux_coverage_common_items_aux,
            common_items_scalar=mocker.sentinel.tgt_aux_coverage_common_items_scalar,
        )
        self.resolve = Resolve()
        root = "iris.common.resolve.Resolve"
        self.m_prepare_common_dim_payload = mocker.patch(
            f"{root}._prepare_common_dim_payload"
        )
        self.m_prepare_common_aux_payload = mocker.patch(
            f"{root}._prepare_common_aux_payload"
        )
        self.m_prepare_local_payload = mocker.patch(f"{root}._prepare_local_payload")
        self.m_prepare_factory_payload = mocker.patch(
            f"{root}._prepare_factory_payload"
        )

    def _check(self):
        assert self.resolve.prepared_category is None
        assert self.resolve.prepared_factories is None
        self.resolve._metadata_prepare()
        expected = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        assert self.resolve.prepared_category == expected
        assert self.resolve.prepared_factories == []
        assert self.m_prepare_common_dim_payload.call_count == 1
        expected = [mock.call(self.src_dim_coverage, self.tgt_dim_coverage)]
        assert self.m_prepare_common_dim_payload.call_args_list == expected
        assert self.m_prepare_common_aux_payload.call_count == 2
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
        assert self.m_prepare_common_aux_payload.call_args_list == expected
        assert self.m_prepare_local_payload.call_count == 1
        expected = [
            mock.call(
                self.src_dim_coverage,
                self.src_aux_coverage,
                self.tgt_dim_coverage,
                self.tgt_aux_coverage,
            )
        ]
        assert self.m_prepare_local_payload.call_args_list == expected
        assert self.m_prepare_factory_payload.call_count == 2
        expected = [
            mock.call(self.tgt_cube, self.tgt_category_local, from_src=False),
            mock.call(self.src_cube, self.src_category_local),
        ]
        assert self.m_prepare_factory_payload.call_args_list == expected

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


class Test__prepare_factory_payload:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
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
        self.m_get_prepared_item = mocker.patch(
            "iris.common.resolve.Resolve._get_prepared_item"
        )
        self.category_local = mocker.sentinel.category_local
        self.from_src = mocker.sentinel.from_src

    def test_no_factory(self):
        cube = self.Cube(aux_factories=[])
        self.resolve._prepare_factory_payload(cube, self.category_local)
        assert len(self.resolve.prepared_factories) == 0

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
        assert self.resolve.prepared_factories == prepared_factories

    def test_factory__dependency_already_prepared(self, mocker):
        coord_a = self.Coord(metadata=mocker.sentinel.coord_a_metadata)
        coord_b = self.Coord(metadata=mocker.sentinel.coord_b_metadata)
        coord_c = self.Coord(metadata=mocker.sentinel.coord_c_metadata)
        side_effect = (coord_a, coord_b, coord_c)
        self.m_get_prepared_item.side_effect = side_effect
        dependencies = dict(name_a=coord_a, name_b=coord_b, name_c=coord_c)
        aux_factory = self.Factory_T1(dependencies=dependencies)
        aux_factories = [aux_factory]
        cube = self.Cube(aux_factories=aux_factories)
        self.resolve._prepare_factory_payload(
            cube, self.category_local, from_src=self.from_src
        )
        assert len(self.resolve.prepared_factories) == 1
        prepared_dependencies = {
            name: coord.metadata for name, coord in dependencies.items()
        }
        expected = [
            _PreparedFactory(
                container=self.container_T1, dependencies=prepared_dependencies
            )
        ]
        assert self.resolve.prepared_factories == expected
        assert self.m_get_prepared_item.call_count == len(side_effect)
        expected = [
            mocker.call(coord_a.metadata, self.category_local, from_src=self.from_src),
            mocker.call(coord_b.metadata, self.category_local, from_src=self.from_src),
            mocker.call(coord_c.metadata, self.category_local, from_src=self.from_src),
        ]
        actual = self.m_get_prepared_item.call_args_list
        for call in expected:
            assert call in actual

    def test_factory__dependency_local_not_prepared(self, mocker):
        coord_a = self.Coord(metadata=mocker.sentinel.coord_a_metadata)
        coord_b = self.Coord(metadata=mocker.sentinel.coord_b_metadata)
        coord_c = self.Coord(metadata=mocker.sentinel.coord_c_metadata)
        side_effect = (None, coord_a, None, coord_b, None, coord_c)
        self.m_get_prepared_item.side_effect = side_effect
        dependencies = dict(name_a=coord_a, name_b=coord_b, name_c=coord_c)
        aux_factory = self.Factory_T1(dependencies=dependencies)
        aux_factories = [aux_factory]
        cube = self.Cube(aux_factories=aux_factories)
        self.resolve._prepare_factory_payload(
            cube, self.category_local, from_src=self.from_src
        )
        assert len(self.resolve.prepared_factories) == 1
        prepared_dependencies = {
            name: coord.metadata for name, coord in dependencies.items()
        }
        expected = [
            _PreparedFactory(
                container=self.container_T1, dependencies=prepared_dependencies
            )
        ]
        assert self.resolve.prepared_factories == expected
        assert self.m_get_prepared_item.call_count == len(side_effect)
        expected = [
            mocker.call(coord_a.metadata, self.category_local, from_src=self.from_src),
            mocker.call(coord_b.metadata, self.category_local, from_src=self.from_src),
            mocker.call(coord_c.metadata, self.category_local, from_src=self.from_src),
            mocker.call(
                coord_a.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
            mocker.call(
                coord_b.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
            mocker.call(
                coord_c.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
        ]
        actual = self.m_get_prepared_item.call_args_list
        for call in expected:
            assert call in actual

    def test_factory__dependency_not_found(self, mocker):
        coord_a = self.Coord(metadata=mocker.sentinel.coord_a_metadata)
        coord_b = self.Coord(metadata=mocker.sentinel.coord_b_metadata)
        coord_c = self.Coord(metadata=mocker.sentinel.coord_c_metadata)
        side_effect = (None, None)
        self.m_get_prepared_item.side_effect = side_effect
        dependencies = dict(name_a=coord_a, name_b=coord_b, name_c=coord_c)
        aux_factory = self.Factory_T1(dependencies=dependencies)
        aux_factories = [aux_factory]
        cube = self.Cube(aux_factories=aux_factories)
        self.resolve._prepare_factory_payload(
            cube, self.category_local, from_src=self.from_src
        )
        assert len(self.resolve.prepared_factories) == 0
        assert self.m_get_prepared_item.call_count == len(side_effect)
        expected = [
            mocker.call(coord_a.metadata, self.category_local, from_src=self.from_src),
            mocker.call(coord_b.metadata, self.category_local, from_src=self.from_src),
            mocker.call(coord_c.metadata, self.category_local, from_src=self.from_src),
            mocker.call(
                coord_a.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
            mocker.call(
                coord_b.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
            mocker.call(
                coord_c.metadata,
                self.category_local,
                from_src=self.from_src,
                from_local=True,
            ),
        ]
        actual = self.m_get_prepared_item.call_args_list
        for call in actual:
            assert call in expected


class Test__get_prepared_item:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        PreparedItem = namedtuple("PreparedItem", ["metadata"])
        self.resolve = Resolve()
        self.prepared_dim_metadata_src = mocker.sentinel.prepared_dim_metadata_src
        self.prepared_dim_metadata_tgt = mocker.sentinel.prepared_dim_metadata_tgt
        self.prepared_items_dim = PreparedItem(
            metadata=_PreparedMetadata(
                combined=None,
                src=self.prepared_dim_metadata_src,
                tgt=self.prepared_dim_metadata_tgt,
            )
        )
        self.prepared_aux_metadata_src = mocker.sentinel.prepared_aux_metadata_src
        self.prepared_aux_metadata_tgt = mocker.sentinel.prepared_aux_metadata_tgt
        self.prepared_items_aux = PreparedItem(
            metadata=_PreparedMetadata(
                combined=None,
                src=self.prepared_aux_metadata_src,
                tgt=self.prepared_aux_metadata_tgt,
            )
        )
        self.prepared_scalar_metadata_src = mocker.sentinel.prepared_scalar_metadata_src
        self.prepared_scalar_metadata_tgt = mocker.sentinel.prepared_scalar_metadata_tgt
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
        self.m_create_prepared_item = mocker.patch(
            "iris.common.resolve.Resolve._create_prepared_item"
        )
        self.local_dim_metadata = mocker.sentinel.local_dim_metadata
        self.local_aux_metadata = mocker.sentinel.local_aux_metadata
        self.local_scalar_metadata = mocker.sentinel.local_scalar_metadata
        self.local_coord = mocker.sentinel.local_coord
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

    def test_missing_prepared_coord__from_src(self, mocker):
        metadata = mocker.sentinel.missing
        category_local = None
        result = self.resolve._get_prepared_item(metadata, category_local)
        assert result is None

    def test_missing_prepared_coord__from_tgt(self, mocker):
        metadata = mocker.sentinel.missing
        category_local = None
        result = self.resolve._get_prepared_item(
            metadata, category_local, from_src=False
        )
        assert result is None

    def test_get_prepared_dim_coord__from_src(self):
        metadata = self.prepared_dim_metadata_src
        category_local = None
        result = self.resolve._get_prepared_item(metadata, category_local)
        assert result == self.prepared_items_dim

    def test_get_prepared_dim_coord__from_tgt(self):
        metadata = self.prepared_dim_metadata_tgt
        category_local = None
        result = self.resolve._get_prepared_item(
            metadata, category_local, from_src=False
        )
        assert result == self.prepared_items_dim

    def test_get_prepared_aux_coord__from_src(self):
        metadata = self.prepared_aux_metadata_src
        category_local = None
        result = self.resolve._get_prepared_item(metadata, category_local)
        assert result == self.prepared_items_aux

    def test_get_prepared_aux_coord__from_tgt(self):
        metadata = self.prepared_aux_metadata_tgt
        category_local = None
        result = self.resolve._get_prepared_item(
            metadata, category_local, from_src=False
        )
        assert result == self.prepared_items_aux

    def test_get_prepared_scalar_coord__from_src(self):
        metadata = self.prepared_scalar_metadata_src
        category_local = None
        result = self.resolve._get_prepared_item(metadata, category_local)
        assert result == self.prepared_items_scalar

    def test_get_prepared_scalar_coord__from_tgt(self):
        metadata = self.prepared_scalar_metadata_tgt
        category_local = None
        result = self.resolve._get_prepared_item(
            metadata, category_local, from_src=False
        )
        assert result == self.prepared_items_scalar

    def test_missing_local_coord__from_src(self, mocker):
        metadata = mocker.sentinel.missing
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_local=True
        )
        assert result is None

    def test_missing_local_coord__from_tgt(self, mocker):
        metadata = mocker.sentinel.missing
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_src=False, from_local=True
        )
        assert result is None

    def test_get_local_dim_coord__from_src(self, mocker):
        created_local_item = mocker.sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_dim_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_local=True
        )
        expected = created_local_item
        assert result == expected
        assert len(self.resolve.prepared_category.items_dim) == 2
        assert self.resolve.prepared_category.items_dim[1] == expected
        assert self.m_create_prepared_item.call_count == 1
        dims = (self.resolve.mapping[self.local_coord_dims[0]],)
        expected = [
            mocker.call(
                self.local_coord,
                dims,
                src_metadata=metadata,
                tgt_metadata=None,
            )
        ]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_get_local_dim_coord__from_tgt(self, mocker):
        created_local_item = mocker.sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_dim_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_src=False, from_local=True
        )
        expected = created_local_item
        assert result == expected
        assert len(self.resolve.prepared_category.items_dim) == 2
        assert self.resolve.prepared_category.items_dim[1] == expected
        assert self.m_create_prepared_item.call_count == 1
        dims = self.local_coord_dims
        expected = [
            mocker.call(
                self.local_coord,
                dims,
                src_metadata=None,
                tgt_metadata=metadata,
            )
        ]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_get_local_aux_coord__from_src(self, mocker):
        created_local_item = mocker.sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_aux_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_local=True
        )
        expected = created_local_item
        assert result == expected
        assert len(self.resolve.prepared_category.items_aux) == 2
        assert self.resolve.prepared_category.items_aux[1] == expected
        assert self.m_create_prepared_item.call_count == 1
        dims = (self.resolve.mapping[self.local_coord_dims[0]],)
        expected = [
            mocker.call(
                self.local_coord,
                dims,
                src_metadata=metadata,
                tgt_metadata=None,
            )
        ]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_get_local_aux_coord__from_tgt(self, mocker):
        created_local_item = mocker.sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_aux_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_src=False, from_local=True
        )
        expected = created_local_item
        assert result == expected
        assert len(self.resolve.prepared_category.items_aux) == 2
        assert self.resolve.prepared_category.items_aux[1] == expected
        assert self.m_create_prepared_item.call_count == 1
        dims = self.local_coord_dims
        expected = [
            mocker.call(
                self.local_coord,
                dims,
                src_metadata=None,
                tgt_metadata=metadata,
            )
        ]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_get_local_scalar_coord__from_src(self, mocker):
        created_local_item = mocker.sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_scalar_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_local=True
        )
        expected = created_local_item
        assert result == expected
        assert len(self.resolve.prepared_category.items_scalar) == 2
        assert self.resolve.prepared_category.items_scalar[1] == expected
        assert self.m_create_prepared_item.call_count == 1
        dims = (self.resolve.mapping[self.local_coord_dims[0]],)
        expected = [
            mocker.call(
                self.local_coord,
                dims,
                src_metadata=metadata,
                tgt_metadata=None,
            )
        ]
        assert self.m_create_prepared_item.call_args_list == expected

    def test_get_local_scalar_coord__from_tgt(self, mocker):
        created_local_item = mocker.sentinel.created_local_item
        self.m_create_prepared_item.return_value = created_local_item
        metadata = self.local_scalar_metadata
        result = self.resolve._get_prepared_item(
            metadata, self.category_local, from_src=False, from_local=True
        )
        expected = created_local_item
        assert result == expected
        assert len(self.resolve.prepared_category.items_scalar) == 2
        assert self.resolve.prepared_category.items_scalar[1] == expected
        assert self.m_create_prepared_item.call_count == 1
        dims = self.local_coord_dims
        expected = [
            mocker.call(
                self.local_coord,
                dims,
                src_metadata=None,
                tgt_metadata=metadata,
            )
        ]
        assert self.m_create_prepared_item.call_args_list == expected


class Test_cube:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
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
        self.m_add_dim_coord = mocker.patch("iris.cube.Cube.add_dim_coord")
        self.m_add_aux_coord = mocker.patch("iris.cube.Cube.add_aux_coord")
        self.m_add_aux_factory = mocker.patch("iris.cube.Cube.add_aux_factory")
        self.m_coord = mocker.patch("iris.cube.Cube.coord")
        #
        # prepared coordinates
        #
        prepared_category = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])
        # prepared dim coordinates
        self.prepared_dim_0_metadata = _PreparedMetadata(
            combined=mocker.sentinel.prepared_dim_0_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_dim_0_points = mocker.sentinel.prepared_dim_0_points
        self.prepared_dim_0_bounds = mocker.sentinel.prepared_dim_0_bounds
        self.prepared_dim_0_dims = (0,)
        self.prepared_dim_0_coord = mocker.Mock(metadata=None)
        self.prepared_dim_0_container = mocker.Mock(
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
            combined=mocker.sentinel.prepared_dim_1_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_dim_1_points = mocker.sentinel.prepared_dim_1_points
        self.prepared_dim_1_bounds = mocker.sentinel.prepared_dim_1_bounds
        self.prepared_dim_1_dims = (1,)
        self.prepared_dim_1_coord = mocker.Mock(metadata=None)
        self.prepared_dim_1_container = mocker.Mock(
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
            combined=mocker.sentinel.prepared_aux_0_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_aux_0_points = mocker.sentinel.prepared_aux_0_points
        self.prepared_aux_0_bounds = mocker.sentinel.prepared_aux_0_bounds
        self.prepared_aux_0_dims = (0,)
        self.prepared_aux_0_coord = mocker.Mock(metadata=None)
        self.prepared_aux_0_container = mocker.Mock(
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
            combined=mocker.sentinel.prepared_aux_1_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_aux_1_points = mocker.sentinel.prepared_aux_1_points
        self.prepared_aux_1_bounds = mocker.sentinel.prepared_aux_1_bounds
        self.prepared_aux_1_dims = (1,)
        self.prepared_aux_1_coord = mocker.Mock(metadata=None)
        self.prepared_aux_1_container = mocker.Mock(
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
            combined=mocker.sentinel.prepared_scalar_0_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_scalar_0_points = mocker.sentinel.prepared_scalar_0_points
        self.prepared_scalar_0_bounds = mocker.sentinel.prepared_scalar_0_bounds
        self.prepared_scalar_0_dims = ()
        self.prepared_scalar_0_coord = mocker.Mock(metadata=None)
        self.prepared_scalar_0_container = mocker.Mock(
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
            combined=mocker.sentinel.prepared_scalar_1_metadata_combined,
            src=None,
            tgt=None,
        )
        self.prepared_scalar_1_points = mocker.sentinel.prepared_scalar_1_points
        self.prepared_scalar_1_bounds = mocker.sentinel.prepared_scalar_1_bounds
        self.prepared_scalar_1_dims = ()
        self.prepared_scalar_1_coord = mocker.Mock(metadata=None)
        self.prepared_scalar_1_container = mocker.Mock(
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
        self.aux_factory = mocker.sentinel.aux_factory
        self.prepared_factory_container = mocker.Mock(return_value=self.aux_factory)
        self.prepared_factory_metadata_a = _PreparedMetadata(
            combined=mocker.sentinel.prepared_factory_metadata_a_combined,
            src=None,
            tgt=None,
        )
        self.prepared_factory_metadata_b = _PreparedMetadata(
            combined=mocker.sentinel.prepared_factory_metadata_b_combined,
            src=None,
            tgt=None,
        )
        self.prepared_factory_metadata_c = _PreparedMetadata(
            combined=mocker.sentinel.prepared_factory_metadata_c_combined,
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
            mocker.sentinel.prepared_factory_coord_a,
            mocker.sentinel.prepared_factory_coord_b,
            mocker.sentinel.prepared_factory_coord_c,
        )
        self.m_coord.side_effect = self.prepared_factory_side_effect
        self.resolve.prepared_category = prepared_category
        self.resolve.prepared_factories = prepared_factories

        # Required to stop mock 'containers' failing in an 'issubclass' call.
        mocker.patch("iris.common.resolve.issubclass", mocker.Mock(return_value=False))

    def test_no_resolved_shape(self):
        self.resolve._broadcast_shape = None
        data = None
        emsg = "Cannot resolve resultant cube, as no candidate cubes have been provided"
        with pytest.raises(ValueError, match=emsg):
            _ = self.resolve.cube(data)

    def test_bad_data_shape(self):
        emsg = "Cannot resolve resultant cube, as the provided data must have shape"
        with pytest.raises(ValueError, match=emsg):
            _ = self.resolve.cube(self.bad_data)

    def test_bad_data_shape__inplace(self):
        self.resolve.lhs_cube = Cube(self.bad_data)
        emsg = "Cannot resolve resultant cube in-place"
        with pytest.raises(ValueError, match=emsg):
            _ = self.resolve.cube(self.data, in_place=True)

    def _check(self):
        # check dim coordinate 0
        assert self.prepared_dim_0.container.call_count == 1
        expected = [
            mock.call(self.prepared_dim_0_points, bounds=self.prepared_dim_0_bounds)
        ]
        assert self.prepared_dim_0.container.call_args_list == expected
        assert (
            self.prepared_dim_0_metadata.combined == self.prepared_dim_0_coord.metadata
        )
        # check dim coordinate 1
        assert self.prepared_dim_1.container.call_count == 1
        expected = [
            mock.call(self.prepared_dim_1_points, bounds=self.prepared_dim_1_bounds)
        ]
        assert self.prepared_dim_1.container.call_args_list == expected
        assert (
            self.prepared_dim_1_metadata.combined == self.prepared_dim_1_coord.metadata
        )
        # check add_dim_coord
        assert self.m_add_dim_coord.call_count == 2
        expected = [
            mock.call(self.prepared_dim_0_coord, self.prepared_dim_0_dims),
            mock.call(self.prepared_dim_1_coord, self.prepared_dim_1_dims),
        ]
        assert self.m_add_dim_coord.call_args_list == expected

        # check aux coordinate 0
        assert self.prepared_aux_0.container.call_count == 1
        expected = [
            mock.call(self.prepared_aux_0_points, bounds=self.prepared_aux_0_bounds)
        ]
        assert self.prepared_aux_0.container.call_args_list == expected
        assert (
            self.prepared_aux_0_metadata.combined == self.prepared_aux_0_coord.metadata
        )
        # check aux coordinate 1
        assert self.prepared_aux_1.container.call_count == 1
        expected = [
            mock.call(self.prepared_aux_1_points, bounds=self.prepared_aux_1_bounds)
        ]
        assert self.prepared_aux_1.container.call_args_list == expected
        assert (
            self.prepared_aux_1_metadata.combined == self.prepared_aux_1_coord.metadata
        )
        # check scalar coordinate 0
        assert self.prepared_scalar_0.container.call_count == 1
        expected = [
            mock.call(
                self.prepared_scalar_0_points,
                bounds=self.prepared_scalar_0_bounds,
            )
        ]
        assert self.prepared_scalar_0.container.call_args_list == expected
        assert (
            self.prepared_scalar_0_metadata.combined
            == self.prepared_scalar_0_coord.metadata
        )
        # check scalar coordinate 1
        assert self.prepared_scalar_1.container.call_count == 1
        expected = [
            mock.call(
                self.prepared_scalar_1_points,
                bounds=self.prepared_scalar_1_bounds,
            )
        ]
        assert self.prepared_scalar_1.container.call_args_list == expected
        assert (
            self.prepared_scalar_1_metadata.combined
            == self.prepared_scalar_1_coord.metadata
        )
        # check add_aux_coord
        assert self.m_add_aux_coord.call_count == 4
        expected = [
            mock.call(self.prepared_aux_0_coord, self.prepared_aux_0_dims),
            mock.call(self.prepared_aux_1_coord, self.prepared_aux_1_dims),
            mock.call(self.prepared_scalar_0_coord, self.prepared_scalar_0_dims),
            mock.call(self.prepared_scalar_1_coord, self.prepared_scalar_1_dims),
        ]
        assert self.m_add_aux_coord.call_args_list == expected

        # check auxiliary factories
        assert self.m_add_aux_factory.call_count == 1
        expected = [mock.call(self.aux_factory)]
        assert self.m_add_aux_factory.call_args_list == expected
        assert self.prepared_factory_container.call_count == 1
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
        assert self.prepared_factory_container.call_args_list == expected
        assert self.m_coord.call_count == 3
        expected = [
            mock.call(self.prepared_factory_metadata_a.combined),
            mock.call(self.prepared_factory_metadata_b.combined),
            mock.call(self.prepared_factory_metadata_c.combined),
        ]
        assert self.m_coord.call_args_list == expected

    def test_resolve(self):
        result = self.resolve.cube(self.data)
        assert result.metadata == self.cube_metadata
        self._check()
        assert self.resolve.lhs_cube is not result

    def test_resolve__inplace(self):
        result = self.resolve.cube(self.data, in_place=True)
        assert result.metadata == self.cube_metadata
        self._check()
        assert self.resolve.lhs_cube is result
