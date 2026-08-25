# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.fileformats.cf.CFGroup` class."""

from copy import deepcopy
from typing import TYPE_CHECKING, NamedTuple
from unittest.mock import MagicMock

import pytest

from iris.fileformats import cf


class MockVariables(NamedTuple):
    formula_var: MagicMock
    anc_var: MagicMock
    aux_var: MagicMock
    bound_var: MagicMock
    climat_var: MagicMock
    coord_var: MagicMock
    data_var: MagicMock
    grid_var: MagicMock
    label_var: MagicMock
    measure_var: MagicMock
    ugrid_aux_var: MagicMock
    ugrid_conn_var: MagicMock
    ugrid_mesh_var: MagicMock


@pytest.fixture
def mock_variables(mocker) -> MockVariables:
    return MockVariables(
        mocker.MagicMock(spec=cf._CFFormulaTermsVariable, cf_name="formula_var"),
        mocker.MagicMock(spec=cf.CFAncillaryDataVariable, cf_name="anc_var"),
        mocker.MagicMock(spec=cf.CFAuxiliaryCoordinateVariable, cf_name="aux_var"),
        mocker.MagicMock(spec=cf.CFBoundaryVariable, cf_name="bound_var"),
        mocker.MagicMock(spec=cf.CFClimatologyVariable, cf_name="climat_var"),
        mocker.MagicMock(spec=cf.CFCoordinateVariable, cf_name="coord_var"),
        mocker.MagicMock(spec=cf.CFDataVariable, cf_name="data_var"),
        mocker.MagicMock(spec=cf.CFGridMappingVariable, cf_name="grid_var"),
        mocker.MagicMock(spec=cf.CFLabelVariable, cf_name="label_var"),
        mocker.MagicMock(spec=cf.CFMeasureVariable, cf_name="measure_var"),
        mocker.MagicMock(
            spec=cf.CFUGridAuxiliaryCoordinateVariable, cf_name="ugrid_aux_var"
        ),
        mocker.MagicMock(spec=cf.CFUGridConnectivityVariable, cf_name="ugrid_conn_var"),
        mocker.MagicMock(spec=cf.CFUGridMeshVariable, cf_name="ugrid_mesh_var"),
    )


@pytest.fixture
def mock_variables_extra(mock_variables) -> MockVariables:
    result = deepcopy(mock_variables)
    for var in result:
        var.cf_name = f"{var.cf_name}_2"
    return result


@pytest.fixture
def cf_group() -> cf.CFGroup:
    return cf.CFGroup()


def test_non_data_names(cf_group, mock_variables, mock_variables_extra):
    data_var = mock_variables.data_var
    aux_var = mock_variables.aux_var
    coord_var = mock_variables.coord_var
    coord_var2 = mock_variables_extra.coord_var
    duplicate_name_var = mock_variables_extra.aux_var
    duplicate_name_var.cf_name = aux_var.cf_name

    for var in (
        data_var,
        aux_var,
        coord_var,
        coord_var2,
        duplicate_name_var,
    ):
        cf_group[var.cf_name] = var

    expected_names = [var.cf_name for var in (aux_var, coord_var, coord_var2)]
    expected = set(expected_names)
    assert cf_group.non_data_variable_names == expected


# TODO: unit tests for existing functionality pre 2021-03-11.


class TestUgrid:
    """Separate class to test UGRID functionality."""

    @pytest.fixture(autouse=True)
    def _setup(self, cf_group, mock_variables):
        self.cf_group = cf_group
        self.variables = mock_variables

    def test_inherited(self):
        coord_var = self.variables.coord_var
        self.cf_group[coord_var.cf_name] = coord_var
        assert self.cf_group.coordinates[coord_var.cf_name] == coord_var

    def test_connectivities(self):
        conn_var = self.variables.ugrid_conn_var
        self.cf_group[conn_var.cf_name] = conn_var
        assert self.cf_group.connectivities[conn_var.cf_name] == conn_var

    def test_ugrid_coords(self):
        coord_var = self.variables.ugrid_aux_var
        self.cf_group[coord_var.cf_name] = coord_var
        assert self.cf_group.ugrid_coords[coord_var.cf_name] == coord_var

    def test_meshes(self):
        mesh_var = self.variables.ugrid_mesh_var
        self.cf_group[mesh_var.cf_name] = mesh_var
        assert self.cf_group.meshes[mesh_var.cf_name] == mesh_var

    def test_non_data_names(self, mock_variables_extra):
        data_var = self.variables.data_var
        coord_var = self.variables.coord_var
        conn_var = self.variables.ugrid_conn_var
        ugrid_coord_var = self.variables.ugrid_aux_var
        mesh_var = self.variables.ugrid_mesh_var
        mesh_var2 = mock_variables_extra.ugrid_mesh_var
        duplicate_name_var = mock_variables_extra.coord_var
        duplicate_name_var.cf_name = coord_var.cf_name

        for var in (
            data_var,
            coord_var,
            conn_var,
            ugrid_coord_var,
            mesh_var,
            mesh_var2,
            duplicate_name_var,
        ):
            self.cf_group[var.cf_name] = var

        expected_names = [
            var.cf_name
            for var in (
                coord_var,
                conn_var,
                ugrid_coord_var,
                mesh_var,
                mesh_var2,
            )
        ]
        expected = set(expected_names)
        assert self.cf_group.non_data_variable_names == expected
