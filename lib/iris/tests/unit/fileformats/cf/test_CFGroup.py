# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.fileformats.cf.CFGroup` class."""

from typing import Callable
from unittest.mock import MagicMock

import pytest

from iris.fileformats import cf

VariableMap: dict[type[cf.CFVariable], Callable] = {
    cf._CFFormulaTermsVariable: cf.CFGroup.formula_terms,
    cf.CFAncillaryDataVariable: cf.CFGroup.ancillary_variables,
    cf.CFAuxiliaryCoordinateVariable: cf.CFGroup.auxiliary_coordinates,
    cf.CFBoundaryVariable: cf.CFGroup.bounds,
    cf.CFClimatologyVariable: cf.CFGroup.climatology,
    cf.CFCoordinateVariable: cf.CFGroup.coordinates,
    cf.CFDataVariable: cf.CFGroup.data_variables,
    cf.CFGridMappingVariable: cf.CFGroup.grid_mappings,
    cf.CFLabelVariable: cf.CFGroup.labels,
    cf.CFMeasureVariable: cf.CFGroup.cell_measures,
    cf.CFUGridAuxiliaryCoordinateVariable: cf.CFGroup.ugrid_coords,
    cf.CFUGridConnectivityVariable: cf.CFGroup.connectivities,
    cf.CFUGridMeshVariable: cf.CFGroup.meshes,
}


def get_mocked_var(class_: type[cf.CFVariable], mocker) -> MagicMock:
    cf_name = f"{class_.__name__}_var"
    mocked = mocker.MagicMock(spec=class_, cf_name=cf_name)
    return mocked


@pytest.fixture(
    params=[
        (class_, scenario)
        for class_ in VariableMap
        for scenario in ("single", "duplicates", "multiple")
    ],
    ids=lambda value: f"{value[0].__name__}-{value[1]}",
)
def mock_variables(
    request, mocker
) -> tuple[type[cf.CFVariable], tuple[MagicMock, ...]]:
    class_, scenario = request.param
    mocked = get_mocked_var(class_, mocker)
    variables = [mocked]

    if scenario == "duplicates":
        variables.append(get_mocked_var(class_, mocker))
    elif scenario == "multiple":
        mocked_2 = get_mocked_var(class_, mocker)
        mocked_2.cf_name = f"{mocked.cf_name}_2"
        variables.append(mocked_2)

    return class_, tuple(variables)


@pytest.fixture(params=["single", "duplicates", "multiple"])
def mock_variables_all(request, mocker) -> tuple[MagicMock, ...]:
    scenario = request.param
    all_variables = []

    for class_ in VariableMap:
        mocked = get_mocked_var(class_, mocker)
        variables = [mocked]

        if scenario == "duplicates":
            variables.append(get_mocked_var(class_, mocker))
        elif scenario == "multiple":
            mocked_2 = get_mocked_var(class_, mocker)
            mocked_2.cf_name = f"{mocked.cf_name}_2"
            variables.append(mocked_2)

        all_variables.extend(variables)

    return tuple(all_variables)


@pytest.fixture
def cf_group() -> cf.CFGroup:
    return cf.CFGroup()


@pytest.fixture
def cf_group_populated(cf_group, mock_variables_all) -> cf.CFGroup:
    for mocked in mock_variables_all:
        cf_group[mocked.cf_name] = mocked
    return cf_group


class TestProperties:
    def test_common(self, cf_group, mock_variables):
        class_, variables = mock_variables
        for mocked in variables:
            cf_group[mocked.cf_name] = mocked

        property_ = VariableMap[class_]
        result = property_.fget(cf_group)
        expected_names = {var.cf_name for var in variables}
        assert len(result) == len(expected_names)
        for expected_name in expected_names:
            assert expected_name in result

    def test_non_data_names(self, cf_group_populated, mock_variables_all):
        expected = {
            mocked.cf_name
            for mocked in mock_variables_all
            if not isinstance(mocked, (cf.CFDataVariable, cf._CFFormulaTermsVariable))
        }
        assert cf_group_populated.non_data_variable_names == expected


class TestReturns:
    @pytest.fixture(autouse=True)
    def _setup(self, cf_group_populated, mock_variables_all):
        self.variables = mock_variables_all
        self.cf_group = cf_group_populated

    def test_keys(self):
        expected_names = {mocked.cf_name for mocked in self.variables}
        assert set(self.cf_group.keys()) == expected_names

    def test_len(self):
        expected_names = {mocked.cf_name for mocked in self.variables}
        assert len(self.cf_group) == len(expected_names)

    def test_iter(self):
        expected_names = {mocked.cf_name for mocked in self.variables}
        actual_names = set()
        for name in self.cf_group:
            actual_names.add(name)

        assert actual_names == expected_names

    def test_getitem(self):
        expected_by_name = {}
        for mocked in self.variables:
            expected_by_name[mocked.cf_name] = mocked

        for expected_name, expected_variable in expected_by_name.items():
            assert self.cf_group[expected_name] is expected_variable

        with pytest.raises(KeyError, match="Cannot get unknown CF-netCDF variable"):
            _ = self.cf_group["unknown_name"]

    def test_repr(self):
        self.cf_group.global_attributes["global_attr"] = "value"
        self.cf_group.promoted["promoted_var"] = self.variables[0]

        expected_names = {mocked.cf_name for mocked in self.variables}
        expected = (
            "<CFGroup of "
            f"variables:{len(expected_names)}, "
            "global_attributes:1, "
            "promoted:1>"
        )
        assert repr(self.cf_group) == expected


class TestMutations:
    def test_setitem(self, cf_group, mock_variables_all):
        mocked = mock_variables_all[0]

        cf_group[mocked.cf_name] = mocked
        assert cf_group[mocked.cf_name] is mocked

        with pytest.raises(TypeError, match="Attempted to add an invalid"):
            cf_group[mocked.cf_name] = object()

        with pytest.raises(ValueError, match="Mismatch between key name"):
            cf_group[f"{mocked.cf_name}_mismatch"] = mocked

    def test_delitem(self, cf_group_populated, mock_variables_all):
        expected_names = {mocked.cf_name for mocked in mock_variables_all}
        name_to_delete = next(iter(expected_names))

        del cf_group_populated[name_to_delete]
        assert name_to_delete not in cf_group_populated
        assert len(cf_group_populated) == len(expected_names) - 1

        with pytest.raises(KeyError, match="Cannot delete unknown CF-netcdf"):
            del cf_group_populated["unknown_name"]
