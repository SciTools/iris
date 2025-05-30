# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.build_and_add_units`."""

from unittest import mock

from cf_units import Unit
import pytest

from iris.cube import Cube
from iris.fileformats._nc_load_rules import helpers
from iris.fileformats.cf import CFDataVariable
from iris.loading import LOAD_PROBLEMS


@pytest.fixture
def mock_cf_data_var():
    yield mock.Mock(
        spec=CFDataVariable,
        units="kelvin",
        cf_name="wibble",
        filename="DUMMY",
        dtype=float,
        cf_data=mock.Mock(spec=[]),
    )


@pytest.fixture
def mock_engine(mock_cf_data_var):
    yield mock.Mock(
        cube=mock.Mock(attributes={}),
        cf_var=mock_cf_data_var,
        filename=mock_cf_data_var.filename,
    )


def test_construction(mock_engine):
    expected_units = Unit("kelvin")
    helpers.build_and_add_units(mock_engine)
    assert mock_engine.cube.units == expected_units


def test_invalid_units(mock_engine, mock_cf_data_var):
    mock_cf_data_var.units = "not_built"
    helpers.build_and_add_units(mock_engine)
    assert mock_engine.cube.attributes["invalid_units"] == "not_built"
    load_problem = LOAD_PROBLEMS.problems[-1]
    assert load_problem.loaded == {"units": "not_built"}


def test_not_built(monkeypatch, mock_engine):
    units_original = mock_engine.cube.units

    def mock_get_attr_units(cf_var, attributes, capture_invalid=False):
        raise RuntimeError("Not built")

    with monkeypatch.context() as m:
        m.setattr(helpers, "get_attr_units", mock_get_attr_units)
        helpers.build_and_add_units(mock_engine)

    load_problem = LOAD_PROBLEMS.problems[-1]
    assert "Not built" in "".join(load_problem.stack_trace.format())
    assert mock_engine.cube.units == units_original


def test_not_added(monkeypatch, mock_engine, mock_cf_data_var):
    units_original = mock_engine.cube.units

    class NoUnits(mock.Mock):
        def __setattr__(self, key, value):
            if key == "units":
                raise RuntimeError("Not added")
            super().__setattr__(key, value)

    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", NoUnits())
        helpers.build_and_add_units(mock_engine)

    load_problem = LOAD_PROBLEMS.problems[-1]
    assert "Not added" in "".join(load_problem.stack_trace.format())
    assert mock_engine.cube.units == units_original


def test_unhandlable_error(monkeypatch, mock_engine):
    # Confirm that the code can redirect an error to LOAD_PROBLEMS even
    #  when there is no specific handling code for it.
    n_problems = len(LOAD_PROBLEMS.problems)

    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", "foo")
        helpers.build_and_add_units(mock_engine)

    assert len(LOAD_PROBLEMS.problems) > n_problems


def test_problem_destination(monkeypatch, mock_engine):
    # Confirm that the destination of the problem is set correctly.
    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", "foo")
        helpers.build_and_add_units(mock_engine)

        destination = LOAD_PROBLEMS.problems[-1].destination
        assert destination.iris_class is Cube
        assert destination.identifier == mock_engine.cf_var.cf_name

    assert mock_engine.cube.attributes == {}
