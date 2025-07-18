# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.build_and_add_cell_methods`."""

from unittest import mock

import pytest

from iris.coords import CellMethod
from iris.cube import Cube
from iris.fileformats._nc_load_rules import helpers
from iris.fileformats.cf import CFDataVariable
from iris.loading import LOAD_PROBLEMS


@pytest.fixture
def mock_cf_data_var():
    yield mock.Mock(
        spec=CFDataVariable,
        cell_methods="time: mean",
        cf_name="wibble",
        filename="DUMMY",
    )


@pytest.fixture
def mock_engine(mock_cf_data_var):
    yield mock.Mock(
        cube=mock.Mock(),
        cf_var=mock_cf_data_var,
        filename=mock_cf_data_var.filename,
    )


def test_construction(mock_engine):
    expected_method = CellMethod("mean", coords=["time"])
    helpers.build_and_add_cell_methods(mock_engine)
    assert mock_engine.cube.cell_methods == (expected_method,)


def test_not_built(monkeypatch, mock_engine, mock_cf_data_var):
    cm_original = mock_engine.cube.cell_methods

    def mock_parse_cell_methods(nc_cell_methods, cf_name=None):
        raise RuntimeError("Not built")

    with monkeypatch.context() as m:
        m.setattr(helpers, "parse_cell_methods", mock_parse_cell_methods)
        helpers.build_and_add_cell_methods(mock_engine)

    load_problem = LOAD_PROBLEMS.problems[-1]
    assert "Not built" in "".join(load_problem.stack_trace.format())
    assert mock_engine.cube.cell_methods == cm_original


def test_not_added(monkeypatch, mock_engine, mock_cf_data_var):
    cm_original = mock_engine.cube.cell_methods

    class NoCellMethods(mock.Mock):
        def __setattr__(self, key, value):
            if key == "cell_methods":
                raise RuntimeError("Not added")
            super().__setattr__(key, value)

    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", NoCellMethods())
        helpers.build_and_add_cell_methods(mock_engine)

    load_problem = LOAD_PROBLEMS.problems[-1]
    assert "Not added" in "".join(load_problem.stack_trace.format())
    assert mock_engine.cube.cell_methods == cm_original


def test_unhandlable_error(monkeypatch, mock_engine):
    # Confirm that the code can redirect an error to LOAD_PROBLEMS even
    #  when there is no specific handling code for it.
    n_problems = len(LOAD_PROBLEMS.problems)

    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", "foo")
        helpers.build_and_add_cell_methods(mock_engine)

    assert len(LOAD_PROBLEMS.problems) > n_problems


def test_problem_destination(monkeypatch, mock_engine):
    # Confirm that the destination of the problem is set correctly.
    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", "foo")
        helpers.build_and_add_cell_methods(mock_engine)

    destination = LOAD_PROBLEMS.problems[-1].destination
    assert destination.iris_class is Cube
    assert destination.identifier == mock_engine.cf_var.cf_name
