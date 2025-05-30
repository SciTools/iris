# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.build_and_add_cell_measure`."""

from unittest import mock

import numpy as np
import pytest

from iris.coords import CellMeasure
from iris.cube import Cube
from iris.exceptions import CannotAddError
from iris.fileformats._nc_load_rules.helpers import build_and_add_cell_measure
from iris.fileformats.cf import CFMeasureVariable
from iris.loading import LOAD_PROBLEMS


@pytest.fixture
def mock_engine():
    return mock.Mock(
        cube=mock.Mock(),
        cf_var=mock.Mock(dimensions=("foo", "bar")),
        filename="DUMMY",
        cube_parts=dict(cell_measures=[]),
    )


@pytest.fixture
def mock_cf_cm_var(monkeypatch, mock_engine):
    data = np.arange(6)
    output = mock.Mock(
        spec=CFMeasureVariable,
        dimensions=("foo",),
        scale_factor=1,
        add_offset=0,
        cf_name="wibble",
        cf_data=mock.MagicMock(chunking=mock.Mock(return_value=None), spec=[]),
        filename=mock_engine.filename,
        standard_name=None,
        long_name="wibble",
        units="m2",
        shape=data.shape,
        size=np.prod(data.shape),
        dtype=data.dtype,
        __getitem__=lambda self, key: data[key],
        cf_measure="area",
    )

    # Create patch for deferred loading that prevents attempted
    # file access. This assumes that output is defined in the test case.
    def patched__getitem__(proxy_self, keys):
        if proxy_self.variable_name == output.cf_name:
            return output[keys]
        raise RuntimeError()

    monkeypatch.setattr(
        "iris.fileformats.netcdf.NetCDFDataProxy.__getitem__",
        patched__getitem__,
    )

    return output


def test_construction(mock_engine, mock_cf_cm_var):
    expected_cm = CellMeasure(
        mock_cf_cm_var[:],
        long_name=mock_cf_cm_var.long_name,
        var_name=mock_cf_cm_var.cf_name,
        units=mock_cf_cm_var.units,
    )

    build_and_add_cell_measure(mock_engine, mock_cf_cm_var)

    # Test that expected coord is built and added to cube.
    mock_engine.cube.add_cell_measure.assert_called_with(expected_cm, [0])


def test_not_added(monkeypatch, mock_engine, mock_cf_cm_var):
    # Confirm that the cell measure will be skipped if a CannotAddError is
    #  raised when attempting to add.
    def mock_add_cell_measure(_, __):
        raise CannotAddError("foo")

    with monkeypatch.context() as m:
        m.setattr(mock_engine.cube, "add_cell_measure", mock_add_cell_measure)
        build_and_add_cell_measure(mock_engine, mock_cf_cm_var)

        load_problem = LOAD_PROBLEMS.problems[-1]
        assert load_problem.stack_trace.exc_type is CannotAddError

    assert mock_engine.cube_parts["cell_measures"] == []


def test_unhandlable_error(monkeypatch, mock_engine, mock_cf_cm_var):
    # Confirm that the code can redirect an error to LOAD_PROBLEMS even
    #  when there is no specific handling code for it.
    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", "foo")
        n_problems = len(LOAD_PROBLEMS.problems)
        build_and_add_cell_measure(mock_engine, mock_cf_cm_var)
        assert len(LOAD_PROBLEMS.problems) > n_problems

    assert mock_engine.cube_parts["cell_measures"] == []


def test_problem_destination(monkeypatch, mock_engine, mock_cf_cm_var):
    # Confirm that the destination of the problem is set correctly.
    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", "foo")
        build_and_add_cell_measure(mock_engine, mock_cf_cm_var)

        destination = LOAD_PROBLEMS.problems[-1].destination
        assert destination.iris_class is Cube
        assert destination.identifier == mock_engine.cf_var.cf_name

    assert mock_engine.cube_parts["cell_measures"] == []
