# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.build_and_add_global_attributes`."""

from unittest import mock

import numpy as np
import pytest

from iris.cube import Cube
from iris.fileformats._nc_load_rules.helpers import build_and_add_global_attributes
from iris.loading import LOAD_PROBLEMS


@pytest.fixture
def mock_engine():
    global_attributes = {
        "Conventions": "CF-1.5",
        "comment": "Mocked test object",
    }
    cf_group = mock.Mock(global_attributes=global_attributes)
    cf_var = mock.MagicMock(
        cf_name="wibble",
        standard_name=None,
        long_name=None,
        units="m",
        dtype=np.float64,
        cell_methods=None,
        cf_group=cf_group,
    )
    engine = mock.Mock(cube=Cube([23]), cf_var=cf_var, filename="foo.nc")
    yield engine


def test_construction(mock_engine):
    expected = mock_engine.cf_var.cf_group.global_attributes
    build_and_add_global_attributes(mock_engine)
    assert mock_engine.cube.attributes.globals == expected


def test_not_added(mock_engine):
    attributes = mock_engine.cf_var.cf_group.global_attributes
    attributes["calendar"] = "standard"
    build_and_add_global_attributes(mock_engine)
    # Check for a load problem.
    load_problem = LOAD_PROBLEMS.problems[-1]
    assert "Skipping disallowed global attribute 'calendar'" in "".join(
        load_problem.stack_trace.format()
    )
    # Check resulting attributes. The invalid entry 'calendar' should be
    #  filtered out.
    attributes.pop("calendar")
    expected = attributes
    assert mock_engine.cube.attributes.globals == expected


def test_unhandlable_error(monkeypatch, mock_engine):
    # Confirm that the code can redirect an error to LOAD_PROBLEMS even
    #  when there is no specific handling code for it.
    n_problems = len(LOAD_PROBLEMS.problems)

    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", "foo")
        build_and_add_global_attributes(mock_engine)

    assert len(LOAD_PROBLEMS.problems) > n_problems


def test_problem_destination(monkeypatch, mock_engine):
    # Confirm that the destination of the problem is set correctly.
    with monkeypatch.context() as m:
        m.setattr(mock_engine, "cube", "foo")
        build_and_add_global_attributes(mock_engine)

    destination = LOAD_PROBLEMS.problems[-1].destination
    assert destination.iris_class is Cube
    assert destination.identifier == mock_engine.cf_var.cf_name
