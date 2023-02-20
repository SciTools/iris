# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.build_cell_measure`.

"""

from unittest import mock

import numpy as np
import pytest

from iris.exceptions import CannotAddError
from iris.fileformats._nc_load_rules.helpers import build_cell_measures


@pytest.fixture
def mock_engine():
    return mock.Mock(
        cube=mock.Mock(),
        cf_var=mock.Mock(dimensions=("foo", "bar")),
        filename="DUMMY",
        cube_parts=dict(cell_measures=[]),
    )


@pytest.fixture
def mock_cf_cm_var(monkeypatch):
    data = np.arange(6)
    output = mock.Mock(
        dimensions=("foo",),
        scale_factor=1,
        add_offset=0,
        cf_name="wibble",
        cf_data=mock.MagicMock(chunking=mock.Mock(return_value=None), spec=[]),
        standard_name=None,
        long_name="wibble",
        units="m2",
        shape=data.shape,
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


def test_not_added(monkeypatch, mock_engine, mock_cf_cm_var):
    # Confirm that the cell measure will be skipped if a CannotAddError is
    #  raised when attempting to add.
    def mock_add_cell_measure(_, __):
        raise CannotAddError("foo")

    with monkeypatch.context() as m:
        m.setattr(mock_engine.cube, "add_cell_measure", mock_add_cell_measure)
        with pytest.warns(match="cell measure not added to Cube: foo"):
            build_cell_measures(mock_engine, mock_cf_cm_var)

    assert mock_engine.cube_parts["cell_measures"] == []
