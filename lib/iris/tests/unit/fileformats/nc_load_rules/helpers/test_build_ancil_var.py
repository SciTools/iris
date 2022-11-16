# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.build_ancil_var`.

"""

from unittest import mock

import numpy as np
import pytest

from iris.exceptions import CannotAddError
from iris.fileformats._nc_load_rules.helpers import build_ancil_var


@pytest.fixture
def mock_engine():
    return mock.Mock(
        cube=mock.Mock(),
        cf_var=mock.Mock(dimensions=("foo", "bar")),
        filename="DUMMY",
        cube_parts=dict(ancillary_variables=[]),
    )


@pytest.fixture
def mock_cf_av_var(monkeypatch):
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


def test_not_added(monkeypatch, mock_engine, mock_cf_av_var):
    # Confirm that the ancillary variable will be skipped if a CannotAddError
    #  is raised when attempting to add.
    def mock_add_ancillary_variable(_, __):
        raise CannotAddError("foo")

    with monkeypatch.context() as m:
        m.setattr(
            mock_engine.cube,
            "add_ancillary_variable",
            mock_add_ancillary_variable,
        )
        with pytest.warns(match="ancillary variable not added to Cube: foo"):
            build_ancil_var(mock_engine, mock_cf_av_var)

    assert mock_engine.cube_parts["ancillary_variables"] == []
