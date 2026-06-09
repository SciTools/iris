# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.build_raw_cube`."""

import numpy as np
import pytest

from iris.common import LimitedAttributeDict
from iris.cube import Cube
from iris.fileformats._nc_load_rules.helpers import build_raw_cube
from iris.fileformats.cf import CFVariable


def _make_array_and_cf_data(mocker, dim_lens: dict[str, int]):
    shape = list(dim_lens.values())
    cf_data = mocker.MagicMock(_FillValue=None, spec=[])
    cf_data.chunking = mocker.MagicMock(return_value=shape)
    data = np.arange(np.prod(shape), dtype=float)
    data = data.reshape(shape)
    return data, cf_data


def cf_attrs():
    return tuple(
        [
            # standard_name is normally forbidden as a basic attribute - expect to
            #  see under IRIS_RAW.
            ("standard_name", "air_temperature"),
            ("my_attribute", "my_value"),
        ]
    )


@pytest.fixture
def cf_variable(mocker):
    dim_lens = {"foo": 3, "bar": 4}
    data, cf_data = _make_array_and_cf_data(mocker, dim_lens)

    cf_var = mocker.MagicMock(
        spec=CFVariable,
        cf_name="wibble",
        cf_attrs=cf_attrs,
        # Minimum attributes to enable data getting.
        dimensions=list(dim_lens.keys()),
        cf_data=cf_data,
        filename="foo.nc",
        shape=data.shape,
        size=data.size,
        dtype=data.dtype,
        __getitem__=lambda self, key: data[key],
    )

    return cf_var


@pytest.fixture
def expected_cube(mocker, cf_variable):
    dim_lens = {k: v for k, v in zip(cf_variable.dimensions, cf_variable.shape)}
    expected_data, _ = _make_array_and_cf_data(mocker, dim_lens)

    raw_attributes = {k: v for k, v in cf_attrs()}
    raw_attributes["var_name"] = cf_variable.cf_name
    return Cube(
        data=expected_data, attributes={LimitedAttributeDict.IRIS_RAW: raw_attributes}
    )


def test(cf_variable, expected_cube):
    result = build_raw_cube(cf_variable)
    assert result == expected_cube
