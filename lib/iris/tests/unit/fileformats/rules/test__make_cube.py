# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.fileformats.rules._make_cube`."""

from unittest import mock

import numpy as np
import pytest

from iris.fileformats.rules import ConversionMetadata, _make_cube


class Test:
    def test_invalid_units(self):
        # Mock converter() function that returns an invalid
        # units string amongst the collection of other elements.
        factories = None
        references = None
        standard_name = None
        long_name = None
        units = "wibble"  # Invalid unit.
        attributes = dict(source="test")
        cell_methods = None
        dim_coords_and_dims = None
        aux_coords_and_dims = None
        metadata = ConversionMetadata(
            factories,
            references,
            standard_name,
            long_name,
            units,
            attributes,
            cell_methods,
            dim_coords_and_dims,
            aux_coords_and_dims,
        )
        converter = mock.Mock(return_value=metadata)

        data = np.arange(3.0)
        field = mock.Mock(
            core_data=lambda: data, bmdi=9999.0, realised_dtype=data.dtype
        )

        exp_emsg = "invalid units {!r}".format(units)
        with pytest.warns(match=exp_emsg):
            cube, factories, references = _make_cube(field, converter)

        # Check attributes dictionary is correctly populated.
        expected_attributes = attributes.copy()
        expected_attributes["invalid_units"] = units
        assert cube.attributes == expected_attributes
