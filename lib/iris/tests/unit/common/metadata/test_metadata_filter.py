# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.common.metadata_filter`."""

import numpy as np
import pytest

from iris.common.metadata import CoordMetadata, DimCoordMetadata, metadata_filter
from iris.coords import AuxCoord


class Test_standard:
    def test_instances_non_iterable(self, mocker):
        item = mocker.Mock()
        item.name.return_value = "one"
        result = metadata_filter(item, item="one")
        assert len(result) == 1
        assert item in result

    def test_name(self, mocker):
        name_one = mocker.Mock()
        name_one.name.return_value = "one"
        name_two = mocker.Mock()
        name_two.name.return_value = "two"
        input_list = [name_one, name_two]
        result = metadata_filter(input_list, item="one")
        assert name_one in result
        assert name_two not in result

    def test_item(self, mocker):
        coord = mocker.Mock(__class__=AuxCoord)
        mock = mocker.Mock()
        input_list = [coord, mock]
        result = metadata_filter(input_list, item=coord)
        assert coord in result
        assert mock not in result

    def test_item_metadata(self, mocker):
        coord = mocker.Mock(metadata=CoordMetadata)
        dim_coord = mocker.Mock(metadata=DimCoordMetadata)
        input_list = [coord, dim_coord]
        result = metadata_filter(input_list, item=coord)
        assert coord in result
        assert dim_coord not in result

    def test_standard_name(self, mocker):
        name_one = mocker.Mock(standard_name="one")
        name_two = mocker.Mock(standard_name="two")
        input_list = [name_one, name_two]
        result = metadata_filter(input_list, standard_name="one")
        assert name_one in result
        assert name_two not in result

    def test_long_name(self, mocker):
        name_one = mocker.Mock(long_name="one")
        name_two = mocker.Mock(long_name="two")
        input_list = [name_one, name_two]
        result = metadata_filter(input_list, long_name="one")
        assert name_one in result
        assert name_two not in result

    def test_var_name(self, mocker):
        name_one = mocker.Mock(var_name="one")
        name_two = mocker.Mock(var_name="two")
        input_list = [name_one, name_two]
        result = metadata_filter(input_list, var_name="one")
        assert name_one in result
        assert name_two not in result

    def test_attributes(self, mocker):
        # Confirm that this can handle attrib dicts including np arrays.
        attrib_one_two = mocker.Mock(
            attributes={"one": np.arange(1), "two": np.arange(2)}
        )
        attrib_three_four = mocker.Mock(
            attributes={"three": np.arange(3), "four": np.arange(4)}
        )
        input_list = [attrib_one_two, attrib_three_four]
        result = metadata_filter(input_list, attributes=attrib_one_two.attributes)
        assert attrib_one_two in result
        assert attrib_three_four not in result

    def test_invalid_attributes(self, mocker):
        attrib_one = mocker.Mock(attributes={"one": 1})
        input_list = [attrib_one]
        emsg = ".*expecting a dictionary.*"
        with pytest.raises(ValueError, match=emsg):
            _ = metadata_filter(input_list, attributes="one")

    def test_axis__by_guess(self, mocker):
        # see https://docs.python.org/3/library/unittest.mock.html#deleting-attributes
        axis_lon = mocker.Mock(standard_name="longitude")
        del axis_lon.axis
        axis_lat = mocker.Mock(standard_name="latitude")
        del axis_lat.axis
        input_list = [axis_lon, axis_lat]
        result = metadata_filter(input_list, axis="x")
        assert axis_lon in result
        assert axis_lat not in result

    def test_axis__by_member(self, mocker):
        axis_x = mocker.Mock(axis="x")
        axis_y = mocker.Mock(axis="y")
        input_list = [axis_x, axis_y]
        result = metadata_filter(input_list, axis="x")
        assert len(result) == 1
        assert axis_x in result

    def test_multiple_args(self, mocker):
        coord_one = mocker.Mock(__class__=AuxCoord, long_name="one")
        coord_two = mocker.Mock(__class__=AuxCoord, long_name="two")
        input_list = [coord_one, coord_two]
        result = metadata_filter(input_list, item=coord_one, long_name="one")
        assert coord_one in result
        assert coord_two not in result
