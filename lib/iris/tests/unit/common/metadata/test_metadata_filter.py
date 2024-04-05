# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.common.metadata_filter`."""

import numpy as np

from iris.common.metadata import CoordMetadata, DimCoordMetadata, metadata_filter
from iris.coords import AuxCoord

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

Mock = tests.mock.Mock


class Test_standard(tests.IrisTest):
    def test_instances_non_iterable(self):
        item = Mock()
        item.name.return_value = "one"
        result = metadata_filter(item, item="one")
        self.assertEqual(1, len(result))
        self.assertIn(item, result)

    def test_name(self):
        name_one = Mock()
        name_one.name.return_value = "one"
        name_two = Mock()
        name_two.name.return_value = "two"
        input_list = [name_one, name_two]
        result = metadata_filter(input_list, item="one")
        self.assertIn(name_one, result)
        self.assertNotIn(name_two, result)

    def test_item(self):
        coord = Mock(__class__=AuxCoord)
        mock = Mock()
        input_list = [coord, mock]
        result = metadata_filter(input_list, item=coord)
        self.assertIn(coord, result)
        self.assertNotIn(mock, result)

    def test_item_metadata(self):
        coord = Mock(metadata=CoordMetadata)
        dim_coord = Mock(metadata=DimCoordMetadata)
        input_list = [coord, dim_coord]
        result = metadata_filter(input_list, item=coord)
        self.assertIn(coord, result)
        self.assertNotIn(dim_coord, result)

    def test_standard_name(self):
        name_one = Mock(standard_name="one")
        name_two = Mock(standard_name="two")
        input_list = [name_one, name_two]
        result = metadata_filter(input_list, standard_name="one")
        self.assertIn(name_one, result)
        self.assertNotIn(name_two, result)

    def test_long_name(self):
        name_one = Mock(long_name="one")
        name_two = Mock(long_name="two")
        input_list = [name_one, name_two]
        result = metadata_filter(input_list, long_name="one")
        self.assertIn(name_one, result)
        self.assertNotIn(name_two, result)

    def test_var_name(self):
        name_one = Mock(var_name="one")
        name_two = Mock(var_name="two")
        input_list = [name_one, name_two]
        result = metadata_filter(input_list, var_name="one")
        self.assertIn(name_one, result)
        self.assertNotIn(name_two, result)

    def test_attributes(self):
        # Confirm that this can handle attrib dicts including np arrays.
        attrib_one_two = Mock(attributes={"one": np.arange(1), "two": np.arange(2)})
        attrib_three_four = Mock(
            attributes={"three": np.arange(3), "four": np.arange(4)}
        )
        input_list = [attrib_one_two, attrib_three_four]
        result = metadata_filter(input_list, attributes=attrib_one_two.attributes)
        self.assertIn(attrib_one_two, result)
        self.assertNotIn(attrib_three_four, result)

    def test_invalid_attributes(self):
        attrib_one = Mock(attributes={"one": 1})
        input_list = [attrib_one]
        self.assertRaisesRegex(
            ValueError,
            ".*expecting a dictionary.*",
            metadata_filter,
            input_list,
            attributes="one",
        )

    def test_axis__by_guess(self):
        # see https://docs.python.org/3/library/unittest.mock.html#deleting-attributes
        axis_lon = Mock(standard_name="longitude")
        del axis_lon.axis
        axis_lat = Mock(standard_name="latitude")
        del axis_lat.axis
        input_list = [axis_lon, axis_lat]
        result = metadata_filter(input_list, axis="x")
        self.assertIn(axis_lon, result)
        self.assertNotIn(axis_lat, result)

    def test_axis__by_member(self):
        axis_x = Mock(axis="x")
        axis_y = Mock(axis="y")
        input_list = [axis_x, axis_y]
        result = metadata_filter(input_list, axis="x")
        self.assertEqual(1, len(result))
        self.assertIn(axis_x, result)

    def test_multiple_args(self):
        coord_one = Mock(__class__=AuxCoord, long_name="one")
        coord_two = Mock(__class__=AuxCoord, long_name="two")
        input_list = [coord_one, coord_two]
        result = metadata_filter(input_list, item=coord_one, long_name="one")
        self.assertIn(coord_one, result)
        self.assertNotIn(coord_two, result)
