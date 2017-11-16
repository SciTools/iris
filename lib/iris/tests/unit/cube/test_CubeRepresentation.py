# (C) British Crown Copyright 2017, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the `iris.cube.CubeRepresentation` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.coords import CellMethod
from iris.cube import _CubeRepresentation
import iris.tests.stock as stock


class Test__instantiation(tests.IrisTest):
    def setUp(self):
        self.cube = stock.simple_3d()
        self.representer = _CubeRepresentation(self.cube)

    def test_cube_attributes(self):
        self.assertEqual(id(self.cube), self.representer.cube_id)
        self.assertStringEqual(str(self.cube), self.representer.cube_str)

    def test_summary(self):
        self.assertIsNone(self.representer.summary)

    def test__heading_contents(self):
        content = set(self.representer.str_headings.values())
        self.assertEqual(len(content), 1)
        self.assertIsNone(list(content)[0])


class Test__get_bits(tests.IrisTest):
    def setUp(self):
        self.cube = stock.realistic_4d()
        cm = CellMethod('mean', 'time', '6hr')
        self.cube.add_cell_method(cm)
        self.representer = _CubeRepresentation(self.cube)
        self.representer._get_bits()
        self.summary = self.representer.summary

    def test_population(self):
        self.assertIsNotNone(self.summary)
        for v in self.representer.str_headings.values():
            self.assertIsNotNone(v)

    def test_summary(self):
        expected = self.cube.summary(True)
        result = self.summary
        self.assertStringEqual(expected, result)

    def test_headings__dimcoords(self):
        contents = self.representer.str_headings['Dimension coordinates:']
        content_str = ','.join(content for content in contents)
        dim_coords = [c.name() for c in self.cube.dim_coords]
        for coord in dim_coords:
            self.assertIn(coord, content_str)

    def test_headings__auxcoords(self):
        contents = self.representer.str_headings['Auxiliary coordinates:']
        content_str = ','.join(content for content in contents)
        aux_coords = [c.name() for c in self.cube.aux_coords
                      if c.shape != (1,)]
        for coord in aux_coords:
            self.assertIn(coord, content_str)

    def test_headings__derivedcoords(self):
        contents = self.representer.str_headings['Auxiliary coordinates:']
        content_str = ','.join(content for content in contents)
        derived_coords = [c.name() for c in self.cube.derived_coords]
        for coord in derived_coords:
            self.assertIn(coord, content_str)

    def test_headings__scalarcoords(self):
        contents = self.representer.str_headings['Scalar coordinates:']
        content_str = ','.join(content for content in contents)
        scalar_coords = [c.name() for c in self.cube.coords()
                         if c.shape == (1,)]
        for coord in scalar_coords:
            self.assertIn(coord, content_str)

    def test_headings__attributes(self):
        contents = self.representer.str_headings['Attributes:']
        content_str = ','.join(content for content in contents)
        for attr_name, attr_value in self.cube.attributes.items():
            self.assertIn(attr_name, content_str)
            self.assertIn(attr_value, content_str)

    def test_headings__cellmethods(self):
        contents = self.representer.str_headings['Cell methods:']
        content_str = ','.join(content for content in contents)
        for cell_method in self.cube.cell_methods:
            self.assertIn(str(cell_method), content_str)


class Test__make_content(tests.IrisTest):
    def setUp(self):
        self.cube = stock.simple_3d()
        self.representer = _CubeRepresentation(self.cube)
        self.representer._get_bits()
        self.result = self.representer._make_content()

    def test_included(self):
        included = 'Dimension coordinates:'
        self.assertIn(included, self.result)
        dim_coord_names = [c.name() for c in self.cube.dim_coords]
        for coord_name in dim_coord_names:
            self.assertIn(coord_name, self.result)

    def test_not_included(self):
        # `stock.simple_3d()` only contains the `Dimension coordinates` attr.
        not_included = list(self.representer.str_headings.keys())
        not_included.pop(not_included.index('Dimension coordinates:'))
        for heading in not_included:
            self.assertNotIn(heading, self.result)


class Test_repr_html(tests.IrisTest):
    def setUp(self):
        self.cube = stock.simple_3d()
        representer = _CubeRepresentation(self.cube)
        self.result = representer.repr_html()

    def test_summary_added(self):
        self.assertIn(self.cube.summary(True), self.result)

    def test_contents_added(self):
        included = 'Dimension coordinates:'
        self.assertIn(included, self.result)
        not_included = 'Auxiliary coordinates:'
        self.assertNotIn(not_included, self.result)


if __name__ == '__main__':
    tests.main()
