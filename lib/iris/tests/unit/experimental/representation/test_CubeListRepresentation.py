# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.cube.CubeRepresentation` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from html import escape

from iris.cube import CubeList
from iris.experimental.representation import CubeListRepresentation
import iris.tests.stock as stock


@tests.skip_data
class Test__instantiation(tests.IrisTest):
    def setUp(self):
        self.cubes = CubeList([stock.simple_3d()])
        self.representer = CubeListRepresentation(self.cubes)

    def test_ids(self):
        self.assertEqual(id(self.cubes), self.representer.cubelist_id)


@tests.skip_data
class Test_make_content(tests.IrisTest):
    def setUp(self):
        self.cubes = CubeList([stock.simple_3d(), stock.lat_lon_cube()])
        self.cubes[0].rename("name & <html>")
        self.representer = CubeListRepresentation(self.cubes)
        self.content = self.representer.make_content()

    def test_repr_len(self):
        self.assertEqual(len(self.cubes), len(self.content))

    def test_summary_lines(self):
        names = [c.name() for c in self.cubes]
        for name, content in zip(names, self.content):
            name = escape(name)
            self.assertIn(name, content)

    def test__cube_name_summary_consistency(self):
        # Just check the first cube in the CubeList.
        single_cube_html = self.content[0]
        # Get a "prettified" cube name, as it should be in the cubelist repr.
        cube_name = self.cubes[0].name()
        pretty_cube_name = cube_name.strip().replace("_", " ").title()
        pretty_escaped_name = escape(pretty_cube_name)
        self.assertIn(pretty_escaped_name, single_cube_html)


@tests.skip_data
class Test_repr_html(tests.IrisTest):
    def setUp(self):
        self.cubes = CubeList([stock.simple_3d(), stock.lat_lon_cube()])
        self.representer = CubeListRepresentation(self.cubes)

    def test_html_length(self):
        html = self.representer.repr_html()
        n_html_elems = html.count("<button")  # One <button> tag per cube.
        self.assertEqual(len(self.cubes), n_html_elems)


if __name__ == "__main__":
    tests.main()
