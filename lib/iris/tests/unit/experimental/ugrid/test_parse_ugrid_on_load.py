# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :func:`iris.experimental.ugrid.parse_ugrid_on_load` function.

todo: remove this module when experimental.ugrid is folded into standard behaviour.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.experimental.ugrid import (
    load_cubes as ugrid_load_cubes,
    parse_ugrid_on_load,
)
from iris.fileformats.netcdf import load_cubes as nc_load_cubes
from iris.fileformats.pp import load_cubes as pp_load_cubes
from iris.io.format_picker import FileElement, FormatSpecification


class TestFlag(tests.IrisTest):
    def test_default(self):
        from iris.experimental.ugrid import _PARSE_UGRID_ON_LOAD

        self.assertFalse(_PARSE_UGRID_ON_LOAD)

    def test_set(self):
        with parse_ugrid_on_load():
            from iris.experimental.ugrid import _PARSE_UGRID_ON_LOAD

            self.assertTrue(_PARSE_UGRID_ON_LOAD)
        from iris.experimental.ugrid import _PARSE_UGRID_ON_LOAD

        self.assertFalse(_PARSE_UGRID_ON_LOAD)


class TestFormatSpecification(tests.IrisTest):
    def setUp(self):
        file_element = tests.mock.Mock(__class__=FileElement)
        self.format_spec_nc = FormatSpecification(
            "nc", file_element, 0, handler=nc_load_cubes
        )
        self.format_spec_pp = FormatSpecification(
            "pp", file_element, 0, handler=pp_load_cubes
        )

    def test_default(self):
        self.assertEqual(pp_load_cubes, self.format_spec_pp.handler)

        self.assertEqual(nc_load_cubes, self.format_spec_nc.handler)
        self.assertNotEqual(ugrid_load_cubes, self.format_spec_nc.handler)

    def test_set(self):
        with parse_ugrid_on_load():
            self.assertEqual(pp_load_cubes, self.format_spec_pp.handler)

            self.assertNotEqual(nc_load_cubes, self.format_spec_nc.handler)
            self.assertEqual(ugrid_load_cubes, self.format_spec_nc.handler)

        self.assertEqual(pp_load_cubes, self.format_spec_pp.handler)

        self.assertEqual(nc_load_cubes, self.format_spec_nc.handler)
        self.assertNotEqual(ugrid_load_cubes, self.format_spec_nc.handler)
