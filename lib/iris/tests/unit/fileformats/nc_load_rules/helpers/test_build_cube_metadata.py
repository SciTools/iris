# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers\
build_cube_metadata`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.cube import Cube
from iris.fileformats._nc_load_rules.helpers import build_cube_metadata


def _make_engine(global_attributes=None, standard_name=None, long_name=None):
    if global_attributes is None:
        global_attributes = {}

    cf_group = mock.Mock(global_attributes=global_attributes)

    cf_var = mock.MagicMock(
        cf_name="wibble",
        standard_name=standard_name,
        long_name=long_name,
        units="m",
        dtype=np.float64,
        cell_methods=None,
        cf_group=cf_group,
    )

    engine = mock.Mock(cube=Cube([23]), cf_var=cf_var, filename="foo.nc")

    return engine


class TestGlobalAttributes(tests.IrisTest):
    def test_valid(self):
        global_attributes = {
            "Conventions": "CF-1.5",
            "comment": "Mocked test object",
        }
        engine = _make_engine(global_attributes)
        build_cube_metadata(engine)
        expected = global_attributes
        self.assertEqual(engine.cube.attributes.globals, expected)

    def test_invalid(self):
        global_attributes = {
            "Conventions": "CF-1.5",
            "comment": "Mocked test object",
            "calendar": "standard",
        }
        engine = _make_engine(global_attributes)
        with mock.patch("warnings.warn") as warn:
            build_cube_metadata(engine)
        # Check for a warning.
        self.assertEqual(warn.call_count, 1)
        self.assertIn(
            "Skipping disallowed global attribute 'calendar'",
            warn.call_args[0][0],
        )
        # Check resulting attributes. The invalid entry 'calendar'
        # should be filtered out.
        global_attributes.pop("calendar")
        expected = global_attributes
        self.assertEqual(engine.cube.attributes.globals, expected)


if __name__ == "__main__":
    tests.main()
