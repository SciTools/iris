# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the class
:class:`iris.fileformats.um._fast_load.FieldCollation`.

This only tests the additional functionality for recording file locations of
PPFields that make loaded cubes.
The original class is the baseclass of this, now renamed 'BasicFieldCollation'.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

import iris
from iris.tests.integration.fast_load.test_fast_load import Mixin_FieldTest


class TestFastCallbackLocationInfo(Mixin_FieldTest, tests.IrisTest):
    do_fast_loads = True

    def setUp(self):
        # Call parent setup.
        super().setUp()

        # Create a basic load test case.
        self.callback_collations = []
        self.callback_filepaths = []

        def fast_load_callback(cube, collation, filename):
            self.callback_collations.append(collation)
            self.callback_filepaths.append(filename)

        flds = self.fields(c_t="11112222", c_h="11221122", phn="01010101")
        self.test_filepath = self.save_fieldcubes(flds)
        iris.load(self.test_filepath, callback=fast_load_callback)

    def test_callback_collations_filepaths(self):
        self.assertEqual(len(self.callback_collations), 2)
        self.assertEqual(
            self.callback_collations[0].data_filepath, self.test_filepath
        )
        self.assertEqual(
            self.callback_collations[1].data_filepath, self.test_filepath
        )

    def test_callback_collations_field_indices(self):
        self.assertEqual(
            self.callback_collations[0].data_field_indices.dtype, np.int64
        )
        self.assertArrayEqual(
            self.callback_collations[0].data_field_indices, [[1, 3], [5, 7]]
        )

        self.assertEqual(
            self.callback_collations[1].data_field_indices.dtype, np.int64
        )
        self.assertArrayEqual(
            self.callback_collations[1].data_field_indices, [[0, 2], [4, 6]]
        )


if __name__ == "__main__":
    tests.main()
