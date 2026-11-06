# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the class
:class:`iris.fileformats.um._fast_load.FieldCollation`.

This only tests the additional functionality for recording file locations of
PPFields that make loaded cubes.
The original class is the baseclass of this, now renamed 'BasicFieldCollation'.

"""

import numpy as np
import pytest

import iris
from iris.tests import _shared_utils
from iris.tests.integration.fast_load.test_fast_load import Mixin_FieldTest


class TestFastCallbackLocationInfo(Mixin_FieldTest):
    do_fast_loads = True

    @pytest.fixture(autouse=True)
    def _setup(self):
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
        assert len(self.callback_collations) == 2
        assert self.callback_collations[0].data_filepath == self.test_filepath
        assert self.callback_collations[1].data_filepath == self.test_filepath

    def test_callback_collations_field_indices(self):
        assert self.callback_collations[0].data_field_indices.dtype == np.int64
        _shared_utils.assert_array_equal(
            self.callback_collations[0].data_field_indices, [[1, 3], [5, 7]]
        )

        assert self.callback_collations[1].data_field_indices.dtype == np.int64
        _shared_utils.assert_array_equal(
            self.callback_collations[1].data_field_indices, [[0, 2], [4, 6]]
        )
