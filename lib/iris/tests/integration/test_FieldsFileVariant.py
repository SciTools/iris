# (C) British Crown Copyright 2014, Met Office
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
"""Integration tests for loading UM FieldsFile variants."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.experimental.um import Field, Field3, FieldsFileVariant


class TestLoad(tests.IrisTest):
    def load(self):
        path = tests.get_data_path(('FF', 'n48_multi_field'))
        return FieldsFileVariant(path)

    def test_fixed_header(self):
        ffv = self.load()
        self.assertEqual(ffv.fixed_length_header.dataset_type, 3)
        self.assertEqual(ffv.fixed_length_header.lookup_shape, (64, 5))

    def test_integer_constants(self):
        ffv = self.load()
        MDI = -32768
        expected = [MDI, MDI, MDI, MDI, MDI,  # 1 - 5
                    96, 73, 70, 70, 4,        # 6 - 10
                    MDI, 70, 50, MDI, MDI,    # 11 - 15
                    MDI, 2, MDI, MDI, MDI,    # 16 - 20
                    MDI, MDI, MDI, 50, 2381,  # 21 - 25
                    MDI, MDI, 4, MDI, MDI,    # 26 - 30
                    MDI, MDI, MDI, MDI, MDI,  # 31 - 35
                    MDI, MDI, MDI, MDI, MDI,  # 36 - 40
                    MDI, MDI, MDI, MDI, MDI,  # 41 - 45
                    MDI]                      # 46
        self.assertArrayEqual(ffv.integer_constants, expected)

    def test_real_constants(self):
        ffv = self.load()
        MDI = -1073741824.0
        expected = [3.75, 2.5, -90.0, 0.0, 90.0,  # 1 - 5
                    0.0, MDI, MDI, MDI, MDI,      # 6 - 10
                    MDI, MDI, MDI, MDI, MDI,      # 11 - 15
                    80000.0, MDI, MDI, MDI, MDI,  # 16 - 20
                    MDI, MDI, MDI, MDI, MDI,      # 21 - 25
                    MDI, MDI, MDI, MDI, MDI,      # 26 - 30
                    MDI, MDI, MDI, MDI, MDI,      # 31 - 35
                    MDI, MDI, MDI]                # 36 - 38
        self.assertArrayEqual(ffv.real_constants, expected)

    def test_fields__length(self):
        ffv = self.load()
        self.assertEqual(len(ffv.fields), 5)

    def test_fields__superclass(self):
        ffv = self.load()
        fields = ffv.fields
        for field in fields:
            self.assertIsInstance(field, Field)

    def test_fields__specific_classes(self):
        ffv = self.load()
        fields = ffv.fields
        for i in range(4):
            self.assertIs(type(fields[i]), Field3)
        self.assertIs(type(fields[4]), Field)

    def test_fields__header(self):
        ffv = self.load()
        self.assertEqual(ffv.fields[0].lbfc, 16)

    def test_fields__data(self):
        ffv = self.load()
        data = ffv.fields[0].read_data()
        self.assertEqual(data.shape, (73, 96))
        self.assertArrayEqual(data[30, :3], [292.125, 292.375, 290.875])


if __name__ == '__main__':
    tests.main()
