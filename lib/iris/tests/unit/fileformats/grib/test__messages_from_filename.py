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
"""
Unit tests for `iris.fileformats.grib._messages_from_filename`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.fileformats.grib import _messages_from_filename


@tests.skip_data
class Test(tests.IrisTest):
    def test(self):
        filename = tests.get_data_path(('GRIB', '3_layer_viz',
                                        '3_layer.grib2'))
        messages = list(_messages_from_filename(filename))
        self.assertEqual(len(messages), 3)


if __name__ == '__main__':
    tests.main()
