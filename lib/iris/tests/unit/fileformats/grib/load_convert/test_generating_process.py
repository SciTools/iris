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
Tests for function
:func:`iris.fileformats.grib._load_convert.generating_process`.

"""
# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.fileformats.grib._load_convert import generating_process


class TestGeneratingProcess(tests.IrisTest):
    def setUp(self):
        self.warn_patch = self.patch('warnings.warn')

    def test_nowarn(self):
        generating_process(None)
        self.assertEqual(self.warn_patch.call_count, 0)

    def test_warn(self):
        module = 'iris.fileformats.grib._load_convert'
        self.patch(module + '.options.warn_on_unsupported', True)
        generating_process(None)
        got_msgs = [call[0][0] for call in self.warn_patch.call_args_list]
        expected_msgs = ['Unable to translate type of generating process',
                         'Unable to translate background generating process',
                         'Unable to translate forecast generating process']
        for expect_msg in expected_msgs:
            matches = [msg for msg in got_msgs if expect_msg in msg]
            self.assertEqual(len(matches), 1)
            got_msgs.remove(matches[0])


if __name__ == '__main__':
    tests.main()
