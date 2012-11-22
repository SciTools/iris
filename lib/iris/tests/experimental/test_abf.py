# (C) British Crown Copyright 2010 - 2012, Met Office
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

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import iris


@iris.tests.skip_data
class TestAbfLoad(tests.GraphicsTest):
    def setUp(self):
        import iris.experimental.fileformats.abf

    def tearDown(self):
        # Undo the effects of the import so as not to affect subsequent tests.
        iris.fileformats.FORMAT_AGENT._format_specs.pop()
        iris.fileformats.FORMAT_AGENT._format_specs.pop()

    def test_load(self):
        cubes = iris.load(tests.get_data_path(('abf',
                                               'AVHRRBUVI01.1985apra.abf')))
        self.assertCML(cubes, ("abf", "load.cml"))


if __name__ == '__main__':
    tests.main()
