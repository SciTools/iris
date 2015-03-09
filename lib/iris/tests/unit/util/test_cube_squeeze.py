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
"""Test function :func:`iris.util.cube_squeeze`."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest

import iris
import iris.tests.stock
from iris.util import new_axis

class Test(iris.tests.IrisTest):

    def setUp(self):
        self.cube = iris.tests.stock.simple_2d_w_multidim_and_scalars()

    def test_no_change(self):
        self.assertEqual(self.cube, iris.util.cube_squeeze(self.cube))

    def test_squeeze_one_dim(self):
        cube_3d = iris.util.new_axis(self.cube, scalar_coord='an_other')
        cube_2d = iris.util.cube_squeeze(cube_3d)

        self.assertEqual(self.cube, cube_2d)

    def test_squeeze_two_dims(self):
        cube_3d = iris.util.new_axis(self.cube, scalar_coord='an_other')
        cube_4d =  iris.util.new_axis(cube_3d, scalar_coord='air_temperature')

        self.assertEqual(self.cube, iris.util.cube_squeeze(cube_4d))

    def test_squeeze_one_anonymous_dim(self):
        cube_3d = iris.util.new_axis(self.cube)
        cube_2d = iris.util.cube_squeeze(cube_3d)

        self.assertEqual(self.cube, cube_2d)

    def test_squeeze_to_scalar_cube(self):
        cube_scalar = self.cube[0, 0]
        cube_1d = iris.util.new_axis(cube_scalar)

        self.assertEqual(cube_scalar, iris.util.cube_squeeze(cube_1d))


if __name__ == '__main__':
    unittest.main()
