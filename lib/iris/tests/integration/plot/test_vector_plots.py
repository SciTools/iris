# (C) British Crown Copyright 2014 - 2016, Met Office
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
Test some key usages of :func:`iris.plot.quiver` and
:func:`iris.plot.streamplot`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
import iris.tests.stock

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt
    from iris.plot import quiver, streamplot

    # TEMP-TEST
    plt.switch_backend('tkagg')


@tests.skip_plot
class MixinVectorPlotCases(object):
    """Test examples mixin, used by separate quiver + streamplot classes."""

#    def test_plain(self):
#        """Basic non-latlon, 1d coords testcase."""
#        x = np.array([0., 2, 3, 5])
#        y = np.array([0., 2.5, 4])
#        uv = np.array([[(0., 0), (0, 1), (0, -1), (2, 1)],
#                       [(-1, 0), (-1, -1), (-1, 1), (-2, 1)],
#                       [(1., 0), (1, -1), (1, 1), (-2, 2)]])
#        uv = np.array(uv)
#        u, v = uv[..., 0], uv[..., 1]
#        x_coord = DimCoord(x, long_name='x')
#        y_coord = DimCoord(y, long_name='y')
#        u_cube = Cube(u, long_name='u', units='ms-1')
#        u_cube.add_dim_coord(y_coord, 0)
#        u_cube.add_dim_coord(x_coord, 1)
#        v_cube = u_cube.copy()
#        v_cube.rename('v')
#        v_cube.data = v
#        self.plot('plain', u_cube, v_cube)
#        plt.xlim(x.min() - 1, x.max() + 2)
#        plt.ylim(y.min() - 1, y.max() + 2)
#
#        # TEMP-TEST
#        plt.show()

    def plot(self, plotname, *args, **kwargs):
        plot_function = self.plot_function_to_test()
        plot_function(*args, **kwargs)
        # TEMP-TEST
        plt.suptitle('{}.{}'.format(str(self.__class__), plotname))
#        plt.show()
#        self.test_graphic()

    def test_2d_nonlatlon(self):
        """Basic non-latlon, 1d coords testcase."""
        x = np.array([0., 2, 3, 5])
        y = np.array([0., 2.5, 4])
        x, y = np.meshgrid(x, y)
        uv = np.array([[(0., 0), (0, 1), (0, -1), (2, 1)],
                       [(-1, 0), (-1, -1), (-1, 1), (-2, 1)],
                       [(1., 0), (1, -1), (1, 1), (-2, 2)]])
        uv = np.array(uv)
        u, v = uv[..., 0], uv[..., 1]
        x_coord = AuxCoord(x, long_name='x')
        y_coord = AuxCoord(y, long_name='y')
        u_cube = Cube(u, long_name='u', units='ms-1')
        u_cube.add_aux_coord(y_coord, (0, 1))
        u_cube.add_aux_coord(x_coord, (0, 1))
        v_cube = u_cube.copy()
        v_cube.rename('v')
        v_cube.data = v
        # Call plot : N.B. default gives wrong coords order.
        self.plot('2d_nonlatlon', u_cube, v_cube) # , coords=('x', 'y'))
        plt.xlim(x.min() - 1, x.max() + 2)
        plt.ylim(y.min() - 1, y.max() + 2)

        # TEMP-TEST
        plt.show()
#        self.test_graphic()


class TestQuiver(MixinVectorPlotCases, tests.GraphicsTest):
    def plot_function_to_test(self):
        return iris.plot.quiver


class TestStreamplot(MixinVectorPlotCases, tests.GraphicsTest):
    def plot_function_to_test(self):
        return iris.plot.streamplot


if __name__ == "__main__":
    tests.main()
