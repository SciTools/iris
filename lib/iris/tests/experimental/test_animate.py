# (C) British Crown Copyright 2013 - 2014, Met Office
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
Test the animation of cubes within iris.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import itertools

import numpy as np

import iris
from iris.coord_systems import GeogCS

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import iris.experimental.animate as animate
    import iris.plot as iplt


@tests.skip_plot
class IntegrationTest(tests.GraphicsTest):
    def setUp(self):
        cube = iris.cube.Cube(np.arange(36, dtype=np.int32).reshape((3, 3, 4)))
        cs = GeogCS(6371229)

        coord = iris.coords.DimCoord(
            points=np.array([1, 2, 3], dtype=np.int32), long_name='time')
        cube.add_dim_coord(coord, 0)

        coord = iris.coords.DimCoord(
            points=np.array([-1, 0, 1], dtype=np.int32),
            standard_name='latitude',
            units='degrees',
            coord_system=cs)
        cube.add_dim_coord(coord, 1)
        coord = iris.coords.DimCoord(
            points=np.array([-1, 0, 1, 2], dtype=np.int32),
            standard_name='longitude',
            units='degrees',
            coord_system=cs)
        cube.add_dim_coord(coord, 2)
        self.cube = cube

    def test_cube_animation(self):
        # This follows :meth:`~matplotlib.animation.FuncAnimation.save`
        # to ensure that each frame corresponds to known accepted frames for
        # the animation.
        cube_iter = self.cube.slices(('latitude', 'longitude'))

        ani = animate.animate(cube_iter, iplt.contourf)

        # Disconnect the first draw callback to stop the animation
        ani._fig.canvas.mpl_disconnect(ani._first_draw_id)

        ani = [ani]
        # Extract frame data
        for data in itertools.izip(*[a.new_saved_frame_seq() for a in ani]):
            # Draw each frame
            for anim, d in zip(ani, data):
                anim._draw_next_frame(d, blit=False)
                self.check_graphic()


if __name__ == "__main__":
    tests.main()
