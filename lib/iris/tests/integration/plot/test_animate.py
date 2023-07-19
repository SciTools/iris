# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Integration tests for :func:`iris.plot.animate`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

import iris
from iris.coord_systems import GeogCS

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class IntegrationTest(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
        cube = iris.cube.Cube(np.arange(36, dtype=np.int32).reshape((3, 3, 4)))
        cs = GeogCS(6371229)

        coord = iris.coords.DimCoord(
            points=np.array([1, 2, 3], dtype=np.int32), long_name="time"
        )
        cube.add_dim_coord(coord, 0)

        coord = iris.coords.DimCoord(
            points=np.array([-1, 0, 1], dtype=np.int32),
            standard_name="latitude",
            units="degrees",
            coord_system=cs,
        )
        cube.add_dim_coord(coord, 1)
        coord = iris.coords.DimCoord(
            points=np.array([-1, 0, 1, 2], dtype=np.int32),
            standard_name="longitude",
            units="degrees",
            coord_system=cs,
        )
        cube.add_dim_coord(coord, 2)
        self.cube = cube

    def test_cube_animation(self):
        # This follows :meth:`~matplotlib.animation.FuncAnimation.save`
        # to ensure that each frame corresponds to known accepted frames for
        # the animation.
        cube_iter = self.cube.slices(("latitude", "longitude"))

        ani = iplt.animate(cube_iter, iplt.contourf)

        # Disconnect the first draw callback to stop the animation.
        ani._fig.canvas.mpl_disconnect(ani._first_draw_id)
        # Update flag to indicate drawing happens.  Without this, a warning is
        # thrown when the ani object is destroyed, and this warning sometimes
        # interferes with unrelated tests (#4330).
        ani._draw_was_started = True

        ani = [ani]
        # Extract frame data
        for data in zip(*[a.new_saved_frame_seq() for a in ani]):
            # Draw each frame
            for anim, d in zip(ani, data):
                anim._draw_next_frame(d, blit=False)
                self.check_graphic()


if __name__ == "__main__":
    tests.main()
