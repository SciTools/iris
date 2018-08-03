# (C) British Crown Copyright 2018, Met Office
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
Unit tests for the function
:func:`iris.analysis.cartography.gridcell_angles`.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.cube import Cube
from iris.coords import AuxCoord

from iris.analysis.cartography import gridcell_angles


class TestGridcellAngles(tests.IrisTest):
    def _check_multiple_orientations_and_latitudes(
            self,
            method='mid-lhs, mid-rhs',
            atol_degrees=0.005,
            cellsize_degrees=1.0):
        ny, nx = 7, 9
        x0, x1 = -164, 164
        y0, y1 = -75, 75
        lats = np.linspace(y0, y1, ny, endpoint=True)
        angles = np.linspace(x0, x1, nx, endpoint=True)
        x_pts_2d, y_pts_2d = np.meshgrid(angles, lats)

        # Make gridcells rectangles surrounding these centrepoints, but also
        # tilted at various angles (= same as x-point lons, as that's easy).
#        dx = cellsize_degrees  # half-width of gridcells, in degrees
#        dy = dx   # half-height of gridcells, in degrees

        # Calculate centrepoint lons+lats : in radians, and shape (ny, nx, 1).
        xangs, yangs = np.deg2rad(x_pts_2d), np.deg2rad(y_pts_2d)
        xangs, yangs = [arr[..., None] for arr in (xangs, yangs)]
        # Program which corners are up+down on each gridcell axis.
        dx_corners = [[[-1, 1, 1, -1]]]
        dy_corners = [[[-1, -1, 1, 1]]]
        # Calculate the relative offsets in x+y at the 4 corners.
        x_ofs_2d = cellsize_degrees * np.cos(xangs) * dx_corners
        x_ofs_2d -= cellsize_degrees * np.sin(xangs) * dy_corners
        y_ofs_2d = cellsize_degrees * np.cos(xangs) * dy_corners
        y_ofs_2d += cellsize_degrees * np.sin(xangs) * dx_corners
        # Apply a latitude stretch to make correct angles on the globe.
        y_ofs_2d *= np.cos(yangs)
        # Make bounds arrays by adding the corner offsets to the centrepoints.
        x_bds_2d = x_pts_2d[..., None] + x_ofs_2d
        y_bds_2d = y_pts_2d[..., None] + y_ofs_2d

        # Create a cube with these points + bounds in its 'X' and 'Y' coords.
        co_x = AuxCoord(points=x_pts_2d, bounds=x_bds_2d,
                        standard_name='longitude', units='degrees')
        co_y = AuxCoord(points=y_pts_2d, bounds=y_bds_2d,
                        standard_name='latitude', units='degrees')
        cube = Cube(np.zeros((ny, nx)))
        cube.add_aux_coord(co_x, (0, 1))
        cube.add_aux_coord(co_y, (0, 1))

        # Calculate gridcell angles at each point.
        angles_cube = gridcell_angles(cube, cell_angle_boundpoints=method)

        # Check that the results are a close match to the original intended
        # gridcell orientation angles.
        # NOTE: neither the above gridcell construction nor the calculation
        # itself are exact :  Errors scale as the square of gridcell sizes.
        angles_cube.convert_units('degrees')
        angles_calculated = angles_cube.data

        # Note: expand the original 1-d test angles into the full result shape,
        # just to please 'np.testing.assert_allclose', which doesn't broadcast.
        angles_expected = np.zeros(angles_cube.shape)
        angles_expected[:] = angles

        # Assert (toleranced) equality, and return results.
        self.assertArrayAllClose(angles_calculated, angles_expected,
                                 atol=atol_degrees)

        return angles_calculated, angles_expected

    def test_various_orientations_and_locations(self):
        self._check_multiple_orientations_and_latitudes()

    def test_bottom_edge_method(self):
        # Get results with the "other" calculation method + check to tolerance.
        # A smallish cellsize should yield similar results in both cases.
        r1, _ = self._check_multiple_orientations_and_latitudes()
        r2, _ = self._check_multiple_orientations_and_latitudes(
            method='lower-left, lower-right',
            cellsize_degrees=0.1, atol_degrees=0.1)

        # They are not the same : checks we selected the 'other' method !
        self.assertFalse(np.allclose(r1, r2))
        # Note: results are rather different at higher latitudes.
        atol = 0.1
        self.assertArrayAllClose(r1, r2, atol=atol)


if __name__ == "__main__":
    tests.main()
