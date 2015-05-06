# (C) British Crown Copyright 2014 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)

import six

import numpy as np

from iris.analysis._interpolation import get_xy_dim_coords, snapshot_grid
import iris
import iris.experimental.regrid as eregrid


class AreaWeightedRegridder(object):
    """
    This class provides support for performing area-weighted regridding.

    """

    def __init__(self, src_grid_cube, target_grid_cube, mdtol=1):
        """
        Create an area-weighted regridder for conversions between the source
        and target grids.

        Args:

        * src_grid_cube:
            The :class:`~iris.cube.Cube` providing the source grid.
        * target_grid_cube:
            The :class:`~iris.cube.Cube` providing the target grid.

        Kwargs:

        * mdtol (float):
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds mdtol. mdtol=0 means no missing data is tolerated while
            mdtol=1 will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked.
            Defaults to 1.

        """
        # Snapshot the state of the cubes to ensure that the regridder is
        # impervious to external changes to the original source cubes.
        self._src_grid = snapshot_grid(src_grid_cube)
        self._target_grid = snapshot_grid(target_grid_cube)
        # Missing data tolerance.
        if not (0 <= mdtol <= 1):
            msg = 'Value for mdtol must be in range 0 - 1, got {}.'
            raise ValueError(msg.format(mdtol))
        self._mdtol = mdtol

        # The need for an actual Cube is an implementation quirk caused by the
        # current usage of the experimental regrid function.
        self._target_grid_cube_cache = None

    @property
    def _target_grid_cube(self):
        if self._target_grid_cube_cache is None:
            x, y = self._target_grid
            data = np.empty((y.points.size, x.points.size))
            cube = iris.cube.Cube(data)
            cube.add_dim_coord(y, 0)
            cube.add_dim_coord(x, 1)
            self._target_grid_cube_cache = cube
        return self._target_grid_cube_cache

    def __call__(self, cube):
        """
        Regrid this :class:`~iris.cube.Cube` onto the target grid of
        this :class:`AreaWeightedRegridder`.

        The given cube must be defined with the same grid as the source
        grid used to create this :class:`AreaWeightedRegridder`.

        Args:

        * cube:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            area-weighted regridding.

        """
        if get_xy_dim_coords(cube) != self._src_grid:
            raise ValueError('The given cube is not defined on the same '
                             'source grid as this regridder.')
        return eregrid.regrid_area_weighted_rectilinear_src_and_grid(
            cube, self._target_grid_cube, mdtol=self._mdtol)
