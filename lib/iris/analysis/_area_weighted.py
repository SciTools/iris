# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from iris.analysis._interpolation import get_xy_dim_coords, snapshot_grid


class AreaWeightedRegridder:
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

        .. Note::

            Both source and target cubes must have an XY grid defined by
            separate X and Y dimensions with dimension coordinates.
            All of the XY dimension coordinates must also be bounded, and have
            the same coordinate system.

        """
        from iris.experimental.regrid import (
            _regrid_area_weighted_rectilinear_src_and_grid__prepare,
        )

        # Snapshot the state of the source cube to ensure that the regridder is
        # impervious to external changes to the original cubes.
        self._src_grid = snapshot_grid(src_grid_cube)

        # Missing data tolerance.
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self._mdtol = mdtol

        # Store regridding information
        _regrid_info = _regrid_area_weighted_rectilinear_src_and_grid__prepare(
            src_grid_cube, target_grid_cube
        )
        (
            src_x,
            src_y,
            src_x_dim,
            src_y_dim,
            self.grid_x,
            self.grid_y,
            self.meshgrid_x,
            self.meshgrid_y,
            self.weights_info,
        ) = _regrid_info

    def __call__(self, cube):
        """
        Regrid this :class:`~iris.cube.Cube` onto the target grid of
        this :class:`AreaWeightedRegridder`.

        The given cube must be defined with the same grid as the source
        grid used to create this :class:`AreaWeightedRegridder`.

        If the source cube has lazy data, the returned cube will also
        have lazy data.

        Args:

        * cube:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            area-weighted regridding.

        .. note::

            If the source cube has lazy data,
            `chunks <https://docs.dask.org/en/latest/array-chunks.html>`__
            in the horizontal dimensions will be combined before regridding.

        """
        from iris.experimental.regrid import (
            _regrid_area_weighted_rectilinear_src_and_grid__perform,
        )

        src_x, src_y = get_xy_dim_coords(cube)
        if (src_x, src_y) != self._src_grid:
            raise ValueError(
                "The given cube is not defined on the same "
                "source grid as this regridder."
            )
        src_x_dim = cube.coord_dims(src_x)[0]
        src_y_dim = cube.coord_dims(src_y)[0]
        _regrid_info = (
            src_x,
            src_y,
            src_x_dim,
            src_y_dim,
            self.grid_x,
            self.grid_y,
            self.meshgrid_x,
            self.meshgrid_y,
            self.weights_info,
        )
        return _regrid_area_weighted_rectilinear_src_and_grid__perform(
            cube, _regrid_info, mdtol=self._mdtol
        )
