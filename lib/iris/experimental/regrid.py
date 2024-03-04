# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Regridding functions.

.. note::

    .. deprecated:: 3.2.0

    This package will be removed in a future release.
    The PointInCell class has now moved to :class:`iris.analysis.PointInCell`.
    All the other content will be withdrawn.

    If you still use any of this, please contact the Iris Developers to
    discuss how to replace it or to retain it.

"""
import copy
import functools
import warnings

import cartopy.crs as ccrs
import numpy as np
import scipy.interpolate

from iris._deprecation import warn_deprecated
from iris.analysis._area_weighted import (
    _regrid_area_weighted_rectilinear_src_and_grid__perform,
    _regrid_area_weighted_rectilinear_src_and_grid__prepare,
)
from iris.analysis._interpolation import (
    get_xy_coords,
    get_xy_dim_coords,
    snapshot_grid,
)
from iris.analysis._regrid import (
    _regrid_weighted_curvilinear_to_rectilinear__perform,
    _regrid_weighted_curvilinear_to_rectilinear__prepare,
)
import iris.analysis.cartography
import iris.coord_systems
import iris.cube
from iris.util import _meshgrid

wmsg = (
    "The 'iris.experimental.regrid' package is deprecated since version 3.2, "
    "and will be removed in a future release.  The PointInCell class has now "
    "moved into iris.analysis.  All its other content will be withdrawn.  "
    "If you still use any of this, please contact the Iris Developers to "
    "discuss how to replace it or to retain it (reverse the deprecation)."
)
warn_deprecated(wmsg)


def regrid_area_weighted_rectilinear_src_and_grid(
    src_cube, grid_cube, mdtol=0
):
    """
    Return a new cube with data values calculated using the area weighted
    mean of data values from src_grid regridded onto the horizontal grid of
    grid_cube.

    .. note::

        .. deprecated:: 3.2.0

        This function is scheduled to be removed in a future release.
        Please use :meth:`iris.cube.Cube.regrid` with the
        :class:`iris.analysis.AreaWeighted` scheme instead : this is an exact
        replacement.

        For example :

        .. code::

            result = src_cube.regrid(grid_cube, AreaWeighted())

    This function requires that the horizontal grids of both cubes are
    rectilinear (i.e. expressed in terms of two orthogonal 1D coordinates)
    and that these grids are in the same coordinate system. This function
    also requires that the coordinates describing the horizontal grids
    all have bounds.

    .. note::

        Elements in data array of the returned cube that lie either partially
        or entirely outside of the horizontal extent of the src_cube will
        be masked irrespective of the value of mdtol.

    Args:

    * src_cube:
        An instance of :class:`iris.cube.Cube` that supplies the data,
        metadata and coordinates.
    * grid_cube:
        An instance of :class:`iris.cube.Cube` that supplies the desired
        horizontal grid definition.

    Kwargs:

    * mdtol:
        Tolerance of missing data. The value returned in each element of the
        returned cube's data array will be masked if the fraction of masked
        data in the overlapping cells of the source cube exceeds mdtol. This
        fraction is calculated based on the area of masked cells within each
        target cell. mdtol=0 means no missing data is tolerated while mdtol=1
        will mean the resulting element will be masked if and only if all the
        overlapping cells of the source cube are masked. Defaults to 0.

    Returns:
        A new :class:`iris.cube.Cube` instance.

    """
    wmsg = (
        "The function "
        "'iris.experimental.regrid."
        "regrid_area_weighted_rectilinear_src_and_grid' "
        "has been deprecated, and will be removed in a future release.  "
        "Please consult the docstring for details."
    )
    warn_deprecated(wmsg)

    regrid_info = _regrid_area_weighted_rectilinear_src_and_grid__prepare(
        src_cube, grid_cube
    )
    result = _regrid_area_weighted_rectilinear_src_and_grid__perform(
        src_cube, regrid_info, mdtol
    )
    return result


def regrid_weighted_curvilinear_to_rectilinear(src_cube, weights, grid_cube):
    r"""
    Return a new cube with the data values calculated using the weighted
    mean of data values from :data:`src_cube` and the weights from
    :data:`weights` regridded onto the horizontal grid of :data:`grid_cube`.

    .. note ::

        .. deprecated:: 3.2.0

        This function is scheduled to be removed in a future release.
        Please use :meth:`iris.cube.Cube.regrid` with the
        :class:`iris.analysis.PointInCell` scheme instead : this is an exact
        replacement.

        For example :

        .. code::

            result = src_cube.regrid(grid_cube, PointInCell())

    This function requires that the :data:`src_cube` has a horizontal grid
    defined by a pair of X- and Y-axis coordinates which are mapped over the
    same cube dimensions, thus each point has an individually defined X and
    Y coordinate value.  The actual dimensions of these coordinates are of
    no significance.
    The :data:`src_cube` grid cube must have a normal horizontal grid,
    i.e. expressed in terms of two orthogonal 1D horizontal coordinates.
    Both grids must be in the same coordinate system, and the :data:`grid_cube`
    must have horizontal coordinates that are both bounded and contiguous.

    Note that, for any given target :data:`grid_cube` cell, only the points
    from the :data:`src_cube` that are bound by that cell will contribute to
    the cell result. The bounded extent of the :data:`src_cube` will not be
    considered here.

    A target :data:`grid_cube` cell result will be calculated as,
    :math:`\sum (src\_cube.data_{ij} * weights_{ij}) / \sum weights_{ij}`, for
    all :math:`ij` :data:`src_cube` points that are bound by that cell.

    .. warning::

        * All coordinates that span the :data:`src_cube` that don't define
          the horizontal curvilinear grid will be ignored.

    Args:

    * src_cube:
        A :class:`iris.cube.Cube` instance that defines the source
        variable grid to be regridded.
    * weights (array or None):
        A :class:`numpy.ndarray` instance that defines the weights
        for the source variable grid cells. Must have the same shape
        as the X and Y coordinates.  If weights is None, all-ones will be used.
    * grid_cube:
        A :class:`iris.cube.Cube` instance that defines the target
        rectilinear grid.

    Returns:
        A :class:`iris.cube.Cube` instance.

    """
    wmsg = (
        "The function "
        "'iris.experimental.regrid."
        "regrid_weighted_curvilinear_to_rectilinear' "
        "has been deprecated, and will be removed in a future release.  "
        "Please consult the docstring for details."
    )
    warn_deprecated(wmsg)
    regrid_info = _regrid_weighted_curvilinear_to_rectilinear__prepare(
        src_cube, weights, grid_cube
    )
    result = _regrid_weighted_curvilinear_to_rectilinear__perform(
        src_cube, regrid_info
    )
    return result


class PointInCell:
    """
    This class describes the point-in-cell regridding scheme for use
    typically with :meth:`iris.cube.Cube.regrid()`.

    .. warning::

        This class is now **disabled**.

        The functionality has been moved to
        :class:`iris.analysis.PointInCell`.

    """

    def __init__(self, weights=None):
        """
        Point-in-cell regridding scheme suitable for regridding over one
        or more orthogonal coordinates.

        .. warning::

            This class is now **disabled**.

            The functionality has been moved to
            :class:`iris.analysis.PointInCell`.

        """
        raise Exception(
            'The class "iris.experimental.PointInCell" has been '
            "moved, and is now in iris.analysis"
            "\nPlease replace "
            '"iris.experimental.PointInCell" with '
            '"iris.analysis.PointInCell".'
        )


class _ProjectedUnstructuredRegridder:
    """
    This class provides regridding that uses scipy.interpolate.griddata.

    """

    def __init__(self, src_cube, tgt_grid_cube, method, projection=None):
        """
        Create a regridder for conversions between the source
        and target grids.

        Args:

        * src_cube:
            The :class:`~iris.cube.Cube` providing the source points.
        * tgt_grid_cube:
            The :class:`~iris.cube.Cube` providing the target grid.
        * method:
            Either 'linear' or 'nearest'.
        * projection:
            The projection in which the interpolation is performed. If None, a
            PlateCarree projection is used. Defaults to None.

        """
        # Validity checks.
        if not isinstance(src_cube, iris.cube.Cube):
            raise TypeError("'src_cube' must be a Cube")
        if not isinstance(tgt_grid_cube, iris.cube.Cube):
            raise TypeError("'tgt_grid_cube' must be a Cube")

        # Snapshot the state of the target cube to ensure that the regridder
        # is impervious to external changes to the original source cubes.
        self._tgt_grid = snapshot_grid(tgt_grid_cube)

        # Check the target grid units.
        for coord in self._tgt_grid:
            self._check_units(coord)

        # Whether to use linear or nearest-neighbour interpolation.
        if method not in ("linear", "nearest"):
            msg = "Regridding method {!r} not supported.".format(method)
            raise ValueError(msg)
        self._method = method

        src_x_coord, src_y_coord = get_xy_coords(src_cube)
        if src_x_coord.coord_system != src_y_coord.coord_system:
            raise ValueError(
                "'src_cube' lateral geographic coordinates have "
                "differing coordinate systems."
            )
        if src_x_coord.coord_system is None:
            raise ValueError(
                "'src_cube' lateral geographic coordinates have "
                "no coordinate system."
            )
        tgt_x_coord, tgt_y_coord = get_xy_dim_coords(tgt_grid_cube)
        if tgt_x_coord.coord_system != tgt_y_coord.coord_system:
            raise ValueError(
                "'tgt_grid_cube' lateral geographic coordinates "
                "have differing coordinate systems."
            )
        if tgt_x_coord.coord_system is None:
            raise ValueError(
                "'tgt_grid_cube' lateral geographic coordinates "
                "have no coordinate system."
            )

        if projection is None:
            globe = src_x_coord.coord_system.as_cartopy_globe()
            projection = ccrs.Sinusoidal(globe=globe)
        self._projection = projection

    def _check_units(self, coord):
        if coord.coord_system is None:
            # No restriction on units.
            pass
        elif isinstance(
            coord.coord_system,
            (iris.coord_systems.GeogCS, iris.coord_systems.RotatedGeogCS),
        ):
            # Units for lat-lon or rotated pole must be 'degrees'. Note
            # that 'degrees_east' etc. are equal to 'degrees'.
            if coord.units != "degrees":
                msg = (
                    "Unsupported units for coordinate system. "
                    "Expected 'degrees' got {!r}.".format(coord.units)
                )
                raise ValueError(msg)
        else:
            # Units for other coord systems must be equal to metres.
            if coord.units != "m":
                msg = (
                    "Unsupported units for coordinate system. "
                    "Expected 'metres' got {!r}.".format(coord.units)
                )
                raise ValueError(msg)

    @staticmethod
    def _regrid(
        src_data,
        xy_dim,
        src_x_coord,
        src_y_coord,
        tgt_x_coord,
        tgt_y_coord,
        projection,
        method,
    ):
        """
        Regrids input data from the source to the target. Calculation is.

        """
        # Transform coordinates into the projection the interpolation will be
        # performed in.
        src_projection = src_x_coord.coord_system.as_cartopy_projection()
        projected_src_points = projection.transform_points(
            src_projection, src_x_coord.points, src_y_coord.points
        )

        tgt_projection = tgt_x_coord.coord_system.as_cartopy_projection()
        tgt_x, tgt_y = _meshgrid(tgt_x_coord.points, tgt_y_coord.points)
        projected_tgt_grid = projection.transform_points(
            tgt_projection, tgt_x, tgt_y
        )

        # Prepare the result data array.
        # XXX TODO: Deal with masked src_data
        (tgt_y_shape,) = tgt_y_coord.shape
        (tgt_x_shape,) = tgt_x_coord.shape
        tgt_shape = (
            src_data.shape[:xy_dim]
            + (tgt_y_shape,)
            + (tgt_x_shape,)
            + src_data.shape[xy_dim + 1 :]
        )
        data = np.empty(tgt_shape, dtype=src_data.dtype)

        iter_shape = list(src_data.shape)
        iter_shape[xy_dim] = 1

        for index in np.ndindex(tuple(iter_shape)):
            src_index = list(index)
            src_index[xy_dim] = slice(None)
            src_subset = src_data[tuple(src_index)]
            tgt_index = (
                index[:xy_dim]
                + (slice(None), slice(None))
                + index[xy_dim + 1 :]
            )
            data[tgt_index] = scipy.interpolate.griddata(
                projected_src_points[..., :2],
                src_subset,
                (projected_tgt_grid[..., 0], projected_tgt_grid[..., 1]),
                method=method,
            )
        data = np.ma.array(data, mask=np.isnan(data))
        return data

    def _create_cube(
        self,
        data,
        src,
        src_xy_dim,
        src_x_coord,
        src_y_coord,
        grid_x_coord,
        grid_y_coord,
        regrid_callback,
    ):
        """
        Return a new Cube for the result of regridding the source Cube onto
        the new grid.

        All the metadata and coordinates of the result Cube are copied from
        the source Cube, with two exceptions:
            - Grid dimension coordinates are copied from the grid Cube.
            - Auxiliary coordinates which span the grid dimensions are
              ignored, except where they provide a reference surface for an
              :class:`iris.aux_factory.AuxCoordFactory`.

        Args:

        * data:
            The regridded data as an N-dimensional NumPy array.
        * src:
            The source Cube.
        * src_xy_dim:
            The dimension the X and Y coord span within the source Cube.
        * src_x_coord:
            The X coordinate (either :class:`iris.coords.AuxCoord` or
            :class:`iris.coords.DimCoord`).
        * src_y_coord:
            The Y coordinate (either :class:`iris.coords.AuxCoord` or
            :class:`iris.coords.DimCoord`).
        * grid_x_coord:
            The :class:`iris.coords.DimCoord` for the new grid's X
            coordinate.
        * grid_y_coord:
            The :class:`iris.coords.DimCoord` for the new grid's Y
            coordinate.
        * regrid_callback:
            The routine that will be used to calculate the interpolated
            values of any reference surfaces.

        Returns:
            The new, regridded Cube.

        """
        # Create a result cube with the appropriate metadata
        result = iris.cube.Cube(data)
        result.metadata = copy.deepcopy(src.metadata)

        # Copy across all the coordinates which don't span the grid.
        # Record a mapping from old coordinate IDs to new coordinates,
        # for subsequent use in creating updated aux_factories.
        coord_mapping = {}

        def copy_coords(src_coords, add_method):
            for coord in src_coords:
                dims = src.coord_dims(coord)
                if coord is src_x_coord:
                    coord = grid_x_coord
                    # Increase dimensionality to account for 1D coord being
                    # regridded onto 2D grid
                    dims = list(dims)
                    dims[0] += 1
                    dims = tuple(dims)
                    add_method = result.add_dim_coord
                elif coord is src_y_coord:
                    coord = grid_y_coord
                    add_method = result.add_dim_coord
                elif src_xy_dim in dims:
                    continue
                result_coord = coord.copy()
                add_method(result_coord, dims)
                coord_mapping[id(coord)] = result_coord

        copy_coords(src.dim_coords, result.add_dim_coord)
        copy_coords(src.aux_coords, result.add_aux_coord)

        def regrid_reference_surface(
            src_surface_coord,
            surface_dims,
            src_xy_dim,
            src_x_coord,
            src_y_coord,
            grid_x_coord,
            grid_y_coord,
            regrid_callback,
        ):
            # Determine which of the reference surface's dimensions span the X
            # and Y dimensions of the source cube.
            surface_xy_dim = surface_dims.index(src_xy_dim)
            surface = regrid_callback(
                src_surface_coord.points,
                surface_xy_dim,
                src_x_coord,
                src_y_coord,
                grid_x_coord,
                grid_y_coord,
            )
            surface_coord = src_surface_coord.copy(surface)
            return surface_coord

        # Copy across any AuxFactory instances, and regrid their reference
        # surfaces where required.
        for factory in src.aux_factories:
            for coord in factory.dependencies.values():
                if coord is None:
                    continue
                dims = src.coord_dims(coord)
                if src_xy_dim in dims:
                    result_coord = regrid_reference_surface(
                        coord,
                        dims,
                        src_xy_dim,
                        src_x_coord,
                        src_y_coord,
                        grid_x_coord,
                        grid_y_coord,
                        regrid_callback,
                    )
                    result.add_aux_coord(result_coord, (dims[0], dims[0] + 1))
                    coord_mapping[id(coord)] = result_coord
            try:
                result.add_aux_factory(factory.updated(coord_mapping))
            except KeyError:
                msg = (
                    "Cannot update aux_factory {!r} because of dropped"
                    " coordinates.".format(factory.name())
                )
                warnings.warn(msg)
        return result

    def __call__(self, src_cube):
        """
        Regrid this :class:`~iris.cube.Cube` on to the target grid of
        this :class:`UnstructuredProjectedRegridder`.

        The given cube must be defined with the same grid as the source
        grid used to create this :class:`UnstructuredProjectedRegridder`.

        Args:

        * src_cube:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            either nearest-neighbour or linear interpolation.

        """
        # Validity checks.
        if not isinstance(src_cube, iris.cube.Cube):
            raise TypeError("'src' must be a Cube")

        src_x_coord, src_y_coord = get_xy_coords(src_cube)
        tgt_x_coord, tgt_y_coord = self._tgt_grid
        src_cs = src_x_coord.coord_system

        if src_x_coord.coord_system != src_y_coord.coord_system:
            raise ValueError(
                "'src' lateral geographic coordinates have "
                "differing coordinate systems."
            )
        if src_cs is None:
            raise ValueError(
                "'src' lateral geographic coordinates have "
                "no coordinate system."
            )

        # Check the source grid units.
        for coord in (src_x_coord, src_y_coord):
            self._check_units(coord)

        (src_x_dim,) = src_cube.coord_dims(src_x_coord)
        (src_y_dim,) = src_cube.coord_dims(src_y_coord)

        if src_x_dim != src_y_dim:
            raise ValueError(
                "'src' lateral geographic coordinates should map "
                "the same dimension."
            )
        src_xy_dim = src_x_dim

        # Compute the interpolated data values.
        data = self._regrid(
            src_cube.data,
            src_xy_dim,
            src_x_coord,
            src_y_coord,
            tgt_x_coord,
            tgt_y_coord,
            self._projection,
            method=self._method,
        )

        # Wrap up the data as a Cube.
        regrid_callback = functools.partial(
            self._regrid, method=self._method, projection=self._projection
        )

        new_cube = self._create_cube(
            data,
            src_cube,
            src_xy_dim,
            src_x_coord,
            src_y_coord,
            tgt_x_coord,
            tgt_y_coord,
            regrid_callback,
        )

        return new_cube


class ProjectedUnstructuredLinear:
    """
    This class describes the linear regridding scheme which uses the
    scipy.interpolate.griddata to regrid unstructured data on to a grid.

    The source cube and the target cube will be projected into a common
    projection for the scipy calculation to be performed.

    """

    def __init__(self, projection=None):
        """
        Linear regridding scheme that uses scipy.interpolate.griddata on
        projected unstructured data.

        .. note::

            .. deprecated:: 3.2.0

            This class is scheduled to be removed in a future release, and no
            replacement is currently planned.
            If you make use of this functionality, please contact the Iris
            Developers to discuss how to retain it (which could include
            reversing the deprecation).

        Optional Args:

        * projection: `cartopy.crs instance`
            The projection that the scipy calculation is performed in.
            If None is given, a PlateCarree projection is used. Defaults to
            None.

        """
        self.projection = projection
        wmsg = (
            "The class iris.experimental.regrid.ProjectedUnstructuredLinear "
            "has been deprecated, and will be removed in a future release.  "
            "Please consult the docstring for details."
        )
        warn_deprecated(wmsg)

    def regridder(self, src_cube, target_grid):
        """
        Creates a linear regridder to perform regridding, using
        scipy.interpolate.griddata from unstructured source points to the
        target grid. The regridding calculation is performed in the given
        projection.

        Typically you should use :meth:`iris.cube.Cube.regrid` for
        regridding a cube. There are, however, some situations when
        constructing your own regridder is preferable. These are detailed in
        the :ref:`user guide <caching_a_regridder>`.

        Does not support lazy regridding.

        Args:

        * src_cube:
            The :class:`~iris.cube.Cube` defining the unstructured source
            points.
        * target_grid:
            The :class:`~iris.cube.Cube` defining the target grid.

        Returns:
            A callable with the interface:

                `callable(cube)`

            where `cube` is a cube with the same grid as `src_cube`
            that is to be regridded to the `target_grid`.

        """
        return _ProjectedUnstructuredRegridder(
            src_cube, target_grid, "linear", self.projection
        )


class ProjectedUnstructuredNearest:
    """
    This class describes the nearest regridding scheme which uses the
    scipy.interpolate.griddata to regrid unstructured data on to a grid.

    The source cube and the target cube will be projected into a common
    projection for the scipy calculation to be performed.

    .. Note::
          The :class:`iris.analysis.UnstructuredNearest` scheme performs
          essentially the same job.  That calculation is more rigorously
          correct and may be applied to larger data regions (including global).
          This one however, where applicable, is substantially faster.

    """

    def __init__(self, projection=None):
        """
        Nearest regridding scheme that uses scipy.interpolate.griddata on
        projected unstructured data.

        .. note::

            .. deprecated:: 3.2.0

            This class is scheduled to be removed in a future release, and no
            exact replacement is currently planned.
            Please use :class:`iris.analysis.UnstructuredNearest` instead, if
            possible.  If you have a need for this exact functionality, please
            contact the Iris Developers to discuss how to retain it (which
            could include reversing the deprecation).

        Optional Args:

        * projection: `cartopy.crs instance`
            The projection that the scipy calculation is performed in.
            If None is given, a PlateCarree projection is used. Defaults to
            None.

        """
        self.projection = projection
        wmsg = (
            "iris.experimental.regrid.ProjectedUnstructuredNearest has been "
            "deprecated, and will be removed in a future release.  "
            "Please use 'iris.analysis.UnstructuredNearest' instead, where "
            "possible.  Consult the docstring for details."
        )
        warn_deprecated(wmsg)

    def regridder(self, src_cube, target_grid):
        """
        Creates a nearest-neighbour regridder to perform regridding, using
        scipy.interpolate.griddata from unstructured source points to the
        target grid. The regridding calculation is performed in the given
        projection.

        Typically you should use :meth:`iris.cube.Cube.regrid` for
        regridding a cube. There are, however, some situations when
        constructing your own regridder is preferable. These are detailed in
        the :ref:`user guide <caching_a_regridder>`.

        Does not support lazy regridding.

        Args:

        * src_cube:
            The :class:`~iris.cube.Cube` defining the unstructured source
            points.
        * target_grid:
            The :class:`~iris.cube.Cube` defining the target grid.

        Returns:
            A callable with the interface:

                `callable(cube)`

            where `cube` is a cube with the same grid as `src_cube`
            that is to be regridded to the `target_grid`.

        """
        return _ProjectedUnstructuredRegridder(
            src_cube, target_grid, "nearest", self.projection
        )
