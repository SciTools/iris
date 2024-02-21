# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculus operations on :class:`iris.cube.Cube` instances.

See also: :mod:`NumPy <numpy>`.

"""

import re
import warnings

import cf_units
import numpy as np

import iris.analysis
from iris.analysis.cartography import (
    DEFAULT_SPHERICAL_EARTH_RADIUS,
    DEFAULT_SPHERICAL_EARTH_RADIUS_UNIT,
)
import iris.analysis.maths
import iris.coord_systems
import iris.coords
from iris.util import delta
from iris.warnings import IrisUserWarning

__all__ = ["DIRECTIONAL_NAMES", "cube_delta", "curl", "differentiate"]


def _construct_delta_coord(coord):
    """Return a coordinate of deltas between the given coordinate's points.

    Return a coordinate of deltas between the given coordinate's points.
    If the original coordinate has length n and is circular then the
    result will be a coordinate of length n, otherwise the result will be
    of length n-1.

    """
    if coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError(coord)
    circular = getattr(coord, "circular", False)
    if coord.shape == (1,) and not circular:
        raise ValueError(
            "Cannot take interval differences of a single valued coordinate."
        )

    if circular:
        circular_kwd = coord.units.modulus or True
    else:
        circular_kwd = False

    if coord.bounds is not None:
        bounds = iris.util.delta(coord.bounds, 0, circular=circular_kwd)
    else:
        bounds = None

    points = iris.util.delta(coord.points, 0, circular=circular_kwd)
    new_coord = iris.coords.AuxCoord.from_coord(coord).copy(points, bounds)
    new_coord.rename("change_in_%s" % new_coord.name())

    return new_coord


def _construct_midpoint_coord(coord, circular=None):
    """Return a coordinate of mid-points from the given coordinate.

    Return a coordinate of mid-points from the given coordinate. If the
    given coordinate has length n and the circular flag set then the
    result will be a coordinate of length n, otherwise the result will be
    of length n-1.

    """
    if circular and not hasattr(coord, "circular"):
        msg = (
            "Cannot produce a circular midpoint for the '{}' coord, "
            "which does not have a 'circular' attribute."
        )
        raise ValueError(msg.format(coord.name()))

    if circular is None:
        circular = getattr(coord, "circular", False)
    elif circular != getattr(coord, "circular", False):
        msg = (
            "Construction coordinate midpoints for the '{}' coordinate, "
            "though it has the attribute 'circular'={}."
        )
        warnings.warn(
            msg.format(circular, coord.circular, coord.name()),
            category=IrisUserWarning,
        )

    if coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError(coord)
    if coord.shape == (1,) and not circular:
        raise ValueError("Cannot take the midpoints of a single valued coordinate.")

    # Calculate the delta of the coordinate
    # (this deals with circularity nicely).
    mid_point_coord = _construct_delta_coord(coord)

    # if the coord is circular then include the last one, else, just take 0:-1
    circular_slice = slice(0, -1 if not circular else None)

    if coord.bounds is not None:
        axis_delta = mid_point_coord.bounds
        mid_point_bounds = axis_delta * 0.5 + coord.bounds[circular_slice, :]
    else:
        mid_point_bounds = None

    # Get the deltas
    axis_delta = mid_point_coord.points
    # Add half of the deltas to the original points
    # if the coord is circular then include the last one, else, just take 0:-1
    mid_point_points = axis_delta * 0.5 + coord.points[circular_slice]

    # Try creating a coordinate of the same type as before, otherwise,
    # make an AuxCoord.
    try:
        mid_point_coord = coord.from_coord(coord).copy(
            mid_point_points, mid_point_bounds
        )
    except ValueError:
        mid_point_coord = iris.coords.AuxCoord.from_coord(coord).copy(
            mid_point_points, mid_point_bounds
        )

    return mid_point_coord


def cube_delta(cube, coord):
    """Given a cube calculate the difference between each value in the coord's direction.

    Parameters
    ----------
    coord :
        Either a Coord instance or the unique name of a coordinate in the cube.
        If a Coord instance is provided, it does not necessarily have to
        exist in the cube.

    Examples
    --------
    ::

        change_in_temperature_wrt_pressure = cube_delta(temperature_cube, 'pressure')

    Notes
    -----
    .. note:: Missing data support not yet implemented.

    .. note::
            This function does not maintain laziness when called; it realises data.
            See more at :doc:`/userguide/real_and_lazy_data`.

    """
    # handle the case where a user passes a coordinate name
    if isinstance(coord, str):
        coord = cube.coord(coord)

    if coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError(coord)

    # Try and get a coord dim
    delta_dims = cube.coord_dims(coord.name())
    if (
        coord.shape[0] == 1 and not getattr(coord, "circular", False)
    ) or not delta_dims:
        raise ValueError(
            "Cannot calculate delta over {!r} as it has length of 1.".format(
                coord.name()
            )
        )
    delta_dim = delta_dims[0]

    # Calculate the actual delta, taking into account whether the given
    # coordinate is circular.
    delta_cube_data = delta(
        cube.data, delta_dim, circular=getattr(coord, "circular", False)
    )

    # If the coord/dim is circular there is no change in cube shape
    if getattr(coord, "circular", False):
        delta_cube = cube.copy(data=delta_cube_data)
    else:
        # Subset the cube to the appropriate new shape by knocking off
        # the last row of the delta dimension.
        subset_slice = [slice(None, None)] * cube.ndim
        subset_slice[delta_dim] = slice(None, -1)
        delta_cube = cube[tuple(subset_slice)]
        delta_cube.data = delta_cube_data

    # Replace the delta_dim coords with midpoints
    # (no shape change if circular).
    for cube_coord in cube.coords(dimensions=delta_dim):
        delta_cube.replace_coord(
            _construct_midpoint_coord(
                cube_coord, circular=getattr(coord, "circular", False)
            )
        )

    delta_cube.rename("change_in_{}_wrt_{}".format(delta_cube.name(), coord.name()))

    return delta_cube


def differentiate(cube, coord_to_differentiate):
    r"""Calculate the differential of a given cube.

    Calculate the differential of a given cube with respect to the
    coord_to_differentiate.

    Parameters
    ----------
    coord_to_differentiate :
        Either a Coord instance or the unique name of a coordinate which
        exists in the cube.
        If a Coord instance is provided, it does not necessarily have to
        exist on the cube.

    Examples
    --------
    ::

        u_wind_acceleration = differentiate(u_wind_cube, 'forecast_time')

    The algorithm used is equivalent to:

    .. math::

        d_i = \frac{v_{i+1}-v_i}{c_{i+1}-c_i}

    Where ``d`` is the differential, ``v`` is the data value, ``c`` is
    the coordinate value and ``i`` is the index in the differential
    direction. Hence, in a normal situation if a cube has a shape
    (x: n; y: m) differentiating with respect to x will result in a cube
    of shape (x: n-1; y: m) and differentiating with respect to y will
    result in (x: n; y: m-1). If the coordinate to differentiate is
    :attr:`circular <iris.coords.DimCoord.circular>` then the resultant
    shape will be the same as the input cube.

    In the returned cube the `coord_to_differentiate` object is
    redefined such that the output coordinate values are set to the
    averages of the original coordinate values (i.e. the mid-points).
    Similarly, the output lower bounds values are set to the averages of
    the original lower bounds values and the output upper bounds values
    are set to the averages of the original upper bounds values. In more
    formal terms:

    * `C[i] = (c[i] + c[i+1]) / 2`
    * `B[i, 0] = (b[i, 0] + b[i+1, 0]) / 2`
    * `B[i, 1] = (b[i, 1] + b[i+1, 1]) / 2`

    where `c` and `b` represent the input coordinate values and bounds,
    and `C` and `B` the output coordinate values and bounds.

    .. note::
        Difference method used is the same as :func:`cube_delta`
        and therefore has the same limitations.

    .. note:: Spherical differentiation does not occur in this routine.

    .. note::
            This function does not maintain laziness when called; it realises data.
            See more at :doc:`/userguide/real_and_lazy_data`.

    """
    # Get the delta cube in the required differential direction.
    # This operation results in a copy of the original cube.
    delta_cube = cube_delta(cube, coord_to_differentiate)

    if isinstance(coord_to_differentiate, str):
        coord = cube.coord(coord_to_differentiate)
    else:
        coord = coord_to_differentiate

    delta_coord = _construct_delta_coord(coord)
    delta_dim = cube.coord_dims(coord.name())[0]

    # calculate delta_cube / delta_coord to give the differential.
    delta_cube = iris.analysis.maths.divide(delta_cube, delta_coord, delta_dim)

    # Update the standard name
    delta_cube.rename("derivative_of_{}_wrt_{}".format(cube.name(), coord.name()))
    return delta_cube


def _curl_subtract(a, b):
    """Straight forward wrapper to :func:`iris.analysis.maths.subtract`.

    Simple wrapper to :func:`iris.analysis.maths.subtract` to subtract
    two cubes, which deals with None in a way that makes sense in the
    context of curl.

    """
    from iris.cube import Cube

    # We are definitely dealing with cubes or None - otherwise we have a
    # programmer error...
    assert isinstance(a, Cube) or a is None
    assert isinstance(b, Cube) or b is None

    if a is None and b is None:
        return None
    elif a is None:
        c = b.copy(data=0 - b.data)
        return c
    elif b is None:
        return a.copy()
    else:
        return iris.analysis.maths.subtract(a, b)


def _curl_differentiate(cube, coord):
    """Straight forward wrapper to :func:`differentiate`.

    Simple wrapper to :func:`differentiate` to differentiate a cube and
    deal with None in a way that makes sense in the context of curl.

    """
    from iris.cube import Cube

    # We are definitely dealing with cubes/coords or None - otherwise we
    # have a programmer error...
    assert isinstance(cube, Cube) or cube is None
    assert isinstance(coord, iris.coords.Coord) or coord is None

    if cube is None:
        return None
    if coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError(coord)
    if coord.shape[0] <= 1:
        return None

    return differentiate(cube, coord)


def _curl_regrid(cube, prototype):
    """Straight forward wrapper to :ref`iris.cube.Cube.regridded`.

    Simple wrapper to :ref`iris.cube.Cube.regridded` to deal with None
    in a way that makes sense in the context of curl.

    """
    from iris.cube import Cube

    # We are definitely dealing with cubes or None - otherwise we have a
    # programmer error...
    assert isinstance(cube, Cube) or cube is None
    assert isinstance(prototype, Cube)

    if cube is None:
        result = None
    else:
        result = cube.regrid(prototype, iris.analysis.Linear())
    return result


def _copy_cube_transformed(src_cube, data, coord_func):
    """Return a new cube with the given data with the coordinates transformed.

    Returns a new cube based on the src_cube, but with the given data,
    and with the coordinates transformed via coord_func.

    The data must have the same number of dimensions as the source cube.

    """
    from iris.cube import Cube

    assert src_cube.ndim == data.ndim

    # Start with just the metadata and the data...
    new_cube = Cube(data)
    new_cube.metadata = src_cube.metadata
    new_cube.metadata = src_cube.metadata

    # ... and then create all the coordinates.

    # Record a mapping from old coordinate IDs to new coordinates,
    # for subsequent use in creating updated aux_factories.
    coord_mapping = {}

    def copy_coords(source_coords, add_method):
        for coord in source_coords:
            new_coord = coord_func(coord)
            add_method(new_coord, src_cube.coord_dims(coord))
            coord_mapping[id(coord)] = new_coord

    copy_coords(src_cube.dim_coords, new_cube.add_dim_coord)
    copy_coords(src_cube.aux_coords, new_cube.add_aux_coord)

    for factory in src_cube.aux_factories:
        new_cube.add_aux_factory(factory.updated(coord_mapping))

    return new_cube


def _curl_change_z(src_cube, z_coord, prototype_diff):
    # New data
    ind = [slice(None, None)] * src_cube.ndim
    z_dim = src_cube.coord_dims(z_coord)[0]
    ind[z_dim] = slice(-1, None)
    new_data = np.append(src_cube.data, src_cube.data[tuple(ind)], z_dim)

    # The existing z_coord doesn't fit the new data so make a
    # new cube using the prototype z_coord.
    local_z_coord = src_cube.coord(z_coord)
    new_local_z_coord = prototype_diff.coord(z_coord).copy()

    def coord_func(coord):
        if coord is local_z_coord:
            new_coord = new_local_z_coord
        else:
            new_coord = coord.copy()
        return new_coord

    result = _copy_cube_transformed(src_cube, new_data, coord_func)
    return result


def _coord_sin(coord):
    """Return a coordinate which represents sin(coord).

    Parameters
    ----------
    coord :
        Coord instance with values in either degrees or radians.

    """
    return _trig_method(coord, np.sin)


def _coord_cos(coord):
    """Return a coordinate which represents cos(coord).

    Parameters
    ----------
    coord :
        Coord instance with values in either degrees or radians.

    """
    return _trig_method(coord, np.cos)


def _trig_method(coord, trig_function):
    """Return a coordinate which represents trig_function(coord).

    Parameters
    ----------
    coord :
        Coord instance with points values in either degrees or radians.
    trig_function :
        Reference to a trigonometric function e.g. numpy.sin.

    """
    # If we are in degrees create a copy that is in radians.
    if coord.units == "degrees":
        coord = coord.copy()
        coord.convert_units("radians")

    trig_coord = iris.coords.AuxCoord.from_coord(coord)
    trig_coord.points = trig_function(coord.points)
    if coord.has_bounds():
        trig_coord.bounds = trig_function(coord.bounds)
    trig_coord.units = "1"
    trig_coord.rename("{}({})".format(trig_function.__name__, coord.name()))

    return trig_coord


def curl(i_cube, j_cube, k_cube=None):
    r"""Calculate the 2 or 3-dimensional spherical or cartesian curl.

    Calculate the 2-dimensional or 3-dimensional spherical or cartesian
    curl of the given vector of cubes.

    The cube standard names must match one of the combinations in
    :data:`DIRECTIONAL_NAMES`.

    As well as the standard x and y coordinates, this function requires each
    cube to possess a vertical or z-like coordinate (representing some form
    of height or pressure).  This can be a scalar or dimension coordinate.

    Parameters
    ----------
    i_cube :
        The i cube of the vector to operate on.
    j_cube :
        The j cube of the vector to operate on.
    k_cube : optional
        The k cube of the vector to operate on.

    Returns
    -------
    List of cubes i_cmpt_curl_cube, j_cmpt_curl_cube, k_cmpt_curl_cube

    Notes
    -----
    If the k-cube is not passed in then the 2-dimensional curl will
    be calculated, yielding the result: [None, None, k_cube].
    If the k-cube is passed in, the 3-dimensional curl will
    be calculated, returning 3 component cubes.

    All cubes passed in must have the same data units, and those units
    must be spatially-derived (e.g. 'm/s' or 'km/h').

    The calculation of curl is dependent on the type of
    :func:`~iris.coord_systems.CoordSystem` in the cube.
    If the :func:`~iris.coord_systems.CoordSystem` is either
    GeogCS or RotatedGeogCS, the spherical curl will be calculated; otherwise
    the cartesian curl will be calculated:

    * Cartesian curl
        * When cartesian calculus is used, i_cube is the u component,
          j_cube is the v component and k_cube is the w component.

          The Cartesian curl is defined as:

          .. math::

              \nabla\times \vec u =
              (\frac{\delta w}{\delta y} - \frac{\delta v}{\delta z})\vec a_i
              -
              (\frac{\delta w}{\delta x} - \frac{\delta u}{\delta z})\vec a_j
              +
              (\frac{\delta v}{\delta x} - \frac{\delta u}{\delta y})\vec a_k

    * Spherical curl
        * When spherical calculus is used, i_cube is the :math:`\phi` vector
          component (e.g. eastward), j_cube is the :math:`\theta` component
          (e.g. northward) and k_cube is the radial component.

          The spherical curl is defined as:

          .. math::

              \nabla\times \vec A = \frac{1}{r cos \theta}
              (\frac{\delta}{\delta \theta}
              (\vec A_\phi cos \theta) -
              \frac{\delta \vec A_\theta}{\delta \phi}) \vec r +
              \frac{1}{r}(\frac{1}{cos \theta}
              \frac{\delta \vec A_r}{\delta \phi} -
              \frac{\delta}{\delta r} (r \vec A_\phi))\vec \theta +
              \frac{1}{r}
              (\frac{\delta}{\delta r}(r \vec A_\theta) -
              \frac{\delta \vec A_r}{\delta \theta}) \vec \phi

          where phi is longitude, theta is latitude.

    .. note::

            This function does not maintain laziness when called; it realises data.
            See more at :doc:`/userguide/real_and_lazy_data`.

    """
    # Get the vector quantity names.
    # (i.e. ['easterly', 'northerly', 'vertical'])
    vector_quantity_names, phenomenon_name = spatial_vectors_with_phenom_name(
        i_cube, j_cube, k_cube
    )

    cubes = filter(None, [i_cube, j_cube, k_cube])

    # get the names of all coords binned into useful comparison groups
    coord_comparison = iris.analysis._dimensional_metadata_comparison(*cubes)

    bad_coords = coord_comparison["ungroupable_and_dimensioned"]
    if bad_coords:
        raise ValueError(
            "Coordinates found in one cube that describe "
            "a data dimension which weren't in the other "
            "cube ({}), try removing this coordinate.".format(
                ", ".join(group.name() for group in bad_coords)
            )
        )

    bad_coords = coord_comparison["resamplable"]
    if bad_coords:
        raise ValueError(
            "Some coordinates are different ({}), consider resampling.".format(
                ", ".join(group.name() for group in bad_coords)
            )
        )

    # Get the dim_coord, or None if none exist, for the xyz dimensions
    x_coord = i_cube.coord(axis="X")
    y_coord = i_cube.coord(axis="Y")
    z_coord = i_cube.coord(axis="Z")

    y_dim = i_cube.coord_dims(y_coord)[0]

    horiz_cs = i_cube.coord_system("CoordSystem")

    # Non-spherical coords?
    spherical_coords = isinstance(
        horiz_cs, (iris.coord_systems.GeogCS, iris.coord_systems.RotatedGeogCS)
    )
    if not spherical_coords:
        # TODO Implement some mechanism for conforming to a common grid
        dj_dx = _curl_differentiate(j_cube, x_coord)
        prototype_diff = dj_dx

        # i curl component (dk_dy - dj_dz)
        dk_dy = _curl_differentiate(k_cube, y_coord)
        dk_dy = _curl_regrid(dk_dy, prototype_diff)
        dj_dz = _curl_differentiate(j_cube, z_coord)
        dj_dz = _curl_regrid(dj_dz, prototype_diff)

        # TODO Implement resampling in the vertical (which regridding
        # does not support).
        if dj_dz is not None and dj_dz.data.shape != prototype_diff.data.shape:
            dj_dz = _curl_change_z(dj_dz, z_coord, prototype_diff)

        i_cmpt = _curl_subtract(dk_dy, dj_dz)
        dj_dz = dk_dy = None

        # j curl component (di_dz - dk_dx)
        di_dz = _curl_differentiate(i_cube, z_coord)
        di_dz = _curl_regrid(di_dz, prototype_diff)

        # TODO Implement resampling in the vertical (which regridding
        # does not support).
        if di_dz is not None and di_dz.data.shape != prototype_diff.data.shape:
            di_dz = _curl_change_z(di_dz, z_coord, prototype_diff)

        dk_dx = _curl_differentiate(k_cube, x_coord)
        dk_dx = _curl_regrid(dk_dx, prototype_diff)
        j_cmpt = _curl_subtract(di_dz, dk_dx)
        di_dz = dk_dx = None

        # k curl component ( dj_dx - di_dy)
        di_dy = _curl_differentiate(i_cube, y_coord)
        di_dy = _curl_regrid(di_dy, prototype_diff)
        # Since prototype_diff == dj_dx we don't need to recalculate dj_dx
        #        dj_dx = _curl_differentiate(j_cube, x_coord)
        #        dj_dx = _curl_regrid(dj_dx, prototype_diff)
        k_cmpt = _curl_subtract(dj_dx, di_dy)
        di_dy = dj_dx = None

        result = [i_cmpt, j_cmpt, k_cmpt]

    # Spherical coords (GeogCS or RotatedGeogCS).
    else:
        # A_\phi = i ; A_\theta = j ; A_\r = k
        # theta = lat ; phi = long ;
        # r_cmpt = 1 / (r * cos(lat)) *
        #    (d/dtheta (i_cube * sin(lat)) - d_j_cube_dphi)
        # phi_cmpt = 1/r * ( d/dr (r * j_cube) - d_k_cube_dtheta)
        # theta_cmpt = 1/r * ( 1/cos(lat) * d_k_cube_dphi - d/dr (r * i_cube)
        if y_coord.name() not in [
            "latitude",
            "grid_latitude",
        ] or x_coord.name() not in ["longitude", "grid_longitude"]:
            raise ValueError(
                "Expecting latitude as the y coord and "
                "longitude as the x coord for spherical curl."
            )

        # Get the radius of the earth - and check for sphericity
        ellipsoid = horiz_cs
        if isinstance(horiz_cs, iris.coord_systems.RotatedGeogCS):
            ellipsoid = horiz_cs.ellipsoid
        if ellipsoid:
            # TODO: Add a test for this
            r = ellipsoid.semi_major_axis
            r_unit = cf_units.Unit("m")
            spherical = ellipsoid.inverse_flattening == 0.0
        else:
            r = DEFAULT_SPHERICAL_EARTH_RADIUS
            r_unit = DEFAULT_SPHERICAL_EARTH_RADIUS_UNIT
            spherical = True

        if not spherical:
            raise ValueError("Cannot take the curl over a non-spherical ellipsoid.")

        lon_coord = x_coord.copy()
        lat_coord = y_coord.copy()
        lon_coord.convert_units("radians")
        lat_coord.convert_units("radians")
        lat_cos_coord = _coord_cos(lat_coord)

        # TODO Implement some mechanism for conforming to a common grid
        temp = iris.analysis.maths.multiply(i_cube, lat_cos_coord, y_dim)
        dicos_dtheta = _curl_differentiate(temp, lat_coord)
        prototype_diff = dicos_dtheta

        # r curl component: 1 / (r * cos(lat)) * (d_j_cube_dphi - dicos_dtheta)
        # Since prototype_diff == dicos_dtheta we don't need to
        # recalculate dicos_dtheta.
        d_j_cube_dphi = _curl_differentiate(j_cube, lon_coord)
        d_j_cube_dphi = _curl_regrid(d_j_cube_dphi, prototype_diff)
        new_lat_coord = d_j_cube_dphi.coord(axis="Y")
        new_lat_cos_coord = _coord_cos(new_lat_coord)
        lat_dim = d_j_cube_dphi.coord_dims(new_lat_coord)[0]
        r_cmpt = iris.analysis.maths.divide(
            _curl_subtract(d_j_cube_dphi, dicos_dtheta),
            r * new_lat_cos_coord,
            dim=lat_dim,
        )
        r_cmpt.units = r_cmpt.units / r_unit
        d_j_cube_dphi = dicos_dtheta = None

        # phi curl component: 1/r * ( drj_dr - d_k_cube_dtheta)
        drj_dr = _curl_differentiate(r * j_cube, z_coord)
        if drj_dr is not None:
            drj_dr.units = drj_dr.units * r_unit
        drj_dr = _curl_regrid(drj_dr, prototype_diff)
        d_k_cube_dtheta = _curl_differentiate(k_cube, lat_coord)
        d_k_cube_dtheta = _curl_regrid(d_k_cube_dtheta, prototype_diff)
        if drj_dr is None and d_k_cube_dtheta is None:
            phi_cmpt = None
        else:
            phi_cmpt = 1 / r * _curl_subtract(drj_dr, d_k_cube_dtheta)
            phi_cmpt.units = phi_cmpt.units / r_unit

        drj_dr = d_k_cube_dtheta = None

        # theta curl component: 1/r * ( 1/cos(lat) * d_k_cube_dphi - dri_dr )
        d_k_cube_dphi = _curl_differentiate(k_cube, lon_coord)
        d_k_cube_dphi = _curl_regrid(d_k_cube_dphi, prototype_diff)
        if d_k_cube_dphi is not None:
            d_k_cube_dphi = iris.analysis.maths.divide(d_k_cube_dphi, lat_cos_coord)
        dri_dr = _curl_differentiate(r * i_cube, z_coord)
        if dri_dr is not None:
            dri_dr.units = dri_dr.units * r_unit
        dri_dr = _curl_regrid(dri_dr, prototype_diff)
        if d_k_cube_dphi is None and dri_dr is None:
            theta_cmpt = None
        else:
            theta_cmpt = 1 / r * _curl_subtract(d_k_cube_dphi, dri_dr)
            theta_cmpt.units = theta_cmpt.units / r_unit
        d_k_cube_dphi = dri_dr = None

        result = [phi_cmpt, theta_cmpt, r_cmpt]

    for direction, cube in zip(vector_quantity_names, result):
        if cube is not None:
            cube.rename("%s curl of %s" % (direction, phenomenon_name))

    return result


#: Acceptable X-Y-Z standard name combinations that
#:  :func:`curl` can use (via :func:`spatial_vectors_with_phenom_name`).
DIRECTIONAL_NAMES: tuple[tuple[str, str, str], ...] = (
    ("u", "v", "w"),
    ("x", "y", "z"),
    ("i", "j", "k"),
    ("eastward", "northward", "upward"),
    ("easterly", "northerly", "vertical"),
    ("easterly", "northerly", "radial"),
)


def spatial_vectors_with_phenom_name(i_cube, j_cube, k_cube=None):
    """Given spatially dependent cubes, return a list of the spatial coordinate names.

    Given 2 or 3 spatially dependent cubes, return a list of the spatial
    coordinate names with appropriate phenomenon name.

    The cube standard names must match one of the combinations in
    :data:`DIRECTIONAL_NAMES`.

    This routine is designed to identify the vector quantites which each
    of the cubes provided represent and return a list of their 3d
    spatial dimension names and associated phenomenon.
    For example, given a cube of "u wind" and "v wind" the return value
    would be (['u', 'v', 'w'], 'wind')::

        >>> spatial_vectors_with_phenom_name(u_wind_cube, v_wind_cube) \
#doctest: +SKIP
        (['u', 'v', 'w'], 'wind')

    Notes
    -----
    This function maintains laziness when called; it does not realise data.
    See more at :doc:`/userguide/real_and_lazy_data`.


    """
    # Create a list of the standard_names of our incoming cubes
    # (excluding the k_cube if it is None).
    cube_standard_names = [
        cube.name() for cube in (i_cube, j_cube, k_cube) if cube is not None
    ]

    # Define a regular expr which represents (direction, phenomenon)
    # from the standard name of a cube.
    # e.g from "w wind" -> ("w", "wind")
    vector_qty = re.compile(r"([^\W_]+)[\W_]+(.*)")

    # Make a dictionary of {direction: phenomenon quantity}
    cube_directions, cube_phenomena = zip(
        *[re.match(vector_qty, std_name).groups() for std_name in cube_standard_names]
    )

    # Check that there is only one distinct phenomenon
    if len(set(cube_phenomena)) != 1:
        raise ValueError(
            "Vector phenomenon name not consistent between "
            "vector cubes. Got cube phenomena: {}; from "
            "standard names: {}.".format(
                ", ".join(cube_phenomena), ", ".join(cube_standard_names)
            )
        )

    # Get the appropriate direction list from the cube_directions we
    # have got from the standard name.
    direction = None
    for possible_direction in DIRECTIONAL_NAMES:
        # If this possible direction (minus the k_cube if it is none)
        # matches direction from the given cubes use it.
        if possible_direction[0 : len(cube_directions)] == cube_directions:
            direction = possible_direction

    # If we didn't get a match, raise an Exception
    if direction is None:
        direction_string = "; ".join(
            ", ".join(possible_direction) for possible_direction in DIRECTIONAL_NAMES
        )
        raise ValueError(
            "{} are not recognised vector cube_directions. "
            "Possible cube_directions are: {}.".format(
                cube_directions, direction_string
            )
        )

    return (direction, cube_phenomena[0])
