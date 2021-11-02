# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Code to implement vector rotation by angles, and inferring gridcell angles
from coordinate points and bounds.

"""

import cartopy.crs as ccrs
import numpy as np

import iris


def _3d_xyz_from_latlon(lon, lat):
    """
    Return locations of (lon, lat) in 3D space.

    Args:

    * lon, lat: (float array)
        Arrays of longitudes and latitudes, in degrees.
        Both the same shape.

    Returns:

    * xyz : (array, dtype=float64)
        Cartesian coordinates on a unit sphere.
        Shape is (3, <input-shape>).
        The x / y / z coordinates are in xyz[0 / 1 / 2].

    """
    lon1 = np.deg2rad(lon).astype(np.float64)
    lat1 = np.deg2rad(lat).astype(np.float64)

    x = np.cos(lat1) * np.cos(lon1)
    y = np.cos(lat1) * np.sin(lon1)
    z = np.sin(lat1)

    result = np.concatenate([array[np.newaxis] for array in (x, y, z)])

    return result


def _latlon_from_xyz(xyz):
    """
    Return arrays of lons+lats angles from xyz locations.

    Args:

    * xyz: (array)
        Array of 3-D cartesian coordinates.
        Shape (3, <input_points_dimensions>).
        x / y / z values are in xyz[0 / 1 / 2],

    Returns:

    * lonlat : (array)
        longitude and latitude position angles, in degrees.
        Shape (2, <input_points_dimensions>).
        The longitudes / latitudes are in lonlat[0 / 1].

    """
    lons = np.rad2deg(np.arctan2(xyz[1], xyz[0]))
    radii = np.sqrt(np.sum(xyz * xyz, axis=0))
    lats = np.rad2deg(np.arcsin(xyz[2] / radii))
    return np.array([lons, lats])


def _angle(p, q, r):
    """
    Estimate grid-angles to true-Eastward direction from positions in the same
    grid row, but at increasing column (grid-Eastward) positions.

    {P, Q, R} are locations of consecutive points in the same grid row.
    These could be successive points in a single grid,
        e.g. {T(i-1,j), T(i,j), T(i+1,j)}
    or a mixture of U/V and T gridpoints if row positions are aligned,
        e.g. {v(i,j), f(i,j), v(i+1,j)}.

    Method:

        Calculate dot product of a unit-vector parallel to P-->R, unit-scaled,
        with the unit eastward (true longitude) vector at Q.
        This value is cos(required angle).
        Discriminate between +/- angles by comparing latitudes of P and R.
        Return NaN where any P-->R are zero.

        .. NOTE::

            This method assumes that the vector PR is parallel to the surface
            at the longitude of Q, as it uses the length of PR as the basis for
            the cosine ratio.
            That is only exact when Q is at the same longitude as the midpoint
            of PR, and this typically causes errors which grow with increasing
            gridcell angle.
            However, we retain this method because it reproduces the "standard"
            gridcell-orientation-angle arrays found in files output by the CICE
            model, which presumably uses an equivalent calculation.

    Args:

    * p, q, r : (float array)
        Arrays of angles, in degrees.
        All the same shape.
        Shape is (2, <input_points_dimensions>).
        Longitudes / latitudes are in array[0 / 1].

    Returns:

    * angle : (float array)
        Grid angles relative to true-East, in degrees.
        Positive when grid-East is anticlockwise from true-East.
        Shape is same as <input_points_dimensions>.

    """
    mid_lons = np.deg2rad(q[0])

    pr = _3d_xyz_from_latlon(r[0], r[1]) - _3d_xyz_from_latlon(p[0], p[1])
    pr_norm = np.sqrt(np.sum(pr ** 2, axis=0))
    pr_top = pr[1] * np.cos(mid_lons) - pr[0] * np.sin(mid_lons)

    index = pr_norm == 0
    pr_norm[index] = 1

    cosine = np.maximum(np.minimum(pr_top / pr_norm, 1), -1)
    cosine[index] = 0

    psi = np.arccos(cosine) * np.sign(r[1] - p[1])
    psi[index] = np.nan

    return np.rad2deg(psi)


def gridcell_angles(x, y=None, cell_angle_boundpoints="mid-lhs, mid-rhs"):
    """
    Calculate gridcell orientations for an arbitrary 2-dimensional grid.

    The input grid is defined by two 2-dimensional coordinate arrays with the
    same dimensions (ny, nx), specifying the geolocations of a 2D mesh.

    Input values may be coordinate points (ny, nx) or bounds (ny, nx, 4).
    However, if points, the edges in the X direction are assumed to be
    connected by wraparound.

    Input can be either two arrays, two coordinates, or a single cube
    containing two suitable coordinates identified with the 'x' and 'y' axes.

    Args:

    The inputs (x [,y]) can be any of the following :

    * x (:class:`~iris.cube.Cube`):
        a grid cube with 2D X and Y coordinates, identified by 'axis'.
        The coordinates must be 2-dimensional with the same shape.
        The two dimensions represent grid dimensions in the order Y, then X.

    * x, y (:class:`~iris.coords.Coord`):
        X and Y coordinates, specifying grid locations on the globe.
        The coordinates must be 2-dimensional with the same shape.
        The two dimensions represent grid dimensions in the order Y, then X.
        If there is no coordinate system, they are assumed to be true
        longitudes and latitudes.  Units must convertible to 'degrees'.

    * x, y (2-dimensional arrays of same shape (ny, nx)):
        longitude and latitude cell center locations, in degrees.
        The two dimensions represent grid dimensions in the order Y, then X.

    * x, y (3-dimensional arrays of same shape (ny, nx, 4)):
        longitude and latitude cell bounds, in degrees.
        The first two dimensions are grid dimensions in the order Y, then X.
        The last index maps cell corners anticlockwise from bottom-left.

    Optional Args:

    * cell_angle_boundpoints (string):
        Controls which gridcell bounds locations are used to calculate angles,
        if the inputs are bounds or bounded coordinates.
        Valid values are 'lower-left, lower-right', which takes the angle from
        the lower left to the lower right corner, and 'mid-lhs, mid-rhs' which
        takes an angles between the average of the left-hand and right-hand
        pairs of corners.  The default is 'mid-lhs, mid-rhs'.

    Returns:

        angles : (2-dimensional cube)

            Cube of angles of grid-x vector from true Eastward direction for
            each gridcell, in degrees.
            It also has "true" longitude and latitude coordinates, with no
            coordinate system.
            When the input has coords, then the output ones are identical if
            the inputs are true-latlons, otherwise they are transformed
            true-latlon versions.
            When the input has bounded coords, then the output coords have
            matching bounds and centrepoints (possibly transformed).
            When the input is 2d arrays, or has unbounded coords, then the
            output coords have matching points and no bounds.
            When the input is 3d arrays, then the output coords have matching
            bounds, and the centrepoints are an average of the 4 boundpoints.

    """
    cube = None
    if hasattr(x, "add_aux_coord"):
        # Passed a cube : extract 'x' and ;'y' axis coordinates.
        cube = x  # Save for later checking.
        x, y = cube.coord(axis="x"), cube.coord(axis="y")

    # Now should have either 2 coords or 2 arrays.
    if not hasattr(x, "shape") or not hasattr(y, "shape"):
        msg = (
            "Inputs (x,y) must have array shape property."
            "Got type(x)={} and type(y)={}."
        )
        raise ValueError(msg.format(type(x), type(y)))

    x_coord, y_coord = None, None
    if hasattr(x, "bounds") and hasattr(y, "bounds"):
        # x and y are Coords.
        x_coord, y_coord = x.copy(), y.copy()

        # They must be angles : convert into degrees
        for coord in (x_coord, y_coord):
            if not coord.units.is_convertible("degrees"):
                msg = (
                    "Input X and Y coordinates must have angular "
                    'units. Got units of "{!s}" and "{!s}".'
                )
                raise ValueError(msg.format(x_coord.units, y_coord.units))
            coord.convert_units("degrees")

        if x_coord.ndim != 2 or y_coord.ndim != 2:
            msg = (
                "Coordinate inputs must have 2-dimensional shape. "
                "Got x-shape of {} and y-shape of {}."
            )
            raise ValueError(msg.format(x_coord.shape, y_coord.shape))
        if x_coord.shape != y_coord.shape:
            msg = (
                "Coordinate inputs must have same shape. "
                "Got x-shape of {} and y-shape of {}."
            )
            raise ValueError(msg.format(x_coord.shape, y_coord.shape))
        if cube:
            x_dims, y_dims = (cube.coord_dims(co) for co in (x, y))
            if x_dims != y_dims:
                msg = (
                    "X and Y coordinates must have the same cube "
                    "dimensions.  Got x-dims = {} and y-dims = {}."
                )
                raise ValueError(msg.format(x_dims, y_dims))
        cs = x_coord.coord_system
        if y_coord.coord_system != cs:
            msg = (
                "Coordinate inputs must have same coordinate system. "
                "Got x of {} and y of {}."
            )
            raise ValueError(msg.format(cs, y_coord.coord_system))

        # Base calculation on bounds if we have them, or points as a fallback.
        if x_coord.has_bounds() and y_coord.has_bounds():
            x, y = x_coord.bounds, y_coord.bounds
        else:
            x, y = x_coord.points, y_coord.points

        # Make sure these arrays are ordinary lats+lons, in degrees.
        if cs is not None:
            # Transform points into true lats + lons.
            crs_src = cs.as_cartopy_crs()
            crs_pc = ccrs.PlateCarree()

            def transform_xy_arrays(x, y):
                # Note: flatten, as transform_points is limited to 2D arrays.
                shape = x.shape
                x, y = (arr.flatten() for arr in (x, y))
                pts = crs_pc.transform_points(crs_src, x, y)
                x = pts[..., 0].reshape(shape)
                y = pts[..., 1].reshape(shape)
                return x, y

            # Transform the main reference points into standard lats+lons.
            x, y = transform_xy_arrays(x, y)

            # Likewise replace the original coordinates with transformed ones,
            # because the calculation also needs the centrepoint values.
            xpts, ypts = (coord.points for coord in (x_coord, y_coord))
            xbds, ybds = (coord.bounds for coord in (x_coord, y_coord))
            xpts, ypts = transform_xy_arrays(xpts, ypts)
            xbds, ybds = transform_xy_arrays(xbds, ybds)
            x_coord = iris.coords.AuxCoord(
                points=xpts,
                bounds=xbds,
                standard_name="longitude",
                units="degrees",
            )
            y_coord = iris.coords.AuxCoord(
                points=ypts,
                bounds=ybds,
                standard_name="latitude",
                units="degrees",
            )

    elif hasattr(x, "bounds") or hasattr(y, "bounds"):
        # One was a Coord, and the other not ?
        is_and_not = ("x", "y")
        if hasattr(y, "bounds"):
            is_and_not = reversed(is_and_not)
        msg = "Input {!r} is a Coordinate, but {!r} is not."
        raise ValueError(msg.format(*is_and_not))

    # Now have either 2 points arrays (ny, nx) or 2 bounds arrays (ny, nx, 4).
    # Construct (lhs, mid, rhs) where these represent 3 points at increasing
    # grid-x indices (columns).
    # Also make suitable X and Y coordinates for the result cube.
    if x.ndim == 2:
        # Data is points arrays.
        # Use previous + subsequent points along grid-x-axis as references.

        # PROBLEM: we assume that the rhs connects to the lhs, so we should
        # really only use this if data is full-longitudes (as a 'circular'
        # coordinate).
        # This is mentioned in the docstring, but we have no general means of
        # checking it.

        # NOTE: we take the 2d grid as presented, so the second dimension is
        # the 'X' dim.  Again, that is implicit + can't be checked.
        mid = np.array([x, y])
        lhs = np.roll(mid, 1, 2)
        rhs = np.roll(mid, -1, 2)
        if not x_coord:
            # Create coords for result cube : with no bounds.
            y_coord = iris.coords.AuxCoord(
                x, standard_name="latitude", units="degrees"
            )
            x_coord = iris.coords.AuxCoord(
                y, standard_name="longitude", units="degrees"
            )
    else:
        # Data is bounds arrays.
        # Use gridcell corners at different grid-x positions as references.
        # NOTE: so with bounds, we *don't* need full circular longitudes.
        xyz = _3d_xyz_from_latlon(x, y)
        # Support two different choices of reference points locations.
        angle_boundpoints_vals = {
            "mid-lhs, mid-rhs": "03_to_12",
            "lower-left, lower-right": "0_to_1",
        }
        bounds_pos = angle_boundpoints_vals.get(cell_angle_boundpoints)
        if bounds_pos == "0_to_1":
            lhs_xyz = xyz[..., 0]
            rhs_xyz = xyz[..., 1]
        elif bounds_pos == "03_to_12":
            lhs_xyz = 0.5 * (xyz[..., 0] + xyz[..., 3])
            rhs_xyz = 0.5 * (xyz[..., 1] + xyz[..., 2])
        else:
            msg = (
                'unrecognised cell_angle_boundpoints of "{}", '
                "must be one of {}"
            )
            raise ValueError(
                msg.format(
                    cell_angle_boundpoints, list(angle_boundpoints_vals.keys())
                )
            )
        if not x_coord:
            # Create bounded coords for result cube.
            # Use average of lhs+rhs points in 3d to get 'mid' points,
            # as coords without points are not allowed.
            mid_xyz = 0.5 * (lhs_xyz + rhs_xyz)
            mid_latlons = _latlon_from_xyz(mid_xyz)
            # Create coords with given bounds, and averaged centrepoints.
            x_coord = iris.coords.AuxCoord(
                points=mid_latlons[0],
                bounds=x,
                standard_name="longitude",
                units="degrees",
            )
            y_coord = iris.coords.AuxCoord(
                points=mid_latlons[1],
                bounds=y,
                standard_name="latitude",
                units="degrees",
            )

        # Convert lhs and rhs points back to latlon form -- IN DEGREES !
        lhs = _latlon_from_xyz(lhs_xyz)
        rhs = _latlon_from_xyz(rhs_xyz)
        # 'mid' is coord.points, whether from input or just made up.
        mid = np.array([x_coord.points, y_coord.points])

    # Do the angle calcs, and return as a suitable cube.
    angles = _angle(lhs, mid, rhs)
    result = iris.cube.Cube(
        angles, long_name="gridcell_angle_from_true_east", units="degrees"
    )
    result.add_aux_coord(x_coord, (0, 1))
    result.add_aux_coord(y_coord, (0, 1))
    return result


def rotate_grid_vectors(
    u_cube, v_cube, grid_angles_cube=None, grid_angles_kwargs=None
):
    """
    Rotate distance vectors from grid-oriented to true-latlon-oriented.

    Can also rotate by arbitrary angles, if they are passed in.

    .. Note::

        This operation overlaps somewhat in function with
        :func:`iris.analysis.cartography.rotate_winds`.
        However, that routine only rotates vectors according to transformations
        between coordinate systems.
        This function, by contrast, can rotate vectors by arbitrary angles.
        Most commonly, the angles are estimated solely from grid sampling
        points, using :func:`gridcell_angles` :  This allows operation on
        complex meshes defined by two-dimensional coordinates, such as most
        ocean grids.

    Args:

    * u_cube, v_cube : (cube)
        Cubes of grid-u and grid-v vector components.
        Units should be differentials of true-distance, e.g. 'm/s'.

    Optional args:

    * grid_angles_cube : (cube)
        gridcell orientation angles.
        Units must be angular, i.e. can be converted to 'radians'.
        If not provided, grid angles are estimated from 'u_cube' using the
        :func:`gridcell_angles` method.

    * grid_angles_kwargs : (dict or None)
        Additional keyword args to be passed to the :func:`gridcell_angles`
        method, if it is used.

    Returns:

        true_u, true_v : (cube)
            Cubes of true-north oriented vector components.
            Units are same as inputs.

        .. Note::

            Vector magnitudes will always be the same as the inputs.

    """
    u_out, v_out = (cube.copy() for cube in (u_cube, v_cube))
    if not grid_angles_cube:
        grid_angles_kwargs = grid_angles_kwargs or {}
        grid_angles_cube = gridcell_angles(u_cube, **grid_angles_kwargs)
    gridangles = grid_angles_cube.copy()
    gridangles.convert_units("radians")
    uu, vv, aa = (cube.data for cube in (u_out, v_out, gridangles))
    mags = np.sqrt(uu * uu + vv * vv)
    angs = np.arctan2(vv, uu) + aa
    uu, vv = mags * np.cos(angs), mags * np.sin(angs)

    # Promote all to masked arrays, and also apply mask at bad (NaN) angles.
    mask = np.isnan(aa)
    for cube in (u_out, v_out, aa):
        if hasattr(cube.data, "mask"):
            mask |= cube.data.mask
    u_out.data = np.ma.masked_array(uu, mask=mask)
    v_out.data = np.ma.masked_array(vv, mask=mask)

    return u_out, v_out
