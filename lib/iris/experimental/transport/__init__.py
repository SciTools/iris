# (C) British Crown Copyright 2013, Met Office
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
Calculating mass transports along lines of constant latitude.

"""

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import iris


class Data(namedtuple('Data', 't u v dx dy dzu dzv')):
    """
    Named tuple collection of gridded cubes.

    Args:

    * t:
        The T-cell gridded :class:`iris.cube.Cube`.

    * u:
        The U-cell gridded :class:`iris.cube.Cube`.

    * v:
        The V-cell gridded :class:`iris.cube.Cube`.

    * dx:
        The :class:`iris.cube.Cube` containing the delta-x components on
        the V grid.

    * dy:
        The :class:`iris.cube.Cube` containing the delta-y components on
        the U grid.

    * dzu:
        The :class:`iris.cube.Cube` containing the delta-z components on
        the U grid.

    * dzv:
        The :class:`iris.cube.Cube` containing the delta-z components on
        the V grid.

    """
    def payload(self):
        """Return the data payload of each :class:`iris.cube.Cube`."""
        item_data = []
        for attr in self._fields:
            value = getattr(self, attr)
            if attr == 't' or not isinstance(value, iris.cube.Cube):
                item_data.append(None)
            else:
                item_data.append(value.data)
        return Data(*item_data)


class PathData(namedtuple('PathData', 'uv dxdy dz')):
    """
    Named tuple collection of combined path data for
    N-vertices.

    Args:

    * uv:
        A :class:`numpy.ndarray` of combined U and V data.

    * dxdy:
        A :class:`numpy.ndarray` of combined DX and DY data.

    * dz:
        A :class:`numpy.ndarray` of combined DZU and DZV data.

    """


def _up_U(start_yx, end_yx, grid_type):
    if grid_type == 'C':
        y = start_yx[0]
        x = start_yx[1] - 1
    elif grid_type == 'B1':
        y = start_yx[0]
        x = start_yx[1]
    elif grid_type == 'B2':
        y = end_yx[0]
        x = start_yx[1]
    return y, x


def _down_U(start_yx, end_yx, grid_type):
    if grid_type == 'C':
        y = end_yx[0]
        x = end_yx[1] - 1
    elif grid_type == 'B1':
        y = start_yx[0]
        x = start_yx[1]
    elif grid_type == 'B2':
        y = end_yx[0]
        x = start_yx[1]
    return y, x


def _right_V(start_yx, end_yx, grid_type):
    if grid_type == 'C':
        y = start_yx[0] - 1
        x = start_yx[1]
    elif grid_type == 'B1':
        y = start_yx[0]
        x = start_yx[1]
    elif grid_type == 'B2':
        y = start_yx[0]
        x = end_yx[1]
    return y, x


def _left_V(start_yx, end_yx, grid_type):
    if grid_type == 'C':
        y = end_yx[0] - 1
        x = end_yx[1]
    elif grid_type == 'B1':
        y = start_yx[0]
        x = start_yx[1]
    elif grid_type == 'B2':
        y = start_yx[0]
        x = end_yx[1]
    return y, x


def _get_points(data, path, grid_type):
    """
    Calculate the path data for each sub-path vertex.

    Args:

    * data:
        The gridded :class:`Data`.

    * path:
        A list containing one or more sub-paths. Each sub-path is a
        list of (row, column) i-j space tuple pairs.

    * grid_type:
        The typle of Arakawa grid.

    Returns:
        The resulting :class:`PathData` for the given path.

    """
    _, u, v, dx, dy, dzu, dzv = data

    # Determine the total number of edges in the path.
    n = sum(len(sub_path) - 1 for sub_path in path)

    # Prepare empty arrays for our results
    uv = ma.empty((u.shape[:-2] + (n,)))
    dxdy = ma.empty(n)
    dz = ma.empty((u.shape[-3], n))

    ni = 0
    for sub_path in path:
        for start_yx, end_yx in zip(sub_path[:-1], sub_path[1:]):
            if not (0 <= start_yx[0] <= u.shape[-2] and
                    0 <= start_yx[1] <= u.shape[-1]):
                msg = 'Invalid sub-path point: {}'.format(start_yx)
                raise ValueError(msg)

            # Up => U
            if start_yx[0] + 1 == end_yx[0] and start_yx[1] == end_yx[1]:
                y, x = _up_U(start_yx, end_yx, grid_type)
                scale = 1
                uv_src = u
                dxdy_src = dy
                dz_src = dzu
            # Down => -U
            elif start_yx[0] - 1 == end_yx[0] and start_yx[1] == end_yx[1]:
                y, x = _down_U(start_yx, end_yx, grid_type)
                scale = -1
                uv_src = u
                dxdy_src = dy
                dz_src = dzu
            # Right => -V
            elif start_yx[1] + 1 == end_yx[1] and start_yx[0] == end_yx[0]:
                y, x = _right_V(start_yx, end_yx, grid_type)
                scale = -1
                uv_src = v
                dxdy_src = dx
                dz_src = dzv
            # Left => V
            elif start_yx[1] - 1 == end_yx[1] and start_yx[0] == end_yx[0]:
                y, x = _left_V(start_yx, end_yx, grid_type)
                scale = 1
                uv_src = v
                dxdy_src = dx
                dz_src = dzv
            else:
                msg = 'Invalid sub-path segment: ' \
                    '{0} -> {1})'.format(start_yx, end_yx)
                raise RuntimeError(msg)

            uv[..., ni] = scale * uv_src[..., y, x]
            dxdy[ni] = dxdy_src[y, x]
            dz[:, ni] = dz_src[:, y, x]
            ni += 1

    return PathData(uv, dxdy, dz)


def path_data(data, path, region_mask=None, grid_type='C'):
    """
    Calculate the path data for each vertex traversed within the
    one or more sub-paths of the given Arakawa grid.

    Args:

    * data:
        The gridded :class:`Data`.

    * path:
        A list containing one or more sub-paths. Each sub-path is a
        list of (row, column) i-j space tuple pairs.

    Kwargs:

    * region_mask:
        A boolean :class:`numpy.ndarray` land/sea mask.

    * grid_type:
        The type of Arakawa grid, either 'B' or 'C'.
        Defaults to 'C'. This is the only grid type currently supported.

    Returns:
        A 'C' grid :class:`PathData`.

    """
    # Only require the data payload of each cube.
    t, u, v, dx, dy, dzu, dzv = data.payload()

    grid_type = grid_type.upper()
    if grid_type not in ['B', 'C']:
        raise ValueError('Invalid grid type {!r}.'.format(grid_type))

    if region_mask is not None:
        u[..., region_mask] = ma.masked
        v[..., region_mask] = ma.masked

    # Package up the data payload.
    payload = Data(t, u, v, dx, dy, dzu, dzv)

    if grid_type == 'C':
        data = _get_points(payload, path, grid_type)
    else:
        # TODO: Extend to support 'B' grids.
        msg = 'Unhandled Arakawa grid type {!r}.'.format(grid_type)
        raise iris.exceptions.NotYetImplemented(msg)

        data1 = _get_points(payload, path, 'B1')
        data2 = _get_points(payload, path, 'B2')
        data = [data1, data2]

    return data


def path_transport(data, path, region_mask=None, grid_type='C'):
    """
    Calculate the transport overt the vertices traversed by the
    one or more sub-paths.

    Args:

    * data:
        The gridded :class:`Data`.

    * path:
        A list containing one or more sub-paths. Each sub-path is a
        list of (row, column) i-j space tuple pairs.

    Kwargs:

    * region_mask:
        A boolean :class:`numpy.ndarray` land/sea mask.

    * grid_type:
        The type of Arakawa grid, either 'B' or 'C'.
        Defaults to 'C'. This is the only grid type currently supported.

    Returns:
        The transport :class:`numpy.ndarray`.

    """
    uv, dxdy, dz = path_data(data, path,
                             region_mask=region_mask,
                             grid_type=grid_type)

    if dz.shape[-1] != uv.shape[-1]:
        dz = dz[:, np.newaxis]
    edge_transport = (uv * dz) + dxdy

    return edge_transport.sum(axis=-1)


def stream_function(data, path, region_mask=None, grid_type='C'):
    """
    Calculate the cumulative sum transport over the vertices traversed by the
    one or more sub-paths.

    Args:

    * data:
        The gridded :class:`Data`.

    * path:
        A list containing one or more sub-paths. Each sub-path is a
        list of (row, column) i-j space tuple pairs.

    Kwargs:

    * region_mask:
        A boolean :class:`numpy.ndarray` land/sea mask.

    * grid_type:
        The type of Arakawa grid, either 'B' or 'C'.
        Defaults to 'C'. This is the only grid type currently supported.

    Returns:
        The cumulative sum transport :class:`numpy.ndarray`.

    """
    transport = path_transport(data, path,
                               region_mask=region_mask,
                               grid_type=grid_type)

    return transport.cumsum(axis=-1)


def net_transport(data, path, region_mask=None, grid_type='C'):
    """
    Calculate the sum transport over the vertices traversed by the
    one or more sub-paths.

    Args:

    * data:
        The gridded :class:`Data`.

    * path:
        A list containing one or more sub-paths. Each sub-path is a
        list of (row, column) i-j space tuple pairs.

    Kwargs:

    * region_mask:
        A boolean :class:`numpy.ndarray` land/sea mask.

    * grid_type:
        The type of Arakawa grid, either 'B' or 'C'.
        Defaults to 'C'. This is the only grid type currently supported.

    Returns:
        The sum transport :class:`numpy.ndarray`.

    """
    transport = path_transport(data, path,
                               region_mask=region_mask,
                               grid_type=grid_type)

    return transport.sum(axis=-1)
