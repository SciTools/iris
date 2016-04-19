# (C) British Crown Copyright 2010 - 2016, Met Office
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
Interpolation and re-gridding routines.

See also: :mod:`NumPy <numpy>`, and :ref:`SciPy <scipy:modindex>`.

.. deprecated:: 1.10

    The module :mod:`iris.analysis.interpolate` is deprecated.
    Please use :meth:`iris.cube.regrid` or :meth:`iris.cube.interpolate` with
    the appropriate regridding and interpolation schemes from
    :mod:`iris.analysis` instead.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import collections
import warnings

import numpy as np
import scipy
import scipy.spatial
from scipy.interpolate.interpolate import interp1d

from iris.analysis import Linear
import iris.cube
import iris.coord_systems
import iris.coords
import iris.exceptions
import iris.analysis._interpolate_private as oldinterp

# Import deprecation support from the underlying module.
# Put it there so we can use it from elsewhere without triggering the
# deprecation warning (!)
from iris.analysis._interpolate_private import (
    _DEPRECATION_WARNSTRING,
    _warn_deprecated,
    _DeprecationWrapperMetaclass)


# Issue a deprecation message when the module is loaded, if enabled.
_warn_deprecated()


def nearest_neighbour_indices(cube, sample_points):
    msg = (_DEPRECATION_WARNSTRING + '\n' +
           'Please replace usage of '
           'iris.analysis.interpolate.nearest_neighbour_indices() '
           'with iris.coords.Coord.nearest_neighbour_index()).')
    _warn_deprecated(msg)
    return oldinterp.nearest_neighbour_indices(cube, sample_points)


def extract_nearest_neighbour(cube, sample_points):
    msg = (_DEPRECATION_WARNSTRING + '\n' +
           'Please replace usage of '
           'iris.analysis.interpolate.extract_nearest_neighbour() with '
           'iris.cube.Cube.interpolate(..., scheme=iris.analysis.Nearest()).')
    _warn_deprecated(msg)
    return oldinterp.extract_nearest_neighbour(cube, sample_points)


def nearest_neighbour_data_value(cube, sample_points):
    msg = (_DEPRECATION_WARNSTRING + '\n' +
           'Please replace usage of '
           'iris.analysis.interpolate.nearest_neighbour_data_value() with '
           'iris.cube.Cube.interpolate(..., scheme=iris.analysis.Nearest()).')
    _warn_deprecated(msg)
    return oldinterp.nearest_neighbour_data_value(cube, sample_points)


def regrid(source_cube, grid_cube, mode='bilinear', **kwargs):
    msg = (_DEPRECATION_WARNSTRING + '\n' +
           'Please replace usage of iris.analysis.interpolate.regrid() '
           'with iris.cube.Cube.regrid().')
    _warn_deprecated(msg)
    return oldinterp.regrid(source_cube, grid_cube, mode=mode, **kwargs)


def regrid_to_max_resolution(cubes, **kwargs):
    msg = (_DEPRECATION_WARNSTRING + '\n' +
           'Please replace usage of '
           'iris.analysis.interpolate.regrid_to_max_resolution() '
           'with iris.cube.Cube.regrid().')
    _warn_deprecated(msg)
    return oldinterp.regrid_to_max_resolution(cubes, **kwargs)


def linear(cube, sample_points, extrapolation_mode='linear'):
    msg = (_DEPRECATION_WARNSTRING + '\n' +
           'Please replace usage of iris.analysis.interpolate.linear() with '
           'iris.cube.Cube.interpolate(..., scheme=iris.analysis.Linear()).')
    _warn_deprecated(msg)
    return oldinterp.linear(cube, sample_points,
                            extrapolation_mode=extrapolation_mode)


class Linear1dExtrapolator(six.with_metaclass(_DeprecationWrapperMetaclass,
                                              oldinterp.Linear1dExtrapolator)):
    pass
