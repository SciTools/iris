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
from functools import wraps
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
from iris._deprecation_helpers import ClassWrapperSameDocstring


_INTERPOLATE_DEPRECATION_WARNING = \
    "The module 'iris.analysis.interpolate' is deprecated."


# Define a common callpoint for deprecation warnings.
def _warn_deprecated(msg=None):
    if msg is None:
        msg = _INTERPOLATE_DEPRECATION_WARNING
    warnings.warn(msg)

# Issue a deprecation message when the module is loaded.
_warn_deprecated()


def nearest_neighbour_indices(cube, sample_points):
    msg = (_INTERPOLATE_DEPRECATION_WARNING + '\n' +
           'Please replace usage of '
           'iris.analysis.interpolate.nearest_neighbour_indices() '
           'with iris.coords.Coord.nearest_neighbour_index()).')
    _warn_deprecated(msg)
    return oldinterp.nearest_neighbour_indices(cube, sample_points)

nearest_neighbour_indices.__doc__ = oldinterp.nearest_neighbour_indices.__doc__


def extract_nearest_neighbour(cube, sample_points):
    msg = (_INTERPOLATE_DEPRECATION_WARNING + '\n' +
           'Please replace usage of '
           'iris.analysis.interpolate.extract_nearest_neighbour() with '
           'iris.cube.Cube.interpolate(..., scheme=iris.analysis.Nearest()).')
    _warn_deprecated(msg)
    return oldinterp.extract_nearest_neighbour(cube, sample_points)

extract_nearest_neighbour.__doc__ = oldinterp.extract_nearest_neighbour.__doc__


def nearest_neighbour_data_value(cube, sample_points):
    msg = (_INTERPOLATE_DEPRECATION_WARNING + '\n' +
           'Please replace usage of '
           'iris.analysis.interpolate.nearest_neighbour_data_value() with '
           'iris.cube.Cube.interpolate(..., scheme=iris.analysis.Nearest()).')
    _warn_deprecated(msg)
    return oldinterp.nearest_neighbour_data_value(cube, sample_points)

nearest_neighbour_data_value.__doc__ = \
    oldinterp.nearest_neighbour_data_value.__doc__


def regrid(source_cube, grid_cube, mode='bilinear', **kwargs):
    msg = (_INTERPOLATE_DEPRECATION_WARNING + '\n' +
           'Please replace usage of iris.analysis.interpolate.regrid() '
           'with iris.cube.Cube.regrid().')
    _warn_deprecated(msg)
    return oldinterp.regrid(source_cube, grid_cube, mode=mode, **kwargs)

regrid.__doc__ = oldinterp.regrid.__doc__


def regrid_to_max_resolution(cubes, **kwargs):
    msg = (_INTERPOLATE_DEPRECATION_WARNING + '\n' +
           'Please replace usage of '
           'iris.analysis.interpolate.regrid_to_max_resolution() '
           'with iris.cube.Cube.regrid().')
    _warn_deprecated(msg)
    return oldinterp.regrid_to_max_resolution(cubes, **kwargs)

regrid_to_max_resolution.__doc__ = oldinterp.regrid_to_max_resolution.__doc__


def linear(cube, sample_points, extrapolation_mode='linear'):
    msg = (_INTERPOLATE_DEPRECATION_WARNING + '\n' +
           'Please replace usage of iris.analysis.interpolate.linear() with '
           'iris.cube.Cube.interpolate(..., scheme=iris.analysis.Linear()).')
    _warn_deprecated(msg)
    return oldinterp.linear(cube, sample_points,
                            extrapolation_mode=extrapolation_mode)

linear.__doc__ = oldinterp.linear.__doc__


class Linear1dExtrapolator(six.with_metaclass(ClassWrapperSameDocstring,
                                              oldinterp.Linear1dExtrapolator)):
    @wraps(oldinterp.Linear1dExtrapolator.__init__)
    def __init__(self, interpolator):
        _warn_deprecated()
        super(Linear1dExtrapolator, self).__init__(interpolator)
