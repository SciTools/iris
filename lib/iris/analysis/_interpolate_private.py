# (C) British Crown Copyright 2010 - 2017, Met Office
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
This is the 'original' content of :mod:`iris.analysis.interpolate`, which has
now been deprecated.

A rename was essential to provide a deprecation warning on import of the
original name, while still providing this code for internal usage (for now)
without triggering the deprecation notice.

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
from iris._deprecation import warn_deprecated
import iris.exceptions


def linear(cube, sample_points, extrapolation_mode='linear'):
    """
    Return a cube of the linearly interpolated points given the desired
    sample points.

    Given a list of tuple pairs mapping coordinates (or coordinate names)
    to their desired values, return a cube with linearly interpolated values.
    If more than one coordinate is specified, the linear interpolation will be
    carried out in sequence, thus providing n-linear interpolation
    (bi-linear, tri-linear, etc.).

    If the input cube's data is masked, the result cube will have a data
    mask interpolated to the new sample points

    .. testsetup::

        import numpy as np

    For example:

        >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
        >>> sample_points = [('latitude', np.linspace(-90, 90, 10)),
        ...                  ('longitude', np.linspace(-180, 180, 20))]
        >>> iris.analysis.interpolate.linear(cube, sample_points)
        <iris 'Cube' of air_temperature / (K) (latitude: 10; longitude: 20)>

    .. note::

        By definition, linear interpolation requires all coordinates to
        be 1-dimensional.

    .. note::

        If a specified coordinate is single valued its value will be
        extrapolated to the desired sample points by assuming a gradient of
        zero.

    Args:

    * cube
        The cube to be interpolated.

    * sample_points
        List of one or more tuple pairs mapping coordinate to desired
        points to interpolate. Points may be a scalar or a numpy array
        of values.  Multi-dimensional coordinates are not supported.

    Kwargs:

    * extrapolation_mode - string - one of 'linear', 'nan' or 'error'

        * If 'linear' the point will be calculated by extending the
          gradient of closest two points.
        * If 'nan' the extrapolation point will be put as a NaN.
        * If 'error' a value error will be raised notifying of the
          attempted extrapolation.

    .. note::

        If the source cube's data, or any of its resampled coordinates,
        have an integer data type they will be promoted to a floating
        point data type in the result.

    .. deprecated:: 1.10

        The module :mod:`iris.analysis.interpolate` is deprecated.
        Please replace usage of
        :func:`iris.analysis.interpolate.linear`
        with :meth:`iris.cube.Cube.interpolate` using the scheme
        :class:`iris.analysis.Linear`.

    """
    if isinstance(sample_points, dict):
        sample_points = list(sample_points.items())

    # catch the case where a user passes a single (coord/name, value) pair rather than a list of pairs
    if sample_points and not (isinstance(sample_points[0], collections.Container) and not isinstance(sample_points[0], six.string_types)):
        raise TypeError('Expecting the sample points to be a list of tuple pairs representing (coord, points), got a list of %s.' % type(sample_points[0]))

    scheme = Linear(extrapolation_mode)
    return cube.interpolate(sample_points, scheme)


def _interp1d_rolls_y():
    """
    Determines if :class:`scipy.interpolate.interp1d` rolls its array `y` by
    comparing the shape of y passed into interp1d to the shape of its internal
    representation of y.

    SciPy v0.13.x+ no longer rolls the axis of its internal representation
    of y so we test for this occurring to prevent us subsequently
    extrapolating along the wrong axis.

    For further information on this change see, for example:
        * https://github.com/scipy/scipy/commit/0d906d0fc54388464603c63119b9e35c9a9c4601
          (the commit that introduced the change in behaviour).
        * https://github.com/scipy/scipy/issues/2621
          (a discussion on the change - note the issue is not resolved
          at time of writing).

    """
    y = np.arange(12).reshape(3, 4)
    f = interp1d(np.arange(3), y, axis=0)
    # If the initial shape of y and the shape internal to interp1d are *not*
    # the same then scipy.interp1d rolls y.
    return y.shape != f.y.shape


class Linear1dExtrapolator(object):
    """
    Extension class to :class:`scipy.interpolate.interp1d` to provide linear extrapolation.

    See also: :mod:`scipy.interpolate`.

    .. deprecated :: 1.10

    """
    roll_y = _interp1d_rolls_y()

    def __init__(self, interpolator):
        """
        Given an already created :class:`scipy.interpolate.interp1d` instance, return a callable object
        which supports linear extrapolation.

        .. deprecated :: 1.10

        """
        self._interpolator = interpolator
        self.x = interpolator.x
        # Store the y values given to the interpolator.
        self.y = interpolator.y
        """
        The y values given to the interpolator object.

        .. note:: These are stored with the interpolator.axis last.

        """
        # Roll interpolator.axis to the end if scipy no longer does it for us.
        if not self.roll_y:
            self.y = np.rollaxis(self.y, self._interpolator.axis, self.y.ndim)

    def all_points_in_range(self, requested_x):
        """Given the x points, do all of the points sit inside the interpolation range."""
        test = (requested_x >= self.x[0]) & (requested_x <= self.x[-1])
        if isinstance(test, np.ndarray):
            test = test.all()
        return test

    def __call__(self, requested_x):
        if not self.all_points_in_range(requested_x):
            # cast requested_x to a numpy array if it is not already.
            if not isinstance(requested_x, np.ndarray):
                requested_x = np.array(requested_x)

            # we need to catch the special case of providing a single value...
            remember_that_i_was_0d = requested_x.ndim == 0

            requested_x = requested_x.flatten()

            gt = np.where(requested_x > self.x[-1])[0]
            lt = np.where(requested_x < self.x[0])[0]
            ok = np.where( (requested_x >= self.x[0]) & (requested_x <= self.x[-1]) )[0]

            data_shape = list(self.y.shape)
            data_shape[-1] = len(requested_x)
            result = np.empty(data_shape, dtype=self._interpolator(self.x[0]).dtype)

            # Make a variable to represent the slice into the resultant data. (This will be updated in each of gt, lt & ok)
            interpolator_result_index = [slice(None, None)] * self.y.ndim

            if len(ok) != 0:
                interpolator_result_index[-1] = ok

                r = self._interpolator(requested_x[ok])
                # Reshape the properly formed array to put the interpolator.axis last i.e. dims 0, 1, 2 -> 0, 2, 1 if axis = 1
                axes = list(range(r.ndim))
                del axes[self._interpolator.axis]
                axes.append(self._interpolator.axis)

                result[interpolator_result_index] = r.transpose(axes)

            if len(lt) != 0:
                interpolator_result_index[-1] = lt

                grad = (self.y[..., 1:2] - self.y[..., 0:1]) / (self.x[1] - self.x[0])
                result[interpolator_result_index] = self.y[..., 0:1] + (requested_x[lt] - self.x[0]) * grad

            if len(gt) != 0:
                interpolator_result_index[-1] = gt

                grad = (self.y[..., -1:] - self.y[..., -2:-1]) / (self.x[-1] - self.x[-2])
                result[interpolator_result_index] = self.y[..., -1:] + (requested_x[gt] - self.x[-1]) * grad

            axes = list(range(len(interpolator_result_index)))
            axes.insert(self._interpolator.axis, axes.pop(axes[-1]))
            result = result.transpose(axes)

            if remember_that_i_was_0d:
                new_shape = list(result.shape)
                del new_shape[self._interpolator.axis]
                result = result.reshape(new_shape)

            return result
        else:
            return self._interpolator(requested_x)
