# (C) British Crown Copyright 2017, Met Office
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
Control runtime options of Iris.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import contextlib
import warnings


class NetCDF(object):
    """Control Iris NetCDF options."""

    def __init__(self, conventions_override=False):
        """
        Set up NetCDF processing options for Iris.

        Currently accepted kwargs:

        * conventions_override (bool):
            Define whether the CF Conventions version (e.g. `CF-1.6`) set when
            saving a cube to a NetCDF file should be defined by
            Iris (the default) or the cube being saved.

            If `False` (the default), specifies that Iris should set the
            CF Conventions version when saving cubes as NetCDF files.
            If `True`, specifies that the cubes being saved to NetCDF should
            set the cf conventions version for the saved NetCDF files.

        Example usages:

        * Specify, for the lifetime of the session, that we want all cubes
          written to NetCDF to define their own CF Conventions versions::

        >>> iris.options.netcdf(conventions_override=True)
        >>> iris.save('my_cube', 'my_dataset.nc')
        >>> iris.save('my_second_cube', 'my_second_dataset.nc')

        * Specify, with a context manager, that we want a cube written to
          NetCDF to define its own CF Conventions version::

        >>> with iris.options.netcdf(conventions_override=True):
        ...     iris.save('my_cube', 'my_dataset.nc')

        """
        self.__dict__['conventions_override'] = conventions_override

    def __str__(self):
        msg = 'NetCDF options: conventions_override={}.'
        return msg.format(self.conventions_override)

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            msg = "'Future' object has no attribute {!r}".format(name)
            raise AttributeError(msg)
        if value not in [True, False]:
            good_value = self._defaults_dict[name]
            wmsg = ('Attempting to set bad value {!r} for attribute {!r}. '
                    'Defaulting to {!r}.')
            warnings.warn(wmsg.format(value, name, good_value))
            value = good_value
        self.__dict__[name] = value

    @property
    def _defaults_dict(self):
        return {'conventions_override': False}

    @contextlib.contextmanager
    def context(self, **kwargs):
        """
        Allow temporary modification of the options via a context manager.
        Accepted kwargs are the same as can be supplied to the Option.

        """
        # Snapshot the starting state for restoration at the end of the
        # contextmanager block.
        starting_state = self.__dict__.copy()
        # Update the state to reflect the requested changes.
        for name, value in six.iteritems(kwargs):
            setattr(self, name, value)
        try:
            yield
        finally:
            # Return the state to the starting state.
            self.__dict__.clear()
            self.__dict__.update(starting_state)


netcdf = NetCDF()
