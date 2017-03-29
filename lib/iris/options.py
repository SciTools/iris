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
        self._override_default = False
        self.__dict__['conventions_override'] = conventions_override

    def __str__(self):
        msg = 'NetCDF options: conventions_override={}.'
        return msg.format(self.conventions_override)

    def __enter__(self):
        return

    def __exit__(self, exception_type, exception_val, exception_traceback):
        self.__dict__['conventions_override'] = self._override_default


netcdf = NetCDF()
