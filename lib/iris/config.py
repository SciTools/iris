# (C) British Crown Copyright 2010 - 2018, Met Office
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
Provides access to Iris-specific configuration values.

The default configuration values can be overridden by creating the file
``iris/etc/site.cfg``. If it exists, this file must conform to the format
defined by :mod:`ConfigParser`.

----------

.. py:data:: iris.config.TEST_DATA_DIR

    Local directory where test data exists.  Defaults to "test_data"
    sub-directory of the Iris package install directory. The test data
    directory supports the subset of Iris unit tests that require data.
    Directory contents accessed via :func:`iris.tests.get_data_path`.

.. py:data:: iris.config.PALETTE_PATH

    The full path to the Iris palette configuration directory

.. py:data:: iris.config.IMPORT_LOGGER

    The [optional] name of the logger to notify when first imported.

----------
"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six
from six.moves import configparser

import contextlib
import os.path
import sys
import warnings


# Returns simple string options
def get_option(section, option, default=None):
    """
    Returns the option value for the given section, or the default value
    if the section/option is not present.

    """
    value = default
    if config.has_option(section, option):
        value = config.get(section, option)
    return value


# Returns directory path options
def get_dir_option(section, option, default=None):
    """
    Returns the directory path from the given option and section, or
    returns the given default value if the section/option is not present
    or does not represent a valid directory.

    """
    path = default
    if config.has_option(section, option):
        c_path = config.get(section, option)
        if os.path.isdir(c_path):
            path = c_path
        else:
            msg = 'Ignoring config item {!r}:{!r} (section:option) as {!r}' \
                  ' is not a valid directory path.'
            warnings.warn(msg.format(section, option, c_path))
    return path


# Figure out the full path to the "iris" package.
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

# The full path to the configuration directory of the active Iris instance.
CONFIG_PATH = os.path.join(ROOT_PATH, 'etc')

# Load the optional "site.cfg" file if it exists.
if sys.version_info >= (3, 2):
    config = configparser.ConfigParser()
else:
    config = configparser.SafeConfigParser()
config.read([os.path.join(CONFIG_PATH, 'site.cfg')])


##################
# Resource options
_RESOURCE_SECTION = 'Resources'


TEST_DATA_DIR = get_dir_option(_RESOURCE_SECTION, 'test_data_dir',
                               default=os.path.join(os.path.dirname(__file__),
                                                    'test_data'))

# Override the data repository if the appropriate environment variable
# has been set.  This is used in setup.py in the TestRunner command to
# enable us to simulate the absence of external data.
override = os.environ.get("OVERRIDE_TEST_DATA_REPOSITORY")
if override:
    TEST_DATA_DIR = None
    if os.path.isdir(os.path.expanduser(override)):
        TEST_DATA_DIR = os.path.abspath(override)

PALETTE_PATH = get_dir_option(_RESOURCE_SECTION, 'palette_path',
                              os.path.join(CONFIG_PATH, 'palette'))

# Runtime options


class NetCDF(object):
    """Control Iris NetCDF options."""

    def __init__(self, conventions_override=None):
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
            set the CF Conventions version for the saved NetCDF files.

        Example usages:

        * Specify, for the lifetime of the session, that we want all cubes
          written to NetCDF to define their own CF Conventions versions::

            iris.config.netcdf.conventions_override = True
            iris.save('my_cube', 'my_dataset.nc')
            iris.save('my_second_cube', 'my_second_dataset.nc')

        * Specify, with a context manager, that we want a cube written to
          NetCDF to define its own CF Conventions version::

            with iris.config.netcdf.context(conventions_override=True):
                iris.save('my_cube', 'my_dataset.nc')

        """
        # Define allowed `__dict__` keys first.
        self.__dict__['conventions_override'] = None

        # Now set specific values.
        setattr(self, 'conventions_override', conventions_override)

    def __repr__(self):
        msg = 'NetCDF options: {}.'
        # Automatically populate with all currently accepted kwargs.
        options = ['{}={}'.format(k, v)
                   for k, v in six.iteritems(self.__dict__)]
        joined = ', '.join(options)
        return msg.format(joined)

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            # Can't add new names.
            msg = 'Cannot set option {!r} for {} configuration.'
            raise AttributeError(msg.format(name, self.__class__.__name__))
        if value is None:
            # Set an unset value to the name's default.
            value = self._defaults_dict[name]['default']
        if self._defaults_dict[name]['options'] is not None:
            # Replace a bad value with a good one if there is a defined set of
            # specified good values. If there isn't, we can assume that
            # anything goes.
            if value not in self._defaults_dict[name]['options']:
                good_value = self._defaults_dict[name]['default']
                wmsg = ('Attempting to set invalid value {!r} for '
                        'attribute {!r}. Defaulting to {!r}.')
                warnings.warn(wmsg.format(value, name, good_value))
                value = good_value
        self.__dict__[name] = value

    @property
    def _defaults_dict(self):
        # Set this as a property so that it isn't added to `self.__dict__`.
        return {'conventions_override': {'default': False,
                                         'options': [True, False]},
                }

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
