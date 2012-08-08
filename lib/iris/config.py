# (C) British Crown Copyright 2010 - 2012, Met Office
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

"""

import ConfigParser
import os.path
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
            warnings.warn("Ignoring config item '%s':'%s' (section:option) as '%s' is not a valid directory path." % \
                    (section, option, c_path))
    return path


# Figure out the full path to the "iris" package.
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

# The full path to the configuration directory of the active Iris instance.
CONFIG_PATH = os.path.join(ROOT_PATH, 'etc')

# Load the optional "site.cfg" file if it exists.
config = ConfigParser.SafeConfigParser()
config.read([os.path.join(CONFIG_PATH, 'site.cfg')])


##################
# Resource options
_RESOURCE_SECTION = 'Resources'


# Determine the default location to look for the sample data
_split_token = os.path.join("iris", "lib", "iris", "config.py")
_iris_parent_path = __file__.split(_split_token)[0]
_default_sample_data_path = os.path.join(_iris_parent_path, "iris_sample_data")

SAMPLE_DATA_DIR = get_dir_option(_RESOURCE_SECTION, 'iris_sample_data',
                                 _default_sample_data_path)
"""The full path to the sample data directory (required to build the documentation)."""



RESOURCE_DIR = get_dir_option(_RESOURCE_SECTION, 'dir',
                              os.path.join(ROOT_PATH, 'resources'))
"""The full path to the Iris resource directory."""


MASTER_DATA_REPOSITORY = get_dir_option(
        _RESOURCE_SECTION, 'master_data_repository')
"""Directory in which the master data repository normally exists. Can be seen
as the master for the :const:`DATA_REPOSITORY`."""


DATA_REPOSITORY = get_dir_option(_RESOURCE_SECTION, 'data_repository')
"""Local directory where test and sample data resources exist."""

# Override the data repository if the appropriate environment variable has been set
# This is used in setup.py in the TestRunner command to enable us to simulate the absence of external data
if os.environ.get("override_data_repository"):
    DATA_REPOSITORY = None

PALETTE_PATH = get_dir_option(_RESOURCE_SECTION, 'palette_path',
                              os.path.join(CONFIG_PATH, 'palette'))
"""The full path to the palette configuration directory of the active Iris instance."""


#################
# Logging options
_LOGGING_SECTION = 'Logging'


RULE_LOG_DIR = get_dir_option(_LOGGING_SECTION, 'rule_dir')
"""The [optional] full path to the rule logging directory."""


RULE_LOG_IGNORE = get_option(_LOGGING_SECTION, 'rule_ignore')
"""The [optional] list of users to ignore when logging rules."""


IMPORT_LOGGER = get_option(_LOGGING_SECTION, 'import_logger')
"""The [optional] name of the logger to notify when first imported."""
