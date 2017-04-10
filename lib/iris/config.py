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
Provides access to Iris-specific configuration values.

The default configuration values can be overridden by creating the file
``iris/etc/site.cfg``. If it exists, this file must conform to the format
defined by :mod:`ConfigParser`.

----------

.. py:data:: iris.config.SAMPLE_DATA_DIR

    Local directory where sample data exists. Defaults to "sample_data"
    sub-directory of the Iris package install directory. The sample data
    directory supports the Iris gallery. Directory contents accessed via
    :func:`iris.sample_data_path`.

    .. deprecated:: 1.10

.. py:data:: iris.config.TEST_DATA_DIR

    Local directory where test data exists.  Defaults to "test_data"
    sub-directory of the Iris package install directory. The test data
    directory supports the subset of Iris unit tests that require data.
    Directory contents accessed via :func:`iris.tests.get_data_path`.

.. py:data:: iris.config.PALETTE_PATH

    The full path to the Iris palette configuration directory

.. py:data:: iris.config.IMPORT_LOGGER

    The [optional] name of the logger to notify when first imported.

.. py:data:: iris.config.RULE_LOG_DIR

    The [optional] full path to the rule logging directory used by
    :func:`iris.fileformats.pp.load()` and
    :func:`iris.fileformats.pp.save()`.

    .. deprecated:: 1.10

.. py:data:: iris.config.RULE_LOG_IGNORE

    The [optional] list of users to ignore when logging rules.

    .. deprecated:: 1.10

----------
"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six
from six.moves import configparser

import contextlib
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import re
import os.path
import warnings

import dask
import dask.multiprocessing

DISTRIBUTED_AVAILABLE = True
try:
    import distributed
except ImportError:
    DISTRIBUTED_AVAILABLE = False


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
config = configparser.SafeConfigParser()
config.read([os.path.join(CONFIG_PATH, 'site.cfg')])


##################
# Resource options
_RESOURCE_SECTION = 'Resources'


SAMPLE_DATA_DIR = get_dir_option(
    _RESOURCE_SECTION, 'sample_data_dir',
    default=os.path.join(os.path.dirname(__file__), 'sample_data'))

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


#################
# Logging options
_LOGGING_SECTION = 'Logging'


RULE_LOG_DIR = get_dir_option(_LOGGING_SECTION, 'rule_dir')


RULE_LOG_IGNORE = get_option(_LOGGING_SECTION, 'rule_ignore')


IMPORT_LOGGER = get_option(_LOGGING_SECTION, 'import_logger')


#################
# Runtime options

class Option(object):
    """
    An abstract superclass to enforce certain key behaviours for all `Option`
    classes.

    """
    @property
    def _defaults_dict(self):
        raise NotImplementedError

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            # Can't add new names.
            msg = "'Option' object has no attribute {!r}".format(name)
            raise AttributeError(msg)
        if value is None:
            # Set an explicitly unset value to the default value for the name.
            value = self._defaults_dict[name]['default']
        if self._defaults_dict[name]['options'] is not None:
            # Replace a bad value with the default if there is a defined set of
            # specified good values.
            if value not in self._defaults_dict[name]['options']:
                good_value = self._defaults_dict[name]['default']
                wmsg = ('Attempting to set bad value {!r} for attribute {!r}. '
                        'Defaulting to {!r}.')
                warnings.warn(wmsg.format(value, name, good_value))
                value = good_value
        self.__dict__[name] = value

    def context(self):
        raise NotImplementedError


class Parallel(Option):
    """
    Control dask parallel processing options for Iris.

    """
    def __init__(self, scheduler=None, num_workers=None):
        """
        Set up options for dask parallel processing.

        Currently accepted kwargs:

        * scheduler:
            The scheduler used to run a dask graph. Must be set to one of:

            * 'threaded': (default)
                The scheduler processes the graph in parallel using a
                thread pool. Good for processing dask arrays and dataframes.
            * 'multiprocessing':
                The scheduler processes the graph in parallel using a
                process pool. Good for processing dask bags.
            * 'async':
                The scheduler runs synchronously (not in parallel). Good for
                debugging.
            * The IP address and port of a distributed scheduler:
                Specifies the location of a distributed scheduler that has
                already been set up. The distributed scheduler will process the
                graph.

            For more information see
            http://dask.pydata.org/en/latest/scheduler-overview.html.

        * num_workers:
            The number of worker threads or processess to use to run the dask
            graph in parallel. Defaults to 1 (that is, processed serially).

            .. note::
                The value for `num_workers` cannot be set to greater than the
                number of CPUs available on the host system. If such a value is
                requested, `num_workers` is automatically set to 1 less than
                the number of CPUs available on the host system.

            .. note::
                Only the 'threaded' and 'multiprocessing' schedulers support
                the `num_workers` kwarg. If it is specified with the `async` or
                `distributed` scheduler, the kwarg is ignored:

                * The 'async' scheduler runs serially so will only use a single
                worker.
                * The number of workers for the 'distributed' scheduler must be
                defined when setting up the distributed scheduler. For more
                information on setting up distributed schedulers, see
                https://distributed.readthedocs.io/en/latest/index.html.

        Example usages:

        * Specify that we want to load a cube with dask parallel processing
        using multiprocessing with six worker processes::

            iris.options.parallel(scheduler='multiprocessing', num_workers=6)
            iris.load('my_dataset.nc')

        * Specify, with a context manager, that we want to load a cube with
        dask parallel processing using four worker threads::

            with iris.options.parallel(scheduler='threaded', num_workers=4):
                iris.load('my_dataset.nc')

        * Run dask parallel processing using a distributed scheduler that has
        been set up at the IP address and port at ``192.168.0.219:8786``::

            iris.options.parallel(scheduler='192.168.0.219:8786')

        """
        # Set `__dict__` keys first.
        self.__dict__['_scheduler'] = scheduler
        self.__dict__['scheduler'] = None
        self.__dict__['num_workers'] = None
        self.__dict__['dask_scheduler'] = None

        # Set `__dict__` values for each kwarg.
        setattr(self, 'scheduler', scheduler)
        setattr(self, 'num_workers', num_workers)
        setattr(self, 'dask_scheduler', self.get('scheduler'))

        # Activate the specified dask options.
        self._set_dask_options()

    def __setattr__(self, name, value):
        if value is None:
            value = self._defaults_dict[name]['default']
        attr_setter = getattr(self, 'set_{}'.format(name))
        value = attr_setter(value)
        super(Parallel, self).__setattr__(name, value)

    @property
    def _defaults_dict(self):
        """
        Define the default value and available options for each settable
        `kwarg` of this `Option`.

        Note: `'options'` can be set to `None` if it is not reasonable to
        specify all possible options. For example, this may be reasonable if
        the `'options'` were a range of numbers.

        """
        return {'_scheduler': {'default': None, 'options': None},
                'scheduler': {'default': 'threaded',
                              'options': ['threaded',
                                          'multiprocessing',
                                          'async',
                                          'distributed']},
                'num_workers': {'default': 1, 'options': None},
                'dask_scheduler': {'default': None, 'options': None},
                }

    def set__scheduler(self, value):
        return value

    def set_scheduler(self, value):
        default = self._defaults_dict['scheduler']['default']
        if value is None:
            value = default
        elif re.match(r'^(\d{1,3}\.){3}\d{1,3}:\d{1,5}$', value):
            if DISTRIBUTED_AVAILABLE:
                value = 'distributed'
            else:
                # Distributed not available.
                wmsg = 'Cannot import distributed. Defaulting to {}.'
                warnings.warn(wmsg.format(default))
                self.set_scheduler(default)
        elif value not in self._defaults_dict['scheduler']['options']:
            # Invalid value for `scheduler`.
            wmsg = 'Invalid value for scheduler: {!r}. Defaulting to {}.'
            warnings.warn(wmsg.format(value, default))
            self.set_scheduler(default)
        return value

    def set_num_workers(self, value):
        default = self._defaults_dict['num_workers']['default']
        scheduler = self.get('scheduler')
        if scheduler == 'async' and value != default:
            wmsg = 'Cannot set `num_workers` for the serial scheduler {!r}.'
            warnings.warn(wmsg.format(scheduler))
            value = None
        elif scheduler == 'distributed' and value != default:
            wmsg = ('Attempting to set `num_workers` with the {!r} scheduler '
                    'requested. Please instead specify number of workers when '
                    'setting up the distributed scheduler. See '
                    'https://distributed.readthedocs.io/en/latest/index.html '
                    'for more details.')
            warnings.warn(wmsg.format(scheduler))
            value = None
        else:
            if value is None:
                value = default
            if value >= cpu_count():
                # Limit maximum CPUs used to 1 fewer than all available CPUs.
                wmsg = ('Requested more CPUs ({}) than total available ({}). '
                        'Limiting number of used CPUs to {}.')
                warnings.warn(wmsg.format(value, cpu_count(), cpu_count()-1))
                value = cpu_count() - 1
        return value

    def set_dask_scheduler(self, scheduler):
        if scheduler == 'threaded':
            value = dask.threaded.get
        elif scheduler == 'multiprocessing':
            value = dask.multiprocessing.get
        elif scheduler == 'async':
            value = dask.async.get_sync
        elif scheduler == 'distributed':
            value = self.get('_scheduler')
        return value

    def _set_dask_options(self):
        """
        Use `dask.set_options` to globally apply the options specified at
        instantiation, either for the lifetime of the session or
        context manager.

        """
        scheduler = self.get('scheduler')
        num_workers = self.get('num_workers')
        get = self.get('dask_scheduler')
        pool = None

        if scheduler in ['threaded', 'multiprocessing']:
            pool = ThreadPool(num_workers)
        if scheduler == 'distributed':
            get = distributed.Client(get).get

        dask.set_options(get=get, pool=pool)

    def get(self, item):
        return getattr(self, item)

    @contextlib.contextmanager
    def context(self, **kwargs):
        # Snapshot the starting state for restoration at the end of the
        # contextmanager block.
        starting_state = self.__dict__.copy()
        # Update the state to reflect the requested changes.
        for name, value in six.iteritems(kwargs):
            setattr(self, name, value)
        self._set_dask_options()
        try:
            yield
        finally:
            # Return the state to the starting state.
            self.__dict__.clear()
            self.__dict__.update(starting_state)
            self._set_dask_options()


parallel = Parallel()