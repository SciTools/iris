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

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import re
import warnings

import dask
import dask.multiprocessing
import distributed


class Parallel(object):
    """
    Control dask parallel processing options for Iris.

    """
    def __init__(self, scheduler='threaded', num_workers=1):
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

        >>> iris.options.parallel(scheduler='multiprocessing', num_workers=6)
        >>> iris.load('my_dataset.nc')

        * Specify, with a context manager, that we want to load a cube with
        dask parallel processing using four worker threads::

        >>> with iris.options.parallel(scheduler='threaded', num_workers=4):
        ...     iris.load('my_dataset.nc')

        * Run dask parallel processing using a distributed scheduler that has
        been set up at the IP address and port at ``192.168.0.219:8786``::

        >>> iris.options.parallel(scheduler='192.168.0.219:8786')

        """
        # Set some defaults first of all.
        self._default_scheduler = 'threaded'
        self._default_num_workers = 1

        self.scheduler = scheduler
        self.num_workers = num_workers

        self._dask_scheduler = None

        # Activate the specified dask options.
        self._set_dask_options()

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        if value is None:
            value = self._default_scheduler
        if value == 'threaded':
            self._scheduler = value
            self.dask_scheduler = dask.threaded.get
        elif value == 'multiprocessing':
            self._scheduler = value
            self.dask_scheduler = dask.multiprocessing.get
        elif value == 'async':
            self._scheduler = value
            self.dask_scheduler = dask.async.get_sync
        elif re.match(r'^(\d{1,3}\.){3}\d{1,3}:\d{1,5}$', value):
            self._scheduler = 'distributed'
            self.dask_scheduler = value
        else:
            # Invalid value for `scheduler`.
            wmsg = 'Invalid value for scheduler: {!r}. Defaulting to {}.'
            warnings.warn(wmsg.format(value, self._default_scheduler))
            self.scheduler = self._default_scheduler

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        if self.scheduler == 'async' and value != self._default_num_workers:
            wmsg = 'Cannot set `num_workers` for the serial scheduler {!r}.'
            warnings.warn(wmsg.format(self.scheduler))
            value = None
        elif (self.scheduler == 'distributed' and
                      value != self._default_num_workers):
            wmsg = ('Attempting to set `num_workers` with the {!r} scheduler '
                    'requested. Please instead specify number of workers when '
                    'setting up the distributed scheduler. See '
                    'https://distributed.readthedocs.io/en/latest/index.html '
                    'for more details.')
            warnings.warn(wmsg.format(self.scheduler))
            value = None
        else:
            if value is None:
                value = self._default_num_workers
            if value >= cpu_count():
                # Limit maximum CPUs used to 1 fewer than all available CPUs.
                wmsg = ('Requested more CPUs ({}) than total available ({}). '
                        'Limiting number of used CPUs to {}.')
                warnings.warn(wmsg.format(value, cpu_count(), cpu_count()-1))
                value = cpu_count() - 1
        self._num_workers = value

    @property
    def dask_scheduler(self):
        return self._dask_scheduler

    @dask_scheduler.setter
    def dask_scheduler(self, value):
        self._dask_scheduler = value

    def _set_dask_options(self):
        """
        Use `dask.set_options` to globally apply the options specified at
        instantiation, either for the lifetime of the session or
        context manager.

        """
        get = self.dask_scheduler
        pool = None
        if self.scheduler in ['threaded', 'multiprocessing']:
            pool = ThreadPool(self.num_workers)
        if self.scheduler == 'distributed':
            get = distributed.Client(self.dask_scheduler).get

        dask.set_options(get=get, pool=pool)

    def get(self, item):
        return getattr(self, item)

    def __enter__(self):
        return

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.num_workers = self._default_num_workers
        self.scheduler = self._default_scheduler
        self._set_dask_options()


parallel = Parallel
