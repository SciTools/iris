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
import re
import warnings

import dask
import distributed


class Parallel(object):
    """
    Control dask parallel-processing options for Iris.

    """
    _default_scheduler = 'threaded'

    def __init__(self, num_workers=1, scheduler=_default_scheduler, pool=None):
        self._num_workers = num_workers
        self._scheduler = scheduler
        self.pool = pool

        self._dask_scheduler = None

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        if value >= cpu_count():
            # Limit maximum CPUs used to 1 fewer than all available CPUs.
            wmsg = ('Requested more CPUs ({}) than total available ({}). '
                    'Limiting number of used CPUs to {}.')
            warnings.warn(wmsg.format(value, cpu_count(), cpu_count()-1))
            value = cpu_count() - 1
        self._num_workers = value

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
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
            self._scheduler = value
            self.dask_scheduler = distributed.Client.get
        else:
            # Invalid value for `scheduler`.
            wmsg = 'Invalid value for scheduler: {!r}. Defaulting to {}.'
            warnings.warn(wmsg.format(value, self._default_scheduler))
            self.scheduler = self._default_scheduler

    @property
    def dask_scheduler(self):
        return self._dask_scheduler

    @dask_scheduler.setter
    def dask_scheduler(self, value):
        self._dask_scheduler = value


parallel = Parallel
