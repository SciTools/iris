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
"""Unit tests for the :class:`iris.config.Parallel` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import warnings

import dask

from iris.config import Parallel
from iris.tests import mock


class Test__operation(tests.IrisTest):
    def setUp(self):
        self.parallel = Parallel()

    def test_bad_name(self):
        # Check we can't do `iris.config.parallel.foo = 'bar`.
        exp_emsg = "'Parallel' object has no attribute 'foo'"
        with self.assertRaisesRegexp(AttributeError, exp_emsg):
            self.parallel.foo = 'bar'

    def test_bad_name__contextmgr(self):
        # Check we can't do `with iris.config.parallel.context('foo'='bar')`.
        exp_emsg = "'Parallel' object has no attribute 'foo'"
        with self.assertRaisesRegexp(AttributeError, exp_emsg):
            with self.parallel.context(foo='bar'):
                pass


class Test__set_dask_options(tests.IrisTest):
    def setUp(self):
        ThreadPool = 'iris.config.ThreadPool'
        self.pool = mock.sentinel.pool
        self.patch_ThreadPool = self.patch(ThreadPool, return_value=self.pool)
        self.default_num_workers = 1

        Client = 'distributed.Client'
        self.address = '192.168.0.128:8786'
        mocker = mock.Mock(get=self.address)
        self.patch_Client = self.patch(Client, return_value=mocker)

        set_options = 'dask.set_options'
        self.patch_set_options = self.patch(set_options)

    def test_default(self):
        Parallel()
        self.assertEqual(self.patch_Client.call_count, 0)
        self.patch_ThreadPool.assert_called_once_with(self.default_num_workers)

        pool = self.pool
        get = dask.threaded.get
        self.patch_set_options.assert_called_once_with(pool=pool, get=get)

    def test__five_workers(self):
        n_workers = 5
        Parallel(num_workers=n_workers)
        self.assertEqual(self.patch_Client.call_count, 0)
        self.patch_ThreadPool.assert_called_once_with(n_workers)

        pool = self.pool
        get = dask.threaded.get
        self.patch_set_options.assert_called_once_with(pool=pool, get=get)

    def test__five_workers__contextmgr(self):
        n_workers = 5
        options = Parallel()
        pool = self.pool
        get = dask.threaded.get

        with options.context(num_workers=n_workers):
            self.assertEqual(self.patch_Client.call_count, 0)
            self.patch_ThreadPool.assert_called_with(n_workers)

            self.patch_set_options.assert_called_with(pool=pool, get=get)

        self.patch_ThreadPool.assert_called_with(self.default_num_workers)
        self.patch_set_options.assert_called_with(pool=pool, get=get)

    def test_threaded(self):
        scheduler = 'threaded'
        Parallel(scheduler=scheduler)
        self.assertEqual(self.patch_Client.call_count, 0)
        self.patch_ThreadPool.assert_called_once_with(self.default_num_workers)

        pool = self.pool
        get = dask.threaded.get
        self.patch_set_options.assert_called_once_with(pool=pool, get=get)

    def test_multiprocessing(self):
        scheduler = 'multiprocessing'
        Parallel(scheduler=scheduler)
        self.assertEqual(self.patch_Client.call_count, 0)
        self.patch_ThreadPool.assert_called_once_with(self.default_num_workers)

        pool = self.pool
        get = dask.multiprocessing.get
        self.patch_set_options.assert_called_once_with(pool=pool, get=get)

    def test_multiprocessing__contextmgr(self):
        scheduler = 'multiprocessing'
        options = Parallel()
        with options.context(scheduler=scheduler):
            self.assertEqual(self.patch_Client.call_count, 0)
            self.patch_ThreadPool.assert_called_with(self.default_num_workers)

            pool = self.pool
            get = dask.multiprocessing.get
            self.patch_set_options.assert_called_with(pool=pool, get=get)

        default_get = dask.threaded.get
        self.patch_ThreadPool.assert_called_with(self.default_num_workers)
        self.patch_set_options.assert_called_with(pool=pool,
                                                  get=default_get)

    def test_async(self):
        scheduler = 'async'
        Parallel(scheduler=scheduler)
        self.assertEqual(self.patch_Client.call_count, 0)
        self.assertEqual(self.patch_ThreadPool.call_count, 0)

        pool = self.pool
        get = dask.async.get_sync
        self.patch_set_options.assert_called_once_with(pool=None, get=get)

    def test_distributed(self):
        scheduler = self.address
        Parallel(scheduler=scheduler)
        self.assertEqual(self.patch_ThreadPool.call_count, 0)

        get = scheduler
        self.patch_Client.assert_called_once_with(get)

        self.patch_set_options.assert_called_once_with(pool=None, get=get)


class Test_set_schedulers(tests.IrisTest):
    # Check that the correct scheduler is chosen given the inputs.
    def setUp(self):
        self.patch('iris.config.Parallel._set_dask_options')

    def test_default(self):
        opts = Parallel()
        result = opts.get('scheduler')
        expected = opts._defaults_dict['scheduler']['default']
        self.assertEqual(result, expected)

    def test_threaded(self):
        scheduler = 'threaded'
        opts = Parallel(scheduler=scheduler)
        result = opts.get('scheduler')
        self.assertEqual(result, scheduler)

    def test_multiprocessing(self):
        scheduler = 'multiprocessing'
        opts = Parallel(scheduler=scheduler)
        result = opts.get('scheduler')
        self.assertEqual(result, scheduler)

    def test_async(self):
        scheduler = 'async'
        opts = Parallel(scheduler=scheduler)
        result = opts.get('scheduler')
        self.assertEqual(result, scheduler)

    def test_distributed(self):
        scheduler = '192.168.0.128:8786'
        opts = Parallel(scheduler=scheduler)
        result = opts.get('scheduler')
        self.assertEqual(result, 'distributed')

    def test_bad(self):
        scheduler = 'wibble'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            opts = Parallel(scheduler=scheduler)
        result = opts.get('scheduler')
        expected = opts._defaults_dict['scheduler']['default']
        self.assertEqual(result, expected)
        exp_wmsg = 'Invalid value for scheduler: {!r}'
        six.assertRegex(self, str(w[0].message), exp_wmsg.format(scheduler))


class Test_set_num_workers(tests.IrisTest):
    # Check that the correct `num_workers` are chosen given the inputs.
    def setUp(self):
        self.patch('iris.config.Parallel._set_dask_options')

    def test_default(self):
        opts = Parallel()
        result = opts.get('num_workers')
        expected = opts._defaults_dict['num_workers']['default']
        self.assertEqual(result, expected)

    def test_basic(self):
        n_workers = 5
        opts = Parallel(num_workers=n_workers)
        result = opts.get('num_workers')
        self.assertEqual(result, n_workers)

    def test_too_many_workers(self):
        max_cpus = 8
        n_workers = 12
        with mock.patch('multiprocessing.cpu_count', return_value=max_cpus):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                opts = Parallel(num_workers=n_workers)
        result = opts.get('num_workers')
        self.assertEqual(result, max_cpus-1)
        exp_wmsg = ('Requested more CPUs ({}) than total available ({}). '
                    'Limiting number of used CPUs to {}.')
        self.assertEqual(str(w[0].message),
                         exp_wmsg.format(n_workers, max_cpus, max_cpus-1))

    def test_async(self):
        scheduler = 'async'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            opts = Parallel(scheduler=scheduler, num_workers=5)
        expected = opts._defaults_dict['num_workers']['default']
        self.assertEqual(opts.get('num_workers'), expected)
        exp_wmsg = 'Cannot set `num_workers` for the serial scheduler {!r}'
        six.assertRegex(self, str(w[0].message), exp_wmsg.format(scheduler))

    def test_distributed(self):
        scheduler = '192.168.0.128:8786'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            opts = Parallel(scheduler=scheduler, num_workers=5)
        expected = opts._defaults_dict['num_workers']['default']
        self.assertEqual(opts.get('num_workers'), expected)
        exp_wmsg = 'Attempting to set `num_workers` with the {!r} scheduler'
        six.assertRegex(self, str(w[0].message),
                        exp_wmsg.format('distributed'))


class Test_set_dask_scheduler(tests.IrisTest):
    # Check that the correct dask scheduler is chosen given the inputs.
    def setUp(self):
        self.patch('iris.config.Parallel._set_dask_options')

    def test_default(self):
        opts = Parallel()
        result = opts.get('dask_scheduler')
        expected = dask.threaded.get
        self.assertIs(result, expected)

    def test_threaded(self):
        opts = Parallel(scheduler='threaded')
        result = opts.get('dask_scheduler')
        expected = dask.threaded.get
        self.assertIs(result, expected)

    def test_multiprocessing(self):
        opts = Parallel(scheduler='multiprocessing')
        result = opts.get('dask_scheduler')
        expected = dask.multiprocessing.get
        self.assertIs(result, expected)

    def test_async(self):
        opts = Parallel(scheduler='async')
        result = opts.get('dask_scheduler')
        expected = dask.async.get_sync
        self.assertIs(result, expected)

    def test_distributed(self):
        scheduler = '192.168.0.128:8786'
        opts = Parallel(scheduler=scheduler)
        result = opts.get('dask_scheduler')
        self.assertEqual(result, scheduler)


if __name__ == '__main__':
    tests.main()
