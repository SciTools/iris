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
"""Unit tests for the `iris.options.Paralle` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import multiprocessing
import warnings

import dask
import distributed

from iris.options import Parallel
from iris.tests import mock


class Test_operation(tests.IrisTest):
    # Check that the options are passed through to 'real' code.
    # NOTE: tests that call the option class directly and as a contextmgr.

    def test_bad_name__contextmgr(self):
        # Check we can't do `with iris.options.parallel.context('foo'='bar')`.
        pass


class Test__set_dask_options(tests.IrisTest):
    # Check the correct dask options are set given the inputs.
    # NOTE: tests that check the correct dask options are set
    # (will require mock :scream:).
    # def setUp(self):
    #     patcher = mock.patch('dask.set_options')
    #     self.addCleanup(patcher.stop)
    #     self.mock_dask_opts = patcher.start()

    def setUp(self):
        self.mock_dask = dask
        self.mock_dask.threaded.get = mock.MagicMock()
        self.mock_dask.set_options = mock.MagicMock()
        self.mock_mul = multiprocessing
        self.mock_mul.pool.ThreadPool = mock.MagicMock()

    def test_default(self):
        pass

    def test_threaded(self):
        scheduler = 'threaded'
        Parallel(scheduler=scheduler)
        # self.mock_mul.pool.ThreadPool.assert_called_once_with(1)
        self.mock_dask.set_options.assert_called_once_with(get=self.mock_dask.threaded.get,
                                                       pool=self.mock_mul.pool.ThreadPool)

    def test_threaded_num_workers(self):
        pass

    # def test_async(self):
    #     scheduler = 'async'
    #     # dask_options = mock.Mock(spec=set_options)
    #     # dask_scheduler = mock.Mock(spec=async.get_sync)
    #     with mock.patch('dask.set_options') as mock_dask_opts:
    #         Parallel(scheduler=scheduler)
    #     # dask_options.assert_called_once_with(get=dask_scheduler)
    #     mock_dask_opts.assert_any_call()


class Test_set_schedulers(tests.IrisTest):
    # Check that the correct scheduler and dask scheduler are chosen given the
    # inputs.
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


class Test_set_dask_scheduler(tests.IrisTest):
    # Check that the correct scheduler and dask scheduler are chosen given the
    # inputs.
    def test_default(self):
        opts = Parallel()
        result = opts.get('dask_scheduler')
        self.assertIs(result, dask.threaded.get)

    def test_threaded(self):
        scheduler = 'threaded'
        opts = Parallel(scheduler=scheduler)
        result = opts.get('dask_scheduler')
        self.assertIs(result, dask.threaded.get)

    def test_multiprocessing(self):
        scheduler = 'multiprocessing'
        opts = Parallel(scheduler=scheduler)
        result = opts.get('dask_scheduler')
        self.assertIs(result, dask.multiprocessing.get)

    def test_async(self):
        scheduler = 'async'
        opts = Parallel(scheduler=scheduler)
        result = opts.get('dask_scheduler')
        self.assertIs(result, dask.async.get_sync)

    def test_distributed(self):
        scheduler = '192.168.0.128:8786'
        with mock.patch('distributed.Client.get') as mock_get:
            opts = Parallel(scheduler=scheduler)
        mock_get.assert_called_once_with(scheduler)


class Test_set_num_workers(tests.IrisTest):
    # Check that the correct `num_workers` are chosen given the inputs.
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

    def test_negative_workers(self):
        n_workers = -2
        exp_emsg = "Number of processes must be at least 1"
        with self.assertRaisesRegexp(ValueError, exp_emsg):
            Parallel(num_workers=n_workers)

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


if __name__ == '__main__':
    tests.main()
