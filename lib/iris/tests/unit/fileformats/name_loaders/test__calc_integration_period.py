# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :func:`iris.fileformats.name_loaders.__calc_integration_period`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import datetime

from iris.fileformats.name_loaders import _calc_integration_period


class Test(tests.IrisTest):
    def test_30_min_av(self):
        time_avgs = ["             30min average"]
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (30 * 60))]
        self.assertEqual(result, expected)

    def test_30_min_av_rspace(self):
        time_avgs = ["             30min average  "]
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (30 * 60))]
        self.assertEqual(result, expected)

    def test_30_min_av_lstrip(self):
        time_avgs = ["             30min average".lstrip()]
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (30 * 60))]
        self.assertEqual(result, expected)

    def test_3_hour_av(self):
        time_avgs = ["          3hr 0min average"]
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (3 * 60 * 60))]
        self.assertEqual(result, expected)

    def test_3_hour_int(self):
        time_avgs = ["         3hr 0min integral"]
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (3 * 60 * 60))]
        self.assertEqual(result, expected)

    def test_12_hour_av(self):
        time_avgs = ["         12hr 0min average"]
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (12 * 60 * 60))]
        self.assertEqual(result, expected)

    def test_5_day_av(self):
        time_avgs = ["    5day 0hr 0min integral"]
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (5 * 24 * 60 * 60))]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    tests.main()
