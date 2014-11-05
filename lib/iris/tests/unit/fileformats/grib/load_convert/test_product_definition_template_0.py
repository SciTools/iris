# (C) British Crown Copyright 2014, Met Office
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
Test function
:func:`iris.fileformats.grib._load_convert.product_definition_template_0`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import deepcopy
import warnings

import mock

from iris.fileformats.grib._load_convert import product_definition_template_0


class Test(tests.IrisTest):
    def setUp(self):
        module = 'iris.fileformats.grib._load_convert'
        self.patch('warnings.warn')
        this = '{}.data_cutoff'.format(module)
        self.patch(this)
        this = '{}.forecast_period_coord'.format(module)
        self.forecast_period = mock.sentinel.forecast_period
        self.patch(this, return_value=self.forecast_period)
        this = '{}.validity_time_coord'.format(module)
        self.time = mock.sentinel.time
        self.patch(this, return_value=self.time)
        this = '{}.vertical_coords'.format(module)
        self.factory = mock.sentinel.factory
        func = lambda s, m: m['factories'].append(self.factory)
        self.patch(this, side_effect=func)
        self.metadata = {'factories': [], 'references': [],
                         'standard_name': None,
                         'long_name': None, 'units': None, 'attributes': {},
                         'cell_methods': [], 'dim_coords_and_dims': [],
                         'aux_coords_and_dims': []}

    def _check(self, request_warning):
        this = 'iris.fileformats.grib._load_convert.options'
        with mock.patch(this, warn_on_unsupported=request_warning):
            metadata = deepcopy(self.metadata)
            indicator = mock.sentinel.indicatorOfUnitOfTimeRange
            section = {'hoursAfterDataCutoff': None,
                       'minutesAfterDataCutoff': None,
                       'indicatorOfUnitOfTimeRange': indicator,
                       'forecastTime': mock.sentinel.forecastTime}
            forecast_reference_time = mock.sentinel.forecast_reference_time
            # The called being tested.
            product_definition_template_0(section, metadata,
                                          forecast_reference_time)
            expected = deepcopy(self.metadata)
            expected['factories'].append(self.factory)
            expected['aux_coords_and_dims'] = [(self.forecast_period, None),
                                               (self.time, None),
                                               (forecast_reference_time, None)]
            self.assertEqual(metadata, expected)
            if request_warning:
                warn_msgs = [mcall[1][0] for mcall in warnings.warn.mock_calls]
                expected_msgs = ['type of generating', 'background generating',
                                 'forecast generating']
                for emsg in expected_msgs:
                    matches = [wmsg for wmsg in warn_msgs if emsg in wmsg]
                    self.assertEqual(len(matches), 1)
                    warn_msgs.remove(matches[0])
            else:
                self.assertEqual(len(warnings.warn.mock_calls), 0)

    def test_pdt_no_warn(self):
        self._check(False)

    def test_pdt_warn(self):
        self._check(True)


if __name__ == '__main__':
    tests.main()
