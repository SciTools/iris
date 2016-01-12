# (C) British Crown Copyright 2016, Met Office
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
:func:`iris.fileformats.grib._load_convert.ensemble_identifier`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import deepcopy
import warnings

from iris.coords import DimCoord
from iris.fileformats.grib._load_convert import ensemble_identifier
from iris.tests import mock


class Test(tests.IrisTest):
    def setUp(self):
        module = 'iris.fileformats.grib._load_convert'
        self.patch('warnings.warn')
        this = '{}.product_definition_template_0'.format(module)

    def _check(self, request_warning):
        section = {'perturbationNumber': 17}
        this = 'iris.fileformats.grib._load_convert.options'
        with mock.patch(this, warn_on_unsupported=request_warning):
            realization = ensemble_identifier(section)
            expected = DimCoord(section['perturbationNumber'],
                                standard_name='realization',
                                units='no_unit')

            if request_warning:
                warn_msgs = [mcall[1][0] for mcall in warnings.warn.mock_calls]
                expected_msgs = ['type of ensemble', 'number of forecasts']
                for emsg in expected_msgs:
                    matches = [wmsg for wmsg in warn_msgs if emsg in wmsg]
                    self.assertEqual(len(matches), 1)
                    warn_msgs.remove(matches[0])
            else:
                self.assertEqual(len(warnings.warn.mock_calls), 0)

    def test_ens_no_warn(self):
        self._check(False)

    def test_ens_warn(self):
        self._check(True)


if __name__ == '__main__':
    tests.main()
