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
:func:`iris.fileformats.grib._load_convert.product_definition_template_1`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import deepcopy
import warnings

import mock

from iris.coords import DimCoord
from iris.fileformats.grib._load_convert import product_definition_template_1


class Test(tests.IrisTest):
    def setUp(self):
        module = 'iris.fileformats.grib._load_convert'
        patch = []
        patch.append(mock.patch('warnings.warn'))
        this = '{}.product_definition_template_0'.format(module)
        self.cell_method = mock.sentinel.cell_method
        func = lambda s, m, f: m['cell_methods'].append(self.cell_method)
        patch.append(mock.patch(this, side_effect=func))
        self.metadata = {'factories': [], 'references': [],
                         'standard_name': None,
                         'long_name': None, 'units': None, 'attributes': {},
                         'cell_methods': [], 'dim_coords_and_dims': [],
                         'aux_coords_and_dims': []}
        for p in patch:
            p.start()
            self.addCleanup(p.stop)

    def _check(self, request_warning):
        this = 'iris.fileformats.grib._load_convert.options'
        with mock.patch(this, warn_on_unsupported=request_warning):
            metadata = deepcopy(self.metadata)
            perturbationNumber = 666
            section = {'perturbationNumber': perturbationNumber}
            forecast_reference_time = mock.sentinel.forecast_reference_time
            # The called being tested.
            product_definition_template_1(section, metadata,
                                          forecast_reference_time)
            expected = deepcopy(self.metadata)
            expected['cell_methods'].append(self.cell_method)
            realization = DimCoord(perturbationNumber,
                                   standard_name='realization',
                                   units='no_unit')
            expected['aux_coords_and_dims'].append((realization, None))
            self.assertEqual(metadata, expected)
            if request_warning:
                self.assertEqual(len(warnings.warn.mock_calls), 2)
                _, args, _ = zip(*warnings.warn.mock_calls)
                args = [arg[0] for arg in args]
                msgs = ['type of ensemble', 'number of forecasts']
                for msg in msgs:
                    self.assertTrue(any([msg in arg for arg in args]))
            else:
                self.assertEqual(len(warnings.warn.mock_calls), 0)

    def test_pdt_no_warn(self):
        self._check(False)

    def test_pdt_warn(self):
        self._check(True)


if __name__ == '__main__':
    tests.main()
