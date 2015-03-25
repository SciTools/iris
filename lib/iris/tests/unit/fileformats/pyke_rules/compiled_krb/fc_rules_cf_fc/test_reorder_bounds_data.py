# (C) British Crown Copyright 2014 - 2015, Met Office
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
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.reorder_bounds_data`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np
import mock

from iris.aux_factory import LazyArray
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    reorder_bounds_data


class Test(tests.IrisTest):
    def test_fastest_varying(self):
        bounds_data = np.arange(24).reshape(2, 3, 4)
        cf_bounds_var = mock.Mock(dimensions=('foo', 'bar', 'nv'),
                                  cf_name='wibble_bnds')
        cf_coord_var = mock.Mock(dimensions=('foo', 'bar'))

        res = reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)
        # Vertex dimension (nv) is already at the end.
        self.assertArrayEqual(res, bounds_data)

    def test_slowest_varying(self):
        bounds_data = np.arange(24).reshape(4, 2, 3)
        cf_bounds_var = mock.Mock(dimensions=('nv', 'foo', 'bar'))
        cf_coord_var = mock.Mock(dimensions=('foo', 'bar'))

        res = reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)
        # Move zeroth dimension (nv) to the end.
        expected = np.rollaxis(bounds_data, 0, bounds_data.ndim)
        self.assertArrayEqual(res, expected)

    def test_slowest_varying_lazy(self):
        bounds_data = np.arange(24).reshape(4, 2, 3)
        func = lambda: bounds_data
        lazy_bounds_data = LazyArray(bounds_data.shape, func,
                                     bounds_data.dtype)
        cf_bounds_var = mock.Mock(dimensions=('nv', 'foo', 'bar'))
        cf_coord_var = mock.Mock(dimensions=('foo', 'bar'))

        res = reorder_bounds_data(lazy_bounds_data, cf_bounds_var,
                                  cf_coord_var)
        # Move zeroth dimension (nv) to the end.
        expected = np.rollaxis(bounds_data, 0, bounds_data.ndim)
        self.assertArrayEqual(res, expected)

    def test_different_dim_names(self):
        bounds_data = np.arange(24).reshape(2, 3, 4)
        cf_bounds_var = mock.Mock(dimensions=('foo', 'bar', 'nv'),
                                  cf_name='wibble_bnds')
        cf_coord_var = mock.Mock(dimensions=('x', 'y'), cf_name='wibble')
        with self.assertRaisesRegexp(ValueError, 'dimension names'):
            reorder_bounds_data(bounds_data, cf_bounds_var, cf_coord_var)


if __name__ == '__main__':
    tests.main()
