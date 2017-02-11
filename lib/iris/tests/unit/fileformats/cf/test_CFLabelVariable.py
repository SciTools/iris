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
Unit tests for the `iris.fileformats.cf.CFLabelVariable` class.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.fileformats.cf import CFLabelVariable, CFDataVariable
from iris.tests import mock


class Test(tests.IrisTest):
    def test_bytes(self):
        # Python 3 and the netCDF4 library gives us bytes for character data.
        var = mock.MagicMock(name='variable', dimensions=['string_dim'],
                             ndim=1)
        var.__getitem__.return_value = np.array([char.encode('utf-8')
                                                 for char in 'wibble'])
 
        data_var = mock.MagicMock(spec=CFDataVariable, name='data_var',
                                  dimensions=[])

        lv = CFLabelVariable('foo', var)

        result = lv.cf_label_data(data_var)
        self.assertEqual(result.shape, (1, ))
        self.assertEqual('wibble', result[0])


if __name__ == '__main__':
    tests.main()
