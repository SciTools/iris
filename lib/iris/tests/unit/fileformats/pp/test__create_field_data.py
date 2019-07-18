# (C) British Crown Copyright 2013 - 2019, Met Office
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
"""Unit tests for the `iris.fileformats.pp._create_field_data` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris.fileformats.pp as pp
from iris.tests import mock


class Test__create_field_data(tests.IrisTest):
    def test_loaded_bytes(self):
        # Check that a field with LoadedArrayBytes in core_data gets the
        # result of a suitable call to _data_bytes_to_shaped_array().
        mock_loaded_bytes = mock.Mock(spec=pp.LoadedArrayBytes)
        core_data = mock.MagicMock(return_value=mock_loaded_bytes)
        field = mock.Mock(core_data=core_data)
        data_shape = mock.Mock()
        land_mask = mock.Mock()
        with mock.patch('iris.fileformats.pp._data_bytes_to_shaped_array') as \
                convert_bytes:
            convert_bytes.return_value = mock.sentinel.array
            pp._create_field_data(field, data_shape, land_mask)

        self.assertIs(field.data, mock.sentinel.array)
        convert_bytes.assert_called_once_with(mock_loaded_bytes.bytes,
                                              field.lbpack,
                                              field.boundary_packing,
                                              data_shape,
                                              mock_loaded_bytes.dtype,
                                              field.bmdi, land_mask)

    def test_deferred_bytes(self):
        # Check that a field with deferred array bytes in core_data gets a
        # dask array.
        fname = mock.sentinel.fname
        position = mock.sentinel.position
        n_bytes = mock.sentinel.n_bytes
        newbyteorder = mock.Mock(return_value=mock.sentinel.dtype)
        dtype = mock.Mock(newbyteorder=newbyteorder)
        deferred_bytes = (fname, position, n_bytes, dtype)
        core_data = mock.MagicMock(return_value=deferred_bytes)
        field = mock.Mock(core_data=core_data)
        data_shape = (100, 120)
        proxy = mock.Mock(dtype=np.dtype('f4'), shape=data_shape,
                          spec=pp.PPDataProxy, ndim=len(data_shape))
        # We can't directly inspect the concrete data source underlying
        # the dask array, so instead we patch the proxy creation and check it's
        # being created and invoked correctly.
        with mock.patch('iris.fileformats.pp.PPDataProxy') as PPDataProxy:
            PPDataProxy.return_value = proxy
            pp._create_field_data(field, data_shape, land_mask_field=None)
        # The data should be assigned via field.data. As this is a mock object
        # we can check the attribute directly.
        self.assertEqual(field.data.shape, data_shape)
        self.assertEqual(field.data.dtype, np.dtype('f4'))
        # Is it making use of a correctly configured proxy?
        # NB. We know it's *using* the result of this call because
        # that's where the dtype came from above.
        PPDataProxy.assert_called_once_with((data_shape), dtype,
                                            fname, position,
                                            n_bytes,
                                            field.raw_lbpack,
                                            field.boundary_packing,
                                            field.bmdi)


if __name__ == "__main__":
    tests.main()
