# (C) British Crown Copyright 2013, Met Office
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

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

import iris.fileformats.pp as pp


class Test__create_field_data(tests.IrisTest):
    def test_loaded_bytes(self):
        # Check that a field with LoadedArrayBytes in _data gets a suitable
        # call to _read_data_bytes.
        mock_loaded_bytes = mock.Mock(spec=pp.LoadedArrayBytes)
        field = mock.Mock(_data=mock_loaded_bytes)
        data_shape = mock.Mock()
        land_mask = mock.Mock()
        with mock.patch('iris.fileformats.pp._read_data_bytes') as read_bytes:
            pp._create_field_data(field, data_shape, land_mask)
            call = mock.call(mock_loaded_bytes.bytes, field.lbpack,
                             data_shape, mock_loaded_bytes.dtype, field.bmdi,
                             land_mask)
            self.assertEqual(read_bytes.call_args, call)
            self.assertEqual(read_bytes.call_count, 1)
        self.assertEqual(field._data, read_bytes.return_value)
        self.assertIsNone(field._data_manager)

    def test_deferred_bytes(self):
        # Check that a field with DeferredArrayBytes in _data gets a data
        # manager.
        mock_deferred_bytes = mock.Mock(spec=pp.DeferredArrayBytes)
        field = mock.Mock(_data=mock_deferred_bytes)
        data_shape = mock.Mock()
        land_mask = mock.Mock()
        proxy = pp.PPDataProxy(mock_deferred_bytes.fname,
                               mock_deferred_bytes.position,
                               mock_deferred_bytes.n_bytes,
                               field.lbpack, land_mask)
        _data = np.array(proxy)
        _data_manager = pp.DataManager(data_shape, mock_deferred_bytes.dtype,
                                       field.bmdi)
        pp._create_field_data(field, data_shape, land_mask)
        self.assertEqual(field._data, _data)
        self.assertEqual(field._data_manager, _data_manager)


if __name__ == "__main__":
    tests.main()
