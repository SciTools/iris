# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp._create_field_data` function."""

import numpy as np

import iris.fileformats.pp as pp


class Test__create_field_data:
    def test_loaded_bytes(self, mocker):
        # Check that a field with LoadedArrayBytes in core_data gets the
        # result of a suitable call to _data_bytes_to_shaped_array().
        mock_loaded_bytes = mocker.Mock(spec=pp.LoadedArrayBytes)
        core_data = mocker.MagicMock(return_value=mock_loaded_bytes)
        field = mocker.Mock(core_data=core_data)
        data_shape = mocker.Mock()
        land_mask = mocker.Mock()
        convert_bytes = mocker.patch("iris.fileformats.pp._data_bytes_to_shaped_array")
        convert_bytes.return_value = mocker.sentinel.array
        pp._create_field_data(field, data_shape, land_mask)

        assert field.data is mocker.sentinel.array
        convert_bytes.assert_called_once_with(
            mock_loaded_bytes.bytes,
            field.lbpack,
            field.boundary_packing,
            data_shape,
            mock_loaded_bytes.dtype,
            field.bmdi,
            land_mask,
        )

    def test_deferred_bytes(self, mocker):
        # Check that a field with deferred array bytes in core_data gets a
        # dask array.
        fname = mocker.sentinel.fname
        position = mocker.sentinel.position
        n_bytes = mocker.sentinel.n_bytes
        newbyteorder = mocker.Mock(return_value=mocker.sentinel.dtype)
        dtype = mocker.Mock(newbyteorder=newbyteorder)
        deferred_bytes = (fname, position, n_bytes, dtype)
        core_data = mocker.MagicMock(return_value=deferred_bytes)
        field = mocker.Mock(core_data=core_data)
        data_shape = (100, 120)
        proxy = mocker.Mock(
            dtype=np.dtype("f4"),
            dask_meta=np.empty((0,) * len(data_shape), dtype=np.dtype("f4")),
            shape=data_shape,
            spec=pp.PPDataProxy,
            ndim=len(data_shape),
        )
        # We can't directly inspect the concrete data source underlying
        # the dask array, so instead we patch the proxy creation and check it's
        # being created and invoked correctly.
        PPDataProxy = mocker.patch("iris.fileformats.pp.PPDataProxy")
        PPDataProxy.return_value = proxy
        pp._create_field_data(field, data_shape, land_mask_field=None)
        # The data should be assigned via field.data. As this is a mock object
        # we can check the attribute directly.
        assert field.data.shape == data_shape
        assert field.data.dtype == np.dtype("f4")
        # Is it making use of a correctly configured proxy?
        # NB. We know it's *using* the result of this call because
        # that's where the dtype came from above.
        PPDataProxy.assert_called_once_with(
            (data_shape),
            dtype,
            fname,
            position,
            n_bytes,
            field.raw_lbpack,
            field.boundary_packing,
            field.bmdi,
        )
