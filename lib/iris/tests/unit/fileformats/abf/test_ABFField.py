# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.abf.ABFField` class."""
from iris.fileformats.abf import ABFField


class Test_data:
    def test_single_read(self, mocker):
        path = "0000000000000000jan00000"
        field = ABFField(path)

        fromfile = mocker.patch("iris.fileformats.abf.np.fromfile")
        getattr = mocker.patch(
            "iris.fileformats.abf.ABFField.__getattr__", wraps=field.__getattr__
        )
        read = mocker.patch("iris.fileformats.abf.ABFField._read", wraps=field._read)

        # do the access
        field.data

        fromfile.assert_called_once_with(path, dtype=">u1")
        assert getattr.call_count == 1
        assert read.call_count == 1
