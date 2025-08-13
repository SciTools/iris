# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import numpy as np
import pytest

import iris
import iris.fileformats.abf
from iris.tests import _shared_utils


@_shared_utils.skip_data
class TestAbfLoad:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.path = _shared_utils.get_data_path(("abf", "AVHRRBUVI01.1985apra.abf"))

    def test_load(self, request):
        cubes = iris.load(self.path)
        # On a 32-bit platform the time coordinate will have 32-bit integers.
        # We force them to 64-bit to ensure consistent test results.
        time_coord = cubes[0].coord("time")
        time_coord.points = np.array(time_coord.points, dtype=np.int64)
        time_coord.bounds = np.array(time_coord.bounds, dtype=np.int64)
        # Normalise the different array orders returned by version 1.6
        # and 1.7 of NumPy.
        cubes[0].data = cubes[0].data.copy(order="C")
        _shared_utils.assert_CML(request, cubes, ("abf", "load.cml"))

    def test_fill_value(self):
        field = iris.fileformats.abf.ABFField(self.path)
        # Make sure the fill value is appropriate. It must avoid the
        # data range (0 to 100 inclusive) but still fit within the dtype
        # range (0 to 255 inclusive).
        assert field.data.fill_value > 100
        assert field.data.fill_value < 256
