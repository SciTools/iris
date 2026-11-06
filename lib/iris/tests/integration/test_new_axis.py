# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for :func:`iris.util.new_axis`."""

import iris
from iris.tests import _shared_utils
from iris.util import new_axis


class Test:
    @_shared_utils.skip_data
    def test_lazy_data(self):
        filename = _shared_utils.get_data_path(("PP", "globClim1", "theta.pp"))
        cube = iris.load_cube(filename)
        new_cube = new_axis(cube, "time")
        assert cube.has_lazy_data()
        assert new_cube.has_lazy_data()
        assert new_cube.shape == (1,) + cube.shape
