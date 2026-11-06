# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test set up of limited area map extents which bridge the date line."""

import iris
from iris.tests import _shared_utils

# Run tests in no graphics mode if matplotlib is not available.
if _shared_utils.MPL_AVAILABLE:
    import matplotlib.pyplot as plt

    from iris.plot import pcolormesh


@_shared_utils.skip_plot
@_shared_utils.skip_data
class TestExtent:
    def test_dateline(self):
        dpath = _shared_utils.get_data_path(["PP", "nzgust.pp"])
        cube = iris.load_cube(dpath)
        pcolormesh(cube)
        # Ensure that the limited area expected for NZ is set.
        # This is set in longitudes with the datum set to the
        # International Date Line.
        assert -10 < plt.gca().get_xlim()[0] < -5
        assert 5 < plt.gca().get_xlim()[1] < 10
