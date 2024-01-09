# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test set up of limited area map extents which bridge the date line."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import iris

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt

    from iris.plot import pcolormesh


@tests.skip_plot
@tests.skip_data
class TestExtent(tests.IrisTest):
    def test_dateline(self):
        dpath = tests.get_data_path(["PP", "nzgust.pp"])
        cube = iris.load_cube(dpath)
        pcolormesh(cube)
        # Ensure that the limited area expected for NZ is set.
        # This is set in longitudes with the datum set to the
        # International Date Line.
        self.assertTrue(
            -10 < plt.gca().get_xlim()[0] < -5 and 5 < plt.gca().get_xlim()[1] < 10
        )


if __name__ == "__main__":
    tests.main()
