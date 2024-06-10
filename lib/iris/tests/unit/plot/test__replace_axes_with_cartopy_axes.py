# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot.__replace_axes_with_cartopy_axes` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from iris.plot import _replace_axes_with_cartopy_axes


@tests.skip_plot
class Test_replace_axes_with_cartopy_axes(tests.IrisTest):
    def setUp(self):
        self.fig = plt.figure()

    def test_preserve_position(self):
        position = [0.17, 0.65, 0.2, 0.2]
        projection = ccrs.PlateCarree()

        plt.axes(position)
        _replace_axes_with_cartopy_axes(projection)
        result = plt.gca()

        # result should be the same as an axes created directly with the projection.
        expected = plt.axes(position, projection=projection)

        # get_position returns mpl.transforms.Bbox object, for which equality does
        # not appear to be implemented.  Compare the bounds (tuple) instead.
        self.assertEqual(expected.get_position().bounds, result.get_position().bounds)

    def test_ax_on_subfigure(self):
        subfig, _ = self.fig.subfigures(nrows=2)
        subfig.subplots()
        _replace_axes_with_cartopy_axes(ccrs.PlateCarree())
        result = plt.gca()
        self.assertIs(result.get_figure(), subfig)

    def tearDown(self):
        plt.close(self.fig)


if __name__ == "__main__":
    tests.main()
