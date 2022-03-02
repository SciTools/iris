# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.plot._get_plot_objects` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import iris.cube

if tests.MPL_AVAILABLE:
    from iris.plot import _get_plot_objects


@tests.skip_plot
class Test_get_plot_objects(tests.IrisTest):
    def test_scalar(self):
        cube1 = iris.cube.Cube(1)
        cube2 = iris.cube.Cube(1)
        expected = (cube1, cube2, 1, 1, ())
        result = _get_plot_objects((cube1, cube2))
        self.assertTupleEqual(expected, result)


if __name__ == "__main__":
    tests.main()
