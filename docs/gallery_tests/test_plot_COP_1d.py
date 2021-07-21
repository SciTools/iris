# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# Import Iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from .gallerytest_util import (
    add_gallery_to_path,
    fail_any_deprecation_warnings,
    show_replaced_by_check_graphic,
)


class TestCOP1DPlot(tests.GraphicsTest):
    """Test the COP_1d_plot gallery code."""

    def test_plot_COP_1d(self):
        with fail_any_deprecation_warnings():
            with add_gallery_to_path():
                import plot_COP_1d
            with show_replaced_by_check_graphic(self):
                plot_COP_1d.main()


if __name__ == "__main__":
    tests.main()
