# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# Import Iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import runpy

from .gallerytest_util import (
    add_gallery_to_path,
    show_replaced_by_check_graphic,
    fail_any_deprecation_warnings,
)


class TestLineplotWithLegend(tests.GraphicsTest):
    """Test the lineplot_with_legend gallery code."""

    def test_plot_lineplot_with_legend(self):
        with fail_any_deprecation_warnings():
            with add_gallery_to_path():
                with show_replaced_by_check_graphic(self):
                    runpy.run_module("plot_lineplot_with_legend")


if __name__ == "__main__":
    tests.main()
