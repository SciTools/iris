# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# Import Iris tests first so that some things can be initialised before
# importing anything else.

import iris.tests as tests

from . import gallerytest_util

with gallerytest_util.add_gallery_to_path():
    import plot_coriolis


class TestCoriolisPlot(tests.GraphicsTest):
    """Test the Coriolis Plot gallery code."""

    def test_plot_coriolis(self):
        with gallerytest_util.show_replaced_by_check_graphic(self):
            plot_coriolis.main()


if __name__ == "__main__":
    tests.main()
