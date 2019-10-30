# Copyright Iris Contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full licensing details.

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import Iris tests first so that some things can be initialised before
# importing anything else.

import iris.tests as tests

from . import extest_util

with extest_util.add_examples_to_path():
    import coriolis_plot


class TestCoriolisPlot(tests.GraphicsTest):
    """Test the Coriolis Plot example code."""
    def test_coriolis_plot(self):
        with extest_util.show_replaced_by_check_graphic(self):
            coriolis_plot.main()


if __name__ == '__main__':
    tests.main()
