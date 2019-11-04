# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from six.moves import (filter, input, map, range, zip)  # noqa

# Import Iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from .extest_util import (add_examples_to_path,
                          show_replaced_by_check_graphic,
                          fail_any_deprecation_warnings)


class TestCustomAggregation(tests.GraphicsTest):
    """Test the custom aggregation example code."""
    def test_custom_aggregation(self):
        with fail_any_deprecation_warnings():
            with add_examples_to_path():
                import custom_aggregation
            with show_replaced_by_check_graphic(self):
                custom_aggregation.main()


if __name__ == '__main__':
    tests.main()
