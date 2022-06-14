# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import importlib

import pytest

from .gallerytest_util import gallery_examples, show_replaced_by_check_graphic


@pytest.mark.parametrize("example_code", gallery_examples())
def test_plot_example(
    example_code, add_gallery_to_path, fail_any_deprecation_warnings
):
    module = importlib.import_module(example_code)
    with show_replaced_by_check_graphic(f"gallery_tests.test_{example_code}"):
        module.main()
