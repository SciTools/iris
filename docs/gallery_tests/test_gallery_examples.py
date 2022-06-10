# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import importlib

import pytest

# Import Iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from .gallerytest_util import (
    add_gallery_to_path,
    fail_any_deprecation_warnings,
    gallery_examples,
    show_replaced_by_check_graphic,
)


@pytest.mark.parametrize("example_code", gallery_examples())
def test_plot_example(example_code):
    with fail_any_deprecation_warnings():
        with add_gallery_to_path():
            module = importlib.import_module(example_code)
            print(module.__file__)
        with show_replaced_by_check_graphic(
            f"gallery_tests.test_{example_code}"
        ):
            module.main()


if __name__ == "__main__":
    tests.main()
