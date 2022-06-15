# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import importlib

import pytest


@pytest.mark.filterwarnings("error::iris.IrisDeprecation")
def test_plot_example(
    add_gallery_to_path, iris_future_defaults, show_replaced_by_check_graphic
):
    module = importlib.import_module(show_replaced_by_check_graphic)
    module.main()
