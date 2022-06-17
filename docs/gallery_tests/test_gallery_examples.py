# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import importlib

import matplotlib.pyplot as plt
import pytest

from iris.tests import check_graphic

from .gallerytest_util import gallery_examples


@pytest.mark.filterwarnings("error::iris.IrisDeprecation")
@pytest.mark.parametrize("example_code", gallery_examples())
def test_plot_example(
    example_code,
    add_gallery_to_path,
    image_setup_teardown,
    iris_future_defaults,
    monkeypatch,
):
    def no_show():
        pass

    monkeypatch.setattr(plt, "show", no_show)

    module = importlib.import_module(example_code)

    module.main()
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        image_id = f"gallery_tests.test_{example_code}.{fig_num - 1}"
        check_graphic(image_id)
