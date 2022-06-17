# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import importlib
import pathlib

import matplotlib.pyplot as plt
import pytest

from iris.tests import check_graphic


def gallery_examples():
    """
    Generator to yield all current gallery examples and their containing
    directories.

    """
    current_dir = pathlib.Path(__file__).resolve()
    gallery_dir = current_dir.parents[1] / "gallery_code"
    for example_file in gallery_dir.glob("*/plot*.py"):
        yield example_file.parent, example_file.stem


@pytest.mark.filterwarnings("error::iris.IrisDeprecation")
@pytest.mark.parametrize("example", gallery_examples(), ids=lambda arg: arg[1])
def test_plot_example(
    example,
    image_setup_teardown,
    iris_future_defaults,
    monkeypatch,
):
    """Test that all figures from example code match KGO."""

    example_dir, example_code = example

    # Replace pyplot.show with a function that does nothing, so all figures from the
    # example are still open after it runs.
    def no_show():
        pass

    monkeypatch.setattr(plt, "show", no_show)

    # Add example code to sys.path and import it.
    monkeypatch.syspath_prepend(example_dir)
    module = importlib.import_module(example_code)

    # Run example.
    module.main()
    # Loop through open figures and set each to be the current figure so check_graphic
    # will find it.
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        image_id = f"gallery_tests.test_{example_code}.{fig_num - 1}"
        check_graphic(image_id)
