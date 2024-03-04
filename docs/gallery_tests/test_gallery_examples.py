# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import importlib

import matplotlib.pyplot as plt
import pytest

from iris.tests import _RESULT_PATH
from iris.tests.graphics import check_graphic

from .conftest import GALLERY_DIR


def gallery_examples():
    """Generator to yield all current gallery examples."""

    for example_file in GALLERY_DIR.glob("*/plot*.py"):
        yield example_file.stem


@pytest.mark.filterwarnings("error::iris.IrisDeprecation")
@pytest.mark.parametrize("example", gallery_examples())
def test_plot_example(
    example,
    image_setup_teardown,
    import_patches,
    iris_future_defaults,
):
    """Test that all figures from example code match KGO."""

    module = importlib.import_module(example)

    # Run example.
    module.main()
    # Loop through open figures and set each to be the current figure so check_graphic
    # will find it.
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        image_id = f"gallery_tests.test_{example}.{fig_num - 1}"
        check_graphic(image_id, _RESULT_PATH)
