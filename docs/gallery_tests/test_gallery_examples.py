# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Test all the gallery examples."""

import importlib

import matplotlib.pyplot as plt
import pytest

from iris.tests import _shared_utils

from .conftest import GALLERY_DIR


def gallery_examples():
    """Entry point for generator to yield all current gallery examples."""
    for example_file in GALLERY_DIR.glob("*/plot*.py"):
        yield example_file.stem


@pytest.mark.filterwarnings("error::iris.IrisDeprecation")
@pytest.mark.parametrize("example", gallery_examples())
def test_plot_example(
    example,
    image_setup_teardown,
    import_patches,
    iris_future_defaults,
    check_graphic_caller,
):
    """Test that all figures from example code match KGO."""
    if example in (
        "plot_TEC",
        "plot_orca_projection",
        "plot_projections_and_annotations",
    ):
        proj_9_8_message = _shared_utils.proj_9_8_incompatible_message()
        incompatible = proj_9_8_message != ""
        if incompatible:
            pytest.skip(proj_9_8_message)

    module = importlib.import_module(example)

    # Run example.
    module.main()
    # Loop through open figures and set each to be the current figure so check_graphic
    # will find it.
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        check_graphic_caller()
