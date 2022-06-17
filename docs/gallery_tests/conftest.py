# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""Pytest fixtures for the gallery tests."""


import matplotlib.pyplot as plt
import pytest

import iris


@pytest.fixture
def image_setup_teardown():
    """
    Setup and teardown fixture.

    Ensures all figures are closed before and after test to prevent one test
    polluting another if it fails with a figure unclosed.

    """
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def iris_future_defaults():
    """
    Create a fixture which resets all the iris.FUTURE settings to the defaults,
    as otherwise changes made in one test can affect subsequent ones.

    """
    # Run with all default settings in iris.FUTURE.
    default_future_kwargs = iris.Future().__dict__.copy()
    for dead_option in iris.Future.deprecated_options:
        # Avoid a warning when setting these !
        del default_future_kwargs[dead_option]
    with iris.FUTURE.context(**default_future_kwargs):
        yield
