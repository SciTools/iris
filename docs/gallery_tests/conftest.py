# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Pytest fixtures for the gallery tests.

"""

import sys

import matplotlib.pyplot as plt
import pytest

import iris

from .gallerytest_util import GALLERY_DIR

GALLERY_DIRECTORIES = [
    str(path) for path in GALLERY_DIR.iterdir() if path.is_dir()
]


@pytest.fixture
def add_gallery_to_path():
    """
    Creates a fixture which can be used to add the iris gallery to the
    PYTHONPATH. The gallery entries are only importable throughout the lifetime
    of the test.

    """
    orig_sys_path = sys.path
    sys.path = sys.path[:]
    sys.path += GALLERY_DIRECTORIES
    yield
    sys.path = orig_sys_path


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
