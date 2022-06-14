# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Provides context managers which are fundamental to the ability
to run the gallery tests.

"""

import sys
import warnings

import pytest

import iris
from iris._deprecation import IrisDeprecation

from .gallerytest_util import gallery_path

GALLERY_DIRECTORY = gallery_path()
GALLERY_DIRECTORIES = [
    str(path) for path in GALLERY_DIRECTORY.iterdir() if path.is_dir()
]


@pytest.fixture
def add_gallery_to_path():
    """
    Creates a fixture which can be used to add the iris gallery to the
    PYTHONPATH. The gallery entries are only importable throughout the lifetime
    of this context manager.

    """
    orig_sys_path = sys.path
    sys.path = sys.path[:]
    sys.path += GALLERY_DIRECTORIES
    yield
    sys.path = orig_sys_path


@pytest.fixture
def fail_any_deprecation_warnings():
    """
    Create a fixture in which any deprecation warning will cause an error.

    The context also resets all the iris.FUTURE settings to the defaults, as
    otherwise changes made in one test can affect subsequent ones.

    """
    with warnings.catch_warnings():
        # Detect and error all and any Iris deprecation warnings.
        warnings.simplefilter("error", IrisDeprecation)
        # Run with all default settings in iris.FUTURE.
        default_future_kwargs = iris.Future().__dict__.copy()
        for dead_option in iris.Future.deprecated_options:
            # Avoid a warning when setting these !
            del default_future_kwargs[dead_option]
        with iris.FUTURE.context(**default_future_kwargs):
            yield
