# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Provides context managers which are fundamental to the ability
to run the gallery tests.

"""

import contextlib
import os.path
import sys
import warnings

import matplotlib.pyplot as plt

import iris
from iris._deprecation import IrisDeprecation
import iris.plot as iplt
import iris.quickplot as qplt

GALLERY_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "gallery_code"
)
GALLERY_DIRECTORIES = [
    os.path.join(GALLERY_DIRECTORY, the_dir)
    for the_dir in os.listdir(GALLERY_DIRECTORY)
]


@contextlib.contextmanager
def add_gallery_to_path():
    """
    Creates a context manager which can be used to add the iris gallery
    to the PYTHONPATH. The gallery entries are only importable throughout the lifetime
    of this context manager.

    """
    orig_sys_path = sys.path
    sys.path = sys.path[:]
    sys.path += GALLERY_DIRECTORIES
    yield
    sys.path = orig_sys_path


@contextlib.contextmanager
def show_replaced_by_check_graphic(test_case):
    """
    Creates a context manager which can be used to replace the functionality
    of matplotlib.pyplot.show with a function which calls the check_graphic
    method on the given test_case (iris.tests.IrisTest.check_graphic).

    """

    def replacement_show():
        # form a closure on test_case and tolerance
        test_case.check_graphic()

    orig_show = plt.show
    plt.show = iplt.show = qplt.show = replacement_show
    yield
    plt.show = iplt.show = qplt.show = orig_show


@contextlib.contextmanager
def fail_any_deprecation_warnings():
    """
    Create a context in which any deprecation warning will cause an error.

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
