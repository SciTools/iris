# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Provides context managers which are fundamental to the ability
to run the gallery tests.

"""

import collections
import contextlib
import os.path
import pathlib
import sys
import warnings

import matplotlib.pyplot as plt

import iris
from iris._deprecation import IrisDeprecation
import iris.plot as iplt
import iris.quickplot as qplt
from iris.tests import check_graphic

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
def show_replaced_by_check_graphic(test_id):
    """
    Creates a context manager which can be used to replace the functionality
    of matplotlib.pyplot.show with a function which calls the check_graphic
    function (iris.tests.check_graphic).

    """
    assertion_counts = collections.defaultdict(int)

    def replacement_show():
        # form a closure on test_case and tolerance
        unique_id = f"{test_id}.{assertion_counts[test_id]}"
        assertion_counts[test_id] += 1
        check_graphic(unique_id)

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


def gallery_examples():
    """Generator to yield all current gallery examples."""
    current_dir = pathlib.Path(__file__).resolve()
    code_dir = current_dir.parents[1] / "gallery_code"
    for example_file in code_dir.glob("*/plot*.py"):
        yield example_file.stem
