# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Provides context manager and utility functions which are fundamental to the ability
to run the gallery tests.

"""

import collections
import contextlib
import pathlib

import matplotlib.pyplot as plt

import iris.plot as iplt
import iris.quickplot as qplt
from iris.tests import check_graphic


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


def gallery_path():
    """Return path to gallery code."""
    current_dir = pathlib.Path(__file__).resolve()
    return current_dir.parents[1] / "gallery_code"


def gallery_examples():
    """Generator to yield all current gallery examples."""
    for example_file in gallery_path().glob("*/plot*.py"):
        yield example_file.stem
