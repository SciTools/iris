# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Provides context manager and utility functions which are fundamental to the ability
to run the gallery tests.

"""


import pathlib


def gallery_path():
    """Return path to gallery code."""
    current_dir = pathlib.Path(__file__).resolve()
    return current_dir.parents[1] / "gallery_code"


def gallery_examples():
    """Generator to yield all current gallery examples."""
    for example_file in gallery_path().glob("*/plot*.py"):
        yield example_file.stem
