# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Provides context manager and generator which are fundamental to the ability
to run the gallery tests.

"""

import pathlib

CURRENT_DIR = pathlib.Path(__file__).resolve()
GALLERY_DIR = CURRENT_DIR.parents[1] / "gallery_code"


def gallery_examples():
    """Generator to yield all current gallery examples."""
    for example_file in GALLERY_DIR.glob("*/plot*.py"):
        yield example_file.stem
