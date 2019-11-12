# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Experimental cube-adjusting functions to assist merge operations.

"""

import numpy as np


def equalise_attributes(cubes):
    """
    Delete cube attributes that are not identical over all cubes in a group.

    This function simply deletes any attributes which are not the same for
    all the given cubes.  The cubes will then have identical attributes.  The
    given cubes are modified in-place.

    Args:

    * cubes (iterable of :class:`iris.cube.Cube`):
        A collection of cubes to compare and adjust.

    """
    # Work out which attributes are identical across all the cubes.
    common_keys = list(cubes[0].attributes.keys())
    for cube in cubes[1:]:
        cube_keys = list(cube.attributes.keys())
        common_keys = [
            key
            for key in common_keys
            if (
                key in cube_keys
                and np.all(cube.attributes[key] == cubes[0].attributes[key])
            )
        ]

    # Remove all the other attributes.
    for cube in cubes:
        for key in list(cube.attributes.keys()):
            if key not in common_keys:
                del cube.attributes[key]
