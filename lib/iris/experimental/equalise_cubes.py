# (C) British Crown Copyright 2013 - 2015, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Experimental cube-adjusting functions to assist merge operations.

"""

from __future__ import (absolute_import, division, print_function)

import six

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
            key for key in common_keys
            if key in cube_keys
            and np.all(cube.attributes[key] == cubes[0].attributes[key])]

    # Remove all the other attributes.
    for cube in cubes:
        for key in list(cube.attributes.keys()):
            if key not in common_keys:
                del cube.attributes[key]
