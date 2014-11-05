# (C) British Crown Copyright 2013 - 2014, Met Office
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
Automatic concatenation of multiple cubes over one or more existing dimensions.

.. warning::

    This functionality has now been moved to
    :meth:`iris.cube.CubeList.concatenate`.

"""

from __future__ import (absolute_import, division, print_function)


def concatenate(cubes):
    """
    Concatenate the provided cubes over common existing dimensions.

    .. warning::

        This function is now **disabled**.

        The functionality has been moved to
        :meth:`iris.cube.CubeList.concatenate`.

    """
    raise Exception(
        'The function "iris.experimental.concatenate.concatenate" has been '
        'moved, and is now a CubeList instance method.'
        '\nPlease replace '
        '"iris.experimental.concatenate.concatenate(<cubes>)" with '
        '"iris.cube.CubeList(<cubes>).concatenate()".')
