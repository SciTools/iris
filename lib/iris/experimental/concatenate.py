# Copyright Iris Contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Automatic concatenation of multiple cubes over one or more existing dimensions.

.. warning::

    This functionality has now been moved to
    :meth:`iris.cube.CubeList.concatenate`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa


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
