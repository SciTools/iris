# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Experimental cube-adjusting functions to assist merge operations.

"""


def equalise_attributes(cubes):
    """
    Delete cube attributes that are not identical over all cubes in a group.

    .. warning::

        This function is now **disabled**.

        The functionality has been moved to
        :func:`iris.util.equalise_attributes`.

    """
    old = "iris.experimental.equalise_cubes.equalise_attributes"
    new = "iris.util.equalise_attributes"
    emsg = (
        f'The function "{old}" has been moved.\n'
        f'Please replace "{old}(<cubes>)" with "{new}(<cubes>)".'
    )
    raise Exception(emsg)
