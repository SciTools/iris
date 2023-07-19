# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Wrapper for animating iris cubes using iris or matplotlib plotting functions

Notes
-----
.. deprecated:: 3.4.0

``iris.experimental.animate.animate()`` has been moved to
:func:`iris.plot.animate`. This module will therefore be removed in a future
release.

"""


def animate(cube_iterator, plot_func, fig=None, **kwargs):
    """
    Animates the given cube iterator.

    Warnings
    --------
    This function is now **disabled**.

    The functionality has been moved to :func:`iris.plot.animate`.

    """
    msg = (
        "The function 'iris.experimental.animate.animate()' has been moved, "
        "and is now at 'iris.plot.animate()'.\n"
        "Please replace 'iris.experimental.animate.animate' with "
        "'iris.plot.animate'."
    )
    raise Exception(msg)
