# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Utilities for producing runtime deprecation messages.

"""

import warnings


class IrisDeprecation(UserWarning):
    """An Iris deprecation warning."""

    pass


def warn_deprecated(msg, stacklevel=2):
    """
    Issue an Iris deprecation warning.

    Calls :func:`warnings.warn', to emit the message 'msg' as a
    :class:`warnings.warning`, of the subclass :class:`IrisDeprecationWarning`.

    The 'stacklevel' keyword is passed through to warnings.warn.  However by
    default this is set to 2, which ensures that the identified code line is in
    the caller, rather than in this routine.
    See :mod:`warnings` module documentation.

    For example::

        >>> from iris._deprecation import warn_deprecated
        >>> def arrgh():
        ...    warn_deprecated('"arrgh" is deprecated since version 3.5.')
        ...    return 1
        ...
        >>> arrgh()
        __main__:2: IrisDeprecation: "arrgh" is deprecated since version 3.5.
        1
        >>> arrgh()
        1
        >>>

    """
    warnings.warn(msg, IrisDeprecation, stacklevel=stacklevel)


# A Mixin for a wrapper class that copies the docstring of the wrapped class
# into the wrapper.
# This is useful in producing wrapper classes that need to mimic the original
# but emit deprecation warnings when used.
class ClassWrapperSameDocstring(type):
    def __new__(metacls, classname, bases, class_dict):
        # Patch the subclass to duplicate the class docstring from the wrapped
        # class, and give it a special '__new__' that issues a deprecation
        # warning when creating an instance.
        parent_class = bases[0]

        # Copy the original class docstring.
        class_dict["__doc__"] = parent_class.__doc__

        # Return the result.
        return super().__new__(metacls, classname, bases, class_dict)
