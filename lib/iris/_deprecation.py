# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Utilities for producing runtime deprecation messages."""

from functools import wraps
import inspect
import warnings

from iris.warnings import IrisUserWarning


def explicit_copy_checker(f):
    """Check for explicitly set parameters in a function.

    This is intended to be used as a decorator for functions that take a
    variable number of parameters, to allow the function to determine which
    parameters were explicitly set by the caller.

    This can be helpful when wanting raise DeprecationWarning of function
    parameters, but only when they are explicitly set by the caller, and not
    when they are left at their default value.

    Parameters
    ----------
    f : function
        The function to be decorated. The function must have a signature that
        allows for variable parameters (e.g. ``*args`` and/or ``**kwargs``), and
        the parameters to be checked must be explicitly listed in the function
        signature (i.e. not just passed via ``**kwargs``).

    Returns
    -------
    function
        The decorated function, which will have an additional keyword argument
        ``explicit_params`` added to its signature. This argument will be a set
        of the names of the parameters that were explicitly set by the caller when
        calling the function.

    Examples
    --------
    The following example shows how to use the ``explicit_copy_checker`` decorator to
    check for explicitly set parameters in a function, and raise a DeprecationWarning
    if a deprecated parameter is explicitly set by the caller.

    >>> from iris._deprecation import explicit_copy_checker, IrisDeprecation
    >>> @explicit_copy_checker
    ... def my_function(a, b=1):
    ...     print(f"a={a}, b={b}")
    ...     if "b" in kwargs["explicit_params"]:
    ...         warnings.warn("Parameter 'b' is deprecated.", IrisDeprecation)
    >>> my_function(1)  # No warning, 'b' is not explicitly set
    >>> my_function(1, b=3)  # Warning, 'b' is explicitly set

    """
    varnames = inspect.getfullargspec(f)[0]

    @wraps(f)
    def wrapper(*a, **kw):
        explicit_params = set(list(varnames[: len(a)]) + list(kw.keys()))
        if "copy" in explicit_params:
            if kw["copy"] is False:
                msg = (
                    "Pandas v3 behaviour defaults to copy=True. The `copy`"
                    f" parameter in `{f.__name__}` is deprecated and"
                    "will be removed in a future release."
                )
                warnings.warn(msg, category=IrisUserWarning)
            else:
                msg = (
                    f"The `copy` parameter in `{f.__name__}` is deprecated and"
                    " will be removed in a future release. The function will"
                    " always make a copy of the data array, to ensure that the"
                    " returned Cubes are independent of the input pandas data."
                )
                warn_deprecated(msg)
        else:
            return f(*a, **kw)

    return wrapper


class IrisDeprecation(UserWarning):
    """An Iris deprecation warning.

    Note this subclasses UserWarning for backwards compatibility with Iris'
    original deprecation warnings. Should subclass DeprecationWarning at the
    next major release.
    """

    pass


def warn_deprecated(msg, stacklevel=2):
    """Issue an Iris deprecation warning.

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
    warnings.warn(msg, category=IrisDeprecation, stacklevel=stacklevel)


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
