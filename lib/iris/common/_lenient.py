# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from inspect import getmodule
from itertools import product
import threading


# TODO: allow *args to specify the ephemeral services that the client wishes to
#       use which are then unpacked in the LENIENT.context
def lenient_client(func):
    """
    Decorator that allows a client function/method to declare at runtime that
    it is executing and requires lenient behaviour from a prior registered
    lenient service function/method.

    Args:

    * func (callable):
        Callable function/method to be wrapped by the decorator.

    Returns:
        Closure wrapped function/method.

    """

    @wraps(func)
    def lenient_inner(*args, **kwargs):
        """
        Closure wrapper function to register the wrapped function/method as
        active at runtime before executing it.

        """
        with LENIENT.context(active=qualname(func)):
            result = func(*args, **kwargs)
        return result

    return lenient_inner


def qualname(func, cls=None):
    """
    Return the fully qualified function/method string name.

    Args:

    * func (callable/string):
        Callable function/method. The fully qualified name of the callable
        function/method is determined, otherwise the string name is used
        instead.

    Kwargs:

    * cls (class):
        If provided, the class is used to qualify the string name.

    .. note::
        Inherited methods will be qualified with the base class that
        defines the method.

    """
    if isinstance(func, str):
        result = func
        if cls is not None:
            result = "{}.{}.{}".format(cls.__module__, cls.__name__, func)
    else:
        module = getmodule(func)
        result = f"{module.__name__}.{func.__qualname__}"

    return result


def lenient_service(func):
    """
    Decorator that allows a function/method to declare that it supports lenient
    behaviour.

    Args:

    * func (callable):
        Callable function/method to be wrapped by the decorator.

    Returns:
        Closure wrapped function/method.

    """
    LENIENT.register(qualname(func))

    @wraps(func)
    def register_inner(*args, **kwargs):
        """
        Closure wrapper function to execute the lenient service
        function/method.

        """
        return func(*args, **kwargs)

    return register_inner


class Lenient(threading.local):
    def __init__(self):
        """
        A container for managing the run-time lenient options for
        pre-defined Iris functions and/or methods.

        To adjust the values simply update the relevant attribute.
        For example::

            iris.LENIENT.example_lenient_flag = False

        Or, equivalently::

            iris.LENIENT["example_lenient_flag"] = False

        Note that, the values of these options are thread-specific.

        """
        # Currently executing lenient client at runtime.
        self.__dict__["active"] = None

        # Define lenient services.
        # Require to be explicit here for subclass methods that inherit parent
        # class behaviour as a service for free.
        classes = [
            "AncillaryVariableMetadata",
            "CellMeasureMetadata",
            "CoordMetadata",
            "CubeMetadata",
        ]
        methods = [
            "__eq__",
            "combine",
            "difference",
        ]
        for cls, method in product(classes, methods):
            self.__dict__[f"iris.common.metadata.{cls}.{method}"] = True

        # Define lenient client/service relationships.
        # client = "iris.analysis.maths.add"
        # services = ("iris.common.metadata.CoordMetadata.__eq__",)
        # self.__dict__[client] = services

        # XXX: testing...
        client = "__main__.myfunc"
        services = ("iris.common.metadata.CoordMetadata.__eq__",)
        self.__dict__[client] = services

    def __call__(self, func, cls=None):
        """
        Determine whether it is valid for the function/method to provide a
        lenient service at runtime to the actively executing lenient client.

        Args:

        * func (callable/string):
            Callable function/method providing the lenient service. The fully
            qualified name of the callable function/method is determined,
            otherwise the string name is used instead.

        Kwargs:

        * cls (class):
            If provided, the class is used to qualify the string name.

        Returns:
            Boolean.

        """
        result = False
        service = qualname(func, cls=cls)
        if service in self and self.__dict__[service]:
            active = self.__dict__["active"]
            if active is not None and active in self:
                services = self.__dict__[active]
                if isinstance(services, str) or not isinstance(
                    services, Iterable
                ):
                    services = (services,)
                result = service in services
        return result

    def __contains__(self, name):
        return name in self.__dict__

    # TODO: Confirm whether this should be part of the API.
    def __getattr__(self, name):
        if name not in self.__dict__:
            cls = self.__class__.__name__
            emsg = f"Invalid {cls!r} option, got {name!r}."
            raise AttributeError(emsg)
        return self.__dict__[name]

    # TODO: Confirm whether this should be part of the API.
    def __getitem__(self, name):
        if name not in self.__dict__:
            cls = self.__class__.__name__
            emsg = f"Invalid {cls!r} option, got {name!r}."
            raise KeyError(emsg)
        return self.__dict__[name]

    def __repr__(self):
        cls = self.__class__.__name__
        width = len(cls) + 1
        kwargs = [
            "{}={!r}".format(name, self.__dict__[name])
            for name in sorted(self.__dict__.keys())
        ]
        joiner = ",\n{}".format(" " * width)
        return "{}({})".format(cls, joiner.join(kwargs))

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            cls = self.__class__.__name__
            emsg = f"Invalid {cls!r} option, got {name!r}."
            raise AttributeError(emsg)
        self.__dict__[name] = value

    def __setitem__(self, name, value):
        if name not in self.__dict__:
            cls = self.__class__.__name__
            emsg = f"Invalid {cls!r} option, got {name!r}."
            raise KeyError(emsg)
        self.__dict__[name] = value

    @contextmanager
    def context(self, *args, **kwargs):
        """
        Return a context manager which allows temporary modification of
        the lenient option state for the active thread.

        On entry to the context manager, all provided keyword arguments are
        applied. On exit from the context manager, the previous lenient option
        state is restored.

        For example::
            with iris.LENIENT.context(example_lenient_flag=False):
                # ... code that expects some non-lenient behaviour

        .. note::
            iris.LENIENT.example_lenient_flag does not exist and is
            provided only as an example.

        """
        # Save the original state.
        current_state = self.__dict__.copy()
        # Temporarily update the state with the kwargs first.
        for name, value in kwargs.items():
            setattr(self, name, value)
        # Temporarily update the client/services, if provided.
        if args:
            active = self.__dict__["active"]
            if active is None:
                active = "context"
                self.__dict__["active"] = active
            self.__dict__[active] = args
        try:
            yield
        finally:
            # Restore the original state.
            self.__dict__.clear()
            self.__dict__.update(current_state)

    def register(self, name):
        """
        Register the provided function/method as providing a lenient service.

        Args:

        * name (string):
            Fully qualified string name of the function/method.

        """
        self.__dict__[name] = True

    def unregister(self, name):
        """
        Unregister the provided function/method as providing a lenient service.

        Args:

        * name (string):
            Fully qualified string name of the function/method.

        """
        if name in self.__dict__:
            self.__dict__[name] = False
        else:
            cls = self.__class__.__name__
            emsg = f"Cannot unregister invalid {cls!r} service, got {name!r}."
            raise ValueError(emsg)


#: Instance that manages all Iris run-time lenient options.
LENIENT = Lenient()
