# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from inspect import getmodule
import threading


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


def qualname(func):
    """Return the fully qualified function/method string name."""
    module = getmodule(func)
    return f"{module.__name__}.{func.__qualname__}"


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
        # Currently active lenient service.
        self.__dict__["active"] = None
        # Define ratified client/service relationships.
        client = "iris.analysis.maths.add"
        self.__dict__[client] = ("iris.common.metadata.CoordMetadata.__eq__",)

    def __call__(self, func):
        result = False
        service = qualname(func)
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
    def context(self, **kwargs):
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

            iris.LENIENT.example_future_flag does not exist and is
            provided only as an example.

        """
        # Save the current context
        current_state = self.__dict__.copy()
        # Temporarily update the state.
        for name, value in kwargs.items():
            setattr(self, name, value)
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
