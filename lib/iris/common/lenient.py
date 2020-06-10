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


__all__ = [
    "lenient_client",
    "lenient_service",
    "qualname",
    "LENIENT",
]


def lenient_client(*dargs, services=None):
    """
    Decorator that allows a client function/method to declare at runtime that
    it is executing and requires lenient behaviour from a prior registered
    lenient service function/method.

    This decorator supports being called with no arguments e.g.,

        @lenient_client()
        def func():
            pass

    This is equivalent to using it as a simple naked decorator e.g.,

        @lenient_client
        def func()
            pass

    Alternatively, this decorator supports the lenient client explicitly
    declaring the lenient services that it wishes to use e.g.,

        @lenient_client(services=(service1, service2, ...)
        def func():
            pass

    Args:

    * dargs (tuple of callable):
        A tuple containing the callable lenient client function/method to be
        wrapped by the decorator. This is automatically populated by Python
        through the decorator interface. No argument requires to be manually
        provided.

    Kwargs:

    * services (callable or str or iterable of callable/str)
        Zero or more function/methods, or equivalent fully qualified string names, of
        lenient service function/methods.

    Returns:
        Closure wrapped function/method.

    """
    ndargs = len(dargs)

    if ndargs:
        assert ndargs == 1, f"Invalid lenient client arguments, expecting 1 got {ndargs}."
        assert callable(dargs[0]), f"Invalid lenient client argument, expecting callable."

    assert not(ndargs and services), f"Invalid lenient client arguments."

    if ndargs:
        # The decorator has been used as a simple naked decorator.
        func = dargs[0]

        @wraps(func)
        def lenient_inner(*args, **kwargs):
            """
            Closure wrapper function to register the wrapped function/method
            as active at runtime before executing it.

            """
            with LENIENT.context(active=qualname(func)):
                result = func(*args, **kwargs)
            return result

        result = lenient_inner
    else:
        # The decorator has been called with None, zero or more explicit lenient services.
        if services is None:
            services = ()

        if isinstance(services, str) or not isinstance(services, Iterable):
            services = (services,)

        def lenient_outer(func):

            @wraps(func)
            def lenient_inner(*args, **kwargs):
                """
                Closure wrapper function to register the wrapped function/method
                as active at runtime before executing it.

                """
                with LENIENT.context(*services, active=qualname(func)):
                    result = func(*args, **kwargs)
                return result

            return lenient_inner

        result = lenient_outer

    return result


def lenient_service(func):
    """
    Decorator that allows a function/method to declare that it supports lenient
    behaviour.

    Registration is at Python interpreter parse time.

    Args:

    * func (callable):
        Callable function/method to be wrapped by the decorator.

    Returns:
        Closure wrapped function/method.

    """
    LENIENT.register(func)

    @wraps(func)
    def register_inner(*args, **kwargs):
        """
        Closure wrapper function to execute the lenient service
        function/method.

        """
        return func(*args, **kwargs)

    return register_inner


def qualname(func):
    """
    Return the fully qualified function/method string name.

    Args:

    * func (callable):
        Callable function/method. Non-callable arguments are simply
        passed through.

    .. note::
        Inherited methods will be qualified with the base class that
        defines the method.

    """
    result = func
    if callable(func):
        module = getmodule(func)
        result = f"{module.__name__}.{func.__qualname__}"

    return result


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

    def __call__(self, func):
        """
        Determine whether it is valid for the function/method to provide a
        lenient service at runtime to the actively executing lenient client.

        Args:

        * func (callable or str):
            A function/method or fully qualified string name of the function/method.

        Returns:
            Boolean.

        """
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
            self.__dict__[active] = tuple([qualname(arg) for arg in args])
        try:
            yield
        finally:
            # Restore the original state.
            self.__dict__.clear()
            self.__dict__.update(current_state)

    def register(self, func):
        """
        Register the provided function/method as providing a lenient service.

        Args:

        * func (callable or str):
            A function/method or fully qualified string name of the function/method.

        """
        func = qualname(func)
        self.__dict__[func] = True

    def unregister(self, func):
        """
        Unregister the provided function/method as providing a lenient service.

        Args:

        * func (callable or str):
            A function/method or fully qualified string name of the function/method.

        """
        func = qualname(func)
        if func in self.__dict__:
            self.__dict__[func] = False
        else:
            cls = self.__class__.__name__
            emsg = f"Cannot unregister invalid {cls!r} service, got {func!r}."
            raise ValueError(emsg)


#: Instance that manages all Iris run-time lenient options.
LENIENT = Lenient()
