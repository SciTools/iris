# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides the infrastructure to support lenient client/service behaviour.

"""

from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import getmodule
import threading

__all__ = [
    "LENIENT",
    "Lenient",
]


#: Default _Lenient services global activation state.
_LENIENT_ENABLE_DEFAULT = True

#: Default Lenient maths feature state.
_LENIENT_MATHS_DEFAULT = True

#: Protected _Lenient internal non-client, non-service keys.
_LENIENT_PROTECTED = ("active", "enable")


def _lenient_client(*dargs, services=None):
    """Decorator that allows a client function/method to declare at runtime that
    it is executing and requires lenient behaviour from a prior registered
    lenient service function/method.

    This decorator supports being called with no arguments e.g.,

        @_lenient_client()
        def func():
            pass

    This is equivalent to using it as a simple naked decorator e.g.,

        @_lenient_client
        def func()
            pass

    Alternatively, this decorator supports the lenient client explicitly
    declaring the lenient services that it wishes to use e.g.,

        @_lenient_client(services=(service1, service2, ...)
        def func():
            pass

    Parameters
    ----------
    dargs : tuple of callable
        A tuple containing the callable lenient client function/method to be
        wrapped by the decorator. This is automatically populated by Python
        through the decorator interface. No argument requires to be manually
        provided.
    services : callable or str or iterable of callable/str, optional, default=None
        Zero or more function/methods, or equivalent fully qualified string names, of
        lenient service function/methods.

    Returns
    -------
    Closure wrapped function/method.

    """
    ndargs = len(dargs)

    if ndargs:
        assert (
            ndargs == 1
        ), f"Invalid lenient client arguments, expecting 1 got {ndargs}."
        assert callable(
            dargs[0]
        ), "Invalid lenient client argument, expecting a callable."

    assert not (
        ndargs and services
    ), "Invalid lenient client, got both arguments and keyword arguments."

    if ndargs:
        # The decorator has been used as a simple naked decorator.
        (func,) = dargs

        @wraps(func)
        def lenient_client_inner_naked(*args, **kwargs):
            """Closure wrapper function to register the wrapped function/method
            as active at runtime before executing it.

            """
            with _LENIENT.context(active=_qualname(func)):
                result = func(*args, **kwargs)
            return result

        result = lenient_client_inner_naked
    else:
        # The decorator has been called with None, zero or more explicit lenient services.
        if services is None:
            services = ()

        if isinstance(services, str) or not isinstance(services, Iterable):
            services = (services,)

        def lenient_client_outer(func):
            @wraps(func)
            def lenient_client_inner(*args, **kwargs):
                """Closure wrapper function to register the wrapped function/method
                as active at runtime before executing it.

                """
                with _LENIENT.context(*services, active=_qualname(func)):
                    result = func(*args, **kwargs)
                return result

            return lenient_client_inner

        result = lenient_client_outer

    return result


def _lenient_service(*dargs):
    """Decorator that allows a function/method to declare that it supports lenient
    behaviour as a service.

    Registration is at Python interpreter parse time.

    The decorator supports being called with no arguments e.g.,

        @_lenient_service()
        def func():
            pass

    This is equivalent to using it as a simple naked decorator e.g.,

        @_lenient_service
        def func():
            pass

    Parameters
    ----------
    dargs : tuple of callable
        A tuple containing the callable lenient service function/method to be
        wrapped by the decorator. This is automatically populated by Python
        through the decorator interface. No argument requires to be manually
        provided.

    Returns
    -------
    Closure wrapped function/method.

    """
    ndargs = len(dargs)

    if ndargs:
        assert (
            ndargs == 1
        ), f"Invalid lenient service arguments, expecting 1 got {ndargs}."
        assert callable(
            dargs[0]
        ), "Invalid lenient service argument, expecting a callable."

    if ndargs:
        # The decorator has been used as a simple naked decorator.
        # Thus the (single) argument is a function to be wrapped.
        # We just register the argument function as a lenient service, and
        # return it unchanged
        (func,) = dargs

        _LENIENT.register_service(func)

        # This decorator registers 'func': the func itself is unchanged.
        result = func

    else:
        # The decorator has been called with no arguments.
        # Return a decorator, to apply to 'func' immediately following.
        def lenient_service_outer(func):
            _LENIENT.register_service(func)

            # Decorator registers 'func', but func itself is unchanged.
            return func

        result = lenient_service_outer

    return result


def _qualname(func):
    """Return the fully qualified function/method string name.

    Parameters
    ----------
    func : callable
        Callable function/method. Non-callable arguments are simply
        passed through.

    Notes
    -----
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
    def __init__(self, **kwargs):
        """A container for managing the run-time lenient features and options.

        Parameters
        ----------
        **kwargs : dict
            Mapping of lenient key/value options to enable/disable. Note that,
            only the lenient "maths" options is available, which controls
            lenient/strict cube arithmetic.

        Examples
        --------
        ::

            Lenient(maths=False)

        Note that, the values of these options are thread-specific.

        """
        # Configure the initial default lenient state.
        self._init()

        if not kwargs:
            # If not specified, set the default behaviour of the maths lenient feature.
            kwargs = dict(maths=_LENIENT_MATHS_DEFAULT)

        # Configure the provided (or default) lenient features.
        for feature, state in kwargs.items():
            self[feature] = state

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        if key not in self.__dict__:
            cls = self.__class__.__name__
            emsg = f"Invalid {cls!r} option, got {key!r}."
            raise KeyError(emsg)
        return self.__dict__[key]

    def __repr__(self):
        cls = self.__class__.__name__
        msg = f"{cls}(maths={self.__dict__['maths']!r})"
        return msg

    def __setitem__(self, key, value):
        cls = self.__class__.__name__

        if key not in self.__dict__:
            emsg = f"Invalid {cls!r} option, got {key!r}."
            raise KeyError(emsg)

        if not isinstance(value, bool):
            emsg = f"Invalid {cls!r} option {key!r} value, got {value!r}."
            raise ValueError(emsg)

        self.__dict__[key] = value
        # Toggle the (private) lenient behaviour.
        _LENIENT.enable = value

    def _init(self):
        """Configure the initial default lenient state."""
        # This is the only public supported lenient feature i.e., cube arithmetic
        self.__dict__["maths"] = None

    @contextmanager
    def context(self, **kwargs):
        """Return a context manager which allows temporary modification of the
        lenient option state within the scope of the context manager.

        On entry to the context manager, all provided keyword arguments are
        applied. On exit from the context manager, the previous lenient
        option state is restored.


        For example::

            with iris.common.Lenient.context(maths=False):
                pass

        """

        def configure_state(state):
            for feature, value in state.items():
                self[feature] = value

        # Save the original state.
        original_state = deepcopy(self.__dict__)

        # Configure the provided lenient features.
        configure_state(kwargs)

        try:
            yield
        finally:
            # Restore the original state.
            self.__dict__.clear()
            self._init()
            configure_state(original_state)


###############################################################################


class _Lenient(threading.local):
    def __init__(self, *args, **kwargs):
        """A container for managing the run-time lenient services and client
        options for pre-defined functions/methods.

        Parameters
        ----------
        *args : callable or str or iterable of callable/str
            A function/method or fully qualified string name of the function/method
            acting as a lenient service.
        **kwargs : dict of callable/str or iterable of callable/str, optional
            Mapping of lenient client function/method, or fully qualified string name
            of the function/method, to one or more lenient service
            function/methods or fully qualified string name of function/methods.

        Examples
        --------
        ::

            _Lenient(service1, service2, client1=service1, client2=(service1, service2))

        Note that, the values of these options are thread-specific.

        """
        # The executing lenient client at runtime.
        self.__dict__["active"] = None
        # The global lenient services state activation switch.
        self.__dict__["enable"] = _LENIENT_ENABLE_DEFAULT

        for service in args:
            self.register_service(service)

        for client, services in kwargs.items():
            self.register_client(client, services)

    def __call__(self, func):
        """Determine whether it is valid for the function/method to provide a
        lenient service at runtime to the actively executing lenient client.

        Parameters
        ----------
        func : callable or str
            A function/method or fully qualified string name of the function/method.

        Returns
        -------
        bool

        """
        result = False
        if self.__dict__["enable"]:
            service = _qualname(func)
            if service in self and self.__dict__[service]:
                active = self.__dict__["active"]
                if active is not None and active in self:
                    services = self.__dict__[active]
                    if isinstance(services, str) or not isinstance(services, Iterable):
                        services = (services,)
                    result = service in services
        return result

    def __contains__(self, name):
        name = _qualname(name)
        return name in self.__dict__

    def __getattr__(self, name):
        if name not in self.__dict__:
            cls = self.__class__.__name__
            emsg = f"Invalid {cls!r} option, got {name!r}."
            raise AttributeError(emsg)
        return self.__dict__[name]

    def __getitem__(self, name):
        name = _qualname(name)
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

    def __setitem__(self, name, value):
        name = _qualname(name)
        cls = self.__class__.__name__

        if name not in self.__dict__:
            emsg = f"Invalid {cls!r} option, got {name!r}."
            raise KeyError(emsg)

        if name == "active":
            value = _qualname(value)
            if not isinstance(value, str) and value is not None:
                emsg = f"Invalid {cls!r} option {name!r}, expected a registered {cls!r} client, got {value!r}."
                raise ValueError(emsg)
            self.__dict__[name] = value
        elif name == "enable":
            self.enable = value
        else:
            if isinstance(value, str) or callable(value):
                value = (value,)
            if isinstance(value, Iterable):
                value = tuple([_qualname(item) for item in value])
            self.__dict__[name] = value

    @contextmanager
    def context(self, *args, **kwargs):
        """Return a context manager which allows temporary modification of
        the lenient option state for the active thread.

        On entry to the context manager, all provided keyword arguments are
        applied. On exit from the context manager, the previous lenient option
        state is restored.

        For example::

            with iris._LENIENT.context(example_lenient_flag=False):
                # ... code that expects some non-lenient behaviour

        .. note::
            iris._LENIENT.example_lenient_flag does not exist and is
            provided only as an example.

        """

        def update_client(client, services):
            if client in self.__dict__:
                existing_services = self.__dict__[client]
            else:
                existing_services = ()

            self.__dict__[client] = tuple(set(existing_services + services))

        # Save the original state.
        original_state = deepcopy(self.__dict__)

        # Temporarily update the state with the kwargs first.
        for name, value in kwargs.items():
            self[name] = value

        # Get the active client.
        active = self.__dict__["active"]

        if args:
            # Update the client with the provided services.
            new_services = tuple([_qualname(arg) for arg in args])

            if active is None:
                # Ensure not to use "context" as the ephemeral name
                # of the context manager runtime "active" lenient client,
                # as this causes a namespace clash with this method
                # i.e., _Lenient.context, via _Lenient.__getattr__
                active = "__context"
                self.__dict__["active"] = active
                self.__dict__[active] = new_services
            else:
                # Append provided services to any pre-existing services of the active client.
                update_client(active, new_services)
        else:
            # Append previous ephemeral services (for non-specific client) to the active client.
            if (
                active is not None
                and active != "__context"
                and "__context" in self.__dict__
            ):
                new_services = self.__dict__["__context"]
                update_client(active, new_services)

        try:
            yield
        finally:
            # Restore the original state.
            self.__dict__.clear()
            self.__dict__.update(original_state)

    @property
    def enable(self):
        """Return the activation state of the lenient services."""
        return self.__dict__["enable"]

    @enable.setter
    def enable(self, state):
        """Set the activate state of the lenient services.

        Setting the state to `False` disables all lenient services, and
        setting the state to `True` enables all lenient services.

        Parameters
        ----------
        state : bool
            Activate state for lenient services.

        """
        if not isinstance(state, bool):
            cls = self.__class__.__name__
            emsg = f"Invalid {cls!r} option 'enable', expected a {type(True)!r}, got {state!r}."
            raise ValueError(emsg)
        self.__dict__["enable"] = state

    def register_client(self, func, services, append=False):
        """Add the provided mapping of lenient client function/method to
        required lenient service function/methods.

        Parameters
        ----------
        func : callable or str
            A client function/method or fully qualified string name of the
            client function/method.
        services : callable or str or iterable of callable/str
            One or more service function/methods or fully qualified string names
            of the required service function/method.
        append : bool, optional
            If True, append the lenient services to any pre-registered lenient
            services for the provided lenient client. Default is False.

        """
        func = _qualname(func)
        cls = self.__class__.__name__

        if func in _LENIENT_PROTECTED:
            emsg = (
                f"Cannot register {cls!r} client. "
                f"Please rename your client to be something other than {func!r}."
            )
            raise ValueError(emsg)
        if isinstance(services, str) or not isinstance(services, Iterable):
            services = (services,)
        if not len(services):
            emsg = f"Require at least one {cls!r} client service."
            raise ValueError(emsg)
        services = tuple([_qualname(service) for service in services])
        if append:
            # The original provided service order is not significant. There is
            # no requirement to preserve it, so it's safe to sort.
            existing = self.__dict__[func] if func in self else ()
            services = tuple(sorted(set(existing) | set(services)))
        self.__dict__[func] = services

    def register_service(self, func):
        """Add the provided function/method as providing a lenient service and
        activate it.

        Parameters
        ----------
        func : callable or str
            A service function/method or fully qualified string name of the
            service function/method.

        """
        func = _qualname(func)
        if func in _LENIENT_PROTECTED:
            cls = self.__class__.__name__
            emsg = (
                f"Cannot register {cls!r} service. "
                f"Please rename your service to be something other than {func!r}."
            )
            raise ValueError(emsg)
        self.__dict__[func] = True

    def unregister_client(self, func):
        """Remove the provided function/method as a lenient client using lenient services.

        Parameters
        ----------
        func : callable or str
            A function/method of fully qualified string name of the function/method.

        """
        func = _qualname(func)
        cls = self.__class__.__name__

        if func in _LENIENT_PROTECTED:
            emsg = f"Cannot unregister {cls!r} client, as {func!r} is a protected {cls!r} option."
            raise ValueError(emsg)

        if func in self.__dict__:
            value = self.__dict__[func]
            if isinstance(value, bool):
                emsg = f"Cannot unregister {cls!r} client, as {func!r} is not a valid {cls!r} client."
                raise ValueError(emsg)
            del self.__dict__[func]
        else:
            emsg = f"Cannot unregister unknown {cls!r} client {func!r}."
            raise ValueError(emsg)

    def unregister_service(self, func):
        """Remove the provided function/method as providing a lenient service.

        Parameters
        ----------
        func : callable or str
            A function/method or fully qualified string name of the function/method.

        """
        func = _qualname(func)
        cls = self.__class__.__name__

        if func in _LENIENT_PROTECTED:
            emsg = f"Cannot unregister {cls!r} service, as {func!r} is a protected {cls!r} option."
            raise ValueError(emsg)

        if func in self.__dict__:
            value = self.__dict__[func]
            if not isinstance(value, bool):
                emsg = f"Cannot unregister {cls!r} service, as {func!r} is not a valid {cls!r} service."
                raise ValueError(emsg)
            del self.__dict__[func]
        else:
            emsg = f"Cannot unregister unknown {cls!r} service {func!r}."
            raise ValueError(emsg)


#: (Private) Instance that manages all Iris run-time lenient client and service options.
_LENIENT = _Lenient()

#: (Public) Instance that manages all Iris run-time lenient features.
LENIENT = Lenient()
