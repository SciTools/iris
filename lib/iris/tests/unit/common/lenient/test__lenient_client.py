# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.common.lenient._lenient_client`."""

from inspect import getmodule

import pytest

from iris.common.lenient import _LENIENT, _lenient_client


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        module_name = getmodule(self).__name__
        self.client = f"{module_name}" + ".Test.{}.<locals>.myclient"
        self.service = f"{module_name}" + ".Test.{}.<locals>.myservice"
        self.active = "active"
        self.args_in = mocker.sentinel.arg1, mocker.sentinel.arg2
        self.kwargs_in = dict(
            kwarg1=mocker.sentinel.kwarg1, kwarg2=mocker.sentinel.kwarg2
        )

    def test_args_too_many(self):
        emsg = "Invalid lenient client arguments, expecting 1"
        with pytest.raises(AssertionError, match=emsg):
            _ = _lenient_client(None, None)

    def test_args_not_callable(self):
        emsg = "Invalid lenient client argument, expecting a callable"
        with pytest.raises(AssertionError, match=emsg):
            _ = _lenient_client(None)

    def test_args_and_kwargs(self):
        def func():
            pass

        emsg = "Invalid lenient client, got both arguments and keyword arguments"
        with pytest.raises(AssertionError, match=emsg):
            _ = _lenient_client(func, services=func)

    def test_call_naked(self):
        @_lenient_client
        def myclient():
            return _LENIENT.__dict__.copy()

        result = myclient()
        assert self.active in result
        qualname_client = self.client.format("test_call_naked")
        assert result[self.active] == qualname_client
        assert qualname_client not in result

    def test_call_naked_alternative(self):
        def myclient():
            return _LENIENT.__dict__.copy()

        result = _lenient_client(myclient)()
        assert self.active in result
        qualname_client = self.client.format("test_call_naked_alternative")
        assert result[self.active] == qualname_client
        assert qualname_client not in result

    def test_call_naked_client_args_kwargs(self):
        @_lenient_client
        def myclient(*args, **kwargs):
            return args, kwargs

        args_out, kwargs_out = myclient(*self.args_in, **self.kwargs_in)
        assert args_out == self.args_in
        assert kwargs_out == self.kwargs_in

    def test_call_naked_doc(self):
        @_lenient_client
        def myclient():
            """Myclient doc-string."""

        assert myclient.__doc__ == "Myclient doc-string."

    def test_call_no_kwargs(self):
        @_lenient_client()
        def myclient():
            return _LENIENT.__dict__.copy()

        result = myclient()
        assert self.active in result
        qualname_client = self.client.format("test_call_no_kwargs")
        assert result[self.active] == qualname_client
        assert qualname_client not in result

    def test_call_no_kwargs_alternative(self):
        def myclient():
            return _LENIENT.__dict__.copy()

        result = (_lenient_client())(myclient)()
        assert self.active in result
        qualname_client = self.client.format("test_call_no_kwargs_alternative")
        assert result[self.active] == qualname_client
        assert qualname_client not in result

    def test_call_kwargs_none(self):
        @_lenient_client(services=None)
        def myclient():
            return _LENIENT.__dict__.copy()

        result = myclient()
        assert self.active in result
        qualname_client = self.client.format("test_call_kwargs_none")
        assert result[self.active] == qualname_client
        assert qualname_client not in result

    def test_call_kwargs_single(self, mocker):
        service = mocker.sentinel.service

        @_lenient_client(services=service)
        def myclient():
            return _LENIENT.__dict__.copy()

        result = myclient()
        assert self.active in result
        qualname_client = self.client.format("test_call_kwargs_single")
        assert result[self.active] == qualname_client
        assert qualname_client in result
        assert result[qualname_client] == (service,)

    def test_call_kwargs_single_callable(self):
        def myservice():
            pass

        @_lenient_client(services=myservice)
        def myclient():
            return _LENIENT.__dict__.copy()

        test_name = "test_call_kwargs_single_callable"
        result = myclient()
        assert self.active in result
        qualname_client = self.client.format(test_name)
        assert result[self.active] == qualname_client
        assert qualname_client in result
        qualname_services = (self.service.format(test_name),)
        assert result[qualname_client] == qualname_services

    def test_call_kwargs_iterable(self, mocker):
        services = (mocker.sentinel.service1, mocker.sentinel.service2)

        @_lenient_client(services=services)
        def myclient():
            return _LENIENT.__dict__.copy()

        result = myclient()
        assert self.active in result
        qualname_client = self.client.format("test_call_kwargs_iterable")
        assert result[self.active] == qualname_client
        assert qualname_client in result
        assert set(result[qualname_client]) == set(services)

    def test_call_client_args_kwargs(self):
        @_lenient_client()
        def myclient(*args, **kwargs):
            return args, kwargs

        args_out, kwargs_out = myclient(*self.args_in, **self.kwargs_in)
        assert args_out == self.args_in
        assert kwargs_out == self.kwargs_in

    def test_call_doc(self):
        @_lenient_client()
        def myclient():
            """Myclient doc-string."""

        assert myclient.__doc__ == "Myclient doc-string."
