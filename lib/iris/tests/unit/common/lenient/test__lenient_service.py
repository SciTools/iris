# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.common.lenient._lenient_service`."""

from inspect import getmodule

import pytest

from iris.common.lenient import _LENIENT, _lenient_service


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        module_name = getmodule(self).__name__
        self.service = f"{module_name}" + ".Test.{}.<locals>.myservice"
        self.args_in = mocker.sentinel.arg1, mocker.sentinel.arg2
        self.kwargs_in = dict(
            kwarg1=mocker.sentinel.kwarg1, kwarg2=mocker.sentinel.kwarg2
        )

    def test_args_too_many(self):
        emsg = "Invalid lenient service arguments, expecting 1"
        with pytest.raises(AssertionError, match=emsg):
            _lenient_service(None, None)

    def test_args_not_callable(self):
        emsg = "Invalid lenient service argument, expecting a callable"
        with pytest.raises(AssertionError, match=emsg):
            _lenient_service(None)

    def test_call_naked(self):
        @_lenient_service
        def myservice():
            return _LENIENT.__dict__.copy()

        qualname_service = self.service.format("test_call_naked")
        state = _LENIENT.__dict__
        assert qualname_service in state
        assert state[qualname_service]
        result = myservice()
        assert qualname_service in result
        assert result[qualname_service]

    def test_call_naked_alternative(self):
        def myservice():
            return _LENIENT.__dict__.copy()

        qualname_service = self.service.format("test_call_naked_alternative")
        result = _lenient_service(myservice)()
        assert qualname_service in result
        assert result[qualname_service]

    def test_call_naked_service_args_kwargs(self):
        @_lenient_service
        def myservice(*args, **kwargs):
            return args, kwargs

        args_out, kwargs_out = myservice(*self.args_in, **self.kwargs_in)
        assert args_out == self.args_in
        assert kwargs_out == self.kwargs_in

    def test_call_naked_doc(self):
        @_lenient_service
        def myservice():
            """Myservice doc-string."""

        assert myservice.__doc__ == "Myservice doc-string."

    def test_call(self):
        @_lenient_service()
        def myservice():
            return _LENIENT.__dict__.copy()

        qualname_service = self.service.format("test_call")
        state = _LENIENT.__dict__
        assert qualname_service in state
        assert state[qualname_service]
        result = myservice()
        assert qualname_service in result
        assert result[qualname_service]

    def test_call_alternative(self):
        def myservice():
            return _LENIENT.__dict__.copy()

        qualname_service = self.service.format("test_call_alternative")
        result = (_lenient_service())(myservice)()
        assert qualname_service in result
        assert result[qualname_service]

    def test_call_service_args_kwargs(self):
        @_lenient_service()
        def myservice(*args, **kwargs):
            return args, kwargs

        args_out, kwargs_out = myservice(*self.args_in, **self.kwargs_in)
        assert args_out == self.args_in
        assert kwargs_out == self.kwargs_in

    def test_call_doc(self):
        @_lenient_service()
        def myservice():
            """Myservice doc-string."""

        assert myservice.__doc__ == "Myservice doc-string."
