# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.common.lenient._qualname`."""

from inspect import getmodule

import pytest

from iris.common.lenient import _qualname


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self):
        module_name = getmodule(self).__name__
        self.locals = f"{module_name}" + ".Test.{}.<locals>.{}"

    def test_pass_thru_non_callable(self, mocker):
        func = mocker.sentinel.func
        result = _qualname(func)
        assert result == func

    def test_callable_function_local(self):
        def myfunc():
            pass

        qualname_func = self.locals.format("test_callable_function_local", "myfunc")
        result = _qualname(myfunc)
        assert result == qualname_func

    def test_callable_function(self):
        import iris

        result = _qualname(iris.load)
        assert result == "iris.loading.load"

    def test_callable_method_local(self):
        class MyClass:
            def mymethod(self):
                pass

        qualname_method = self.locals.format(
            "test_callable_method_local", "MyClass.mymethod"
        )
        result = _qualname(MyClass.mymethod)
        assert result == qualname_method

    def test_callable_method(self):
        import iris

        result = _qualname(iris.cube.Cube.add_ancillary_variable)
        assert result == "iris.cube.Cube.add_ancillary_variable"
