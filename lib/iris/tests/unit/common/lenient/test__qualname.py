# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :func:`iris.common.lenient._qualname`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from inspect import getmodule
from unittest.mock import sentinel

from iris.common.lenient import _qualname


class Test(tests.IrisTest):
    def setUp(self):
        module_name = getmodule(self).__name__
        self.locals = f"{module_name}" + ".Test.{}.<locals>.{}"

    def test_pass_thru_non_callable(self):
        func = sentinel.func
        result = _qualname(func)
        self.assertEqual(result, func)

    def test_callable_function_local(self):
        def myfunc():
            pass

        qualname_func = self.locals.format(
            "test_callable_function_local", "myfunc"
        )
        result = _qualname(myfunc)
        self.assertEqual(result, qualname_func)

    def test_callable_function(self):
        import iris

        result = _qualname(iris.load)
        self.assertEqual(result, "iris.load")

    def test_callable_method_local(self):
        class MyClass:
            def mymethod(self):
                pass

        qualname_method = self.locals.format(
            "test_callable_method_local", "MyClass.mymethod"
        )
        result = _qualname(MyClass.mymethod)
        self.assertEqual(result, qualname_method)

    def test_callable_method(self):
        import iris

        result = _qualname(iris.cube.Cube.add_ancillary_variable)
        self.assertEqual(result, "iris.cube.Cube.add_ancillary_variable")


if __name__ == "__main__":
    tests.main()
