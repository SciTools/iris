# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.log.IrisFormatter`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip
import unittest

from iris.log import IrisFormatter, get_logger

# create test child logger of iris
NAME = "iris.test"
LEVEL = "DEBUG"
logger = get_logger(NAME, level=LEVEL)

# test message
MESSAGE = "test message"


# test function that logs
def my_test_func():
    logger.debug(MESSAGE)


# test class that logs
class MyTestClass:
    def __init__(self):
        self.cls = self.__class__.__name__

    def my_test_method__no_cls(self):
        logger.debug(MESSAGE)

    def my_test_method__with_cls(self):
        logger.debug(MESSAGE, extra=dict(cls=self.cls))


class Test(tests.IrisTest):
    def setUp(self):
        self.formatter = IrisFormatter()
        self.output = [f"{LEVEL}:{NAME}:{MESSAGE}"]
        self.obj = MyTestClass()

    def test_format__my_test_func(self):
        expected = r"[caller:my_test_func]"
        with self.assertLogs(logger, level=LEVEL) as cm:
            my_test_func()
        self.assertEqual(cm.output, self.output)
        actual = self.formatter.format(cm.records[0])
        self.assertRegex(actual, expected)

    def test_format__my_test_method__no_cls(self):
        expected = r"[caller:my_test_method__no_cls]"
        with self.assertLogs(logger, level=LEVEL) as cm:
            self.obj.my_test_method__no_cls()
        self.assertEqual(cm.output, self.output)
        actual = self.formatter.format(cm.records[0])
        self.assertRegex(actual, expected)

    def test_format__my_test_method__with_cls(self):
        expected = r"[caller:my_test_method__with_cls]"
        with self.assertLogs(logger, level=LEVEL) as cm:
            self.obj.my_test_method__with_cls()
        self.assertEqual(cm.output, self.output)
        actual = self.formatter.format(cm.records[0])
        self.assertRegex(actual, expected)


if __name__ == "__main__":
    unittest.main()
