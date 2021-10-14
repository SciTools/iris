# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.logging.get_logger`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip
import logging
import unittest

from iris import logger as ilogger
from iris.logging import IrisFormatter, get_logger


class Test(tests.IrisTest):
    def setUp(self):
        # Get the singleton logging root logger
        self.root = logging.getLogger()

    @staticmethod
    def _filter_handlers(logger, name):
        return list(
            filter(lambda handler: handler.get_name() == name, logger.handlers)
        )

    def test_get_root__with_none(self):
        logger = get_logger(None)
        self.assertIs(self.root, logger)

    def test_get_root__with_name(self):
        logger = get_logger("root")
        self.assertIs(self.root, logger)

    def test_get_iris(self):
        logger = get_logger("iris")
        self.assertIs(logger, ilogger)

    def test_get_child(self):
        name = "iris.child"
        child = get_logger(name)
        logger = get_logger(name)
        self.assertIs(logger, child)

    def test_root_level(self):
        logger = get_logger("root")
        self.assertEqual(logger.level, logging._nameToLevel["WARN"])

    def test_iris_level(self):
        logger = get_logger("iris")
        self.assertEqual(logger.level, logging._nameToLevel["NOTSET"])

    def test_child_level(self):
        logger = get_logger("iris.child")
        self.assertEqual(logger.level, logging._nameToLevel["INFO"])

    def test_root_propagate(self):
        logger = get_logger(None)
        # default propagate behaviour is not overridden
        self.assertTrue(logger.propagate)

    def test_iris_propagate(self):
        logger = get_logger("iris")
        self.assertFalse(logger.propagate)

    def test_child_propagate(self):
        logger = get_logger("iris.child")
        self.assertTrue(logger.propagate)

    def test_root_handler(self):
        logger = get_logger(None)
        handlers = self._filter_handlers(logger, "root_handler")
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(handlers[0].formatter, IrisFormatter)

    def test_root_handler__multiple(self):
        logger = get_logger(None)
        other = get_logger(None)
        self.assertIs(logger, other)
        handlers = self._filter_handlers(logger, "root_handler")
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(handlers[0].formatter, IrisFormatter)

    def test_iris_handler(self):
        logger = get_logger("iris")
        handlers = self._filter_handlers(logger, "iris_handler")
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(handlers[0].formatter, IrisFormatter)

    def test_iris_handler__multiple(self):
        logger = get_logger("iris")
        other = get_logger("iris")
        self.assertIs(logger, other)
        handlers = self._filter_handlers(logger, "iris_handler")
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(handlers[0].formatter, IrisFormatter)

    def test_child_handler(self):
        logger = get_logger("iris.child")
        self.assertEqual(len(logger.handlers), 0)

    def test_child_handler__multiple(self):
        logger = get_logger("iris.child")
        other = get_logger("iris.child")
        self.assertIs(logger, other)
        self.assertEqual(len(logger.handlers), 0)


if __name__ == "__main__":
    unittest.main()
