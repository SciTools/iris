# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.experimental.ugrid.load.ParseUgridOnLoad` class.

todo: remove this module when experimental.ugrid is folded into standard behaviour.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.experimental.ugrid.load import PARSE_UGRID_ON_LOAD, ParseUGridOnLoad


class TestClass(tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        cls.cls = ParseUGridOnLoad()

    def test_default(self):
        self.assertFalse(self.cls)

    def test_context(self):
        self.assertFalse(self.cls)
        with self.cls.context():
            self.assertTrue(self.cls)
        self.assertFalse(self.cls)


class TestConstant(tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        cls.constant = PARSE_UGRID_ON_LOAD

    def test_default(self):
        self.assertFalse(self.constant)

    def test_context(self):
        self.assertFalse(self.constant)
        with self.constant.context():
            self.assertTrue(self.constant)
        self.assertFalse(self.constant)
