# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.mixin.LimitedAttributeDict`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.common.mixin import LimitedAttributeDict


class Test(tests.IrisTest):
    def setUp(self):
        self.forbidden_keys = LimitedAttributeDict.CF_ATTRS_FORBIDDEN
        self.emsg = "{!r} is not a permitted attribute"

    def test__invalid_keys(self):
        for key in self.forbidden_keys:
            with self.assertRaisesRegex(ValueError, self.emsg.format(key)):
                _ = LimitedAttributeDict(**{key: None})

    def test___eq__(self):
        values = dict(
            one=mock.sentinel.one,
            two=mock.sentinel.two,
            three=mock.sentinel.three,
        )
        left = LimitedAttributeDict(**values)
        right = LimitedAttributeDict(**values)
        self.assertEqual(left, right)
        self.assertEqual(left, values)

    def test___eq___numpy(self):
        values = dict(one=np.arange(1), two=np.arange(2), three=np.arange(3))
        left = LimitedAttributeDict(**values)
        right = LimitedAttributeDict(**values)
        self.assertEqual(left, right)
        self.assertEqual(left, values)
        values = dict(one=np.arange(1), two=np.arange(1), three=np.arange(1))
        left = LimitedAttributeDict(dict(one=0, two=0, three=0))
        right = LimitedAttributeDict(**values)
        self.assertEqual(left, right)
        self.assertEqual(left, values)

    def test___setitem__(self):
        for key in self.forbidden_keys:
            item = LimitedAttributeDict()
            with self.assertRaisesRegex(ValueError, self.emsg.format(key)):
                item[key] = None

    def test_update(self):
        for key in self.forbidden_keys:
            item = LimitedAttributeDict()
            with self.assertRaisesRegex(ValueError, self.emsg.format(key)):
                other = {key: None}
                item.update(other)


if __name__ == "__main__":
    tests.main()
