# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.mixin.LimitedAttributeDict`."""

import numpy as np
import pytest

from iris.common.mixin import LimitedAttributeDict


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.forbidden_keys = LimitedAttributeDict.CF_ATTRS_FORBIDDEN
        self.emsg = "{!r} is not a permitted attribute"

    def test__invalid_keys(self):
        for key in self.forbidden_keys:
            with pytest.raises(ValueError, match=self.emsg.format(key)):
                _ = LimitedAttributeDict(**{key: None})

    def test___eq__(self, mocker):
        values = dict(
            one=mocker.sentinel.one,
            two=mocker.sentinel.two,
            three=mocker.sentinel.three,
        )
        left = LimitedAttributeDict(**values)
        right = LimitedAttributeDict(**values)
        assert left == right
        assert left == values

    def test___eq___numpy(self):
        values = dict(one=np.arange(1), two=np.arange(2), three=np.arange(3))
        left = LimitedAttributeDict(**values)
        right = LimitedAttributeDict(**values)
        assert left == right
        assert left == values
        values = dict(one=np.arange(1), two=np.arange(1), three=np.arange(1))
        left = LimitedAttributeDict(dict(one=0, two=0, three=0))
        right = LimitedAttributeDict(**values)
        assert left == right
        assert left == values

        # Test inequality:
        values = dict(one=np.arange(1), two=np.arange(2), three=np.arange(3))
        left = LimitedAttributeDict(**values)
        right = LimitedAttributeDict(
            one=np.arange(3), two=np.arange(2), three=np.arange(1)
        )
        assert right != left
        assert right != values

    def test___setitem__(self):
        for key in self.forbidden_keys:
            item = LimitedAttributeDict()
            with pytest.raises(ValueError, match=self.emsg.format(key)):
                item[key] = None

    def test_update(self):
        for key in self.forbidden_keys:
            item = LimitedAttributeDict()
            other = {key: None}
            with pytest.raises(ValueError, match=self.emsg.format(key)):
                item.update(other)
