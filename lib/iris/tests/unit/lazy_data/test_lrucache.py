# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._lazy data.LRUCache`."""

from iris._lazy_data import LRUCache


def test_lrucache():
    cache = LRUCache(2)

    cache["a"] = 1

    assert "a" in cache
    assert cache["a"] == 1

    cache["b"] = 2
    cache["c"] = 3

    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache

    assert str(cache) == "<LRUCache maxsize=2 cache={'b': 2, 'c': 3} >"
