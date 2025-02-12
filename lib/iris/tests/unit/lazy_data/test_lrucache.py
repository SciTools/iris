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
