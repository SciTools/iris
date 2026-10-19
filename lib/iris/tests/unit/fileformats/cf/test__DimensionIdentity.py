# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""`_DimensionIdentity`, which decides what counts as the same dimension.

Equality and hashing are hand-written, and `CFVariable.spans` and
`iris.fileformats._nc_load_rules.helpers._common_data_dims` both lean on
them through `set.issubset`, `in` and `tuple.index`. The pairs below are the
whole contract.
"""

from iris.fileformats.cf._variables import _DimensionIdentity


def test_same_name_and_size_are_equal():
    assert _DimensionIdentity("time", 4) == _DimensionIdentity("time", 4)


def test_same_name_and_size_hash_alike():
    # Equal members must land in the same bucket, or `set.issubset` misses.
    assert hash(_DimensionIdentity("time", 4)) == hash(_DimensionIdentity("time", 4))


def test_same_name_at_two_sizes_is_not_equal():
    # The reason matching is a pair: Zarr lets one name mean two lengths.
    assert _DimensionIdentity("time", 4) != _DimensionIdentity("time", 5)


def test_different_names_at_one_size_are_not_equal():
    assert _DimensionIdentity("time", 4) != _DimensionIdentity("level", 4)


def test_an_anonymous_dimension_equals_itself():
    identity = _DimensionIdentity(None, 4)
    assert identity == identity


def test_two_anonymous_dimensions_are_not_equal():
    # Not even at the same size. There is nothing to match on.
    assert _DimensionIdentity(None, 4) != _DimensionIdentity(None, 4)


def test_an_anonymous_dimension_does_not_equal_a_named_one():
    assert _DimensionIdentity(None, 4) != _DimensionIdentity("time", 4)
    assert _DimensionIdentity("time", 4) != _DimensionIdentity(None, 4)


def test_a_foreign_object_is_not_equal():
    # `__eq__` returns NotImplemented, so Python falls back to identity.
    assert _DimensionIdentity("time", 4) != ("time", 4)


def test_named_identities_work_as_a_subset():
    source = {_DimensionIdentity("time", 4)}
    target = {_DimensionIdentity("time", 4), _DimensionIdentity("level", 3)}
    assert source.issubset(target)


def test_an_anonymous_dimension_is_never_a_subset():
    source = {_DimensionIdentity(None, 4)}
    target = {_DimensionIdentity(None, 4), _DimensionIdentity("level", 3)}
    assert not source.issubset(target)
