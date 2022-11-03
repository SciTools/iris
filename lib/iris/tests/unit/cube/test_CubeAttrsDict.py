# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.cube.CubeAttrsDict` class."""

import pickle

import numpy as np
import pytest

from iris.common.mixin import LimitedAttributeDict
from iris.cube import CubeAttrsDict


@pytest.fixture
def sample_attrs() -> CubeAttrsDict:
    return CubeAttrsDict(
        locals={"a": 1, "z": "this"}, globals={"b": 2, "z": "that"}
    )


def check_content(attrs, locals=None, globals=None, matches=None):
    # Check a CubeAttrsDict for expected properties.
    # If locals/globals are set (including =None), test for equality and not identity.
    assert isinstance(attrs, CubeAttrsDict)
    attr_locals, attr_globals = attrs.locals, attrs.globals
    assert type(attr_locals) == LimitedAttributeDict
    assert type(attr_globals) == LimitedAttributeDict
    if matches:
        locals, globals = matches.locals, matches.globals

    def check(arg, content):
        if not arg:
            arg = {}
        if not isinstance(arg, LimitedAttributeDict):
            arg = LimitedAttributeDict(arg)
        # N.B. if 'arg' is an actual given LimitedAttributeDict, it is not changed..
        # .. we proceed to ensure that the stored content is equal but NOT the same
        assert content == arg
        assert content is not arg

    check(locals, attr_locals)
    check(globals, attr_globals)


class Test___init__:
    def test_empty(self):
        attrs = CubeAttrsDict()
        check_content(attrs, None, None)

    def test_from_combined_dict(self):
        attrs = CubeAttrsDict({"q": 3, "history": "something"})
        check_content(attrs, locals={"q": 3}, globals={"history": "something"})

    def test_from_separate_dicts(self):
        locals = {"q": 3}
        globals = {"history": "something"}
        attrs = CubeAttrsDict(locals=locals, globals=globals)
        check_content(attrs, locals=locals, globals=globals)

    def test_from_cubeattrsdict(self, sample_attrs):
        result = CubeAttrsDict(sample_attrs)
        check_content(
            result, locals=sample_attrs.locals, globals=sample_attrs.globals
        )

    def test_from_cubeattrsdict_like(self):
        class MyDict:
            pass

        mydict = MyDict()
        locals, globals = {"a": 1}, {"b": 2}
        mydict.locals = locals
        mydict.globals = globals
        attrs = CubeAttrsDict(mydict)
        check_content(attrs, locals=locals, globals=globals)


class Test_OddMethods:
    def test_pickle(self, sample_attrs):
        bytes = pickle.dumps(sample_attrs)
        result = pickle.loads(bytes)
        check_content(result, matches=sample_attrs)

    def test_clear(self, sample_attrs):
        sample_attrs.clear()
        check_content(sample_attrs, {}, {})

    def test_del(self, sample_attrs):
        # 'z' is in boht locals+globals.  Delete removes both.
        assert "z" in sample_attrs.keys()
        del sample_attrs["z"]
        assert "z" not in sample_attrs.keys()

    def test_copy(self, sample_attrs):
        copy = sample_attrs.copy()
        assert copy is not sample_attrs
        check_content(copy, matches=sample_attrs)

    def test_update(self, sample_attrs):
        updated = sample_attrs.copy()
        updated.update({"q": 77})
        expected_locals = sample_attrs.locals.copy()
        expected_locals["q"] = 77
        check_content(
            updated, globals=sample_attrs.globals, locals=expected_locals
        )

    def test_to_dict(self, sample_attrs):
        result = dict(sample_attrs)
        expected = sample_attrs.globals.copy()
        expected.update(sample_attrs.locals)
        assert result == expected

    def test_array_copies(self):
        array = np.array([3, 2, 1, 4])
        map = {"array": array}
        attrs = CubeAttrsDict(map)
        check_content(attrs, globals=None, locals=map)
        attrs_array = attrs["array"]
        assert np.all(attrs_array == array)
        assert attrs_array is not array

    def test__str__(self, sample_attrs):
        result = str(sample_attrs)
        assert result == "{'b': 2, 'z': 'this', 'a': 1}"

    def test__repr__(self, sample_attrs):
        result = repr(sample_attrs)
        expected = (
            "CubeAttrsDict("
            "globals={'b': 2, 'z': 'that'}, "
            "locals={'a': 1, 'z': 'this'})"
        )
        assert result == expected


class TestEq:
    def test_eq_empty(self):
        attrs_1 = CubeAttrsDict()
        attrs_2 = CubeAttrsDict()
        assert attrs_1 == attrs_2

    def test_eq_nonempty(self, sample_attrs):
        attrs_1 = sample_attrs
        attrs_2 = sample_attrs.copy()
        assert attrs_1 == attrs_2

    @pytest.mark.parametrize("aspect", ["locals", "globals"])
    def test_ne_missing(self, sample_attrs, aspect):
        attrs_1 = sample_attrs
        attrs_2 = sample_attrs.copy()
        del getattr(attrs_2, aspect)["z"]
        assert attrs_1 != attrs_2
        assert attrs_2 != attrs_1

    @pytest.mark.parametrize("aspect", ["locals", "globals"])
    def test_ne_different(self, sample_attrs, aspect):
        attrs_1 = sample_attrs
        attrs_2 = sample_attrs.copy()
        getattr(attrs_2, aspect)["z"] = 99
        assert attrs_1 != attrs_2
        assert attrs_2 != attrs_1

    def test_ne_locals_vs_globals(self):
        attrs_1 = CubeAttrsDict(locals={"a": 1})
        attrs_2 = CubeAttrsDict(globals={"a": 1})
        assert attrs_1 != attrs_2
        assert attrs_2 != attrs_1

    def test_eq_dict(self):
        # A CubeAttrsDict can be equal to a plain dictionary (which would create it)
        vals_dict = {"a": 1, "b": 2, "history": "this"}
        attrs = CubeAttrsDict(vals_dict)
        assert attrs == vals_dict
        assert vals_dict == attrs

    def test_ne_dict_local_global(self):
        # Dictionary equivalence fails if the local/global assignments are wrong.
        # sample dictionary
        vals_dict = {"title": "b"}
        # these attrs are *not* the same, because 'title' is global by default
        attrs = CubeAttrsDict(locals={"title": "b"})
        assert attrs != vals_dict
        assert vals_dict != attrs

    def test_empty_not_none(self):
        # An empty CubeAttrsDict is not None, and does not compare to 'None'
        # N.B. this for compatibility with the LimitedAttributeDict
        attrs = CubeAttrsDict()
        assert attrs is not None
        with pytest.raises(TypeError, match="iterable"):
            # Cannot *compare* to None (or anything non-iterable)
            # N.B. not actually testing against None, as it upsets black (!)
            attrs == 0

    def test_empty_eq_iterables(self):
        # An empty CubeAttrsDict is "equal" to various empty containers
        attrs = CubeAttrsDict()
        assert attrs == {}
        assert attrs == []
        assert attrs == ()


class TestDictOrderBehaviour:
    def test_ordering(self):
        attrs = CubeAttrsDict({"a": 1, "b": 2})
        assert list(attrs.keys()) == ["a", "b"]
        # Remove, then reinstate 'a' : it will go to the back
        del attrs["a"]
        attrs["a"] = 1
        assert list(attrs.keys()) == ["b", "a"]

    def test_globals_locals_ordering(self):
        # create attrs with a global attribute set *before* a local one ..
        attrs = CubeAttrsDict()
        attrs.globals.update(dict(a=1, m=3))
        attrs.locals.update(dict(f=7, z=4))
        # .. and check key order of combined attrs
        assert list(attrs.keys()) == ["a", "m", "f", "z"]
        # create the "same" thing with locals before globals, *and* different key order
        attrs = CubeAttrsDict()
        attrs.locals.update(dict(z=4, f=7))
        attrs.globals.update(dict(m=3, a=1))
        # .. this shows that the result is not affected either by alphabetical key
        # order, or the order of adding locals/globals
        # I.E. result is globals-in-create-order, then locals-in-create-order
        assert list(attrs.keys()) == ["m", "a", "z", "f"]


class TestSettingBehaviours:
    def test_add_localtype(self):
        attrs = CubeAttrsDict()
        attrs["z"] = 3
        check_content(attrs, locals={"z": 3})

    def test_add_globaltype(self):
        attrs = CubeAttrsDict()
        attrs["history"] = "this"
        check_content(attrs, globals={"history": "this"})

    def test_overwrite_local(self):
        attrs = CubeAttrsDict(locals={"a": 1})
        attrs["a"] = "this"
        check_content(attrs, locals={"a": "this"})

    def test_overwrite_global(self):
        attrs = CubeAttrsDict(globals={"a": 1})
        # Since a global attr exists, it will write that instead of creating a local
        attrs["a"] = "this"
        check_content(attrs, globals={"a": "this"})

    def test_overwrite_forced_local(self):
        attrs = CubeAttrsDict(locals={"history": 1})
        # The attr remains local, even though it would be created global by default
        attrs["history"] = 2
        check_content(attrs, locals={"history": 2})

    def test_overwrite_forced_global(self):
        attrs = CubeAttrsDict(globals={"data": 1})
        # The attr remains global, even though it would be created global by default
        attrs["data"] = 2
        check_content(attrs, globals={"data": 2})

    def test_overwrite_both(self):
        attrs = CubeAttrsDict(locals={"z": 1}, globals={"z": 1})
        # Where both exits, it will update the local one
        attrs["z"] = 2
        check_content(attrs, locals={"z": 2}, globals={"z": 1})

    def test_local_global_masking(self, sample_attrs):
        # initially, local 'z' masks the global one
        assert sample_attrs["z"] == sample_attrs.locals["z"]
        # remove local, global will show
        del sample_attrs.locals["z"]
        assert sample_attrs["z"] == sample_attrs.globals["z"]
        # re-set local
        sample_attrs.locals["z"] = "new"
        assert sample_attrs["z"] == "new"
        # change the global, makes no difference
        sample_attrs.globals["z"] == "other"
        assert sample_attrs["z"] == "new"
