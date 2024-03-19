# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.lenient.Lenient`."""

import pytest

from iris.common.lenient import _LENIENT, _LENIENT_PROTECTED, Lenient


@pytest.fixture()
def lenient():
    # setup
    state = {key: _LENIENT.__dict__[key] for key in _LENIENT_PROTECTED}
    # call
    yield Lenient()
    # teardown
    for key, value in state.items():
        _LENIENT.__dict__[key] = value


class Test___init__:
    def test_default(self, lenient):
        expected = dict(maths=True)
        assert lenient.__dict__ == expected

    def test_kwargs(self, lenient):
        actual = Lenient(maths=False)
        expected = dict(maths=False)
        assert actual.__dict__ == expected

    def test_kwargs_invalid(self, lenient):
        emsg = "Invalid .* option, got 'merge'."
        with pytest.raises(KeyError, match=emsg):
            _ = Lenient(merge=True)


class Test___contains__:
    def test_in(self, lenient):
        assert "maths" in lenient

    def test_not_in(self, lenient):
        assert "concatenate" not in lenient


class Test___getitem__:
    def test_in(self, lenient):
        assert bool(lenient["maths"]) is True

    def test_not_in(self, lenient):
        emsg = "Invalid .* option, got 'MATHS'."
        with pytest.raises(KeyError, match=emsg):
            _ = lenient["MATHS"]


class Test___repr__:
    def test(self, lenient):
        expected = "Lenient(maths=True)"
        assert repr(lenient) == expected


class Test___setitem__:
    def test_key_invalid(self, lenient):
        emsg = "Invalid .* option, got 'MATHS."
        with pytest.raises(KeyError, match=emsg):
            lenient["MATHS"] = False

    def test_maths_value_invalid(self, mocker, lenient):
        value = mocker.sentinel.value
        emsg = f"Invalid .* option 'maths' value, got {value!r}."
        with pytest.raises(ValueError, match=emsg):
            lenient["maths"] = value

    def test_maths_disable__lenient_enable_true(self, lenient):
        assert bool(_LENIENT.enable) is True
        lenient["maths"] = False
        assert bool(lenient.__dict__["maths"]) is False
        assert bool(_LENIENT.enable) is False

    def test_maths_disable__lenient_enable_false(self, lenient):
        _LENIENT.__dict__["enable"] = False
        assert bool(_LENIENT.enable) is False
        lenient["maths"] = False
        assert bool(lenient.__dict__["maths"]) is False
        assert bool(_LENIENT.enable) is False

    def test_maths_enable__lenient_enable_true(self, lenient):
        assert bool(_LENIENT.enable) is True
        lenient["maths"] = True
        assert bool(lenient.__dict__["maths"]) is True
        assert bool(_LENIENT.enable) is True

    def test_maths_enable__lenient_enable_false(self, lenient):
        _LENIENT.__dict__["enable"] = False
        assert bool(_LENIENT.enable) is False
        lenient["maths"] = True
        assert bool(lenient.__dict__["maths"]) is True
        assert bool(_LENIENT.enable) is True


class Test_context:
    def test_nop(self, lenient):
        assert bool(lenient["maths"]) is True

        with lenient.context():
            assert bool(lenient["maths"]) is True

        assert bool(lenient["maths"]) is True

    def test_maths_disable__lenient_true(self, lenient):
        # synchronised
        assert bool(_LENIENT.enable) is True
        assert bool(lenient["maths"]) is True

        with lenient.context(maths=False):
            # still synchronised
            assert bool(_LENIENT.enable) is False
            assert bool(lenient["maths"]) is False

        # still synchronised
        assert bool(_LENIENT.enable) is True
        assert bool(lenient["maths"]) is True

    def test_maths_disable__lenient_false(self, lenient):
        # not synchronised
        _LENIENT.__dict__["enable"] = False
        assert bool(_LENIENT.enable) is False
        assert bool(lenient["maths"]) is True

        with lenient.context(maths=False):
            # now synchronised
            assert bool(_LENIENT.enable) is False
            assert bool(lenient["maths"]) is False

        # still synchronised
        assert bool(_LENIENT.enable) is True
        assert bool(lenient["maths"]) is True

    def test_maths_enable__lenient_true(self, lenient):
        # not synchronised
        assert bool(_LENIENT.enable) is True
        lenient.__dict__["maths"] = False
        assert bool(lenient["maths"]) is False

        with lenient.context(maths=True):
            # now synchronised
            assert bool(_LENIENT.enable) is True
            assert bool(lenient["maths"]) is True

        # still synchronised
        assert bool(_LENIENT.enable) is False
        assert bool(lenient["maths"]) is False

    def test_maths_enable__lenient_false(self, lenient):
        # synchronised
        _LENIENT.__dict__["enable"] = False
        assert bool(_LENIENT.enable) is False
        lenient.__dict__["maths"] = False
        assert bool(lenient["maths"]) is False

        with lenient.context(maths=True):
            # still synchronised
            assert bool(_LENIENT.enable) is True
            assert bool(lenient["maths"]) is True

        # still synchronised
        assert bool(_LENIENT.enable) is False
        assert bool(lenient["maths"]) is False
