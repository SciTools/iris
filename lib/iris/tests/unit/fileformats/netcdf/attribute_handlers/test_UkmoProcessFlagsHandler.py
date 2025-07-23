# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for
:class:`iris.fileformats.netcdf._attribute_handlers.UkmoProcessFlagsHandler`.
"""

import numpy as np
import pytest

from iris.fileformats.netcdf._attribute_handlers import ATTRIBUTE_HANDLERS

UPF_HANDLER = ATTRIBUTE_HANDLERS["ukmo__process_flags"]


class TestEncodeObject:
    """Test how 'ukmo__process_flags' attributes convert to tuples in netcdf files."""

    def test_basic_tuple(self):
        """A tuple of strings converts to a single string."""
        test_tuple = ("one", "two", "three")
        result = UPF_HANDLER.encode_object(test_tuple)
        expected = "one two three"
        assert result == ("ukmo__process_flags", expected)

    def test_single(self):
        test_tuple = ("one",)
        result = UPF_HANDLER.encode_object(test_tuple)
        expected = "one"
        assert result == ("ukmo__process_flags", expected)

    def test_0_tuple(self):
        test_tuple = ()
        result = UPF_HANDLER.encode_object(test_tuple)
        expected = ""
        assert result == ("ukmo__process_flags", expected)

    def test_empty_element(self):
        test_tuple = ("one", "", "two")
        result = UPF_HANDLER.encode_object(test_tuple)
        expected = "one <EMPTY> two"
        assert result == ("ukmo__process_flags", expected)

    def test_spaced_element(self):
        test_tuple = ("one", "two three")
        result = UPF_HANDLER.encode_object(test_tuple)
        expected = "one two_three"
        assert result == ("ukmo__process_flags", expected)

    def test_underscores(self):
        """Can't distinguish original underscores and spaces. Doesn't really matter."""
        test_tuple = ("_", " ", "a_b", "a b")
        result = UPF_HANDLER.encode_object(test_tuple)
        expected = "_ _ a_b a_b"
        assert result == ("ukmo__process_flags", expected)

    @pytest.mark.parametrize(
        "badval",
        ["this", 1, ("a", 1, "b"), ("a", None), ["a", "b"], None],
        ids=["string", "int", "tuplewithInt", "tuplewithNone", "listofStr", "none"],
    )
    def test_non_tuple__fail(self, badval):
        """Won't convert anything but a tuple of strings."""
        with pytest.raises(TypeError, match="Invalid 'ukmo__process_flags' attribute"):
            UPF_HANDLER.encode_object(badval)


class TestDecodeAttribute:
    """Test how 'ukmo__process_flags' converts from file string back to a tuple."""

    def test_standard(self):
        test_string = "one two"
        result = UPF_HANDLER.decode_attribute(test_string)
        assert result == ("one", "two")

    def test_empty(self):
        test_string = ""
        result = UPF_HANDLER.decode_attribute(test_string)
        assert result == ()

    def test_empty_element(self):
        test_string = "<EMPTY>"
        result = UPF_HANDLER.decode_attribute(test_string)
        assert result == ("",)

    def test_empty_among_elements(self):
        test_string = "a <EMPTY> b"
        result = UPF_HANDLER.decode_attribute(test_string)
        assert result == ("a", "", "b")

    def test_embedded_spaces(self):
        """Extra spaces result in additional empty elements. Never mind!."""
        test_string = "a  b   c"
        result = UPF_HANDLER.decode_attribute(test_string)
        assert result == ("a", "", "b", "", "", "c")

    def test_underscores(self):
        """Extra spaces result in additional empty elements. Never mind!."""
        test_string = "_a b_c _ d_"
        result = UPF_HANDLER.decode_attribute(test_string)
        assert result == (" a", "b c", " ", "d ")

    def test_junk_string(self):
        """There's no such thing as an undecodable string."""
        test_string = "xxx"
        result = UPF_HANDLER.decode_attribute(test_string)
        assert result == ("xxx",)

    @pytest.mark.parametrize("badtype", ("int", "intarray", "floatarray"))
    def test_numeric_values(self, badtype):
        """Even array attributes get converted to a string + split."""
        if badtype == "int":
            test_value = 1
            expected = ("1",)
        elif badtype == "intarray":
            test_value = np.array([1, 2])
            expected = ("[1", "2]")
        elif badtype == "floatarray":
            test_value = np.array([1.2, 2.5, 3.7])
            expected = ("[1.2", "2.5", "3.7]")
        else:
            raise ValueError(f"Unrecognised param : {badtype}")

        result = UPF_HANDLER.decode_attribute(test_value)
        assert result == expected
