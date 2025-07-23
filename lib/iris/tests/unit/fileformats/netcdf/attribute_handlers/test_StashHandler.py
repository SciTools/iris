# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.netcdf._attribute_handlers.StashHandler`."""

import pytest

from iris.fileformats.netcdf._attribute_handlers import ATTRIBUTE_HANDLERS
from iris.fileformats.pp import STASH

STASH_HANDLER = ATTRIBUTE_HANDLERS["STASH"]


class TestEncodeObject:
    """Test how STASH attributes convert to strings for storage in actual netcdf files.

    These are mostly STASH objects, but we must also allow correctly formed STASH
    strings, and error other types of content.
    """

    def test_stash_object(self):
        """A STASH object is converted to its 'str'."""
        test_stash = STASH(3, 5, 123)
        result = STASH_HANDLER.encode_object(test_stash)
        expected = ("um_stash_source", str(test_stash))
        assert result == expected

    def test_stash_string(self):
        """A STASH-convertible str is regularised."""
        test_string = "m2s5i23"
        result = STASH_HANDLER.encode_object(test_string)
        expected = ("um_stash_source", "m02s05i023")  # Numbers filled to N digits
        assert result == expected

    def test_invalid_string__fail(self):
        with pytest.raises(ValueError, match="Expected STASH code MSI string"):
            STASH_HANDLER.encode_object("xxx")

    def test_empty_string__fail(self):
        with pytest.raises(ValueError, match="Expected STASH code MSI string"):
            STASH_HANDLER.encode_object("")

    def test_none_object__fail(self):
        with pytest.raises(TypeError, match="Invalid STASH attribute"):
            STASH_HANDLER.encode_object(None)

    def test_nonstash_object__fail(self):
        with pytest.raises(TypeError, match="Invalid STASH attribute"):
            STASH_HANDLER.encode_object({})


class TestDecodeAttribute:
    """Test how STASH string attributes convert back to STASH objects."""

    def test_standard(self):
        """Test valid MSI string."""
        test_string = "m01s02i213"
        result = STASH_HANDLER.decode_attribute(test_string)
        expected = STASH(1, 2, 213)
        assert result == expected

    def test_alternate_format(self):
        """Test the slight tolerances in formatting."""
        test_string = "  m1S002i3  "
        result = STASH_HANDLER.decode_attribute(test_string)
        expected = STASH(1, 2, 3)
        assert result == expected

    def test_invalid(self):
        test_string = "xxx"
        with pytest.raises(ValueError, match="Expected STASH code MSI"):
            STASH_HANDLER.decode_attribute(test_string)

    def test_empty(self):
        test_string = ""
        with pytest.raises(ValueError, match="Expected STASH code MSI"):
            STASH_HANDLER.decode_attribute(test_string)
