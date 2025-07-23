# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.netcdf._attribute_handlers.StashHandler`."""

import pytest

iris_grib = pytest.importorskip("iris_grib")
from iris_grib.grib_phenom_translation._gribcode import (
    GenericConcreteGRIBCode,
    GRIBCode,
)
import numpy as np

from iris.fileformats.netcdf._attribute_handlers import ATTRIBUTE_HANDLERS

GP_HANDLER = ATTRIBUTE_HANDLERS["GRIB_PARAM"]


class TestEncodeObject:
    """Test how GRIB_PARAM attributes convert to strings for storage in netcdf files."""

    def test_edition1_codeobject(self):
        test_code = GRIBCode(1, 2, 3, 4)
        result = GP_HANDLER.encode_object(test_code)
        expected = "GRIBCode(edition=1, table_version=2, centre_number=3, number=4)"
        assert result == ("GRIB_PARAM", expected)

    def test_edition1_minimal_string(self):
        test_string = "1 2 3 4"
        result = GP_HANDLER.encode_object(test_string)
        expected = "GRIBCode(edition=1, table_version=2, centre_number=3, number=4)"
        assert result == ("GRIB_PARAM", expected)

    def test_edition1_alternate_string(self):
        test_string = "##1-a22bb33@z!44##"
        result = GP_HANDLER.encode_object(test_string)
        expected = "GRIBCode(edition=1, table_version=22, centre_number=33, number=44)"
        assert result == ("GRIB_PARAM", expected)

    def test_bad_string__toofew__fail(self):
        test_string = "1, 2, 3"
        msg = "Invalid argument for GRIBCode creation.*requires 4 numbers"
        with pytest.raises(ValueError, match=msg):
            GP_HANDLER.encode_object(test_string)

    def test_edition1_string_toomany(self):
        """No objection to extra numbers -- ignored."""
        test_string = "1, 2, 3, 4, 5, 6"
        result = GP_HANDLER.encode_object(test_string)
        expected = "GRIBCode(edition=1, table_version=2, centre_number=3, number=4)"
        assert result == ("GRIB_PARAM", expected)

    def test_bad_edition(self):
        test_string = "7,1,2,3"
        msg = "Invalid grib edition.*for GRIBcode : can only be 1 or 2"
        with pytest.raises(ValueError, match=msg):
            GP_HANDLER.encode_object(test_string)

    def test_edition2_codeobject(self):
        test_code = GRIBCode(2, 3, 4, 5)
        result = GP_HANDLER.encode_object(test_code)
        expected = "GRIBCode(edition=2, discipline=3, category=4, number=5)"
        assert result == ("GRIB_PARAM", expected)

    def test_string_edition2(self):
        """A STASH object is converted to its 'str'."""
        test_code = "2, 3, 4, 5"
        result = GP_HANDLER.encode_object(test_code)
        expected = "GRIBCode(edition=2, discipline=3, category=4, number=5)"
        assert result == ("GRIB_PARAM", expected)


class TestDecodeAttribute:
    """Test how GRIB_PARAM attributes convert back to GRIBCode objects."""

    def test_grib1(self):
        test_string = "GRIBCode(edition=1, table_version=2, centre_number=3, number=4)"
        result = GP_HANDLER.decode_attribute(test_string)
        expected = GRIBCode(1, 2, 3, 4)
        assert isinstance(result, GenericConcreteGRIBCode)
        assert result == expected

    def test_grib2(self):
        test_string = "GRIBCode(edition=2, discipline=3, category=4, number=5)"
        result = GP_HANDLER.decode_attribute(test_string)
        expected = GRIBCode(2, 3, 4, 5)
        assert isinstance(result, GenericConcreteGRIBCode)
        assert result == expected

    def test_odd_array_case(self):
        test_value = np.array([1.7, 5.4])
        # Bizarrely, this converts to a string which *will* parse
        result = GP_HANDLER.decode_attribute(test_value)
        expected = GRIBCode(1, 7, 5, 4)
        assert isinstance(result, GenericConcreteGRIBCode)
        assert result == expected

    @pytest.mark.parametrize(
        "badval",
        [np.array(1), 2.5, np.array([9.3, 5, 2.4])],
        ids=["int", "float", "array"],
    )
    def test_badvalue__fail(self, badval):
        # It can convert random values to strings, but they mostly won't satisfy.
        with pytest.raises(ValueError):
            GP_HANDLER.decode_attribute(badval)
