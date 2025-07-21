# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""NetCDF attribute translations for Iris attributes with special convenience types.

These are things which are stored differently in an Iris cube attribute from how they
are actually stored in a netcdf file.  E.G. a STASH code is stored as a special object,
but in a file it is just a string.

These conversions are intended to be automatic and lossless, like a serialization.

At present, there are 3 of these :
  * "STASH": records/controls the exact file encoding of data loaded from or saved to
     UM file formats (PP/FF).
  * "GRIB_PARAM": does the same for GRIB data (using iris_grib).
  * "ukmo__process_flags": internally a tuple of strings, but stored as a single string
    with underscore separators.

"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

from iris.fileformats.pp import STASH


class AttributeHandler(metaclass=ABCMeta):
    #: The user-visible attribute name used within Iris, which identifies attributes
    #  which we should attempt to encode with this coder.
    IrisIdentifyingName: str = ""
    #: The storage name(s) which identify this type of data in actual files, which thus
    #  identify attributes which we should attempt to decode with this coder.
    # NOTES:
    # (1) for save the attribute name is dynamically determined by the "encode" call.
    # (2) for load, in (presumably extremely rare) case of multiples appearing, "the"
    #  internal attribute is taken from the earliest appearing name: The other values
    #  are lost, and a warning will be issued.
    NetcdfIdentifyingNames: List[str] = []

    @abstractmethod
    def encode_object(self, content) -> Tuple[str, str]:
        """Encode an object as an attribute name and value.

        We already do change the name of STASH attributes to "um_stash_source" on save
        (as-of Iris 3.12).  This structure also allows that we might produce different
        names for different codes.
        """
        pass

    @abstractmethod
    def decode_attribute(self, attr_name: str, attr_value) -> Any:
        """Decode an attribute name and string to an attribute object."""
        pass


class StashHandler(AttributeHandler):
    """Convert STASH object attribute to/from a netcdf string attribute."""

    IrisIdentifyingName = "STASH"
    # Note: two possible in-file attribute names, the first one is a 'legacy' version
    #  but takes priority in a conflict.
    NetcdfIdentifyingNames = ["ukmo__um_stash_source", "um_stash_source"]

    def encode_object(self, stash):
        if isinstance(stash, STASH):
            stash_object = stash
        elif isinstance(stash, str):
            # Attempt to convert as an MSI string to a STASH object.
            # NB this will normalise the content.
            stash_object = STASH.from_msi(stash)
        else:
            msg = (
                f"Invalid STASH attribute can not be written to netcdf file: {stash!r}. "
                "Can only be a 'iris.fileformats.pp.STASH' object, or a string of the "
                "form 'mXXsXXiXXX', where XX are decimal numbers."
            )
            raise TypeError(msg)

        msi_string = str(stash_object)  # convert to standard MSI string representation
        # We always write "um_stash_source", not the legacy one.
        return self.NetcdfIdentifyingNames[1], msi_string

    def decode_attribute(self, attr_name: str, attr_value):
        # In this case the attribute name does not matter.
        from iris.fileformats.pp import STASH

        attr_value = str(attr_value)
        return STASH.from_msi(attr_value)


class UkmoProcessFlagsHandler(AttributeHandler):
    """Convert ukmo__process_flags tuple attribute to/from a netcdf string attribute."""

    IrisIdentifyingName = "ukmo__process_flags"
    NetcdfIdentifyingNames = ["ukmo__process_flags"]

    def encode_object(self, value):
        if not isinstance(value, tuple) or any(
            not isinstance(elem, str) for elem in value
        ):
            msg = (
                f"Invalid 'ukmo__process_flags' attribute : {value!r}. "
                "Must be a tuple of str."
            )
            raise TypeError(msg)

        def value_fix(value):
            value = value.replace(" ", "_")
            if value == "":
                # Special handling for an empty string entry, which otherwise upsets
                #  the split/join process.
                value = "<EMPTY>"
            return value

        value = " ".join([value_fix(x) for x in value])
        return self.NetcdfIdentifyingNames[0], value

    def decode_attribute(self, attr_name: str, attr_value):
        # In this case the attribute name does not matter.
        attr_value = str(attr_value)

        def value_unfix(value):
            value = value.replace("_", " ")
            if value == "<EMPTY>":
                # A special placeholder flagging where the original was an empty string.
                value = ""
            return value

        if attr_value == "":
            # This is basically a fix for the odd behaviour of 'str.split'.
            flags = []
        else:
            flags = [value_unfix(x) for x in attr_value.split(" ")]

        return tuple(flags)


class GribParamHandler(AttributeHandler):
    """Convert iris_grib GRIB_PARAM object attribute to/from a netcdf string attribute.

    Use the mechanisms in iris_grib.
    """

    IrisIdentifyingName = "GRIB_PARAM"
    NetcdfIdentifyingNames = ["GRIB_PARAM"]

    def encode_object(self, iris_value):
        # grib_param should be an
        #  iris_grib.grib_phenom_translation._gribcode.GenericConcreteGRIBCode
        # Not typing this, as we need iris_grib to remain an optional import.
        from iris_grib.grib_phenom_translation._gribcode import (
            GenericConcreteGRIBCode,
            GRIBCode,
        )

        if isinstance(iris_value, GenericConcreteGRIBCode):
            gribcode = iris_value
        else:
            # Attempt to convert to string, if not already
            gribcode = str(iris_value)
            # Attempt to create a gribcode from that.
            # NB let it fail if it will -- caller deals with this !
            gribcode = GRIBCode(gribcode)

        # The correct file attribute is the repr of a GRIBCode object.
        grib_string = repr(gribcode)
        return self.NetcdfIdentifyingNames[0], grib_string

    def decode_attribute(self, attr_name: str, attr_value):
        from iris_grib.grib_phenom_translation._gribcode import GRIBCode

        attr_value = str(attr_value)
        result = GRIBCode(attr_value)
        return result


# Define the available attribute handlers.
ATTRIBUTE_HANDLERS: Dict[str, AttributeHandler] = {}


def _add_handler(handler: AttributeHandler):
    ATTRIBUTE_HANDLERS[handler.IrisIdentifyingName] = handler


# Always include the "STASH" and "ukmo__process_flags" handlers.
_add_handler(StashHandler())
_add_handler(UkmoProcessFlagsHandler())

try:
    import iris_grib  # noqa: F401

    # If iris-grib is available, also include the "GRIB_PARAM" handler.
    _add_handler(GribParamHandler())

except ImportError:
    pass


#
# Mechanism tests
#
def _decode_gribcode(grib_code: str):
    return GribParamHandler().decode_attribute("x", grib_code)
    # from iris_grib.grib_phenom_translation._gribcode import GRIBCode
    #
    # result = None
    # # Use the helper function to construct a suitable GenericConcreteGRIBCode object.
    # try:
    #     result = GRIBCode(grib_code)
    # except (TypeError, ValueError):
    #     pass
    #
    # return result


def make_gribcode(*args, **kwargs):
    from iris_grib.grib_phenom_translation._gribcode import GRIBCode

    return GRIBCode(*args, **kwargs)


class TestGribDecode:
    def test_grib_1(self):
        assert _decode_gribcode(
            "GRIBCode(edition=1, table_version=2, centre_number=3, number=4)"
        ) == make_gribcode(1, 2, 3, 4)

    def test_grib_2(self):
        assert _decode_gribcode("GRIBCode(2,5,7,13)") == make_gribcode(2, 5, 7, 13)

    def test_grib_3(self):
        assert _decode_gribcode(
            "GRIBCode(2,5, number=13, centre_number=7)"
        ) == make_gribcode(2, 5, 7, 13)

    def test_grib_4(self):
        assert _decode_gribcode("GRIBxXCode(2,5,7,13)") == make_gribcode(2, 5, 7, 13)

    def test_grib_5(self):
        assert _decode_gribcode("GRIBCode()") is None

    def test_grib_6(self):
        assert _decode_gribcode("GRIBCode(xxx)") is None

    def test_grib_7(self):
        assert _decode_gribcode(
            "GRIBCode(xxx-any-junk..1, 2,qytw3dsa, 4)"
        ) == make_gribcode(1, 2, 3, 4)


def _sample_decode_rawlbproc(lbproc):
    from iris.fileformats._pp_lbproc_pairs import LBPROC_MAP

    return tuple(
        sorted(
            [
                name
                for value, name in LBPROC_MAP.items()
                if isinstance(value, int) and lbproc & value
            ]
        )
    )


def _check_pf_roundtrip(contents):
    print(f"original: {contents!r}")
    handler = UkmoProcessFlagsHandler()
    name, val = handler.encode_object(contents)
    reconstruct = handler.decode_attribute(name, val)
    print(f"  -> encoded: {val!r}")
    print(f"  -> reconstructed: {reconstruct!r}")
    assert name == "ukmo__process_flags"
    n_val = 0 if val == "" else len(val.split(" "))  # because split is odd
    assert n_val == len(contents)
    assert reconstruct == contents


class TestProcessFlagsRoundtrip:
    def test_pf_1(self):
        sample = ("A example", "b", "another-thing with spaces")
        _check_pf_roundtrip(sample)

    def test_pf_2(self):
        sample = ("single",)
        _check_pf_roundtrip(sample)

    def test_pf_3(self):
        sample = ("nonempty", "", "nonempty2")
        _check_pf_roundtrip(sample)

    def test_pf_4(self):
        sample = ()
        _check_pf_roundtrip(sample)

    def test_pf_5(self):
        sample = ("a", "")
        _check_pf_roundtrip(sample)

    def test_pf_6(self):
        sample = ("", "b")
        _check_pf_roundtrip(sample)

    def test_pf_7(self):
        sample = ("",)
        _check_pf_roundtrip(sample)

    def test_pf_8(self):
        sample = (" ",)
        _check_pf_roundtrip(sample)

    def test_pf_9(self):
        sample = ("", "")
        _check_pf_roundtrip(sample)

    def test_pf_10(self):
        sample = (" a", "b")
        _check_pf_roundtrip(sample)

    def test_pf_11(self):
        sample = ("a ", "b")
        _check_pf_roundtrip(sample)

    def test_pf_12(self):
        sample = ("a", " b")
        _check_pf_roundtrip(sample)

    def test_pf_13(self):
        sample = ("a", "b ")
        _check_pf_roundtrip(sample)


#
# NOTE: also need to test both encode + decode separately, as there are corner cases.
# LIKE: leading+trailing, empty entries ...
#
