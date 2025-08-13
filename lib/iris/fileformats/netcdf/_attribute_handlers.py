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
from typing import Any

from iris.fileformats.pp import STASH


class AttributeHandler(metaclass=ABCMeta):
    #: The user-visible attribute name used within Iris, which identifies attributes
    #  which we should attempt to encode with this coder.
    _IrisIdentifyingName: str = ""
    #: The storage name(s) which identify this type of data in actual files, which thus
    #  identify attributes which we should attempt to decode with this coder.
    # NOTES:
    # (1) for load, in (presumably extremely rare) case of multiples appearing, "the"
    #  internal attribute is taken from the earliest appearing name: The other values
    #  are lost, and a warning will be issued.
    # (2) for save ,the attribute name is dynamically determined by the "encode" call.
    #  On translation failure, however, we assume it is the last name listed -- since
    #  it is so for StashHandler, the only one it currently matters for.
    _NetcdfIdentifyingNames: list[str] = []

    @property
    def iris_name(self) -> str:
        """Provide the iris attribute name which this handler deals with.

        Read-only access to the information configured at the class-level.
        """
        return self._IrisIdentifyingName

    @property
    def netcdf_names(self) -> list[str]:
        """Provide the netcdf attribute name(s) which this handler deals with.

        Read-only access to the information configured at the class-level.
        """
        # N.B. return a list copy to avoid any possibility of in-place change !
        return list(self._NetcdfIdentifyingNames)

    @property
    def _primary_nc_name(self):
        """The "usual" file attribute name."""
        # N.B. for now, this only matters for STASH, so take the *last* name. Because
        #  the first name is dominant, but that is the 'legacy' version.
        return self._NetcdfIdentifyingNames[-1]

    @abstractmethod
    def encode_object(self, content: Any) -> tuple[str, str]:
        """Encode an object as an attribute name and value.

        We already do change the name of STASH attributes to "um_stash_source" on save
        (as-of Iris 3.12).  This structure also allows that we might produce different
        names for different codes.

        The 'content' may be a custom object or string equivalent, depending on what
        specific implementation allows.

        This should raise TypeError or ValueError if 'content' is unsuitable.
        """
        pass

    @abstractmethod
    def decode_attribute(self, attr_value: Any) -> Any:
        """Decode an attribute name and value into the appropriate attribute object.

        The 'value' is typically a string, but possibly other attribute content types,
        depending on the specific implementation.

        This should raise TypeError or ValueError if 'value' is unsuitable.
        """
        pass


class StashHandler(AttributeHandler):
    """Convert STASH object attribute to/from a netcdf string attribute."""

    _IrisIdentifyingName = "STASH"
    # Note: two possible in-file attribute names, the first one is a 'legacy' version
    #  but takes priority in a conflict.
    _NetcdfIdentifyingNames = ["ukmo__um_stash_source", "um_stash_source"]

    def encode_object(self, stash: Any) -> tuple[str, str]:
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
        return self._primary_nc_name, msi_string

    def decode_attribute(self, attr_value: Any) -> Any:
        # In this case the attribute name does not matter.
        from iris.fileformats.pp import STASH

        attr_value = str(attr_value)
        return STASH.from_msi(attr_value)


class UkmoProcessFlagsHandler(AttributeHandler):
    """Convert ukmo__process_flags tuple attribute to/from a netcdf string attribute."""

    _IrisIdentifyingName = "ukmo__process_flags"
    _NetcdfIdentifyingNames = ["ukmo__process_flags"]

    def encode_object(self, value: Any) -> tuple[str, str]:
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
        return self._primary_nc_name, value

    def decode_attribute(self, attr_value: Any) -> Any:
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

    _IrisIdentifyingName = "GRIB_PARAM"
    _NetcdfIdentifyingNames = ["GRIB_PARAM"]

    def encode_object(self, iris_value: Any) -> Any:
        # 'iris_value' is typically an
        #  iris_grib.grib_phenom_translation._gribcode.GenericConcreteGRIBCode
        # Not typing this, as we need iris_grib to remain an optional import.
        from iris_grib.grib_phenom_translation._gribcode import (
            GenericConcreteGRIBCode,
            GRIBCode,
        )

        if isinstance(iris_value, GenericConcreteGRIBCode):
            gribcode = iris_value
        else:
            # Create a gribcode from that.
            # N.B. (1) implicitly uses str() to convert the arg
            # N.B. (2) can fail : let it, caller deals with this !
            gribcode = GRIBCode(iris_value)

        # The correct file attribute is the repr of a GRIBCode object.
        grib_string = repr(gribcode)
        return self._primary_nc_name, grib_string

    def decode_attribute(self, attr_value: Any) -> Any:
        from iris_grib.grib_phenom_translation._gribcode import GRIBCode

        # As above, a str() conversion is implied here.
        result = GRIBCode(attr_value)
        return result


# Define the available attribute handlers.
ATTRIBUTE_HANDLERS: dict[str, AttributeHandler] = {}


def _add_handler(handler: AttributeHandler):
    ATTRIBUTE_HANDLERS[handler._IrisIdentifyingName] = handler


# Always include the "STASH" and "ukmo__process_flags" handlers.
_add_handler(StashHandler())
_add_handler(UkmoProcessFlagsHandler())

try:
    import iris_grib  # noqa: F401

    # If iris-grib is available, also include the "GRIB_PARAM" handler.
    _add_handler(GribParamHandler())

except ImportError:
    pass
