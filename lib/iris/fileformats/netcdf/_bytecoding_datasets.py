# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Module providing to netcdf datasets with automatic character encoding.

The requirement is to convert numpy fixed-width unicode arrays on writing to a variable
which is declared as a byte (character) array with a fixed-length string dimension.

Numpy unicode string arrays are ones with dtypes of the form "U<character-width>".
Numpy character variables have the dtype "S1", and map to a fixed-length "string
dimension".

In principle, netCDF4 already performs these translations, but in practice current
releases are not functional for anything other than "ascii" encoding -- including UTF-8,
which is the most obvious and desirable "general" solution.

There is also the question of whether we should like to implement UTF-8 as our default.
Current discussions on this are inconclusive and neither CF conventions nor the NetCDF
User Guide are definite on what possible values of "_Encoding" are, or what the effective
default is, even though they do both mention the "_Encoding" attribute as a potential
way to handle the issue.

Because of this, we interpret as follows:
  * when reading bytes : in the absence of an "_Encoding" attribute, we will attempt to
    decode bytes as UTF-8
  * when writing strings : in the absence of an "_Encoding" attribute (on the Iris
    cube or coord object), we will attempt to encode data with "ascii" : If this fails,
    it raise an error prompting the user to supply an "_Encoding" attribute.

Where an "_Encoding" attribute is provided to Iris, we will honour it where possible,
identifying with "codecs.lookup" :  This means we support the encodings in the Python
Standard Library, and the name aliases which it recognises.

See:

* known problems https://github.com/Unidata/netcdf4-python/issues/1440
* suggestions for how this "ought" to work, discussed in the netcdf-c library
   * https://github.com/Unidata/netcdf-c/issues/402

"""

import codecs
import warnings

import numpy as np

from iris.fileformats.netcdf._thread_safe_nc import DatasetWrapper, VariableWrapper


def decode_bytesarray_to_stringarray(
    byte_array: np.ndarray, encoding="utf-8", string_width: int | None = None
) -> np.ndarray:
    """Convert an array of bytes to an array of strings, with one less dimension.

    N.B. for now at least, we assume the string dim is **always the last one**.
    If 'string_width' is not given, it is set to the final dimension of 'byte_array'.
    """
    bytes_shape = byte_array.shape
    var_shape = bytes_shape[:-1]
    if string_width is None:
        string_width = bytes_shape[-1]
    string_dtype = f"U{string_width}"
    result = np.empty(var_shape, dtype=string_dtype)
    for ndindex in np.ndindex(var_shape):
        element_bytes = byte_array[ndindex]
        bytes = b"".join([b if b else b"\0" for b in element_bytes])
        string = bytes.decode(encoding)
        result[ndindex] = string
    return result


def encode_stringarray_as_bytearray(
    data: np.ndarray, encoding=None, string_dimension_length: int | None = None
) -> np.ndarray:
    """Encode strings as bytearray.

    Note: if 'string_dimension_length' is not given (None), it is set to the longest
    encoded bytes element.  If 'string_dimension_length' is specified, the last array
    dimension is set to this and content strings are truncated or extended as required.
    """
    element_shape = data.shape
    max_length = 1  # this is a MINIMUM - i.e. not zero!
    data_elements = np.zeros(element_shape, dtype=object)
    for index in np.ndindex(element_shape):
        data_element = data[index].encode(encoding=encoding)
        element_length = len(data_element)
        data_elements[index] = data_element
        if element_length > max_length:
            max_length = element_length

    if string_dimension_length is None:
        string_dimension_length = max_length

    # We already encoded all the strings, but stored them in an object-array as
    #  we didn't yet know the fixed byte-length to convert to.
    # Now convert to a fixed-width byte array with an extra string-length dimension
    result = np.zeros(element_shape + (string_dimension_length,), dtype="S1")
    right_pad = b"\0" * string_dimension_length
    for index in np.ndindex(element_shape):
        bytes = data_elements[index]
        bytes = (bytes + right_pad)[:string_dimension_length]
        result[index] = [bytes[i : i + 1] for i in range(string_dimension_length)]

    return result


DEFAULT_ENCODING = "utf-8"


class EncodedVariable(VariableWrapper):
    """A variable wrapper that translates variable data according to byte encodings."""

    def __getitem__(self, keys):
        if self.is_chardata():
            super().set_auto_chartostring(False)

        data = super().__getitem__(keys)

        if self.is_chardata():
            encoding = self.get_byte_encoding()
            strlen = self.get_string_length()
            data = decode_bytesarray_to_stringarray(data, encoding, strlen)

        return data

    def __setitem__(self, keys, data):
        if self.is_chardata():
            encoding = self.get_byte_encoding()
            strlen = self.get_string_length()
            if encoding is not None:
                data = encode_stringarray_as_bytearray(data, encoding, strlen)
            else:
                try:
                    # Check if all characters are valid ascii
                    data = encode_stringarray_as_bytearray(data, "ascii", strlen)
                except UnicodeEncodeError:
                    data = encode_stringarray_as_bytearray(
                        data, DEFAULT_ENCODING, strlen
                    )
                    # As this was necessary, record the new encoding on the variable
                    self.set_ncattr("_Encoding", DEFAULT_ENCODING)
                    msg = (
                        f"Non-ascii data written to label variable {self.name}. "
                        f"Applied {DEFAULT_ENCODING!r} encoding, "
                        f"and set attribute _Encoding={DEFAULT_ENCODING!r}."
                    )
                    warnings.warn(msg, UserWarning)

            super().set_auto_chartostring(False)

        super().__setitem__(keys, data)

    def is_chardata(self):
        return np.issubdtype(self.dtype, np.bytes_)

    def get_encoding(self) -> str | None:
        """Get the effective byte encoding to be used for this variable."""
        # utf-8 is a reasonable "safe" default, equivalent to 'ascii' for ascii data
        result = getattr(self, "_Encoding", None)
        if result is not None:
            try:
                # Accept + normalise naming of encodings
                result = codecs.lookup(result).name
                # NOTE: if encoding does not suit data, errors can occur.
                # For example, _Encoding = "ascii", with non-ascii content.
            except LookupError:
                # Replace some invalid setting with "safe"(ish) fallback.
                msg = f"Unknown encoding for variable {self.name!r}: {result!r}"
                warnings.warn(msg, UserWarning)

        return result

    def get_string_length(self):
        """Return the string-length defined for this variable (or None)."""
        return getattr(self, "iris_string_length", None)


class EncodedDataset(DatasetWrapper):
    """A specialised DatasetWrapper whose variables perform byte encoding."""

    VAR_WRAPPER_CLS = EncodedVariable
