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
import contextlib
import threading
import warnings

import numpy as np

from iris.fileformats.netcdf._thread_safe_nc import DatasetWrapper, VariableWrapper


def decode_bytesarray_to_stringarray(
    byte_array: np.ndarray, encoding: str, string_width: int
) -> np.ndarray:
    """Convert an array of bytes to an array of strings, with one less dimension.

    N.B. for now at least, we assume the string dim is **always the last one**.
    If 'string_width' is not given, it is set to the final dimension of 'byte_array'.
    """
    if np.ma.isMaskedArray(byte_array):
        # netCDF4-python sees zeros as "missing" -- we don't need or want that
        byte_array = byte_array.data
    bytes_shape = byte_array.shape
    var_shape = bytes_shape[:-1]
    string_dtype = f"U{string_width}"
    result = np.empty(var_shape, dtype=string_dtype)
    for ndindex in np.ndindex(var_shape):
        element_bytes = byte_array[ndindex]
        bytes = b"".join([b if b else b"\0" for b in element_bytes])
        string = bytes.decode(encoding)
        result[ndindex] = string
    return result


#
# TODO: remove?
# this older version is "overly flexible", less efficient and not needed here.
#
def flexi_encode_stringarray_as_bytearray(
    data: np.ndarray, encoding=None, string_dimension_length: int | None = None
) -> np.ndarray:
    """Encode strings as bytearray.

    Note: if 'string_dimension_length' is not given (None), it is set to the longest
    encoded bytes element, **OR** the dtype size, if that is greater.
    If 'string_dimension_length' is specified, the last array
    dimension is set to this and content strings are truncated or extended as required.
    """
    if np.ma.isMaskedArray(data):
        # netCDF4-python sees zeros as "missing" -- we don't need or want that
        data = data.data
    element_shape = data.shape
    # Encode all the strings + see which is longest
    max_length = 1  # this is a MINIMUM - i.e. not zero!
    data_elements = np.zeros(element_shape, dtype=object)
    for index in np.ndindex(element_shape):
        data_element = data[index].encode(encoding=encoding)
        element_length = len(data_element)
        data_elements[index] = data_element
        if element_length > max_length:
            max_length = element_length

    if string_dimension_length is None:
        # If the string length was not specified, it is the maximum encoded length
        # (n-bytes), **or** the dtype string-length, if greater.
        string_dimension_length = max_length
        array_string_length = int(str(data.dtype)[2:])  # Yuck. No better public way?
        if array_string_length > string_dimension_length:
            string_dimension_length = array_string_length

    # We maybe *already* encoded all the strings above, but stored them in an
    #  object-array as we didn't yet know the fixed byte-length to convert to.
    # Now convert to a fixed-width byte array with an extra string-length dimension
    result = np.zeros(element_shape + (string_dimension_length,), dtype="S1")
    right_pad = b"\0" * string_dimension_length
    for index in np.ndindex(element_shape):
        bytes = data_elements[index]
        bytes = (bytes + right_pad)[:string_dimension_length]
        result[index] = [bytes[i : i + 1] for i in range(string_dimension_length)]

    return result


def encode_stringarray_as_bytearray(
    data: np.ndarray, encoding: str, string_dimension_length: int
) -> np.ndarray:
    """Encode strings as a bytes array."""
    element_shape = data.shape
    result = np.zeros(element_shape + (string_dimension_length,), dtype="S1")
    right_pad = b"\0" * string_dimension_length
    for index in np.ndindex(element_shape):
        bytes = data[index].encode(encoding=encoding)
        # It's all a bit nasty ...
        bytes = (bytes + right_pad)[:string_dimension_length]
        result[index] = [bytes[i : i + 1] for i in range(string_dimension_length)]

    return result


class NetcdfStringDecodeSetting(threading.local):
    def __init__(self, perform_encoding: bool = True):
        self.set(perform_encoding)

    def set(self, perform_encoding: bool):
        self.perform_encoding = perform_encoding

    def __bool__(self):
        return self.perform_encoding

    @contextlib.contextmanager
    def context(self, perform_encoding: bool):
        old_setting = self.perform_encoding
        self.perform_encoding = perform_encoding
        yield
        self.perform_encoding = old_setting


DECODE_TO_STRINGS_ON_READ = NetcdfStringDecodeSetting()
DEFAULT_READ_ENCODING = "utf-8"
DEFAULT_WRITE_ENCODING = "ascii"


class EncodedVariable(VariableWrapper):
    """A variable wrapper that translates variable data according to byte encodings."""

    def __getitem__(self, keys):
        if self._is_chardata():
            # N.B. we never need to UNset this, as we totally control it
            self._contained_instance.set_auto_chartostring(False)

        data = super().__getitem__(keys)

        if DECODE_TO_STRINGS_ON_READ and self._is_chardata():
            encoding = self._get_encoding() or DEFAULT_READ_ENCODING
            # N.B. typically, read encoding default is UTF-8 --> a "usually safe" choice
            strlen = self._get_string_length()
            try:
                data = decode_bytesarray_to_stringarray(data, encoding, strlen)
            except UnicodeDecodeError as err:
                msg = (
                    f"Character data in variable {self.name!r} could not be decoded"
                    f"with the {encoding!r} encoding.  This can be fixed by setting the "
                    "variable '_Encoding' attribute to suit the content."
                )
                raise ValueError(msg) from err

        return data

    def __setitem__(self, keys, data):
        if self._is_chardata():
            # N.B. we never need to UNset this, as we totally control it
            self._contained_instance.set_auto_chartostring(False)

            encoding = self._get_encoding() or DEFAULT_WRITE_ENCODING
            # N.B. typically, write encoding default is "ascii" --> fails bad content
            if data.dtype.kind == "U":
                try:
                    strlen = self._get_string_length()
                    data = encode_stringarray_as_bytearray(data, encoding, strlen)
                except UnicodeEncodeError as err:
                    msg = (
                        f"String data written to netcdf character variable {self.name!r} "
                        f"could not be represented in encoding {encoding!r}.  This can be "
                        "fixed by setting a suitable variable '_Encoding' attribute, "
                        'e.g. <variable>._Encoding="UTF-8".'
                    )
                    raise ValueError(msg) from err

        super().__setitem__(keys, data)

    def _is_chardata(self):
        return np.issubdtype(self.dtype, np.bytes_)

    def _get_encoding(self) -> str | None:
        """Get the byte encoding defined for this variable (or None)."""
        result = getattr(self, "_Encoding", None)
        if result is not None:
            try:
                # Accept + normalise naming of encodings
                result = codecs.lookup(result).name
                # NOTE: if encoding does not suit data, errors can occur.
                # For example, _Encoding = "ascii", with non-ascii content.
            except LookupError:
                # Unrecognised encoding name : handle this as just a warning
                msg = f"Unknown encoding for variable {self.name!r}: {result!r}"
                warnings.warn(msg, UserWarning)

        return result

    def _get_string_length(self):
        """Return the string-length defined for this variable."""
        if not hasattr(self, "_strlen"):
            # Work out the string length from the parent dataset dimensions.
            strlen = self.group().dimensions[self.dimensions[-1]].size
            # Cache this on the variable -- but not as a netcdf attribute (!)
            self.__dict__["_strlen"] = strlen

        return self._strlen

    def set_auto_chartostring(self, onoff: bool):
        msg = "auto_chartostring is not supported by Iris 'EncodedVariable' type."
        raise TypeError(msg)


class EncodedDataset(DatasetWrapper):
    """A specialised DatasetWrapper whose variables perform byte encoding."""

    VAR_WRAPPER_CLS = EncodedVariable

    def set_auto_chartostring(self, onoff: bool):
        msg = "auto_chartostring is not supported by Iris 'EncodedDataset' type."
        raise TypeError(msg)
